#![cfg(feature = "std")]

//! Boundary tests for the async waiting paths and the rejection metadata
//! exposed to middleware.
//!
//! These tests pin down four properties that are easy to get off-by-one:
//!
//! * when the clock reaches the advertised retry time *exactly*, the
//!   limiter admits the cell immediately (it must not wait an additional
//!   cell period);
//! * the same holds for `until_n_ready` batches;
//! * dropping a future that has not yet been admitted consumes no quota,
//!   neither for direct nor for keyed limiters;
//! * the rejection information observed through a custom clock (and
//!   derived by middleware into visible header fields) agrees with the
//!   limiter's actual next admission time.
//!
//! The real-clock tests wait for at most one short cell period (~100ms);
//! all other tests use the deterministic [`ManualClock`].

mod common;

use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Barrier,
};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread;
use std::time::{Duration, Instant};

use futures_executor::block_on;
use governor::clock::Clock;
use governor::middleware::StateInformationMiddleware;
use governor::nanos::Nanos;
use governor::state::{InMemoryState, StateStore};
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;

use common::ManualClock;

// ---------------------------------------------------------------------------
// A no-op waker so futures can be polled without an executor.
// ---------------------------------------------------------------------------

fn noop_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker {
        noop_raw()
    }
    fn noop(_: *const ()) {}
    fn noop_raw() -> RawWaker {
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    unsafe { Waker::from_raw(noop_raw()) }
}

fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    future.poll(&mut cx)
}

// ---------------------------------------------------------------------------
// Real-clock: arriving exactly at the wait endpoint must not cost an
// additional cell period.
// ---------------------------------------------------------------------------

#[test]
fn until_ready_waits_exactly_one_cell_period_at_the_boundary() {
    // 10 cells/s, burst 10: after exhausting the burst the next cell is
    // due in exactly 100ms. Waiting ~200ms here would mean one extra cell
    // period slipped into the wait computation.
    let lim = RateLimiter::direct(Quota::per_second(nonzero!(10u32)));
    while lim.check().is_ok() {}
    let start = Instant::now();
    block_on(lim.until_ready());
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(100),
        "waited less than one full cell period"
    );
    assert!(
        elapsed < Duration::from_millis(180),
        "waited nearly two cell periods: the exact endpoint was missed"
    );
}

#[test]
fn until_n_ready_batch_does_not_wait_more_than_the_owed_weight() {
    // Burst of 2, then request a batch of 2 immediately. The batch becomes
    // conforming after two cell periods; the implementation sleeps until
    // that point and then re-checks. A mutant that always waits an
    // *additional* period past the batch's weighted endpoint (e.g. using
    // single-cell retry timing) lands near three periods and fails the
    // upper bound. The lower bound proves the full two-period weight was
    // actually waited for.
    let lim = RateLimiter::direct(Quota::per_second(nonzero!(10u32)).allow_burst(nonzero!(2u32)));
    lim.check_n(nonzero!(2u32)).unwrap().unwrap();
    let start = Instant::now();
    block_on(lim.until_n_ready(nonzero!(2u32))).unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(200),
        "waited less than the two-period batch weight"
    );
    assert!(
        elapsed < Duration::from_millis(280),
        "waited close to three periods: an extra cell period was added past the batch endpoint"
    );
}

#[test]
fn keyed_until_ready_waits_exactly_one_cell_period() {
    let lim = RateLimiter::keyed(Quota::per_second(nonzero!(10u32)));
    let key = 42u32;
    while lim.check_key(&key).is_ok() {}
    let start = Instant::now();
    block_on(lim.until_key_ready(&key));
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(100));
    assert!(elapsed < Duration::from_millis(180));
}

// ---------------------------------------------------------------------------
// Deterministic clock: dropping a not-yet-admitted future is free.
// ---------------------------------------------------------------------------

#[test]
fn dropping_a_pending_direct_future_consumes_no_quota() {
    // A real clock is used here: the async future registers its delay with
    // the OS timer. The cell period is short (50ms) so the test stays quick.
    let lim = RateLimiter::direct(Quota::per_second(nonzero!(20u32)).allow_burst(nonzero!(2u32)));
    lim.check().unwrap();
    lim.check().unwrap(); // burst of 2 exhausted

    // A not-yet-admitted waiter, canceled long before its delay elapses.
    let mut pending = Box::pin(lim.until_ready());
    assert!(poll_once(pending.as_mut()).is_pending());
    drop(pending);

    // Immediately after cancellation the limiter is still exhausted.
    assert!(lim.check().is_err());

    // At exactly one cell period later exactly *one* cell is available:
    // if the canceled future had consumed a cell, the first check here
    // would be rejected and only one would succeed a period later.
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        lim.check().is_ok(),
        "the one replenished cell must be usable"
    );
    assert!(
        lim.check().is_err(),
        "a second cell was consumed by the dropped future"
    );
}

#[test]
fn dropping_a_pending_keyed_future_consumes_no_quota() {
    let lim = RateLimiter::keyed(Quota::per_second(nonzero!(20u32)).allow_burst(nonzero!(2u32)));
    let key = 9u32;
    lim.check_key(&key).unwrap();
    lim.check_key(&key).unwrap(); // burst of 2 exhausted

    let mut pending = Box::pin(lim.until_key_ready(&key));
    assert!(poll_once(pending.as_mut()).is_pending());
    drop(pending);

    assert!(lim.check_key(&key).is_err());
    std::thread::sleep(Duration::from_millis(50));
    assert!(lim.check_key(&key).is_ok());
    assert!(lim.check_key(&key).is_err());
}

// ---------------------------------------------------------------------------
// Rejection metadata under a custom clock must match the actual next
// admission - including the exact-boundary point and batch weights.
// ---------------------------------------------------------------------------

#[test]
fn rejection_metadata_tracks_the_actual_next_admission_on_custom_clock() {
    let t = Duration::from_nanos(10);
    let quota = Quota::with_period(t).unwrap().allow_burst(nonzero!(3u32));
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(quota, clock.clone())
        .with_middleware::<StateInformationMiddleware>();

    lim.check().unwrap(); // tat=10
    lim.check().unwrap(); // tat=20
    lim.check().unwrap(); // tat=30, burst exhausted

    let denial = lim.check().unwrap_err();
    let earliest = denial.earliest_possible();
    assert_eq!(earliest.as_u64(), 10, "one owed single cell is due at t=10");
    assert_eq!(
        denial.wait_time_from(clock.now()),
        Duration::from_nanos(10),
        "advertised wait must be exactly one cell period"
    );

    // One nanosecond early the same rejection stands.
    clock.advance(Duration::from_nanos(9));
    let denial = lim.check().unwrap_err();
    assert_eq!(denial.wait_time_from(clock.now()), Duration::from_nanos(1));

    // Exactly at the endpoint the cell is admitted.
    clock.advance(Duration::from_nanos(1));
    assert!(lim.check().is_ok(), "admission at the exact endpoint");
    assert!(lim.check().is_err());
}

#[test]
fn batch_rejection_metadata_tracks_the_weighted_next_admission() {
    let t = Duration::from_nanos(10);
    let quota = Quota::with_period(t).unwrap().allow_burst(nonzero!(3u32));
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(quota, clock.clone())
        .with_middleware::<StateInformationMiddleware>();

    lim.check_n(nonzero!(3u32)).unwrap().unwrap(); // tat=30

    // A fresh batch of 3 needs the whole bucket: earliest at 30.
    let denial = lim.check_n(nonzero!(3u32)).unwrap().unwrap_err();
    assert_eq!(denial.earliest_possible().as_u64(), 30);
    assert_eq!(denial.wait_time_from(clock.now()), Duration::from_nanos(30));

    // A batch of 2 needs two cells of room, which exists at t=20.
    let denial = lim.check_n(nonzero!(2u32)).unwrap().unwrap_err();
    assert_eq!(denial.earliest_possible().as_u64(), 20);

    // The rejected batches consumed nothing.
    clock.advance(Duration::from_nanos(20));
    assert!(lim.check_n(nonzero!(2u32)).unwrap().is_ok());
    assert!(lim.check_n(nonzero!(2u32)).unwrap().is_err());
}

// ---------------------------------------------------------------------------
// Forced CAS collision: even when two threads update one state at exactly
// the same instant, the stored theoretical arrival time reflects precisely
// the two admissions (no lost update, no advancement on retry).
// ---------------------------------------------------------------------------

#[test]
fn forced_cas_collision_records_exactly_two_admissions() {
    use governor::state::NotKeyed;

    let state = Arc::new(InMemoryState::default());
    // Two barriers: threads synchronize *once* on their first closure
    // invocation (later CAS retries skip the barrier, otherwise a losing
    // thread would block waiting for a barrier that is never entered again).
    let first_attempt = Arc::new(Barrier::new(2));
    let attempts = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();
    for _ in 0..2 {
        let state = Arc::clone(&state);
        let first_attempt = Arc::clone(&first_attempt);
        let attempts = Arc::clone(&attempts);
        handles.push(thread::spawn(move || {
            let did_sync = Arc::new(std::sync::atomic::AtomicBool::new(false));
            state
                .measure_and_replace(&NotKeyed::NonKey, move |old| {
                    let n = attempts.fetch_add(1, Ordering::AcqRel);
                    if n < 2 && !did_sync.swap(true, Ordering::AcqRel) {
                        first_attempt.wait();
                    }
                    let prev = old.map(Nanos::as_u64).unwrap_or(0);
                    let next = Nanos::from(prev + 1);
                    Ok::<((), Nanos), ()>(((), next))
                })
                .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Two admissions must produce a final TAT of exactly 2: no lost update
    // and no TAT advancement introduced on a CAS retry.
    let final_state = Arc::try_unwrap(state).ok().expect("unique owner");
    let value = final_state
        .measure_and_replace(&NotKeyed::NonKey, |old| {
            let prev = old.map(Nanos::as_u64).unwrap_or(0);
            Ok::<(u64, Nanos), ()>((prev, Nanos::from(prev)))
        })
        .unwrap();
    assert_eq!(value, 2, "exactly two admissions must be recorded");
    assert!(
        attempts.load(Ordering::Acquire) > 2,
        "synchronizing inside the closures must force at least one CAS retry, got {}",
        attempts.load(Ordering::Acquire)
    );
}
