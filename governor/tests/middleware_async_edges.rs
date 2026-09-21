//! Boundary tests for middleware-visible rejection information and
//! asynchronous waiting.
//!
//! These tests pin down three behaviors:
//!   * arriving at *exactly* the announced retry instant conforms (no
//!     extra cell period of waiting),
//!   * cancelling a future that was never admitted consumes no capacity,
//!   * the rejection information exposed through the middleware
//!     (`NotUntil`, `StateSnapshot`) agrees with the limiter's actual
//!     next positive decision, also under a custom clock that can be
//!     rewound.
#![cfg(feature = "std")]

mod common;

use common::RewindableClock;
use futures_executor::block_on;
use futures_util::task::noop_waker;
use governor::clock::{Clock, FakeRelativeClock};
use governor::middleware::StateInformationMiddleware;
use governor::nanos::Nanos;
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use std::future::Future;
use std::task::Context;
use std::time::{Duration, Instant};

/// One cell per 500ms, no burst tolerance.
fn strict_quota() -> Quota {
    Quota::per_second(nonzero!(2u32)).allow_burst(nonzero!(1u32))
}

/// After a rejection, waiting exactly the announced `wait_time_from`
/// (and not one nanosecond more) must be enough. Catches an off-by-one
/// in the conforming comparison (`<` vs `<=`).
#[test]
fn exact_wait_endpoint_needs_no_extra_cell_period() {
    let clock = FakeRelativeClock::default();
    let lim = RateLimiter::direct_with_clock(strict_quota(), clock.clone());

    assert_eq!(Ok(()), lim.check());
    let not_until = lim.check().expect_err("second cell must be rejected");
    assert_eq!(
        Duration::from_millis(500),
        not_until.wait_time_from(clock.now())
    );

    // One nanosecond before the announced instant: still rejected.
    clock.advance(Duration::from_millis(499));
    assert!(
        lim.check().is_err(),
        "cell arriving 1ms early must not conform"
    );

    // Exactly at the announced instant: must conform without waiting
    // for another cell period.
    clock.advance(Duration::from_millis(1));
    assert_eq!(
        Ok(()),
        lim.check(),
        "cell arriving exactly at the announced retry instant must conform"
    );
    assert!(lim.check().is_err());
}

/// Same boundary, for the keyed limiter.
#[test]
fn exact_wait_endpoint_needs_no_extra_cell_period_keyed() {
    let clock = FakeRelativeClock::default();
    let lim = RateLimiter::hashmap_with_clock(strict_quota(), clock.clone());

    assert_eq!(Ok(()), lim.check_key(&1u32));
    let not_until = lim.check_key(&1u32).expect_err("must be rejected");
    clock.advance(not_until.wait_time_from(clock.now()));
    assert_eq!(
        Ok(()),
        lim.check_key(&1u32),
        "keyed limiter must admit exactly at the announced instant"
    );
}

/// `until_ready` must resolve as soon as the first cell conforms, not
/// one cell period later. Uses a real clock with a 100ms period; the
/// whole test waits ~100ms.
#[test]
fn until_ready_resolves_at_first_conforming_moment() {
    let lim = RateLimiter::direct(Quota::with_period(Duration::from_millis(100)).unwrap());
    lim.check().expect("first cell conforms");

    let start = Instant::now();
    block_on(lim.until_ready());
    let elapsed = start.elapsed();

    assert!(
        elapsed >= Duration::from_millis(90),
        "until_ready resolved suspiciously early ({:?}); it must wait for the cell",
        elapsed
    );
    assert!(
        elapsed < Duration::from_millis(190),
        "until_ready waited {:?}: resolving one full cell period (100ms) late",
        elapsed
    );
    // The future consumed exactly one cell:
    assert!(lim.check().is_err());
}

/// A pending `until_ready` future that is dropped before being admitted
/// must not consume any capacity: the announced wait must not grow, and
/// exactly one cell becomes available after one period.
#[test]
fn cancelled_pending_future_consumes_no_capacity() {
    let lim = RateLimiter::direct(Quota::with_period(Duration::from_millis(100)).unwrap());
    lim.check().expect("first cell conforms");
    let wait_before = lim
        .check()
        .expect_err("bucket exhausted")
        .wait_time_from(lim.clock().now());

    // Poll the future once so it observes the rejection and registers
    // its timer, then cancel it before it was ever admitted.
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut future = Box::pin(lim.until_ready());
    assert!(
        future.as_mut().poll(&mut cx).is_pending(),
        "future must be pending while the limiter is exhausted"
    );
    drop(future);

    let wait_after = lim
        .check()
        .expect_err("bucket still exhausted")
        .wait_time_from(lim.clock().now());
    assert!(
        wait_after <= wait_before,
        "cancelled future consumed capacity: wait grew from {:?} to {:?}",
        wait_before,
        wait_after
    );

    // After one period, exactly one cell is available (none was eaten
    // by the cancelled future):
    std::thread::sleep(wait_after + Duration::from_millis(20));
    assert!(
        lim.check().is_ok(),
        "cell must be available after one period"
    );
    assert!(
        lim.check().is_err(),
        "and exactly one cell must have arrived"
    );
}

/// Rejection information exposed through the middleware must agree with
/// the limiter's actual next positive decision, under a custom clock
/// that can advance and rewind.
#[test]
fn rejection_info_matches_actual_next_allowance_custom_clock() {
    let clock = RewindableClock::default();
    let quota = Quota::per_second(nonzero!(2u32)).allow_burst(nonzero!(2u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone())
        .with_middleware::<StateInformationMiddleware>();

    // Visible "header" fields on positive outcomes:
    assert_eq!(1, lim.check().unwrap().remaining_burst_capacity());
    assert_eq!(0, lim.check().unwrap().remaining_burst_capacity());

    // The rejection: quota, earliest retry and wait must be consistent.
    let rejected = lim.check().expect_err("burst of 2 exhausted");
    assert_eq!(quota, rejected.quota());
    assert_eq!(Nanos::from(500_000_000u64), rejected.earliest_possible());
    assert_eq!(
        Duration::from_millis(500),
        rejected.wait_time_from(clock.now())
    );

    // A rejection must not mutate state: asking again yields the same answer.
    let rejected_again = lim.check().expect_err("still exhausted");
    assert_eq!(
        rejected.earliest_possible(),
        rejected_again.earliest_possible()
    );

    // The reported wait tracks the custom clock, forwards and backwards:
    clock.advance(Duration::from_millis(200));
    assert_eq!(
        Duration::from_millis(300),
        rejected_again.wait_time_from(clock.now())
    );
    clock.rewind(Duration::from_millis(100));
    assert_eq!(
        Duration::from_millis(400),
        rejected_again.wait_time_from(clock.now())
    );

    // Arriving exactly at the announced instant conforms, and the next
    // cell then needs a full period:
    clock.advance(Duration::from_millis(400));
    assert!(
        lim.check().is_ok(),
        "limiter must admit exactly at the instant its rejection announced"
    );
    let next = lim.check().expect_err("only one cell was due");
    assert_eq!(Duration::from_millis(500), next.wait_time_from(clock.now()));
}

/// Keyed rejection information is per-key: a rejection on one key must
/// report (and cause) no delay on another key.
#[test]
fn keyed_rejection_info_is_per_key() {
    let clock = RewindableClock::default();
    let lim = RateLimiter::hashmap_with_clock(strict_quota(), clock.clone())
        .with_middleware::<StateInformationMiddleware>();

    assert!(lim.check_key(&1u32).is_ok());
    let rejected = lim.check_key(&1u32).expect_err("key 1 exhausted");
    assert_eq!(
        Duration::from_millis(500),
        rejected.wait_time_from(clock.now())
    );

    // Key 2 is unaffected by key 1's rejection:
    assert!(lim.check_key(&2u32).is_ok());

    // And key 1 conforms exactly when its rejection said it would:
    clock.advance(rejected.wait_time_from(clock.now()));
    assert!(lim.check_key(&1u32).is_ok());
}
