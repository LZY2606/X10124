#![cfg(feature = "std")]

//! Deterministic concurrency tests for the keyed (and direct) GCRA state.
//!
//! These tests use the controllable [`ManualClock`] plus thread barriers so
//! that every thread hits the limiter at the *same* fake instant. Nothing
//! here depends on OS scheduling or wall-clock timing: each round releases
//! all threads, waits for every check to complete, then advances the fake
//! clock by exactly one cell period. Assertions are on exact counts summed
//! over all threads, so a CAS retry that wrongly advances the theoretical
//! arrival time, a lost update, or cross-key state leakage fails
//! deterministically on every run.

mod common;

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Barrier,
};
use std::thread;
use std::time::Duration;

use governor::clock::Clock;
use governor::middleware::NoOpMiddleware;
use governor::state::keyed::{KeyedStateStore, ShrinkableKeyedStateStore};
use governor::{nanos::Nanos, Quota, RateLimiter};
use nonzero_ext::nonzero;

use common::ManualClock;

const T_NS: u64 = 1_000_000; // 1ms per cell
const BURST: u32 = 10;

fn quota() -> Quota {
    Quota::with_period(Duration::from_nanos(T_NS))
        .unwrap()
        .allow_burst(nonzero!(BURST))
}

/// Outcome counters shared between worker threads.
#[derive(Default)]
struct Counters {
    allowed: AtomicU64,
    denied: AtomicU64,
    /// Sum of `wait_time_from(now)` in nanoseconds across all denials.
    denied_wait_ns: AtomicU64,
}

/// Run `rounds` synchronized rounds: in every round `threads` workers each
/// call `check_key(key)` once, all released by a barrier; the main thread
/// waits for completion on a second barrier before advancing the clock.
///
/// With burst `BURST` and one cell period per round, exactly `BURST` of the
/// `threads * rounds` checks may succeed (and at most one extra per round
/// only if the bucket somehow over-replenishes).
fn run_single_cell_contention<S>(
    make: impl Fn(ManualClock) -> Arc<RateLimiter<u32, S, ManualClock, NoOpMiddleware<Nanos>>>,
) where
    S: KeyedStateStore<u32> + Send + Sync + 'static,
{
    const THREADS: u64 = 8;
    const ROUNDS: u64 = 60;

    let clock = ManualClock::default();
    let limiter = make(clock.clone());
    let counters = Arc::new(Counters::default());

    for round in 0..ROUNDS {
        let start = Arc::new(Barrier::new(THREADS as usize + 1));
        let done = Arc::new(Barrier::new(THREADS as usize + 1));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let limiter = Arc::clone(&limiter);
            let counters = Arc::clone(&counters);
            let start = Arc::clone(&start);
            let done = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                start.wait();
                match limiter.check_key(&1u32) {
                    Ok(()) => {
                        counters.allowed.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(negative) => {
                        counters.denied.fetch_add(1, Ordering::AcqRel);
                        let wait = negative.wait_time_from(limiter.clock().now());
                        counters
                            .denied_wait_ns
                            .fetch_add(wait.as_nanos() as u64, Ordering::AcqRel);
                    }
                }
                done.wait();
            }));
        }
        start.wait();
        done.wait();
        for h in handles {
            h.join().unwrap();
        }
        // The last round must not advance the clock further.
        if round + 1 < ROUNDS {
            clock.advance(Duration::from_nanos(T_NS));
        }
    }

    let allowed = counters.allowed.load(Ordering::Acquire);
    let denied = counters.denied.load(Ordering::Acquire);
    assert_eq!(
        allowed + denied,
        THREADS * ROUNDS,
        "every check must be recorded exactly once"
    );
    // Burst available in round 0, plus exactly one fresh cell per round.
    assert_eq!(
        allowed,
        BURST as u64 + (ROUNDS - 1),
        "exact quota must be admitted regardless of thread scheduling"
    );

    // Every single-cell rejection at the end of a round advertises a wait
    // of at least one cell period (a bogus TAT pushed forward on a CAS retry
    // makes some waits exceed one period in the same round).
    let denied_count = denied;
    let total_wait = counters.denied_wait_ns.load(Ordering::Acquire);
    assert!(
        total_wait >= denied_count * T_NS,
        "no rejection may advertise a sub-period wait"
    );
}

#[test]
fn same_key_dashmap_admits_exactly_the_quota_under_contention() {
    run_single_cell_contention(|clock| Arc::new(RateLimiter::dashmap_with_clock(quota(), clock)));
}

#[test]
fn same_key_hashmap_admits_exactly_the_quota_under_contention() {
    run_single_cell_contention(|clock| Arc::new(RateLimiter::hashmap_with_clock(quota(), clock)));
}

/// Many keys are hit simultaneously: each key's quota is independent.
fn run_multi_key_isolation<S>(
    make: impl Fn(ManualClock) -> Arc<RateLimiter<u32, S, ManualClock, NoOpMiddleware<Nanos>>>,
) where
    S: KeyedStateStore<u32> + ShrinkableKeyedStateStore<u32> + Send + Sync + 'static,
{
    const KEYS: u32 = 4;
    const ROUNDS: u64 = 40;
    // One worker per key, issuing 3 checks per round => first-round burst
    // admits min(BURST, 3) and subsequent rounds admit exactly one each.
    const CHECKS_PER_ROUND: u64 = 3;

    let clock = ManualClock::default();
    let limiter = make(clock.clone());
    let per_key: Arc<Vec<AtomicU64>> = Arc::new((0..KEYS).map(|_| AtomicU64::new(0)).collect());

    for round in 0..ROUNDS {
        let start = Arc::new(Barrier::new(KEYS as usize + 1));
        let done = Arc::new(Barrier::new(KEYS as usize + 1));
        let mut handles = Vec::new();
        for key_idx in 0..KEYS {
            let limiter = Arc::clone(&limiter);
            let per_key = Arc::clone(&per_key);
            let start = Arc::clone(&start);
            let done = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                start.wait();
                let key = key_idx + 1;
                for _ in 0..CHECKS_PER_ROUND {
                    if limiter.check_key(&key).is_ok() {
                        per_key[key_idx as usize].fetch_add(1, Ordering::AcqRel);
                    }
                }
                done.wait();
            }));
        }
        start.wait();
        done.wait();
        for h in handles {
            h.join().unwrap();
        }
        if round + 1 < ROUNDS {
            clock.advance(Duration::from_nanos(T_NS));
        }
    }

    // Independent reference simulation of one key: burst=BURST, one cell
    // period per round, CHECKS_PER_ROUND checks at each round's instant.
    let expected = {
        let tau = (BURST - 1) as i64;
        let mut tat: Option<i64> = None;
        let mut admitted = 0u64;
        for round in 0..ROUNDS as i64 {
            for _ in 0..CHECKS_PER_ROUND {
                let prev = tat.unwrap_or(round);
                if round >= prev - tau {
                    tat = Some(prev.max(round) + 1);
                    admitted += 1;
                }
            }
        }
        admitted
    };
    for key_idx in 0..KEYS {
        assert_eq!(
            per_key[key_idx as usize].load(Ordering::Acquire),
            expected,
            "key {} got an out-of-quota admission count: {}",
            key_idx + 1,
            per_key[key_idx as usize].load(Ordering::Acquire)
        );
    }

    // Exactly KEYS entries should exist (no state copied between keys).
    assert_eq!(limiter.len(), KEYS as usize);
}

#[test]
fn different_keys_dashmap_do_not_share_state() {
    run_multi_key_isolation(|clock| Arc::new(RateLimiter::dashmap_with_clock(quota(), clock)));
}

#[test]
fn different_keys_hashmap_do_not_share_state() {
    run_multi_key_isolation(|clock| Arc::new(RateLimiter::hashmap_with_clock(quota(), clock)));
}

/// Batch checks under contention: admissions must be all-or-nothing. With
/// `THREADS` workers each requesting half the burst, only an even total
/// weighted admission matching the exact quota is acceptable.
fn run_batch_atomicity<S>(
    make: impl Fn(ManualClock) -> Arc<RateLimiter<u32, S, ManualClock, NoOpMiddleware<Nanos>>>,
) where
    S: KeyedStateStore<u32> + Send + Sync + 'static,
{
    const THREADS: u64 = 8;
    const ROUNDS: u64 = 41; // odd number; the weighted result is unambiguous
    const N: u32 = BURST / 2; // 5

    let clock = ManualClock::default();
    let limiter = make(clock.clone());
    let admitted_batches = Arc::new(AtomicU64::new(0));
    let rejected_batches = Arc::new(AtomicU64::new(0));

    for round in 0..ROUNDS {
        let start = Arc::new(Barrier::new(THREADS as usize + 1));
        let done = Arc::new(Barrier::new(THREADS as usize + 1));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let limiter = Arc::clone(&limiter);
            let admitted_batches = Arc::clone(&admitted_batches);
            let rejected_batches = Arc::clone(&rejected_batches);
            let start = Arc::clone(&start);
            let done = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                start.wait();
                // check_key_n returns Ok(Ok) / Ok(Err) for conforming/rejected.
                let result = limiter
                    .check_key_n(&7u32, nonzero!(N))
                    .expect("5 cells fit a burst of 10");
                if result.is_ok() {
                    admitted_batches.fetch_add(1, Ordering::AcqRel);
                } else {
                    rejected_batches.fetch_add(1, Ordering::AcqRel);
                }
                done.wait();
            }));
        }
        start.wait();
        done.wait();
        for h in handles {
            h.join().unwrap();
        }
        if round + 1 < ROUNDS {
            // Replenish exactly the cells a full batch consumes each round:
            clock.advance(Duration::from_nanos(T_NS * N as u64));
        }
    }

    let admitted = admitted_batches.load(Ordering::Acquire);
    let rejected = rejected_batches.load(Ordering::Acquire);
    assert_eq!(admitted + rejected, THREADS * ROUNDS);

    // Round 0 admits exactly BURST/N = 2 batches; every later round (which
    // replenishes N cells) admits exactly one. No partial batch may ever be
    // charged, which is the "batch != multiple single calls" invariant.
    assert_eq!(
        admitted,
        (BURST / N) as u64 + (ROUNDS - 1),
        "batch admissions must be atomic and exactly match the quota"
    );
}

#[test]
fn batch_checks_dashmap_are_atomic_under_contention() {
    run_batch_atomicity(|clock| Arc::new(RateLimiter::dashmap_with_clock(quota(), clock)));
}

#[test]
fn batch_checks_hashmap_are_atomic_under_contention() {
    run_batch_atomicity(|clock| Arc::new(RateLimiter::hashmap_with_clock(quota(), clock)));
}

/// Direct (unkeyed) limiter contention: same exact-quota invariant.
#[test]
fn direct_limiter_admits_exactly_the_quota_under_contention() {
    use governor::state::InMemoryState;

    const THREADS: u64 = 8;
    const ROUNDS: u64 = 50;

    let clock = ManualClock::default();
    let limiter: Arc<
        RateLimiter<governor::state::NotKeyed, InMemoryState, ManualClock, NoOpMiddleware<Nanos>>,
    > = Arc::new(RateLimiter::new(
        quota(),
        InMemoryState::default(),
        clock.clone(),
    ));
    let allowed = Arc::new(AtomicU64::new(0));
    let denied = Arc::new(AtomicU64::new(0));

    for round in 0..ROUNDS {
        let start = Arc::new(Barrier::new(THREADS as usize + 1));
        let done = Arc::new(Barrier::new(THREADS as usize + 1));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let limiter = Arc::clone(&limiter);
            let allowed = Arc::clone(&allowed);
            let denied = Arc::clone(&denied);
            let start = Arc::clone(&start);
            let done = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                start.wait();
                if limiter.check().is_ok() {
                    allowed.fetch_add(1, Ordering::AcqRel);
                } else {
                    denied.fetch_add(1, Ordering::AcqRel);
                }
                done.wait();
            }));
        }
        start.wait();
        done.wait();
        for h in handles {
            h.join().unwrap();
        }
        if round + 1 < ROUNDS {
            clock.advance(Duration::from_nanos(T_NS));
        }
    }

    let allowed_n = allowed.load(Ordering::Acquire);
    let denied_n = denied.load(Ordering::Acquire);
    assert_eq!(
        allowed_n,
        BURST as u64 + ROUNDS - 1,
        "direct limiter must admit exactly the quota across threads"
    );
    assert_eq!(allowed_n + denied_n, THREADS * ROUNDS);
}
