//! Concurrency tests for contended rate-limiting decisions.
//!
//! All threads are released simultaneously behind a barrier against a
//! frozen fake clock, so every thread observes the exact same instant
//! and the outcomes are fully determined by the quota, not by
//! scheduling: exactly the burst capacity may be admitted, rejections
//! must report the exact theoretical retry time, and distinct keys must
//! never influence each other. Each scenario is repeated for many
//! rounds to smoke out races without ever relying on a particular
//! interleaving.
#![cfg(feature = "std")]

use governor::clock::{Clock, FakeRelativeClock};
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use std::sync::Barrier;
use std::thread;
use std::time::Duration;

const ROUNDS: usize = 25;

/// One cell every 125ms, burst of 8.
fn quota8() -> Quota {
    Quota::per_second(nonzero!(8u32))
}

/// Runs `threads` closures simultaneously against a barrier and
/// collects their results.
fn run_contended<T: Send>(threads: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    let barrier = Barrier::new(threads);
    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                let f = &f;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    f(i)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Many threads hammering the *same* key at the *same* instant:
/// exactly the burst capacity is admitted, and every rejection reports
/// the exact same wait time. A limiter that advances its theoretical
/// arrival time before a lost CAS race would admit too few or too many.
#[test]
fn barrier_same_key_admits_exactly_burst() {
    const THREADS: usize = 12;
    const BURST: usize = 8;
    for round in 0..ROUNDS {
        let clock = FakeRelativeClock::default();
        let lim = RateLimiter::hashmap_with_clock(quota8(), clock.clone());
        let outcomes = run_contended(THREADS, |_| {
            lim.check_key(&1u32)
                .map_err(|nu| nu.wait_time_from(clock.now()))
                .err()
        });
        let admitted = outcomes.iter().filter(|o| o.is_none()).count();
        assert_eq!(
            BURST, admitted,
            "round {round}: expected exactly {BURST} admitted out of {THREADS} contenders"
        );
        for wait in outcomes.iter().flatten() {
            assert_eq!(
                Duration::from_millis(125),
                *wait,
                "round {round}: all rejections at t=0 must report the same 125ms wait"
            );
        }
    }
}

/// Same as above, but against the concurrent dashmap state store.
#[cfg(feature = "dashmap")]
#[test]
fn barrier_same_key_admits_exactly_burst_dashmap() {
    const THREADS: usize = 12;
    const BURST: usize = 8;
    for round in 0..ROUNDS {
        let clock = FakeRelativeClock::default();
        let lim = RateLimiter::dashmap_with_clock(quota8(), clock.clone());
        let outcomes = run_contended(THREADS, |_| {
            lim.check_key(&1u32)
                .map_err(|nu| nu.wait_time_from(clock.now()))
                .err()
        });
        let admitted = outcomes.iter().filter(|o| o.is_none()).count();
        assert_eq!(
            BURST, admitted,
            "round {round}: dashmap store admitted {admitted}, expected {BURST}"
        );
        for wait in outcomes.iter().flatten() {
            assert_eq!(
                Duration::from_millis(125),
                *wait,
                "round {round}: all rejections at t=0 must report the same 125ms wait"
            );
        }
    }
}

/// The direct (un-keyed) limiter under the same kind of contention.
#[test]
fn barrier_direct_admits_exactly_burst() {
    const THREADS: usize = 12;
    const BURST: usize = 8;
    for round in 0..ROUNDS {
        let clock = FakeRelativeClock::default();
        let lim = RateLimiter::direct_with_clock(quota8(), clock.clone());
        let outcomes = run_contended(THREADS, |_| {
            lim.check()
                .map_err(|nu| nu.wait_time_from(clock.now()))
                .err()
        });
        let admitted = outcomes.iter().filter(|o| o.is_none()).count();
        assert_eq!(
            BURST, admitted,
            "round {round}: direct limiter admitted {admitted}, expected {BURST}"
        );
        for wait in outcomes.iter().flatten() {
            assert_eq!(
                Duration::from_millis(125),
                *wait,
                "round {round}: rejection wait must be exactly one cell period"
            );
        }
    }
}

/// Each thread owns a distinct key: every thread must see the full
/// burst capacity for its own key, no matter how the threads interleave.
#[test]
fn barrier_distinct_keys_do_not_share_state() {
    const KEYS: usize = 8;
    const BURST: u32 = 4;
    for round in 0..ROUNDS {
        let clock = FakeRelativeClock::default();
        let lim = RateLimiter::hashmap_with_clock(Quota::per_second(nonzero!(4u32)), clock.clone());
        run_contended(KEYS, |i| {
            let key = i as u32;
            for attempt in 0..BURST {
                assert!(
                    lim.check_key(&key).is_ok(),
                    "round {}, key {}, attempt {}: \
                     distinct keys must each get a full burst",
                    round,
                    key,
                    attempt
                );
            }
            let wait = match lim.check_key(&key) {
                Err(not_until) => not_until.wait_time_from(clock.now()),
                Ok(()) => panic!("round {}, key {}: burst must be exhausted", round, key),
            };
            assert_eq!(
                Duration::from_millis(250),
                wait,
                "round {}, key {}: wait must be exactly one 250ms cell period",
                round,
                key
            );
        });
        assert_eq!(
            KEYS,
            lim.len(),
            "round {round}: expected one state entry per key"
        );
    }
}

/// Batch checks under contention are all-or-nothing: with a burst of 8
/// and batches of 3, exactly two batches (6 cells) can be admitted at
/// the same instant, and afterwards exactly two single cells fit.
#[test]
fn barrier_batches_are_all_or_nothing() {
    const THREADS: usize = 6;
    for round in 0..ROUNDS {
        let clock = FakeRelativeClock::default();
        let lim = RateLimiter::hashmap_with_clock(quota8(), clock.clone());
        let outcomes = run_contended(THREADS, |_| {
            lim.check_key_n(&1u32, nonzero!(3u32))
                .unwrap_or_else(|e| {
                    panic!("round {}: batch of 3 fits in burst of 8: {:?}", round, e)
                })
                .is_ok()
        });
        let admitted = outcomes.iter().filter(|ok| **ok).count();
        assert_eq!(
            2, admitted,
            "round {round}: exactly 2 batches of 3 fit in a burst of 8, got {admitted}"
        );
        // The remaining 2 cells of capacity are usable individually:
        assert!(lim.check_key(&1u32).is_ok(), "round {}: 7th cell", round);
        assert!(lim.check_key(&1u32).is_ok(), "round {}: 8th cell", round);
        let wait = match lim.check_key(&1u32) {
            Err(not_until) => not_until.wait_time_from(clock.now()),
            Ok(()) => panic!("round {}: bucket must be exhausted after 8 cells", round),
        };
        assert_eq!(Duration::from_millis(125), wait, "round {round}");
    }
}
