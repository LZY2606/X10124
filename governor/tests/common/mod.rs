//! Shared test helpers: a fully manual clock that can advance *and* rewind,
//! so state-machine tests can exercise time travelling backwards without any
//! production-code changes.
#![allow(dead_code)]

use governor::clock::{Clock, ReasonablyRealtime};
use governor::nanos::Nanos;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A deterministic, manually-driven clock.
///
/// Unlike [`governor::clock::FakeRelativeClock`], this clock can also move
/// backwards (saturating at zero), which lets tests simulate clock
/// adjustments that a GCRA implementation must tolerate.
#[derive(Debug, Clone, Default)]
pub struct ManualClock {
    now: Arc<AtomicU64>,
}

impl ManualClock {
    /// Advances the clock by `ns` nanoseconds.
    pub fn advance(&self, ns: u64) {
        self.now.fetch_add(ns, Ordering::Relaxed);
    }

    /// Moves the clock backwards by `ns` nanoseconds, saturating at zero.
    pub fn rewind(&self, ns: u64) {
        let mut current = self.now.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(ns);
            match self.now.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// The current time in nanoseconds since the clock's (zero) start point.
    pub fn now_ns(&self) -> u64 {
        self.now.load(Ordering::Relaxed)
    }
}

impl Clock for ManualClock {
    type Instant = Nanos;

    fn now(&self) -> Nanos {
        Nanos::new(self.now_ns())
    }
}

/// Allows the manual clock to be used with the `until_ready` family of
/// async methods. The reference point is simply "now".
impl ReasonablyRealtime for ManualClock {}
