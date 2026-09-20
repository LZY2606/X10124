#![cfg(feature = "std")]
#![allow(dead_code)] // small test-only clock; not every method is used by every test file

//! Deterministic test-only clock shared by the integration tests.
//!
//! Unlike [`governor::clock::FakeRelativeClock`], this clock can also be
//! moved backwards, which lets the state-machine tests exercise clock
//! adjustments (including saturating rewinds) without changing any
//! production code.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use governor::clock::{Clock, ReasonablyRealtime};
use governor::nanos::Nanos;

#[derive(Clone, Default, Debug)]
pub struct ManualClock {
    now: Arc<AtomicU64>,
}

impl ManualClock {
    pub fn new(nanos: u64) -> Self {
        Self {
            now: Arc::new(AtomicU64::new(nanos)),
        }
    }

    /// Move the clock forward by `by` (nanoseconds saturate at u64::MAX).
    pub fn advance(&self, by: Duration) {
        self.now.fetch_add(by.as_nanos() as u64, Ordering::AcqRel);
    }

    /// Move the clock backwards by `by`, saturating at the epoch (0).
    pub fn rewind(&self, by: Duration) {
        let by = by.as_nanos() as u64;
        let mut current = self.now.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(by);
            match self
                .now
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn set(&self, nanos: u64) {
        self.now.store(nanos, Ordering::Release);
    }

    pub fn as_nanos(&self) -> u64 {
        self.now.load(Ordering::Acquire)
    }
}

impl Clock for ManualClock {
    type Instant = Nanos;

    fn now(&self) -> Nanos {
        Nanos::from(self.now.load(Ordering::Acquire))
    }
}

impl ReasonablyRealtime for ManualClock {}
