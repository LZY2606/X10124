//! Shared helpers for governor's integration tests.
//!
//! This module is not a test target of its own; it is included via
//! `mod common;` from the integration tests in this directory.
#![allow(dead_code)]

use governor::clock::Clock;
use governor::nanos::Nanos;
use std::convert::TryInto;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A fully deterministic, manually driven clock.
///
/// Unlike [`governor::clock::FakeRelativeClock`], this clock can also
/// move *backwards* (saturating at zero), which lets tests exercise the
/// limiter's behavior when a clock is rewound (e.g. NTP adjustments).
/// Clones share the same time.
#[derive(Debug, Clone, Default)]
pub struct RewindableClock {
    now: Arc<AtomicU64>,
}

impl RewindableClock {
    /// Moves the clock forward by `by`.
    pub fn advance(&self, by: Duration) {
        let by: u64 = by
            .as_nanos()
            .try_into()
            .expect("unreasonably large duration");
        self.now.fetch_add(by, Ordering::SeqCst);
    }

    /// Moves the clock backwards by `by`, saturating at zero.
    pub fn rewind(&self, by: Duration) {
        let by: u64 = by
            .as_nanos()
            .try_into()
            .expect("unreasonably large duration");
        let mut current = self.now.load(Ordering::SeqCst);
        loop {
            let next = current.saturating_sub(by);
            match self
                .now
                .compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }
}

impl Clock for RewindableClock {
    type Instant = Nanos;

    fn now(&self) -> Nanos {
        self.now.load(Ordering::SeqCst).into()
    }
}
