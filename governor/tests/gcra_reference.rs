#![cfg(feature = "std")]

//! Differential state-machine tests for the GCRA implementation.
//!
//! These tests drive a deliberately simple independent reference model
//! ([`RefGcra`]) and the library's direct and keyed limiters through the
//! exact same sequences of operations. After every single step they
//! compare:
//!
//! * whether the request was admitted (`Ok` / `Err` / `InsufficientCapacity`),
//! * on a rejection, the earliest time the request can be retried,
//! * the theoretical arrival time (GCRA's stored state) via recording
//!   state stores that observe every CAS closure invocation, and
//! * the remaining burst capacity reported by the state-information
//!   middleware on positive decisions.
//!
//! The sequences cover forward clock movement, clock rewinds, single
//! cells, batches that exactly consume the burst, batches larger than the
//! burst, batches that only partially fit, and per-key cleanup. Random
//! sequences use fixed seeds; on failure the failing operation prefix is
//! shrunken to the shortest sequence that still reproduces the mismatch.

mod common;

use std::fmt::{self, Write};
use std::hash::Hash;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use governor::middleware::StateInformationMiddleware;
use governor::nanos::Nanos;
use governor::state::keyed::{DashMapStateStore, HashMapStateStore, ShrinkableKeyedStateStore};
use governor::state::{InMemoryState, NotKeyed, StateStore};
use governor::{InsufficientCapacity, Quota, RateLimiter};
use nonzero_ext::nonzero;

use common::ManualClock;

// ---------------------------------------------------------------------------
// Test-only state stores that observe the state the GCRA writes.
// ---------------------------------------------------------------------------

/// What the rate limiter observed on the *last* CAS closure invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Observation {
    /// The old state the closure was called with (`None` for a fresh key).
    old: Option<u64>,
    /// The state the closure wants to store. On a rejection this equals
    /// the unchanged old state (0 for a fresh key); for an oversized batch
    /// no closure runs at all, so no observation is recorded.
    next: u64,
    /// Whether the closure admitted (`true`) or rejected (`false`).
    admitted: bool,
}

/// Wrap a GCRA decision closure so every invocation is recorded. Under
/// CAS contention the closure may run several times; the tracker ends up
/// holding the *final* invocation, which matches the caller-visible result.
///
/// The state store demands an `Fn` (it invokes the closure through a shared
/// reference inside its CAS loop), while the GCRA decision code is generic
/// over `FnMut`; the inner closure therefore lives in a `RefCell` and the
/// observation is published through the mutex after each invocation.
fn observe<'a, T, F, E>(
    last: &'a Mutex<Option<Observation>>,
    f: F,
) -> impl Fn(Option<Nanos>) -> Result<(T, Nanos), E> + 'a
where
    F: FnMut(Option<Nanos>) -> Result<(T, Nanos), E> + 'a,
{
    let f = std::cell::RefCell::new(f);
    move |old| {
        let result = f.borrow_mut()(old);
        let old_ns = old.map(Nanos::as_u64);
        *last.lock().unwrap() = Some(match &result {
            Ok((_, new_data)) => Observation {
                old: old_ns,
                next: new_data.as_u64(),
                admitted: true,
            },
            Err(_) => Observation {
                old: old_ns,
                next: old_ns.unwrap_or(0),
                admitted: false,
            },
        });
        result
    }
}

/// Shared recording state. The same `Arc` is handed to the rate limiter
/// and retained by the test so the last CAS invocation can be inspected.
#[derive(Default)]
struct Recorder {
    last: Mutex<Option<Observation>>,
}

impl Recorder {
    fn take(&self) -> Option<Observation> {
        self.last.lock().unwrap().take()
    }
}

/// Direct state store recording the final closure invocation.
#[derive(Clone, Default)]
struct RecordingDirect {
    inner: Arc<InMemoryState>,
    rec: Arc<Recorder>,
}

impl StateStore for RecordingDirect {
    type Key = NotKeyed;

    fn measure_and_replace<T, F, E>(&self, key: &NotKeyed, f: F) -> Result<T, E>
    where
        F: FnMut(Option<Nanos>) -> Result<(T, Nanos), E>,
    {
        self.inner
            .measure_and_replace(key, observe(&self.rec.last, f))
    }
}

/// Keyed state store wrapper recording the final closure invocation.
struct RecordingKeyed<K, S> {
    inner: Arc<S>,
    rec: Arc<Recorder>,
    marker: std::marker::PhantomData<fn() -> K>,
}

impl<K, S> Clone for RecordingKeyed<K, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            rec: self.rec.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<K, S: Default> Default for RecordingKeyed<K, S> {
    fn default() -> Self {
        Self {
            inner: Arc::new(S::default()),
            rec: Arc::new(Recorder::default()),
            marker: std::marker::PhantomData,
        }
    }
}

impl<K, S> StateStore for RecordingKeyed<K, S>
where
    K: Eq + Hash + Clone,
    S: StateStore<Key = K>,
{
    type Key = K;

    fn measure_and_replace<T, F, E>(&self, key: &K, f: F) -> Result<T, E>
    where
        F: FnMut(Option<Nanos>) -> Result<(T, Nanos), E>,
    {
        self.inner
            .measure_and_replace(key, observe(&self.rec.last, f))
    }
}

impl<K, S> ShrinkableKeyedStateStore<K> for RecordingKeyed<K, S>
where
    K: Eq + Hash + Clone,
    S: ShrinkableKeyedStateStore<K>,
{
    fn retain_recent(&self, drop_below: Nanos) {
        self.inner.retain_recent(drop_below)
    }

    fn shrink_to_fit(&self) {
        self.inner.shrink_to_fit()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// The independent reference GCRA model.
// ---------------------------------------------------------------------------

/// A direct translation of the GCRA spec, written independently of the
/// library code. `tat` is the theoretical arrival time in nanoseconds
/// since limiter creation; `None` means no cell has been admitted yet.
#[derive(Clone)]
struct RefGcra {
    t: u64,
    tau: u64,
    tat: Option<u64>,
}

#[derive(Debug, PartialEq, Eq)]
enum RefDecision {
    /// Admitted; carries the new theoretical arrival time and the remaining
    /// burst capacity *after* the decision.
    Allowed { tat: u64, remaining: u32 },
    /// Rejected; `earliest` is when this request could conform, `tat` is
    /// the unchanged stored theoretical arrival time.
    Denied { earliest: u64, tat: u64 },
    /// The batch exceeds the burst size and can never conform.
    TooLarge { capacity: u32 },
}

impl RefGcra {
    fn new(quota: Quota) -> Self {
        let t = quota.replenish_interval().as_nanos().max(1) as u64;
        let tau = t * (quota.burst_size().get() as u64 - 1);
        RefGcra { t, tau, tat: None }
    }

    fn capacity(&self) -> u32 {
        1 + (self.tau / self.t) as u32
    }

    fn check(&mut self, now: u64, n: u32) -> RefDecision {
        let additional_weight = self.t * u64::from(n - 1);
        if additional_weight > self.tau {
            return RefDecision::TooLarge {
                capacity: self.capacity(),
            };
        }
        let tat = self.tat.unwrap_or(now);
        let earliest = tat
            .saturating_add(additional_weight)
            .saturating_sub(self.tau);
        if now < earliest {
            RefDecision::Denied { earliest, tat }
        } else {
            let next = tat.max(now) + self.t + additional_weight;
            self.tat = Some(next);
            let remaining = ((now + self.tau + self.t).saturating_sub(next) / self.t) as u32;
            RefDecision::Allowed {
                tat: next,
                remaining,
            }
        }
    }

    /// A cleanup pass removes exactly keys whose state is
    /// indistinguishable from a fresh bucket at `now`.
    fn is_stale(&self, now: u64) -> bool {
        match self.tat {
            Some(tat) => tat <= now.saturating_sub(self.t),
            None => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Operations, sequences and error reporting.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// Move the clock forward by `ns` nanoseconds.
    Advance(u64),
    /// Move the clock backwards by `ns` (saturating at the epoch).
    Rewind(u64),
    /// Check `n` cells at the current time.
    Check(u32),
    /// Run the keyed cleanup pass (keyed harness only).
    #[allow(dead_code)] // only exercised by the keyed KOp harness
    Retain,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Op::Advance(ns) => write!(f, "advance({ns}ns)"),
            Op::Rewind(ns) => write!(f, "rewind({ns}ns)"),
            Op::Check(n) => write!(f, "check_n({n})"),
            Op::Retain => write!(f, "retain_recent()"),
        }
    }
}

/// Render a short, replayable prefix of an operation sequence.
fn render_ops(ops: &[Op]) -> String {
    let mut out = String::new();
    for op in ops {
        let _ = writeln!(out, "            {op},");
    }
    out
}

// ---------------------------------------------------------------------------
// Library result normalization and comparison.
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum LibDecision {
    Allowed { remaining: u32 },
    Denied { earliest: u64 },
    TooLarge { capacity: u32 },
}

fn normalize_single(
    result: Result<governor::middleware::StateSnapshot, governor::NotUntil<Nanos>>,
) -> LibDecision {
    match result {
        Ok(snapshot) => LibDecision::Allowed {
            remaining: snapshot.remaining_burst_capacity(),
        },
        Err(negative) => LibDecision::Denied {
            earliest: negative.earliest_possible().as_u64(),
        },
    }
}

fn normalize_batch(
    result: Result<
        Result<governor::middleware::StateSnapshot, governor::NotUntil<Nanos>>,
        InsufficientCapacity,
    >,
) -> LibDecision {
    match result {
        Err(InsufficientCapacity(capacity)) => LibDecision::TooLarge { capacity },
        Ok(inner) => normalize_single(inner),
    }
}

fn assert_decisions_match(
    reference: &RefDecision,
    library: &LibDecision,
    observed: Option<Observation>,
    now: u64,
    label: &str,
    prefix: &[Op],
) {
    let mismatch = |detail: &str| -> ! {
        panic!(
            "{label}: {detail} at now={now}ns\nreference = {reference:?}\nlibrary   = {library:?}\nshortest reproducing sequence:\n{}",
            render_ops(prefix)
        );
    };

    match (reference, library) {
        (
            RefDecision::Allowed { tat, remaining },
            LibDecision::Allowed {
                remaining: got_remaining,
            },
        ) => {
            if remaining != got_remaining {
                mismatch("remaining burst capacity differs");
            }
            let obs = match observed {
                Some(obs) => obs,
                None => mismatch("admission recorded no state observation"),
            };
            if !obs.admitted {
                mismatch("recording store observed a rejecting closure");
            }
            if obs.next != *tat {
                mismatch("stored theoretical arrival time differs");
            }
        }
        (RefDecision::Denied { earliest, tat }, LibDecision::Denied { earliest: got }) => {
            if earliest != got {
                mismatch("earliest retry time differs");
            }
            let obs = match observed {
                Some(obs) => obs,
                None => mismatch("rejection recorded no state observation"),
            };
            if obs.admitted {
                mismatch("recording store observed an admitting closure");
            }
            // A rejection must leave the stored TAT exactly unchanged, and
            // that stored TAT must equal the reference model's.
            if obs.old != Some(*tat) || obs.next != *tat {
                mismatch("rejection modified (or misread) the stored TAT");
            }
        }
        (RefDecision::TooLarge { capacity }, LibDecision::TooLarge { capacity: got }) => {
            if capacity != got {
                mismatch("InsufficientCapacity limit differs");
            }
            if observed.is_some() {
                mismatch("an oversized batch must not touch the limiter state");
            }
        }
        _ => mismatch("decision kind differs"),
    }
}

// ---------------------------------------------------------------------------
// Direct harness
// ---------------------------------------------------------------------------

type DirectLimiter =
    RateLimiter<NotKeyed, RecordingDirect, ManualClock, StateInformationMiddleware>;

struct DirectHarness {
    clock: ManualClock,
    limiter: DirectLimiter,
    recorder: RecordingDirect,
    model: RefGcra,
}

impl DirectHarness {
    fn new(quota: Quota) -> Self {
        let clock = ManualClock::default();
        // The recorder and the limiter share the same Arc-backed store.
        let recorder = RecordingDirect::default();
        let limiter = RateLimiter::new(quota, recorder.clone(), clock.clone());
        let model = RefGcra::new(quota);
        Self {
            clock,
            limiter,
            recorder,
            model,
        }
    }

    fn now(&self) -> u64 {
        self.clock.as_nanos()
    }

    /// Runs one full sequence, comparing state after each check.
    fn run(&mut self, sequence: &Sequence) {
        let mut prefix: Vec<Op> = Vec::with_capacity(sequence.ops.len());
        for &op in &sequence.ops {
            prefix.push(op);
            match op {
                Op::Advance(ns) => self.clock.advance(Duration::from_nanos(ns)),
                Op::Rewind(ns) => self.clock.rewind(Duration::from_nanos(ns)),
                Op::Retain => panic!("retain_recent is only valid for keyed limiters"),
                Op::Check(n) => {
                    let nz = NonZeroU32::new(n).unwrap();
                    let library = normalize_batch(self.limiter.check_n(nz));
                    let observed = self.recorder.rec.take();
                    let now = self.now();
                    let reference = self.model.check(now, n);
                    assert_decisions_match(
                        &reference,
                        &library,
                        observed,
                        now,
                        sequence.name,
                        &prefix,
                    );
                }
            }
        }
    }
}

#[derive(Clone)]
struct Sequence {
    name: &'static str,
    ops: Vec<Op>,
}

// ---------------------------------------------------------------------------
// Keyed harness (generic over the inner keyed state store)
// ---------------------------------------------------------------------------

type KeyedLimiter<K, S> =
    RateLimiter<K, RecordingKeyed<K, S>, ManualClock, StateInformationMiddleware>;

struct KeyedHarness<K, S>
where
    K: Eq + Hash + Clone,
    S: StateStore<Key = K> + Default,
{
    clock: ManualClock,
    limiter: KeyedLimiter<K, S>,
    recorder: RecordingKeyed<K, S>,
    models: std::collections::BTreeMap<K, RefGcra>,
    quota: Quota,
}

impl<K, S> KeyedHarness<K, S>
where
    K: Eq + Hash + Clone + Ord + Copy + fmt::Debug,
    S: StateStore<Key = K> + Default,
{
    fn new(quota: Quota) -> Self {
        let clock = ManualClock::default();
        let recorder = RecordingKeyed::<K, S>::default();
        let limiter = RateLimiter::new(quota, recorder.clone(), clock.clone());
        Self {
            clock,
            limiter,
            recorder,
            models: std::collections::BTreeMap::new(),
            quota,
        }
    }

    fn now(&self) -> u64 {
        self.clock.as_nanos()
    }

    fn model_mut(&mut self, key: K) -> &mut RefGcra {
        let quota = self.quota;
        self.models
            .entry(key)
            .or_insert_with(|| RefGcra::new(quota))
    }

    fn run(&mut self, sequence: &KeyedSequence<K>)
    where
        S: ShrinkableKeyedStateStore<K>,
    {
        let mut prefix: Vec<KOp<K>> = Vec::with_capacity(sequence.ops.len());
        for &op in &sequence.ops {
            prefix.push(op);
            match op {
                KOp::Advance(ns) => self.clock.advance(Duration::from_nanos(ns)),
                KOp::Rewind(ns) => self.clock.rewind(Duration::from_nanos(ns)),
                KOp::Check { key, n } => {
                    let nz = NonZeroU32::new(n).unwrap();
                    let library = normalize_batch(self.limiter.check_key_n(&key, nz));
                    let observed = self.recorder.rec.take();
                    let now = self.now();
                    let reference = self.model_mut(key).check(now, n);
                    assert_keyed_decision(
                        sequence.name,
                        &key,
                        now,
                        &reference,
                        &library,
                        observed,
                        &prefix,
                    );
                }
                KOp::Retain => {
                    let now = self.now();
                    self.limiter.retain_recent();
                    self.models.retain(|key, model| {
                        let keep = !model.is_stale(now);
                        if keep {
                            // Surviving keys must keep their quota state.
                        } else {
                            let _ = key;
                        }
                        keep
                    });
                }
            }
        }
    }
}

fn assert_keyed_decision<K: fmt::Debug>(
    name: &str,
    key: &K,
    now: u64,
    reference: &RefDecision,
    library: &LibDecision,
    observed: Option<Observation>,
    prefix: &[KOp<K>],
) {
    let label = format!("{name} [key={key:?}]");
    let mismatch = |detail: &str| -> ! {
        panic!(
            "{label}: {detail} at now={now}ns\nreference = {reference:?}\nlibrary   = {library:?}\nshortest reproducing sequence:\n{}",
            render_keyed_ops(prefix)
        );
    };
    match (reference, library) {
        (RefDecision::Allowed { tat, remaining }, LibDecision::Allowed { remaining: got }) => {
            if remaining != got {
                mismatch("remaining burst capacity differs");
            }
            let obs = observed.unwrap_or_else(|| mismatch("admission recorded no observation"));
            if !obs.admitted {
                mismatch("recording store observed a rejecting closure");
            }
            if obs.next != *tat {
                mismatch("stored theoretical arrival time differs");
            }
        }
        (RefDecision::Denied { earliest, tat }, LibDecision::Denied { earliest: got }) => {
            if earliest != got {
                mismatch("earliest retry time differs");
            }
            let obs = observed.unwrap_or_else(|| mismatch("rejection recorded no observation"));
            if obs.admitted {
                mismatch("recording store observed an admitting closure");
            }
            if obs.old != Some(*tat) || obs.next != *tat {
                mismatch("rejection modified (or misread) the stored TAT");
            }
        }
        (RefDecision::TooLarge { capacity }, LibDecision::TooLarge { capacity: got }) => {
            if capacity != got {
                mismatch("InsufficientCapacity limit differs");
            }
            if observed.is_some() {
                mismatch("an oversized batch must not touch the limiter state");
            }
        }
        _ => mismatch("decision kind differs"),
    }
}

#[derive(Clone, Copy, Debug)]
enum KOp<K> {
    Advance(u64),
    Rewind(u64),
    Check { key: K, n: u32 },
    Retain,
}

impl<K: fmt::Debug> fmt::Display for KOp<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KOp::Advance(ns) => write!(f, "advance({ns}ns)"),
            KOp::Rewind(ns) => write!(f, "rewind({ns}ns)"),
            KOp::Check { key, n } => write!(f, "check_key({key:?}, {n})"),
            KOp::Retain => write!(f, "retain_recent()"),
        }
    }
}

fn render_keyed_ops<K: fmt::Debug>(ops: &[KOp<K>]) -> String {
    let mut out = String::new();
    for op in ops {
        let _ = writeln!(out, "            {op},");
    }
    out
}

#[derive(Clone)]
struct KeyedSequence<K> {
    name: &'static str,
    ops: Vec<KOp<K>>,
}

// ---------------------------------------------------------------------------
// Deterministic direct scenarios
// ---------------------------------------------------------------------------

// Burst of 5 cells, 10ns per cell: tau = 40ns.
const T: u64 = 10;
const BURST: u32 = 5;

fn quota() -> Quota {
    Quota::with_period(Duration::from_nanos(T))
        .unwrap()
        .allow_burst(nonzero!(BURST))
}

#[test]
fn direct_fresh_bucket_allows_exactly_the_burst_then_blocks() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "fresh bucket burst boundary",
        ops: vec![
            Op::Check(1), // allow, remaining 4, tat = 10
            Op::Check(1), // allow, remaining 3, tat = 20
            Op::Check(1), // allow, remaining 2, tat = 30
            Op::Check(1), // allow, remaining 1, tat = 40
            Op::Check(1), // allow (5th), remaining 0, tat = 50
            // The 6th cell is denied: only one cell is owed, so the wait
            // is exactly one cell period (`>=` vs `>` is exercised below):
            Op::Check(1), // denied at 0, earliest = 10
            Op::Advance(9),
            Op::Check(1), // denied one ns early at 9, earliest = 10
            Op::Advance(1),
            Op::Check(1), // allowed exactly at now=10, tat=20, remaining 3
        ],
    });
}

#[test]
fn direct_batch_exactly_consumes_full_burst_atomically() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "batch exactly the burst",
        ops: vec![
            Op::Check(BURST), // allow: tat = 50, remaining 0
            Op::Check(1),     // denied at 0: earliest 10 (one owed cell)
            // A batch that "counts as independent single calls" would have
            // admitted several of these sequentially; the atomic batch
            // must deny while only a partial recharge is available:
            Op::Advance(3 * T),
            Op::Check(BURST), // denied at 30: earliest=50 (full burst only at 50)
            Op::Check(2),     // allowed at 30: tat=50, remaining 0
            Op::Check(2),     // denied at 30: earliest=50
            Op::Advance(2 * T),
            Op::Check(2), // allowed at 50: tat=70, remaining 3
            Op::Check(1), // denied at 50: earliest=60
            Op::Advance(2 * T),
            Op::Check(3), // allowed at 70: tat=100, remaining 2
            Op::Check(1), // denied at 70: earliest=80
            Op::Advance(3 * T),
            Op::Check(BURST), // allowed at 100: exactly full burst again
        ],
    });
}

#[test]
fn direct_oversized_batch_is_rejected_without_touching_state() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "oversized batch",
        ops: vec![
            Op::Check(BURST + 1), // TooLarge(5), state untouched
            Op::Check(BURST + 5), // Still TooLarge(5)
            Op::Check(BURST),     // allowed: full burst intact, tat=50
            Op::Check(BURST + 1), // TooLarge even with no budget left
            Op::Advance(T),
            Op::Check(1), // denied at 10: earliest=20, proving nothing was consumed
        ],
    });
}

#[test]
fn direct_denied_batch_consumes_no_cells() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "denied batch is free",
        ops: vec![
            Op::Check(3), // allow at 0: tat=30, remaining 2
            Op::Check(3), // denied at 0: earliest=20 (needs 2 cells of room)
            Op::Check(2), // allow at 0: tat=50 (the failed 3-batch cost nothing)
            Op::Check(3), // denied at 0: earliest=40
            Op::Check(1), // denied at 0: earliest=10
        ],
    });
}

#[test]
fn direct_clock_rewind_does_not_grant_extra_burst() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "clock rewind",
        ops: vec![
            Op::Advance(100),
            Op::Check(BURST), // allow at 100: tat=150, remaining 0
            Op::Check(1),     // denied at 100: earliest=110
            Op::Rewind(50),   // now=50
            Op::Check(1),     // still denied: earliest=110
            Op::Rewind(100),  // saturating rewind to now=0
            Op::Check(1),     // still denied: earliest=110
            Op::Advance(109), // now=109
            Op::Check(1),     // denied one ns early: earliest=110
            Op::Advance(1),   // now=110
            Op::Check(1),     // allowed exactly: tat=120
        ],
    });
}

#[test]
fn direct_steady_rate_never_rejects_at_boundary() {
    let mut h = DirectHarness::new(quota());
    let mut ops = vec![Op::Check(1)];
    for _ in 0..20 {
        ops.push(Op::Advance(T));
        ops.push(Op::Check(1)); // exactly one cell replenished each tick
    }
    h.run(&Sequence {
        name: "steady rate at the boundary",
        ops,
    });
}

#[test]
fn direct_long_pause_restores_no_more_than_the_burst() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "long pause restores exactly the burst",
        ops: vec![
            Op::Check(BURST), // tat=50
            Op::Advance(100_000),
            Op::Check(BURST),     // allow at 100000: tat=100050
            Op::Check(BURST + 1), // TooLarge
            Op::Check(1),         // denied at 100000: earliest=100010
        ],
    });
}

#[test]
fn direct_single_cell_arriving_exactly_at_earliest_is_admitted() {
    // Every check happens *exactly* one cell period after the previous
    // admission, i.e. precisely at the reference's earliest conforming
    // instant. A `>` (in place of `>=`) comparison rejects all of them.
    let mut h = DirectHarness::new(quota());
    let mut ops = vec![Op::Check(1)]; // tat=10
    for _ in 0..15 {
        ops.push(Op::Advance(T));
        ops.push(Op::Check(1));
    }
    h.run(&Sequence {
        name: "single cell exactly at the boundary (>= vs >)",
        ops,
    });
}

#[test]
fn direct_batch_arriving_exactly_at_earliest_is_admitted() {
    let mut h = DirectHarness::new(quota());
    h.run(&Sequence {
        name: "batch exactly at the boundary (>= vs >)",
        ops: vec![
            Op::Check(BURST), // tat=50
            // arrive exactly when the *full burst* fits again:
            Op::Advance(5 * T),
            Op::Check(BURST), // must admit at now=50, tat=100
            // and a partial batch arriving exactly when only it fits:
            Op::Advance(2 * T),
            Op::Check(2), // at 70: earliest=100 for a full, but 2 fit: tat=90
            Op::Advance(2 * T),
            Op::Check(2), // at 90, exactly conforming: tat=110
        ],
    });
}

#[test]
fn direct_single_cell_burst_of_one_is_strictly_sequential() {
    let q = Quota::with_period(Duration::from_nanos(7))
        .unwrap()
        .allow_burst(nonzero!(1u32));
    let mut h = DirectHarness::new(q);
    h.run(&Sequence {
        name: "burst size one",
        ops: vec![
            Op::Check(1), // allow tat=7, remaining 0
            Op::Check(1), // denied at 0, earliest=7
            Op::Check(2), // TooLarge(1)
            Op::Advance(6),
            Op::Check(1), // denied at 6, earliest=7
            Op::Advance(1),
            Op::Check(1), // allow exactly at 7, tat=14
            Op::Check(1), // denied at 7, earliest=14
        ],
    });
}

// ---------------------------------------------------------------------------
// Deterministic keyed scenarios (run against both state stores)
// ---------------------------------------------------------------------------

fn keyed_isolation_sequence() -> KeyedSequence<u32> {
    KeyedSequence {
        name: "keyed state isolation",
        ops: vec![
            KOp::Check { key: 1, n: BURST }, // key1: tat=50
            KOp::Check { key: 2, n: 1 },     // key2 independent: tat=10
            KOp::Check { key: 1, n: 1 },     // key1 denied at 0: earliest=10
            KOp::Check { key: 2, n: 1 },     // key2 allowed at 0: tat=20
            KOp::Advance(3 * T),             // now=30
            KOp::Check { key: 1, n: 3 },     // key1 denied at 30: earliest=30
            KOp::Check { key: 2, n: BURST }, // key2 denied at 30: earliest=60
            KOp::Check { key: 3, n: BURST }, // fresh key3 allowed at 30: tat=80
            KOp::Advance(2 * T),             // now=50
            KOp::Check { key: 1, n: 1 },     // key1 allowed at 50: tat=60
            KOp::Check { key: 2, n: 1 },     // key2 denied at 50: earliest=60
            KOp::Check { key: 3, n: 1 },     // key3 denied at 50: earliest=40
        ],
    }
}

fn keyed_cleanup_sequence() -> KeyedSequence<u32> {
    KeyedSequence {
        name: "keyed cleanup removes only stale keys",
        ops: vec![
            // key1 spends its whole burst long before cleanup, so it is stale.
            KOp::Check { key: 1, n: BURST }, // key1 tat=50
            KOp::Advance(9 * T),             // now=90
            // key2 is used right before cleanup and still owes a cell,
            // so its state is definitely live (tat=100 > drop_below=80).
            KOp::Check { key: 2, n: 1 }, // key2 tat=100
            KOp::Retain,
            // Probe *immediately*, while key2 still owes a cell: a live
            // key that survived cleanup must deny here, whereas a wrongly
            // evicted (fresh) key would admit the whole burst. This is the
            // assertion that catches "cleaning one key damages another".
            KOp::Check { key: 2, n: 1 }, // live key2 denied at 90: earliest=100
            KOp::Check { key: 2, n: BURST }, // still denied at 90: earliest=100
            // key1 was stale and is fresh again (complete burst available):
            KOp::Check { key: 1, n: BURST }, // fresh key1 allowed at 90: tat=140
            // key1 is independent of key2's activity (cleanup isolation):
            KOp::Check { key: 1, n: 1 }, // denied at 90: earliest=100
            // Much later, both used keys are stale and collectable; a
            // previously unseen key then behaves exactly like a fresh key.
            KOp::Advance(20 * T),
            KOp::Retain,
            KOp::Check { key: 3, n: BURST }, // brand-new key allowed at 290
        ],
    }
}

#[test]
fn keyed_reference_diff_dashmap() {
    let mut h = KeyedHarness::<u32, DashMapStateStore<u32>>::new(quota());
    h.run(&keyed_isolation_sequence());
    let mut h = KeyedHarness::<u32, DashMapStateStore<u32>>::new(quota());
    h.run(&keyed_cleanup_sequence());
}

#[test]
fn keyed_reference_diff_hashmap() {
    let mut h = KeyedHarness::<u32, HashMapStateStore<u32>>::new(quota());
    h.run(&keyed_isolation_sequence());
    let mut h = KeyedHarness::<u32, HashMapStateStore<u32>>::new(quota());
    h.run(&keyed_cleanup_sequence());
}

// ---------------------------------------------------------------------------
// Fixed-seed randomized differential fuzzing with prefix shrinking
// ---------------------------------------------------------------------------

/// Tiny deterministic LCG (Numerical Recipes constants).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

fn random_direct_ops(seed: u64, count: usize) -> Vec<Op> {
    let mut rng = Rng(seed);
    let mut ops = Vec::with_capacity(count);
    for _ in 0..count {
        ops.push(match rng.below(10) {
            0..=3 => {
                // small forward steps, sometimes a long pause
                Op::Advance(if rng.below(5) == 0 {
                    rng.below(50) * T
                } else {
                    1 + rng.below(3 * T)
                })
            }
            4 => Op::Rewind(1 + rng.below(4 * T)),
            5 => Op::Check(1 + rng.below(BURST as u64 + 2) as u32),
            _ => Op::Check(1),
        });
    }
    ops
}

fn random_keyed_ops(seed: u64, count: usize) -> Vec<KOp<u32>> {
    let mut rng = Rng(seed);
    let mut ops = Vec::with_capacity(count);
    for _ in 0..count {
        ops.push(match rng.below(12) {
            0..=3 => KOp::Advance(1 + rng.below(3 * T)),
            4 => KOp::Rewind(1 + rng.below(4 * T)),
            5 => KOp::Retain,
            6 => KOp::Check {
                key: 1 + (rng.below(4) as u32),
                n: 1 + rng.below(BURST as u64 + 2) as u32,
            },
            _ => KOp::Check {
                key: 1 + (rng.below(4) as u32),
                n: 1,
            },
        });
    }
    ops
}

/// Replays a direct op prefix; returns Err with a message on mismatch.
fn replay_direct(quota: Quota, ops: &[Op]) -> Result<(), String> {
    let clock = ManualClock::default();
    let store = RecordingDirect::default();
    let limiter: DirectLimiter = RateLimiter::new(quota, store.clone(), clock.clone());
    let mut model = RefGcra::new(quota);
    for (idx, &op) in ops.iter().enumerate() {
        match op {
            Op::Advance(ns) => clock.advance(Duration::from_nanos(ns)),
            Op::Rewind(ns) => clock.rewind(Duration::from_nanos(ns)),
            Op::Retain => return Err("retain on direct limiter".into()),
            Op::Check(n) => {
                let nz = NonZeroU32::new(n).unwrap();
                let library = normalize_batch(limiter.check_n(nz));
                let observed = store.rec.take();
                let now = clock.as_nanos();
                let reference = model.check(now, n);
                if let Err(msg) = check_one(&reference, &library, observed, now, idx, ops.len()) {
                    return Err(msg);
                }
            }
        }
    }
    Ok(())
}

fn replay_keyed<S>(quota: Quota, ops: &[KOp<u32>]) -> Result<(), String>
where
    S: StateStore<Key = u32> + Default + ShrinkableKeyedStateStore<u32>,
{
    let clock = ManualClock::default();
    let store = RecordingKeyed::<u32, S>::default();
    let limiter: KeyedLimiter<u32, S> = RateLimiter::new(quota, store.clone(), clock.clone());
    let mut models = std::collections::BTreeMap::<u32, RefGcra>::new();
    for (idx, &op) in ops.iter().enumerate() {
        match op {
            KOp::Advance(ns) => clock.advance(Duration::from_nanos(ns)),
            KOp::Rewind(ns) => clock.rewind(Duration::from_nanos(ns)),
            KOp::Retain => {
                let now = clock.as_nanos();
                limiter.retain_recent();
                models.retain(|_, m| !m.is_stale(now));
            }
            KOp::Check { key, n } => {
                let nz = NonZeroU32::new(n).unwrap();
                let library = normalize_batch(limiter.check_key_n(&key, nz));
                let observed = store.rec.take();
                let now = clock.as_nanos();
                let model = models.entry(key).or_insert_with(|| RefGcra::new(quota));
                let reference = model.check(now, n);
                if let Err(msg) = check_one(&reference, &library, observed, now, idx, ops.len()) {
                    return Err(format!("key={key}: {msg}"));
                }
            }
        }
    }
    Ok(())
}

fn check_one(
    reference: &RefDecision,
    library: &LibDecision,
    observed: Option<Observation>,
    now: u64,
    idx: usize,
    total: usize,
) -> Result<(), String> {
    let obs_ok = match (reference, &library) {
        (RefDecision::Allowed { tat, remaining }, LibDecision::Allowed { remaining: got }) => {
            let obs = observed.ok_or("no observation")?;
            obs.admitted && obs.next == *tat && remaining == got
        }
        (RefDecision::Denied { earliest, tat }, LibDecision::Denied { earliest: got }) => {
            let obs = observed.ok_or("no observation")?;
            !obs.admitted && earliest == got && obs.old == Some(*tat) && obs.next == *tat
        }
        (RefDecision::TooLarge { capacity }, LibDecision::TooLarge { capacity: got }) => {
            capacity == got && observed.is_none()
        }
        _ => false,
    };
    if obs_ok {
        Ok(())
    } else {
        Err(format!(
            "mismatch at op {idx}/{total}, now={now}ns: reference={reference:?}, library={library:?}, observed={observed:?}"
        ))
    }
}

/// Greedy shrink: keep removing operations while the prefix still fails,
/// producing the shortest failing sequence.
fn shrink_direct(quota: Quota, mut ops: Vec<Op>, seed_name: &str) {
    let mut i = 0;
    while i < ops.len() {
        let mut candidate = ops.clone();
        candidate.remove(i);
        if replay_direct(quota, &candidate).is_err() {
            ops = candidate;
        } else {
            i += 1;
        }
    }
    let detail = replay_direct(quota, &ops).unwrap_err();
    panic!(
        "direct differential fuzz ({seed_name}) found a mismatch:\n{detail}\nshortest reproducing sequence:\n{}",
        render_ops(&ops)
    );
}

fn shrink_keyed<S>(quota: Quota, mut ops: Vec<KOp<u32>>, seed_name: &str)
where
    S: StateStore<Key = u32> + Default + ShrinkableKeyedStateStore<u32>,
{
    let mut i = 0;
    while i < ops.len() {
        let mut candidate = ops.clone();
        candidate.remove(i);
        if replay_keyed::<S>(quota, &candidate).is_err() {
            ops = candidate;
        } else {
            i += 1;
        }
    }
    let detail = replay_keyed::<S>(quota, &ops).unwrap_err();
    panic!(
        "keyed differential fuzz ({seed_name}) found a mismatch:\n{detail}\nshortest reproducing sequence:\n{}",
        render_keyed_ops(&ops)
    );
}

const FUZZ_OPS: usize = 800;

#[test]
fn direct_reference_fuzz_fixed_seeds() {
    let q = quota();
    for seed in 0..6u64 {
        let ops = random_direct_ops(seed, FUZZ_OPS);
        if let Err(_) = replay_direct(q, &ops) {
            shrink_direct(q, ops, &format!("seed={seed}"));
        }
    }
}

#[test]
fn keyed_reference_fuzz_dashmap_fixed_seeds() {
    let q = quota();
    for seed in 0..6u64 {
        let ops = random_keyed_ops(seed, FUZZ_OPS);
        if replay_keyed::<DashMapStateStore<u32>>(q, &ops).is_err() {
            shrink_keyed::<DashMapStateStore<u32>>(q, ops, &format!("seed={seed}, store=DashMap"));
        }
    }
}

#[test]
fn keyed_reference_fuzz_hashmap_fixed_seeds() {
    let q = quota();
    for seed in 0..6u64 {
        let ops = random_keyed_ops(seed, FUZZ_OPS);
        if replay_keyed::<HashMapStateStore<u32>>(q, &ops).is_err() {
            shrink_keyed::<HashMapStateStore<u32>>(q, ops, &format!("seed={seed}, store=HashMap"));
        }
    }
}

// Also fuzz a burst size of one, where tau=0 and every boundary is exact.
#[test]
fn direct_reference_fuzz_burst_one() {
    let q = Quota::with_period(Duration::from_nanos(3))
        .unwrap()
        .allow_burst(nonzero!(1u32));
    for seed in 100..106u64 {
        let ops = random_direct_ops(seed, FUZZ_OPS);
        if replay_direct(q, &ops).is_err() {
            shrink_direct(q, ops, &format!("seed={seed}, burst=1"));
        }
    }
}
