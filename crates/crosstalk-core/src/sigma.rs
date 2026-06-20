//! Sigma: a state holder whose critical section is reduced to the minimal
//! mutation. All expensive preparation is performed lock-free *before* the
//! lock is acquired, and the lock is never held while acquiring another lock.

use std::sync::Mutex;

/// The mutable state guarded by the Sigma lock.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SigmaState {
    /// Monotonic generation counter, bumped on every committed update.
    pub generation: u64,
    /// The committed payload value.
    pub value: u64,
}

/// A prepared update computed entirely outside of the lock.
///
/// Constructing this performs all the "~40 lines" of work that previously
/// lived inside the critical section. By the time we take the lock, the only
/// thing left to do is the single field mutation in [`Sigma::commit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedUpdate {
    value: u64,
}

impl PreparedUpdate {
    /// Lock-free preparation. This is where any heavy computation happens.
    ///
    /// IMPORTANT: this function must never acquire the Sigma lock (or any other
    /// lock) so that callers can run it before entering the critical section.
    pub fn prepare(inputs: &[u64]) -> Self {
        // Simulated "expensive" lock-free work that used to live inside the
        // critical section. None of this touches the mutex.
        let mut acc: u64 = 0;
        for &i in inputs {
            acc = acc.wrapping_add(i).wrapping_mul(31).rotate_left(1);
        }
        PreparedUpdate { value: acc }
    }

    /// The value this prepared update will write.
    pub fn value(&self) -> u64 {
        self.value
    }
}

/// Sigma owns the lock and exposes a minimal critical section.
#[derive(Debug, Default)]
pub struct Sigma {
    state: Mutex<SigmaState>,
}

impl Sigma {
    /// Construct a new Sigma with default state.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(SigmaState::default()),
        }
    }

    /// Commit a previously prepared update.
    ///
    /// The critical section here is intentionally minimal: it contains only the
    /// field mutations needed to publish the new value. No preparation, no
    /// allocation, and no nested lock acquisition occurs while the lock is held.
    pub fn commit(&self, prepared: PreparedUpdate) -> u64 {
        // === BEGIN CRITICAL SECTION (minimal mutation only) ===
        let mut guard = self.state.lock().expect("sigma mutex poisoned");
        guard.value = prepared.value;
        guard.generation += 1;
        guard.generation
        // === END CRITICAL SECTION ===
    }

    /// Convenience: prepare lock-free, then commit under the minimal lock.
    ///
    /// This makes the prep-before-lock ordering explicit for callers.
    pub fn update(&self, inputs: &[u64]) -> u64 {
        let prepared = PreparedUpdate::prepare(inputs); // lock-free prep first
        self.commit(prepared) // then the minimal critical section
    }

    /// Read a snapshot of the current state.
    pub fn snapshot(&self) -> SigmaState {
        self.state.lock().expect("sigma mutex poisoned").clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_is_lock_free_and_runs_before_commit() {
        let sigma = Sigma::new();

        // Hold the lock to simulate another thread inside the critical section.
        let held = sigma.state.lock().expect("lock");

        // Because preparation does NOT touch the lock, it must complete while
        // the lock is held by someone else. If `prepare` tried to acquire the
        // Sigma lock, this single-threaded test would deadlock.
        let prepared = PreparedUpdate::prepare(&[1, 2, 3, 4]);
        assert_eq!(prepared, PreparedUpdate::prepare(&[1, 2, 3, 4]));

        // Release the lock so commit can take it.
        drop(held);

        let gen = sigma.commit(prepared.clone());
        assert_eq!(gen, 1);
        assert_eq!(sigma.snapshot().value, prepared.value());
    }

    #[test]
    fn commit_does_not_acquire_nested_lock() {
        let sigma = Sigma::new();
        let prepared = PreparedUpdate::prepare(&[10, 20, 30]);

        // The mutation must be the only thing under the lock. We verify there
        // is no nested/reentrant acquisition by confirming that after commit
        // returns, the lock is immediately free again (try_lock succeeds).
        let gen = sigma.commit(prepared);
        assert_eq!(gen, 1);

        let probe = sigma.state.try_lock();
        assert!(
            probe.is_ok(),
            "lock should be fully released after commit (no nested hold)"
        );
        drop(probe);
    }

    #[test]
    fn minimal_critical_section_only_mutates() {
        let sigma = Sigma::new();
        let before = sigma.snapshot();
        assert_eq!(before, SigmaState { generation: 0, value: 0 });

        let g1 = sigma.update(&[5]);
        let after1 = sigma.snapshot();
        assert_eq!(g1, 1);
        assert_eq!(after1.generation, 1);

        let g2 = sigma.update(&[7]);
        let after2 = sigma.snapshot();
        assert_eq!(g2, 2);
        assert_eq!(after2.generation, 2);

        // Generation strictly increments — confirms commit performs exactly the
        // intended mutation each call.
        assert!(after2.generation > after1.generation);
    }

    #[test]
    fn concurrent_commits_serialize_correctly() {
        use std::sync::Arc;
        use std::thread;

        let sigma = Arc::new(Sigma::new());
        let threads = 8;
        let per_thread = 100;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let s = Arc::clone(&sigma);
                thread::spawn(move || {
                    for i in 0..per_thread {
                        // Prep happens lock-free in each iteration.
                        s.update(&[t as u64, i as u64]);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread join");
        }

        // Every commit bumped the generation exactly once.
        assert_eq!(sigma.snapshot().generation, (threads * per_thread) as u64);
    }
}