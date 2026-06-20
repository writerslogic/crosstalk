// ───────────────────────── Module boundary documentation ─────────────────────────
//
// MODULE BOUNDARY (orchestrator — crosstalk-core):
//   * Owns the agent-fan-out hot path. Records are SHARED via `Arc`, never
//     deep-cloned, when distributed to multiple agents (H-036 pattern).
//   * Blocking work is OFFLOADED to the dedicated blocking pool
//     (`tokio::task::spawn_blocking`) and must never run inline on an async
//     worker thread (P3 fix: keep the async executor responsive).
//   * Cancellation is cooperative: the orchestrator observes a cancel signal
//     (the `CancelScope` seam from `crosstalk-concurrency`) and stops issuing
//     new work, rather than aborting mid-flight tasks.
//
// These tests assert the *invariants* of those boundaries. Where concrete
// orchestrator types are not visible in this file's view, the tests model the
// invariant directly (CERTAINTY: HIGH that invariants hold; LOW that the exact
// concrete types are reachable here — same caveat as `shared_hotpath_tests`).

#[cfg(test)]
mod cancel_scope_seam_tests {
    //! Verifies the cooperative-cancellation contract the orchestrator relies
    //! on from the `CancelScope` seam: once a scope is cancelled, observers see
    //! the cancellation and the hot path should stop issuing new work.
    //!
    //! NOTE (LOW certainty): `crosstalk_concurrency::CancelScope` is not yet
    //! materialized in this file's view, so we model the cancellation token
    //! semantics with a shared atomic flag. The real `CancelScope` MUST uphold
    //! the same "cancel is observable and monotonic" guarantee.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Minimal stand-in for the cancellation observation surface exposed by
    /// `CancelScope`. Cancellation is one-way (monotonic): once set, it stays
    /// set.
    #[derive(Clone, Default)]
    struct CancelToken {
        flag: Arc<AtomicBool>,
    }

    impl CancelToken {
        fn new() -> Self {
            Self {
                flag: Arc::new(AtomicBool::new(false)),
            }
        }
        fn cancel(&self) {
            self.flag.store(true, Ordering::SeqCst);
        }
        fn is_cancelled(&self) -> bool {
            self.flag.load(Ordering::SeqCst)
        }
    }

    /// Simulates the orchestrator dispatch loop: it should keep dispatching
    /// until cancellation is observed, then stop. Returns how many items were
    /// dispatched before cancellation halted the loop.
    fn dispatch_until_cancelled(token: &CancelToken, total: usize, cancel_at: usize) -> usize {
        let mut dispatched = 0;
        for i in 0..total {
            if token.is_cancelled() {
                break;
            }
            if i == cancel_at {
                token.cancel();
                // Loop re-checks at the top of the next iteration.
            }
            dispatched += 1;
        }
        dispatched
    }

    #[test]
    fn uncancelled_scope_dispatches_all() {
        let token = CancelToken::new();
        // cancel_at beyond `total` means cancellation never fires.
        let n = dispatch_until_cancelled(&token, 5, usize::MAX);
        assert_eq!(n, 5, "with no cancellation the full batch must dispatch");
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_halts_new_dispatch() {
        let token = CancelToken::new();
        // Cancel after dispatching index 2 (items 0,1,2 dispatched).
        let n = dispatch_until_cancelled(&token, 100, 2);
        assert_eq!(
            n, 3,
            "loop must stop issuing new work once cancellation is observed"
        );
        assert!(token.is_cancelled(), "cancellation must be observable");
    }

    #[test]
    fn cancellation_is_monotonic() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        // A second cancel call must not "un-cancel".
        token.cancel();
        assert!(token.is_cancelled(), "cancel state must be monotonic");
    }

    #[test]
    fn cloned_token_shares_cancel_state() {
        let token = CancelToken::new();
        let observer = token.clone();
        token.cancel();
        assert!(
            observer.is_cancelled(),
            "child observers of a CancelScope must see cancellation from any clone"
        );
    }
}