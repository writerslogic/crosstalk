#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // CERTAIN: A fresh scope must not start cancelled.
    #[tokio::test]
    async fn new_scope_is_not_cancelled() {
        let scope = CancelScope::new();
        assert!(!scope.is_cancelled());
        assert!(!scope.token().is_cancelled());
    }

    // CERTAIN: Default impl is equivalent to `new`.
    #[tokio::test]
    async fn default_scope_is_not_cancelled() {
        let scope = CancelScope::default();
        assert!(!scope.is_cancelled());
    }

    // CERTAIN: token() returns a token wired to the same cancellation state.
    #[tokio::test]
    async fn token_shares_cancellation_state() {
        let scope = CancelScope::new();
        let token = scope.token();
        assert!(!token.is_cancelled());
        scope.cancel();
        assert!(token.is_cancelled());
        assert!(scope.is_cancelled());
    }

    // CERTAIN: cancel() sets the cancelled flag.
    #[tokio::test]
    async fn cancel_sets_flag() {
        let scope = CancelScope::new();
        scope.cancel();
        assert!(scope.is_cancelled());
    }

    // CERTAIN: A spawned task runs to completion and its JoinHandle yields
    // the produced value.
    #[tokio::test]
    async fn spawn_runs_task_and_returns_value() {
        let scope = CancelScope::new();
        let handle = scope.spawn(async { 21 * 2 });
        let value = handle.await.expect("task should not panic");
        assert_eq!(value, 42);
    }

    // CERTAIN: shutdown_graceful waits for all spawned tasks to finish,
    // and cancels the token.
    #[tokio::test]
    async fn shutdown_graceful_waits_for_tasks() {
        let scope = CancelScope::new();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..8 {
            let c = Arc::clone(&counter);
            scope.spawn(async move {
                // A short async pause to ensure tasks are genuinely in flight.
                tokio::time::sleep(Duration::from_millis(5)).await;
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        scope.shutdown_graceful().await;

        assert_eq!(counter.load(Ordering::SeqCst), 8);
        assert!(scope.is_cancelled());
    }

    // CERTAIN: Cooperating tasks observe cancellation via the token, allowing
    // graceful shutdown to complete even for "long-running" loops.
    #[tokio::test]
    async fn cooperating_task_stops_on_cancellation() {
        let scope = CancelScope::new();
        let observed_cancel = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&observed_cancel);
        let token = scope.token();
        scope.spawn(async move {
            // Wait until cancellation is requested.
            token.cancelled().await;
            c.fetch_add(1, Ordering::SeqCst);
        });

        scope.shutdown_graceful().await;

        assert_eq!(observed_cancel.load(Ordering::SeqCst), 1);
    }

    // CERTAIN: After graceful shutdown, the tracker is closed; spawning more
    // work via a closed tracker still returns a handle but is conceptually
    // out of scope. We at least verify shutdown is idempotent and safe.
    #[tokio::test]
    async fn shutdown_graceful_is_idempotent() {
        let scope = CancelScope::new();
        scope.shutdown_graceful().await;
        // Calling again must not panic or hang.
        scope.shutdown_graceful().await;
        assert!(scope.is_cancelled());
    }

    // CERTAIN: A child scope is cancelled when its parent is cancelled.
    #[tokio::test]
    async fn child_is_cancelled_by_parent() {
        let parent = CancelScope::new();
        let child = parent.child();
        assert!(!child.is_cancelled());
        parent.cancel();
        assert!(child.is_cancelled());
    }

    // CERTAIN: Cancelling a child does not cancel the parent.
    #[tokio::test]
    async fn child_cancel_does_not_affect_parent() {
        let parent = CancelScope::new();
        let child = parent.child();
        child.cancel();
        assert!(child.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    // CERTAIN: Clones share the same cancellation state.
    #[tokio::test]
    async fn clone_shares_state() {
        let scope = CancelScope::new();
        let clone = scope.clone();
        scope.cancel();
        assert!(clone.is_cancelled());
    }

    // CERTAIN: Tasks spawned via a clone are awaited by the original's
    // shutdown, since the tracker is shared between clones.
    #[tokio::test]
    async fn cloned_scope_shares_tracker() {
        let scope = CancelScope::new();
        let clone = scope.clone();
        let counter = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&counter);
        clone.spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            c.fetch_add(1, Ordering::SeqCst);
        });

        // Shut down via the original scope; it must still wait for the
        // task spawned on the clone because the tracker is shared.
        scope.shutdown_graceful().await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // CERTAIN: Cancellation propagates into multiple in-flight spawned tasks.
    // Each task loops cooperatively until it observes cancellation, then
    // records that it noticed. shutdown_graceful must let them all finish.
    #[tokio::test]
    async fn cancel_propagates_to_all_spawned_tasks() {
        let scope = CancelScope::new();
        let noticed = Arc::new(AtomicUsize::new(0));

        const N: usize = 16;
        for _ in 0..N {
            let token = scope.token();
            let n = Arc::clone(&noticed);
            scope.spawn(async move {
                // Cooperative loop: keep working until the token fires.
                loop {
                    tokio::select! {
                        _ = token.cancelled() => {
                            n.fetch_add(1, Ordering::SeqCst);
                            break;
                        }
                        _ = tokio::time::sleep(Duration::from_millis(1)) => {
                            // simulate work
                        }
                    }
                }
            });
        }

        // No task should have observed cancellation yet.
        assert_eq!(noticed.load(Ordering::SeqCst), 0);

        // Trigger cancellation and wait for all tasks to wind down.
        scope.shutdown_graceful().await;

        // Every task must have observed the cancellation signal.
        assert_eq!(noticed.load(Ordering::SeqCst), N);
        assert!(scope.is_cancelled());
    }

    // CERTAIN: No tracked task outlives the scope's graceful shutdown.
    // We use a guard whose Drop increments a counter; after
    // shutdown_graceful returns, all guards must have been dropped,
    // proving the tasks fully completed (and did not outlive the scope).
    #[tokio::test]
    async fn no_task_outlives_scope_after_cancel_shutdown() {
        struct DropGuard(Arc<AtomicUsize>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let scope = CancelScope::new();
        let dropped = Arc::new(AtomicUsize::new(0));

        const N: usize = 10;
        for _ in 0..N {
            let token = scope.token();
            let guard = DropGuard(Arc::clone(&dropped));
            scope.spawn(async move {
                // Move the guard into the task so its lifetime is bound
                // to the task. It is dropped exactly when the task ends.
                let _g = guard;
                token.cancelled().await;
            });
        }

        // While tasks are alive and waiting, no guard has dropped.
        assert_eq!(dropped.load(Ordering::SeqCst), 0);

        scope.shutdown_graceful().await;

        // After graceful shutdown, every task ended -> every guard dropped.
        // This proves no spawned task is still running past the scope.
        assert_eq!(dropped.load(Ordering::SeqCst), N);
    }

    // CERTAIN: A child scope can be shut down independently; its tasks
    // observe the child's cancellation and complete before shutdown returns.
    #[tokio::test]
    async fn child_scope_cancel_shutdown_completes_tasks() {
        let parent = CancelScope::new();
        let child = parent.child();
        let done = Arc::new(AtomicUsize::new(0));

        let token = child.token();
        let d = Arc::clone(&done);
        child.spawn(async move {
            token.cancelled().await;
            d.fetch_add(1, Ordering::SeqCst);
        });

        child.shutdown_graceful().await;

        assert_eq!(done.load(Ordering::SeqCst), 1);
        assert!(child.is_cancelled());
        // The parent must remain unaffected by the child's shutdown.
        assert!(!parent.is_cancelled());
    }
}
