//! Multi-agent orchestrator.
//!
//! All background tasks spawned by the orchestrator are routed through a
//! [`CancelScope`] so they participate in structured cancellation and
//! graceful shutdown. When [`Orchestrator::shutdown`] is called, the scope is
//! cancelled and all tracked tasks are awaited.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crosstalk_concurrency::cancel::CancelScope;
use tokio::task::JoinHandle;

/// Coordinates background agent tasks under a single structured-cancellation
/// scope.
///
/// Every background task MUST be spawned via [`Orchestrator::spawn`] (which
/// delegates to [`CancelScope::spawn`]) so that it is cancelled and awaited
/// during [`Orchestrator::shutdown`].
#[derive(Clone)]
pub struct Orchestrator {
    scope: CancelScope,
    spawned: Arc<AtomicUsize>,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    /// Create a new orchestrator with a fresh cancellation scope.
    pub fn new() -> Self {
        Self {
            scope: CancelScope::new(),
            spawned: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Spawn a background task on the orchestrator's cancellation scope.
    ///
    /// The returned [`JoinHandle`] resolves to the task's output. The task is
    /// tracked by the underlying [`CancelScope`] and will be cancelled and
    /// awaited during [`Orchestrator::shutdown`].
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.spawned.fetch_add(1, Ordering::SeqCst);
        self.scope.spawn(future)
    }

    /// Returns the orchestrator's cancellation token, so cooperating tasks can
    /// observe shutdown requests.
    pub fn cancel_token(&self) -> tokio_util::sync::CancellationToken {
        self.scope.token()
    }

    /// Returns `true` if the orchestrator has been cancelled / shut down.
    pub fn is_cancelled(&self) -> bool {
        self.scope.is_cancelled()
    }

    /// Total number of background tasks spawned through this orchestrator.
    pub fn spawned_count(&self) -> usize {
        self.spawned.load(Ordering::SeqCst)
    }

    /// Create a child orchestrator whose scope is cancelled when this
    /// orchestrator's scope is cancelled.
    pub fn child(&self) -> Orchestrator {
        Orchestrator {
            scope: self.scope.child(),
            spawned: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Request cancellation of all background tasks without awaiting them.
    pub fn cancel(&self) {
        self.scope.cancel();
    }

    /// Gracefully shut down the orchestrator: cancel all background tasks and
    /// await their completion.
    pub async fn shutdown(&self) {
        self.scope.shutdown_graceful().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // CERTAIN: A new orchestrator is not cancelled and has spawned nothing.
    #[tokio::test]
    async fn new_orchestrator_is_idle_and_not_cancelled() {
        let orch = Orchestrator::new();
        assert!(!orch.is_cancelled());
        assert_eq!(orch.spawned_count(), 0);
    }

    // CERTAIN: Default impl matches `new`.
    #[tokio::test]
    async fn default_orchestrator_is_not_cancelled() {
        let orch = Orchestrator::default();
        assert!(!orch.is_cancelled());
        assert_eq!(orch.spawned_count(), 0);
    }

    // CERTAIN: spawn runs the task and returns its produced value.
    #[tokio::test]
    async fn spawn_runs_task_and_returns_value() {
        let orch = Orchestrator::new();
        let handle = orch.spawn(async { 21 * 2 });
        let value = handle.await.expect("task should not panic");
        assert_eq!(value, 42);
        assert_eq!(orch.spawned_count(), 1);
    }

    // CERTAIN: spawn increments the spawned counter for each task.
    #[tokio::test]
    async fn spawn_counts_tasks() {
        let orch = Orchestrator::new();
        for _ in 0..5 {
            orch.spawn(async {});
        }
        // Await graceful shutdown so the no-op tasks are reaped.
        orch.shutdown().await;
        assert_eq!(orch.spawned_count(), 5);
    }

    // CERTAIN: shutdown awaits all spawned background tasks (structured).
    #[tokio::test]
    async fn shutdown_awaits_all_background_tasks() {
        let orch = Orchestrator::new();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..8 {
            let c = Arc::clone(&counter);
            orch.spawn(async move {
                tokio::time::sleep(Duration::from_millis(5)).await;
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        orch.shutdown().await;

        assert_eq!(counter.load(Ordering::SeqCst), 8);
        assert!(orch.is_cancelled());
    }

    // CERTAIN: Cooperating background tasks observe cancellation via the token.
    #[tokio::test]
    async fn cooperating_task_stops_on_shutdown() {
        let orch = Orchestrator::new();
        let observed = Arc::new(AtomicUsize::new(0));

        let token = orch.cancel_token();
        let o = Arc::clone(&observed);
        orch.spawn(async move {
            token.cancelled().await;
            o.fetch_add(1, Ordering::SeqCst);
        });

        orch.shutdown().await;

        assert_eq!(observed.load(Ordering::SeqCst), 1);
    }

    // CERTAIN: shutdown is idempotent and safe to call multiple times.
    #[tokio::test]
    async fn shutdown_is_idempotent() {
        let orch = Orchestrator::new();
        orch.shutdown().await;
        orch.shutdown().await;
        assert!(orch.is_cancelled());
    }

    // CERTAIN: cancel() requests cancellation without awaiting tasks.
    #[tokio::test]
    async fn cancel_sets_flag_without_awaiting() {
        let orch = Orchestrator::new();
        orch.cancel();
        assert!(orch.is_cancelled());
        assert!(orch.cancel_token().is_cancelled());
    }

    // CERTAIN: A clone shares the same cancellation state.
    #[tokio::test]
    async fn clone_shares_cancellation_state() {
        let orch = Orchestrator::new();
        let clone = orch.clone();
        orch.cancel();
        assert!(clone.is_cancelled());
    }

    // CERTAIN: Tasks spawned on a clone are awaited by the original's
    // shutdown because the underlying scope tracker is shared.
    #[tokio::test]
    async fn cloned_orchestrator_shares_tracker() {
        let orch = Orchestrator::new();
        let clone = orch.clone();
        let counter = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&counter);
        clone.spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            c.fetch_add(1, Ordering::SeqCst);
        });

        orch.shutdown().await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // CERTAIN: A child orchestrator is cancelled when its parent is cancelled.
    #[tokio::test]
    async fn child_is_cancelled_by_parent() {
        let parent = Orchestrator::new();
        let child = parent.child();
        assert!(!child.is_cancelled());
        parent.cancel();
        assert!(child.is_cancelled());
    }

    // CERTAIN: Cancelling/shutting down a child does not affect the parent.
    #[tokio::test]
    async fn child_shutdown_does_not_affect_parent() {
        let parent = Orchestrator::new();
        let child = parent.child();
        let done = Arc::new(AtomicUsize::new(0));

        let token = child.cancel_token();
        let d = Arc::clone(&done);
        child.spawn(async move {
            token.cancelled().await;
            d.fetch_add(1, Ordering::SeqCst);
        });

        child.shutdown().await;

        assert_eq!(done.load(Ordering::SeqCst), 1);
        assert!(child.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    // CERTAIN: Cancellation propagates to all in-flight background tasks and
    // shutdown lets them all wind down cooperatively.
    #[tokio::test]
    async fn shutdown_propagates_to_all_background_tasks() {
        let orch = Orchestrator::new();
        let noticed = Arc::new(AtomicUsize::new(0));

        const N: usize = 16;
        for _ in 0..N {
            let token = orch.cancel_token();
            let n = Arc::clone(&noticed);
            orch.spawn(async move {
                loop {
                    tokio::select! {
                        _ = token.cancelled() => {
                            n.fetch_add(1, Ordering::SeqCst);
                            break;
                        }
                        _ = tokio::time::sleep(Duration::from_millis(1)) => {}
                    }
                }
            });
        }

        assert_eq!(noticed.load(Ordering::SeqCst), 0);

        orch.shutdown().await;

        assert_eq!(noticed.load(Ordering::SeqCst), N);
        assert!(orch.is_cancelled());
    }

    // CERTAIN: No background task outlives the orchestrator's shutdown.
    #[tokio::test]
    async fn no_background_task_outlives_shutdown() {
        struct DropGuard(Arc<AtomicUsize>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let orch = Orchestrator::new();
        let dropped = Arc::new(AtomicUsize::new(0));

        const N: usize = 10;
        for _ in 0..N {
            let token = orch.cancel_token();
            let guard = DropGuard(Arc::clone(&dropped));
            orch.spawn(async move {
                let _g = guard;
                token.cancelled().await;
            });
        }

        assert_eq!(dropped.load(Ordering::SeqCst), 0);

        orch.shutdown().await;

        assert_eq!(dropped.load(Ordering::SeqCst), N);
    }
}