//! Cancellation token primitives.
//!
//! Provides [`CancelScope`], a structured-concurrency helper that wraps a
//! [`CancellationToken`] together with a [`TaskTracker`]. All background work
//! is expected to be spawned through [`CancelScope::spawn`], which guarantees
//! that the scope can be shut down gracefully — cancelling the token and then
//! waiting for every tracked task to complete.
//!
//! Addresses H-040 (structured shutdown of background work) and partially
//! H-038 (cancellation propagation).

use std::future::Future;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// A structured cancellation scope.
///
/// `CancelScope` couples a [`CancellationToken`] with a [`TaskTracker`] so that
/// all background tasks spawned via [`CancelScope::spawn`] can be cancelled and
/// awaited together. This guarantees structured shutdown: once
/// [`CancelScope::shutdown_graceful`] returns, no tracked task is still running.
///
/// Cloning a `CancelScope` yields a handle to the *same* underlying token and
/// tracker, so cancellation and tracking are shared across clones.
#[derive(Clone, Debug)]
pub struct CancelScope {
    token: CancellationToken,
    tracker: TaskTracker,
}

impl CancelScope {
    /// Create a new, empty cancellation scope with a fresh token and tracker.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            tracker: TaskTracker::new(),
        }
    }

    /// Create a child scope whose token is derived from this scope's token.
    ///
    /// Cancelling the parent cancels the child, but cancelling the child does
    /// not affect the parent. The child uses its own [`TaskTracker`].
    pub fn child(&self) -> Self {
        Self {
            token: self.token.child_token(),
            tracker: TaskTracker::new(),
        }
    }

    /// Return a clone of the underlying [`CancellationToken`].
    ///
    /// Tasks can use this to observe cancellation, e.g. via
    /// [`CancellationToken::cancelled`].
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Returns `true` if this scope's token has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Cancel the scope's token without waiting for tasks to finish.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Spawn a future onto the Tokio runtime, tracked by this scope.
    ///
    /// The returned [`JoinHandle`] can be used to await the individual task,
    /// but tasks are also collectively awaited by
    /// [`CancelScope::shutdown_graceful`].
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.tracker.spawn(future)
    }

    /// Gracefully shut down the scope.
    ///
    /// This cancels the token (signalling all cooperating tasks to stop),
    /// closes the tracker so no further tasks can be spawned, and then waits
    /// for every tracked task to complete.
    pub async fn shutdown_graceful(&self) {
        self.token.cancel();
        self.tracker.close();
        self.tracker.wait().await;
    }
}

impl Default for CancelScope {
    fn default() -> Self {
        Self::new()
    }
}

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
}
