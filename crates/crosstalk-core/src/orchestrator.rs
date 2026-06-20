//! Orchestrator scheduling configuration.
//!
//! Magic literals for scheduling have been swept into [`crate::consts`].

use std::time::Duration;

use crosstalk_concurrency::CancelScope;

use crate::consts::{
    DEFAULT_AGENT_TIMEOUT, DEFAULT_MAX_CONCURRENT_AGENTS, DEFAULT_MAX_RETRIES,
};

/// Tunable parameters controlling orchestration behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrchestratorConfig {
    /// Maximum number of agents scheduled concurrently.
    pub max_concurrent_agents: usize,
    /// Retry attempts for a failed step.
    pub max_retries: u32,
    /// Per-agent request timeout.
    pub agent_timeout: Duration,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_agents: DEFAULT_MAX_CONCURRENT_AGENTS,
            max_retries: DEFAULT_MAX_RETRIES,
            agent_timeout: DEFAULT_AGENT_TIMEOUT,
        }
    }
}

/// A background orchestrator that owns a cancellable [`CancelScope`].
///
/// All background work is spawned via [`Orchestrator::spawn`], so that
/// [`Orchestrator::shutdown`] cancels every in-flight child task.
pub struct Orchestrator {
    config: OrchestratorConfig,
    scope: CancelScope,
}

impl Orchestrator {
    /// Create a new orchestrator with the given configuration and a fresh
    /// root cancellation scope.
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config,
            scope: CancelScope::new(),
        }
    }

    /// Configuration backing this orchestrator.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Borrow the orchestrator's cancellation scope.
    pub fn scope(&self) -> &CancelScope {
        &self.scope
    }

    /// Spawn a background task that is cancelled on [`Orchestrator::shutdown`].
    ///
    /// Tasks should observe their [`CancelScope::token`] (or the parent scope
    /// token) to react to cancellation.
    pub fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.scope.spawn(future)
    }

    /// Returns whether the orchestrator's scope has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.scope.is_cancelled()
    }

    /// Shut the orchestrator down, cancelling all in-flight background tasks.
    pub fn shutdown(&self) {
        self.scope.cancel();
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new(OrchestratorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn default_uses_named_consts() {
        let cfg = OrchestratorConfig::default();
        assert_eq!(cfg.max_concurrent_agents, DEFAULT_MAX_CONCURRENT_AGENTS);
        assert_eq!(cfg.max_retries, DEFAULT_MAX_RETRIES);
        assert_eq!(cfg.agent_timeout, DEFAULT_AGENT_TIMEOUT);
    }

    #[test]
    fn new_orchestrator_starts_uncancelled() {
        let orch = Orchestrator::default();
        assert!(!orch.is_cancelled());
    }

    #[test]
    fn shutdown_marks_scope_cancelled() {
        let orch = Orchestrator::default();
        assert!(!orch.is_cancelled());
        orch.shutdown();
        assert!(orch.is_cancelled());
    }

    #[tokio::test]
    async fn background_task_observes_cancellation_after_shutdown() {
        let orch = Orchestrator::default();
        let observed = Arc::new(AtomicBool::new(false));

        let token = orch.scope().token();
        let observed_clone = Arc::clone(&observed);

        // Spawn a long-lived task that should react to cancellation.
        let handle = orch.spawn(async move {
            // Wait until the scope is cancelled.
            token.cancelled().await;
            observed_clone.store(true, Ordering::SeqCst);
        });

        // Ensure the task is actually running and not yet cancelled.
        tokio::task::yield_now().await;
        assert!(!observed.load(Ordering::SeqCst));
        assert!(!orch.is_cancelled());

        // Trigger shutdown -> cancellation propagates to the child task.
        orch.shutdown();
        assert!(orch.is_cancelled());

        // The task should complete (observe cancellation) promptly.
        let _ = handle.await;
        assert!(
            observed.load(Ordering::SeqCst),
            "background task should observe cancellation after shutdown"
        );
    }
}