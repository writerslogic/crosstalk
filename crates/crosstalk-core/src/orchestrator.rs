//! Orchestrator scheduling configuration.
//!
//! Magic literals for scheduling have been swept into [`crate::consts`].

use std::time::Duration;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_named_consts() {
        let cfg = OrchestratorConfig::default();
        assert_eq!(cfg.max_concurrent_agents, DEFAULT_MAX_CONCURRENT_AGENTS);
        assert_eq!(cfg.max_retries, DEFAULT_MAX_RETRIES);
        assert_eq!(cfg.agent_timeout, DEFAULT_AGENT_TIMEOUT);
    }
}