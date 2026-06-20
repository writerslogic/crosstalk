//! Centralized named constants, swept from magic literals across the codebase.
//!
//! This module collects previously-inlined magic literals into named,
//! documented constants. Behavior is unchanged; this is a low-risk cleanup.

use std::time::Duration;

/// Default maximum number of memory recall results returned to a caller.
pub const DEFAULT_RECALL_LIMIT: usize = 10;

/// Minimum similarity score (0.0..=1.0) for a recall result to be considered relevant.
pub const DEFAULT_RECALL_THRESHOLD: f32 = 0.75;

/// Default maximum number of concurrent agents the orchestrator will schedule.
pub const DEFAULT_MAX_CONCURRENT_AGENTS: usize = 4;

/// Default number of retry attempts for a failed orchestration step.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default per-request timeout for orchestrated agent calls.
pub const DEFAULT_AGENT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default MCP gateway listen port.
pub const DEFAULT_GATEWAY_PORT: u16 = 8787;

/// Default MCP gateway request timeout.
pub const DEFAULT_GATEWAY_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum allowed inbound payload size (in bytes) for the MCP gateway.
pub const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consts_are_exported_and_sane() {
        // Reference at least one const to confirm exports compile and link.
        assert_eq!(DEFAULT_RECALL_LIMIT, 10);
        assert!((0.0..=1.0).contains(&DEFAULT_RECALL_THRESHOLD));
        assert!(DEFAULT_MAX_CONCURRENT_AGENTS >= 1);
        assert!(DEFAULT_MAX_RETRIES >= 1);
        assert_eq!(DEFAULT_AGENT_TIMEOUT, Duration::from_secs(30));
        assert_eq!(DEFAULT_GATEWAY_PORT, 8787);
        assert_eq!(DEFAULT_GATEWAY_TIMEOUT, Duration::from_secs(60));
        assert_eq!(MAX_PAYLOAD_BYTES, 1024 * 1024);
    }
}