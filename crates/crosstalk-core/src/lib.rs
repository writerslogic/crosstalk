//! Core types and shared infrastructure for Crosstalk.

pub mod consts;
pub mod error;
pub mod orchestrator;

pub use error::{CrossTalkError, Result};

#[cfg(test)]
mod tests {
    // Confirm the consts module is re-exported and reachable from the crate root.
    use super::*;
    use crate::consts::{
        DEFAULT_AGENT_TIMEOUT, DEFAULT_GATEWAY_PORT, DEFAULT_GATEWAY_TIMEOUT,
        DEFAULT_MAX_CONCURRENT_AGENTS, DEFAULT_MAX_RETRIES, DEFAULT_RECALL_LIMIT,
        DEFAULT_RECALL_THRESHOLD, MAX_PAYLOAD_BYTES,
    };

    #[test]
    fn consts_module_is_reachable() {
        assert!(DEFAULT_RECALL_LIMIT > 0);
    }

    #[test]
    fn recall_limit_has_expected_default() {
        // Contract from T-023: DEFAULT_RECALL_LIMIT == 10
        assert_eq!(DEFAULT_RECALL_LIMIT, 10);
    }

    #[test]
    fn recall_threshold_is_within_valid_range() {
        // Contract from T-023: DEFAULT_RECALL_THRESHOLD == 0.75
        assert!((0.0..=1.0).contains(&DEFAULT_RECALL_THRESHOLD));
        assert!((DEFAULT_RECALL_THRESHOLD - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn concurrency_defaults_are_positive() {
        assert!(DEFAULT_MAX_CONCURRENT_AGENTS > 0);
        assert_eq!(DEFAULT_MAX_CONCURRENT_AGENTS, 4);
    }

    #[test]
    fn retry_default_is_expected() {
        assert_eq!(DEFAULT_MAX_RETRIES, 3);
    }

    #[test]
    fn timeout_defaults_are_nonzero() {
        assert!(!DEFAULT_AGENT_TIMEOUT.is_zero());
        assert!(!DEFAULT_GATEWAY_TIMEOUT.is_zero());
        assert_eq!(DEFAULT_AGENT_TIMEOUT.as_secs(), 30);
        assert_eq!(DEFAULT_GATEWAY_TIMEOUT.as_secs(), 60);
    }

    #[test]
    fn gateway_port_has_expected_default() {
        assert_eq!(DEFAULT_GATEWAY_PORT, 8787);
    }

    #[test]
    fn max_payload_bytes_is_one_megabyte() {
        assert_eq!(MAX_PAYLOAD_BYTES, 1024 * 1024);
    }

    #[test]
    fn result_alias_is_usable() {
        // Exercise the re-exported `Result` alias to ensure it is reachable
        // and that the `Ok` path type-checks.
        fn ok_value() -> Result<u32> {
            Ok(42)
        }
        assert_eq!(ok_value().unwrap(), 42);
    }

    #[test]
    fn crosstalk_error_can_be_constructed_via_from() {
        // `CrossTalkError` is re-exported; ensure the name resolves at the
        // crate root. We only verify the type is nameable to avoid coupling
        // to a specific variant signature.
        fn _accepts_error(_e: CrossTalkError) {}
        // Compile-time reachability check only; no runtime assertion needed.
    }
}