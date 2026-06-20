//! Memory recall configuration.
//!
//! Magic literals for recall tuning have been swept into
//! `crosstalk_core::consts`.

use crosstalk_core::consts::{DEFAULT_RECALL_LIMIT, DEFAULT_RECALL_THRESHOLD};

/// Parameters controlling a recall query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallConfig {
    /// Maximum number of results returned.
    pub limit: usize,
    /// Minimum similarity threshold (0.0..=1.0).
    pub threshold: f32,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            limit: DEFAULT_RECALL_LIMIT,
            threshold: DEFAULT_RECALL_THRESHOLD,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_named_consts() {
        let cfg = RecallConfig::default();
        assert_eq!(cfg.limit, DEFAULT_RECALL_LIMIT);
        assert_eq!(cfg.threshold, DEFAULT_RECALL_THRESHOLD);
    }
}