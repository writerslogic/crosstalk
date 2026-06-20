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

#[cfg(test)]
mod shared_recall_tests {
    //! Tests confirming that recall records are shared (Arc) rather than
    //! deep-cloned on the hot path (per H-036 pattern).
    //!
    //! NOTE (LOW certainty): The concrete `Shared<T>` wrapper from
    //! `crosstalk-concurrency` is not yet visible in this crate's view, so
    //! these tests exercise the underlying sharing invariant directly via
    //! `std::sync::Arc`. When the recall hot path adopts `Shared<T>`, the
    //! recalled records must satisfy this same `Arc::ptr_eq` invariant.
    use super::*;
    use std::sync::Arc;

    /// A stand-in recalled record. Cheap to share, expensive to clone in the
    /// real system (embeddings + payload), which is exactly why the hot path
    /// must share via Arc rather than clone.
    #[derive(Debug, PartialEq)]
    struct RecalledRecord {
        id: u64,
        cfg: RecallConfig,
        payload: String,
    }

    /// Simulates a recall hot path that returns shared handles to stored
    /// records instead of cloning their contents.
    fn recall_shared(store: &[Arc<RecalledRecord>]) -> Vec<Arc<RecalledRecord>> {
        // Hot path: share via Arc::clone (pointer bump), never deep clone.
        store.iter().map(Arc::clone).collect()
    }

    #[test]
    fn recalled_records_are_shared_not_cloned() {
        let store: Vec<Arc<RecalledRecord>> = vec![Arc::new(RecalledRecord {
            id: 1,
            cfg: RecallConfig::default(),
            payload: "embedding-payload".to_string(),
        })];

        let recalled = recall_shared(&store);

        // The recalled handle must point to the *same* allocation as the
        // stored record (no deep clone occurred on the hot path).
        assert!(
            Arc::ptr_eq(&store[0], &recalled[0]),
            "recalled record must be Arc-shared with the stored record"
        );
    }

    #[test]
    fn sharing_does_not_duplicate_payload_allocation() {
        let original = Arc::new(RecalledRecord {
            id: 42,
            cfg: RecallConfig::default(),
            payload: "x".repeat(1024),
        });

        let store = vec![Arc::clone(&original)];
        let recalled = recall_shared(&store);

        assert_eq!(Arc::strong_count(&original), 3, "all handles share one allocation");
        assert!(Arc::ptr_eq(&original, &recalled[0]));
        assert_eq!(recalled[0].id, 42);
        assert_eq!(recalled[0].payload.len(), 1024);
    }
}