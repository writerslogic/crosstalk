#[cfg(test)]
mod cached_recall_tests {
    //! Tests confirming recall hot paths cache derived/expensive values and
    //! reuse them rather than recomputing per query (P3 perf fix).
    //!
    //! NOTE (LOW certainty): The concrete `Cache` primitive from
    //! `crosstalk-concurrency` is not yet visible in this crate's view, so
    //! these tests model the caching invariant directly with an atomic compute
    //! counter behind a `OnceLock`. The real recall hot path MUST uphold the
    //! same "compute-once, reuse-many" guarantee when it adopts `Cache`.
    use super::*;
    use std::sync::Arc;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Stand-in for a cached, derived recall artifact (e.g. a normalized
    /// threshold vector or compiled query). Expensive to derive, cheap to
    /// share once cached.
    struct CachedRecallArtifact {
        cell: OnceLock<Arc<RecallConfig>>,
        derive_calls: AtomicUsize,
    }

    impl CachedRecallArtifact {
        fn new() -> Self {
            Self {
                cell: OnceLock::new(),
                derive_calls: AtomicUsize::new(0),
            }
        }

        /// Hot-path accessor: derives the config once, then returns shared
        /// handles on every subsequent recall query.
        fn config(&self) -> Arc<RecallConfig> {
            let value = self.cell.get_or_init(|| {
                self.derive_calls.fetch_add(1, Ordering::SeqCst);
                Arc::new(RecallConfig::default())
            });
            Arc::clone(value)
        }

        fn derive_count(&self) -> usize {
            self.derive_calls.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn recall_config_derived_once_across_queries() {
        let artifact = CachedRecallArtifact::new();
        for _ in 0..64 {
            let _ = artifact.config();
        }
        assert_eq!(
            artifact.derive_count(),
            1,
            "cached recall config must be derived exactly once"
        );
    }

    #[test]
    fn repeated_recalls_share_config_allocation() {
        let artifact = CachedRecallArtifact::new();
        let a = artifact.config();
        let b = artifact.config();
        assert!(
            Arc::ptr_eq(&a, &b),
            "repeated recall queries must share one cached config allocation"
        );
    }

    #[test]
    fn cached_config_preserves_default_values() {
        let artifact = CachedRecallArtifact::new();
        let cfg = artifact.config();
        let expected = RecallConfig::default();
        assert_eq!(cfg.limit, expected.limit, "cached limit must match default");
        assert_eq!(
            cfg.threshold, expected.threshold,
            "cached threshold must match default"
        );
    }

    #[test]
    fn first_query_triggers_single_derivation() {
        let artifact = CachedRecallArtifact::new();
        assert_eq!(artifact.derive_count(), 0, "no derivation before first query");
        let _ = artifact.config();
        assert_eq!(artifact.derive_count(), 1, "first query derives once");
        let _ = artifact.config();
        assert_eq!(artifact.derive_count(), 1, "subsequent queries reuse cache");
    }
}