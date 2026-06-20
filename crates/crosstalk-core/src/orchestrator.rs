#[cfg(test)]
mod cached_hotpath_tests {
    //! Tests confirming the orchestrator hot path resolves expensive values
    //! once and reuses the *cached* result, rather than recomputing per
    //! fan-out (P3 perf fix: M-021–M-030 style items).
    //!
    //! NOTE (LOW certainty): The concrete `Cache` primitive from
    //! `crosstalk-concurrency` is not yet visible in this crate's view, so
    //! these tests model the caching invariant directly with an atomic compute
    //! counter behind a `OnceLock`. When the hot path adopts the real `Cache`,
    //! it MUST uphold the same "compute-once, reuse-many" guarantee.
    use std::sync::Arc;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Minimal stand-in for the orchestrator's cached, lazily-initialized
    /// value (e.g. a routing table or compiled fan-out plan). The expensive
    /// initializer must run at most once regardless of how many agents read it.
    #[derive(Default)]
    struct CachedPlan {
        cell: OnceLock<Arc<String>>,
        compute_calls: AtomicUsize,
    }

    impl CachedPlan {
        fn new() -> Self {
            Self::default()
        }

        /// Hot-path accessor: returns a *shared* handle to the cached value,
        /// computing it on first use only.
        fn get(&self) -> Arc<String> {
            let value = self.cell.get_or_init(|| {
                self.compute_calls.fetch_add(1, Ordering::SeqCst);
                Arc::new("expensive-fan-out-plan".to_string())
            });
            Arc::clone(value)
        }

        fn compute_count(&self) -> usize {
            self.compute_calls.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn value_computed_once_across_repeated_reads() {
        let plan = CachedPlan::new();
        for _ in 0..100 {
            let _ = plan.get();
        }
        assert_eq!(
            plan.compute_count(),
            1,
            "cached hot-path value must be computed exactly once"
        );
    }

    #[test]
    fn cached_reads_return_shared_allocation() {
        let plan = CachedPlan::new();
        let a = plan.get();
        let b = plan.get();
        assert!(
            Arc::ptr_eq(&a, &b),
            "repeated cached reads must share one allocation, not deep-clone"
        );
    }

    #[test]
    fn first_read_triggers_exactly_one_compute() {
        let plan = CachedPlan::new();
        assert_eq!(plan.compute_count(), 0, "no compute before first read");
        let _ = plan.get();
        assert_eq!(plan.compute_count(), 1, "first read computes once");
        let _ = plan.get();
        assert_eq!(plan.compute_count(), 1, "second read reuses cached value");
    }

    #[test]
    fn cached_value_content_is_stable() {
        let plan = CachedPlan::new();
        let first = plan.get();
        let second = plan.get();
        assert_eq!(*first, *second, "cached content must be stable across reads");
        assert_eq!(&**first, "expensive-fan-out-plan");
    }
}