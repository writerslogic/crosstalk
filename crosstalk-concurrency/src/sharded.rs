#[cfg(test)]
mod recall_arc_sharing_tests {
    use super::*;
    use std::sync::Arc as StdArc;

    // CERTAIN: recall must return an Arc that shares the SAME inner allocation
    // as the one held internally — i.e. the inner record V is NOT cloned, only
    // the Arc handle is reference-counted. This is the hot-tier zero-copy
    // contract (criterion: "recall returns Arc with zero clone of inner record").
    #[test]
    fn recall_shares_inner_record_without_cloning() {
        let map: Sharded<u32, Vec<u8>> = Sharded::new();
        map.insert(99, vec![9, 8, 7, 6]);

        // Two independent recalls.
        let a: StdArc<Vec<u8>> = map.recall(&99).expect("present");
        let b: StdArc<Vec<u8>> = map.recall(&99).expect("present");

        // The two Arcs must point at the exact same heap allocation, proving
        // the inner Vec<u8> record was shared (not deep-cloned) across recalls.
        assert!(
            StdArc::ptr_eq(&a, &b),
            "recall must share the inner record via Arc, not clone it"
        );

        // Cross-path consistency: get() shares the same allocation too.
        let c = map.get(&99).expect("present");
        assert!(StdArc::ptr_eq(&a, &c));

        // Strong count reflects shared handles plus the one retained internally.
        // (3 outstanding clones a/b/c + 1 stored = 4). We assert a lower bound
        // to stay robust against internal bookkeeping changes.
        assert!(
            StdArc::strong_count(&a) >= 4,
            "expected shared Arc refcount to reflect outstanding handles"
        );
    }

    // CERTAIN: recall on an absent key allocates nothing and returns None.
    #[test]
    fn recall_absent_key_returns_none() {
        let map: Sharded<u32, String> = Sharded::new();
        assert!(map.recall(&12345).is_none());
    }
}
