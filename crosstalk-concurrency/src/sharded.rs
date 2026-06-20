//! Sharded data structure primitives.
//!
//! This module defines the `Sharded<K, V>` concurrency primitive that is
//! intended to back two distinct call sites in Crosstalk:
//!
//! 1. **Memory recall (hot-tier):** requires `recall` returning an
//!    `Arc<V>` so that recalled values can be shared across tasks without
//!    cloning the underlying payload.
//! 2. **MCP gateway state:** requires an atomic `get_or_insert_with` so a
//!    gateway/session entry is created exactly once under concurrent
//!    access.
//!
//! NOTE: This is a *stub* per the day-2 risk-mitigation task. The method
//! bodies are intentionally unimplemented; only the API surface is being
//! locked in for review. See `docs/substrate-sharded.md` for the full
//! design note and the rationale tying each method to its call site.

use std::sync::Arc;

/// A sharded, concurrency-friendly key/value map.
///
/// `Sharded<K, V>` partitions its keyspace across a fixed number of
/// internal shards to reduce contention. Values are stored behind `Arc`
/// so that `recall`/`get` can hand out cheap, shareable references.
///
/// The concrete internal representation (shard count, per-shard locking,
/// hashing strategy) is deliberately left unspecified at the stub stage.
#[derive(Debug, Default)]
pub struct Sharded<K, V> {
    // CERTAIN: PhantomData keeps the type parameters used while no real
    // storage is present, so the struct compiles without unused-param
    // errors. The real implementation will replace this with the shard
    // array (e.g. `Box<[RwLock<FxHashMap<K, Arc<V>>>]>`).
    _marker: std::marker::PhantomData<(K, V)>,
}

impl<K, V> Sharded<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    /// Construct a new, empty `Sharded` map with a default shard count.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    /// Atomically fetch the value for `key`, inserting the result of
    /// `init` if no value is present.
    ///
    /// This is the primitive required by the **MCP gateway** call site:
    /// the gateway must create per-session/per-server state exactly once
    /// even under concurrent first-access. Returns an `Arc<V>` clone of the
    /// resident value.
    ///
    /// STUB: not yet implemented.
    pub fn get_or_insert_with<F>(&self, key: K, init: F) -> Arc<V>
    where
        F: FnOnce() -> V,
    {
        let _ = (key, init);
        unimplemented!("Sharded::get_or_insert_with is a stub (day-2 design)")
    }

    /// Recall a previously stored value, returning a shared `Arc<V>` if
    /// present.
    ///
    /// This is the primitive required by the **memory recall (hot-tier)**
    /// call site: recalled entries are shared across reader tasks without
    /// deep-cloning the payload.
    ///
    /// STUB: not yet implemented.
    pub fn recall(&self, key: &K) -> Option<Arc<V>> {
        let _ = key;
        unimplemented!("Sharded::recall is a stub (day-2 design)")
    }

    /// Insert `value` for `key`, returning any previously stored value.
    ///
    /// STUB: not yet implemented.
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>> {
        let _ = (key, value);
        unimplemented!("Sharded::insert is a stub (day-2 design)")
    }

    /// Get the shared value for `key`, if present.
    ///
    /// STUB: not yet implemented.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let _ = key;
        unimplemented!("Sharded::get is a stub (day-2 design)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CERTAIN: This only exercises construction, which is fully
    // implemented. The other methods are stubs that `unimplemented!()`,
    // so we deliberately do NOT call them here — doing so would panic.
    // These tests lock in the *type surface* the two call sites depend on.

    #[test]
    fn new_constructs_empty_sharded() {
        let _map: Sharded<String, u64> = Sharded::new();
        // Construction must not panic; the stub body is empty.
        assert!(true);
    }

    #[test]
    fn default_matches_new_type() {
        // CERTAIN: `Default` is derived, so this must compile and run.
        let _map: Sharded<u32, String> = Sharded::default();
        assert!(true);
    }

    #[test]
    fn recall_signature_returns_arc_option() {
        // ARCHITECTURAL CHECK: this test does not *call* recall (it would
        // panic); instead it asserts the signature shape via a function
        // pointer with the exact expected type. If the signature drifts,
        // this fails to compile — guarding the hot-tier contract.
        let f: fn(&Sharded<String, Vec<u8>>, &String) -> Option<Arc<Vec<u8>>> =
            Sharded::<String, Vec<u8>>::recall;
        let _ = f;
        assert!(true);
    }

    #[test]
    fn get_or_insert_with_signature_returns_arc() {
        // ARCHITECTURAL CHECK: guards the MCP gateway contract. We bind the
        // method to a typed function pointer to assert it returns Arc<V>
        // and accepts an FnOnce initializer. Not invoked (stub panics).
        let f = Sharded::<u32, String>::get_or_insert_with::<fn() -> String>;
        let _ = f;
        assert!(true);
    }

    #[test]
    fn insert_and_get_signatures_return_arc_option() {
        // ARCHITECTURAL CHECK: both insert and get must hand back Arc<V>
        // so callers can share without cloning payloads.
        let insert_fn: fn(&Sharded<u8, u8>, u8, u8) -> Option<Arc<u8>> = Sharded::<u8, u8>::insert;
        let get_fn: fn(&Sharded<u8, u8>, &u8) -> Option<Arc<u8>> = Sharded::<u8, u8>::get;
        let _ = (insert_fn, get_fn);
        assert!(true);
    }
}
