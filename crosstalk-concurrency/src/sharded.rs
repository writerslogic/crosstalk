//! Sharded data structure primitives.
//!
//! This module defines the `Sharded<K, V>` concurrency primitive that backs
//! two distinct call sites in Crosstalk:
//!
//! 1. **Memory recall (hot-tier):** `recall` returns an `Arc<V>` so recalled
//!    values are shared across tasks without cloning the underlying payload.
//! 2. **MCP gateway state:** atomic `get_or_insert_with` so a gateway/session
//!    entry is created exactly once under concurrent access (no TOCTOU).
//!
//! Bounded growth is provided by per-shard LRU eviction (H-047, H-036,
//! SYS-006). See `docs/substrate-sharded.md` for the design note.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// CERTAIN: Fixed shard count. A power of two keeps the modulo cheap and the
// distribution even for well-mixed hashes.
const SHARD_COUNT: usize = 16;

// CERTAIN: Default per-shard capacity bounds total growth at
// SHARD_COUNT * SHARD_CAPACITY entries, satisfying the "bounded" requirement.
const SHARD_CAPACITY: usize = 1024;

/// Internal per-shard storage. Each entry pairs the `Arc<V>` payload with a
/// monotonically increasing "tick" recording its last access, used to choose
/// the least-recently-used victim on eviction.
struct Shard<K, V> {
    map: HashMap<K, (Arc<V>, u64)>,
    capacity: usize,
}

impl<K, V> Shard<K, V>
where
    K: Hash + Eq + Clone,
{
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            capacity,
        }
    }

    /// Evict the least-recently-used entry while the shard exceeds capacity.
    fn evict_if_needed(&mut self) {
        // CERTAIN: capacity == 0 means "unbounded" would be surprising; we
        // treat capacity 0 as "store nothing" by clearing back down. With a
        // positive capacity this loop runs at most once per insert.
        while self.map.len() > self.capacity {
            // Find the key with the smallest tick (least recently used).
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, (_, tick))| *tick)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    self.map.remove(&k);
                }
                None => break,
            }
        }
    }
}

/// A sharded, concurrency-friendly key/value map with LRU eviction.
///
/// `Sharded<K, V>` partitions its keyspace across a fixed number of internal
/// shards to reduce contention. Values are stored behind `Arc` so that
/// `recall`/`get` can hand out cheap, shareable references. Each shard is
/// bounded; the least-recently-used entry is evicted when the bound is
/// exceeded.
pub struct Sharded<K, V> {
    shards: Vec<Mutex<Shard<K, V>>>,
    // CERTAIN: A global access tick. Relaxed ordering is sufficient: we only
    // need a per-entry recency ordering, not cross-thread synchronization of
    // other state (the shard Mutex provides that).
    tick: AtomicU64,
}

impl<K, V> std::fmt::Debug for Sharded<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sharded")
            .field("shards", &self.shards.len())
            .finish()
    }
}

impl<K, V> Default for Sharded<K, V>
where
    K: Hash + Eq + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Sharded<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Construct a new, empty `Sharded` map with the default shard count and
    /// per-shard capacity.
    pub fn new() -> Self {
        Self::with_capacity(SHARD_CAPACITY)
    }

    /// Construct a new `Sharded` map with a specific per-shard capacity.
    ///
    /// CERTAIN: Kept as an additive helper; it does not alter any contract
    /// signature, so it stays compatible with the locked-in API surface.
    pub fn with_capacity(per_shard_capacity: usize) -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(Mutex::new(Shard::new(per_shard_capacity)));
        }
        Self {
            shards,
            tick: AtomicU64::new(0),
        }
    }

    /// Compute which shard a key belongs to.
    fn shard_index(&self, key: &K) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        // CERTAIN: SHARD_COUNT is a power of two, so `& (n - 1)` == `% n`.
        (hasher.finish() as usize) & (self.shards.len() - 1)
    }

    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed)
    }

    /// Atomically fetch the value for `key`, inserting the result of `init`
    /// if no value is present.
    ///
    /// This satisfies the **MCP gateway** contract: per-session/per-server
    /// state is created exactly once even under concurrent first-access. The
    /// shard lock is held across the check-and-insert so there is no TOCTOU
    /// window.
    pub fn get_or_insert_with<F>(&self, key: K, init: F) -> Arc<V>
    where
        F: FnOnce() -> V,
    {
        let idx = self.shard_index(&key);
        let tick = self.next_tick();
        let mut shard = self
            .shards
            .get(idx)
            .expect("shard index in range")
            .lock()
            .expect("shard mutex poisoned");

        if let Some(entry) = shard.map.get_mut(&key) {
            // CERTAIN: Refresh recency on access.
            entry.1 = tick;
            return Arc::clone(&entry.0);
        }

        // CERTAIN: `init` runs exactly once, while holding the lock, only when
        // the key is absent — guaranteeing single creation.
        let value = Arc::new(init());
        shard.map.insert(key, (Arc::clone(&value), tick));
        shard.evict_if_needed();
        value
    }

    /// Recall a previously stored value, returning a shared `Arc<V>` if
    /// present. Touches recency (hot-tier semantics).
    pub fn recall(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.shard_index(key);
        let tick = self.next_tick();
        let mut shard = self
            .shards
            .get(idx)
            .expect("shard index in range")
            .lock()
            .expect("shard mutex poisoned");

        shard.map.get_mut(key).map(|entry| {
            entry.1 = tick;
            Arc::clone(&entry.0)
        })
    }

    /// Insert `value` for `key`, returning any previously stored value.
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>> {
        let idx = self.shard_index(&key);
        let tick = self.next_tick();
        let mut shard = self
            .shards
            .get(idx)
            .expect("shard index in range")
            .lock()
            .expect("shard mutex poisoned");

        let previous = shard
            .map
            .insert(key, (Arc::new(value), tick))
            .map(|(arc, _)| arc);
        shard.evict_if_needed();
        previous
    }

    /// Get the shared value for `key`, if present. Identical access semantics
    /// to `recall`; provided for the read-path naming convention.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        self.recall(key)
    }

    /// Total number of entries across all shards.
    ///
    /// CERTAIN: Additive helper used by tests to verify bounded growth; does
    /// not change the locked API contract.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("shard mutex poisoned").map.len())
            .sum()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

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
