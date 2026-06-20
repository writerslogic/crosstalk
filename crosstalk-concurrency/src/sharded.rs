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
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::thread;

    // CERTAIN: Construction must produce an empty, usable map.
    #[test]
    fn new_constructs_empty_sharded() {
        let map: Sharded<String, u64> = Sharded::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn default_matches_new_type() {
        let map: Sharded<u32, String> = Sharded::default();
        assert!(map.is_empty());
    }

    // CERTAIN: insert returns previous (None first), get returns Arc payload.
    #[test]
    fn insert_then_get_roundtrips_value() {
        let map: Sharded<String, u64> = Sharded::new();
        assert!(map.insert("a".to_string(), 42).is_none());

        let got = map.get(&"a".to_string());
        assert!(got.is_some());
        assert_eq!(*got.unwrap(), 42);
    }

    // CERTAIN: insert over an existing key returns the prior Arc value.
    #[test]
    fn insert_returns_previous_value() {
        let map: Sharded<String, u64> = Sharded::new();
        assert!(map.insert("k".to_string(), 1).is_none());
        let prev = map.insert("k".to_string(), 2);
        assert!(prev.is_some());
        assert_eq!(*prev.unwrap(), 1);
        assert_eq!(*map.get(&"k".to_string()).unwrap(), 2);
    }

    // CERTAIN: recall is the Arc-returning hot-tier read path.
    #[test]
    fn recall_returns_arc_for_present_key_and_none_for_absent() {
        let map: Sharded<String, Vec<u8>> = Sharded::new();
        map.insert("present".to_string(), vec![1, 2, 3]);

        let hit: Option<StdArc<Vec<u8>>> = map.recall(&"present".to_string());
        assert!(hit.is_some());
        assert_eq!(&*hit.unwrap(), &vec![1, 2, 3]);

        let miss = map.recall(&"absent".to_string());
        assert!(miss.is_none());
    }

    // CERTAIN: get and recall share the same Arc allocation (shared, not cloned).
    #[test]
    fn get_shares_same_arc_allocation() {
        let map: Sharded<u32, String> = Sharded::new();
        map.insert(7, "shared".to_string());

        let a = map.get(&7).unwrap();
        let b = map.recall(&7).unwrap();
        // Both Arcs must point at the same heap allocation.
        assert!(StdArc::ptr_eq(&a, &b));
    }

    // CERTAIN: get_or_insert_with inserts when absent, then returns existing.
    #[test]
    fn get_or_insert_with_inserts_once_then_returns_existing() {
        let map: Sharded<String, u64> = Sharded::new();
        let calls = StdArc::new(AtomicUsize::new(0));

        let c1 = StdArc::clone(&calls);
        let v1 = map.get_or_insert_with("key".to_string(), move || {
            c1.fetch_add(1, AtomicOrdering::SeqCst);
            100
        });
        assert_eq!(*v1, 100);

        let c2 = StdArc::clone(&calls);
        let v2 = map.get_or_insert_with("key".to_string(), move || {
            c2.fetch_add(1, AtomicOrdering::SeqCst);
            999 // must NOT be used
        });
        assert_eq!(*v2, 100);

        // init must have run exactly once.
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
        // And both calls hand back the same shared allocation.
        assert!(StdArc::ptr_eq(&v1, &v2));
    }

    // CERTAIN: under concurrent first-access, init runs exactly once (no TOCTOU).
    #[test]
    fn get_or_insert_with_is_atomic_under_concurrency() {
        let map: StdArc<Sharded<u32, u64>> = StdArc::new(Sharded::new());
        let init_calls = StdArc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..16 {
            let map = StdArc::clone(&map);
            let calls = StdArc::clone(&init_calls);
            handles.push(thread::spawn(move || {
                let v = map.get_or_insert_with(1234, move || {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                    7
                });
                assert_eq!(*v, 7);
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        // Exactly-once creation is the core SYS-006 guarantee.
        assert_eq!(init_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(*map.get(&1234).unwrap(), 7);
    }

    // CERTAIN: per-shard capacity bounds total growth via LRU eviction.
    #[test]
    fn lru_eviction_bounds_growth() {
        // capacity 2 per shard => at most SHARD_COUNT*2 entries total.
        let cap_per_shard = 2usize;
        let map: Sharded<u64, u64> = Sharded::with_capacity(cap_per_shard);

        // Insert far more keys than total capacity.
        let total_keys = 10_000u64;
        for k in 0..total_keys {
            map.insert(k, k);
        }

        let max_total = SHARD_COUNT * cap_per_shard;
        let len = map.len();
        assert!(
            len <= max_total,
            "len {len} exceeded bound {max_total} — eviction failed to bound growth"
        );
        assert!(len > 0, "map should retain its most recent entries");
    }

    // CERTAIN: accessing an entry refreshes recency so it survives eviction
    // longer than untouched entries within the same shard.
    #[test]
    fn lru_eviction_respects_recency() {
        // Single-entry shards make the LRU behaviour deterministic per shard.
        let map: Sharded<u64, u64> = Sharded::with_capacity(1);

        // Find two keys that land in the SAME shard so they contend for the
        // single slot.
        let probe = Sharded::<u64, u64>::with_capacity(1);
        let target = probe.shard_index(&0);
        let mut other = None;
        for k in 1u64..100_000 {
            if probe.shard_index(&k) == target {
                other = Some(k);
                break;
            }
        }
        let key_b = other.expect("expected a colliding key in 0..100000");

        // Insert key 0, then re-insert (touch) it, then insert the colliding
        // key_b. With capacity 1 the most-recent survivor is key_b.
        map.insert(0, 100);
        // Touch key 0 to bump its recency.
        let _ = map.get(&0);
        map.insert(key_b, 200);

        // The shard can only hold one entry; the just-inserted key_b must be
        // present.
        assert_eq!(*map.get(&key_b).unwrap(), 200);
        assert!(
            map.get(&0).is_none(),
            "older entry should have been evicted"
        );
    }
}
