//! A single moka-backed async cache primitive.
//!
//! This addresses H-033 and H-050: both `embed_text()` and metacognition
//! embeddings route through one shared cache instance with `get_or_insert_async`.

use std::future::Future;
use std::hash::Hash;
use std::time::Duration;

use moka::future::Cache as MokaCache;

/// Configuration for constructing a [`Cache`].
// CERTAIN: A small config struct keeps capacity/TTL configurable without
// proliferating constructor overloads.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Maximum number of entries the cache may hold.
    pub capacity: u64,
    /// Optional time-to-live for entries. `None` disables TTL expiry.
    pub ttl: Option<Duration>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        // CERTAIN: Reasonable defaults; callers may override.
        Self {
            capacity: 10_000,
            ttl: Some(Duration::from_secs(3600)),
        }
    }
}

/// An async, concurrent cache with `get_or_insert_async` semantics.
///
/// `K` and `V` are constrained to satisfy moka's `future::Cache` bounds.
// CERTAIN: moka requires Send + Sync + 'static for both key and value, plus
// Hash + Eq for the key and Clone for the value.
#[derive(Clone)]
pub struct Cache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    inner: MokaCache<K, V>,
}

impl<K, V> Cache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Construct a cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        let mut builder = MokaCache::builder().max_capacity(config.capacity);
        if let Some(ttl) = config.ttl {
            builder = builder.time_to_live(ttl);
        }
        Self {
            inner: builder.build(),
        }
    }

    /// Construct a cache with explicit capacity and TTL.
    pub fn with_capacity_and_ttl(capacity: u64, ttl: Option<Duration>) -> Self {
        Self::new(CacheConfig { capacity, ttl })
    }

    /// Get the value for `key`, computing and inserting it via `init` if absent.
    ///
    /// The `init` future is only awaited on a cache miss. moka ensures only
    /// one concurrent initialization per key.
    pub async fn get_or_insert_async<F, Fut>(&self, key: K, init: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = V>,
    {
        self.inner.get_with(key, init()).await
    }

    /// Fetch a value if present without inserting.
    pub async fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key).await
    }

    /// Insert a value directly.
    pub async fn insert(&self, key: K, value: V) {
        self.inner.insert(key, value).await;
    }

    /// Invalidate a single entry.
    pub async fn invalidate(&self, key: &K) {
        self.inner.invalidate(key).await;
    }

    /// Approximate number of entries currently in the cache.
    pub fn entry_count(&self) -> u64 {
        self.inner.entry_count()
    }
}

impl<K, V> Default for Cache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // CERTAIN: Verifies the cache compiles and basic insert/get round-trips.
    #[tokio::test]
    async fn insert_and_get_round_trip() {
        let cache: Cache<String, i32> = Cache::with_capacity_and_ttl(100, None);
        cache.insert("a".to_string(), 1).await;
        assert_eq!(cache.get(&"a".to_string()).await, Some(1));
        assert_eq!(cache.get(&"missing".to_string()).await, None);
    }

    // CERTAIN: On a miss, init runs and the value is cached; a subsequent call
    // for the same key must NOT re-run init.
    #[tokio::test]
    async fn get_or_insert_async_computes_once() {
        let cache: Cache<u32, u32> = Cache::default();
        let calls = Arc::new(AtomicUsize::new(0));

        let c1 = calls.clone();
        let v1 = cache
            .get_or_insert_async(7, || async move {
                c1.fetch_add(1, Ordering::SeqCst);
                42
            })
            .await;
        assert_eq!(v1, 42);

        let c2 = calls.clone();
        let v2 = cache
            .get_or_insert_async(7, || async move {
                c2.fetch_add(1, Ordering::SeqCst);
                99
            })
            .await;
        // Second call returns the cached value, not the new init result.
        assert_eq!(v2, 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    // LIKELY: invalidate removes the entry so init runs again.
    #[tokio::test]
    async fn invalidate_forces_recompute() {
        let cache: Cache<u32, u32> = Cache::default();
        cache.insert(1, 10).await;
        assert_eq!(cache.get(&1).await, Some(10));

        cache.invalidate(&1).await;
        // moka's invalidate is async-applied; run_pending via get path.
        let v = cache.get_or_insert_async(1, || async { 20 }).await;
        assert_eq!(v, 20);
    }

    // CERTAIN: Default config produces a usable cache.
    #[tokio::test]
    async fn default_config_is_usable() {
        let cache: Cache<&'static str, &'static str> = Cache::default();
        let v = cache.get_or_insert_async("k", || async { "v" }).await;
        assert_eq!(v, "v");
    }

    // CERTAIN: TTL-configured construction compiles and works.
    #[tokio::test]
    async fn ttl_config_constructs() {
        let cache: Cache<u8, u8> = Cache::with_capacity_and_ttl(8, Some(Duration::from_secs(60)));
        cache.insert(1, 2).await;
        assert_eq!(cache.get(&1).await, Some(2));
    }
}
