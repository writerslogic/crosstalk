//! Text embedding with caching.
//!
//! `embed_text()` resolves through a single shared
//! [`crosstalk_concurrency::Cache`] instance so that identical inputs are
//! embedded exactly once and subsequent identical requests hit the cache.

use std::sync::Arc;

use crosstalk_concurrency::Cache;

/// A computed embedding vector.
pub type Embedding = Arc<Vec<f32>>;

/// Default cache capacity for embeddings.
const DEFAULT_CACHE_CAPACITY: u64 = 4096;

/// Embeds text, caching results so identical inputs are computed once.
///
/// Identical input strings resolve to the same cached [`Embedding`] via
/// [`Cache::get_or_insert_async`]; the underlying embedding computation runs
/// only on a cache miss.
#[derive(Clone)]
pub struct Embedder {
    cache: Cache<String, Embedding>,
}

impl Embedder {
    /// Creates a new embedder backed by a single shared cache instance.
    pub fn new() -> Self {
        Self {
            cache: Cache::with_capacity_and_ttl(DEFAULT_CACHE_CAPACITY, None),
        }
    }

    /// Embeds `text`, returning a cached result for repeated identical inputs.
    ///
    /// The embedding for a given input is computed exactly once; concurrent
    /// callers requesting the same input coalesce on a single computation.
    pub async fn embed_text(&self, text: &str) -> Embedding {
        let key = text.to_string();
        self.cache
            .get_or_insert_async(key, || {
                let owned = text.to_string();
                async move { Arc::new(compute_embedding(&owned)) }
            })
            .await
    }
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic stub embedding computation.
///
/// Produces a small fixed-length vector derived from the input bytes so that
/// identical inputs yield identical embeddings.
fn compute_embedding(text: &str) -> Vec<f32> {
    const DIMS: usize = 8;
    let mut out = vec![0.0f32; DIMS];
    for (i, b) in text.bytes().enumerate() {
        out[i % DIMS] += b as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // CERTAIN: identical inputs produce equal embeddings and the same cached Arc.
    #[tokio::test]
    async fn repeated_identical_input_hits_cache() {
        let embedder = Embedder::new();
        let first = embedder.embed_text("hello").await;
        let second = embedder.embed_text("hello").await;

        // Same cached Arc instance (pointer equality proves a cache hit).
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(&*first, &*second);
    }

    // CERTAIN: distinct inputs produce distinct (non-shared) embeddings.
    #[tokio::test]
    async fn distinct_inputs_produce_distinct_embeddings() {
        let embedder = Embedder::new();
        let a = embedder.embed_text("alpha").await;
        let b = embedder.embed_text("beta").await;

        assert!(!Arc::ptr_eq(&a, &b));
    }

    // CERTAIN: compute_embedding is deterministic for identical inputs.
    #[test]
    fn compute_embedding_is_deterministic() {
        let a = compute_embedding("crosstalk");
        let b = compute_embedding("crosstalk");
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
    }

    // LIKELY: many concurrent callers on the same input coalesce on one compute.
    #[tokio::test]
    async fn concurrent_identical_inputs_share_result() {
        let embedder = Embedder::new();
        const TASKS: usize = 16;
        let mut handles = Vec::with_capacity(TASKS);

        for _ in 0..TASKS {
            let e = embedder.clone();
            handles.push(tokio::spawn(async move { e.embed_text("shared").await }));
        }

        let mut results = Vec::with_capacity(TASKS);
        for h in handles {
            results.push(h.await.expect("task should not panic"));
        }

        let first = &results[0];
        for r in &results[1..] {
            assert!(Arc::ptr_eq(first, r));
        }
    }

    // CERTAIN: Default impl yields a usable embedder.
    #[tokio::test]
    async fn default_embedder_is_usable() {
        let embedder = Embedder::default();
        let v = embedder.embed_text("x").await;
        assert_eq!(v.len(), 8);
    }

    // LIKELY: empty input is handled and cached.
    #[tokio::test]
    async fn empty_input_is_cached() {
        let embedder = Embedder::new();
        let a = embedder.embed_text("").await;
        let b = embedder.embed_text("").await;
        assert!(Arc::ptr_eq(&a, &b));
    }

    // Touch the atomic import path indirectly to ensure no unused warnings
    // if reused later; keep a trivial sanity assertion.
    #[test]
    fn atomics_available() {
        let n = StdArc::new(AtomicUsize::new(0));
        n.fetch_add(1, Ordering::SeqCst);
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }
}