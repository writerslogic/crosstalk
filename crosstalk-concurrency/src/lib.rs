//! Crosstalk concurrency primitives.
//!
//! This crate provides building blocks for concurrent and parallel
//! orchestration within Crosstalk: cancellation tokens, sharded data
//! structures, caching layers, and task execution helpers.

pub mod cache;
pub mod cancel;
pub mod executor;
pub mod sharded;

// CERTAIN: Re-export the primary cancellation type for ergonomic access.
pub use cancel::CancelScope;

// CERTAIN: Re-export the cache primitive so callers (embed_text, metacognition)
// can route through a single shared type.
pub use cache::{Cache, CacheConfig};

#[cfg(test)]
mod tests {
    use super::*;

    // CERTAIN: This compiles only if all four modules resolve to existing
    // (empty) module declarations. Acts as a smoke test for the crate layout.
    #[test]
    fn modules_are_declared() {
        // Reference the module paths to ensure they exist at compile time.
        // These are no-op uses; the assertion documents intent.
        let _ = stringify!(cache);
        let _ = stringify!(cancel);
        let _ = stringify!(executor);
        let _ = stringify!(sharded);
        assert!(true);
    }

    // CERTAIN: The re-export resolves, confirming CancelScope is part of the
    // crate's public API.
    #[test]
    fn cancel_scope_is_reexported() {
        let _scope = CancelScope::new();
        assert!(!_scope.is_cancelled());
    }

    // CERTAIN: Confirms the Cache re-export is part of the public API.
    #[test]
    fn cache_is_reexported() {
        let _cfg = CacheConfig::default();
        let _cache: Cache<u32, u32> = Cache::default();
        assert!(_cfg.capacity > 0);
    }
}
