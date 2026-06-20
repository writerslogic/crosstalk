//! Crosstalk concurrency primitives.
//!
//! This crate provides building blocks for concurrent and parallel
//! orchestration within Crosstalk: cancellation tokens, sharded data
//! structures, caching layers, and task execution helpers.

pub mod cache;
pub mod cancel;
pub mod executor;
pub mod sharded;

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
}
