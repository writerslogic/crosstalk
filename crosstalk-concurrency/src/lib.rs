//! Crosstalk concurrency primitives.
//!
//! This crate provides the cancellation primitives used for cooperative
//! shutdown of concurrent orchestration work within Crosstalk.

pub mod cancel;

// Re-export the primary cancellation type for ergonomic access.
pub use cancel::CancelScope;

#[cfg(test)]
mod tests {
    use super::*;

    // The re-export resolves, confirming CancelScope is part of the
    // crate's public API.
    #[test]
    fn cancel_scope_is_reexported() {
        let _scope = CancelScope::new();
        assert!(!_scope.is_cancelled());
    }
}
