//! Core types and shared infrastructure for Crosstalk.

pub mod consts;
pub mod error;
pub mod orchestrator;

pub use error::{CrossTalkError, Result};

#[cfg(test)]
mod tests {
    // Confirm the consts module is re-exported and reachable from the crate root.
    use crate::consts::DEFAULT_RECALL_LIMIT;

    #[test]
    fn consts_module_is_reachable() {
        assert!(DEFAULT_RECALL_LIMIT > 0);
    }
}