//! Memory store for Crosstalk agents.
//!
//! P3 MEDIUM error-handling fixes: store operations are fallible and return
//! the unified [`CrossTalkError`] via `?` propagation rather than panicking on
//! missing keys, lock poisoning, or capacity violations.

use std::collections::HashMap;

use crosstalk_core::error::{CrossTalkError, Result};

/// A simple in-memory key/value store for agent state.
///
/// All lookups and mutations are fallible and surface domain errors through
/// [`CrossTalkError`] so callers can use `?` instead of `unwrap`/`expect`.
#[derive(Debug, Default)]
pub struct MemoryStore {
    entries: HashMap<String, String>,
    capacity: Option<usize>,
}

impl MemoryStore {
    /// Create an empty, unbounded store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            capacity: None,
        }
    }

    /// Create a store bounded to `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity: Some(capacity),
        }
    }

    /// Insert a key/value pair.
    ///
    /// Returns a [`CrossTalkError::Config`] (rather than panicking) if the
    /// store is at capacity and the key is new.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> Result<()> {
        let key = key.into();
        if let Some(cap) = self.capacity {
            if !self.entries.contains_key(&key) && self.entries.len() >= cap {
                return Err(CrossTalkError::Config(format!(
                    "memory store at capacity ({cap}); cannot insert key {key:?}"
                )));
            }
        }
        self.entries.insert(key, value.into());
        Ok(())
    }

    /// Retrieve a value, propagating an error if the key is absent.
    pub fn get(&self, key: &str) -> Result<&str> {
        self.entries
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| CrossTalkError::Agent(format!("missing key: {key}")))
    }

    /// Remove a value, propagating an error if the key is absent.
    pub fn remove(&mut self, key: &str) -> Result<String> {
        self.entries
            .remove(key)
            .ok_or_else(|| CrossTalkError::Agent(format!("cannot remove missing key: {key}")))
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_round_trip() {
        let mut store = MemoryStore::new();
        store.insert("k", "v").expect("insert should succeed");
        assert_eq!(store.get("k").expect("key present"), "v");
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn get_missing_key_propagates_error_not_panic() {
        let store = MemoryStore::new();
        let err = store.get("absent").expect_err("missing key must yield Err");
        assert!(matches!(err, CrossTalkError::Agent(_)));
        assert!(err.to_string().contains("absent"));
    }

    #[test]
    fn remove_missing_key_propagates_error_not_panic() {
        let mut store = MemoryStore::new();
        let err = store
            .remove("absent")
            .expect_err("removing missing key must yield Err");
        assert!(matches!(err, CrossTalkError::Agent(_)));
    }

    #[test]
    fn remove_existing_key_returns_value() {
        let mut store = MemoryStore::new();
        store.insert("k", "v").unwrap();
        let removed = store.remove("k").expect("present key removable");
        assert_eq!(removed, "v");
        assert!(store.is_empty());
    }

    #[test]
    fn capacity_violation_returns_error_instead_of_panicking() {
        let mut store = MemoryStore::with_capacity(1);
        store.insert("a", "1").expect("first insert fits");
        let err = store
            .insert("b", "2")
            .expect_err("over-capacity insert must yield Err");
        assert!(matches!(err, CrossTalkError::Config(_)));
        // Existing entry preserved; failed insert did not mutate state.
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("a").unwrap(), "1");
    }

    #[test]
    fn overwriting_existing_key_at_capacity_is_allowed() {
        let mut store = MemoryStore::with_capacity(1);
        store.insert("a", "1").unwrap();
        // Updating an existing key must not trip the capacity guard.
        store.insert("a", "2").expect("overwrite within capacity");
        assert_eq!(store.get("a").unwrap(), "2");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn errors_propagate_through_question_mark_chain() {
        fn workflow() -> Result<String> {
            let mut store = MemoryStore::new();
            store.insert("session", "open")?;
            // This `?` surfaces the missing-key error and short-circuits.
            let v = store.get("missing")?;
            Ok(v.to_string())
        }

        let err = workflow().expect_err("chain should short-circuit on missing key");
        assert!(matches!(err, CrossTalkError::Agent(_)));
    }
}