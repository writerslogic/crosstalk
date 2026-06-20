#[cfg(test)]
mod result_propagation_tests {
    //! Additional tests (HIGH certainty) confirming `SessionStore` semantics
    //! that error-propagation refactors must preserve. These exercise the
    //! public API only; they do not depend on `CrossTalkError` directly since
    //! the visible `SessionStore` surface is infallible by design (atomic
    //! insert never returns a discarded `Result`).
    use super::*;
    use std::sync::Arc;

    /// `get_or_insert_with` must return the *already-present* session and must
    /// NOT invoke the init closure when the key already exists. A swallowed
    /// or ignored insert result would manifest as a duplicate instance here.
    #[test]
    fn get_or_insert_with_does_not_reinit_existing() {
        let store = SessionStore::new();
        let first = store.get_or_insert_with("dup".to_string(), || Session::new("dup"));

        let mut init_ran = false;
        let second = store.get_or_insert_with("dup".to_string(), || {
            init_ran = true;
            Session::new("dup-other")
        });

        assert!(
            !init_ran,
            "init closure must not run for an existing key"
        );
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(second.id(), "dup");
        assert_eq!(store.len(), 1);
    }

    /// Round-trip: a value inserted via `get_or_create` must be observable via
    /// `get`, proving the insert result is committed (not discarded).
    #[test]
    fn insert_result_is_committed_and_observable() {
        let store = SessionStore::new();
        assert!(store.get("committed").is_none());

        let created = store.get_or_create("committed".to_string());
        let observed = store
            .get("committed")
            .expect("inserted session must be retrievable");

        assert!(Arc::ptr_eq(&created, &observed));
        assert!(!store.is_empty());
    }

    /// The store transitions from empty to non-empty exactly once per new key,
    /// and `len`/`is_empty` stay consistent — guarding against lost insertions.
    #[test]
    fn len_tracks_committed_inserts_consistently() {
        let store = SessionStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        for i in 0..10 {
            let id = format!("s{i}");
            let _session = store.get_or_create(id);
        }
        assert!(!store.is_empty());
        assert_eq!(store.len(), 10);

        // Re-requesting existing keys must not grow the store.
        for i in 0..10 {
            let id = format!("s{i}");
            let _session = store.get_or_create(id);
        }
        assert_eq!(store.len(), 10);
    }
}