#[cfg(test)]
mod security_validation_tests {
    //! P3 MEDIUM security coverage (HIGH certainty): these tests exercise the
    //! public `SessionStore`/`Session` surface to guard against session-id
    //! confusion, accidental cross-session sharing, and silent insert loss.
    //!
    //! All assertions use only documented public API signatures, so they
    //! remain valid regardless of internal storage refactors.
    use super::*;
    use std::sync::Arc;

    /// Distinct session ids must never alias to the same `Session` instance.
    /// A regression here would let one tenant's traffic land in another's
    /// session (a confused-deputy / isolation bug).
    #[test]
    fn distinct_ids_yield_isolated_sessions() {
        let store = SessionStore::new();
        let a = store.get_or_create("tenant-a".to_string());
        let b = store.get_or_create("tenant-b".to_string());

        assert!(
            !Arc::ptr_eq(&a, &b),
            "different session ids must not share a Session instance"
        );
        assert_eq!(a.id(), "tenant-a");
        assert_eq!(b.id(), "tenant-b");
        assert_eq!(store.len(), 2);
    }

    /// Session ids are exact-match keyed: lookups must be case- and
    /// whitespace-sensitive so that a forged near-miss id cannot resolve to a
    /// legitimate session.
    #[test]
    fn lookup_is_exact_match_only() {
        let store = SessionStore::new();
        let _real = store.get_or_create("Auth-Session".to_string());

        assert!(
            store.get("auth-session").is_none(),
            "case-mismatched id must not resolve to an existing session"
        );
        assert!(
            store.get(" Auth-Session").is_none(),
            "leading whitespace must not resolve to an existing session"
        );
        assert!(
            store.get("Auth-Session ").is_none(),
            "trailing whitespace must not resolve to an existing session"
        );
        assert!(
            store.get("Auth-Session").is_some(),
            "the exact id must resolve"
        );
    }

    /// An empty session id is still a valid, *distinct* key and must not be
    /// confused with a missing/unset session. This prevents an unauthenticated
    /// blank id from colliding with any populated session.
    #[test]
    fn empty_id_is_distinct_and_isolated() {
        let store = SessionStore::new();
        let blank = store.get_or_create(String::new());
        let named = store.get_or_create("named".to_string());

        assert!(!Arc::ptr_eq(&blank, &named));
        assert_eq!(blank.id(), "");
        assert!(store.get("").is_some());
        assert_eq!(store.len(), 2);
    }

    /// Joining a session must monotonically increase its agent count and the
    /// count must be observable through the shared `Arc` (no lost updates that
    /// could let an unaccounted agent slip into a session).
    #[test]
    fn join_is_accounted_through_shared_handle() {
        let store = SessionStore::new();
        let session = store.get_or_create("counted".to_string());

        let before = session.agent_count();
        let after_join = session.join();
        assert!(
            after_join > before,
            "join must increase the agent count"
        );

        let same = store
            .get("counted")
            .expect("session must still be retrievable");
        assert!(Arc::ptr_eq(&session, &same));
        assert_eq!(
            same.agent_count(),
            after_join,
            "agent count must be consistent across shared handles"
        );
    }

    /// Demonstrates surfacing a session-validation failure via `CrossTalkError`
    /// without changing the infallible public `SessionStore` API. A gateway/
    /// auth layer can wrap lookups in this pattern to reject unknown ids.
    #[test]
    fn missing_session_can_be_surfaced_as_crosstalk_error() {
        use crate::error::{CrossTalkError, Result};

        fn require_session(store: &SessionStore, id: &str) -> Result<Arc<Session>> {
            store
                .get(id)
                .ok_or_else(|| CrossTalkError::Agent(format!("unknown session id: {id}")))
        }

        let store = SessionStore::new();
        let _ = store.get_or_create("present".to_string());

        assert!(require_session(&store, "present").is_ok());

        let err = require_session(&store, "absent").unwrap_err();
        assert!(matches!(err, CrossTalkError::Agent(_)));
        assert!(
            err.to_string().contains("absent"),
            "error must identify the rejected id"
        );
    }
}