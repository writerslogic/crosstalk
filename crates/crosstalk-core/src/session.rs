//! Session management for Crosstalk multi-agent orchestration.
//!
//! Sessions are stored in a concurrent map and accessed via an *atomic*
//! `entry().or_insert_with()` API to avoid the TOCTOU (time-of-check to
//! time-of-use) race that the previous check-then-act pattern exhibited.

use std::sync::Arc;

use dashmap::DashMap;

/// Unique identifier for a session.
pub type SessionId = String;

/// A single orchestration session.
#[derive(Debug)]
pub struct Session {
    id: SessionId,
    /// Number of agents that have joined this session.
    agent_count: std::sync::atomic::AtomicUsize,
}

impl Session {
    /// Create a new, empty session with the given id.
    pub fn new(id: impl Into<SessionId>) -> Self {
        Self {
            id: id.into(),
            agent_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The session identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Record that an agent has joined; returns the new count.
    pub fn join(&self) -> usize {
        self.agent_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1
    }

    /// Current number of agents in the session.
    pub fn agent_count(&self) -> usize {
        self.agent_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Concurrent store of sessions keyed by [`SessionId`].
#[derive(Debug, Default)]
pub struct SessionStore {
    sessions: DashMap<SessionId, Arc<Session>>,
}

impl SessionStore {
    /// Create an empty session store.
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// Atomically fetch an existing session or insert a freshly created one.
    ///
    /// This uses DashMap's `entry().or_insert_with()` API so that the
    /// check-and-insert is performed under a single shard lock, eliminating
    /// the TOCTOU race present in a `get`-then-`insert` sequence.
    pub fn get_or_insert_with<F>(&self, id: SessionId, init: F) -> Arc<Session>
    where
        F: FnOnce() -> Session,
    {
        self.sessions
            .entry(id)
            .or_insert_with(|| Arc::new(init()))
            .value()
            .clone()
    }

    /// Convenience helper: get an existing session or create a default one
    /// with the supplied id, atomically.
    pub fn get_or_create(&self, id: SessionId) -> Arc<Session> {
        let id_for_init = id.clone();
        self.get_or_insert_with(id, move || Session::new(id_for_init))
    }

    /// Look up a session by id without inserting.
    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions.get(id).map(|r| r.value().clone())
    }

    /// Number of sessions currently stored.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the store contains no sessions.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    // --- Basic behaviour (HIGH certainty) ---

    #[test]
    fn new_store_is_empty() {
        let store = SessionStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn get_or_create_inserts_once() {
        let store = SessionStore::new();
        let s1 = store.get_or_create("alpha".to_string());
        let s2 = store.get_or_create("alpha".to_string());
        assert_eq!(s1.id(), "alpha");
        // Same underlying Arc should be returned for the same key.
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn distinct_keys_create_distinct_sessions() {
        let store = SessionStore::new();
        let a = store.get_or_create("a".to_string());
        let b = store.get_or_create("b".to_string());
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn get_returns_none_for_missing() {
        let store = SessionStore::new();
        assert!(store.get("nope").is_none());
    }

    #[test]
    fn get_returns_existing() {
        let store = SessionStore::new();
        let created = store.get_or_create("k".to_string());
        let fetched = store.get("k").expect("should exist");
        assert!(Arc::ptr_eq(&created, &fetched));
    }

    #[test]
    fn session_join_increments_count() {
        let s = Session::new("x");
        assert_eq!(s.agent_count(), 0);
        assert_eq!(s.join(), 1);
        assert_eq!(s.join(), 2);
        assert_eq!(s.agent_count(), 2);
    }

    // --- T-004 concurrency harness: TOCTOU race detection (HIGH certainty
    //     on intent; verifies atomic insert returns a single shared instance) ---

    /// Many threads racing to create the *same* session id must all observe
    /// the exact same `Arc<Session>` instance. If the implementation used a
    /// check-then-act pattern, some threads would receive distinct instances
    /// (lost-update / duplicate-insert) — this asserts the atomic guarantee.
    #[test]
    fn concurrent_get_or_create_yields_single_instance() {
        let store = Arc::new(SessionStore::new());
        let threads = 64;
        let mut handles = Vec::with_capacity(threads);

        for _ in 0..threads {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || -> usize {
                let s = store.get_or_create("shared".to_string());
                // Return the pointer address to compare canonical identity.
                Arc::as_ptr(&s) as usize
            }));
        }

        let mut ptrs = Vec::with_capacity(threads);
        for h in handles {
            ptrs.push(h.join().expect("thread panicked"));
        }

        // Exactly one canonical session instance must exist.
        let first = ptrs[0];
        assert!(
            ptrs.iter().all(|&p| p == first),
            "TOCTOU race: multiple distinct session instances created concurrently"
        );
        assert_eq!(store.len(), 1, "exactly one session must be stored");
    }

    /// The init closure must run exactly once across a concurrent race for a
    /// single key — proving `or_insert_with` is atomic, not check-then-act.
    #[test]
    fn init_closure_runs_exactly_once_under_contention() {
        let store = Arc::new(SessionStore::new());
        let init_calls = Arc::new(AtomicUsize::new(0));
        let threads = 128;
        let mut handles = Vec::with_capacity(threads);

        for _ in 0..threads {
            let store = Arc::clone(&store);
            let init_calls = Arc::clone(&init_calls);
            handles.push(thread::spawn(move || {
                store.get_or_insert_with("once".to_string(), || {
                    init_calls.fetch_add(1, Ordering::SeqCst);
                    Session::new("once")
                });
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(
            init_calls.load(Ordering::SeqCst),
            1,
            "atomic entry API must invoke init exactly once per key"
        );
        assert_eq!(store.len(), 1);
    }

    /// Concurrent creation across *many distinct* keys must not lose any
    /// insertions and must not corrupt the map under contention.
    #[test]
    fn concurrent_distinct_keys_all_present() {
        let store = Arc::new(SessionStore::new());
        let threads = 32;
        let per_thread = 50usize;
        let mut handles = Vec::with_capacity(threads);

        for t in 0..threads {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let id = format!("{t}-{i}");
                    let s = store.get_or_create(id.clone());
                    assert_eq!(s.id(), id);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(store.len(), threads * per_thread);
    }

    /// Agents joining the same shared session concurrently must produce a
    /// consistent total count — exercising shared-instance correctness that
    /// the atomic insert guarantees.
    #[test]
    fn concurrent_joins_to_shared_session_are_consistent() {
        let store = Arc::new(SessionStore::new());
        let threads = 50;
        let joins_each = 20usize;
        let mut handles = Vec::with_capacity(threads);

        for _ in 0..threads {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let s = store.get_or_create("group".to_string());
                for _ in 0..joins_each {
                    s.join();
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        let s = store.get("group").expect("session must exist");
        assert_eq!(
            s.agent_count(),
            threads * joins_each,
            "join count lost under contention — shared instance not atomic"
        );
        assert_eq!(store.len(), 1);
    }
}