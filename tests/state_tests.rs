use crosstalk::core::state::StateManager;
use crosstalk::types::conversation::ConversationState;
use tempfile::tempdir;

#[test]
fn test_state_manager_checkpoints() {
    let dir = tempdir().expect("temp dir");
    let manager = StateManager::new(dir.path().to_str().expect("path")).expect("state manager");
    let mut sigma = ConversationState::new("test-session");

    manager.checkpoint(&sigma).expect("first checkpoint");
    sigma.iteration_index = 1;
    manager.checkpoint(&sigma).expect("second checkpoint");

    let restored = manager.restore(0).expect("restore 0").expect("exists");
    assert_eq!(restored.iteration_index, 0);

    let restored1 = manager.restore(1).expect("restore 1").expect("exists");
    assert_eq!(restored1.iteration_index, 1);
}

// ── execute_with_rollback ─────────────────────────────────────────────────────

#[test]
fn rollback_success_commits_new_state() {
    let dir = tempdir().unwrap();
    let mgr = StateManager::new(dir.path().to_str().unwrap()).unwrap();
    let mut sigma = ConversationState::new("rb-ok");

    mgr.execute_with_rollback(&mut sigma, |s| {
        s.iteration_index = 7;
        Ok(())
    })
    .expect("should succeed");

    assert_eq!(sigma.iteration_index, 7);
    let stored = mgr.restore(7).unwrap().expect("committed state must be readable");
    assert_eq!(stored.iteration_index, 7);
}

#[test]
fn rollback_on_failure_restores_original_state() {
    let dir = tempdir().unwrap();
    let mgr = StateManager::new(dir.path().to_str().unwrap()).unwrap();
    let mut sigma = ConversationState::new("rb-fail");
    sigma.iteration_index = 3;

    let result = mgr.execute_with_rollback(&mut sigma, |s| {
        s.iteration_index = 99;
        Err(anyhow::anyhow!("simulated failure"))
    });

    assert!(result.is_err());
    assert_eq!(sigma.iteration_index, 3, "in-memory state must be restored");
    assert!(
        mgr.restore(99).unwrap().is_none(),
        "failed state must not be persisted"
    );
}

#[test]
fn rollback_does_not_leave_rollback_marker_on_success() {
    let dir = tempdir().unwrap();
    let mgr = StateManager::new(dir.path().to_str().unwrap()).unwrap();
    let mut sigma = ConversationState::new("marker-check");

    mgr.execute_with_rollback(&mut sigma, |s| {
        s.iteration_index = 1;
        Ok(())
    })
    .unwrap();

    let checkpoints = mgr.list_checkpoints().unwrap();
    assert!(
        checkpoints.iter().all(|&idx| idx != u32::MAX),
        "no rollback marker key should persist after success"
    );
}

#[test]
fn rollback_does_not_leave_rollback_marker_on_failure() {
    let dir = tempdir().unwrap();
    let mgr = StateManager::new(dir.path().to_str().unwrap()).unwrap();
    let mut sigma = ConversationState::new("marker-fail");

    let _ = mgr.execute_with_rollback(&mut sigma, |_| {
        Err(anyhow::anyhow!("fail"))
    });

    assert_eq!(mgr.list_checkpoints().unwrap().len(), 0);
}
