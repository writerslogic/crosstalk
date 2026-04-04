use crosstalk::state::StateManager;
use crosstalk::types::ConversationState;
use tempfile::tempdir;

#[test]
fn test_state_persistence() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");

    let manager = StateManager::new(path).expect("Failed to create StateManager");

    let mut state = ConversationState::new("test-session");
    state.iteration_index = 5;

    manager.checkpoint(&state).expect("Failed to checkpoint");

    let restored = manager
        .restore(5)
        .expect("Failed to restore")
        .expect("State not found");
    assert_eq!(restored.session_id, "test-session");
    assert_eq!(restored.iteration_index, 5);
}

#[test]
fn test_list_checkpoints() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");
    let manager = StateManager::new(path).expect("Failed to create StateManager");

    for i in 1..=3 {
        let mut state = ConversationState::new("test");
        state.iteration_index = i;
        manager.checkpoint(&state).expect("Failed to checkpoint");
    }

    let checkpoints = manager.list_checkpoints();
    assert_eq!(checkpoints.len(), 3);
    assert!(checkpoints.contains(&1));
    assert!(checkpoints.contains(&2));
    assert!(checkpoints.contains(&3));
}
