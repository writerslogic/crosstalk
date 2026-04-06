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
