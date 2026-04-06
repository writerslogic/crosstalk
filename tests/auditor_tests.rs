use crosstalk::engines::verification::{ContinuousAuditor, HashChain};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use std::time::Duration;

#[tokio::test]
async fn test_continuous_auditor_spawns_and_receives() {
    let tx = ContinuousAuditor::spawn();

    let mut state1 = ConversationState::new("audit-session");
    state1.iteration_index = 0;
    state1.turns.push(Turn {
        index: 0,
        model_id: "auditor-test-model".to_string(),
        content: "Test turn 1".to_string(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: Some(0.8),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    });
    let prev_hash = [0u8; 32];
    state1.state_hash = HashChain::compute(&state1, &prev_hash).expect("Hash computation failed");

    let send_result = tx.send(state1.clone()).await;
    assert!(send_result.is_ok(), "Should successfully send state to auditor");

    let mut state2 = ConversationState::new("audit-session");
    state2.iteration_index = 1;
    state2.turns.push(state1.turns[0].clone());
    state2.turns.push(Turn {
        index: 1,
        model_id: "auditor-test-model".to_string(),
        content: "Test turn 2".to_string(),
        timestamp: ConversationState::now() + 1,
        diffs: vec![],
        certainty: Some(0.85),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    });
    state2.state_hash = HashChain::compute(&state2, &state1.state_hash).expect("Hash computation failed");

    let send_result = tx.send(state2.clone()).await;
    assert!(send_result.is_ok(), "Should successfully send second state to auditor");

    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(state1.iteration_index, 0);
    assert_eq!(state2.iteration_index, 1);
}
