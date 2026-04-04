use crosstalk::types::{Artifact, ArtifactDiff, ConversationState, Turn};
use std::collections::HashMap;

#[test]
fn test_turn_creation() {
    let turn = Turn {
        index: 1,
        model_id: "test-model".to_string(),
        content: "Hello world".to_string(),
        timestamp: 123456789,
        diffs: vec![],
        certainty: Some(1.0),
        outcome: crosstalk::types::TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    };
    assert_eq!(turn.index, 1);
    assert_eq!(turn.model_id, "test-model");
    assert_eq!(turn.content, "Hello world");
    assert_eq!(turn.timestamp, 123456789);
}

#[test]
fn test_conversation_state_serialization() {
    let mut state = ConversationState::new("test-session");
    let turn = Turn {
        index: 0,
        model_id: "user".to_string(),
        content: "init".to_string(),
        timestamp: 1000,
        diffs: vec![],
        certainty: Some(1.0),
        outcome: crosstalk::types::TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    };
    state.turns.push(turn);

    let serialized = serde_json::to_string(&state).expect("Failed to serialize");
    let deserialized: ConversationState =
        serde_json::from_str(&serialized).expect("Failed to deserialize");

    assert_eq!(deserialized.session_id, "test-session");
    assert_eq!(deserialized.turns.len(), 1);
    assert_eq!(deserialized.turns[0].content, "init");
}

#[test]
fn test_artifact_and_diff() {
    let diff = ArtifactDiff {
        original_version: 0,
        new_version: 1,
        diff_text: "some diff".to_string(),
    };

    let artifact = Artifact {
        name: "test.txt".to_string(),
        language: "text".to_string(),
        content: "original".to_string(),
        version: 1,
        history: vec![diff],
        ast_versions: HashMap::new(),
        proof_attachments: vec![],
        metrics: crosstalk::quality::ArtifactMetrics::default(),
        skeleton: String::new(),
    };

    let mut artifacts = HashMap::new();
    artifacts.insert("test.txt".to_string(), artifact);

    let state = ConversationState {
        session_id: "session-2".to_string(),
        iteration_index: 1,
        turns: vec![],
        artifacts,
        agent_weights: HashMap::new(),
        completion_probability: 0.0,
        state_hash: [0u8; 32],
        budget: crosstalk::types::BudgetLedger::default(),
        goal_tree: crosstalk::types::GoalTree::default(),
    };

    let serialized = serde_json::to_vec(&state).expect("Failed to serialize");
    let deserialized: ConversationState =
        serde_json::from_slice(&serialized).expect("Failed to deserialize");

    assert_eq!(deserialized.artifacts.get("test.txt").unwrap().version, 1);
    assert_eq!(
        deserialized.artifacts.get("test.txt").unwrap().history[0].diff_text,
        "some diff"
    );
}
