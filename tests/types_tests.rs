use crosstalk::engines::quality::ArtifactMetrics;
use crosstalk::types::artifact::{Artifact, ArtifactDiff};
use crosstalk::types::compute::BudgetLedger;
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use crosstalk::types::planning::GoalTree;
use std::collections::HashMap;

#[test]
fn test_turn_serialization() {
    let turn = Turn {
        index: 1,
        model_id: "test-model".to_string(),
        content: "test content".to_string(),
        timestamp: 123456789,
        diffs: vec![],
        certainty: Some(0.8),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    };

    let serialized = serde_json::to_string(&turn).unwrap();
    let deserialized: Turn = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.index, turn.index);
    assert_eq!(deserialized.model_id, turn.model_id);
}

#[test]
fn test_conversation_state_serialization() {
    let mut state = ConversationState::new("test-session");
    state.turns.push(Turn {
        index: 0,
        model_id: "test-model".to_string(),
        content: "initial".to_string(),
        timestamp: 123456789,
        diffs: vec![],
        certainty: Some(1.0),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    });

    let serialized = serde_json::to_string(&state).unwrap();
    let deserialized: ConversationState = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.session_id, state.session_id);
    assert_eq!(deserialized.turns.len(), 1);
}

#[test]
fn test_artifact_serialization() {
    let mut artifact = Artifact {
        name: "test.rs".to_string(),
        language: "rust".to_string(),
        content: "fn main() {}".to_string(),
        version: 1,
        history: vec![ArtifactDiff {
            original_version: 0,
            new_version: 1,
            diff_text: "initial".to_string(),
        }],
        ast_versions: HashMap::new(),
        proof_attachments: vec![],
        metrics: ArtifactMetrics::default(),
        skeleton: "fn main() { ... }".to_string(),
    };

    artifact
        .ast_versions
        .insert("fn:main".to_string(), vec![(0, "fn main() {}".to_string())]);

    let serialized = serde_json::to_string(&artifact).unwrap();
    let deserialized: Artifact = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.name, artifact.name);
    assert_eq!(deserialized.version, artifact.version);
    assert_eq!(deserialized.history.len(), 1);
}

#[test]
fn test_conversation_state_new() {
    let session_id = "new-session";
    let state = ConversationState {
        session_id: session_id.to_string(),
        iteration_index: 0,
        turns: vec![],
        artifacts: HashMap::new(),
        completion_probability: 0.0,
        budget: BudgetLedger::default(),
        goal_tree: GoalTree::default(),
        agent_weights: HashMap::new(),
        node_consensus: HashMap::new(),
        state_hash: [0u8; 32],
    };

    assert_eq!(state.session_id, session_id);
    assert_eq!(state.iteration_index, 0);
    assert!(state.turns.is_empty());
}
