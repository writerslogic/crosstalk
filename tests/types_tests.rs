use crosstalk::types::compute::{BudgetLedger, BudgetMode};
use crosstalk::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};

#[test]
fn test_conversation_state_initialization() {
    let s = ConversationState::new("test-session");
    assert_eq!(s.session_id, "test-session");
    assert_eq!(s.iteration_index, 0);
    assert!(s.artifacts.is_empty());
}

#[test]
fn test_budget_ledger_mode_transitions() {
    let mut ledger = BudgetLedger {
        session_budget: 10.0,
        spent: 0.0,
        entries: vec![],
    };
    assert_eq!(ledger.mode(), BudgetMode::Normal);

    ledger.spent = 8.5; // 15% left
    assert_eq!(ledger.mode(), BudgetMode::CostReduction);

    ledger.spent = 9.8; // 2% left
    assert_eq!(ledger.mode(), BudgetMode::Emergency);
}

#[test]
fn test_turn_serialization_roundtrip() {
    let turn = Turn {
        index: 42,
        model_id: "gpt-4".to_string(),
        content: "Proposing a change.".to_string(),
        timestamp: 123456789,
        diffs: vec![],
        certainty: Some(0.85),
        outcome: TurnOutcome::Compiled,
        task_category: Some(TaskCategory::CodeGeneration),
        structure: Some(TurnStructure::StepByStep),
        signature: vec![1, 2, 3],
        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
    };

    let serialized = serde_json::to_string(&turn).unwrap();
    let deserialized: Turn = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.index, 42);
    assert_eq!(deserialized.model_id, "gpt-4");
}
