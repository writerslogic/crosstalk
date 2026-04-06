use crosstalk::engines::intelligence::IntelligenceEngine;
use crosstalk::engines::security::{SecretScanner, ShellSanity, TurnSigner};
use crosstalk::engines::validation::AstValidator;
use crosstalk::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use crosstalk::types::intelligence::ModelProfile;
use std::collections::HashMap;

#[test]
fn test_generate_skeleton() {
    let code = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
        struct Point { x: i32, y: i32 }
        impl Point {
            fn new() -> Self { Point { x: 0, y: 0 } }
        }
    "#;
    let skeleton = AstValidator::generate_skeleton(code, "rust");
    assert!(skeleton.contains("pub fn add(a: i32, b: i32) -> i32 { ... }"));
    assert!(skeleton.contains("struct Point { x: i32, y: i32 }"));
    assert!(skeleton.contains("impl Point {"));
    assert!(skeleton.contains("fn new() -> Self { ... }"));
    assert!(!skeleton.contains("a + b"));
}

#[test]
fn test_secret_scanner() {
    let content = "My key is AKIA1234567890ABCDEF";
    assert_eq!(SecretScanner::scan(content).len(), 1);
}

#[test]
fn test_shell_sanity() {
    assert!(ShellSanity::is_dangerous("rm -rf /"));
    assert!(!ShellSanity::is_dangerous("cargo test"));
}

#[test]
fn test_turn_signer() {
    let signer = TurnSigner::new();
    let data = b"turn data";
    let sig = signer.sign(data);
    assert!(signer.verify(data, &sig));
}

#[test]
fn test_detect_regression() {
    let mut engine = IntelligenceEngine::new();

    let baseline_score = 0.8;
    let mut profile = ModelProfile {
        model_id: "test-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: Default::default(),
    };

    let mut baseline_avg = crosstalk::types::intelligence::RunningAverage::default();
    for _ in 0..10 {
        baseline_avg.update(baseline_score);
    }
    profile.task_scores.insert(TaskCategory::CodeGeneration, baseline_avg);
    engine.profiles.insert("test-model".to_string(), profile);

    let mut recent_turns = Vec::new();
    let _low_score = baseline_score * 0.8;
    for i in 0..5 {
        let turn = Turn {
            index: i,
            model_id: "test-model".to_string(),
            content: "Low quality output".to_string(),
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: None,
            outcome: TurnOutcome::Unknown,
            task_category: Some(TaskCategory::CodeGeneration),
            structure: None,
            signature: vec![],

            surprise_signal: None,
        };
        recent_turns.push(turn);
    }

    let alert = engine.detect_regression("test-model", &recent_turns);
    assert!(alert.is_some(), "Expected regression to be detected");

    let alert = alert.unwrap();
    assert_eq!(alert.agent_id, "test-model");
    assert!(alert.recent_mean < alert.baseline_mean * 0.9);
}

#[test]
fn test_no_regression_when_above_threshold() {
    let mut engine = IntelligenceEngine::new();

    let baseline_score = 0.8;
    let mut profile = ModelProfile {
        model_id: "test-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: Default::default(),
    };

    let mut baseline_avg = crosstalk::types::intelligence::RunningAverage::default();
    for _ in 0..10 {
        baseline_avg.update(baseline_score);
    }
    profile.task_scores.insert(TaskCategory::CodeGeneration, baseline_avg);
    engine.profiles.insert("test-model".to_string(), profile);

    let mut recent_turns = Vec::new();
    for i in 0..5 {
        let turn = Turn {
            index: i,
            model_id: "test-model".to_string(),
            content: "Good quality output with evidence and code".to_string(),
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: None,
            outcome: TurnOutcome::TestsPassed,
            task_category: Some(TaskCategory::CodeGeneration),
            structure: None,
            signature: vec![],

            surprise_signal: None,
        };
        recent_turns.push(turn);
    }

    let alert = engine.detect_regression("test-model", &recent_turns);
    assert!(alert.is_none(), "Expected no regression when above threshold");
}
