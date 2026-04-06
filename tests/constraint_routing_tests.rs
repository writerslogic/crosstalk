use crosstalk::engines::intelligence::IntelligenceEngine;
use crosstalk::types::conversation::{ConversationState, TaskCategory};
use crosstalk::types::intelligence::{ModelProfile, RunningAverage};
use std::collections::HashMap;

#[test]
fn test_route_task_constrained_respects_budget() {
    let mut engine = IntelligenceEngine::new();

    let mut profile = ModelProfile {
        model_id: "model-a".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: RunningAverage::default(),
    };

    let mut score = RunningAverage {
        mean: 0.8,
        count: 5,
        variance: 0.01,
    };
    score.update(0.82);
    profile
        .task_scores
        .insert(TaskCategory::CodeGeneration, score);

    engine.profiles.insert("model-a".to_string(), profile);

    let available = vec!["model-a".to_string()];
    let budget = 2000;
    let latency_ms = 500;
    let blacklist = vec![];

    let result = engine.route_task_constrained(
        TaskCategory::CodeGeneration,
        &available,
        budget,
        latency_ms,
        &blacklist,
    );

    assert!(result.is_ok(), "Route failed: {:?}", result.err());
    assert_eq!(result.unwrap(), "model-a");
}

#[test]
fn test_route_task_constrained_exceeds_budget() {
    let engine = IntelligenceEngine::new();

    let available = vec!["model-a".to_string()];
    let budget = 1000;
    let latency_ms = 500;
    let blacklist = vec![];

    let result = engine.route_task_constrained(
        TaskCategory::CodeGeneration,
        &available,
        budget,
        latency_ms,
        &blacklist,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("exceeds budget"));
}

#[test]
fn test_route_task_constrained_respects_latency() {
    let mut engine = IntelligenceEngine::new();

    let mut profile = ModelProfile {
        model_id: "slow-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: RunningAverage {
            mean: 600.0,
            count: 10,
            variance: 0.1,
        },
    };

    let mut score = RunningAverage {
        mean: 0.9,
        count: 5,
        variance: 0.01,
    };
    score.update(0.92);
    profile
        .task_scores
        .insert(TaskCategory::Debugging, score);

    engine
        .profiles
        .insert("slow-model".to_string(), profile);

    let available = vec!["slow-model".to_string()];
    let budget = 2000;
    let latency_ms = 500;
    let blacklist = vec![];

    let result = engine.route_task_constrained(
        TaskCategory::Debugging,
        &available,
        budget,
        latency_ms,
        &blacklist,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No models satisfy constraints"));
}

#[test]
fn test_route_task_constrained_respects_blacklist() {
    let mut engine = IntelligenceEngine::new();

    let mut profile = ModelProfile {
        model_id: "blacklisted-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: RunningAverage::default(),
    };

    let mut score = RunningAverage {
        mean: 0.95,
        count: 5,
        variance: 0.01,
    };
    score.update(0.96);
    profile
        .task_scores
        .insert(TaskCategory::Architecture, score);

    engine
        .profiles
        .insert("blacklisted-model".to_string(), profile);

    let available = vec!["blacklisted-model".to_string()];
    let budget = 3000;
    let latency_ms = 500;
    let blacklist = vec!["blacklisted-model".to_string()];

    let result = engine.route_task_constrained(
        TaskCategory::Architecture,
        &available,
        budget,
        latency_ms,
        &blacklist,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No models satisfy constraints"));
}

#[test]
fn test_route_task_constrained_selects_highest_quality() {
    let mut engine = IntelligenceEngine::new();

    let mut fast_profile = ModelProfile {
        model_id: "fast-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: RunningAverage {
            mean: 100.0,
            count: 10,
            variance: 0.05,
        },
    };

    let mut fast_score = RunningAverage {
        mean: 0.7,
        count: 5,
        variance: 0.01,
    };
    fast_score.update(0.72);
    fast_profile
        .task_scores
        .insert(TaskCategory::Testing, fast_score);

    let mut good_profile = ModelProfile {
        model_id: "good-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: RunningAverage {
            mean: 200.0,
            count: 10,
            variance: 0.05,
        },
    };

    let mut good_score = RunningAverage {
        mean: 0.85,
        count: 5,
        variance: 0.01,
    };
    good_score.update(0.87);
    good_profile
        .task_scores
        .insert(TaskCategory::Testing, good_score);

    engine
        .profiles
        .insert("fast-model".to_string(), fast_profile);
    engine.profiles.insert("good-model".to_string(), good_profile);

    let available = vec!["fast-model".to_string(), "good-model".to_string()];
    let budget = 2000;
    let latency_ms = 500;
    let blacklist = vec![];

    let result = engine.route_task_constrained(
        TaskCategory::Testing,
        &available,
        budget,
        latency_ms,
        &blacklist,
    );

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "good-model");
}

#[test]
fn test_estimate_tokens() {
    assert_eq!(
        IntelligenceEngine::estimate_tokens(TaskCategory::CodeGeneration),
        2000
    );
    assert_eq!(
        IntelligenceEngine::estimate_tokens(TaskCategory::Debugging),
        1500
    );
    assert_eq!(
        IntelligenceEngine::estimate_tokens(TaskCategory::Architecture),
        2500
    );
    assert_eq!(
        IntelligenceEngine::estimate_tokens(TaskCategory::Refactoring),
        1800
    );
    assert_eq!(
        IntelligenceEngine::estimate_tokens(TaskCategory::Research),
        2200
    );
    assert_eq!(
        IntelligenceEngine::estimate_tokens(TaskCategory::Testing),
        1500
    );
}

#[test]
fn test_update_profile_with_latency() {
    use crosstalk::engines::intelligence::QualityScorer;
    use crosstalk::types::conversation::{Turn, TurnOutcome};

    let mut engine = IntelligenceEngine::new();

    let turn = Turn {
        index: 1,
        model_id: "model-a".to_string(),
        content: "Test response".to_string(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: None,
        outcome: TurnOutcome::TestsPassed,
        task_category: Some(TaskCategory::CodeGeneration),
        structure: None,
        signature: vec![],

        surprise_signal: None,
    };

    let quality_score = QualityScorer::score(&turn, &ConversationState::new("test"));
    engine.update_profile_with_latency(&turn, quality_score, 250);

    let profile = engine.profiles.get("model-a").expect("profile should exist");
    assert!(profile.latency_ms.mean > 0.0);
    assert_eq!(profile.latency_ms.count, 1);
}
