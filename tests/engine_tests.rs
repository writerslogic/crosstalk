use crosstalk::engines::intelligence::{IntelligenceEngine, QualityScorer};
use crosstalk::engines::reasoning::{
    ExtractedSignals, ReasoningScorer, ReportGenerator, StructureSelector,
};
use crosstalk::types::conversation::{TaskCategory, Turn, TurnOutcome, TurnStructure};
use crosstalk::types::intelligence::{ModelProfile, RunningAverage};
use std::collections::BTreeMap;

fn make_profile(model_id: &str, category: TaskCategory, mean: f64) -> ModelProfile {
    let mut task_scores = BTreeMap::new();
    task_scores.insert(
        category,
        RunningAverage {
            mean,
            count: 1,
            variance: 0.0,
        },
    );
    ModelProfile {
        model_id: model_id.to_string(),
        task_scores,
        total_turns: 1,
        last_updated: 0,
        latency_ms: RunningAverage::default(),
    }
}

#[tokio::test]
async fn test_intelligence_routing_logic() {
    let engine = IntelligenceEngine::new();
    let p1 = make_profile("gpt-4", TaskCategory::CodeGeneration, 0.9);
    let p2 = make_profile("gpt-3.5", TaskCategory::CodeGeneration, 0.4);

    engine.profiles.insert("gpt-4".to_string(), p1);
    engine.profiles.insert("gpt-3.5".to_string(), p2);

    let best = engine.route_task(
        TaskCategory::CodeGeneration,
        &["gpt-4".to_string(), "gpt-3.5".to_string()],
    );
    assert_eq!(best, "gpt-4");
}

#[test]
fn test_quality_scorer_penalizes_failure() {
    let turn_fail = Turn {
        index: 0,
        model_id: "m".to_string(),
        content: "oops".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.1),
        outcome: TurnOutcome::Rejected,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    };
    let turn_pass = Turn {
        index: 1,
        model_id: "m".to_string(),
        content: "success".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.9),
        outcome: TurnOutcome::TestsPassed,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    };

    assert!(QualityScorer::score(&turn_pass) > QualityScorer::score(&turn_fail));
}

#[test]
fn test_reasoning_scorer_detects_evidence() {
    let turn = Turn {
        index: 0,
        model_id: "m".to_string(),
        content: "Based on the evidence in line 42, we should use a Mutex.".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.8),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    };
    let score = ReasoningScorer::score(&turn);
    assert!(score > 0.5);
}

#[test]
fn test_structure_selector_prefers_step_by_step_for_complex_tasks() {
    let selector = StructureSelector::new();
    // Run multiple times; Architecture should predominantly yield StepByStep
    let mut step_count = 0;
    for _ in 0..20 {
        let s = selector.recommend(TaskCategory::Architecture, "m1");
        if s == TurnStructure::StepByStep {
            step_count += 1;
        }
    }
    assert!(step_count >= 10, "StepByStep should dominate for Architecture tasks, got {}/20", step_count);
}

#[test]
fn test_report_generator_contains_summary() {
    let signals = ExtractedSignals {
        decisions: vec!["d1".to_string()],
        problems: vec!["p1".to_string()],
        questions: vec!["q1".to_string()],
        code_blocks: vec!["c1".to_string()],
    };
    let report = ReportGenerator::generate(&signals, &[], &[], 0.8);
    assert!(report.contains("**Overall Score**: 0.80"));
}
