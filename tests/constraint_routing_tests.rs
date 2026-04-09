use crosstalk::engines::intelligence::{IntelligenceEngine, ParetoOptimizer, ParetoPoint};
use crosstalk::types::conversation::TaskCategory;
use crosstalk::types::intelligence::{ModelProfile, RunningAverage};
use std::collections::BTreeMap;

fn make_profile(model_id: &str, category: TaskCategory, mean: f64) -> ModelProfile {
    let mut task_scores = BTreeMap::new();
    task_scores.insert(category, RunningAverage { mean, count: 1, variance: 0.0 });
    ModelProfile {
        model_id: model_id.to_string(),
        task_scores,
        total_turns: 1,
        last_updated: 0,
        latency_ms: RunningAverage { mean: 100.0, count: 1, variance: 0.0 },
    }
}

#[tokio::test]
async fn test_routing_respects_latency_constraint() {
    let engine = IntelligenceEngine::new();
    let p1 = make_profile("fast", TaskCategory::CodeGeneration, 0.7);
    let mut p2 = make_profile("slow", TaskCategory::CodeGeneration, 0.9);
    p2.latency_ms.mean = 500.0;

    engine.profiles.insert("fast".to_string(), p1);
    engine.profiles.insert("slow".to_string(), p2);

    // Latency limit 200ms -> should pick "fast"
    let selected = engine
        .route_task_constrained(
            TaskCategory::CodeGeneration,
            &["fast".to_string(), "slow".to_string()],
            5000,
            200,
            &[],
        )
        .expect("routing failed");
    assert_eq!(selected, "fast");
}

#[tokio::test]
async fn test_routing_respects_token_budget() {
    let engine = IntelligenceEngine::new();
    let p1 = make_profile("m1", TaskCategory::Architecture, 0.8);
    engine.profiles.insert("m1".to_string(), p1);

    // Architecture estimate is 2500. Budget 1000 -> should fail.
    let res = engine.route_task_constrained(
        TaskCategory::Architecture,
        &["m1".to_string()],
        1000,
        1000,
        &[],
    );
    assert!(res.is_err());
}

#[tokio::test]
async fn test_routing_respects_blacklist() {
    let engine = IntelligenceEngine::new();
    engine.profiles.insert("m1".to_string(), make_profile("m1", TaskCategory::CodeGeneration, 0.8));
    engine.profiles.insert("m2".to_string(), make_profile("m2", TaskCategory::CodeGeneration, 0.9));

    let selected = engine
        .route_task_constrained(
            TaskCategory::CodeGeneration,
            &["m1".to_string(), "m2".to_string()],
            5000,
            1000,
            &["m2".to_string()],
        )
        .expect("routing failed");
    assert_eq!(selected, "m1");
}

#[test]
fn test_pareto_frontier_computation() {
    let points = vec![
        ParetoPoint { model_id: "cheap_bad".to_string(), quality: 0.4, cost_tokens: 500 },
        ParetoPoint { model_id: "expensive_good".to_string(), quality: 0.9, cost_tokens: 2000 },
        ParetoPoint { model_id: "mid".to_string(), quality: 0.7, cost_tokens: 1000 },
        ParetoPoint { model_id: "dominated".to_string(), quality: 0.6, cost_tokens: 1200 },
    ];

    let frontier = ParetoOptimizer::compute_frontier(points);
    assert_eq!(frontier.len(), 3);
    assert!(frontier.iter().any(|p| p.model_id == "mid"));
    assert!(!frontier.iter().any(|p| p.model_id == "dominated"));
    assert_eq!(frontier[0].model_id, "expensive_good"); // Sorted by quality desc
}

#[test]
fn test_pareto_selection_with_constraints() {
    let frontier = vec![
        ParetoPoint { model_id: "p1".to_string(), quality: 0.9, cost_tokens: 2000 },
        ParetoPoint { model_id: "p2".to_string(), quality: 0.7, cost_tokens: 1000 },
        ParetoPoint { model_id: "p3".to_string(), quality: 0.5, cost_tokens: 500 },
    ];

    // min_quality 0.6, max_tokens 1500 -> should pick p2
    let selected = ParetoOptimizer::select(&frontier, 0.6, 1500).expect("none selected");
    assert_eq!(selected.model_id, "p2");

    // max_tokens 800 -> should pick p3
    let selected2 = ParetoOptimizer::select(&frontier, 0.4, 800).expect("none selected");
    assert_eq!(selected2.model_id, "p3");
}
