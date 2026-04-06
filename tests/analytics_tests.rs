use crosstalk::engines::analytics::{
    AnalyticsEngine, DecisionReplay, FailureTaxonomy, MetaLearningEngine, QualityTrendDetector,
    StrategyRecommender,
};
use crosstalk::engines::collective_intelligence::{
    DynamicTeamComposer, EnsembleEngine, MetaStrategyOptimizer, PeerReview,
};
use crosstalk::engines::release::{ConvergenceReport, CpopVerifier, ReleaseManager};
use crosstalk::ui::visualization::{
    ForceDirectedGraph, HeatmapGenerator, LatentMapper, Node, ReplayEngine, SvgExporter,
    TimelineManager,
};
use crosstalk::types::analytics::QualityTrend;
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use crosstalk::types::intelligence::AgentProfile;
use std::collections::HashMap;

fn make_state(session_id: &str, turns: u32, prob: f64) -> ConversationState {
    let mut s = ConversationState::new(session_id);
    s.completion_probability = prob;
    for i in 0..turns {
        s.turns.push(Turn {
            index: i,
            model_id: format!("agent-{}", i % 2),
            content: "test".to_string(),
            timestamp: i as u64,
            diffs: vec![],
            certainty: Some(0.8),
            outcome: if i % 3 == 0 { TurnOutcome::TestsPassed } else { TurnOutcome::Compiled },
            task_category: None,
            structure: None,
            signature: vec![],
            surprise_signal: None,
        });
    }
    s
}

// ── Track 18: Analytics ───────────────────────────────────────────────────────

#[test]
fn analytics_report_has_correct_session_id() {
    let sigma = make_state("session-abc", 5, 0.6);
    let report = AnalyticsEngine::generate_report(&sigma);
    assert_eq!(report.session_id, "session-abc");
    assert_eq!(report.agent_performances.len(), 2);
}

#[test]
fn quality_trend_detector_improving() {
    let scores = vec![0.4, 0.5, 0.6, 0.7, 0.8];
    assert_eq!(QualityTrendDetector::detect(&scores), QualityTrend::Improving);
}

#[test]
fn quality_trend_detector_regressing() {
    let scores = vec![0.8, 0.7, 0.6, 0.5, 0.4];
    assert_eq!(QualityTrendDetector::detect(&scores), QualityTrend::Regressing);
}

#[test]
fn quality_trend_detector_plateau() {
    let scores = vec![0.6, 0.61, 0.59, 0.60, 0.605];
    assert_eq!(QualityTrendDetector::detect(&scores), QualityTrend::Plateau);
}

#[test]
fn quality_trend_detector_too_few_samples_is_plateau() {
    let scores = vec![0.9, 0.1];
    assert_eq!(QualityTrendDetector::detect(&scores), QualityTrend::Plateau);
}

#[test]
fn decision_replay_reconstructs_turn() {
    let sigma = make_state("replay-test", 5, 0.7);
    let replay = DecisionReplay::reconstruct(&sigma, 2);
    assert!(replay.is_some());
    let s = replay.unwrap();
    assert!(s.contains("Turn 2"));
    assert!(s.contains("agent=agent-0"));
}

#[test]
fn decision_replay_missing_turn_returns_none() {
    let sigma = make_state("test", 3, 0.5);
    assert!(DecisionReplay::reconstruct(&sigma, 99).is_none());
}

#[test]
fn strategy_recommender_flags_low_success_rate() {
    let mut sigma = make_state("strat", 12, 0.3);
    for t in &mut sigma.turns {
        t.outcome = TurnOutcome::Rejected;
    }
    let recs = StrategyRecommender::recommend(&sigma);
    assert!(recs.iter().any(|r| r.action == "switch_to_critique_protocol"));
}

#[test]
fn meta_learning_engine_empty_sessions() {
    let insight = MetaLearningEngine::compute_insight(&[]);
    assert_eq!(insight.session_count, 0);
    assert_eq!(insight.quality_growth_rate, 0.0);
}

#[test]
fn meta_learning_engine_identifies_best_model() {
    let mut s1 = make_state("s1", 6, 0.7);
    for t in &mut s1.turns {
        if t.index % 2 == 0 {
            t.model_id = "specialist".to_string();
            t.outcome = TurnOutcome::TestsPassed;
        }
    }
    let insight = MetaLearningEngine::compute_insight(&[&s1]);
    assert_eq!(insight.best_model, Some("specialist".to_string()));
}

#[test]
fn failure_taxonomy_categorizes_five_types() {
    assert_eq!(FailureTaxonomy::categorize("mismatched types"), "TypeError");
    assert_eq!(FailureTaxonomy::categorize("timeout waiting for response"), "Timeout");
    assert_eq!(FailureTaxonomy::categorize("thread panicked"), "Panic");
    assert_eq!(FailureTaxonomy::categorize("circular dependency"), "CircularReasoning");
    assert_eq!(FailureTaxonomy::categorize("quality regression detected"), "QualityRegression");
}

// ── Track 19: Release ─────────────────────────────────────────────────────────

#[test]
fn cpop_verifier_empty_history_passes() {
    assert!(CpopVerifier::verify_history(&[]));
}

#[test]
fn convergence_report_contains_session_id() {
    let sigma = make_state("report-test", 3, 0.85);
    let report = ConvergenceReport::generate(&sigma);
    assert!(report.contains("report-test"));
    assert!(report.contains("0.85"));
}

#[test]
fn stability_audit_detects_hash_mismatch() {
    let sigma = ConversationState::new("empty");
    // Default state_hash = [0u8;32] never matches a real computed hash
    let err = ReleaseManager::run_stability_audit(&sigma);
    assert!(err.is_err(), "mismatched hash should fail the audit");
    assert!(err.unwrap_err().to_string().contains("Hash chain integrity"));
}

#[test]
fn stability_audit_monotonic_turns_pass() {
    let sigma = make_state("monotonic", 4, 0.7);
    // Monotonic turns pass; only the hash check may fail
    let result = ReleaseManager::run_stability_audit(&sigma);
    if let Err(e) = &result {
        assert!(
            e.to_string().contains("Hash chain"),
            "only acceptable failure is hash: {}", e
        );
    }
}

#[test]
fn homebrew_formula_contains_version() {
    let formula = ReleaseManager::generate_homebrew_formula("1.2.3", "abcdef1234");
    assert!(formula.contains("1.2.3"));
    assert!(formula.contains("abcdef1234"));
    assert!(formula.contains("class Crosstalk < Formula"));
}

// ── Track 20: Collective Intelligence ────────────────────────────────────────

#[test]
fn peer_review_flags_todo() {
    let report = PeerReview::review("critic", "fn foo() { /* TODO: implement */ }");
    assert!(!report.comments.is_empty());
    assert!(report.correctness < 0.8);
}

#[test]
fn ensemble_merges_to_highest_quality() {
    let proposals = vec![
        ("a".to_string(), "proposal A content".to_string(), 0.4),
        ("b".to_string(), "proposal B content".to_string(), 0.9),
        ("c".to_string(), "proposal C content".to_string(), 0.6),
    ];
    let merged = EnsembleEngine::merge_proposals(proposals);
    assert!(merged.contains("B"), "highest-quality proposal should dominate merge");
}

#[test]
fn dynamic_team_composer_assigns_roles() {
    let mut profiles = HashMap::new();
    for id in ["alpha", "beta", "gamma"] {
        profiles.insert(
            id.to_string(),
            AgentProfile {
                model_id: id.to_string(),
                capabilities: HashMap::new(),
                total_turns: 5,
                compilation_success_rate: 0.8,
            },
        );
    }
    let team = DynamicTeamComposer::compose(&profiles, "code");
    assert!(team.architect.is_some() || team.coder.is_some(), "should assign at least one role");
}

#[test]
fn meta_strategy_optimizer_best_after_three_trials() {
    let mut opt = MetaStrategyOptimizer::new();
    for _ in 0..3 { opt.record("adversarial", 0.9); }
    for _ in 0..3 { opt.record("collaborative", 0.6); }
    assert_eq!(opt.best_strategy(), Some("adversarial"));
}

#[test]
fn meta_strategy_optimizer_no_best_before_three_trials() {
    let mut opt = MetaStrategyOptimizer::new();
    opt.record("adversarial", 0.9);
    opt.record("adversarial", 0.9);
    assert!(opt.best_strategy().is_none(), "needs 3+ trials to commit");
}

// ── Track 21: Visualization ───────────────────────────────────────────────────

#[test]
fn timeline_manager_seek_finds_correct_state() {
    let mut tm = TimelineManager::new();
    let mut s1 = ConversationState::new("a");
    s1.iteration_index = 1;
    let mut s2 = ConversationState::new("b");
    s2.iteration_index = 2;
    tm.push(s1);
    tm.push(s2);
    assert_eq!(tm.seek(2).map(|s| s.session_id.as_str()), Some("b"));
    assert!(tm.seek(99).is_none());
}

#[test]
fn timeline_manager_step_navigation() {
    let mut tm = TimelineManager::new();
    for i in 0..3u32 {
        let mut s = ConversationState::new("x");
        s.iteration_index = i;
        tm.push(s);
    }
    assert_eq!(tm.current().map(|s| s.iteration_index), Some(0));
    tm.step_forward();
    assert_eq!(tm.current().map(|s| s.iteration_index), Some(1));
    tm.step_back();
    assert_eq!(tm.current().map(|s| s.iteration_index), Some(0));
}

#[test]
fn replay_engine_records_and_advances() {
    let mut engine = ReplayEngine::new(1.0);
    let s1 = make_state("replay", 3, 0.5);
    let s2 = make_state("replay", 6, 0.8);
    engine.record_frame(&s1);
    engine.record_frame(&s2);
    assert_eq!(engine.frame_count(), 2);
    assert_eq!(engine.current_frame().map(|f| f.turn_count), Some(3));
    assert!(engine.advance());
    assert_eq!(engine.current_frame().map(|f| f.turn_count), Some(6));
    assert!(!engine.advance());
}

#[test]
fn svg_exporter_graph_produces_valid_svg() {
    let mut graph = ForceDirectedGraph::new();
    graph.nodes.push(Node {
        id: "A".to_string(), x: 0.0, y: 0.0, dx: 0.0, dy: 0.0, weight: 1.0,
    });
    graph.nodes.push(Node {
        id: "B".to_string(), x: 10.0, y: 10.0, dx: 0.0, dy: 0.0, weight: 2.0,
    });
    graph.edges.push(crosstalk::ui::visualization::Edge { source: 0, target: 1, strength: 1.0 });
    let svg = SvgExporter::export_graph(&graph, 400.0, 300.0);
    assert!(svg.starts_with("<svg"), "must start with <svg");
    assert!(svg.ends_with("</svg>"), "must end with </svg>");
    assert!(svg.contains("A"), "must contain node A");
    assert!(svg.contains("B"), "must contain node B");
}

#[test]
fn svg_exporter_heatmap_produces_valid_svg() {
    let heatmap = vec![0.0, 0.5, 1.0, 0.3];
    let svg = SvgExporter::export_heatmap("main.rs", &heatmap, 200, 50);
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
    assert!(svg.contains("main.rs"));
}

#[test]
fn latent_mapper_returns_3d_point() {
    let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
    let point = LatentMapper::project_to_3d(&embedding);
    assert_eq!(point.len(), 3);
    assert!(point.iter().all(|v| v.is_finite()));
}

#[test]
fn heatmap_generator_maps_focus_points() {
    let content = "hello world";
    let focus = vec![0, 1, 6];
    let hm = HeatmapGenerator::generate_focus_map(content, focus);
    assert_eq!(hm.len(), content.len());
    assert!(hm[0] > 0.0);
    assert!(hm[6] > 0.0);
    assert_eq!(hm[5], 0.0);
}

#[test]
fn force_directed_graph_layout_step_moves_nodes() {
    let mut graph = ForceDirectedGraph::new();
    graph.nodes.push(Node { id: "A".to_string(), x: 0.0, y: 0.0, dx: 0.0, dy: 0.0, weight: 1.0 });
    graph.nodes.push(Node { id: "B".to_string(), x: 1.0, y: 0.0, dx: 0.0, dy: 0.0, weight: 1.0 });
    let x0 = graph.nodes[0].x;
    graph.compute_layout_step();
    let x1 = graph.nodes[0].x;
    assert_ne!(x0, x1, "layout step must move nodes");
}
