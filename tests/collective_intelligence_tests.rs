use crosstalk::engines::collective_intelligence::{
    CapabilityGapScanner, CollectiveIntelligenceEngine, DynamicTeamComposer, EnsembleEngine,
    KnowledgeTransfer, MetaStrategy, MetaStrategyOptimizer, PeerReview, RoleSequenceRecorder,
    SkillProgressionTracker, SwarmPremiumCalculator, UCB1ProtocolSelector,
};
use crosstalk::types::conversation::{TaskCategory, Turn, TurnOutcome};
use crosstalk::types::intelligence::AgentProfile;
use std::collections::BTreeMap;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_turn(model_id: &str, outcome: TurnOutcome, category: Option<TaskCategory>) -> Turn {
    Turn {
        index: 0,
        model_id: model_id.to_string(),
        content: "content".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: None,
        outcome,
        task_category: category,
        structure: None,
        signature: vec![],
        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    }
}

fn profile_with_score(model_id: &str, cat: TaskCategory, score: f64) -> AgentProfile {
    let mut capabilities = BTreeMap::new();
    capabilities.insert(cat, score);
    AgentProfile {
        model_id: model_id.to_string(),
        capabilities,
        total_turns: 1,
        compilation_success_rate: 1.0,
    }
}

// ── CollectiveIntelligenceEngine ──────────────────────────────────────────────

#[test]
fn specialization_creates_profile_on_first_turn() {
    let mut engine = CollectiveIntelligenceEngine::new();
    let turn = make_turn(
        "m1",
        TurnOutcome::TestsPassed,
        Some(TaskCategory::CodeGeneration),
    );
    engine.update_specialization(&turn);
    assert!(engine.profiles.contains_key("m1"));
}

#[test]
fn specialization_ema_moves_toward_score() {
    let mut engine = CollectiveIntelligenceEngine::new();
    // Default 0.5. TestPassed = 1.0. EMA: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
    let turn = make_turn(
        "m1",
        TurnOutcome::TestsPassed,
        Some(TaskCategory::CodeGeneration),
    );
    engine.update_specialization(&turn);
    let p = engine.profiles.get("m1").unwrap();
    let score = p.capabilities.get(&TaskCategory::CodeGeneration).unwrap();
    assert!((score - 0.55).abs() < 0.001);
}

// ── KnowledgeTransfer ─────────────────────────────────────────────────────────

#[test]
fn knowledge_transfer_packs_successful_lesson() {
    let turn = make_turn("m1", TurnOutcome::TestsPassed, None);
    let lesson = KnowledgeTransfer::pack_lesson(&turn).expect("lesson should be packed");
    assert_eq!(lesson.category, "success_pattern");
}

#[test]
fn knowledge_transfer_injects_relevant_lesson() {
    let lessons = vec![crosstalk::types::memory::TransferableLesson {
        category: "coding".to_string(),
        content: "use a mutex".to_string(),
        confidence: 0.9,
        applicability_tags: vec!["async".to_string()],
    }];
    let injected = KnowledgeTransfer::inject("base context", &lessons, "coding", &[]);
    assert!(injected.contains("use a mutex"));
}

// ── PeerReview ────────────────────────────────────────────────────────────────

#[test]
fn peer_review_penalizes_incomplete_code() {
    let report = PeerReview::review("rev1", "// TODO: implement this");
    assert!(report.correctness < 0.7);
    assert!(report.comments.iter().any(|c| c.contains("TODO")));
}

// ── EnsembleEngine ────────────────────────────────────────────────────────────

#[test]
fn ensemble_merge_selects_best_paragraphs() {
    let proposals = vec![
        (
            "m1".to_string(),
            "Para 1 from m1.\n\nPara 2 from m1.".to_string(),
            0.8,
        ),
        (
            "m2".to_string(),
            "Para 1 from m2.\n\nPara 2 from m2.".to_string(),
            0.9,
        ),
    ];
    let merged = EnsembleEngine::merge_proposals(proposals, TaskCategory::Research, "");
    assert!(merged.contains("Para 1 from m2"));
    assert!(merged.contains("Para 2 from m2"));
}

// ── SwarmPremiumCalculator ────────────────────────────────────────────────────

#[test]
fn swarm_premium_computes_improvement_percentage() {
    let mut scores = BTreeMap::new();
    scores.insert("m1".to_string(), 0.8);
    scores.insert("m2".to_string(), 0.7);
    let p = SwarmPremiumCalculator::compute("coding", 0.95, &scores);
    // (0.95 - 0.8) / 0.8 = 0.1875 = 18.75%
    assert!((p.premium_pct - 18.75).abs() < 0.01);
}

// ── MetaStrategyOptimizer ─────────────────────────────────────────────────────

#[test]
fn meta_optimizer_records_and_ranks_strategies() {
    let mut opt = MetaStrategyOptimizer::new();
    for _ in 0..5 {
        opt.record(MetaStrategy::DebateAndCritique, 0.9);
        opt.record(MetaStrategy::DirectImplementation, 0.4);
    }
    assert_eq!(opt.best_strategy(), Some(MetaStrategy::DebateAndCritique));
}

// ── DynamicTeamComposer ───────────────────────────────────────────────────────

#[test]
fn team_composer_selects_best_models_per_role() {
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "m1".to_string(),
        profile_with_score("m1", TaskCategory::Architecture, 0.9),
    );
    profiles.insert(
        "m2".to_string(),
        profile_with_score("m2", TaskCategory::Architecture, 0.7),
    );
    profiles.insert(
        "m3".to_string(),
        profile_with_score("m3", TaskCategory::Architecture, 0.5),
    );

    let team = DynamicTeamComposer::compose(&profiles, "arch");
    assert_eq!(team.architect, Some("m1".to_string()));
    assert_eq!(team.coder, Some("m2".to_string()));
}

// ── SkillProgressionTracker ───────────────────────────────────────────────────

#[test]
fn skill_tracker_detects_plateau() {
    let mut tracker = SkillProgressionTracker::new();
    for _ in 0..10 {
        tracker.record("m1", "coding", 0.8);
    }
    assert!(tracker.get("m1", "coding").unwrap().is_plateauing());
}

// ── CapabilityGapScanner ──────────────────────────────────────────────────────

#[test]
fn gap_scanner_identifies_weak_task_types() {
    let mut profiles = BTreeMap::new();
    // Best score for CodeGeneration is 0.4, threshold is 0.6
    profiles.insert(
        "m1".to_string(),
        profile_with_score("m1", TaskCategory::CodeGeneration, 0.4),
    );

    let gaps = CapabilityGapScanner::scan_default(&profiles);
    assert!(gaps.iter().any(|g| g.task_type == "CodeGeneration"));
}

// ── UCB1ProtocolSelector ──────────────────────────────────────────────────────

#[test]
fn protocol_selector_prefers_untried_arms() {
    let mut sel = UCB1ProtocolSelector::new(&["p1", "p2"]);
    sel.update("p1", 0.9);
    // p2 is untried (inf), should be selected
    assert_eq!(sel.select(), Some("p2"));
}

// ── RoleSequenceRecorder ──────────────────────────────────────────────────────

#[test]
fn role_recorder_returns_best_ordering() {
    let mut rec = RoleSequenceRecorder::new();
    rec.record("coding", vec![("m1".to_string(), "arch".to_string())], 0.5);
    rec.record("coding", vec![("m2".to_string(), "arch".to_string())], 0.9);

    let best = rec.best_ordering("coding").unwrap();
    assert_eq!(best[0].0, "m2");
}
