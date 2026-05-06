use crosstalk::engines::consensus::{
    InfluenceWeightManager, KalmanConvergence, NashSolver, PayoffCalculator, RefinementRound,
    ResolutionStrategy,
};
use crosstalk::engines::quality::ArtifactMetrics;
use crosstalk::types::artifact::{Artifact, ProofAttachment};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};

// ── helper ────────────────────────────────────────────────────────────────────

fn solver() -> NashSolver {
    NashSolver
}

// ── run_refinement_rounds ─────────────────────────────────────────────────────

#[test]
fn refinement_empty_proposals_returns_empty() {
    let rounds = solver().run_refinement_rounds(&[], 3);
    assert!(rounds.is_empty());
}

#[test]
fn refinement_max_rounds_zero_returns_empty() {
    let proposals = vec![("agent_a", "do X"), ("agent_b", "do Y")];
    let rounds = solver().run_refinement_rounds(&proposals, 0);
    assert!(rounds.is_empty());
}

#[test]
fn refinement_round0_contains_initial_proposals() {
    let proposals = vec![("alice", "proposal_a"), ("bob", "proposal_b")];
    let rounds = solver().run_refinement_rounds(&proposals, 3);
    let round0: Vec<&RefinementRound> = rounds.iter().filter(|r| r.round_index == 0).collect();
    assert_eq!(round0.len(), 2);
    assert!(
        round0
            .iter()
            .any(|r| r.agent_id == "alice" && r.proposed_resolution == "proposal_a")
    );
    assert!(
        round0
            .iter()
            .any(|r| r.agent_id == "bob" && r.proposed_resolution == "proposal_b")
    );
}

#[test]
fn refinement_max_rounds_respected() {
    let proposals = vec![("a", "x"), ("b", "y"), ("c", "z")];
    let max = 4usize;
    let rounds = solver().run_refinement_rounds(&proposals, max);
    let max_round = rounds.iter().map(|r| r.round_index).max().unwrap_or(0);
    assert!(max_round < max as u32, "round index must be < max_rounds");
}

#[test]
fn refinement_equilibrium_terminates_early_when_all_agree() {
    let proposals = vec![("a", "same"), ("b", "same"), ("c", "same")];
    let rounds = solver().run_refinement_rounds(&proposals, 10);
    let max_round = rounds.iter().map(|r| r.round_index).max().unwrap_or(0);
    assert_eq!(
        max_round, 0,
        "identical proposals should terminate at round 0"
    );
}

#[test]
fn refinement_single_agent_single_round() {
    let proposals = vec![("solo", "unique")];
    let rounds = solver().run_refinement_rounds(&proposals, 5);
    assert!(!rounds.is_empty());
    assert_eq!(rounds[0].agent_id, "solo");
    assert_eq!(rounds[0].proposed_resolution, "unique");
}

// ── resolve ───────────────────────────────────────────────────────────────────

#[test]
fn resolve_voting_picks_majority() {
    let proposals = vec![
        ("a", 1.0, "option_A"),
        ("b", 1.0, "option_B"),
        ("c", 1.0, "option_A"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Voting);
    assert_eq!(result, "option_A");
}

#[test]
fn resolve_weighted_average_picks_highest_weight() {
    let proposals = vec![
        ("a", 0.3, "plan_X"),
        ("b", 0.9, "plan_Y"),
        ("c", 0.2, "plan_X"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::WeightedAverage);
    assert_eq!(result, "plan_Y");
}

#[test]
fn resolve_expert_deference_picks_highest_weight_agent() {
    let proposals = vec![
        ("junior", 0.2, "my_proposal"),
        ("senior", 0.95, "expert_proposal"),
        ("mid", 0.5, "mid_proposal"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::ExpertDeference);
    assert_eq!(result, "expert_proposal");
}

#[test]
fn resolve_mediation_returns_non_empty() {
    let proposals = vec![
        ("a", 1.0, "use async runtime"),
        ("b", 1.0, "prefer async approach"),
        ("c", 1.0, "async is better"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Mediation);
    assert!(!result.is_empty());
}

#[test]
fn resolve_mediation_extracts_common_words() {
    let proposals = vec![
        ("a", 1.0, "use tokio runtime"),
        ("b", 1.0, "use tokio scheduler"),
        ("c", 1.0, "use tokio channels"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Mediation);
    assert!(result.contains("use") && result.contains("tokio"));
}

#[test]
fn resolve_empty_proposals_returns_empty() {
    let result = NashSolver::resolve(&[], ResolutionStrategy::Voting);
    assert_eq!(result, "");
}

// ── KalmanConvergence ─────────────────────────────────────────────────────────

#[test]
fn kalman_update_moves_estimate_toward_measurement() {
    let mut k = KalmanConvergence::new(0.0);
    let updated = k.update(1.0);
    assert!(
        updated > 0.0,
        "estimate should increase toward measurement 1.0"
    );
    assert!(updated <= 1.0);
}

#[test]
fn kalman_variance_decreases_after_update() {
    let mut k = KalmanConvergence::new(0.5);
    let initial_var = k.variance;
    k.update(0.8);
    assert!(
        k.variance < initial_var,
        "posterior variance must be less than prior"
    );
}

#[test]
fn kalman_innovation_stored_correctly() {
    let mut k = KalmanConvergence::new(0.3);
    k.update(0.8);
    assert!(
        k.innovation.abs() > 0.0,
        "innovation should be non-zero for differing measurement and prior"
    );
}

#[test]
fn kalman_is_converged_false_initially() {
    let k = KalmanConvergence::new(0.5);
    assert!(
        !k.is_converged(0.001),
        "fresh filter with variance=1.0 must not be converged"
    );
}

#[test]
fn kalman_is_converged_true_after_many_updates() {
    let mut k = KalmanConvergence::new(0.5);
    for _ in 0..50 {
        k.update(0.9);
    }
    assert!(
        k.is_converged(0.1),
        "variance should drop below 0.1 after 50 updates"
    );
}

#[test]
fn kalman_confidence_interval_contains_p_c() {
    let mut k = KalmanConvergence::new(0.5);
    k.update(0.7);
    let (lo, hi) = k.confidence_interval();
    assert!(
        lo <= k.p_c && k.p_c <= hi,
        "p_c must be inside its own confidence interval"
    );
    assert!(
        lo >= 0.0 && hi <= 1.0,
        "confidence interval must be clamped to [0, 1]"
    );
}

#[test]
fn kalman_p_c_bounded_after_extreme_measurements() {
    let mut k = KalmanConvergence::new(0.5);
    let v1 = k.update(2.0);
    let v2 = k.update(-1.0);
    assert!(
        v1 <= 1.0 && v2 >= 0.0,
        "p_c must stay in [0, 1] after out-of-range measurements"
    );
}

// ── InfluenceWeightManager ────────────────────────────────────────────────────

fn make_turn(model: &str, outcome: TurnOutcome, certainty: f64) -> Turn {
    Turn {
        index: 0,
        model_id: model.to_string(),
        content: String::new(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(certainty),
        outcome,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    }
}

#[test]
fn weights_empty_state_returns_empty_map() {
    let sigma = ConversationState::new("s");
    let w = InfluenceWeightManager::calculate_weights(&sigma);
    assert!(w.is_empty());
}

#[test]
fn weights_tests_passed_scores_higher_than_rolled_back() {
    let mut sigma = ConversationState::new("s");
    sigma
        .turns
        .push(make_turn("good", TurnOutcome::TestsPassed, 0.9));
    sigma
        .turns
        .push(make_turn("bad", TurnOutcome::RolledBack, 0.5));
    let w = InfluenceWeightManager::calculate_weights(&sigma);
    assert!(
        w["good"] > w["bad"],
        "TestsPassed agent must outweigh RolledBack agent"
    );
}

#[test]
fn weights_with_recency_recent_turn_outweighs_old() {
    let mut sigma = ConversationState::new("s");
    // agent_old has a good old turn, agent_new has same quality recent turn
    sigma
        .turns
        .push(make_turn("agent_old", TurnOutcome::TestsPassed, 0.9));
    sigma
        .turns
        .push(make_turn("agent_new", TurnOutcome::TestsPassed, 0.9));
    // add more new turns for agent_new to push agent_old further back
    sigma
        .turns
        .push(make_turn("agent_new", TurnOutcome::TestsPassed, 0.9));
    let w = InfluenceWeightManager::calculate_weights_with_recency(&sigma, 0.5);
    assert!(
        w["agent_new"] >= w["agent_old"],
        "recent agent should have >= weight with decay=0.5"
    );
}

#[test]
fn weights_surprise_penalty_reduces_score() {
    let mut sigma_clean = ConversationState::new("s");
    sigma_clean.turns.push(Turn {
        surprise_signal: Some(0.0),
        ..make_turn("m", TurnOutcome::Compiled, 0.8)
    });
    let mut sigma_surprised = ConversationState::new("s");
    sigma_surprised.turns.push(Turn {
        surprise_signal: Some(1.0),
        ..make_turn("m", TurnOutcome::Compiled, 0.8)
    });
    let w_clean = InfluenceWeightManager::calculate_weights(&sigma_clean);
    let w_surprised = InfluenceWeightManager::calculate_weights(&sigma_surprised);
    assert!(
        w_clean["m"] > w_surprised["m"],
        "high surprise signal must reduce agent weight"
    );
}

#[test]
fn rank_returns_descending_order() {
    let weights = std::collections::BTreeMap::from([
        ("c".to_string(), 0.3),
        ("a".to_string(), 1.5),
        ("b".to_string(), 0.9),
    ]);
    let ranked = InfluenceWeightManager::rank(&weights);
    assert_eq!(ranked[0].0, "a");
    assert_eq!(ranked[1].0, "b");
    assert_eq!(ranked[2].0, "c");
}

// ── PayoffCalculator ──────────────────────────────────────────────────────────

fn empty_artifact(name: &str) -> Artifact {
    Artifact {
        name: name.to_string(),
        language: "rust".to_string(),
        content: String::new(),
        version: 1,
        history: vec![],
        ast_versions: std::collections::BTreeMap::new(),
        proof_attachments: vec![],
        metrics: ArtifactMetrics::default(),
        skeleton: String::new(),
    }
}

#[test]
fn evaluate_returns_value_in_unit_interval() {
    let a = empty_artifact("a");
    let score = PayoffCalculator::evaluate(&a);
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn evaluate_proof_attachments_increase_score() {
    let base = empty_artifact("a");
    let mut with_proof = base.clone();
    with_proof.proof_attachments.push(ProofAttachment {
        artifact_name: "a".to_string(),
        proven_properties: vec!["safety".to_string()],
        proof_hash: "abc".to_string(),
        verified_at: 0,
    });
    assert!(
        PayoffCalculator::evaluate(&with_proof) > PayoffCalculator::evaluate(&base),
        "proof attachment must raise payoff"
    );
}

#[test]
fn evaluate_high_complexity_reduces_score() {
    let mut simple = empty_artifact("a");
    simple.metrics = ArtifactMetrics {
        cyclomatic_complexity: 1,
        line_count: 100,
        ..Default::default()
    };
    let mut complex = empty_artifact("a");
    complex.metrics = ArtifactMetrics {
        cyclomatic_complexity: 50,
        line_count: 100,
        ..Default::default()
    };
    assert!(
        PayoffCalculator::evaluate(&simple) > PayoffCalculator::evaluate(&complex),
        "high cyclomatic complexity must reduce payoff"
    );
}

#[test]
fn compute_payoff_matrix_has_correct_dimensions() {
    let a = empty_artifact("a");
    let b = empty_artifact("b");
    let c = empty_artifact("c");
    let proposals = vec![("ag1", &a, TurnOutcome::Compiled), ("ag2", &b, TurnOutcome::Compiled), ("ag3", &c, TurnOutcome::Compiled)];
    let current = empty_artifact("cur");
    let matrix = PayoffCalculator::compute_payoff_matrix(&proposals, &current);
    assert_eq!(matrix.len(), 3);
    for row in &matrix {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn compute_payoff_matrix_values_in_unit_interval() {
    let a = empty_artifact("a");
    let b = empty_artifact("b");
    let proposals = vec![("x", &a, TurnOutcome::Compiled), ("y", &b, TurnOutcome::Compiled)];
    let current = empty_artifact("cur");
    let matrix = PayoffCalculator::compute_payoff_matrix(&proposals, &current);
    for row in &matrix {
        for &v in row {
            assert!((0.0..=1.0).contains(&v), "payoff {v} out of [0,1]");
        }
    }
}

#[test]
fn best_response_returns_index_in_bounds() {
    let mine = vec![0.3, 0.7, 0.5];
    let theirs = vec![0.6, 0.2, 0.4];
    let idx = PayoffCalculator::best_response(&mine, &theirs);
    assert!(idx < mine.len());
}

#[test]
fn best_response_picks_highest_joint_payoff() {
    let mine = vec![0.1, 0.9, 0.5];
    let theirs = vec![0.1, 0.8, 0.5];
    let idx = PayoffCalculator::best_response(&mine, &theirs);
    assert_eq!(
        idx, 1,
        "strategy 1 has highest combined payoff (0.9+0.8=1.7)"
    );
}

// ── NashSolver::resolve additional strategy tests ────────────────────────────

#[test]
fn resolve_voting_tie_picks_one() {
    let proposals = vec![("a", 1.0, "X"), ("b", 1.0, "Y")];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Voting);
    assert!(result == "X" || result == "Y");
}

#[test]
fn resolve_voting_single_proposal() {
    let proposals = vec![("a", 0.5, "only_option")];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Voting);
    assert_eq!(result, "only_option");
}

#[test]
fn resolve_voting_all_same() {
    let proposals = vec![
        ("a", 0.3, "consensus"),
        ("b", 0.7, "consensus"),
        ("c", 0.1, "consensus"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Voting);
    assert_eq!(result, "consensus");
}

#[test]
fn resolve_weighted_average_single_proposal() {
    let proposals = vec![("solo", 0.42, "solo_plan")];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::WeightedAverage);
    assert_eq!(result, "solo_plan");
}

#[test]
fn resolve_weighted_average_equal_weights_picks_one() {
    let proposals = vec![("a", 0.5, "plan_A"), ("b", 0.5, "plan_B")];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::WeightedAverage);
    assert!(result == "plan_A" || result == "plan_B");
}

#[test]
fn resolve_expert_deference_single_expert() {
    let proposals = vec![("expert", 1.0, "expert_says")];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::ExpertDeference);
    assert_eq!(result, "expert_says");
}

#[test]
fn resolve_expert_deference_ignores_low_weight() {
    let proposals = vec![
        ("novice1", 0.1, "bad_idea"),
        ("novice2", 0.15, "worse_idea"),
        ("expert", 0.99, "good_idea"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::ExpertDeference);
    assert_eq!(result, "good_idea");
}

#[test]
fn resolve_mediation_single_proposal_returns_all_words() {
    let proposals = vec![("a", 1.0, "deploy to production")];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Mediation);
    // threshold = 1/2 = 0, so all words with count > 0 pass
    assert!(result.contains("deploy"));
    assert!(result.contains("production"));
}

#[test]
fn resolve_mediation_no_common_words_returns_empty() {
    let proposals = vec![
        ("a", 1.0, "alpha"),
        ("b", 1.0, "beta"),
        ("c", 1.0, "gamma"),
        ("d", 1.0, "delta"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Mediation);
    // threshold = 4/2 = 2, no word appears more than once
    assert!(result.is_empty());
}

#[test]
fn resolve_mediation_common_words_extracted() {
    let proposals = vec![
        ("a", 1.0, "refactor the auth module"),
        ("b", 1.0, "refactor the session module"),
        ("c", 1.0, "refactor the cache module"),
    ];
    let result = NashSolver::resolve(&proposals, ResolutionStrategy::Mediation);
    // threshold = 3/2 = 1, words appearing > 1 time pass
    assert!(result.contains("refactor"));
    assert!(result.contains("the"));
    assert!(result.contains("module"));
}

#[test]
fn resolve_all_strategies_handle_empty() {
    for strategy in [
        ResolutionStrategy::Voting,
        ResolutionStrategy::WeightedAverage,
        ResolutionStrategy::ExpertDeference,
        ResolutionStrategy::Mediation,
    ] {
        let result = NashSolver::resolve(&[], strategy);
        assert_eq!(result, "", "empty proposals should return empty for {:?}", strategy);
    }
}
