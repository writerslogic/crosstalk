use crosstalk::engines::consensus::{NashSolver, RefinementRound, ResolutionStrategy};

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
    assert!(round0.iter().any(|r| r.agent_id == "alice" && r.proposed_resolution == "proposal_a"));
    assert!(round0.iter().any(|r| r.agent_id == "bob" && r.proposed_resolution == "proposal_b"));
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
    assert_eq!(max_round, 0, "identical proposals should terminate at round 0");
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
