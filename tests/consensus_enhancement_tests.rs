use crosstalk::engines::consensus::{CertaintyAnalyzer, KalmanConvergence, NashSolver};

#[test]
fn test_nash_solver_prisoners_dilemma() {
    let matrix = [[(3.0, 3.0), (0.0, 5.0)], [(5.0, 0.0), (1.0, 1.0)]];
    let equilibria = NashSolver::solve_2x2_pure(&matrix);
    assert!(equilibria.contains(&(1, 1)));
}

#[test]
fn test_nash_solver_pareto_optimality() {
    // Stag Hunt: Two Nash Equilibria (0,0) and (1,1)
    // (0,0) is payoff dominant (Pareto Optimal)
    let matrix = [[(5.0, 5.0), (0.0, 4.0)], [(4.0, 0.0), (2.0, 2.0)]];
    let equilibria = NashSolver::solve_2x2_pure(&matrix);
    assert!(equilibria.contains(&(0, 0)));
    assert!(equilibria.contains(&(1, 1)));
}

#[test]
fn test_kalman_adaptive_noise_rejection() {
    let mut kalman = KalmanConvergence::new(0.5);

    // High certainty measurement (0.1 noise)
    let p1 = kalman.update_adaptive(1.0, 1.0);
    let delta1 = (p1 - 0.5).abs();

    // Low certainty measurement (10.0 noise)
    let mut kalman2 = KalmanConvergence::new(0.5);
    let p2 = kalman2.update_adaptive(1.0, 0.01);
    let delta2 = (p2 - 0.5).abs();

    assert!(
        delta1 > delta2,
        "High certainty update should move the state more than low certainty update"
    );
}

#[test]
fn test_certainty_analyzer_sovereign_heuristic() {
    let confident =
        "I have successfully implemented the proof and verified it matches all invariants.";
    let unsure = "I think maybe it works but i am not totally sure about the edge cases.";

    let score_confident = CertaintyAnalyzer::compute(confident, 0.0);
    let score_unsure = CertaintyAnalyzer::compute(unsure, 0.0);

    assert!(score_confident > score_unsure);
    assert!(score_confident > 0.8);
}
