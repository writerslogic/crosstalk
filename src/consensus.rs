use crate::types::{Artifact, ConversationState};
use std::collections::HashMap;

pub struct ConsensusEngine;

impl ConsensusEngine {
    pub fn init() {
        println!("Initialized consensus engine");
    }
}

/// Computes a "Self-Certainty" score for an agent's turn.
pub struct CertaintyAnalyzer;

impl CertaintyAnalyzer {
    /// Computes certainty score in [0.0, 1.0].
    /// In this implementation, we use a heuristic based on hedging language and consistency.
    pub fn compute(content: &str) -> f64 {
        let hedging_terms = [
            "maybe", "perhaps", "i think", "possibly", "could be", "unsure", "not sure", "might",
        ];
        let content_lower = content.to_lowercase();

        let mut penalty = 0.0;
        for term in hedging_terms {
            if content_lower.contains(term) {
                penalty += 0.1;
            }
        }

        // Higher length might imply more detail/certainty, but too long might be rambling.
        // Let's keep it simple: 1.0 - penalty, clamped.
        f64::clamp(1.0 - penalty, 0.1, 1.0)
    }
}

/// Solves for Nash Equilibrium in game-theoretical payoffs.
pub struct NashSolver;

impl NashSolver {
    /// Solves a 2x2 payoff matrix for pure strategy equilibria.
    /// Returns a list of (row_idx, col_idx) that are Nash Equilibria.
    #[must_use]
    pub fn solve_2x2_pure(matrix: &[[(f64, f64); 2]; 2]) -> Vec<(usize, usize)> {
        let mut equilibria = vec![];
        for r in 0..2 {
            for c in 0..2 {
                let (p1_payoff, p2_payoff) = matrix[r][c];

                // Check if Player 1 has a better move given Player 2's choice c
                let p1_is_best = p1_payoff >= matrix[1 - r][c].0;

                // Check if Player 2 has a better move given Player 1's choice r
                let p2_is_best = p2_payoff >= matrix[r][1 - c].1;

                if p1_is_best && p2_is_best {
                    equilibria.push((r, c));
                }
            }
        }
        equilibria
    }
}

/// Kalman Filter for convergence estimation.
pub struct KalmanConvergence {
    pub p_c: f64, // State: Probability of completion
    pub variance: f64,
}

impl KalmanConvergence {
    #[must_use]
    pub fn new() -> Self {
        Self {
            p_c: 0.1, // Start with low probability
            variance: 1.0,
        }
    }

    /// Updates the state with a new measurement (progress signal).
    /// measurement: [0.0, 1.0] where 1.0 is "perfectly consistent/correct".
    pub fn update(&mut self, measurement: f64) -> f64 {
        let process_noise = 0.01;
        let measurement_noise = 0.1;

        // Prediction
        self.variance += process_noise;

        // Update
        let kalman_gain = self.variance / (self.variance + measurement_noise);
        self.p_c += kalman_gain * (measurement - self.p_c);
        self.variance *= 1.0 - kalman_gain;

        self.p_c = self.p_c.clamp(0.0, 1.0);
        self.p_c
    }
}

impl Default for KalmanConvergence {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages agent influence weights based on certainty history.
pub struct InfluenceWeightManager;

impl InfluenceWeightManager {
    #[must_use]
    pub fn calculate_weights(sigma: &ConversationState) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        let mut agent_scores: HashMap<String, Vec<f64>> = HashMap::new();

        for turn in &sigma.turns {
            if let Some(c) = turn.certainty {
                agent_scores
                    .entry(turn.model_id.clone())
                    .or_default()
                    .push(c);
            }
        }

        for (model_id, scores) in agent_scores {
            if scores.is_empty() {
                weights.insert(model_id, 1.0);
                continue;
            }

            // Weighted average with exponential decay (recency bias)
            let mut total_weight = 0.0;
            let mut weighted_sum = 0.0;
            for (i, &score) in scores.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let recency = (i as f64 + 1.0).powf(0.5); // Square root growth
                weighted_sum += score * recency;
                total_weight += recency;
            }

            weights.insert(model_id, weighted_sum / total_weight);
        }

        weights
    }
}

/// Evaluates artifact quality payoffs.
pub struct PayoffCalculator;

impl PayoffCalculator {
    #[must_use]
    pub fn evaluate(artifact: &Artifact) -> f64 {
        // Simplified scoring
        #[allow(clippy::cast_precision_loss)]
        let length_bonus = (artifact.content.len() as f64 / 1000.0).min(0.2);
        #[allow(clippy::cast_precision_loss)]
        let history_penalty = (artifact.history.len() as f64 * 0.05).min(0.3);

        (0.7 + length_bonus - history_penalty).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certainty_analyzer() {
        let sure = "I have implemented the solution correctly.";
        let unsure = "Maybe we could try another approach, perhaps.";
        assert!(CertaintyAnalyzer::compute(sure) > CertaintyAnalyzer::compute(unsure));
    }

    #[test]
    fn test_nash_solver_prisoners_dilemma() {
        // (P1, P2) payoffs
        // Both Cooperate: (3, 3)
        // P1 Defects, P2 Cooperates: (5, 0)
        // P1 Cooperates, P2 Defects: (0, 5)
        // Both Defect: (1, 1)
        let matrix = [
            [(3.0, 3.0), (0.0, 5.0)], // P1 Cooperate
            [(5.0, 0.0), (1.0, 1.0)], // P1 Defect
        ];
        let eq = NashSolver::solve_2x2_pure(&matrix);
        assert_eq!(eq.len(), 1);
        assert_eq!(eq[0], (1, 1)); // (Defect, Defect) is the Nash Equilibrium
    }

    #[test]
    fn test_kalman_convergence() {
        let mut kalman = KalmanConvergence::new();
        let initial = kalman.p_c;

        // Sequence of high progress
        for _ in 0..5 {
            kalman.update(0.9);
        }
        assert!(kalman.p_c > initial);
        assert!(kalman.p_c > 0.5);
    }
}
