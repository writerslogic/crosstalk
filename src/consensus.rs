use crate::types::{Artifact, ConversationState, TurnOutcome};
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
    /// Hardened: Analyzes commitment language and structural stability.
    pub fn compute(content: &str, volatility: f64) -> f64 {
        let hedging_terms = [
            "maybe", "perhaps", "i think", "possibly", "could be", "unsure", "not sure", "might",
        ];
        let strong_terms = [
            "certainly", "definitely", "correct", "fix", "optimal", "verified", "must",
        ];
        
        let content_lower = content.to_lowercase();

        let mut score: f64 = 0.7; // Base score

        for term in hedging_terms {
            if content_lower.contains(term) {
                score -= 0.1;
            }
        }
        for term in strong_terms {
            if content_lower.contains(term) {
                score += 0.05;
            }
        }

        // Penalty for high volatility (node is changing too often)
        score -= volatility * 0.2;

        score.clamp(0.1, 1.0)
    }
}

/// Solves for Nash Equilibrium in game-theoretical payoffs.
pub struct NashSolver;

impl NashSolver {
    /// Solves a 2x2 payoff matrix for pure strategy equilibria.
    #[must_use]
    pub fn solve_2x2_pure(matrix: &[[(f64, f64); 2]; 2]) -> Vec<(usize, usize)> {
        let mut equilibria = vec![];
        for r in 0..2 {
            for c in 0..2 {
                let (p1_payoff, p2_payoff) = matrix[r][c];
                let p1_is_best = p1_payoff >= matrix[1 - r][c].0;
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
    pub p_c: f64,
    pub variance: f64,
}

impl KalmanConvergence {
    #[must_use]
    pub fn new(initial_p: f64) -> Self {
        Self {
            p_c: initial_p,
            variance: 1.0,
        }
    }

    pub fn update(&mut self, measurement: f64) -> f64 {
        let process_noise = 0.01;
        let measurement_noise = 0.1;
        self.variance += process_noise;
        let kalman_gain = self.variance / (self.variance + measurement_noise);
        self.p_c += kalman_gain * (measurement - self.p_c);
        self.variance *= 1.0 - kalman_gain;
        self.p_c.clamp(0.0, 1.0)
    }
}

/// Manages agent influence weights based on certainty history and calibration.
pub struct InfluenceWeightManager;

impl InfluenceWeightManager {
    #[must_use]
    pub fn calculate_weights(sigma: &ConversationState) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        let mut agent_stats: HashMap<String, (f64, f64)> = HashMap::new(); // (weighted_score, total_turns)

        for turn in &sigma.turns {
            let (score, weight) = agent_stats.entry(turn.model_id.clone()).or_insert((0.0, 0.0));
            
            // Hardening: Calibrate weight based on outcome
            let outcome_factor = match turn.outcome {
                TurnOutcome::TestsPassed => 1.2,
                TurnOutcome::Compiled => 1.0,
                TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.5,
                _ => 0.8,
            };
            
            let certainty = turn.certainty.unwrap_or(0.5);
            *score += certainty * outcome_factor;
            *weight += 1.0;
        }

        for (id, (score, count)) in agent_stats {
            weights.insert(id, (score / count).clamp(0.1, 2.0));
        }

        weights
    }
}

pub struct PayoffCalculator;

impl PayoffCalculator {
    #[must_use]
    pub fn evaluate(artifact: &Artifact) -> f64 {
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
    fn test_certainty_analyzer_hardened() {
        let sure = "I have definitely verified this fix.";
        let unsure = "Maybe we could try this, possibly.";
        assert!(CertaintyAnalyzer::compute(sure, 0.0) > CertaintyAnalyzer::compute(unsure, 0.0));
        assert!(CertaintyAnalyzer::compute(sure, 0.0) > CertaintyAnalyzer::compute(sure, 0.8));
    }

    #[test]
    fn test_kalman_convergence() {
        let mut kalman = KalmanConvergence::new(0.1);
        for _ in 0..5 { kalman.update(0.9); }
        assert!(kalman.p_c > 0.5);
    }
}
