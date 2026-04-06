use crate::types::artifact::Artifact;
use crate::types::conversation::{ConversationState, TurnOutcome};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RefinementRound {
    pub round_index: u32,
    pub agent_id: String,
    pub critique: String,
    pub proposed_resolution: String,
    pub accepted: bool,
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    Voting,
    WeightedAverage,
    ExpertDeference,
    Mediation,
}

pub struct ConsensusEngine;

impl ConsensusEngine {
    pub fn init() {}
}

/// Computes a "Self-Certainty" score for an agent's turn.
pub struct CertaintyAnalyzer;

impl CertaintyAnalyzer {
    /// Computes certainty score in [0.0, 1.0].
    pub fn compute(content: &str, volatility: f64) -> f64 {
        let hedging_terms = [
            "maybe", "perhaps", "i think", "possibly", "could be", "unsure", "not sure", "might",
        ];
        let strong_terms = [
            "certainly",
            "definitely",
            "correct",
            "fix",
            "optimal",
            "verified",
            "must",
        ];

        let content_lower = content.to_lowercase();
        let mut score: f64 = 0.7;

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

        score -= volatility * 0.2;
        score.clamp(0.1, 1.0)
    }
}

/// Solves for Nash Equilibrium in game-theoretical payoffs.
pub struct NashSolver;

impl NashSolver {
    /// Multi-round refinement: agents critique each other until equilibrium or max_rounds.
    #[must_use]
    pub fn run_refinement_rounds(
        &self,
        proposals: &[(&str, &str)],
        max_rounds: usize,
    ) -> Vec<RefinementRound> {
        let mut rounds = Vec::new();

        if proposals.is_empty() || max_rounds == 0 {
            return rounds;
        }

        // Round 0: collect initial proposals
        for (agent_id, proposal) in proposals {
            rounds.push(RefinementRound {
                round_index: 0,
                agent_id: agent_id.to_string(),
                critique: String::new(),
                proposed_resolution: proposal.to_string(),
                accepted: false,
            });
        }

        // Rounds 1..max_rounds: each agent critiques others
        for round in 1..max_rounds {
            let prev_proposals: Vec<String> = proposals.iter().map(|(_, p)| p.to_string()).collect();
            let all_same = prev_proposals.windows(2).all(|w| w[0] == w[1]);
            if all_same {
                // Nash equilibrium: mark last round as accepted and stop
                for r in rounds.iter_mut().filter(|r| r.round_index == round as u32 - 1) {
                    r.accepted = true;
                }
                break;
            }

            for (i, (agent_id, proposal)) in proposals.iter().enumerate() {
                let others: Vec<&str> = prev_proposals
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, p)| p.as_str())
                    .collect();
                let critique = if others.is_empty() {
                    String::new()
                } else {
                    format!("Disputes: {}", others.join("; "))
                };
                rounds.push(RefinementRound {
                    round_index: round as u32,
                    agent_id: agent_id.to_string(),
                    critique,
                    proposed_resolution: proposal.to_string(),
                    accepted: false,
                });
            }
        }

        rounds
    }

    /// Resolve disagreements using the given strategy.
    /// `proposals`: (agent_id, influence_weight, proposal_text)
    #[must_use]
    pub fn resolve(proposals: &[(&str, f64, &str)], strategy: ResolutionStrategy) -> String {
        if proposals.is_empty() {
            return String::new();
        }

        match strategy {
            ResolutionStrategy::Voting => {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for (_, _, text) in proposals {
                    *counts.entry(text).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, _)| t.to_string())
                    .unwrap_or_default()
            }
            ResolutionStrategy::WeightedAverage => {
                let mut weights: HashMap<&str, f64> = HashMap::new();
                for (_, w, text) in proposals {
                    *weights.entry(text).or_insert(0.0) += w;
                }
                weights
                    .into_iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(t, _)| t.to_string())
                    .unwrap_or_default()
            }
            ResolutionStrategy::ExpertDeference => proposals
                .iter()
                .max_by(|(_, a, _), (_, b, _)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(_, _, t)| t.to_string())
                .unwrap_or_default(),
            ResolutionStrategy::Mediation => {
                let all_words: Vec<Vec<&str>> = proposals
                    .iter()
                    .map(|(_, _, t)| t.split_whitespace().collect())
                    .collect();
                let common: Vec<&&str> = if let Some(first) = all_words.first() {
                    first
                        .iter()
                        .filter(|w| all_words.iter().all(|words| words.contains(w)))
                        .collect()
                } else {
                    vec![]
                };
                if common.is_empty() {
                    proposals[0].2.to_string()
                } else {
                    common.iter().map(|w| **w).collect::<Vec<_>>().join(" ")
                }
            }
        }
    }

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
        let mut agent_stats: HashMap<String, (f64, f64)> = HashMap::new();

        for turn in &sigma.turns {
            let (score, weight) = agent_stats
                .entry(turn.model_id.clone())
                .or_insert((0.0, 0.0));

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
