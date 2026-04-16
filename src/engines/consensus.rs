use crate::types::artifact::Artifact;
use crate::types::conversation::{ConversationState, TaskCategory, TurnOutcome};
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

        let hedging_count = hedging_terms
            .iter()
            .filter(|&&t| content_lower.contains(t))
            .count();
        let hedging_penalty = (hedging_count.min(4) as f64) * 0.1;

        let strong_bonus: f64 = strong_terms
            .iter()
            .filter(|&&t| content_lower.contains(t))
            .count() as f64
            * 0.05;

        let adj_volatility = volatility.clamp(0.0, 1.0);
        let score = 0.7 - hedging_penalty + strong_bonus - adj_volatility * 0.2;
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
        if proposals.is_empty() || max_rounds == 0 { return rounds; }

        for (agent_id, proposal) in proposals {
            rounds.push(RefinementRound {
                round_index: 0,
                agent_id: agent_id.to_string(),
                critique: String::new(),
                proposed_resolution: proposal.to_string(),
                accepted: false,
            });
        }

        let mut prev_resolutions: Vec<String> = proposals.iter().map(|(_, p)| p.to_string()).collect();
        for round in 1..max_rounds {
            let all_same = prev_resolutions.windows(2).all(|w| w[0] == w[1]);
            if all_same { break; }

            let mut current_resolutions = Vec::new();
            for (i, (agent_id, proposal)) in proposals.iter().enumerate() {
                let others: Vec<&str> = prev_resolutions.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, p)| p.as_str()).collect();
                current_resolutions.push(proposal.to_string());
                let critique = if others.is_empty() { String::new() } else { format!("Disputes: {}", others.join("; ")) };
                rounds.push(RefinementRound {
                    round_index: round as u32,
                    agent_id: agent_id.to_string(),
                    critique,
                    proposed_resolution: proposal.to_string(),
                    accepted: false,
                });
            }
            if current_resolutions.windows(2).all(|w| w[0] == w[1]) { break; }
            prev_resolutions = current_resolutions;
        }
        rounds
    }

    /// Resolve disagreements using the given strategy.
    #[must_use]
    pub fn resolve(proposals: &[(&str, f64, &str)], strategy: ResolutionStrategy) -> String {
        if proposals.is_empty() { return String::new(); }
        match strategy {
            ResolutionStrategy::Voting => {
                let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
                for (_, _, text) in proposals { *counts.entry(text).or_insert(0) += 1; }
                counts.into_iter().max_by(|(t1, c1), (t2, c2)| c1.cmp(c2).then_with(|| t1.cmp(t2))).map(|(t, _)| t.to_string()).unwrap_or_default()
            }
            ResolutionStrategy::WeightedAverage => {
                let mut weights: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
                for (_, w, text) in proposals { *weights.entry(text).or_insert(0.0) += w; }
                weights.into_iter().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(t, _)| t.to_string()).unwrap_or_default()
            }
            ResolutionStrategy::ExpertDeference => proposals.iter().max_by(|(_, a, _), (_, b, _)| a.total_cmp(b)).map(|(_, _, t)| t.to_string()).unwrap_or_default(),
            ResolutionStrategy::Mediation => {
                let all_words: Vec<std::collections::HashSet<&str>> = proposals.iter().map(|(_, _, t)| t.split_whitespace().collect()).collect();
                proposals.iter().max_by_key(|(_, _, t)| {
                    let words: std::collections::HashSet<&str> = t.split_whitespace().collect();
                    all_words.iter().map(|other| words.intersection(other).count()).sum::<usize>()
                }).map(|(_, _, t)| t.to_string()).unwrap_or_default()
            }
        }
    }

    /// Find the Pure Strategy Nash Equilibria (PSNE) for an N-player game.
    pub fn find_nash_equilibrium(matrix: &[Vec<Vec<f64>>]) -> Option<usize> {
        let n = matrix.len();
        if n == 0 { return None; }
        let strategies = matrix[0].len();
        let mut psne_indices = Vec::new();
        for s in 0..strategies {
            let mut is_nash = true;
            for i in 0..n {
                let current_payoff = matrix[i][s][s];
                for alt_s in 0..strategies {
                    if alt_s != s && matrix[i][alt_s][s] > current_payoff + 1e-5 {
                        is_nash = false;
                        break;
                    }
                }
                if !is_nash { break; }
            }
            if is_nash { psne_indices.push(s); }
        }
        // Pareto Efficiency: maximize total welfare among Nash Equilibria
        psne_indices.into_iter().max_by(|&a, &b| {
            let welfare_a: f64 = (0..n).map(|i| matrix[i][a][a]).sum();
            let welfare_b: f64 = (0..n).map(|i| matrix[i][b][b]).sum();
            welfare_a.total_cmp(&welfare_b)
        })
    }

    /// Resolves the optimal artifact proposal using Game Theoretical consensus.
    pub fn resolve_optimal_proposal(proposals: &[(&str, &Artifact, TurnOutcome)], current: &Artifact) -> usize {
        let matrix = PayoffCalculator::compute_payoff_matrix(proposals, current);
        Self::find_nash_equilibrium(&matrix).unwrap_or_else(|| {
            // Fallback to Social Welfare Maximization
            let n = proposals.len();
            (0..n).max_by(|&a, &b| {
                let welfare_a: f64 = (0..n).map(|i| matrix[i][a][a]).sum();
                let welfare_b: f64 = (0..n).map(|i| matrix[i][b][b]).sum();
                welfare_a.total_cmp(&welfare_b)
            }).unwrap_or(0)
        })
    }

    /// Legacy 2x2 solver for backward compatibility and testing.
    pub fn solve_2x2_pure(matrix: &[[(f64, f64); 2]; 2]) -> Vec<(usize, usize)> {
        let mut equilibria = vec![];
        for r in 0..2 {
            for c in 0..2 {
                let (p1_payoff, p2_payoff) = matrix[r][c];
                if p1_payoff >= matrix[1 - r][c].0 && p2_payoff >= matrix[r][1 - c].1 {
                    equilibria.push((r, c));
                }
            }
        }
        equilibria
    }
}

/// Kalman Filter for convergence estimation.
///
/// Tracks the completion probability `p_c` as a latent state, updating it
/// with noisy measurements from each turn. Stores the last innovation
/// (measurement residual) for diagnostics and exposes a `is_converged`
/// predicate once posterior variance drops below a caller-supplied threshold.
pub struct KalmanConvergence {
    /// Posterior estimate of completion probability.
    pub p_c: f64,
    /// Posterior error covariance (uncertainty in `p_c`).
    pub variance: f64,
    /// Last measurement residual (measurement − prior prediction).
    pub innovation: f64,
}

impl KalmanConvergence {
    const PROCESS_NOISE: f64 = 0.002; // Low drift assumption

    pub fn new(initial_p: f64) -> Self {
        Self {
            p_c: initial_p.clamp(0.0, 1.0),
            variance: 0.2,
            innovation: 0.0,
        }
    }

    /// Adaptive Update: adjust measurement noise R based on turn certainty.
    /// Certainty 2208 [0, 1]. R = R_base / (Certainty + 03b5)
    pub fn update_adaptive(&mut self, measurement: f64, certainty: f64) -> f64 {
        self.variance += Self::PROCESS_NOISE;

        // Adaptive Measurement Noise: lower certainty = higher noise = lower Kalman Gain
        let r_base = 0.1;
        let r_adaptive = r_base / (certainty.max(0.01));

        let innovation_covariance = self.variance + r_adaptive;
        let kalman_gain = self.variance / innovation_covariance;

        self.innovation = measurement - self.p_c;
        self.p_c += kalman_gain * self.innovation;
        self.variance *= 1.0 - kalman_gain;
        self.variance = self.variance.max(1e-10);

        self.p_c.clamp(0.0, 1.0)
    }


    pub fn check_stalling(&self, history: &[f64]) -> bool {
        if history.len() < 3 { return false; }
        let mean_velocity = history.windows(2).map(|w| w[1] - w[0]).sum::<f64>() / (history.len() - 1) as f64;
        mean_velocity.abs() < 0.01
    }


    /// Returns `true` once the posterior variance falls below `threshold`,
    /// indicating the estimate has stabilised.
    #[must_use]
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.variance < threshold
    }

    /// 95 % confidence interval around the current `p_c` estimate.
    /// Interval is clamped to [0, 1].
    #[must_use]
    pub fn confidence_interval(&self) -> (f64, f64) {
        let half_width = 1.96 * self.variance.sqrt();
        (
            (self.p_c - half_width).max(0.0),
            (self.p_c + half_width).min(1.0),
        )
    }
}

fn outcome_factor(outcome: &TurnOutcome) -> f64 {
    match outcome {
        TurnOutcome::TestsPassed => 1.2,
        TurnOutcome::Compiled => 1.0,
        TurnOutcome::AdvancedConvergence => 1.1,
        TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.4,
        TurnOutcome::Stalled => 0.6,
        TurnOutcome::Unknown => 0.8,
    }
}

/// Manages agent influence weights based on certainty history and calibration.
pub struct InfluenceWeightManager;

impl InfluenceWeightManager {
    /// Compute weights using a flat average of certainty × outcome factor.
    /// Delegates to `calculate_weights_with_recency` with a 0.9 decay factor.
    #[must_use]
    pub fn calculate_weights(sigma: &ConversationState) -> std::collections::BTreeMap<String, f64> {
        Self::calculate_weights_with_recency(sigma, 0.9)
    }

    /// Compute weights with exponential recency decay and surprise calibration.
    ///
    /// `decay ∈ (0, 1]`: weight for a turn `k` steps ago = `decay^k`.
    /// A `surprise_signal` close to 1.0 reduces the weight (high surprise =
    /// the agent was less reliable on that turn).
    #[must_use]
    pub fn calculate_weights_with_recency(
        sigma: &ConversationState,
        decay: f64,
    ) -> std::collections::BTreeMap<String, f64> {
        let n = sigma.turns.len();
        let mut agent_stats: std::collections::BTreeMap<String, (f64, f64)> =
            std::collections::BTreeMap::new();

        for (i, turn) in sigma.turns.iter().enumerate() {
            let steps_ago = (n - 1).saturating_sub(i);
            let recency = decay.powi(steps_ago as i32);

            let of = outcome_factor(&turn.outcome);

            let certainty = turn.certainty.unwrap_or(0.5);
            // Surprise > 0.5 means the agent behaved unexpectedly — reduce trust.
            let surprise_factor = turn
                .surprise_signal
                .map(|s| 1.0 - (s - 0.5).max(0.0) * 0.4)
                .unwrap_or(1.0);

            let contribution = certainty * of * surprise_factor * recency;
            let (score, weight) = agent_stats
                .entry(turn.model_id.clone())
                .or_insert((0.0, 0.0));
            *score += contribution;
            *weight += recency;
        }

        agent_stats
            .into_iter()
            .map(|(id, (score, weight))| {
                let w = if weight > 0.0 {
                    (score / weight).clamp(0.1, 2.0)
                } else {
                    1.0
                };
                (id, w)
            })
            .collect()
    }

    /// Compute weights filtered to turns matching `category`.
    ///
    /// Agents with no turns in the category fall back to a dampened global
    /// weight (`global * 0.3`), so specialists dominate in their domain while
    /// generalists still contribute a baseline.
    #[must_use]
    pub fn calculate_weights_for_category(
        sigma: &ConversationState,
        category: TaskCategory,
        decay: f64,
    ) -> std::collections::BTreeMap<String, f64> {
        let category_turns: Vec<_> = sigma
            .turns
            .iter()
            .enumerate()
            .filter(|(_, t)| t.task_category == Some(category))
            .collect();

        if category_turns.is_empty() {
            return Self::calculate_weights_with_recency(sigma, decay);
        }

        let n = category_turns.len();
        let mut agent_stats: std::collections::BTreeMap<String, (f64, f64)> =
            std::collections::BTreeMap::new();

        for (rank, (_, turn)) in category_turns.iter().enumerate() {
            let steps_ago = (n - 1).saturating_sub(rank);
            let recency = decay.powi(steps_ago as i32);

            let of = outcome_factor(&turn.outcome);

            let certainty = turn.certainty.unwrap_or(0.5);
            let surprise_factor = turn
                .surprise_signal
                .map(|s| 1.0 - (s - 0.5).max(0.0) * 0.4)
                .unwrap_or(1.0);

            let contribution = certainty * of * surprise_factor * recency;
            let (score, weight) = agent_stats
                .entry(turn.model_id.clone())
                .or_insert((0.0, 0.0));
            *score += contribution;
            *weight += recency;
        }

        let category_weights: std::collections::BTreeMap<String, f64> = agent_stats
            .into_iter()
            .map(|(id, (score, weight))| {
                let w = if weight > 0.0 {
                    (score / weight).clamp(0.1, 2.0)
                } else {
                    1.0
                };
                (id, w)
            })
            .collect();

        let global_weights = Self::calculate_weights_with_recency(sigma, decay);
        let mut merged = category_weights.clone();
        for (id, gw) in &global_weights {
            merged.entry(id.clone()).or_insert(gw * 0.3);
        }
        merged
    }

    /// Return agents sorted by weight descending, highest-influence first.
    #[must_use]
    pub fn rank(weights: &std::collections::BTreeMap<String, f64>) -> Vec<(String, f64)> {
        let mut sorted: Vec<(String, f64)> = weights.iter().map(|(k, v)| (k.clone(), *v)).collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        sorted
    }
}

/// Game-theoretical payoff evaluator for artifact proposals.
///
/// `evaluate` scores a single artifact on [0, 1].
/// `compute_payoff_matrix` produces an N×N matrix of payoffs for N competing
/// proposals so that NashSolver can find pure strategy equilibria.
/// `best_response` returns the strategy index that maximises the caller's
/// expected payoff given an opponent's payoff vector.
pub struct PayoffCalculator;

impl PayoffCalculator {
    pub fn evaluate(artifact: &Artifact, outcome: &TurnOutcome) -> f64 {
        let m = &artifact.metrics;
        let correctness = match outcome {
            TurnOutcome::TestsPassed => 1.0,
            TurnOutcome::Compiled => 0.8,
            TurnOutcome::AdvancedConvergence => 0.7,
            TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.0,
            _ => 0.5,
        };
        let performance = if m.line_count > 0 {
            let size_penalty = (m.line_count as f64 / 2000.0).min(0.2);
            1.0 - size_penalty
        } else { 0.6 };
        let maintainability = if m.line_count > 0 {
            let complexity_penalty = (m.cyclomatic_complexity as f64 * 0.02).min(0.3);
            let coupling_penalty = (m.coupling_factor as f64 * 0.01).min(0.2);
            let comment_bonus = (m.comment_density * 0.2).min(0.15);
            (0.8 - complexity_penalty - coupling_penalty + comment_bonus).clamp(0.0, 1.0)
        } else { 0.6 };
        let alignment = (artifact.proof_attachments.len() as f64 * 0.05).min(1.0);
        (correctness * 0.4 + performance * 0.2 + maintainability * 0.3 + alignment * 0.1).clamp(0.0, 1.0)
    }

    pub fn compute_payoff_matrix(proposals: &[(&str, &Artifact, TurnOutcome)], current: &Artifact) -> Vec<Vec<f64>> {
        let current_score = Self::evaluate(current, &TurnOutcome::Unknown);
        let scores: Vec<f64> = proposals.iter().map(|(_, a, o)| Self::evaluate(a, o)).collect();
        scores.iter().map(|&my_score| {
            scores.iter().map(|&their_score| {
                let relative = ((my_score - their_score) * 0.5 + 0.5).clamp(0.0, 1.0);
                let coordination = (1.0 - (my_score - current_score).abs()).clamp(0.0, 1.0);
                relative * 0.7 + coordination * 0.3
            }).collect()
        }).collect()
    } 
}
