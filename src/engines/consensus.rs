use crate::types::artifact::Artifact;
use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct RefinementRound {
    pub round_index: u32,
    pub agent_id: String,
    pub critique: String,
    pub proposed_resolution: String,
    pub accepted: bool,
    pub source_turn: Option<Turn>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    Voting,
    WeightedAverage,
    ExpertDeference,
    Mediation,
}

// Formal proofs for NashSolver: verus/consensus.rs
// Proved: psne_selection_is_deterministic, search_terminates, optimal_is_psne_if_exists.
pub struct NashSolver;

impl NashSolver {
    pub fn find_nash_equilibrium(matrix: &[Vec<Vec<f64>>]) -> Option<usize> {
        let n = matrix.len();
        if n == 0 {
            return None;
        }
        let strategies = matrix[0].len();
        let mut psne_indices = Vec::new();
        for s in 0..strategies {
            let mut is_nash = true;
            for player in matrix.iter().take(n) {
                let current_payoff = player[s][s];
                if (0..strategies)
                    .any(|alt_s| alt_s != s && player[alt_s][s] > current_payoff + 1e-5)
                {
                    is_nash = false;
                    break;
                }
            }
            if is_nash {
                psne_indices.push(s);
            }
        }
        psne_indices.into_iter().max_by(|&a, &b| {
            let wa: f64 = matrix.iter().map(|m| m[a][a]).sum();
            let wb: f64 = matrix.iter().map(|m| m[b][b]).sum();
            wa.total_cmp(&wb)
        })
    }

    pub fn resolve_optimal_proposal(
        proposals: &[(&str, &Artifact, TurnOutcome)],
        current: &Artifact,
    ) -> usize {
        if proposals.is_empty() {
            return 0;
        }
        let matrix = PayoffCalculator::compute_payoff_matrix(proposals, current);
        let n = matrix.len();
        if n == 0 {
            return 0;
        }

        let mut n_matrix = vec![vec![vec![0.0; proposals.len()]; proposals.len()]; proposals.len()];
        for (i, row) in matrix.iter().enumerate().take(n) {
            for s in 0..proposals.len() {
                for entry in n_matrix[i][s].iter_mut().take(proposals.len()) {
                    *entry = row[s];
                }
            }
        }

        Self::find_nash_equilibrium(&n_matrix).unwrap_or_else(|| {
            tracing::warn!("no Nash equilibrium found; falling back to argmax");
            (0..proposals.len())
                .max_by(|&a, &b| {
                    let wa: f64 = matrix.iter().map(|row| row[a]).sum();
                    let wb: f64 = matrix.iter().map(|row| row[b]).sum();
                    wa.total_cmp(&wb)
                })
                .unwrap_or(0)
        })
    }

    pub fn run_refinement_rounds(
        &self,
        proposals: &[(&str, &str)],
        max_rounds: usize,
    ) -> Vec<RefinementRound> {
        if proposals.is_empty() || max_rounds == 0 {
            return vec![];
        }
        let mut rounds = Vec::new();
        for round_idx in 0..max_rounds {
            let mut all_same = true;
            let first = proposals[0].1;
            for &(agent_id, proposed) in proposals {
                if proposed != first {
                    all_same = false;
                }
                rounds.push(RefinementRound {
                    round_index: round_idx as u32,
                    agent_id: agent_id.to_string(),
                    critique: String::new(),
                    proposed_resolution: proposed.to_string(),
                    accepted: true,
                    source_turn: None,
                });
            }
            if all_same {
                break;
            }
        }
        rounds
    }

    pub fn resolve(proposals: &[(&str, f64, &str)], strategy: ResolutionStrategy) -> String {
        if proposals.is_empty() {
            return String::new();
        }
        match strategy {
            ResolutionStrategy::Voting => {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for &(_, _, text) in proposals {
                    *counts.entry(text).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(t, _)| t.to_string())
                    .unwrap_or_default()
            }
            ResolutionStrategy::WeightedAverage | ResolutionStrategy::ExpertDeference => proposals
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|p| p.2.to_string())
                .unwrap_or_default(),
            ResolutionStrategy::Mediation => {
                let mut word_counts: HashMap<&str, usize> = HashMap::new();
                for &(_, _, text) in proposals {
                    for word in text.split_whitespace() {
                        *word_counts.entry(word).or_insert(0) += 1;
                    }
                }
                let threshold = proposals.len() / 2;
                let mut common: Vec<&str> = word_counts
                    .into_iter()
                    .filter(|&(_, c)| c > threshold)
                    .map(|(w, _)| w)
                    .collect();
                common.sort();
                common.join(" ")
            }
        }
    }

    /// Returns ALL pure-strategy Nash equilibrium indices over proposals.
    /// Used to detect when multiple proposals tie at equilibrium so they can be synthesized.
    pub fn find_all_equilibria(
        proposals: &[(&str, &Artifact, TurnOutcome)],
        current: &Artifact,
    ) -> Vec<usize> {
        let matrix = PayoffCalculator::compute_payoff_matrix(proposals, current);
        let n = proposals.len();
        if n == 0 {
            return vec![];
        }

        let mut n_matrix = vec![vec![vec![0.0; n]; n]; n];
        for (i, row) in matrix.iter().enumerate().take(n) {
            for s in 0..n {
                for entry in n_matrix[i][s].iter_mut().take(n) {
                    *entry = row[s];
                }
            }
        }

        let mut equilibria = Vec::new();
        for s in 0..n {
            let mut is_nash = true;
            for player in n_matrix.iter().take(n) {
                let current_payoff = player[s][s];
                if (0..n).any(|alt_s| alt_s != s && player[alt_s][s] > current_payoff + 1e-5) {
                    is_nash = false;
                    break;
                }
            }
            if is_nash {
                equilibria.push(s);
            }
        }
        if equilibria.is_empty() {
            let fallback = (0..n)
                .max_by(|&a, &b| {
                    let wa: f64 = matrix.iter().map(|row| row[a]).sum();
                    let wb: f64 = matrix.iter().map(|row| row[b]).sum();
                    wa.total_cmp(&wb)
                })
                .unwrap_or(0);
            equilibria.push(fallback);
        }
        equilibria
    }

    /// Resolves proposals to a content string.
    /// When multiple Nash equilibria survive, synthesizes them rather than picking one winner.
    pub fn resolve_with_synthesis(
        proposals: &[(&str, &Artifact, TurnOutcome)],
        current: &Artifact,
    ) -> String {
        if proposals.is_empty() {
            return String::new();
        }
        let equilibria = Self::find_all_equilibria(proposals, current);
        if equilibria.len() == 1 {
            return proposals[equilibria[0]].1.content.clone();
        }
        let surviving: Vec<(String, String)> = equilibria
            .iter()
            .map(|&idx| {
                (
                    proposals[idx].0.to_string(),
                    proposals[idx].1.content.clone(),
                )
            })
            .collect();
        crate::engines::reasoning::SynthesisEngine::synthesize_proposals(&surviving)
    }

    pub fn solve_2x2_pure(matrix: &[[(f64, f64); 2]; 2]) -> Vec<(usize, usize)> {
        let mut equilibria = vec![];
        for r in 0..2 {
            for c in 0..2 {
                if matrix[r][c].0 >= matrix[1 - r][c].0 && matrix[r][c].1 >= matrix[r][1 - c].1 {
                    equilibria.push((r, c));
                }
            }
        }
        equilibria
    }

    /// Returns a normalized Nash score per agent in `proposals`, in the same order.
    /// Score is derived from each agent's column-sum payoff relative to the total.
    pub fn compute_nash_scores(proposals: &[(&str, &Artifact, TurnOutcome)]) -> Vec<(String, f64)> {
        if proposals.is_empty() {
            return vec![];
        }
        let matrix = PayoffCalculator::compute_payoff_matrix(proposals, proposals[0].1);
        let n = proposals.len();
        let col_sums: Vec<f64> = (0..n)
            .map(|col| matrix.iter().map(|row| row[col]).sum())
            .collect();
        let total: f64 = col_sums.iter().sum();
        proposals
            .iter()
            .zip(col_sums.iter())
            .map(|((agent_id, _, _), &col_sum)| {
                let score = if total > 0.0 {
                    col_sum / total
                } else {
                    1.0 / n as f64
                };
                (agent_id.to_string(), score.clamp(0.0, 1.0))
            })
            .collect()
    }
}

pub struct KalmanConvergence {
    pub p_c: f64,
    pub variance: f64,
    pub innovation: f64,
    process_noise: f64,
    prev_innovation: f64,
    oscillation_count: u32,
    trend_count: u32,
}

impl KalmanConvergence {
    const BASE_PROCESS_NOISE: f64 = 0.002;
    pub fn new(initial_p: f64) -> Self {
        Self {
            p_c: initial_p.clamp(0.0, 1.0),
            variance: 0.2,
            innovation: 0.0,
            process_noise: Self::BASE_PROCESS_NOISE,
            prev_innovation: 0.0,
            oscillation_count: 0,
            trend_count: 0,
        }
    }
    pub fn update(&mut self, measurement: f64) -> f64 {
        self.update_adaptive(measurement, 1.0)
    }

    pub fn is_converged(&self, threshold: f64) -> bool {
        // Require low variance AND no recent oscillation
        self.variance < threshold && self.oscillation_count < 2
    }

    pub fn confidence_interval(&self) -> (f64, f64) {
        let margin = 1.96 * self.variance.sqrt();
        let lo = (self.p_c - margin).clamp(0.0, 1.0);
        let hi = (self.p_c + margin).clamp(0.0, 1.0);
        (lo, hi)
    }

    pub fn update_adaptive(&mut self, measurement: f64, certainty: f64) -> f64 {
        // Detect oscillation: innovation sign flips indicate thrashing
        let new_innovation = measurement - self.p_c;
        if self.prev_innovation != 0.0 && new_innovation.signum() != self.prev_innovation.signum() {
            self.oscillation_count = self.oscillation_count.saturating_add(1);
            self.trend_count = 0;
            // Increase process noise when oscillating — system is less predictable
            self.process_noise = (self.process_noise * 1.5).min(0.05);
        } else {
            self.trend_count = self.trend_count.saturating_add(1);
            self.oscillation_count = self.oscillation_count.saturating_sub(1);
            // Decrease toward base when trending steadily
            if self.trend_count > 3 {
                self.process_noise = (self.process_noise * 0.8).max(Self::BASE_PROCESS_NOISE);
            }
        }
        self.prev_innovation = new_innovation;

        self.variance += self.process_noise;
        let r_adaptive = 0.1 / (certainty.max(0.01));
        let gain = self.variance / (self.variance + r_adaptive);
        self.innovation = new_innovation;
        self.p_c += gain * self.innovation;
        self.variance *= 1.0 - gain;
        self.p_c.clamp(0.0, 1.0)
    }
}

pub struct CertaintyAnalyzer;
impl CertaintyAnalyzer {
    pub fn compute(content: &str, volatility: f64) -> f64 {
        let lower = content.to_lowercase();
        let word_count = content.split_whitespace().count().max(1) as f64;

        // Signal 1: Hedging language (reduces certainty)
        const HEDGES: &[&str] = &[
            "maybe",
            "perhaps",
            "might",
            "could be",
            "not sure",
            "i think",
            "possibly",
            "unclear",
            "hard to say",
            "it depends",
            "arguably",
            "i'm not certain",
            "one possibility",
        ];
        let hedge_count = HEDGES.iter().filter(|h| lower.contains(*h)).count() as f64;
        let hedge_penalty = (hedge_count * 0.12).min(0.4);

        // Signal 2: Assertive language (increases certainty)
        const ASSERTIVE: &[&str] = &[
            "verified",
            "confirmed",
            "optimal",
            "correct",
            "proven",
            "tests pass",
            "all tests",
            "successfully",
            "no issues",
            "the solution is",
            "this fixes",
            "this resolves",
            "implemented",
            "works",
            "complete",
            "done",
        ];
        let assert_count = ASSERTIVE.iter().filter(|a| lower.contains(*a)).count() as f64;
        let assert_boost = (assert_count * 0.11).min(0.35);

        // Signal 3: Code presence (concrete output = higher certainty)
        let code_blocks = content.matches("```").count() / 2;
        let code_boost = (code_blocks as f64 * 0.05).min(0.15);

        // Signal 4: Quantitative claims (numbers, measurements, percentages)
        let numeric_density =
            content.chars().filter(|c| c.is_ascii_digit()).count() as f64 / word_count;
        let quant_boost = (numeric_density * 0.3).min(0.1);

        // Signal 5: Response length (very short with no assertions = low effort)
        let length_signal = if word_count < 20.0 && assert_count == 0.0 {
            -0.1
        } else if word_count > 500.0 {
            0.05
        } else {
            0.0
        };

        // Signal 6: Structural markers (organized thought = higher certainty)
        let has_steps =
            lower.contains("step 1") || lower.contains("1.") || lower.contains("first,");
        let has_reasoning =
            lower.contains("because") || lower.contains("therefore") || lower.contains("since");
        let structure_boost =
            if has_steps { 0.05 } else { 0.0 } + if has_reasoning { 0.05 } else { 0.0 };

        // Signal 7: Self-contradiction (agent contradicts itself within response)
        let contradiction_penalty = {
            let sentences: Vec<&str> = content
                .split(['.', '!', '\n'])
                .map(str::trim)
                .filter(|s| s.len() > 10)
                .collect();
            let mut contradictions = 0u32;
            for s in &sentences {
                let sl = s.to_lowercase();
                if sl.contains("however")
                    || sl.contains("but actually")
                    || sl.contains("on second thought")
                    || sl.contains("wait,")
                    || sl.contains("correction:")
                    || sl.contains("i was wrong")
                {
                    contradictions += 1;
                }
            }
            (contradictions as f64 * 0.08).min(0.25)
        };

        // Signal 8: Evidence grounding (references to specific artifacts, lines, functions)
        let grounding_boost = {
            let refs = content.matches("line ").count()
                + content.matches("fn ").count()
                + content.matches("function ").count()
                + content.matches("class ").count()
                + content.matches("file ").count();
            (refs as f64 * 0.02).min(0.1)
        };

        let base = 0.50;
        let raw = base
            + assert_boost
            + code_boost
            + quant_boost
            + length_signal
            + structure_boost
            + grounding_boost
            - hedge_penalty
            - contradiction_penalty;
        (raw - volatility * 0.1).clamp(0.05, 0.98)
    }
}

pub struct PayoffCalculator;
impl PayoffCalculator {
    pub fn evaluate(artifact: &Artifact) -> f64 {
        let base = 0.5;
        let proof_bonus = (artifact.proof_attachments.len() as f64 * 0.1).min(0.3);
        let complexity_penalty = if artifact.metrics.line_count > 0 {
            let ratio =
                artifact.metrics.cyclomatic_complexity as f64 / artifact.metrics.line_count as f64;
            (ratio * 0.5).min(0.3)
        } else {
            0.0
        };

        // --- Sovereign-Tier: Multi-Modal Visual Fidelity ---
        let visual_bonus = (artifact.metrics.visual_fidelity * 0.2).min(0.2);

        (base + proof_bonus + artifact.metrics.health_score * 0.3 + visual_bonus
            - complexity_penalty)
            .clamp(0.0, 1.0)
    }
    pub fn evaluate_with_outcome(artifact: &Artifact, outcome: &TurnOutcome) -> f64 {
        let correctness = match outcome {
            TurnOutcome::TestsPassed => 1.0,
            TurnOutcome::Compiled => 0.8,
            _ => 0.5,
        };
        (correctness * 0.7 + (artifact.metrics.health_score * 0.3)).clamp(0.0, 1.0)
    }
    pub fn compute_payoff_matrix(
        proposals: &[(&str, &Artifact, TurnOutcome)],
        current: &Artifact,
    ) -> Vec<Vec<f64>> {
        let current_score = Self::evaluate_with_outcome(current, &TurnOutcome::Unknown);
        let scores: Vec<f64> = proposals
            .iter()
            .map(|(_, a, o)| Self::evaluate_with_outcome(a, o))
            .collect();
        scores
            .iter()
            .map(|&my_score| {
                scores
                    .iter()
                    .map(|&their_score| {
                        let relative = ((my_score - their_score) * 0.5 + 0.5).clamp(0.0, 1.0);
                        let coordination = (1.0 - (my_score - current_score).abs()).clamp(0.0, 1.0);
                        relative * 0.7 + coordination * 0.3
                    })
                    .collect()
            })
            .collect()
    }
    pub fn best_response(mine: &[f64], theirs: &[f64]) -> usize {
        mine.iter()
            .zip(theirs.iter())
            .enumerate()
            .max_by(|(_, (a_m, a_t)), (_, (b_m, b_t))| {
                let sa = *a_m + *a_t;
                let sb = *b_m + *b_t;
                sa.total_cmp(&sb)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Detects stall patterns from per-turn proposal content hashes.
/// Tracks the last 6 turns of (agent_id → content_hash) snapshots.
pub struct StallDetector {
    proposal_history: VecDeque<HashMap<String, String>>,
    /// Consecutive-disagreement counters: (agent_a, agent_b) → run length.
    disagreement_runs: HashMap<(String, String), u32>,
    /// Entropy values per turn for non-decreasing entropy detection.
    entropy_history: VecDeque<f64>,
}

impl StallDetector {
    pub fn new() -> Self {
        Self {
            proposal_history: VecDeque::with_capacity(6),
            disagreement_runs: HashMap::new(),
            entropy_history: VecDeque::with_capacity(5),
        }
    }

    /// Record proposals for this turn (agent_id → content_hash) and an entropy value.
    /// Returns stall_risk in [0.0, 1.0].
    pub fn push_turn(&mut self, proposals: HashMap<String, String>, turn_entropy: f64) -> f64 {
        if self.proposal_history.len() >= 6 {
            self.proposal_history.pop_front();
        }

        if self.entropy_history.len() >= 5 {
            self.entropy_history.pop_front();
        }
        self.entropy_history.push_back(turn_entropy);

        // Update disagreement runs: compare each pair in this turn vs previous turn.
        if let Some(prev) = self.proposal_history.back() {
            let agents: Vec<&String> = proposals.keys().collect();
            for i in 0..agents.len() {
                for j in (i + 1)..agents.len() {
                    let a = agents[i];
                    let b = agents[j];
                    let hash_a = proposals.get(a).map(|s| s.as_str()).unwrap_or("");
                    let hash_b = proposals.get(b).map(|s| s.as_str()).unwrap_or("");
                    let prev_a = prev.get(a).map(|s| s.as_str()).unwrap_or("");
                    let prev_b = prev.get(b).map(|s| s.as_str()).unwrap_or("");
                    let key = if a < b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    if hash_a != hash_b && prev_a != prev_b {
                        *self.disagreement_runs.entry(key).or_insert(0) += 1;
                    } else {
                        self.disagreement_runs.insert(key, 0);
                    }
                }
            }
        }

        let stall_risk = self.compute_stall_risk(&proposals);
        self.proposal_history.push_back(proposals);
        stall_risk
    }

    fn compute_stall_risk(&self, current: &HashMap<String, String>) -> f64 {
        let mut risk = 0.0_f64;

        // Oscillation: agent produces same hash at turn N and N-2.
        if self.proposal_history.len() >= 3 {
            let n_minus_2 = &self.proposal_history[self.proposal_history.len() - 3];
            for (agent, hash) in current {
                if n_minus_2.get(agent) == Some(hash) {
                    risk += 0.4;
                    break;
                }
            }
        }

        // Non-decreasing entropy over last 5 turns.
        if self.entropy_history.len() >= 5 {
            let non_decreasing = self
                .entropy_history
                .iter()
                .zip(self.entropy_history.iter().skip(1))
                .all(|(a, b)| b >= a);
            if non_decreasing {
                risk += 0.3;
            }
        }

        // Repeated pair disagreement in 3+ consecutive turns.
        if self.disagreement_runs.values().any(|&run| run >= 3) {
            risk += 0.3;
        }

        risk.min(1.0)
    }

    pub fn stall_risk(&self) -> f64 {
        if let Some(current) = self.proposal_history.back() {
            self.compute_stall_risk(current)
        } else {
            0.0
        }
    }
}

impl Default for StallDetector {
    fn default() -> Self {
        Self::new()
    }
}

pub struct InfluenceWeightManager;
impl InfluenceWeightManager {
    pub fn calculate_weights(sigma: &ConversationState) -> std::collections::BTreeMap<String, f64> {
        if sigma.agent_weights.is_empty() && !sigma.turns.is_empty() {
            return Self::compute_agent_weights_from_turns(&sigma.turns, 0.9);
        }
        sigma.agent_weights.clone()
    }
    pub fn calculate_weights_for_category(
        sigma: &ConversationState,
        category: TaskCategory,
        recency: f64,
    ) -> std::collections::BTreeMap<String, f64> {
        let category_turns: Vec<&Turn> = sigma
            .turns
            .iter()
            .filter(|t| t.task_category.as_ref() == Some(&category))
            .collect();

        if category_turns.is_empty() {
            // Fallback: compute global weights from all turns, dampened.
            let global = Self::compute_agent_weights_from_turns(&sigma.turns, recency);
            return global;
        }

        Self::compute_agent_weights_from_turns(
            &category_turns.into_iter().cloned().collect::<Vec<_>>(),
            recency,
        )
    }
    fn compute_agent_weights_from_turns(
        turns: &[Turn],
        recency: f64,
    ) -> std::collections::BTreeMap<String, f64> {
        use std::collections::BTreeMap;
        let mut scores: HashMap<String, (f64, f64)> = HashMap::new();
        let n = turns.len();
        for (i, t) in turns.iter().enumerate() {
            let outcome_score = match t.outcome {
                TurnOutcome::TestsPassed => 1.0,
                TurnOutcome::Compiled => 0.5,
                TurnOutcome::Unknown => 0.3,
                TurnOutcome::AdvancedConvergence => 0.9,
                TurnOutcome::Stalled => 0.2,
                TurnOutcome::Rejected => 0.1,
                TurnOutcome::RolledBack => 0.0,
                TurnOutcome::VerificationFailed => 0.0,
            };
            let certainty = t.certainty.unwrap_or(0.5);
            let surprise_penalty = 1.0 - t.surprise_signal.unwrap_or(0.0) * 0.5;
            // Exponential recency decay: recent turns weighted more heavily
            let age = (n - 1 - i) as f64;
            let decay = recency.powf(age);
            let entry = scores.entry(t.model_id.clone()).or_insert((0.0, 0.0));
            entry.0 += outcome_score * certainty * surprise_penalty * decay;
            entry.1 += decay;
        }
        let mut weights = BTreeMap::new();
        for (agent, (weighted_sum, weight_total)) in &scores {
            let w = if *weight_total > 0.0 {
                weighted_sum / weight_total
            } else {
                0.3
            };
            weights.insert(agent.clone(), w);
        }
        weights
    }
    pub fn calculate_weights_with_recency(
        sigma: &ConversationState,
        recency: f64,
    ) -> std::collections::BTreeMap<String, f64> {
        if sigma.agent_weights.is_empty() && !sigma.turns.is_empty() {
            return Self::compute_agent_weights_from_turns(&sigma.turns, recency);
        }
        sigma.agent_weights.clone()
    }
    pub fn rank(weights: &std::collections::BTreeMap<String, f64>) -> Vec<(String, f64)> {
        let mut sorted: Vec<(String, f64)> = weights.iter().map(|(k, v)| (k.clone(), *v)).collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        sorted
    }
    pub fn vote_tally(proposals: &[RefinementRound]) -> HashMap<String, u32> {
        let mut votes = HashMap::new();
        for round in proposals {
            if round.accepted {
                *votes.entry(round.agent_id.clone()).or_insert(0) += 1;
            }
        }
        votes
    }
}
