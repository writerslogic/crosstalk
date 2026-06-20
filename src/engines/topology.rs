//! Dynamic Debate Topology Manager
//!
//! Controls *how* agents interact — not what they say, but the structural
//! format of the debate.  The topology is selected and shifted automatically
//! based on progress signals drawn from `ConversationState` and turn
//! outcomes, with a history-weighted preference for topologies that have
//! produced good outcomes in this session.

use crate::types::compute::BudgetMode;
use crate::types::conversation::{ConversationState, TaskCategory, TurnOutcome};
use crate::types::intelligence::RunningAverage;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

const MIN_TURNS_BETWEEN_SHIFTS: u32 = 2;

/// Deadlock counter threshold that triggers the first escalation step
/// (e.g., RoundRobin → Critique).
const DEADLOCK_ESCALATION_SOFT: u32 = 5;

/// Deadlock counter threshold that triggers the hard escalation to Mediated.
const DEADLOCK_ESCALATION_HARD: u32 = 8;

/// Minimum number of quality observations required before a topology is
/// eligible for historical-best selection.
const MIN_TOPOLOGY_OBSERVATIONS: u32 = 3;

/// Minimum window size for quality trend detection.
const QUALITY_TREND_MIN_WINDOW: usize = 4;

/// Minimum quality improvement (late half vs early half) to consider the
/// current topology to be making positive progress.
const QUALITY_TREND_THRESHOLD: f64 = 0.05;

// =====================================================================
// TOPOLOGY VARIANTS
// =====================================================================

/// The structural shape of multi-agent interaction for a given phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DebateTopology {
    /// Agents take turns sequentially in round-robin order.
    RoundRobin,
    /// Agents are paired 1-vs-1; pairs rotate each round.
    Adversarial,
    /// All agents respond in parallel; responses are merged via consensus.
    Ensemble,
    /// Agents branch into N distinct hypothetical explorations; low-scoring
    /// branches are pruned before the next round.
    TreeOfThoughts,
    /// A dedicated mediator agent synthesises after every exchange.
    Mediated,
    /// One agent proposes; all others critique; proposer revises.
    Critique,
}

impl DebateTopology {
    /// Human-readable label used in prompt modifiers.
    fn label(self) -> &'static str {
        match self {
            DebateTopology::RoundRobin => "Round-Robin",
            DebateTopology::Adversarial => "Adversarial",
            DebateTopology::Ensemble => "Ensemble",
            DebateTopology::TreeOfThoughts => "Tree-of-Thoughts",
            DebateTopology::Mediated => "Mediated",
            DebateTopology::Critique => "Critique",
        }
    }

    /// Whether this topology requires at least `n` agents to be meaningful.
    fn minimum_agents(self) -> usize {
        match self {
            DebateTopology::RoundRobin => 1,
            DebateTopology::Adversarial => 2,
            DebateTopology::Ensemble => 3,
            DebateTopology::TreeOfThoughts => 1,
            DebateTopology::Mediated => 2,
            DebateTopology::Critique => 2,
        }
    }
}

// =====================================================================
// AGENT GROUPING
// =====================================================================

/// How agents are partitioned for the next turn under a given topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentGrouping {
    /// All agents participate together.
    All,
    /// Explicit 1-vs-1 pairs (indices into the active agent list).
    Pairs(Vec<(usize, usize)>),
    /// A single agent acts (index into the active agent list).
    Single(usize),
    /// Multiple branches, each containing the indices of contributing agents.
    Branches(Vec<Vec<usize>>),
}

// =====================================================================
// TOPOLOGY DIRECTIVE
// =====================================================================

/// Returned by `TopologyManager::shift_to`; tells the orchestrator exactly
/// how to configure the next turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyDirective {
    /// The topology now in effect.
    pub topology: DebateTopology,
    /// How to partition the active agent list.
    pub agent_grouping: AgentGrouping,
    /// Optional text injected into the prompt to enforce the topology
    /// (e.g., "You are the proposer — state your position clearly").
    pub prompt_modifier: Option<String>,
    /// Auto-revert to `RoundRobin` if no quality improvement is seen
    /// within this many turns.
    pub max_turns_in_topology: u32,
}

// =====================================================================
// TOPOLOGY REASON
// =====================================================================

/// Why the topology was shifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyReason {
    /// Too many consecutive turns without progress.
    Deadlock,
    /// Average quality scores declined.
    QualityDrop,
    /// The number of active agents changed.
    AgentCountChange,
    /// Caller explicitly requested a topology.
    ManualOverride,
    /// Periodic rotation to keep agents from settling into local optima.
    ScheduledRotation,
}

// =====================================================================
// TOPOLOGY MANAGER
// =====================================================================

/// Manages topology selection, shift execution, and outcome tracking.
pub struct TopologyManager {
    /// The topology currently in effect.
    pub current: DebateTopology,
    /// Full shift history: `(turn_index, topology, reason)`.
    pub history: Vec<(u32, DebateTopology, TopologyReason)>,
    /// Consecutive turns without meaningful progress.
    pub deadlock_counter: u32,
    /// Per-(topology, category) rolling quality averages used for UCB1 selection.
    pub topology_scores: FxHashMap<(DebateTopology, TaskCategory), RunningAverage>,
    /// Per-(topology, category) rolling cost averages in USD.
    topology_cost_scores: FxHashMap<(DebateTopology, TaskCategory), RunningAverage>,
    /// Per-(topology, category) rolling latency averages in milliseconds.
    topology_latency_scores: FxHashMap<(DebateTopology, TaskCategory), RunningAverage>,
    /// Global running average of cost across all topologies (used for ratio normalization).
    global_mean_cost: RunningAverage,
    /// Global running average of latency in ms across all topologies (used for ratio normalization).
    global_mean_latency: RunningAverage,
    /// Total turns recorded across all topologies and categories (UCB1 denominator).
    total_topology_turns: u32,
    /// Number of active agents (updated by caller when the pool changes).
    agent_count: usize,
    /// Recent per-turn quality scores for trend detection (capped at 8).
    recent_quality: std::collections::VecDeque<f64>,
    /// Recent turn outcomes for pattern detection (capped at 8).
    recent_outcomes: std::collections::VecDeque<TurnOutcome>,
    /// Turn index at which the last shift occurred; used to enforce dwell time.
    last_shift_turn: u32,
}

impl TopologyManager {
    /// Create a new manager starting with `RoundRobin`.
    pub fn new(agent_count: usize) -> Self {
        Self {
            current: DebateTopology::RoundRobin,
            history: Vec::new(),
            deadlock_counter: 0,
            topology_scores: FxHashMap::default(),
            topology_cost_scores: FxHashMap::default(),
            topology_latency_scores: FxHashMap::default(),
            global_mean_cost: RunningAverage::default(),
            global_mean_latency: RunningAverage::default(),
            total_topology_turns: 0,
            agent_count,
            recent_quality: std::collections::VecDeque::with_capacity(8),
            recent_outcomes: std::collections::VecDeque::with_capacity(8),
            last_shift_turn: 0,
        }
    }

    // ── Agent management ──────────────────────────────────────────────

    /// Update the number of active agents.  If the count changed, the
    /// topology may need to be re-evaluated.
    pub fn set_agent_count(&mut self, count: usize, turn_idx: u32) -> Option<TopologyDirective> {
        if count == self.agent_count {
            return None;
        }
        self.agent_count = count;
        // If the current topology can no longer be satisfied, shift.
        if count < self.current.minimum_agents() {
            let next = self.fallback_for_count(count);
            Some(self.shift_to(next, turn_idx, TopologyReason::AgentCountChange))
        } else {
            None
        }
    }

    // ── Turn recording ────────────────────────────────────────────────

    /// Record the outcome of a completed turn, updating deadlock tracking
    /// and topology quality scores.
    pub fn record_turn_outcome(
        &mut self,
        outcome: TurnOutcome,
        quality_score: f64,
        task_category: TaskCategory,
        cost_usd: f64,
        latency_ms: u64,
    ) {
        // Deadlock counter
        let made_progress = matches!(
            outcome,
            TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
        );
        if made_progress {
            self.deadlock_counter = 0;
        } else {
            self.deadlock_counter += 1;
        }

        // Rolling quality keyed by (topology, category) for UCB1
        self.topology_scores
            .entry((self.current, task_category))
            .or_default()
            .update(quality_score);

        // Rolling cost and latency for efficiency scoring
        self.topology_cost_scores
            .entry((self.current, task_category))
            .or_default()
            .update(cost_usd);
        self.topology_latency_scores
            .entry((self.current, task_category))
            .or_default()
            .update(latency_ms as f64);
        self.global_mean_cost.update(cost_usd);
        self.global_mean_latency.update(latency_ms as f64);

        self.total_topology_turns += 1;

        // Recent windows
        self.recent_quality.push_back(quality_score);
        if self.recent_quality.len() > 8 {
            self.recent_quality.pop_front();
        }
        self.recent_outcomes.push_back(outcome);
        if self.recent_outcomes.len() > 8 {
            self.recent_outcomes.pop_front();
        }
    }

    // ── Routing and selection ─────────────────────────────────────────

    /// Recommend the best topology given the current debate state.
    ///
    /// This does NOT mutate state — it only advises.  Call `shift_to` to
    /// commit the recommendation.
    pub fn recommend_topology(
        &self,
        sigma: &ConversationState,
        task_category: TaskCategory,
    ) -> DebateTopology {
        let n_agents = self.agent_count;

        // --- Hard deadlock escalation ladder ---
        if self.deadlock_counter >= DEADLOCK_ESCALATION_HARD {
            return DebateTopology::Mediated;
        }
        if self.deadlock_counter >= DEADLOCK_ESCALATION_SOFT {
            return match self.current {
                DebateTopology::RoundRobin => DebateTopology::Critique,
                DebateTopology::Critique => DebateTopology::TreeOfThoughts,
                _ => DebateTopology::Mediated,
            };
        }

        // --- Three consecutive Stalled turns → TreeOfThoughts ---
        if self.recent_outcomes.len() >= 3 {
            let last3: Vec<TurnOutcome> =
                self.recent_outcomes.iter().rev().take(3).copied().collect();
            if last3.iter().all(|o| *o == TurnOutcome::Stalled) {
                return DebateTopology::TreeOfThoughts;
            }
        }

        // --- Too few agents for complex topologies ---
        if n_agents < 3 {
            return if n_agents < 2 {
                DebateTopology::TreeOfThoughts
            } else {
                DebateTopology::Adversarial
            };
        }

        // --- UCB1: primary signal when sufficient data exists ---
        const ALL_TOPOLOGIES: &[DebateTopology] = &[
            DebateTopology::RoundRobin,
            DebateTopology::Adversarial,
            DebateTopology::Ensemble,
            DebateTopology::TreeOfThoughts,
            DebateTopology::Mediated,
            DebateTopology::Critique,
        ];
        let sufficient_data = ALL_TOPOLOGIES.iter().all(|t| {
            self.topology_scores
                .get(&(*t, task_category))
                .map(|a| a.count >= 5)
                .unwrap_or(false)
        });
        if sufficient_data {
            let bm = sigma.budget.mode();
            if let Some(ucb_winner) = ALL_TOPOLOGIES
                .iter()
                .filter(|t| self.agent_count >= t.minimum_agents())
                .max_by(|a, b| {
                    self.ucb1_score(**a, task_category, bm)
                        .partial_cmp(&self.ucb1_score(**b, task_category, bm))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
            {
                let current_ucb = self.ucb1_score(self.current, task_category, bm);
                let winner_ucb = self.ucb1_score(ucb_winner, task_category, bm);
                if ucb_winner != self.current && winner_ucb - current_ucb > 0.1 {
                    return ucb_winner;
                }
            }
        }

        // --- If quality is improving, stay the course ---
        if self.quality_is_improving() {
            return self.current;
        }

        // --- Quality drop: prefer historically best topology ---
        let best = self.best_historical_topology(task_category);

        if let Some(candidate) = best
            && candidate != self.current
        {
            return candidate;
        }

        // Fall back based on completion probability signal from sigma.
        if sigma.completion_probability < 0.3 {
            DebateTopology::Ensemble
        } else if sigma.completion_probability < 0.7 {
            DebateTopology::Critique
        } else {
            DebateTopology::RoundRobin
        }
    }

    // ── Shift execution ───────────────────────────────────────────────

    /// Execute a topology shift, recording history and returning a directive.
    pub fn shift_to(
        &mut self,
        new: DebateTopology,
        turn_idx: u32,
        reason: TopologyReason,
    ) -> TopologyDirective {
        self.history.push((turn_idx, new, reason));
        self.current = new;
        self.deadlock_counter = 0;
        self.last_shift_turn = turn_idx;

        self.build_directive(new)
    }

    /// Convenience: check whether a shift is warranted and execute it if so,
    /// returning the directive only when a shift actually happened.
    pub fn maybe_shift(
        &mut self,
        sigma: &ConversationState,
        turn_idx: u32,
        task_category: TaskCategory,
    ) -> Option<TopologyDirective> {
        if turn_idx.saturating_sub(self.last_shift_turn) < MIN_TURNS_BETWEEN_SHIFTS {
            return None;
        }
        let recommended = self.recommend_topology(sigma, task_category);
        if recommended == self.current {
            return None;
        }
        let reason = self.classify_reason(recommended);
        Some(self.shift_to(recommended, turn_idx, reason))
    }

    // ── Directive building ────────────────────────────────────────────

    /// Generate `n_branches` variant prompts for Tree-of-Thoughts exploration.
    ///
    /// Each branch receives a distinct epistemic directive so agents explore
    /// complementary corners of the solution space rather than converging
    /// prematurely.
    pub fn generate_thought_branches(base_prompt: &str, n_branches: usize) -> Vec<String> {
        let directives = [
            "Explore the OPTIMISTIC scenario where every assumption holds and \
             the best-case outcome is achievable. Focus on enabling conditions.",
            "Explore the PESSIMISTIC scenario where the most dangerous risks \
             materialise. Identify the critical failure mode and how to prevent it.",
            "Explore the UNCONVENTIONAL approach: discard received wisdom and \
             propose a lateral solution that most practitioners would not consider.",
            "Explore the MINIMAL change approach: achieve the goal with the \
             fewest possible modifications to the existing system.",
            "Explore the RADICAL redesign: assume the current implementation is \
             discarded entirely and reason from first principles.",
        ];

        (0..n_branches.min(directives.len()))
            .map(|i| {
                format!(
                    "<context>\n{}\n</context>\n\n[THOUGHT BRANCH {}]\n{}",
                    base_prompt,
                    i + 1,
                    directives[i]
                )
            })
            .collect()
    }

    /// Build a `TopologyDirective` for the given topology, populating
    /// reasonable defaults for grouping, prompt modifier, and TTL.
    pub fn current_directive(&self) -> TopologyDirective {
        self.build_directive(self.current)
    }

    fn build_directive(&self, topology: DebateTopology) -> TopologyDirective {
        let n = self.agent_count;

        const EPISTEMIC_SUFFIX: &str = "\n\nYou MUST end your response with an \
             epistemic state block:\n\
             [confidence] <0-100>%\n\
             [assumption] <your key assumption>\n\
             [evidence] <supporting evidence or citation>";

        let (agent_grouping, prompt_modifier, max_turns) = match topology {
            DebateTopology::RoundRobin => (
                AgentGrouping::All,
                Some(format!(
                    "[TOPOLOGY: {}] Agents will respond in sequential order. \
                     Each agent must build on or explicitly respond to the \
                     immediately preceding turn.{EPISTEMIC_SUFFIX}",
                    topology.label()
                )),
                12,
            ),

            DebateTopology::Adversarial => {
                let pairs = Self::build_pairs(n);
                (
                    AgentGrouping::Pairs(pairs),
                    Some(format!(
                        "[TOPOLOGY: {}] You are in a direct 1-vs-1 exchange. \
                         Identify the strongest counter-argument to your \
                         opponent's last turn before advancing your own position.\
                         {EPISTEMIC_SUFFIX}",
                        topology.label()
                    )),
                    8,
                )
            }

            DebateTopology::Ensemble => (
                AgentGrouping::All,
                Some(format!(
                    "[TOPOLOGY: {}] All agents respond independently and in \
                     parallel. Do not reference other agents' turns — produce \
                     your best standalone answer. Responses will be merged via \
                     consensus.{EPISTEMIC_SUFFIX}",
                    topology.label()
                )),
                6,
            ),

            DebateTopology::TreeOfThoughts => {
                let branches: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
                (
                    AgentGrouping::Branches(branches),
                    Some(format!(
                        "[TOPOLOGY: {}] You are exploring one branch of a \
                         hypothesis tree. Follow your branch directive exactly. \
                         Low-scoring branches will be pruned; commit fully to \
                         your assigned exploration direction.{EPISTEMIC_SUFFIX}",
                        topology.label()
                    )),
                    6,
                )
            }

            DebateTopology::Mediated => {
                // Agent 0 is the mediator by convention.
                (
                    AgentGrouping::All,
                    Some(format!(
                        "[TOPOLOGY: {}] Agent 0 is the MEDIATOR. After all \
                         participants submit their positions, the mediator will \
                         synthesise and produce a consensus summary. \
                         Participants: state your position concisely. \
                         Mediator: identify shared ground and remaining \
                         disagreements.{EPISTEMIC_SUFFIX}",
                        topology.label()
                    )),
                    10,
                )
            }

            DebateTopology::Critique => {
                // Agent 0 proposes; others critique.
                let proposer = AgentGrouping::Single(0);
                (
                    proposer,
                    Some(format!(
                        "[TOPOLOGY: {}] Agent 0 is the PROPOSER — state your \
                         solution clearly and completely. All other agents are \
                         CRITICS — identify specific flaws, gaps, or risks. \
                         The proposer will revise in the next turn based on \
                         critiques received.{EPISTEMIC_SUFFIX}",
                        topology.label()
                    )),
                    8,
                )
            }
        };

        TopologyDirective {
            topology,
            agent_grouping,
            prompt_modifier,
            max_turns_in_topology: max_turns,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────

    /// Build a round-robin pairing of `n` agents.
    /// With n=4 this gives (0,1), (2,3); with n=5: (0,1), (2,3), (4,0).
    fn build_pairs(n: usize) -> Vec<(usize, usize)> {
        if n < 2 {
            return vec![];
        }
        let mut pairs = Vec::with_capacity(n / 2 + 1);
        let mut i = 0;
        while i + 1 < n {
            pairs.push((i, i + 1));
            i += 2;
        }
        // If odd, pair the last agent with agent 0.
        if n % 2 == 1 {
            pairs.push((n - 1, 0));
        }
        pairs
    }

    /// True if the recent quality window shows a non-trivial upward trend.
    fn quality_is_improving(&self) -> bool {
        if self.recent_quality.len() < QUALITY_TREND_MIN_WINDOW {
            return false;
        }
        let half = self.recent_quality.len() / 2;
        let half = half.max(1);
        let early: f64 = self.recent_quality.iter().take(half).sum::<f64>() / half as f64;
        let late: f64 = self.recent_quality.iter().rev().take(half).sum::<f64>() / half as f64;
        late > early + QUALITY_TREND_THRESHOLD
    }

    /// Return the topology with the highest historical mean quality score,
    /// excluding topologies for which we have fewer than 3 observations or
    /// that require more agents than currently available.
    fn best_historical_topology(&self, task_category: TaskCategory) -> Option<DebateTopology> {
        self.topology_scores
            .iter()
            .filter(|((topo, cat), avg)| {
                *cat == task_category
                    && avg.count >= MIN_TOPOLOGY_OBSERVATIONS
                    && self.agent_count >= topo.minimum_agents()
            })
            .max_by(|(_, a), (_, b)| {
                a.mean
                    .partial_cmp(&b.mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|((topo, _), _)| *topo)
    }

    /// Raw efficiency ratio for a (topology, category) pair without the UCB1 exploration bonus.
    /// Returns `None` when no quality data exists.
    fn raw_efficiency(
        &self,
        topology: DebateTopology,
        task_category: TaskCategory,
        alpha: f64,
        beta: f64,
    ) -> Option<f64> {
        const EPSILON_COST: f64 = 1e-6;
        const EPSILON_LATENCY: f64 = 50.0;
        const EPSILON_FLOOR: f64 = 1e-4;

        let quality_avg = self.topology_scores.get(&(topology, task_category))?;
        if quality_avg.count == 0 {
            return None;
        }
        let mean_cost = self
            .topology_cost_scores
            .get(&(topology, task_category))
            .map(|a| a.mean)
            .unwrap_or(0.0);
        let mean_latency = self
            .topology_latency_scores
            .get(&(topology, task_category))
            .map(|a| a.mean)
            .unwrap_or(0.0);
        let global_cost = self.global_mean_cost.mean.max(EPSILON_COST);
        let global_latency = self.global_mean_latency.mean.max(EPSILON_LATENCY);
        let norm_cost = (mean_cost + EPSILON_COST) / global_cost;
        let norm_latency = (mean_latency + EPSILON_LATENCY) / global_latency;
        Some(quality_avg.mean / (alpha * norm_cost + beta * norm_latency + EPSILON_FLOOR))
    }

    /// Efficiency-weighted UCB1 score for a (topology, category) pair.
    ///
    /// Efficiency = quality / (α·norm_cost + β·norm_latency + ε_floor).
    /// Both cost and latency are ratio-normalized against their global running
    /// averages so the two dimensions stay dimensionless and comparably scaled.
    /// The UCB1 exploration bonus is scaled by the global mean raw efficiency
    /// so it stays proportional when the efficiency range is large.
    /// Returns `f64::MAX` (force explore) when no data exists for the pair.
    fn ucb1_score(
        &self,
        topology: DebateTopology,
        task_category: TaskCategory,
        budget_mode: BudgetMode,
    ) -> f64 {
        let (alpha, beta) = match budget_mode {
            BudgetMode::Normal => (0.3, 0.2),
            BudgetMode::CostReduction => (0.7, 0.3),
            BudgetMode::Emergency => (1.5, 0.5),
        };

        let quality_avg = match self.topology_scores.get(&(topology, task_category)) {
            None => return f64::MAX,
            Some(a) if a.count == 0 => return f64::MAX,
            Some(a) => a,
        };

        let efficiency = match self.raw_efficiency(topology, task_category, alpha, beta) {
            Some(e) => e,
            None => return f64::MAX,
        };

        let n = quality_avg.count as f64;
        if n == 0.0 {
            return f64::MAX;
        }
        let total = self.total_topology_turns.max(1) as f64;
        let exploration = (2.0 * total.ln() / n).sqrt();
        let global_mean_eff = self.global_mean_efficiency(alpha, beta);

        efficiency + global_mean_eff * exploration
    }

    /// Mean raw efficiency across all topology/category pairs that have observations.
    /// Used to scale the UCB1 exploration bonus. Falls back to 1.0 when no data exists.
    /// Does NOT call `ucb1_score` to avoid mutual recursion.
    fn global_mean_efficiency(&self, alpha: f64, beta: f64) -> f64 {
        let scores: Vec<f64> = self
            .topology_scores
            .keys()
            .filter_map(|key| self.raw_efficiency(key.0, key.1, alpha, beta))
            .collect();
        if scores.is_empty() {
            1.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }

    /// Select the best topology when the agent count is below 3.
    fn fallback_for_count(&self, count: usize) -> DebateTopology {
        match count {
            0 | 1 => DebateTopology::TreeOfThoughts,
            _ => DebateTopology::Adversarial,
        }
    }

    /// Serialize per-topology quality scores to JSON for cross-session persistence.
    pub fn export_scores_json(&self) -> String {
        let scores: Vec<((DebateTopology, TaskCategory), RunningAverage)> = self
            .topology_scores
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        serde_json::to_string(&scores).unwrap_or_default()
    }

    /// Merge topology scores from a prior session's JSON into the current map.
    ///
    /// Uses `entry().or_insert()` so current-session scores always win.
    pub fn import_scores_json(&mut self, json: &str) {
        if let Ok(scores) =
            serde_json::from_str::<Vec<((DebateTopology, TaskCategory), RunningAverage)>>(json)
        {
            for (key, avg) in scores {
                self.topology_scores.entry(key).or_insert(avg);
            }
        }
    }

    /// Classify the reason for an imminent shift given the recommended topology.
    fn classify_reason(&self, recommended: DebateTopology) -> TopologyReason {
        if self.deadlock_counter >= DEADLOCK_ESCALATION_SOFT {
            return TopologyReason::Deadlock;
        }
        if recommended == DebateTopology::TreeOfThoughts {
            // Could be quality drop or stall pattern; stall takes precedence.
            if self.recent_outcomes.len() >= 3
                && self
                    .recent_outcomes
                    .iter()
                    .rev()
                    .take(3)
                    .all(|o| *o == TurnOutcome::Stalled)
            {
                return TopologyReason::Deadlock;
            }
        }
        TopologyReason::QualityDrop
    }
}

impl Default for TopologyManager {
    fn default() -> Self {
        Self::new(2)
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::conversation::{ConversationState, TaskCategory};

    fn make_state() -> ConversationState {
        ConversationState::new("test-session")
    }

    #[test]
    fn deadlock_escalation_to_critique() {
        let mut mgr = TopologyManager::new(4);
        // Pump 5 non-progress outcomes
        for _ in 0..5 {
            mgr.record_turn_outcome(
                TurnOutcome::Stalled,
                0.2,
                TaskCategory::Research,
                0.01,
                1000,
            );
        }
        let sigma = make_state();
        assert_eq!(
            mgr.recommend_topology(&sigma, TaskCategory::Research),
            DebateTopology::Critique,
            "5 stalls from RoundRobin should recommend Critique"
        );
    }

    #[test]
    fn deadlock_escalation_to_mediated() {
        let mut mgr = TopologyManager::new(4);
        for _ in 0..8 {
            mgr.record_turn_outcome(
                TurnOutcome::Stalled,
                0.1,
                TaskCategory::Research,
                0.01,
                1000,
            );
        }
        let sigma = make_state();
        assert_eq!(
            mgr.recommend_topology(&sigma, TaskCategory::Research),
            DebateTopology::Mediated,
            "8 stalls should recommend Mediated"
        );
    }

    #[test]
    fn three_consecutive_stalls_trigger_tree_of_thoughts() {
        let mut mgr = TopologyManager::new(4);
        // Two non-stall turns first to avoid the deadlock counter firing
        mgr.record_turn_outcome(
            TurnOutcome::Compiled,
            0.8,
            TaskCategory::Research,
            0.01,
            1000,
        );
        mgr.record_turn_outcome(
            TurnOutcome::Compiled,
            0.8,
            TaskCategory::Research,
            0.01,
            1000,
        );
        mgr.record_turn_outcome(
            TurnOutcome::Stalled,
            0.2,
            TaskCategory::Research,
            0.01,
            1000,
        );
        mgr.record_turn_outcome(
            TurnOutcome::Stalled,
            0.2,
            TaskCategory::Research,
            0.01,
            1000,
        );
        mgr.record_turn_outcome(
            TurnOutcome::Stalled,
            0.2,
            TaskCategory::Research,
            0.01,
            1000,
        );
        let sigma = make_state();
        assert_eq!(
            mgr.recommend_topology(&sigma, TaskCategory::Research),
            DebateTopology::TreeOfThoughts
        );
    }

    #[test]
    fn small_agent_count_forces_adversarial() {
        let mgr = TopologyManager::new(2);
        let sigma = make_state();
        let rec = mgr.recommend_topology(&sigma, TaskCategory::Research);
        assert!(
            rec == DebateTopology::Adversarial || rec == DebateTopology::Critique,
            "2 agents should not recommend Ensemble"
        );
    }

    #[test]
    fn improving_quality_keeps_current_topology() {
        let mut mgr = TopologyManager::new(4);
        // Feed improving scores
        for q in [0.4f64, 0.5, 0.6, 0.7, 0.8] {
            mgr.record_turn_outcome(
                TurnOutcome::AdvancedConvergence,
                q,
                TaskCategory::Research,
                0.01,
                1000,
            );
        }
        let sigma = make_state();
        assert_eq!(
            mgr.recommend_topology(&sigma, TaskCategory::Research),
            DebateTopology::RoundRobin,
            "improving quality should keep current topology"
        );
    }

    #[test]
    fn shift_to_records_history_and_resets_counter() {
        let mut mgr = TopologyManager::new(4);
        mgr.deadlock_counter = 7;
        let directive = mgr.shift_to(DebateTopology::Ensemble, 10, TopologyReason::ManualOverride);
        assert_eq!(mgr.current, DebateTopology::Ensemble);
        assert_eq!(mgr.deadlock_counter, 0);
        assert_eq!(mgr.history.len(), 1);
        assert_eq!(
            mgr.history[0],
            (10, DebateTopology::Ensemble, TopologyReason::ManualOverride)
        );
        assert_eq!(directive.topology, DebateTopology::Ensemble);
    }

    #[test]
    fn generate_thought_branches_count() {
        let branches = TopologyManager::generate_thought_branches("solve this", 3);
        assert_eq!(branches.len(), 3);
        assert!(branches[0].contains("OPTIMISTIC"));
        assert!(branches[1].contains("PESSIMISTIC"));
        assert!(branches[2].contains("UNCONVENTIONAL"));
    }

    #[test]
    fn generate_thought_branches_capped_at_five() {
        let branches = TopologyManager::generate_thought_branches("test", 10);
        assert_eq!(branches.len(), 5, "capped at directive count");
    }

    #[test]
    fn build_pairs_even() {
        let pairs = TopologyManager::build_pairs(4);
        assert_eq!(pairs, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn build_pairs_odd() {
        let pairs = TopologyManager::build_pairs(5);
        assert_eq!(pairs, vec![(0, 1), (2, 3), (4, 0)]);
    }

    #[test]
    fn set_agent_count_triggers_shift_when_topology_unsatisfied() {
        let mut mgr = TopologyManager::new(4);
        mgr.current = DebateTopology::Ensemble; // requires >= 3
        let directive = mgr.set_agent_count(2, 5);
        assert!(
            directive.is_some(),
            "dropping to 2 agents should trigger shift"
        );
        let d = directive.unwrap();
        assert_ne!(
            d.topology,
            DebateTopology::Ensemble,
            "Ensemble needs >= 3 agents"
        );
    }

    #[test]
    fn critique_directive_uses_single_grouping() {
        let mut mgr = TopologyManager::new(4);
        let directive = mgr.shift_to(DebateTopology::Critique, 1, TopologyReason::ManualOverride);
        assert!(
            matches!(directive.agent_grouping, AgentGrouping::Single(0)),
            "Critique proposer is always agent 0"
        );
    }

    #[test]
    fn adversarial_directive_uses_pairs() {
        let mut mgr = TopologyManager::new(4);
        let directive = mgr.shift_to(
            DebateTopology::Adversarial,
            1,
            TopologyReason::ManualOverride,
        );
        assert!(
            matches!(directive.agent_grouping, AgentGrouping::Pairs(_)),
            "Adversarial should yield Pairs grouping"
        );
    }
}
