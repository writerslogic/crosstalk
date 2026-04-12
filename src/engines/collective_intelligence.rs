use crate::types::conversation::{Turn, TurnOutcome, TaskCategory};
use crate::types::intelligence::AgentProfile;
use crate::types::memory::TransferableLesson;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct RefinementRound {
    pub round_index: u32,
    pub agent_contributions: Vec<(String, String)>,
}

pub struct CollectiveIntelligenceEngine {
    pub profiles: BTreeMap<String, AgentProfile>,
    pub meta_optimizer: MetaStrategyOptimizer,
}

impl Default for CollectiveIntelligenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectiveIntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: BTreeMap::new(),
            meta_optimizer: MetaStrategyOptimizer::new(),
        }
    }

    pub fn update_specialization(&mut self, turn: &Turn) {
        let profile = self
            .profiles
            .entry(turn.model_id.clone())
            .or_insert(AgentProfile {
                model_id: turn.model_id.clone(),
                capabilities: BTreeMap::new(),
                total_turns: 0,
                compilation_success_rate: 0.0,
            });

        if let Some(cat) = turn.task_category {
            let score = match turn.outcome {
                TurnOutcome::TestsPassed => 1.0,
                TurnOutcome::Compiled => 0.8,
                TurnOutcome::AdvancedConvergence => 0.7,
                TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.0,
                TurnOutcome::Stalled => 0.3,
                _ => 0.5,
            };
            let current = profile.capabilities.entry(cat).or_insert(0.5);
            // Exponential moving average: 0.9 decay
            *current = (*current * 0.9) + (score * 0.1);
        }
        profile.total_turns += 1;
    }

    pub fn select_strategy(&self, task_category: TaskCategory) -> MetaStrategy {
        self.meta_optimizer.select_best(task_category)
    }
}

pub struct KnowledgeTransfer;

impl KnowledgeTransfer {
    #[must_use]
    pub fn pack_lesson(turn: &Turn) -> Option<TransferableLesson> {
        if turn.outcome == TurnOutcome::TestsPassed {
            return Some(TransferableLesson {
                category: "success_pattern".to_string(),
                content: format!("Pattern discovered by {}: {}", turn.model_id, turn.content),
                confidence: 0.9,
                applicability_tags: vec!["success".to_string()],
            });
        }
        None
    }

    /// Prepend relevant lessons to `agent_context`, returning the enriched prompt.
    ///
    /// A lesson is injected when it shares at least one applicability tag with
    /// `task_tags`, or when its category matches `task_category`.
    #[must_use]
    pub fn inject(
        agent_context: &str,
        lessons: &[TransferableLesson],
        task_category: &str,
        task_tags: &[&str],
    ) -> String {
        let relevant: Vec<&TransferableLesson> = lessons
            .iter()
            .filter(|l| {
                l.category == task_category
                    || l.applicability_tags.iter().any(|t| task_tags.contains(&t.as_str()))
            })
            .collect();

        if relevant.is_empty() {
            return agent_context.to_string();
        }

        let prefix: String = relevant
            .iter()
            .map(|l| format!("[lesson] {}", l.content))
            .collect::<Vec<_>>()
            .join("\n");

        format!("{prefix}\n\n{agent_context}")
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PeerReviewReport {
    pub reviewer_id: String,
    pub correctness: f64,
    pub efficiency: f64,
    pub maintainability: f64,
    pub comments: Vec<String>,
}

pub struct PeerReview;

impl PeerReview {
    #[must_use]
    pub fn review(reviewer_id: &str, proposal: &str) -> PeerReviewReport {
        let mut comments = vec![];
        let mut correctness: f64 = 0.7;
        let mut maintainability: f64 = 0.8;
        let mut efficiency: f64 = 0.8;

        if proposal.contains("TODO") || proposal.contains("FIXME") {
            correctness -= 0.2;
            comments.push("Incomplete implementation detected (TODO/FIXME found)".to_string());
        }
        if proposal.contains("unwrap()") || proposal.contains("expect(") {
            correctness -= 0.1;
            comments.push("Potential panic: unwrap/expect usage detected.".to_string());
        }
        if proposal.contains("clone()") {
            efficiency -= 0.05;
            comments.push("Unnecessary clone may indicate borrow issue.".to_string());
        }
        if proposal.contains("unsafe") {
            correctness -= 0.15;
            comments.push("Unsafe block present: requires manual safety justification.".to_string());
        }
        if !proposal.contains("///") && proposal.contains("pub fn") {
            maintainability -= 0.1;
            comments.push("Public function missing doc comment.".to_string());
        }

        PeerReviewReport {
            reviewer_id: reviewer_id.to_string(),
            correctness: correctness.clamp(0.0, 1.0),
            efficiency: efficiency.clamp(0.0, 1.0),
            maintainability: maintainability.clamp(0.0, 1.0),
            comments,
        }
    }
}

pub struct EnsembleEngine;

impl EnsembleEngine {
    /// Quality-weighted merge: select paragraphs from the proposal with the
    /// highest score for that segment, falling back to the best overall.
    #[must_use]
    pub fn merge_proposals(proposals: Vec<(String, String, f64)>) -> String {
        if proposals.is_empty() {
            return String::new();
        }

        let best_idx = proposals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.2.total_cmp(&b.2))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let base_paragraphs: Vec<&str> = proposals[best_idx].1.split("\n\n").collect();
        let mut merged = String::with_capacity(proposals[best_idx].1.len());

        for para in &base_paragraphs {
            let mut best_para = (*para, proposals[best_idx].2);
            for (_, content, score) in &proposals {
                for candidate in content.split("\n\n") {
                    let overlap = para
                        .split_whitespace()
                        .filter(|w| candidate.contains(*w))
                        .count();
                    let total = para.split_whitespace().count().max(1);
                    if overlap as f64 / total as f64 > 0.5 && *score > best_para.1 {
                        best_para = (candidate, *score);
                    }
                }
            }
            if !merged.is_empty() {
                merged.push_str("\n\n");
            }
            merged.push_str(best_para.0);
        }

        merged
    }
}

#[derive(Debug, Clone)]
pub struct TeamComposition {
    pub architect: Option<String>,
    pub coder: Option<String>,
    pub critic: Option<String>,
}

pub struct DynamicTeamComposer;

impl DynamicTeamComposer {
    #[must_use]
    pub fn compose(
        profiles: &BTreeMap<String, AgentProfile>,
        task_category: &str,
    ) -> TeamComposition {
        let mut ranked: Vec<(&String, f64)> = profiles
            .iter()
            .map(|(id, p)| {
                let score = p
                    .capabilities
                    .iter()
                    .find(|(cat, _)| format!("{:?}", cat).to_lowercase().contains(task_category))
                    .map(|(_, s)| *s)
                    .unwrap_or(0.5);
                (id, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        TeamComposition {
            architect: ranked.first().map(|(id, _)| (*id).clone()),
            coder: ranked.get(1).map(|(id, _)| (*id).clone()),
            critic: ranked.get(2).map(|(id, _)| (*id).clone()),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum MetaStrategy {
    #[default]
    DirectImplementation,
    DebateAndCritique,
    StepByStepReasoning,
    EnsembleVoting,
}

#[derive(Debug, Clone, Default)]
pub struct StrategyOutcome {
    pub strategy: MetaStrategy,
    pub quality_sum: f64,
    pub trial_count: u32,
}

impl StrategyOutcome {
    #[must_use]
    pub fn avg_quality(&self) -> f64 {
        if self.trial_count == 0 {
            0.0
        } else {
            self.quality_sum / self.trial_count as f64
        }
    }
}

pub struct MetaStrategyOptimizer {
    pub outcomes: BTreeMap<MetaStrategy, StrategyOutcome>,
}

impl MetaStrategyOptimizer {
    pub fn new() -> Self {
        let mut outcomes = BTreeMap::new();
        for strategy in [
            MetaStrategy::DirectImplementation,
            MetaStrategy::DebateAndCritique,
            MetaStrategy::StepByStepReasoning,
            MetaStrategy::EnsembleVoting,
        ] {
            outcomes.insert(strategy, StrategyOutcome { strategy, quality_sum: 0.0, trial_count: 0 });
        }
        Self { outcomes }
    }

    pub fn record(&mut self, strategy: MetaStrategy, quality: f64) {
        let e = self.outcomes.entry(strategy).or_insert(StrategyOutcome {
            strategy,
            quality_sum: 0.0,
            trial_count: 0,
        });
        e.quality_sum += quality;
        e.trial_count += 1;
    }

    #[must_use]
    pub fn best_strategy(&self) -> Option<MetaStrategy> {
        self.outcomes
            .values()
            .filter(|o| o.trial_count >= 3)
            .max_by(|a, b| a.avg_quality().total_cmp(&b.avg_quality()))
            .map(|o| o.strategy)
    }

    #[must_use]
    pub fn select_best(&self, _task_category: TaskCategory) -> MetaStrategy {
        let mut rng = rand::rng();

        self.outcomes
            .values()
            .max_by(|a, b| {
                let sample_a = Self::thompson_sample(a, &mut rng);
                let sample_b = Self::thompson_sample(b, &mut rng);
                sample_a.total_cmp(&sample_b)
            })
            .map(|o| o.strategy)
            .unwrap_or(MetaStrategy::DirectImplementation)
    }

    fn thompson_sample(outcome: &StrategyOutcome, rng: &mut impl rand::Rng) -> f64 {
        if outcome.trial_count == 0 {
            return rng.random::<f64>();
        }
        let avg = outcome.avg_quality().clamp(0.0, 1.0);
        let successes = (avg * outcome.trial_count as f64) as u32;
        let failures = outcome.trial_count.saturating_sub(successes);
        let alpha = successes as f64 + 1.0;
        let beta = failures as f64 + 1.0;
        sample_beta(rng, alpha, beta)
    }
}

impl Default for MetaStrategyOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

fn sample_beta(rng: &mut impl rand::Rng, alpha: f64, beta: f64) -> f64 {
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    if x + y == 0.0 { 0.5 } else { x / (x + y) }
}

fn sample_gamma(rng: &mut impl rand::Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let u: f64 = rng.random();
        return sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            (u1.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 { continue; }
        let u: f64 = rng.random();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// ── SwarmPremiumCalculator ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SwarmPremium {
    pub task_type: String,
    pub multi_agent_score: f64,
    pub best_single_agent_score: f64,
    pub premium_pct: f64,
}

pub struct SwarmPremiumCalculator;

impl SwarmPremiumCalculator {
    /// Compute how much better the multi-agent result was versus the best
    /// single-agent score. `agent_scores` maps agent_id → quality score.
    #[must_use]
    pub fn compute(
        task_type: &str,
        multi_agent_score: f64,
        agent_scores: &BTreeMap<String, f64>,
    ) -> SwarmPremium {
        let best_single = agent_scores
            .values()
            .cloned()
            .fold(0.0_f64, f64::max);
        let premium_pct = if best_single > 0.0 {
            ((multi_agent_score - best_single) / best_single) * 100.0
        } else {
            0.0
        };
        SwarmPremium {
            task_type: task_type.to_string(),
            multi_agent_score,
            best_single_agent_score: best_single,
            premium_pct,
        }
    }
}

// ── SkillProgressionTracker ───────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct SkillRecord {
    pub agent_id: String,
    pub task_type: String,
    pub score_history: Vec<f64>,
}

impl SkillRecord {
    /// Linear regression slope over `score_history` as improvement rate.
    #[must_use]
    pub fn improvement_rate(&self) -> f64 {
        let n = self.score_history.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let sum_x: f64 = (0..n).map(|i| i as f64).sum();
        let sum_y: f64 = self.score_history.iter().sum();
        let sum_xy: f64 = self.score_history.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..n).map(|i| (i * i) as f64).sum();
        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom == 0.0 { 0.0 } else { (n_f * sum_xy - sum_x * sum_y) / denom }
    }

    #[must_use]
    pub fn is_plateauing(&self) -> bool {
        self.score_history.len() >= 5 && self.improvement_rate().abs() < 0.005
    }
}

#[derive(Debug, Clone, Default)]
pub struct SkillProgressionTracker {
    pub records: BTreeMap<(String, String), SkillRecord>, // (agent_id, task_type)
}

impl SkillProgressionTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, agent_id: &str, task_type: &str, score: f64) {
        let key = (agent_id.to_string(), task_type.to_string());
        let rec = self.records.entry(key).or_insert_with(|| SkillRecord {
            agent_id: agent_id.to_string(),
            task_type: task_type.to_string(),
            score_history: vec![],
        });
        rec.score_history.push(score);
    }

    #[must_use]
    pub fn get(&self, agent_id: &str, task_type: &str) -> Option<&SkillRecord> {
        self.records.get(&(agent_id.to_string(), task_type.to_string()))
    }
}

// ── CapabilityGapScanner ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CapabilityGap {
    pub task_type: String,
    pub best_score: f64,
    pub threshold: f64,
}

pub struct CapabilityGapScanner;

impl CapabilityGapScanner {
    const DEFAULT_THRESHOLD: f64 = 0.6;

    /// Find task types where the best agent's score falls below `threshold`.
    #[must_use]
    pub fn scan(
        profiles: &BTreeMap<String, AgentProfile>,
        threshold: f64,
    ) -> Vec<CapabilityGap> {
        let mut best_per_task: BTreeMap<String, f64> = BTreeMap::new();
        for profile in profiles.values() {
            for (cat, &score) in &profile.capabilities {
                let key = format!("{cat:?}");
                let entry = best_per_task.entry(key).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            }
        }
        best_per_task
            .into_iter()
            .filter(|(_, best)| *best < threshold)
            .map(|(task_type, best_score)| CapabilityGap {
                task_type,
                best_score,
                threshold,
            })
            .collect()
    }

    #[must_use]
    pub fn scan_default(profiles: &BTreeMap<String, AgentProfile>) -> Vec<CapabilityGap> {
        Self::scan(profiles, Self::DEFAULT_THRESHOLD)
    }
}

// ── ReviewerCalibrator ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ReviewerCalibrator {
    /// (reviewer_id) -> (true_positives, false_positives, false_negatives)
    stats: BTreeMap<String, (u32, u32, u32)>,
}

impl ReviewerCalibrator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a review outcome.
    /// `flagged` = reviewer flagged an issue; `real_issue` = issue was genuine.
    pub fn record(&mut self, reviewer_id: &str, flagged: bool, real_issue: bool) {
        let (tp, fp, fn_) = self.stats.entry(reviewer_id.to_string()).or_insert((0, 0, 0));
        match (flagged, real_issue) {
            (true, true) => *tp += 1,
            (true, false) => *fp += 1,
            (false, true) => *fn_ += 1,
            (false, false) => {}
        }
    }

    #[must_use]
    pub fn precision(&self, reviewer_id: &str) -> f64 {
        let Some(&(tp, fp, _)) = self.stats.get(reviewer_id) else { return 0.0 };
        let denom = tp + fp;
        if denom == 0 { 0.0 } else { tp as f64 / denom as f64 }
    }

    #[must_use]
    pub fn recall(&self, reviewer_id: &str) -> f64 {
        let Some(&(tp, _, fn_)) = self.stats.get(reviewer_id) else { return 0.0 };
        let denom = tp + fn_;
        if denom == 0 { 0.0 } else { tp as f64 / denom as f64 }
    }
}

// ── UCB1ProtocolSelector ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ProtocolArm {
    pub name: String,
    pub total_reward: f64,
    pub trials: u32,
}

impl ProtocolArm {
    fn ucb1(&self, total_trials: u32) -> f64 {
        if self.trials == 0 {
            return f64::INFINITY;
        }
        let avg = self.total_reward / self.trials as f64;
        let exploration = (2.0 * (total_trials as f64 + 1.0).ln() / self.trials as f64).sqrt();
        avg + exploration
    }
}

#[derive(Debug, Clone, Default)]
pub struct UCB1ProtocolSelector {
    pub arms: Vec<ProtocolArm>,
    pub total_trials: u32,
}

impl UCB1ProtocolSelector {
    #[must_use]
    pub fn new(protocol_names: &[&str]) -> Self {
        Self {
            arms: protocol_names
                .iter()
                .map(|n| ProtocolArm { name: n.to_string(), total_reward: 0.0, trials: 0 })
                .collect(),
            total_trials: 0,
        }
    }

    /// Select the arm with the highest UCB1 score.
    #[must_use]
    pub fn select(&self) -> Option<&str> {
        self.arms
            .iter()
            .max_by(|a, b| a.ucb1(self.total_trials).total_cmp(&b.ucb1(self.total_trials)))
            .map(|a| a.name.as_str())
    }

    /// Record the reward for a protocol after a session.
    pub fn update(&mut self, protocol_name: &str, reward: f64) {
        if let Some(arm) = self.arms.iter_mut().find(|a| a.name == protocol_name) {
            arm.total_reward += reward;
            arm.trials += 1;
        }
        self.total_trials += 1;
    }
}

// ── RoleSequenceRecorder ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RoleSequence {
    pub task_type: String,
    pub agent_order: Vec<(String, String)>, // (agent_id, role)
    pub outcome_quality: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RoleSequenceRecorder {
    pub sequences: Vec<RoleSequence>,
}

impl RoleSequenceRecorder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, task_type: &str, agent_order: Vec<(String, String)>, quality: f64) {
        self.sequences.push(RoleSequence {
            task_type: task_type.to_string(),
            agent_order,
            outcome_quality: quality,
        });
    }

    /// Return the agent ordering with the highest mean quality for a given task type.
    #[must_use]
    pub fn best_ordering(&self, task_type: &str) -> Option<&Vec<(String, String)>> {
        self.sequences
            .iter()
            .filter(|s| s.task_type == task_type)
            .max_by(|a, b| a.outcome_quality.total_cmp(&b.outcome_quality))
            .map(|s| &s.agent_order)
    }
}
