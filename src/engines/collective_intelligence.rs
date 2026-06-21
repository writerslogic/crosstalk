use crate::engines::metacognition::MetacognitiveObserver;
use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
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

    /// Serialize agent profiles and meta-strategy outcomes to JSON for cross-session persistence.
    pub fn export_state_json(&self) -> String {
        serde_json::to_string(&(&self.profiles, &self.meta_optimizer.outcomes)).unwrap_or_default()
    }

    /// Restore agent profiles and meta-strategy outcomes from a prior session's JSON.
    ///
    /// Profiles are merged (prior entries only fill gaps; existing entries win).
    /// Strategy outcomes are accumulated (trial counts and quality sums are added).
    pub fn import_state_json(&mut self, json: &str) {
        type State = (
            BTreeMap<String, AgentProfile>,
            BTreeMap<MetaStrategy, StrategyOutcome>,
        );
        if let Ok((profiles, outcomes)) = serde_json::from_str::<State>(json) {
            for (id, profile) in profiles {
                self.profiles.entry(id).or_insert(profile);
            }
            for (strategy, prior) in outcomes {
                let e = self.meta_optimizer.outcomes.entry(strategy).or_default();
                e.quality_sum += prior.quality_sum;
                e.trial_count += prior.trial_count;
            }
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

    pub fn select_strategy_adaptive(
        &self,
        sigma: &ConversationState,
        observer: &MetacognitiveObserver,
    ) -> AdaptiveSelection {
        self.meta_optimizer.select_adaptive(sigma, observer)
    }
}

pub struct KnowledgeTransfer;

impl KnowledgeTransfer {
    /// Maximum bytes of AI-generated content embedded in a lesson. Caps the
    /// blast radius of a prompt-injection attempt carried in turn content.
    const MAX_LESSON_CONTENT_BYTES: usize = 4096;

    /// Strip characters and substrings that act as prompt-delimiter signals in
    /// common LLM prompt formats (ChatML, Llama-2 [INST], H4 ###, XML-style).
    fn sanitize_for_lesson(raw: &str) -> String {
        // Dangerous delimiter patterns that could break out of the lesson block
        // and inject instructions into the surrounding agent prompt.
        const STRIP_PATTERNS: &[&str] = &[
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>",
            "###",
            "<|im_start|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ];
        let truncated = if raw.len() > Self::MAX_LESSON_CONTENT_BYTES {
            // Truncate at a UTF-8 character boundary.
            let mut end = Self::MAX_LESSON_CONTENT_BYTES;
            while !raw.is_char_boundary(end) {
                end -= 1;
            }
            &raw[..end]
        } else {
            raw
        };
        let mut out = truncated.to_string();
        for pat in STRIP_PATTERNS {
            out = out.replace(pat, "");
        }
        out
    }

    #[must_use]
    pub fn pack_lesson(turn: &Turn) -> Option<TransferableLesson> {
        if turn.outcome == TurnOutcome::TestsPassed {
            let safe_content = Self::sanitize_for_lesson(&turn.content);
            return Some(TransferableLesson {
                category: "success_pattern".to_string(),
                content: format!("Pattern discovered by {}: {}", turn.model_id, safe_content),
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
                    || l.applicability_tags
                        .iter()
                        .any(|t| task_tags.contains(&t.as_str()))
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
    pub fn review(reviewer_id: &str, proposal: &str) -> PeerReviewReport {
        let mut comments = vec![];
        let mut correctness: f64 = 0.8;
        let maintainability: f64 = 0.8;

        if proposal.contains("TODO") || proposal.contains("FIXME") {
            correctness -= 0.2;
            comments.push("Incomplete implementation (TODO found)".to_string());
        }

        // Use reviewer_id to potentially adjust strictness in the future
        if reviewer_id.contains("strict") {
            correctness -= 0.1;
        }

        PeerReviewReport {
            reviewer_id: reviewer_id.to_string(),
            correctness: correctness.clamp(0.0, 1.0),
            efficiency: 0.9,
            maintainability: maintainability.clamp(0.0, 1.0),
            comments,
        }
    }
}

pub struct EnsembleEngine;

impl EnsembleEngine {
    /// Quality-weighted merge: uses AST-based merging for code and paragraph selection for text.
    #[must_use]
    pub fn merge_proposals(
        proposals: Vec<(String, String, f64)>, // (agent_id, content, score)
        task_category: TaskCategory,
        language: &str,
    ) -> String {
        if proposals.is_empty() {
            return String::new();
        }
        if proposals.len() == 1 {
            return proposals
                .into_iter()
                .next()
                .expect("len == 1 checked above")
                .1;
        }

        // For Code Generation, use high-fidelity AST merging
        if task_category == TaskCategory::CodeGeneration || !language.is_empty() {
            let mut diff_proposals = Vec::new();
            let base = &proposals[0].1; // Use first proposal as baseline
            for (id, content, _score) in &proposals {
                let diff = crate::engines::diff::DiffEngine::generate_delta(base, content, 0);
                diff_proposals.push((id.clone(), diff));
            }

            if let Some(merged_code) =
                crate::engines::reasoning::SynthesisEngine::merge(base, diff_proposals, language)
            {
                return merged_code;
            }
        }

        // Fallback to paragraph-level selection for Research/Reasoning
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

        // Pre-flatten all (candidate, score) pairs once to avoid O(n²) re-splitting.
        let all_candidates: Vec<(&str, f64)> = proposals
            .iter()
            .flat_map(|(_, content, score)| content.split("\n\n").map(move |c| (c, *score)))
            .collect();

        // Pre-compute all candidate word-sets ONCE to avoid quadratic HashSet allocation.
        let candidate_word_sets: Vec<std::collections::HashSet<&str>> = all_candidates
            .iter()
            .map(|(c, _)| c.split_whitespace().collect())
            .collect();

        let base_word_set: std::collections::HashSet<&str> =
            proposals[best_idx].1.split_whitespace().collect();

        for para in &base_paragraphs {
            let para_words: std::collections::HashSet<&str> = para.split_whitespace().collect();
            let total = para_words.len().max(1);
            let mut best_para = (*para, proposals[best_idx].2);
            for (i, &(candidate, score)) in all_candidates.iter().enumerate() {
                let cand_words = &candidate_word_sets[i];
                let overlap = para_words.intersection(cand_words).count();
                if overlap as f64 / total as f64 > 0.6 && score > best_para.1 {
                    best_para = (candidate, score);
                }
            }
            if !merged.is_empty() {
                merged.push_str("\n\n");
            }
            merged.push_str(best_para.0);
        }

        // Append unique insights from non-best proposals that have low overlap with the merged text.
        // This captures novel perspectives that the best proposal missed.
        for (i, &(candidate, score)) in all_candidates.iter().enumerate() {
            if score < 0.3 {
                continue;
            }
            let cand_words = &candidate_word_sets[i];
            if cand_words.len() < 5 {
                continue;
            }
            let overlap = base_word_set.intersection(cand_words).count();
            let novelty = 1.0 - (overlap as f64 / cand_words.len().max(1) as f64);
            if novelty > 0.5 && candidate.len() > 20 {
                merged.push_str("\n\n[Additional insight] ");
                merged.push_str(candidate);
            }
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

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Default,
)]
pub enum MetaStrategy {
    #[default]
    DirectImplementation,
    DebateAndCritique,
    StepByStepReasoning,
    EnsembleVoting,
    /// Inject relevant memory lessons before proceeding; used when convergence stalls.
    MemoryInjection,
}

/// Result of adaptive strategy selection, including the recommended strategy and
/// the top-rated agent (if a dominant specialist was detected).
#[derive(Debug, Clone)]
pub struct AdaptiveSelection {
    pub strategy: MetaStrategy,
    /// Agent ID of the highest-Elo specialist, when `strategy` is `DirectImplementation`
    /// and a dominant agent was detected.
    pub preferred_agent: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
            MetaStrategy::MemoryInjection,
        ] {
            outcomes.insert(
                strategy,
                StrategyOutcome {
                    strategy,
                    quality_sum: 0.0,
                    trial_count: 0,
                },
            );
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
    pub fn select_best(&self, task_category: TaskCategory) -> MetaStrategy {
        // Apply task-category specific strategy bias
        if task_category == TaskCategory::Debugging {
            return MetaStrategy::DebateAndCritique;
        }
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

    /// Adaptive selection driven by live agent signals.  Scores each strategy
    /// based on observable conversation signals, then falls back to Thompson
    /// sampling when no signal is strong enough to override.
    pub fn select_adaptive(
        &self,
        sigma: &ConversationState,
        observer: &MetacognitiveObserver,
    ) -> AdaptiveSelection {
        let mut scores: BTreeMap<MetaStrategy, f64> = BTreeMap::new();

        // --- Signal 1: high proposal variance → DebateAndCritique ---
        let recent_certainties: Vec<f64> = sigma
            .turns
            .iter()
            .rev()
            .take(3)
            .filter_map(|t| t.certainty)
            .collect();
        if recent_certainties.len() >= 2 {
            let mean = recent_certainties.iter().sum::<f64>() / recent_certainties.len() as f64;
            let variance = recent_certainties
                .iter()
                .map(|c| (c - mean).powi(2))
                .sum::<f64>()
                / recent_certainties.len() as f64;
            if variance.sqrt() > 0.2 {
                *scores.entry(MetaStrategy::DebateAndCritique).or_insert(0.0) += 2.0;
            }
        }

        // --- Signal 2: dominant specialist → DirectImplementation ---
        let mut preferred_agent: Option<String> = None;
        let ranked = observer.ranked_agents();
        if let Some((top_id, top_elo)) = ranked.first() {
            let turn_count = sigma.turns.iter().filter(|t| &t.model_id == top_id).count();
            if *top_elo > 1600.0 && turn_count > 3 {
                *scores
                    .entry(MetaStrategy::DirectImplementation)
                    .or_insert(0.0) += 2.0;
                preferred_agent = Some(top_id.clone());
            }
        }

        // --- Signal 3: convergence stalling → MemoryInjection ---
        if sigma.completion_probability < 0.3 && sigma.turns.len() >= 5 {
            *scores.entry(MetaStrategy::MemoryInjection).or_insert(0.0) += 2.0;
        }

        // Pick the highest-scored strategy, breaking ties by Thompson sample.
        let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let strategy = if max_score > 0.0 {
            let mut rng = rand::rng();
            scores
                .iter()
                .filter(|(_, s)| **s == max_score)
                .max_by(|(strat_a, _), (strat_b, _)| {
                    let outcome_a = self.outcomes.get(strat_a);
                    let outcome_b = self.outcomes.get(strat_b);
                    let sample_a = outcome_a
                        .map(|o| Self::thompson_sample(o, &mut rng))
                        .unwrap_or(0.5);
                    let sample_b = outcome_b
                        .map(|o| Self::thompson_sample(o, &mut rng))
                        .unwrap_or(0.5);
                    sample_a.total_cmp(&sample_b)
                })
                .map(|(strat, _)| *strat)
                .unwrap_or(MetaStrategy::DirectImplementation)
        } else {
            // No signal — fall back to Thompson sampling over all strategies.
            let mut rng = rand::rng();
            self.outcomes
                .values()
                .max_by(|a, b| {
                    let sa = Self::thompson_sample(a, &mut rng);
                    let sb = Self::thompson_sample(b, &mut rng);
                    sa.total_cmp(&sb)
                })
                .map(|o| o.strategy)
                .unwrap_or(MetaStrategy::DirectImplementation)
        };

        // Clear preferred_agent if the winning strategy is not DirectImplementation.
        if strategy != MetaStrategy::DirectImplementation {
            preferred_agent = None;
        }

        AdaptiveSelection {
            strategy,
            preferred_agent,
        }
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
    if shape > 100.0 {
        return shape;
    }
    if shape < 1.0 {
        let u: f64 = rng.random();
        return sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    let mut best_dv = d * f64::EPSILON;
    for _ in 0..1000 {
        let x: f64 = {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            (u1.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        best_dv = d * v;
        let u: f64 = rng.random();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return best_dv;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return best_dv;
        }
    }
    // Iteration cap reached: return best approximation or shape as fallback.
    if best_dv > d * f64::EPSILON {
        best_dv
    } else {
        shape
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
        let best_single = agent_scores.values().cloned().fold(0.0_f64, f64::max);
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
        let sum_xy: f64 = self
            .score_history
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * y)
            .sum();
        let sum_xx: f64 = (0..n).map(|i| (i * i) as f64).sum();
        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom == 0.0 {
            0.0
        } else {
            (n_f * sum_xy - sum_x * sum_y) / denom
        }
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
        self.records
            .get(&(agent_id.to_string(), task_type.to_string()))
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
    pub fn scan(profiles: &BTreeMap<String, AgentProfile>, threshold: f64) -> Vec<CapabilityGap> {
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
        let (tp, fp, fn_) = self
            .stats
            .entry(reviewer_id.to_string())
            .or_insert((0, 0, 0));
        match (flagged, real_issue) {
            (true, true) => *tp += 1,
            (true, false) => *fp += 1,
            (false, true) => *fn_ += 1,
            (false, false) => {}
        }
    }

    #[must_use]
    pub fn precision(&self, reviewer_id: &str) -> f64 {
        let Some(&(tp, fp, _)) = self.stats.get(reviewer_id) else {
            return 0.0;
        };
        let denom = tp + fp;
        if denom == 0 {
            0.0
        } else {
            tp as f64 / denom as f64
        }
    }

    #[must_use]
    pub fn recall(&self, reviewer_id: &str) -> f64 {
        let Some(&(tp, _, fn_)) = self.stats.get(reviewer_id) else {
            return 0.0;
        };
        let denom = tp + fn_;
        if denom == 0 {
            0.0
        } else {
            tp as f64 / denom as f64
        }
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
                .map(|n| ProtocolArm {
                    name: n.to_string(),
                    total_reward: 0.0,
                    trials: 0,
                })
                .collect(),
            total_trials: 0,
        }
    }

    /// Select the arm with the highest UCB1 score.
    #[must_use]
    pub fn select(&self) -> Option<&str> {
        self.arms
            .iter()
            .max_by(|a, b| {
                a.ucb1(self.total_trials)
                    .total_cmp(&b.ucb1(self.total_trials))
            })
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
