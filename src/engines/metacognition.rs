//! Metacognitive Observer — the executive function of the swarm.
//!
//! Monitors debate telemetry in real-time, detects epistemic failures,
//! injects corrective directives, and drives agent evolution through
//! Bayesian confidence tracking and Elo-based selection pressure.

use crate::engines::memory::{cosine_sim, local_embed_text};
use crate::engines::reasoning::{FallacyDetector, sanitize_directive_content};
use crate::engines::surprise::SurpriseEngine;
use crate::types::conversation::{TaskCategory, Turn, TurnOutcome};
use crate::types::security::FallacyReport;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

const DEFAULT_CONCESSION_THRESHOLD: f64 = 0.20;
const DEFAULT_DEADLOCK_THRESHOLD: u32 = 5;

/// Per-turn confidence decay rate applied to agents with no fresh evidence.
const CONFIDENCE_DECAY_RATE: f64 = 0.02;

/// Cosine similarity threshold above which a turn is considered globally
/// repetitive (any agent repeated something similar).
const REPETITION_SIM_GLOBAL: f32 = 0.82;

/// Cosine similarity threshold above which a *same-agent* turn is
/// considered repetitive.
const REPETITION_SIM_SAME_AGENT: f32 = 0.75;

/// How many same-agent near-duplicate turns trigger a repetition intervention.
const REPETITION_SAME_AGENT_COUNT: usize = 3;

/// Sliding window size for semantic embedding history.
const SEMANTIC_WINDOW: usize = 20;

/// Number of recent intervention history entries checked for dedup.
const INTERVENTION_DEDUP_WINDOW: usize = 3;

/// Maximum entries retained in the intervention history deque.
const INTERVENTION_HISTORY_CAP: usize = 50;

/// Cosine similarity threshold for cross-agent assumption defeat.
const DEFEAT_SIM_THRESHOLD: f32 = 0.75;

/// Minimum fraction of token overlap required before flagging a refutation.
const REFUTATION_OVERLAP_THRESHOLD: f64 = 0.3;

// =====================================================================
// EPISTEMIC STATE — what does the agent know that it knows?
// =====================================================================

/// Structured epistemic payload extracted from or attached to each turn.
/// Agents that output explicit assumptions can be Bayesian-updated.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EpistemicState {
    /// Explicit assumptions the agent relies on.
    pub assumptions: Vec<Assumption>,
    /// Confidence interval [0.0, 1.0] for the agent's overall position.
    pub confidence: f64,
    /// Evidence citations supporting the position.
    pub evidence: Vec<String>,
    /// Assumptions that were defeated by opposing agents.
    pub defeated: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Assumption {
    pub claim: String,
    pub confidence: f64,
    pub basis: AssumptionBasis,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum AssumptionBasis {
    Empirical,   // backed by data/code/tests
    Theoretical, // backed by reasoning
    Heuristic,   // rules of thumb
    Ungrounded,  // no stated basis
}

impl Default for EpistemicState {
    fn default() -> Self {
        Self {
            assumptions: vec![],
            confidence: 0.5,
            evidence: vec![],
            defeated: vec![],
        }
    }
}

impl EpistemicState {
    /// Bayesian update: when an assumption is defeated, reduce confidence
    /// proportional to how much weight that assumption carried.
    pub fn defeat_assumption(&mut self, claim: &str) {
        let total_weight: f64 = self.assumptions.iter().map(|a| a.confidence).sum();
        if total_weight == 0.0 {
            return;
        }
        if let Some(pos) = self.assumptions.iter().position(|a| a.claim == claim) {
            let weight = self.assumptions[pos].confidence / total_weight;
            self.confidence *= 1.0 - weight;
            self.assumptions[pos].confidence = 0.0;
            self.defeated.push(claim.to_string());
        }
    }

    /// Has this agent's confidence dropped below the concession threshold?
    pub fn should_concede(&self, threshold: f64) -> bool {
        self.confidence < threshold
    }

    /// Extract epistemic state from a turn's content by parsing structured markers.
    pub fn extract_from_content(content: &str) -> Self {
        const MAX_ASSUMPTIONS: usize = 20;
        let mut state = Self::default();
        let mut in_assumptions = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("[assumption]") || trimmed.starts_with("ASSUMPTION:") {
                let claim = trimmed
                    .trim_start_matches("[assumption]")
                    .trim_start_matches("ASSUMPTION:")
                    .trim()
                    .to_string();
                if !claim.is_empty() && state.assumptions.len() < MAX_ASSUMPTIONS {
                    state.assumptions.push(Assumption {
                        claim,
                        confidence: 0.7,
                        basis: AssumptionBasis::Theoretical,
                    });
                }
                in_assumptions = true;
            } else if trimmed.starts_with("[confidence]") || trimmed.starts_with("CONFIDENCE:") {
                let val = trimmed
                    .trim_start_matches("[confidence]")
                    .trim_start_matches("CONFIDENCE:")
                    .trim()
                    .trim_end_matches('%');
                if let Ok(c) = val.parse::<f64>() {
                    state.confidence = if c > 1.0 { c / 100.0 } else { c };
                }
            } else if trimmed.starts_with("[evidence]") || trimmed.starts_with("EVIDENCE:") {
                let ev = trimmed
                    .trim_start_matches("[evidence]")
                    .trim_start_matches("EVIDENCE:")
                    .trim()
                    .to_string();
                if !ev.is_empty() {
                    state.evidence.push(ev);
                }
            } else if in_assumptions
                && trimmed.starts_with("- ")
                && state.assumptions.len() < MAX_ASSUMPTIONS
            {
                state.assumptions.push(Assumption {
                    claim: trimmed[2..].to_string(),
                    confidence: 0.6,
                    basis: AssumptionBasis::Heuristic,
                });
            } else if trimmed.is_empty() {
                in_assumptions = false;
            }
        }
        state
    }
}

// =====================================================================
// METACOGNITIVE OBSERVER — the executive function
// =====================================================================

/// Intervention directive injected into an agent's next prompt.
#[derive(Debug, Clone)]
pub struct Intervention {
    pub target_agent: String,
    pub directive: String,
    pub severity: InterventionSeverity,
    pub source: InterventionSource,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterventionSeverity {
    /// Informational: agent may incorporate or ignore.
    Advisory,
    /// Agent must address this in its next response.
    Corrective,
    /// Agent must regenerate its response from scratch.
    Mandatory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterventionSource {
    FallacyDetection,
    EpistemicCollapse,
    CircularReasoning,
    StaleRepetition,
    TopologyShift,
}

/// Cumulative observer statistics across the lifetime of a debate session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObserverMetrics {
    pub total_interventions_issued: u64,
    pub total_assumptions_defeated: u64,
    pub total_concessions_forced: u64,
    pub total_topology_shifts: u64,
    /// Map of agent_id → mean epistemic confidence over all recorded turns.
    pub avg_confidence_by_agent: FxHashMap<String, f64>,
}

/// The Observer monitors debate telemetry and produces interventions.
pub struct MetacognitiveObserver {
    /// Per-agent epistemic state tracking.
    pub agent_states: FxHashMap<String, EpistemicState>,
    /// Per-agent Elo ratings for selection pressure (overall average across categories).
    pub elo_ratings: FxHashMap<String, f64>,
    /// Per-agent per-category Elo: index maps TaskCategory variants
    /// [CodeGeneration=0, Debugging=1, Architecture=2, Refactoring=3, Research=4, Testing=5, General=6].
    elo_by_category: FxHashMap<String, [f64; 7]>,
    /// Per-agent Beta-posterior calibration: (alpha, beta).
    /// alpha counts high-certainty turns that were verified correct.
    /// beta counts high-certainty turns that were verified wrong.
    /// Both start at 1.0 (uniform prior).
    pub calibration: std::collections::HashMap<String, (f64, f64)>,
    /// Recent interventions for dedup / rate-limiting.
    intervention_history: VecDeque<(String, InterventionSource)>,
    /// Concession threshold: agents below this must publicly concede.
    pub concession_threshold: f64,
    /// Recent turn embeddings for semantic repetition detection.
    /// Each entry is (agent_id, embedding).
    semantic_embeddings: VecDeque<(String, Vec<f32>)>,
    /// Turn counter for topology shift decisions.
    turns_since_progress: u32,
    /// Topology shift threshold.
    deadlock_threshold: u32,
    /// Per-agent intervention outcome history: (source, improved).
    intervention_outcomes: FxHashMap<String, Vec<(InterventionSource, bool)>>,
    /// Thompson sampling Beta(α, β) parameters per intervention source.
    source_alpha: FxHashMap<InterventionSource, u32>,
    source_beta: FxHashMap<InterventionSource, u32>,
    /// Cumulative metrics for this session.
    metrics: ObserverMetrics,
    /// Per-agent running confidence sum and sample count for avg_confidence tracking.
    confidence_accum: FxHashMap<String, (f64, u64)>,
}

impl MetacognitiveObserver {
    pub fn new() -> Self {
        Self {
            agent_states: FxHashMap::default(),
            elo_ratings: FxHashMap::default(),
            elo_by_category: FxHashMap::default(),
            calibration: std::collections::HashMap::new(),
            intervention_history: VecDeque::with_capacity(INTERVENTION_HISTORY_CAP),
            concession_threshold: DEFAULT_CONCESSION_THRESHOLD,
            semantic_embeddings: VecDeque::with_capacity(SEMANTIC_WINDOW),
            turns_since_progress: 0,
            deadlock_threshold: DEFAULT_DEADLOCK_THRESHOLD,
            intervention_outcomes: FxHashMap::default(),
            source_alpha: FxHashMap::default(),
            source_beta: FxHashMap::default(),
            metrics: ObserverMetrics::default(),
            confidence_accum: FxHashMap::default(),
        }
    }

    // ── Public entry point ────────────────────────────────────────────

    /// Analyze a completed turn and produce any necessary interventions.
    /// This is the main entry point called after each agent response.
    pub fn observe_turn(
        &mut self,
        turn: &Turn,
        all_recent_turns: &[Turn],
        surprise: &mut SurpriseEngine,
    ) -> Vec<Intervention> {
        let mut interventions = Vec::new();

        // 1. Extract/update epistemic state and running confidence average.
        let epistemic = self.update_epistemic_state(turn);

        // 2. Confidence decay for agents with no fresh evidence this turn.
        self.decay_confidence(CONFIDENCE_DECAY_RATE, &FxHashMap::default());

        // 3. Fallacy detection with corrective injection.
        self.detect_and_inject_fallacy_corrections(turn, all_recent_turns, &mut interventions);

        // 4. Epistemic collapse detection.
        self.detect_epistemic_collapse(turn, &epistemic, &mut interventions);

        // 5. Cross-agent assumption defeat.
        self.defeat_cross_agent_assumptions(turn);

        // 6. Stale repetition detection via cosine similarity.
        self.detect_stale_repetition(turn, &mut interventions);

        // 7. Surprise-calibrated Elo update.
        let surprise_val = surprise.compute_surprise(&turn.model_id, turn.outcome);
        let turn_category = turn.task_category.unwrap_or(TaskCategory::Research);
        self.update_elo(&turn.model_id, turn.outcome, surprise_val, turn_category);

        // 8. Progress tracking and topology shift recommendation.
        self.update_progress_and_topology(turn, &mut interventions);

        // Rate-limit: suppress duplicate interventions for the same agent+source
        // within the last INTERVENTION_DEDUP_WINDOW entries in history.
        self.dedup_and_record_interventions(&mut interventions);

        interventions
    }

    // ── Private phase helpers ─────────────────────────────────────────

    /// Phase 1: Extract epistemic state from the turn, update agent map and
    /// running confidence average.  Returns the freshly-extracted state.
    fn update_epistemic_state(&mut self, turn: &Turn) -> EpistemicState {
        let epistemic = EpistemicState::extract_from_content(&turn.content);
        self.agent_states
            .insert(turn.model_id.clone(), epistemic.clone());

        let (sum, count) = self
            .confidence_accum
            .entry(turn.model_id.clone())
            .or_insert((0.0, 0));
        *sum += epistemic.confidence;
        *count += 1;
        self.metrics
            .avg_confidence_by_agent
            .insert(turn.model_id.clone(), *sum / *count as f64);

        epistemic
    }

    /// Phase 3: Run fallacy detection and push corrective interventions.
    fn detect_and_inject_fallacy_corrections(
        &mut self,
        turn: &Turn,
        prior_turns: &[Turn],
        interventions: &mut Vec<Intervention>,
    ) {
        let fallacies = FallacyDetector::scan(&turn.content, prior_turns);
        for fallacy in &fallacies {
            if self.intervention_suppressed(&turn.model_id, InterventionSource::FallacyDetection) {
                continue;
            }
            interventions.push(Intervention {
                target_agent: turn.model_id.clone(),
                directive: self.compose_fallacy_correction(fallacy),
                severity: InterventionSeverity::Corrective,
                source: InterventionSource::FallacyDetection,
            });
        }
    }

    /// Phase 4: Detect epistemic collapse and push mandatory concession directive.
    fn detect_epistemic_collapse(
        &mut self,
        turn: &Turn,
        epistemic: &EpistemicState,
        interventions: &mut Vec<Intervention>,
    ) {
        if epistemic.should_concede(self.concession_threshold)
            && !self.intervention_suppressed(&turn.model_id, InterventionSource::EpistemicCollapse)
        {
            interventions.push(Intervention {
                target_agent: turn.model_id.clone(),
                directive: {
                    let safe_defeated: Vec<String> = epistemic.defeated
                        .iter()
                        .take(20)
                        .map(|d| sanitize_directive_content(d))
                        .collect();
                    format!(
                        "Your epistemic confidence has dropped to {:.0}%. \
                         You must publicly concede the points where your assumptions were defeated: {:?}. \
                         Refocus on your remaining strong positions.",
                        epistemic.confidence * 100.0,
                        safe_defeated
                    )
                },
                severity: InterventionSeverity::Mandatory,
                source: InterventionSource::EpistemicCollapse,
            });
            self.metrics.total_concessions_forced += 1;
        }
    }

    /// Phase 5: Cross-agent assumption defeat.
    ///
    /// If this turn contains negation/refutation keywords AND the turn's
    /// embedding is semantically close to another agent's assumption, mark
    /// that assumption defeated.  Embeddings for all live assumptions are
    /// pre-computed once before the per-agent loop to avoid redundant calls.
    fn defeat_cross_agent_assumptions(&mut self, turn: &Turn) {
        let defeater_content = turn.content.to_lowercase();
        let defeater_embedding = local_embed_text(&turn.content);

        let has_refutation = defeater_content.contains("incorrect")
            || defeater_content.contains("wrong")
            || defeater_content.contains("flawed")
            || defeater_content.contains("actually");
        let has_negation = defeater_content.contains("not ")
            || defeater_content.contains("no ")
            || defeater_content.contains("never")
            || defeater_content.contains("doesn't")
            || defeater_content.contains("isn't")
            || defeater_content.contains("won't")
            || defeater_content.contains("cannot");

        // Pre-compute all assumption claim embeddings per-agent so we
        // call local_embed_text once per (agent, claim) across the whole
        // function, not once per assumption per iteration.
        let mut per_agent_claim_embeddings: FxHashMap<String, Vec<(String, Vec<f32>)>> =
            FxHashMap::default();
        for (other_id, other_state) in &self.agent_states {
            if *other_id == turn.model_id {
                continue;
            }
            let embeddings: Vec<(String, Vec<f32>)> = other_state
                .assumptions
                .iter()
                .filter(|a| a.confidence > 0.0)
                .map(|a| (a.claim.clone(), local_embed_text(&a.claim)))
                .collect();
            if !embeddings.is_empty() {
                per_agent_claim_embeddings.insert(other_id.clone(), embeddings);
            }
        }

        // Now evaluate defeats using cached embeddings.
        for (other_id, claim_embeddings) in &per_agent_claim_embeddings {
            let defeated: Vec<String> = claim_embeddings
                .iter()
                .filter(|(claim, claim_emb)| {
                    let sim = cosine_sim(claim_emb, &defeater_embedding);
                    if sim <= DEFEAT_SIM_THRESHOLD {
                        return false;
                    }
                    // Both a semantic hit AND a keyword signal are required.
                    let keyword_match = has_refutation || has_negation;
                    if !keyword_match {
                        return false;
                    }
                    // Require token overlap with the turn content to filter
                    // negations that are unrelated to this specific claim.
                    token_overlap(claim, &defeater_content) > REFUTATION_OVERLAP_THRESHOLD
                })
                .map(|(claim, _)| claim.clone())
                .collect();

            let n = defeated.len() as u64;
            if n > 0 {
                if let Some(other_state) = self.agent_states.get_mut(other_id) {
                    for claim in &defeated {
                        other_state.defeat_assumption(claim);
                    }
                }
                self.metrics.total_assumptions_defeated += n;
            }
        }
    }

    /// Phase 6: Detect stale/repetitive turns via cosine similarity on the
    /// semantic embedding window.
    fn detect_stale_repetition(&mut self, turn: &Turn, interventions: &mut Vec<Intervention>) {
        let embedding = local_embed_text(&turn.content);
        let is_repetitive = self
            .semantic_embeddings
            .iter()
            .any(|(_, prev)| cosine_sim(prev, &embedding) > REPETITION_SIM_GLOBAL)
            || self
                .semantic_embeddings
                .iter()
                .filter(|(id, prev)| {
                    id == &turn.model_id && cosine_sim(prev, &embedding) > REPETITION_SIM_SAME_AGENT
                })
                .count()
                >= REPETITION_SAME_AGENT_COUNT;

        self.semantic_embeddings
            .push_back((turn.model_id.clone(), embedding));
        if self.semantic_embeddings.len() > SEMANTIC_WINDOW {
            self.semantic_embeddings.pop_front();
        }

        if is_repetitive
            && !self.intervention_suppressed(&turn.model_id, InterventionSource::StaleRepetition)
        {
            interventions.push(Intervention {
                target_agent: turn.model_id.clone(),
                directive: "Your response is semantically redundant with a prior turn. \
                           You must introduce novel analysis, a new perspective, or concede. \
                           Restating the same position is not permitted."
                    .to_string(),
                severity: InterventionSeverity::Corrective,
                source: InterventionSource::StaleRepetition,
            });
        }
    }

    /// Phase 8: Update progress counter and emit topology shift if deadlocked.
    fn update_progress_and_topology(&mut self, turn: &Turn, interventions: &mut Vec<Intervention>) {
        let made_progress = matches!(
            turn.outcome,
            TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
        );
        if made_progress {
            self.turns_since_progress = 0;
        } else {
            self.turns_since_progress += 1;
        }

        if self.turns_since_progress >= self.deadlock_threshold {
            interventions.push(Intervention {
                target_agent: "System".to_string(),
                directive: format!(
                    "TOPOLOGY_SHIFT: {} consecutive turns without progress. \
                     Recommend: spawn mediator agent, switch to Tree of Thoughts, \
                     or force agent concession cascade.",
                    self.turns_since_progress
                ),
                severity: InterventionSeverity::Mandatory,
                source: InterventionSource::TopologyShift,
            });
            self.turns_since_progress = 0;
            self.metrics.total_topology_shifts += 1;
        }
    }

    /// Post-processing: remove duplicate interventions (same agent+source seen
    /// in the last INTERVENTION_DEDUP_WINDOW history entries), record survivors.
    fn dedup_and_record_interventions(&mut self, interventions: &mut Vec<Intervention>) {
        // Suppress sources with pessimistic Thompson sample (< 0.2) before recording.
        interventions.retain(|i| !self.intervention_suppressed(&i.target_agent, i.source));

        // Dedup: skip interventions recently issued to the same agent from the same source.
        interventions.retain(|i| {
            let key = (i.target_agent.clone(), i.source);
            let dominated = self
                .intervention_history
                .iter()
                .rev()
                .take(INTERVENTION_DEDUP_WINDOW)
                .any(|(a, s)| *a == key.0 && *s == key.1);
            if !dominated {
                self.intervention_history.push_back(key);
                if self.intervention_history.len() > INTERVENTION_HISTORY_CAP {
                    self.intervention_history.pop_front();
                }
                self.metrics.total_interventions_issued += 1;
                true
            } else {
                false
            }
        });

        // For the same target_agent, keep only the intervention with the highest Thompson sample.
        let mut best_per_agent: FxHashMap<String, (f64, usize)> = FxHashMap::default();
        for (idx, i) in interventions.iter().enumerate() {
            let q = self.sample_source_quality(i.source);
            let entry = best_per_agent
                .entry(i.target_agent.clone())
                .or_insert((f64::NEG_INFINITY, idx));
            if q > entry.0 {
                *entry = (q, idx);
            }
        }
        let keep: HashSet<usize> = best_per_agent.values().map(|(_, i)| *i).collect();
        let mut idx = 0;
        interventions.retain(|_| {
            let keep_it = keep.contains(&idx);
            idx += 1;
            keep_it
        });
    }

    /// Global success rate for a given intervention source across all agents.
    /// Returns 1.0 (optimistic prior) when no history exists for that source.
    pub fn source_success_rate(&self, source: InterventionSource) -> f64 {
        let matching: Vec<bool> = self
            .intervention_outcomes
            .values()
            .flat_map(|v| v.iter())
            .filter(|(s, _)| *s == source)
            .map(|(_, ok)| *ok)
            .collect();
        if matching.is_empty() {
            return 1.0;
        }
        matching.iter().filter(|&&ok| ok).count() as f64 / matching.len() as f64
    }

    // ── Remaining public / private methods ───────────────────────────

    /// Apply a per-turn confidence decay to all agents that provided no fresh
    /// evidence this turn.  Agents with evidence in their current state are
    /// exempt, as are agents with recent quality > 0.6.  Confidence is
    /// floor-clamped at 0.0.
    pub fn decay_confidence(&mut self, rate: f64, recent_quality: &FxHashMap<String, f64>) {
        for (id, state) in self.agent_states.iter_mut() {
            if !state.evidence.is_empty() {
                continue;
            }
            if recent_quality.get(id).is_some_and(|q| *q > 0.6) {
                continue;
            }
            state.confidence = (state.confidence - rate).max(0.0);
        }
    }

    /// Record whether an intervention issued to `agent_id` from `source`
    /// resulted in an improved subsequent turn.
    pub fn record_intervention_outcome(
        &mut self,
        agent_id: &str,
        source: InterventionSource,
        improved: bool,
    ) {
        let outcomes = self
            .intervention_outcomes
            .entry(agent_id.to_string())
            .or_default();
        outcomes.push((source, improved));
        if outcomes.len() > 500 {
            outcomes.drain(..outcomes.len() - 500);
        }
        if improved {
            *self.source_alpha.entry(source).or_insert(1) += 1;
        } else {
            *self.source_beta.entry(source).or_insert(1) += 1;
        }
    }

    /// Thompson sample from Beta(α, β) for the given intervention source.
    /// Returns a value in (0, 1) representing estimated source quality.
    pub fn sample_source_quality(&self, source: InterventionSource) -> f64 {
        let alpha = *self.source_alpha.get(&source).unwrap_or(&1) as f64;
        let beta = *self.source_beta.get(&source).unwrap_or(&1) as f64;
        // Simple Beta approximation: mean + small noise scaled by variance
        let mean = alpha / (alpha + beta);
        let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let noise = (rand::random::<f64>() - 0.5) * variance.sqrt();
        (mean + noise).clamp(0.0, 1.0)
    }

    /// Fraction of interventions for `agent_id` that led to improvement.
    /// Returns 1.0 when no history exists (optimistic prior, don't suppress).
    pub fn intervention_success_rate(&self, agent_id: &str) -> f64 {
        match self.intervention_outcomes.get(agent_id) {
            None => 1.0,
            Some(outcomes) if outcomes.is_empty() => 1.0,
            Some(outcomes) => {
                let successes = outcomes.iter().filter(|(_, ok)| *ok).count();
                successes as f64 / outcomes.len() as f64
            }
        }
    }

    /// Returns true when interventions for this agent should be withheld
    /// because the success rate has fallen below the suppression floor.
    fn intervention_suppressed(&self, _agent_id: &str, source: InterventionSource) -> bool {
        self.sample_source_quality(source) < 0.2
    }

    /// Compose a corrective prompt for a detected fallacy.
    /// Corrections are tailored to the specific fallacy type when recognised.
    fn compose_fallacy_correction(&self, fallacy: &FallacyReport) -> String {
        let type_lower = fallacy.fallacy_type.to_lowercase();
        let specific = if type_lower.contains("circular") {
            "You are restating your conclusion as a premise. \
             Break the cycle: identify an independent warrant."
                .to_string()
        } else if type_lower.contains("authority") || type_lower.contains("appeal") {
            "Citation without analysis is not argumentation. \
             Explain WHY the authority's position applies here."
                .to_string()
        } else if type_lower.contains("dichotomy") || type_lower.contains("binary") {
            "You have presented a false binary. \
             Enumerate at least one additional option you excluded."
                .to_string()
        } else if type_lower.contains("straw") {
            "You are misrepresenting the opposing position. \
             Quote their actual claim before responding."
                .to_string()
        } else {
            format!(
                "Remove the fallacious reasoning and reconstruct \
                 your argument from valid premises only. \
                 (evidence: \"{}\", confidence: {:.0}%)",
                sanitize_directive_content(&fallacy.evidence_span),
                fallacy.confidence * 100.0
            )
        };

        format!(
            "METACOGNITIVE CORRECTION: Your argument contains a {} fallacy. {}",
            sanitize_directive_content(&fallacy.fallacy_type),
            specific
        )
    }

    /// Format all pending interventions into a prompt injection block.
    pub fn format_interventions(interventions: &[Intervention], target: &str) -> Option<String> {
        let relevant: Vec<&Intervention> = interventions
            .iter()
            .filter(|i| i.target_agent == target || i.target_agent == "System")
            .collect();
        if relevant.is_empty() {
            return None;
        }
        let mut block = String::from("\n[METACOGNITIVE OBSERVER]\n");
        for i in &relevant {
            let severity_tag = match i.severity {
                InterventionSeverity::Advisory => "ADVISORY",
                InterventionSeverity::Corrective => "CORRECTIVE",
                InterventionSeverity::Mandatory => "MANDATORY",
            };
            block.push_str(&format!("[{severity_tag}] {}\n", i.directive));
        }
        block.push_str("[/METACOGNITIVE OBSERVER]\n");
        Some(block)
    }

    /// Update the Beta-posterior calibration for `agent_id`.
    ///
    /// When the agent expressed high certainty (> 0.7) and the outcome was
    /// correct (`verified = true`), alpha is incremented.  When the agent
    /// expressed high certainty but the outcome was wrong, beta is incremented.
    /// Low-certainty turns are ignored: the agent did not commit, so there is
    /// nothing to calibrate against.
    pub fn update_calibration(&mut self, agent_id: &str, certainty: f64, verified: bool) {
        if certainty > 0.7 {
            let entry = self
                .calibration
                .entry(agent_id.to_string())
                .or_insert((1.0, 1.0));
            if verified {
                entry.0 += 1.0;
            } else {
                entry.1 += 1.0;
            }
        }
    }

    /// Returns the mean of the Beta posterior for `agent_id`: alpha/(alpha+beta).
    /// Defaults to 0.5 (maximum uncertainty) when no data has been collected.
    pub fn calibration_score(&self, agent_id: &str) -> f64 {
        self.calibration
            .get(agent_id)
            .map(|(alpha, beta)| alpha / (alpha + beta))
            .unwrap_or(0.5)
    }

    /// Elo rating update: winner gains, loser loses, magnitude scaled by
    /// the expected outcome (so upsets cause bigger swings) and amplified
    /// by the surprise value for the turn.
    fn update_elo(
        &mut self,
        agent_id: &str,
        outcome: TurnOutcome,
        surprise_val: f64,
        task_category: TaskCategory,
    ) {
        let k = 32.0; // standard Elo K-factor
        let rating = *self.elo_ratings.get(agent_id).unwrap_or(&1500.0);

        let score = match outcome {
            TurnOutcome::TestsPassed => 1.0,
            TurnOutcome::Compiled | TurnOutcome::AdvancedConvergence => 0.75,
            TurnOutcome::Unknown => 0.5,
            TurnOutcome::Stalled => 0.25,
            TurnOutcome::RolledBack | TurnOutcome::Rejected | TurnOutcome::VerificationFailed => {
                0.0
            }
        };

        // Update calibration: the turn's certainty is the epistemic state's
        // confidence for this agent; verified = outcome was not VerificationFailed.
        let certainty = self
            .agent_states
            .get(agent_id)
            .map(|s| s.confidence)
            .unwrap_or(0.5);
        let verified = outcome != TurnOutcome::VerificationFailed;
        self.update_calibration(agent_id, certainty, verified);

        // Compute expected score against the field average.
        let field_avg = if self.elo_ratings.is_empty() {
            1500.0
        } else {
            self.elo_ratings.values().sum::<f64>() / self.elo_ratings.len() as f64
        };
        let expected = 1.0 / (1.0 + 10f64.powf((field_avg - rating) / 400.0));

        // Surprise amplifier: unexpected outcomes (high surprise) swing Elo harder.
        let amplifier = 1.0 + surprise_val.clamp(0.0, 1.0);
        let new_rating = rating + k * amplifier * (score - expected);

        self.elo_ratings.insert(agent_id.to_string(), new_rating);

        // Update the per-category slot and recompute overall average.
        let slot: usize = match task_category {
            TaskCategory::CodeGeneration => 0,
            TaskCategory::Debugging => 1,
            TaskCategory::Architecture => 2,
            TaskCategory::Refactoring => 3,
            TaskCategory::Research => 4,
            TaskCategory::Testing => 5,
            TaskCategory::General => 6,
        };
        let slots = self
            .elo_by_category
            .entry(agent_id.to_string())
            .or_insert([1500.0; 7]);
        slots[slot] = new_rating;
        let avg = slots.iter().sum::<f64>() / 7.0;
        self.elo_ratings.insert(agent_id.to_string(), avg);
    }

    /// Return the category-specific Elo for `agent_id` (defaults to 1500.0 if unseen).
    pub fn elo_for_category(&self, agent_id: &str, category: TaskCategory) -> f64 {
        let slot: usize = match category {
            TaskCategory::CodeGeneration => 0,
            TaskCategory::Debugging => 1,
            TaskCategory::Architecture => 2,
            TaskCategory::Refactoring => 3,
            TaskCategory::Research => 4,
            TaskCategory::Testing => 5,
            TaskCategory::General => 6,
        };
        self.elo_by_category
            .get(agent_id)
            .map(|slots| slots[slot])
            .unwrap_or(1500.0)
    }

    /// Get agents sorted by Elo (highest first) for selection pressure.
    pub fn ranked_agents(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .elo_ratings
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Should this agent be "killed" (removed from the active pool)?
    /// Agents below the kill threshold after sufficient turns are eliminated.
    pub fn should_eliminate(&self, agent_id: &str, min_turns: u32) -> bool {
        let rating = *self.elo_ratings.get(agent_id).unwrap_or(&1500.0);
        let agent_turns = self
            .confidence_accum
            .get(agent_id)
            .map(|&(_, c)| c as u32)
            .unwrap_or(0);
        agent_turns >= min_turns && rating < 1200.0
    }

    /// Get the epistemic state for an agent (for prompt injection).
    pub fn epistemic_state(&self, agent_id: &str) -> Option<&EpistemicState> {
        self.agent_states.get(agent_id)
    }

    /// Snapshot of cumulative session metrics.
    pub fn metrics(&self) -> ObserverMetrics {
        self.metrics.clone()
    }

    /// Serialize Elo ratings to JSON for persistence.
    pub fn export_elo_ratings(&self) -> String {
        serde_json::to_string(
            &self
                .elo_ratings
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect::<Vec<_>>(),
        )
        .unwrap_or_default()
    }

    /// Load Elo ratings from a prior session.
    pub fn import_elo_ratings(&mut self, json: &str) {
        match serde_json::from_str::<Vec<(String, f64)>>(json) {
            Ok(ratings) => {
                for (agent_id, elo) in ratings {
                    self.elo_ratings.insert(agent_id, elo);
                }
            }
            Err(e) => tracing::warn!(err = %e, "failed to parse Elo ratings; starting fresh"),
        }
    }

    /// Get a summary of the observer's current state for diagnostics.
    pub fn diagnostic_summary(&self) -> ObserverDiagnostics {
        ObserverDiagnostics {
            agent_count: self.agent_states.len(),
            ranked: self.ranked_agents(),
            turns_since_progress: self.turns_since_progress,
            total_interventions: self.intervention_history.len(),
            agents_below_concession: self
                .agent_states
                .iter()
                .filter(|(_, s)| s.should_concede(self.concession_threshold))
                .count(),
        }
    }
}

impl Default for MetacognitiveObserver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ObserverDiagnostics {
    pub agent_count: usize,
    pub ranked: Vec<(String, f64)>,
    pub turns_since_progress: u32,
    pub total_interventions: usize,
    pub agents_below_concession: usize,
}

// =====================================================================
// UTILITY
// =====================================================================

/// Compute the Jaccard token overlap between two strings.
///
/// Tokens are whitespace-split lowercase words.  Returns a value in
/// [0.0, 1.0] where 1.0 means the two strings share exactly the same
/// vocabulary.  Used by refutation detection to ensure a negation keyword
/// is actually about the claim being checked, not about something else.
fn token_overlap(a: &str, b: &str) -> f64 {
    use std::collections::HashSet;
    let tokens_a: HashSet<&str> = a.split_whitespace().collect();
    let tokens_b: HashSet<&str> = b.split_whitespace().collect();
    if tokens_a.is_empty() && tokens_b.is_empty() {
        return 1.0;
    }
    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}
