//! Metacognitive Observer — the executive function of the swarm.
//!
//! Monitors debate telemetry in real-time, detects epistemic failures,
//! injects corrective directives, and drives agent evolution through
//! Bayesian confidence tracking and Elo-based selection pressure.

use crate::engines::memory::{cosine_sim, local_embed_text};
use crate::engines::reasoning::FallacyDetector;
use crate::engines::surprise::SurpriseEngine;
use crate::types::conversation::{Turn, TurnOutcome};
use crate::types::security::FallacyReport;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

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
    Empirical,    // backed by data/code/tests
    Theoretical,  // backed by reasoning
    Heuristic,    // rules of thumb
    Ungrounded,   // no stated basis
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

#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Per-agent Elo ratings for selection pressure.
    pub elo_ratings: FxHashMap<String, f64>,
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
            calibration: std::collections::HashMap::new(),
            intervention_history: VecDeque::with_capacity(50),
            concession_threshold: 0.20,
            semantic_embeddings: VecDeque::with_capacity(20),
            turns_since_progress: 0,
            deadlock_threshold: 5,
            intervention_outcomes: FxHashMap::default(),
            metrics: ObserverMetrics::default(),
            confidence_accum: FxHashMap::default(),
        }
    }

    /// Analyze a completed turn and produce any necessary interventions.
    /// This is the main entry point called after each agent response.
    pub fn observe_turn(
        &mut self,
        turn: &Turn,
        all_recent_turns: &[Turn],
        surprise: &mut SurpriseEngine,
    ) -> Vec<Intervention> {
        let mut interventions = Vec::new();

        // 1. Extract or update epistemic state
        let epistemic = EpistemicState::extract_from_content(&turn.content);
        self.agent_states
            .insert(turn.model_id.clone(), epistemic.clone());

        // Update running confidence average for metrics.
        let (sum, count) = self
            .confidence_accum
            .entry(turn.model_id.clone())
            .or_insert((0.0, 0));
        *sum += epistemic.confidence;
        *count += 1;
        self.metrics
            .avg_confidence_by_agent
            .insert(turn.model_id.clone(), *sum / *count as f64);

        // 2. Confidence decay: reduce confidence by a fixed rate per turn for
        //    agents that provided no fresh evidence this turn.
        self.decay_confidence(0.02, &FxHashMap::default());

        // 3. Fallacy detection with corrective injection
        let fallacies = FallacyDetector::scan(&turn.content);
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

        // 4. Epistemic collapse detection
        if epistemic.should_concede(self.concession_threshold)
            && !self.intervention_suppressed(&turn.model_id, InterventionSource::EpistemicCollapse)
        {
            interventions.push(Intervention {
                target_agent: turn.model_id.clone(),
                directive: format!(
                    "Your epistemic confidence has dropped to {:.0}%. \
                     You must publicly concede the points where your assumptions were defeated: {:?}. \
                     Refocus on your remaining strong positions.",
                    epistemic.confidence * 100.0,
                    epistemic.defeated
                ),
                severity: InterventionSeverity::Mandatory,
                source: InterventionSource::EpistemicCollapse,
            });
            self.metrics.total_concessions_forced += 1;
        }

        // 5. Cross-agent assumption defeat: check if this turn's content
        //    directly contradicts another agent's stated assumptions.
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
        for (other_id, other_state) in &mut self.agent_states {
            if *other_id == turn.model_id {
                continue;
            }
            let defeated: Vec<String> = other_state
                .assumptions
                .iter()
                .filter(|a| {
                    if a.confidence <= 0.0 {
                        return false;
                    }
                    let claim_embedding = local_embed_text(&a.claim);
                    let sim = cosine_sim(&claim_embedding, &defeater_embedding);
                    sim > 0.75 && (has_refutation || has_negation)
                })
                .map(|a| a.claim.clone())
                .collect();
            let n = defeated.len() as u64;
            for claim in &defeated {
                other_state.defeat_assumption(claim);
            }
            self.metrics.total_assumptions_defeated += n;
        }

        // 6. Stale repetition detection via cosine similarity on embeddings.
        let embedding = local_embed_text(&turn.content);
        let is_repetitive = self
            .semantic_embeddings
            .iter()
            .any(|(_, prev)| cosine_sim(prev, &embedding) > 0.82)
            || self
                .semantic_embeddings
                .iter()
                .filter(|(id, prev)| {
                    id == &turn.model_id && cosine_sim(prev, &embedding) > 0.75
                })
                .count()
                >= 3;
        self.semantic_embeddings
            .push_back((turn.model_id.clone(), embedding));
        if self.semantic_embeddings.len() > 20 {
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

        // 7. Surprise-calibrated Elo update.
        let surprise_val = surprise.compute_surprise(&turn.model_id, turn.outcome);
        self.update_elo(&turn.model_id, turn.outcome, all_recent_turns, surprise_val);

        // 8. Progress tracking for topology shift.
        let made_progress = matches!(
            turn.outcome,
            TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
        );
        if made_progress {
            self.turns_since_progress = 0;
        } else {
            self.turns_since_progress += 1;
        }

        // 9. Topology shift recommendation.
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

        // Rate-limit: suppress duplicate interventions for the same agent+source
        // within the last 3 entries in history.
        interventions.retain(|i| {
            let key = (i.target_agent.clone(), i.source);
            let dominated = self
                .intervention_history
                .iter()
                .rev()
                .take(3)
                .any(|(a, s)| *a == key.0 && *s == key.1);
            if !dominated {
                self.intervention_history.push_back(key);
                if self.intervention_history.len() > 50 {
                    self.intervention_history.pop_front();
                }
                self.metrics.total_interventions_issued += 1;
                true
            } else {
                false
            }
        });

        interventions
    }

    /// Apply a per-turn confidence decay to all agents that provided no fresh
    /// evidence this turn.  Agents with evidence in their current state are
    /// exempt, as are agents with recent quality > 0.6.  Confidence is
    /// floor-clamped at 0.0.
    pub fn decay_confidence(&mut self, rate: f64, recent_quality: &FxHashMap<String, f64>) {
        for (id, state) in self.agent_states.iter_mut() {
            if !state.evidence.is_empty() {
                continue;
            }
            if recent_quality.get(id).map_or(false, |q| *q > 0.6) {
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
        self.intervention_outcomes
            .entry(agent_id.to_string())
            .or_default()
            .push((source, improved));
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
    fn intervention_suppressed(&self, agent_id: &str, _source: InterventionSource) -> bool {
        self.intervention_success_rate(agent_id) < 0.3
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
                fallacy.evidence_span,
                fallacy.confidence * 100.0
            )
        };

        format!(
            "METACOGNITIVE CORRECTION: Your argument contains a {} fallacy. {}",
            fallacy.fallacy_type, specific
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
            let entry = self.calibration
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
        _recent: &[Turn],
        surprise_val: f64,
    ) {
        let k = 32.0; // standard Elo K-factor
        let rating = *self.elo_ratings.get(agent_id).unwrap_or(&1500.0);

        let score = match outcome {
            TurnOutcome::TestsPassed => 1.0,
            TurnOutcome::Compiled | TurnOutcome::AdvancedConvergence => 0.75,
            TurnOutcome::Unknown => 0.5,
            TurnOutcome::Stalled => 0.25,
            TurnOutcome::RolledBack | TurnOutcome::Rejected | TurnOutcome::VerificationFailed => 0.0,
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
    }

    /// Get agents sorted by Elo (highest first) for selection pressure.
    pub fn ranked_agents(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self.elo_ratings.iter().map(|(k, v)| (k.clone(), *v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Should this agent be "killed" (removed from the active pool)?
    /// Agents below the kill threshold after sufficient turns are eliminated.
    pub fn should_eliminate(&self, agent_id: &str, min_turns: u32) -> bool {
        let rating = *self.elo_ratings.get(agent_id).unwrap_or(&1500.0);
        let turns: u32 = self.elo_ratings.len() as u32 * min_turns;
        turns >= min_turns * 3 && rating < 1200.0
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
            &self.elo_ratings.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>(),
        )
        .unwrap_or_default()
    }

    /// Load Elo ratings from a prior session.
    pub fn import_elo_ratings(&mut self, json: &str) {
        if let Ok(ratings) = serde_json::from_str::<Vec<(String, f64)>>(json) {
            for (agent_id, elo) in ratings {
                self.elo_ratings.insert(agent_id, elo);
            }
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
