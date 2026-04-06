use crate::types::conversation::{Turn, TurnOutcome};
use crate::types::intelligence::AgentProfile;
use crate::types::memory::TransferableLesson;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Default)]
pub struct CollectiveIntelligenceEngine {
    pub profiles: HashMap<String, AgentProfile>,
}

impl CollectiveIntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_specialization(&mut self, turn: &Turn) {
        let profile = self
            .profiles
            .entry(turn.model_id.clone())
            .or_insert(AgentProfile {
                model_id: turn.model_id.clone(),
                capabilities: HashMap::new(),
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
        let mut correctness = 0.7;

        if proposal.contains("TODO") || proposal.contains("FIXME") {
            correctness -= 0.2;
            comments.push("Incomplete implementation detected (TODO found)".to_string());
        }

        PeerReviewReport {
            reviewer_id: reviewer_id.to_string(),
            correctness,
            efficiency: 0.8,
            maintainability: 0.8,
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
        profiles: &HashMap<String, AgentProfile>,
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

#[derive(Debug, Clone, Default)]
pub struct StrategyOutcome {
    pub strategy_name: String,
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
    pub outcomes: HashMap<String, StrategyOutcome>,
}

impl MetaStrategyOptimizer {
    pub fn new() -> Self {
        Self { outcomes: HashMap::new() }
    }

    pub fn record(&mut self, strategy: &str, quality: f64) {
        let e = self.outcomes.entry(strategy.to_string()).or_default();
        e.strategy_name = strategy.to_string();
        e.quality_sum += quality;
        e.trial_count += 1;
    }

    #[must_use]
    pub fn best_strategy(&self) -> Option<&str> {
        self.outcomes
            .values()
            .filter(|o| o.trial_count >= 3)
            .max_by(|a, b| a.avg_quality().total_cmp(&b.avg_quality()))
            .map(|o| o.strategy_name.as_str())
    }
}

impl Default for MetaStrategyOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
