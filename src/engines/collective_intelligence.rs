use crate::types::conversation::{Turn, TurnOutcome};
use crate::types::intelligence::AgentProfile;
use crate::types::memory::TransferableLesson;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct CollectiveIntelligenceEngine {
    pub profiles: HashMap<String, AgentProfile>,
}

impl CollectiveIntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
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
    pub fn merge_proposals(proposals: Vec<(String, String, f64)>) -> String {
        if proposals.is_empty() {
            return String::new();
        }
        // Heuristic: pick the highest quality proposal as the base
        let mut best = &proposals[0];
        for p in &proposals {
            if p.2 > best.2 {
                best = p;
            }
        }
        best.1.clone()
    }
}
