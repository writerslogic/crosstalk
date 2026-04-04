use crate::types::{AgentProfile, TaskCategory, Turn, TurnOutcome, TransferableLesson, ArtifactDiff};
use std::collections::HashMap;

pub struct CollectiveIntelligenceEngine {
    pub profiles: HashMap<String, AgentProfile>,
}

impl CollectiveIntelligenceEngine {
    pub fn new() -> Self {
        Self { profiles: HashMap::new() }
    }

    pub fn update_specialization(&mut self, turn: &Turn) {
        let profile = self.profiles.entry(turn.model_id.clone()).or_insert(AgentProfile {
            model_id: turn.model_id.clone(),
            capabilities: HashMap::new(),
            total_turns: 0,
            compilation_success_rate: 0.0,
        });

        if let Some(cat) = turn.task_category {
            let score = match turn.outcome {
                TurnOutcome::TestsPassed => 1.0,
                TurnOutcome::Compiled => 0.8,
                TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.0,
                _ => 0.5,
            };
            let current = profile.capabilities.entry(cat).or_insert(0.5);
            *current = (*current * 0.9) + (score * 0.1);
        }
        profile.total_turns += 1;
    }
}

pub struct KnowledgeTransfer;

impl KnowledgeTransfer {
    pub fn pack_lesson(turn: &Turn) -> Option<TransferableLesson> {
        if turn.outcome == TurnOutcome::TestsPassed {
            return Some(TransferableLesson {
                category: "success_pattern".to_string(),
                content: format!("Pattern discovered by {}: {}", turn.model_id, turn.content),
                confidence: 0.9,
            });
        }
        None
    }
}

pub struct PeerReview;

impl PeerReview {
    pub fn review(reviewer_id: &str, proposal: &str) -> String {
        format!("Review by {}: Proposal quality assessment logic here.", reviewer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialization_update() {
        let mut engine = CollectiveIntelligenceEngine::new();
        let turn = Turn {
            index: 1,
            model_id: "m1".into(),
            content: "c".into(),
            timestamp: 0,
            diffs: vec![],
            certainty: Some(1.0),
            outcome: TurnOutcome::TestsPassed,
            task_category: Some(TaskCategory::CodeGeneration),
            structure: None,
            signature: vec![],
        };
        engine.update_specialization(&turn);
        let profile = engine.profiles.get("m1").unwrap();
        assert!(profile.capabilities.get(&TaskCategory::CodeGeneration).unwrap() > &0.5);
    }
}
