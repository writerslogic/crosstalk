use crate::types::{ModelProfile, TaskCategory, Turn, TurnOutcome, ConversationState};
use std::collections::HashMap;

pub struct IntelligenceEngine {
    pub profiles: HashMap<String, ModelProfile>,
}

impl IntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    pub fn update_profile(&mut self, turn: &Turn, quality_score: f64) {
        let profile = self.profiles.entry(turn.model_id.clone()).or_insert(ModelProfile {
            model_id: turn.model_id.clone(),
            task_scores: HashMap::new(),
            total_turns: 0,
            last_updated: ConversationState::now(),
        });

        if let Some(cat) = turn.task_category {
            profile.task_scores.entry(cat).or_default().update(quality_score);
        }
        profile.total_turns += 1;
        profile.last_updated = ConversationState::now();
    }

    #[must_use]
    pub fn route_task(&self, category: TaskCategory, available_models: &[String]) -> String {
        let mut best_model = available_models[0].clone();
        let mut best_score = -1.0;

        for model_id in available_models {
            if let Some(profile) = self.profiles.get(model_id) {
                let score = profile.task_scores.get(&category).map(|ra| ra.mean).unwrap_or(0.5);
                if score > best_score {
                    best_score = score;
                    best_model = model_id.clone();
                }
            }
        }
        best_model
    }
}

pub struct QualityScorer;

impl QualityScorer {
    #[must_use]
    pub fn score(turn: &Turn) -> f64 {
        let mut score: f64 = 0.5;
        match turn.outcome {
            TurnOutcome::TestsPassed => score += 0.4,
            TurnOutcome::Compiled => score += 0.2,
            TurnOutcome::AdvancedConvergence => score += 0.1,
            TurnOutcome::RolledBack | TurnOutcome::Rejected => score -= 0.4,
            TurnOutcome::Stalled => score -= 0.1,
            TurnOutcome::Unknown => {}
        }
        score.clamp(0.0, 1.0)
    }
}

pub struct ContextBudgeter;

impl ContextBudgeter {
    #[must_use]
    pub fn allocate(available_tokens: usize, segments: Vec<(&str, usize)>) -> Vec<usize> {
        let total_weight: usize = segments.iter().map(|s| s.1).sum();
        if total_weight == 0 {
            let n = segments.len().max(1);
            return vec![available_tokens / n; segments.len()];
        }

        segments.iter().map(|s| (s.1 * available_tokens) / total_weight).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RunningAverage;

    #[test]
    fn test_running_average() {
        let mut ra = RunningAverage::default();
        ra.update(1.0);
        ra.update(0.0);
        assert!((ra.mean - 0.5).abs() < f64::EPSILON);
        assert_eq!(ra.count, 2);
    }

    #[test]
    fn test_quality_scorer() {
        let mut turn = Turn {
            index: 1,
            model_id: "test".to_string(),
            content: "code".to_string(),
            timestamp: 0,
            diffs: vec![],
            certainty: Some(1.0),
            outcome: TurnOutcome::TestsPassed,
            task_category: Some(TaskCategory::CodeGeneration),
        };
        assert!(QualityScorer::score(&turn) > 0.8);
        turn.outcome = TurnOutcome::Rejected;
        assert!(QualityScorer::score(&turn) < 0.2);
    }
}
