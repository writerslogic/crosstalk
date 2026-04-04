use crate::types::{SessionEvaluation, ImprovementHypothesis, ConversationState};
use std::collections::HashMap;
use anyhow::{Result, anyhow};

pub struct SelfImprovementEngine;

impl SelfImprovementEngine {
    pub fn evaluate_session(sigma: &ConversationState) -> SessionEvaluation {
        let mut metrics = HashMap::new();
        metrics.insert("turn_count".to_string(), sigma.turns.len() as f64);
        metrics.insert("convergence_p".to_string(), sigma.completion_probability);
        metrics.insert("cost_spent".to_string(), sigma.budget.spent);
        
        SessionEvaluation {
            session_id: sigma.session_id.clone(),
            metrics,
            timestamp: ConversationState::now(),
        }
    }
}

pub struct AbTestManager {
    pub active_tests: HashMap<String, ImprovementHypothesis>,
}

impl AbTestManager {
    pub fn new() -> Self {
        Self { active_tests: HashMap::new() }
    }

    pub fn enroll_session(&self, _session_id: &str) -> Option<String> {
        // Deterministic split for A/B testing
        None
    }
}

pub struct SafetyInterlock;

impl SafetyInterlock {
    pub fn is_modification_allowed(file_path: &str) -> bool {
        let protected = ["src/security.rs", "src/verification.rs", "src/self_improvement.rs"];
        for p in protected {
            if file_path.contains(p) { return false; }
        }
        true
    }
}

pub struct SelfCodeModifier;

impl SelfCodeModifier {
    pub fn propose_improvement(_file_path: &str, _current_content: &str) -> Result<String> {
        // Real implementation would use an LLM turn to generate a diff.
        Err(anyhow!("No improvements found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_interlock() {
        assert!(!SafetyInterlock::is_modification_allowed("src/security.rs"));
        assert!(SafetyInterlock::is_modification_allowed("src/diff.rs"));
    }

    #[test]
    fn test_session_evaluation() {
        let sigma = ConversationState::new("test");
        let eval = SelfImprovementEngine::evaluate_session(&sigma);
        assert_eq!(eval.session_id, "test");
        assert!(eval.metrics.contains_key("turn_count"));
    }
}
