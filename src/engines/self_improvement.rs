use crate::types::conversation::ConversationState;
use crate::types::self_improvement::{ImprovementHypothesis, SessionEvaluation};
use anyhow::{Result, anyhow};
use std::collections::HashMap;

pub struct SelfImprovementEngine;

impl SelfImprovementEngine {
    #[must_use]
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_tests: HashMap::new(),
        }
    }

    pub fn check_significance(control: &[f64], test: &[f64]) -> bool {
        if control.len() < 5 || test.len() < 5 {
            return false;
        }
        let mean_c = control.iter().sum::<f64>() / control.len() as f64;
        let mean_t = test.iter().sum::<f64>() / test.len() as f64;
        let diff = (mean_t - mean_c).abs();
        diff > 0.1 * mean_c
    }
}

impl Default for AbTestManager {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SafetyInterlock;

impl SafetyInterlock {
    #[must_use]
    pub fn is_modification_allowed(file_path: &str) -> bool {
        let protected = [
            "security.rs",
            "verification.rs",
            "self_improvement.rs",
            "orchestrator.rs",
        ];
        for p in protected {
            if file_path.contains(p) {
                return false;
            }
        }
        true
    }
}

pub struct SelfCodeModifier;

impl SelfCodeModifier {
    pub fn propose_improvement(_file_path: &str, current_content: &str) -> Result<String> {
        if current_content.contains(".collect::<Vec<_>>().iter()") {
            return Ok(current_content.replace(".collect::<Vec<_>>().iter()", ".iter()"));
        }
        Err(anyhow!("No improvements found"))
    }
}
