use crate::types::conversation::ConversationState;
use crate::types::self_improvement::{ImprovementHypothesis, SessionEvaluation};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;

/// Engine responsible for extracting metrics from a conversation session.
pub struct SelfImprovementEngine;

impl SelfImprovementEngine {
    #[must_use]
    pub fn evaluate_session(sigma: &ConversationState) -> SessionEvaluation {
        // Pre-allocate capacity to prevent reallocation overhead
        let mut metrics = HashMap::with_capacity(3);
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

/// Manages active A/B tests and evaluates statistical significance.
#[derive(Debug, Default, Clone)]
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

    /// Evaluates if the test group shows a statistically significant improvement
    /// using a simplified Welch's t-test approximation.
    #[must_use]
    pub fn check_significance(control: &[f64], test: &[f64]) -> bool {
        let n_c = control.len() as f64;
        let n_t = test.len() as f64;

        // Require a statistically relevant sample size (N >= 30 is standard rule of thumb)
        if n_c < 30.0 || n_t < 30.0 {
            return false;
        }

        let mean_c = control.iter().sum::<f64>() / n_c;
        let mean_t = test.iter().sum::<f64>() / n_t;

        // Calculate variance
        let var_c = control.iter().map(|&x| (x - mean_c).powi(2)).sum::<f64>() / (n_c - 1.0);
        let var_t = test.iter().map(|&x| (x - mean_t).powi(2)).sum::<f64>() / (n_t - 1.0);

        // Welch's t-test statistic calculation
        let t_stat = (mean_t - mean_c) / ((var_c / n_c) + (var_t / n_t)).sqrt();

        // Approximate threshold for 95% confidence (t > 1.96)
        t_stat > 1.96
    }
}

/// Prevents the engine from modifying core security and operational logic.
pub struct SafetyInterlock;

impl SafetyInterlock {
    // Use a compile-time array for protected files
    const PROTECTED_FILES: &'static [&'static str] = &[
        "security.rs",
        "verification.rs",
        "self_improvement.rs",
        "orchestrator.rs",
    ];

    #[must_use]
    pub fn is_modification_allowed(file_path: &str) -> bool {
        let path = Path::new(file_path);
        
        // Extract strictly the file name to prevent path traversal bypasses
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            return false; // Fail-secure: if we can't parse the filename, block it
        };

        !Self::PROTECTED_FILES.contains(&file_name)
    }
}

/// Proposes code improvements based on static analysis.
pub struct SelfCodeModifier;

impl SelfCodeModifier {
    /// Evaluates file content and proposes modifications.
    pub fn propose_improvement(file_path: &str, current_content: &str) -> Result<String> {
        // Enforce safety interlock before performing any string analysis
        if !SafetyInterlock::is_modification_allowed(file_path) {
            return Err(anyhow!("Modification rejected: {} is a protected file", file_path));
        }

        let target_pattern = ".collect::<Vec<_>>().iter()";
        let replacement = ".iter()";

        if current_content.contains(target_pattern) {
            Ok(current_content.replace(target_pattern, replacement))
        } else {
            Err(anyhow!("No sub-optimal code patterns identified in {}", file_path))
        }
    }
}