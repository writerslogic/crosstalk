use crate::types::conversation::TurnOutcome;
use rustc_hash::FxHashMap;

pub struct SurpriseEngine {
    predictions: FxHashMap<String, Vec<f64>>,
    surprises: FxHashMap<String, Vec<f64>>,
}

impl SurpriseEngine {
    pub fn new() -> Self {
        Self {
            predictions: FxHashMap::default(),
            surprises: FxHashMap::default(),
        }
    }

    /// Record a predicted success probability for a model before execution.
    pub fn record_prediction(&mut self, model_id: &str, certainty: f64) {
        if !certainty.is_finite() || !(0.0..=1.0).contains(&certainty) {
            return;
        }
        self.predictions
            .entry(model_id.to_string())
            .or_default()
            .push(certainty);
    }

    /// Compute surprise after execution and store it. Returns the surprise value.
    pub fn compute_surprise(&mut self, model_id: &str, actual_outcome: TurnOutcome) -> f64 {
        let actual_score = match actual_outcome {
            TurnOutcome::Compiled | TurnOutcome::TestsPassed => 1.0,
            _ => 0.0,
        };

        let predicted = self
            .predictions
            .get(model_id)
            .and_then(|v| v.last().copied())
            .unwrap_or(0.5);

        let surprise = (actual_score - predicted).abs();
        self.surprises
            .entry(model_id.to_string())
            .or_default()
            .push(surprise);
        surprise
    }

    /// Adjust influence weight based on recent surprise history.
    /// - 3+ consecutive surprises > 0.5: decrease by 20%
    /// - 5+ consecutive surprises < 0.1: increase by 10%
    /// - Result clamped to [0.5, 2.0]
    pub fn calibrate_weight(&self, model_id: &str, current_weight: f64) -> f64 {
        let history = match self.surprises.get(model_id) {
            Some(h) if !h.is_empty() => h,
            _ => return current_weight,
        };

        let tail3: Vec<f64> = history.iter().rev().take(3).copied().collect();
        if tail3.len() == 3 && tail3.iter().all(|&s| s > 0.5) {
            return (current_weight * 0.8).clamp(0.5, 2.0);
        }

        let tail5: Vec<f64> = history.iter().rev().take(5).copied().collect();
        if tail5.len() == 5 && tail5.iter().all(|&s| s < 0.1) {
            return (current_weight * 1.1).clamp(0.5, 2.0);
        }

        current_weight.clamp(0.5, 2.0)
    }

    /// Return the surprise history for a model (for testing / inspection).
    pub fn surprise_history(&self, model_id: &str) -> &[f64] {
        self.surprises
            .get(model_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

impl Default for SurpriseEngine {
    fn default() -> Self {
        Self::new()
    }
}
