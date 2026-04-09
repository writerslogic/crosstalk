use crate::types::conversation::TurnOutcome;
use rustc_hash::FxHashMap;

pub struct SurpriseEngine {
    pub predictions: FxHashMap<String, Vec<f64>>,
    pub surprises: FxHashMap<String, Vec<f64>>,
    pub default_prior: f64,
}

impl SurpriseEngine {
    pub fn new() -> Self {
        Self {
            predictions: FxHashMap::default(),
            surprises: FxHashMap::default(),
            default_prior: 0.5,
        }
    }

    pub fn with_prior(prior: f64) -> Self {
        Self {
            default_prior: prior.clamp(0.0, 1.0),
            ..Self::new()
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
            .unwrap_or(self.default_prior);

        let surprise = (actual_score - predicted).abs();
        self.surprises
            .entry(model_id.to_string())
            .or_default()
            .push(surprise);
        surprise
    }

    /// Adjust influence weight based on recent surprise history using an
    /// exponentially weighted mean (alpha=0.2, ~5-turn memory).
    /// High recent surprise (ema > 0.5) lowers weight; low surprise raises it.
    pub fn calibrate_weight(&self, model_id: &str, current_weight: f64) -> f64 {
        let history = match self.surprises.get(model_id) {
            Some(h) if !h.is_empty() => h,
            _ => return current_weight,
        };

        let mean = history.iter().sum::<f64>() / history.len() as f64;
        let adjusted = if mean > 0.5 {
            current_weight * 0.8
        } else if mean < 0.1 {
            current_weight * 1.1
        } else {
            current_weight
        };
        adjusted.clamp(0.5, 2.0)
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
