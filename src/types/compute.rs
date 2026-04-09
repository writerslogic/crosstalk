use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Spending tier that gates model selection and prompt verbosity.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum BudgetMode {
    /// More than 20 % of session budget remaining.
    Normal,
    /// Between 5 % and 20 % remaining; prefer cheaper models.
    CostReduction,
    /// Below 5 % remaining; cheapest path only.
    Emergency,
}

/// Token counts for a single API call.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Per-turn cost record stored in the [`BudgetLedger`].
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CostEntry {
    pub turn_id: u32,
    pub model_id: String,
    pub usage: TokenUsage,
    pub cost_usd: f64,
    pub latency_ms: u64,
    pub timestamp: u64,
}

/// Tracks API spend across a session and derives the current [`BudgetMode`].
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BudgetLedger {
    pub session_budget: f64,
    pub spent: f64,
    pub entries: Vec<CostEntry>,
}

impl BudgetLedger {
    #[must_use]
    pub fn remaining(&self) -> f64 {
        (self.session_budget - self.spent).max(0.0)
    }

    #[must_use]
    pub fn burn_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        self.spent / self.entries.len() as f64
    }

    #[must_use]
    pub fn burn_rate_defined(&self) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        Some(self.spent / self.entries.len() as f64)
    }

    #[must_use]
    pub fn mode(&self) -> BudgetMode {
        let pct = if self.session_budget > f64::EPSILON {
            self.remaining() / self.session_budget
        } else {
            0.0
        };
        if pct < 0.05 {
            BudgetMode::Emergency
        } else if pct < 0.20 {
            BudgetMode::CostReduction
        } else {
            BudgetMode::Normal
        }
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "budget={:.4} spent={:.4} remaining={:.4} burn_rate={:.6} mode={:?}",
            self.session_budget,
            self.spent,
            self.remaining(),
            self.burn_rate(),
            self.mode()
        )
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ModelCapabilityMatrix {
    pub scores: BTreeMap<String, BTreeMap<String, f64>>,
}

impl ModelCapabilityMatrix {
    #[must_use]
    pub fn score(&self, model_id: &str, capability: &str) -> f64 {
        self.scores
            .get(model_id)
            .and_then(|caps| caps.get(capability))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn register(&mut self, model_id: &str, capability: &str, score: f64) {
        self.scores
            .entry(model_id.to_string())
            .or_default()
            .insert(capability.to_string(), score.clamp(0.0, 1.0));
    }
}
