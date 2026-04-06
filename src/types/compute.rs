use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CostEntry {
    pub turn_id: u32,
    pub model_id: String,
    pub usage: TokenUsage,
    pub cost_usd: f64,
    pub latency_ms: u64,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BudgetLedger {
    pub session_budget: f64,
    pub spent: f64,
    pub entries: Vec<CostEntry>,
}
