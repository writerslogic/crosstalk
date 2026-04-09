use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvergenceDiagnostic {
    pub velocity: f64,
    pub delta_trend: f64,
    pub quality_trend: f64,
    pub blockers: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentPerformanceReport {
    pub agent_id: String,
    pub success_rate: f64,
    pub avg_quality: f64,
    pub cost_per_turn: f64,
    pub improvement_slope: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnalyticsReport {
    pub session_id: String,
    pub convergence: ConvergenceDiagnostic,
    pub agent_performances: Vec<AgentPerformanceReport>,
    pub recommendations: Vec<Recommendation>,
    pub timestamp: u64,
}

impl AnalyticsReport {
    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow::anyhow!(e))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum QualityTrend {
    Improving,
    Plateau,
    Regressing,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Recommendation {
    pub action: String,
    pub expected_impact: f64,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetaLearningInsight {
    pub session_count: usize,
    pub avg_turns_to_convergence: f64,
    pub quality_growth_rate: f64,
    pub best_model: Option<String>,
}
