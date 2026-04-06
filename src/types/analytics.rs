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
    pub timestamp: u64,
}
