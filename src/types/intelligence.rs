use crate::types::conversation::TaskCategory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RunningAverage {
    pub mean: f64,
    pub count: u32,
    pub variance: f64, // Added for regression detection
}

impl RunningAverage {
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / f64::from(self.count);
        let delta2 = value - self.mean;
        self.variance += delta * delta2;
    }

    pub fn stddev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.variance / f64::from(self.count - 1)).sqrt()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelProfile {
    pub model_id: String,
    pub task_scores: HashMap<TaskCategory, RunningAverage>,
    pub total_turns: u32,
    pub last_updated: u64,
    #[serde(default)]
    pub latency_ms: RunningAverage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentProfile {
    pub model_id: String,
    pub capabilities: HashMap<TaskCategory, f64>,
    pub total_turns: u32,
    pub compilation_success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptTemplate {
    pub id: String,
    pub version: u32,
    pub template_text: String,
    pub task_category: TaskCategory,
    pub variables: Vec<String>,
    pub performance_history: Vec<(String, f64)>, // (session_id, quality_score)
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct RegressionAlert {
    pub agent_id: String,
    pub task_category: TaskCategory,
    pub baseline_mean: f64,
    pub recent_mean: f64,
    pub severity: f64,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FailurePattern {
    pub pattern_id: String,
    pub error_type: String,
    pub context_signature: Vec<f32>,
    pub agent_id: String,
    pub frequency: u32,
    pub last_seen: u64,
}
