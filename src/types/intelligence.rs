use crate::types::conversation::TaskCategory;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    pub task_scores: BTreeMap<TaskCategory, RunningAverage>,
    pub total_turns: u32,
    pub last_updated: u64,
    #[serde(default)]
    pub latency_ms: RunningAverage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentProfile {
    pub model_id: String,
    pub capabilities: BTreeMap<TaskCategory, f64>,
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
    pub performance_history: Vec<(String, f64)>,
}

impl PromptTemplate {
    pub fn render(&self, vars: &BTreeMap<String, String>) -> Result<String> {
        let mut out = self.template_text.clone();
        for var in &self.variables {
            let placeholder = format!("{{{{{}}}}}", var);
            let value = vars.get(var.as_str())
                .ok_or_else(|| anyhow!("Missing template variable '{}'; available: [{}]", var, vars.keys().cloned().collect::<Vec<_>>().join(", ")))?;
            out = out.replace(&placeholder, value);
        }
        Ok(out)
    }

    pub fn is_corrective(&self) -> bool {
        self.id.contains("corrective")
    }

    pub fn category(&self) -> TaskCategory {
        self.task_category
    }
}

#[derive(Debug, Clone)]
pub enum MutationStrategy {
    /// Append a suffix (e.g. emphasis clause) to the template body.
    Append(String),
    /// Trim the template body to at most `max_chars` characters.
    Trim(usize),
    /// Prepend a prefix (e.g. role framing) before the template body.
    Prefix(String),
    /// Inject an `{{examples}}` slot if one is not already present.
    InjectExamples,
}

impl PromptTemplate {
    /// Produce a mutated copy of this template. The copy increments `version`
    /// and appends `_v{version}` to the id; the original is not modified.
    #[must_use]
    pub fn mutate(&self, strategy: MutationStrategy) -> PromptTemplate {
        let mut m = self.clone();
        m.version += 1;
        m.id = format!("{}_v{}", self.id, m.version);
        match strategy {
            MutationStrategy::Append(suffix) => {
                m.template_text.push_str("\n\n");
                m.template_text.push_str(&suffix);
            }
            MutationStrategy::Trim(max_chars) => {
                m.template_text = m.template_text.chars().take(max_chars).collect();
            }
            MutationStrategy::Prefix(prefix) => {
                m.template_text = format!("{}\n\n{}", prefix, m.template_text);
            }
            MutationStrategy::InjectExamples => {
                if !m.template_text.contains("{{examples}}") {
                    m.template_text.push_str("\n\nExamples:\n{{examples}}");
                    if !m.variables.contains(&"examples".to_string()) {
                        m.variables.push("examples".to_string());
                    }
                }
            }
        }
        m
    }

    /// Record a quality score for a specific turn/outcome.
    pub fn record_performance(&mut self, outcome_id: String, quality: f64) {
        self.performance_history.push((outcome_id, quality));
    }

    /// Mean quality across all recorded performance observations (0.5 if none).
    #[must_use]
    pub fn mean_performance(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.5;
        }
        self.performance_history.iter().map(|(_, q)| q).sum::<f64>()
            / self.performance_history.len() as f64
    }
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
