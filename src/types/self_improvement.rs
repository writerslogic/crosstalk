use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SessionEvaluation {
    pub session_id: String,
    pub metrics: HashMap<String, f64>,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisStatus {
    Queued,
    Testing,
    Adopted,
    Rejected,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImprovementHypothesis {
    pub id: String,
    pub description: String,
    pub expected_impact: f64,
    /// 0.0–1.0 confidence in the predicted effect.
    pub confidence: f64,
    /// Relative cost to run the test (arbitrary units).
    pub estimated_cost: f64,
    pub status: HypothesisStatus,
}

impl ImprovementHypothesis {
    /// Priority score used by `HypothesisPrioritizer`.
    #[must_use]
    pub fn priority(&self) -> f64 {
        let denom = self.estimated_cost.max(f64::EPSILON);
        (self.expected_impact * self.confidence) / denom
    }
}

// ── PromptTemplate ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptTemplate {
    pub id: String,
    pub version: u32,
    pub content: String,
    pub task_types: Vec<String>,
    /// `(session_id, quality_score)` pairs, most-recent last.
    pub performance_history: Vec<(String, f64)>,
}

impl PromptTemplate {
    #[must_use]
    pub fn mean_quality(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.performance_history.iter().map(|(_, q)| q).sum();
        sum / self.performance_history.len() as f64
    }
}

// ── StrategyEntry ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StrategyEntry {
    pub id: String,
    pub task_features: Vec<f64>,
    pub approach: String,
    pub steps: Vec<String>,
    pub outcome_quality: f64,
    pub sessions_used: u32,
}

impl StrategyEntry {
    /// Squared Euclidean distance to a query feature vector.
    #[must_use]
    pub fn distance_sq(&self, query: &[f64]) -> f64 {
        let len = self.task_features.len().min(query.len());
        self.task_features[..len]
            .iter()
            .zip(&query[..len])
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }
}

// ── PostMortem ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum FailureCause {
    TypeMismatch,
    MissingContext,
    AgentCapabilityLimit,
    ComplexityExceeded,
    InsufficientBudget,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PostMortem {
    pub session_id: String,
    pub failure_turn_indices: Vec<u32>,
    pub root_cause: FailureCause,
    pub missing_context: Vec<String>,
    pub alternative_approaches: Vec<String>,
}

// ── CalibrationRecord ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CalibrationRecord {
    pub session_id: String,
    pub predicted_difficulty: f64,
    pub actual_difficulty: f64,
    pub predicted_outcome: f64,
    pub actual_outcome: f64,
}

impl CalibrationRecord {
    #[must_use]
    pub fn difficulty_error(&self) -> f64 {
        (self.predicted_difficulty - self.actual_difficulty).abs()
    }

    #[must_use]
    pub fn outcome_error(&self) -> f64 {
        (self.predicted_outcome - self.actual_outcome).abs()
    }
}

// ── ErrorBudget ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementLevel {
    Relaxed,
    Normal,
    Strict,
    Suspended,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ErrorBudget {
    pub task_type: String,
    pub allowed_rate: f64,
    pub actual_rate: f64,
    pub budget_remaining: f64,
    pub enforcement_level: EnforcementLevel,
}

// ── BenchmarkTask ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkCategory {
    CodeGeneration,
    BugFixing,
    Refactoring,
    ArchitectureDesign,
    ResearchSynthesis,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkTask {
    pub id: String,
    pub category: BenchmarkCategory,
    pub input_spec: String,
    pub quality_rubric: Vec<String>,
    pub reference_solution: String,
    pub difficulty: f64,
}

// ── DegradationStrategy ───────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DegradationTrigger {
    TaskComplexityExceeded,
    BudgetExhausted,
    AllModelsFailing,
    ConvergenceImpossible,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DegradationResponse {
    Checkpoint,
    DocumentBlocker,
    SuggestHumanIntervention,
    AttemptSimplerSubGoal,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DegradationStrategy {
    pub trigger: DegradationTrigger,
    pub response: DegradationResponse,
}

// ── BenchmarkResult ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkResult {
    pub task_id: String,
    pub score: f64,
    pub timestamp: u64,
}

// ── ParameterAdjustment ───────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterAdjustment {
    pub parameter: String,
    pub old_value: f64,
    pub new_value: f64,
    pub rationale: String,
    pub applied_at: u64,
}

// ── LearningOutcome ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LearningOutcome {
    pub action: String,
    pub metric: String,
    pub before: f64,
    pub after: f64,
}

impl LearningOutcome {
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.after - self.before
    }
}

// ── ProgressReport ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProgressReport {
    pub session_id: String,
    pub turns_completed: u32,
    pub turns_expected: u32,
    pub completion_probability: f64,
    pub estimated_turns_remaining: Option<u32>,
    pub success_probability: f64,
}

// ── HandoffPackage ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HandoffPackage {
    pub session_id: String,
    pub trigger: String,
    pub failure_summary: String,
    pub hypotheses_tried: Vec<String>,
    pub last_successful_turn: Option<u32>,
    pub recommended_next_action: String,
    pub context_snapshot: String,
}
