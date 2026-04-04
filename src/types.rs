use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    Compiled,
    TestsPassed,
    AdvancedConvergence,
    RolledBack,
    Rejected,
    Stalled,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Lesson {
    pub id: String,
    pub category: String,
    pub description: String,
    pub evidence_turn_ids: Vec<u32>,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FailureSignature {
    pub error_type: String,
    pub context_hash: String,
    pub agent_id: String,
    pub occurrence_count: u32,
    pub context_embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryRecord {
    pub turn_id: u32,
    pub session_id: String,
    pub embedding: Vec<f32>,
    pub content_hash: String,
    pub timestamp: u64,
    pub metadata_json: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RunningAverage {
    pub mean: f64,
    pub count: u32,
}

impl RunningAverage {
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.mean += (value - self.mean) / f64::from(self.count);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskCategory {
    CodeGeneration,
    Debugging,
    Architecture,
    Refactoring,
    Research,
    Testing,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelProfile {
    pub model_id: String,
    pub task_scores: HashMap<TaskCategory, RunningAverage>,
    pub total_turns: u32,
    pub last_updated: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum TurnStructure {
    FreeForm,
    StepByStep,
    ProsCons,
    HypothesisTest,
    CodeFirst,
}

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
    pub status: HypothesisStatus,
}

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

/// μ_n: An atomic turn in the debate
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Turn {
    pub index: u32,
    pub model_id: String,
    pub content: String,
    pub timestamp: u64,
    pub diffs: Vec<(String, ArtifactDiff)>,
    #[serde(default)]
    pub certainty: Option<f64>,
    #[serde(default = "default_outcome")]
    pub outcome: TurnOutcome,
    pub task_category: Option<TaskCategory>,
    pub structure: Option<TurnStructure>,
    #[serde(default)]
    pub signature: Vec<u8>,
}

fn default_outcome() -> TurnOutcome {
    TurnOutcome::Unknown
}

/// Δα: Represents a change to an artifact
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArtifactDiff {
    pub original_version: u32,
    pub new_version: u32,
    pub diff_text: String,
}

use crate::quality::ArtifactMetrics;

/// α: A project artifact (code, docs, research)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Artifact {
    pub name: String,
    pub language: String, // e.g., "rust"
    pub content: String,
    pub version: u32,
    pub history: Vec<ArtifactDiff>,
    #[serde(default)]
    pub ast_versions: HashMap<String, Vec<(u32, String)>>,
    #[serde(default)]
    pub proof_attachments: Vec<ProofAttachment>,
    #[serde(default)]
    pub metrics: ArtifactMetrics,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProofAttachment {
    pub artifact_name: String,
    pub proven_properties: Vec<String>,
    pub proof_hash: String,
    pub verified_at: u64,
}

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

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    Pending,
    InProgress,
    Complete,
    Blocked,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoalNode {
    pub id: String,
    pub title: String,
    pub children: Vec<GoalNode>,
    pub status: GoalStatus,
    pub assigned_turn: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GoalTree {
    pub root: Option<GoalNode>,
}

/// σ: The Global State
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationState {
    pub session_id: String,
    pub iteration_index: u32,
    pub turns: Vec<Turn>,
    pub artifacts: HashMap<String, Artifact>,
    #[serde(default)]
    pub agent_weights: HashMap<String, f64>,
    #[serde(default)]
    pub completion_probability: f64,
    #[serde(default)]
    pub state_hash: [u8; 32],
    #[serde(default)]
    pub budget: BudgetLedger,
    #[serde(default)]
    pub goal_tree: GoalTree,
}

/// Events emitted by the Orchestrator to the UI
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TokenReceived(String),
    TurnComplete(Turn),
    CheckpointWritten(u32),
    Error(String),
}

/// Control signals from the UI to the Orchestrator
#[derive(Debug, Clone)]
pub enum ControlSignal {
    Pause,
    Resume,
    Rewind(u32),
    Shutdown,
    Inject(String),
}

impl ConversationState {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            iteration_index: 0,
            turns: vec![],
            artifacts: HashMap::new(),
            agent_weights: HashMap::new(),
            completion_probability: 0.0,
            state_hash: [0u8; 32],
            budget: BudgetLedger::default(),
            goal_tree: GoalTree::default(),
        }
    }

    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
