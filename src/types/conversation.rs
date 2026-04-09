use crate::types::artifact::{Artifact, ArtifactDiff};
use crate::types::compute::BudgetLedger;
use crate::types::planning::GoalTree;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// The observable result of a single model turn.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TurnOutcome {
    /// The generated artifact compiled successfully.
    Compiled,
    /// All tests passed after applying the turn's changes.
    TestsPassed,
    /// The turn moved the session closer to the goal.
    AdvancedConvergence,
    /// The turn was reverted; prior state restored.
    RolledBack,
    /// The turn was rejected by the consensus engine.
    Rejected,
    /// No meaningful progress was made.
    Stalled,
    /// Outcome has not been evaluated yet.
    Unknown,
}

/// The prompt layout strategy used when generating a turn.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TurnStructure {
    /// Unconstrained free-form output.
    FreeForm,
    /// Numbered step-by-step reasoning.
    StepByStep,
    /// Explicit pros/cons enumeration.
    ProsCons,
    /// State a hypothesis then validate it.
    HypothesisTest,
    /// Lead with code, follow with explanation.
    CodeFirst,
    /// Symbolic logic and mathematical notation.
    Symbolic,
}

/// High-level category used for routing, analytics, and prompt selection.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TaskCategory {
    CodeGeneration,
    Debugging,
    Architecture,
    Refactoring,
    Research,
    Testing,
}

/// A single model response within a session, including its diff, metadata, and
/// cryptographic signature.
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
    #[serde(default)]
    pub surprise_signal: Option<f64>,
}

fn default_outcome() -> TurnOutcome {
    TurnOutcome::Unknown
}

/// Full mutable state of a running session, persisted to Sled on every checkpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationState {
    pub session_id: String,
    pub iteration_index: u32,
    pub turns: Vec<Turn>,
    pub artifacts: BTreeMap<String, Arc<Artifact>>,
    #[serde(default)]
    pub agent_weights: BTreeMap<String, f64>,
    #[serde(default)]
    pub completion_probability: f64,
    #[serde(default)]
    pub state_hash: [u8; 32],
    #[serde(default)]
    pub budget: BudgetLedger,
    #[serde(default)]
    pub goal_tree: GoalTree,
    #[serde(default)]
    pub node_consensus: BTreeMap<String, f64>,
}

impl ConversationState {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            iteration_index: 0,
            turns: vec![],
            artifacts: BTreeMap::new(),
            agent_weights: BTreeMap::new(),
            completion_probability: 0.0,
            state_hash: [0u8; 32],
            budget: BudgetLedger::default(),
            goal_tree: GoalTree::default(),
            node_consensus: BTreeMap::new(),
        }
    }

    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
