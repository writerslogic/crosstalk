use crate::types::artifact::{Artifact, ArtifactDiff};
use crate::types::compute::BudgetLedger;
use crate::types::planning::GoalTree;
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum TurnStructure {
    FreeForm,
    StepByStep,
    ProsCons,
    HypothesisTest,
    CodeFirst,
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
    #[serde(default)]
    pub node_consensus: HashMap<String, f64>,
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
            node_consensus: HashMap::new(),
        }
    }

    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
