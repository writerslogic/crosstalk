use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// μ_n: An atomic turn in the debate
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Turn {
    pub index: u32,
    pub model_id: String,
    pub content: String,
    pub timestamp: u64,
    pub diffs: Vec<(String, ArtifactDiff)>,
    #[serde(default)]
    pub certainty: Option<f64>, // [0.0, 1.0] confidence score
}

/// Δα: Represents a change to an artifact
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArtifactDiff {
    pub original_version: u32,
    pub new_version: u32,
    pub diff_text: String, // Standard unified diff format
}

/// α: A project artifact (code, docs, research)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Artifact {
    pub name: String,
    pub language: String, // e.g., "rust"
    pub content: String,
    pub version: u32,
    pub history: Vec<ArtifactDiff>,
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
    pub completion_probability: f64, // P(C) from Kalman Filter
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
        }
    }

    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
