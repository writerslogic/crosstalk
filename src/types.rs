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
    pub diffs: Vec<(String, ArtifactDiff)>, // New: (artifact_name, delta)
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
}

impl ConversationState {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            iteration_index: 0,
            turns: vec![],
            artifacts: HashMap::new(),
        }
    }

    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
