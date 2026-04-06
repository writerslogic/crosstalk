use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryRecord {
    pub turn_id: u32,
    pub session_id: String,
    pub embedding: Vec<f32>,
    pub content_hash: String,
    pub timestamp: u64,
    pub metadata_json: String,
    pub outcome: Option<OutcomeRecord>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutcomeRecord {
    pub compiled: bool,
    pub tests_passed: bool,
    pub quality_delta: f64,
    pub was_rolled_back: bool,
    pub convergence_contribution: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransferableLesson {
    pub category: String,
    pub content: String,
    pub confidence: f64,
    pub applicability_tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FailureSignature {
    pub error_type: String,
    pub error_message: String,
    pub context_hash: String,
    pub agent_id: String,
    pub occurrence_count: u32,
    pub context_embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Lesson {
    pub context_type: String,
    pub approach: String,
    pub outcome: String,
    pub confidence: f64,
    pub applicability_tags: Vec<String>,
}
