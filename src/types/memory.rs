use crate::types::conversation::TurnOutcome;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single turn stored in the LanceDB vector table for semantic recall.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryRecord {
    pub turn_id: u32,
    pub session_id: String,
    pub embedding: Vec<f32>,
    pub content_hash: String,
    pub timestamp: u64,
    pub metadata_json: String,
    pub outcome: Option<OutcomeRecord>,
    /// Negative examples represent antipatterns extracted from failed sessions.
    #[serde(default)]
    pub is_negative: bool,
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MemoryStoreStats {
    pub total_records: usize,
    pub unique_sessions: usize,
    pub avg_cluster_size: f64,
    pub storage_size: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SnapshotMetadata {
    pub session_id: String,
    pub created_at: u64,
    pub record_count: usize,
    pub content_hash: [u8; 32],
    pub compressed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotBundle {
    pub metadata: SnapshotMetadata,
    pub records: Vec<MemoryRecord>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SessionContext {
    pub session_id: String,
    pub start_time: u64,
    pub last_recall_time: Option<u64>,
    pub linked_sessions: Vec<String>,
    pub total_turns: u32,
    pub outcome_summary: BTreeMap<TurnOutcome, u32>,
}

impl SessionContext {
    pub fn new(session_id: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            session_id: session_id.to_string(),
            start_time: now,
            last_recall_time: None,
            linked_sessions: Vec::new(),
            total_turns: 0,
            outcome_summary: BTreeMap::new(),
        }
    }

    pub fn record_turn(&mut self, outcome: TurnOutcome) {
        self.total_turns += 1;
        *self.outcome_summary.entry(outcome).or_insert(0) += 1;
    }

    pub fn link_session(&mut self, prior_session_id: &str) {
        if self.linked_sessions.len() >= 100 {
            return;
        }
        if !self.linked_sessions.contains(&prior_session_id.to_string()) {
            self.linked_sessions.push(prior_session_id.to_string());
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeletionLogEntry {
    pub turn_id: u32,
    pub session_id: String,
    pub deleted_at: u64,
}
