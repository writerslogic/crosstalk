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
    pub status: HypothesisStatus,
}
