use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FallacyReport {
    pub fallacy_type: String,
    pub evidence_span: String,
    pub confidence: f64,
}
