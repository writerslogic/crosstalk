use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MergeVote {
    pub node_id: String,
    pub approve: bool,
    pub reason: String,
}
