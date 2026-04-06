use crate::engines::quality::ArtifactMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Artifact {
    pub name: String,
    pub language: String,
    pub content: String,
    pub version: u32,
    pub history: Vec<ArtifactDiff>,
    #[serde(default)]
    pub ast_versions: HashMap<String, Vec<(u32, String)>>,
    #[serde(default)]
    pub proof_attachments: Vec<ProofAttachment>,
    #[serde(default)]
    pub metrics: ArtifactMetrics,
    #[serde(default)]
    pub skeleton: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArtifactDiff {
    pub original_version: u32,
    pub new_version: u32,
    pub diff_text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProofAttachment {
    pub artifact_name: String,
    pub proven_properties: Vec<String>,
    pub proof_hash: String,
    pub verified_at: u64,
}
