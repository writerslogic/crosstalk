use crate::engines::quality::ArtifactMetrics;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A versioned, language-tagged code artifact tracked across session turns.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Artifact {
    pub name: String,
    pub language: String,
    pub content: String,
    pub version: u32,
    pub history: Vec<ArtifactDiff>,
    #[serde(default)]
    pub ast_versions: BTreeMap<String, Vec<(u32, String)>>,
    #[serde(default)]
    pub proof_attachments: Vec<ProofAttachment>,
    #[serde(default)]
    pub metrics: ArtifactMetrics,
    #[serde(default)]
    pub skeleton: String,
}

/// A unified-diff record capturing the change between two artifact versions.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArtifactDiff {
    pub original_version: u32,
    pub new_version: u32,
    pub diff_text: String,
}

impl ArtifactDiff {
    const MAX_DIFF_SIZE: usize = 1024 * 1024;

    pub fn new(original_version: u32, new_version: u32, diff_text: String) -> Result<Self> {
        let artifact = ArtifactDiff {
            original_version,
            new_version,
            diff_text,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<()> {
        if self.diff_text.len() > Self::MAX_DIFF_SIZE {
            return Err(anyhow!("diff_text exceeds 1MB limit"));
        }
        if self.diff_text.contains('\0') {
            return Err(anyhow!("diff_text contains null bytes"));
        }
        Ok(())
    }
}

/// A Verus proof result attached to an artifact, recording which properties were verified.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProofAttachment {
    pub artifact_name: String,
    pub proven_properties: Vec<String>,
    pub proof_hash: String,
    pub verified_at: u64,
}
