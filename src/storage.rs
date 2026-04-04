use crate::types::Artifact;
use anyhow::Result;
use std::fs;
use std::path::Path;

pub struct ArtifactStorage {
    base_path: String,
}

impl ArtifactStorage {
    pub fn new(path: &str) -> Result<Self> {
        fs::create_dir_all(path)?;
        Ok(Self {
            base_path: path.to_string(),
        })
    }

    /// Write α to disk
    pub fn save_artifact(&self, artifact: &Artifact) -> Result<()> {
        let path = Path::new(&self.base_path).join(&artifact.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &artifact.content)?;
        Ok(())
    }
}
