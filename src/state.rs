use crate::types::ConversationState;
use anyhow::{Context, Result};
use sled::Db;

pub struct StateManager {
    db: Db,
}

impl StateManager {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path).context("Failed to open Sled DB")?;
        Ok(Self { db })
    }

    /// ∀ i ⇒ Checkpoint(σ)
    pub fn checkpoint(&self, state: &ConversationState) -> Result<()> {
        let key = format!("state:{}", state.iteration_index);
        let encoded = serde_json::to_vec(state)?;
        self.db.insert(key, encoded)?;
        self.db.flush()?; 
        Ok(())
    }

    /// Rewind :: σ_t ← σ_{t-k}
    pub fn restore(&self, index: u32) -> Result<Option<ConversationState>> {
        let key = format!("state:{}", index);
        match self.db.get(key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    pub fn list_checkpoints(&self) -> Vec<u32> {
        self.db.range("state:0".."state:999999")
            .filter_map(|r| r.ok())
            .filter_map(|(k, _)| {
                std::str::from_utf8(&k).ok()?
                    .strip_prefix("state:")?
                    .parse::<u32>().ok()
            })
            .collect()
    }
}