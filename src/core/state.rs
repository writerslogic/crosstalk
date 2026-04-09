use crate::types::conversation::ConversationState;
use anyhow::{Context, Result, bail};
use sled::Db;

/// Increment this constant when the on-disk layout changes in a
/// backwards-incompatible way, and add a matching arm to `run_migrations`.
const SCHEMA_VERSION: u64 = 1;

const SCHEMA_KEY: &[u8] = b"__schema_version__";

/// Persistent checkpoint store backed by Sled.  All writes go through an atomic
/// `sled::Batch` so partial failures leave the database in the pre-operation state.
pub struct StateManager {
    db: Db,
}

impl StateManager {
    /// Open (or create) the Sled database at `path` and run any pending schema migrations.
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path).context("Failed to open Sled DB")?;
        run_migrations(&db)?;
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

    /// Atomically execute `f` against `state`, checkpointing the result on
    /// success or restoring both the in-memory state and the sled record on
    /// failure.  Uses a sled `Batch` so the write-then-cleanup is atomic.
    pub fn execute_with_rollback<F>(
        &self,
        state: &mut ConversationState,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(&mut ConversationState) -> Result<()>,
    {
        let snapshot = state.clone();
        let rollback_key = format!("state:{}:rollback", state.iteration_index);
        let snapshot_bytes = serde_json::to_vec(&snapshot)?;

        // Write rollback marker atomically before running the operation.
        let mut pre_batch = sled::Batch::default();
        pre_batch.insert(rollback_key.as_bytes(), snapshot_bytes.as_slice());
        self.db.apply_batch(pre_batch)?;

        match f(state) {
            Ok(()) => {
                let new_bytes = serde_json::to_vec(state)?;
                let mut commit_batch = sled::Batch::default();
                commit_batch.insert(
                    format!("state:{}", state.iteration_index).as_bytes(),
                    new_bytes.as_slice(),
                );
                commit_batch.remove(rollback_key.as_bytes());
                self.db.apply_batch(commit_batch)?;
                self.db.flush()?;
                Ok(())
            }
            Err(e) => {
                *state = snapshot;
                let mut rollback_batch = sled::Batch::default();
                rollback_batch.remove(rollback_key.as_bytes());
                self.db.apply_batch(rollback_batch)?;
                Err(e)
            }
        }
    }

    pub fn list_checkpoints(&self) -> Vec<u32> {
        self.db
            .scan_prefix("state:")
            .filter_map(|r| r.ok())
            .filter_map(|(k, _)| {
                std::str::from_utf8(&k)
                    .ok()?
                    .strip_prefix("state:")?
                    .parse::<u32>()
                    .ok()
            })
            .collect()
    }
}

/// Read the stored schema version, run any missing migrations in order, then
/// write the current `SCHEMA_VERSION` so the check is idempotent on restart.
fn run_migrations(db: &Db) -> Result<()> {
    let stored: u64 = db
        .get(SCHEMA_KEY)?
        .and_then(|v| {
            let arr: [u8; 8] = v.as_ref().try_into().ok()?;
            Some(u64::from_le_bytes(arr))
        })
        .unwrap_or(0);

    if stored > SCHEMA_VERSION {
        bail!(
            "DB schema version {} is newer than binary schema version {}; upgrade the binary",
            stored,
            SCHEMA_VERSION
        );
    }

    // Each arm migrates from `version` → `version + 1`.
    // Add new arms here when SCHEMA_VERSION is incremented.
    for version in stored..SCHEMA_VERSION {
        match version {
            0 => {
                // v0 → v1: no structural changes; initial schema baseline.
            }
            v => bail!("No migration defined for schema version {}", v),
        }
    }

    db.insert(SCHEMA_KEY, &SCHEMA_VERSION.to_le_bytes())?;
    db.flush()?;
    Ok(())
}
