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

    pub fn db(&self) -> &sled::Db {
        &self.db
    }

    /// ∀ i ⇒ Checkpoint(σ)
    pub fn checkpoint(&self, state: &ConversationState) -> Result<()> {
        let key = format!("state:{}", state.iteration_index);
        let encoded = serde_json::to_vec(state)?;
        self.db.insert(key, encoded)?;
        self.db.flush()?;
        Ok(())
    }

    /// Async-safe variant of [`Self::checkpoint`]: the blocking sled insert + fsync
    /// are moved to the blocking-thread pool so the reactor is never stalled.
    pub async fn checkpoint_async(&self, state: &ConversationState) -> Result<()> {
        let key = format!("state:{}", state.iteration_index);
        let encoded = serde_json::to_vec(state)?;
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            db.insert(key.as_bytes(), encoded)?;
            db.flush()?;
            Ok(())
        })
        .await
        .context("checkpoint task panicked")?
    }

    /// Rewind :: σ_t ← σ_{t-k}
    pub fn restore(&self, index: u32) -> Result<Option<ConversationState>> {
        let key = format!("state:{}", index);
        match self.db.get(key.as_bytes()).context("sled read failed for state checkpoint")? {
            Some(data) => Ok(Some(serde_json::from_slice(&data).context("failed to deserialize state checkpoint")?)),
            None => Ok(None),
        }
    }

    /// Async-safe variant of [`Self::restore`]: the blocking sled read is moved
    /// to the blocking-thread pool so the reactor is never stalled.
    pub async fn restore_async(&self, index: u32) -> Result<Option<ConversationState>> {
        let key = format!("state:{}", index);
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<Option<ConversationState>> {
            match db.get(key.as_bytes())? {
                Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
                None => Ok(None),
            }
        })
        .await
        .context("restore task panicked")?
    }

    /// Atomically execute `f` against `state`, checkpointing the result on
    /// success or restoring both the in-memory state and the sled record on
    /// failure.  Uses a sled `Batch` so the write-then-cleanup is atomic.
    pub fn execute_with_rollback<F>(&self, state: &mut ConversationState, f: F) -> Result<()>
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

    pub fn list_checkpoints(&self) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for entry in self.db.scan_prefix("state:") {
            let (k, _) = entry.context("failed to scan checkpoint entry")?;
            let key = std::str::from_utf8(&k)
                .with_context(|| format!("non-utf8 checkpoint key: {:?}", k.as_ref()))?;
            let Some(suffix) = key.strip_prefix("state:") else {
                continue;
            };
            if suffix.ends_with(":rollback") {
                continue;
            }
            let idx = suffix
                .parse::<u32>()
                .with_context(|| format!("invalid checkpoint index in key: {key}"))?;
            out.push(idx);
        }
        Ok(out)
    }
}

/// Read the stored schema version, run any missing migrations in order, then
/// write the current `SCHEMA_VERSION` so the check is idempotent on restart.
fn run_migrations(db: &Db) -> Result<()> {
    let stored: u64 = match db.get(SCHEMA_KEY)? {
        None => 0,
        Some(v) => {
            let arr: [u8; 8] = v.as_ref().try_into()
                .context("SCHEMA_KEY corrupted: invalid byte length")?;
            u64::from_le_bytes(arr)
        }
    };

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
                // Write the version bump atomically so a crash here doesn't
                // replay this migration on restart.
                let mut batch = sled::Batch::default();
                batch.insert(SCHEMA_KEY, &1u64.to_le_bytes());
                db.apply_batch(batch)?;
            }
            v => bail!("No migration defined for schema version {}", v),
        }
    }

    db.flush()?;
    Ok(())
}
