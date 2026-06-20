use crate::types::fiduciary::FiduciaryDutyEvent;
use crate::types::principal::PrincipalConstraints;
use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

/// Enforces the principal's data-retention policy against the sled database.
/// Deletes records whose storage timestamp exceeds `data_retention_days`.
/// Emits a `RetentionPurge` fiduciary event summarising the deletion count.
pub struct DataMinimizer;

impl DataMinimizer {
    /// Scan the sled tree prefix and remove entries older than the policy.
    /// Returns a `RetentionPurge` event that the caller should emit.
    pub fn enforce(
        db: &sled::Db,
        session_id: &str,
        constraints: &PrincipalConstraints,
    ) -> Result<Option<FiduciaryDutyEvent>> {
        let retention_days = match constraints.data_retention_days {
            Some(d) => d,
            None => return Ok(None), // session-scoped: no cross-session purge needed here
        };

        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let cutoff_secs = now_secs.saturating_sub(retention_days as u64 * 86_400);

        if session_id.contains(':') || session_id.contains('/') || session_id.is_empty() {
            return Err(anyhow::anyhow!(
                "Invalid session_id for sled key: {:?}",
                session_id
            ));
        }
        let prefix = format!("fiduciary:account:{}:", session_id);
        let tree = db.open_tree("audit_log")?;

        let mut to_delete: Vec<sled::IVec> = Vec::new();
        for entry in tree.scan_prefix(prefix.as_bytes()) {
            let (k, v) = entry?;
            // Each audit entry has a `timestamp` field (u64 secs) encoded as JSON.
            if let Ok(obj) = serde_json::from_slice::<serde_json::Value>(&v) {
                let ts = obj.get("timestamp").and_then(|t| t.as_u64()).unwrap_or(0);
                if ts < cutoff_secs {
                    to_delete.push(k);
                }
            }
        }

        let records_deleted = to_delete.len();
        if records_deleted > 0 {
            let mut batch = sled::Batch::default();
            for k in to_delete {
                batch.remove(k);
            }
            tree.apply_batch(batch)?;
            tree.flush()?;
        }

        Ok(Some(FiduciaryDutyEvent::RetentionPurge {
            session_id: session_id.to_string(),
            records_deleted,
        }))
    }
}
