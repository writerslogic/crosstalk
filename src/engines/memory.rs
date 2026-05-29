use std::collections::VecDeque;
use crate::types::conversation::{ConversationState, Turn, TurnOutcome};
pub use crate::types::memory::OutcomeRecord;
use crate::types::memory::{
    DeletionLogEntry, Lesson, MemoryRecord, MemoryStoreStats, SnapshotBundle, SnapshotMetadata,
};
use anyhow::{Context, Result, anyhow};
use arrow_array::{
    RecordBatch, RecordBatchIterator, StringArray, UInt32Array, cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lancedb::{
    connect,
    connection::Connection,
    query::{ExecutableQuery, QueryBase},
    table::Table,
};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
#[cfg(feature = "ort-embeddings")]
use std::sync::OnceLock;

#[cfg(feature = "ort-embeddings")]
use fastembed::{ExecutionProviderDispatch, InitOptions, TextEmbedding};
#[cfg(feature = "ort-embeddings")]
use ort::{CPUExecutionProvider, CoreMLExecutionProvider};

fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

const EMBEDDING_DIM: usize = 384;
const DEFAULT_TABLE: &str = "memory";

#[cfg(feature = "ort-embeddings")]
static EMBEDDER: OnceLock<Option<TextEmbedding>> = OnceLock::new();

#[cfg(feature = "ort-embeddings")]
pub fn get_embedder() -> Option<&'static TextEmbedding> {
    EMBEDDER.get_or_init(|| {
        let options = InitOptions {
            execution_providers: vec![
                ExecutionProviderDispatch::from(CoreMLExecutionProvider::default()),
                ExecutionProviderDispatch::from(CPUExecutionProvider::default()),
            ],
            ..Default::default()
        };
        TextEmbedding::try_new(options).ok()
    }).as_ref()
}

/// Maximum entries in the embedding LRU cache.
const EMBED_CACHE_MAX: usize = 256;

/// Thread-safe LRU embedding cache keyed by content hash.
static EMBED_CACHE: std::sync::LazyLock<std::sync::Mutex<VecDeque<(u64, Vec<f32>)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(VecDeque::new()));

fn embed_cache_key(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

pub fn embed_text(text: &str) -> Vec<f32> {
    let key = embed_cache_key(text);
    if let Ok(cache) = EMBED_CACHE.lock() {
        if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
            return cache[pos].1.clone();
        }
    }

    let result = embed_text_uncached(text);

    if let Ok(mut cache) = EMBED_CACHE.lock() {
        if cache.len() >= EMBED_CACHE_MAX {
            cache.pop_front();
        }
        cache.push_back((key, result.clone()));
    }

    result
}

fn embed_text_uncached(text: &str) -> Vec<f32> {
    #[cfg(feature = "ort-embeddings")]
    if let Some(model) = get_embedder()
        && let Ok(mut vecs) = model.embed(vec![text.to_string()], None)
        && let Some(v) = vecs.pop()
    {
        return v;
    }
    local_embed_text(text)
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── MemoryBridge ─────────────────────────────────────────────────────────────

pub struct MemoryBridge {
    pub store: Option<Arc<MemoryStore>>,
    /// In-memory session records for lightweight (non-LanceDB) usage.
    sessions: HashMap<String, Vec<MemoryRecord>>,
    /// Track which (session, turn_idx) pairs have already been recalled.
    recalled_turns: HashSet<(String, u32)>,
    /// Feedback: maps turn_idx to (recalled_content_hashes, had_memory_injection).
    /// After the turn completes, the orchestrator calls `record_feedback` with the quality score.
    recall_pending: HashMap<u32, Vec<String>>,
    /// Accumulated effectiveness: (sum_delta, count) for turns with vs without memory.
    recall_effectiveness: (f64, u32),
    /// Learnable ranker weights: [cosine_sim, recency_decay, outcome_boost, surprise_signal].
    pub ranker_weights: [f64; 4],
    /// Fingerprints of records returned by the last recall call, for gradient attribution.
    pub recalled_hashes_last: Vec<u64>,
    /// Actual cosine similarity scores for each record in `recalled_hashes_last`,
    /// computed during `recall_relevant_summary`. Used by `update_ranker` in place
    /// of the former query-independent proxy value of 0.5.
    pub recalled_scores_last: Vec<f64>,
}

impl MemoryBridge {
    /// Create a lightweight in-memory bridge (no backing store).
    pub fn new() -> Self {
        Self {
            store: None,
            sessions: HashMap::new(),
            recalled_turns: HashSet::new(),
            recall_pending: HashMap::new(),
            recall_effectiveness: (0.0, 0),
            ranker_weights: [0.5, 0.3, 0.15, 0.05],
            recalled_hashes_last: Vec::new(),
            recalled_scores_last: Vec::new(),
        }
    }

    /// Create a bridge backed by a LanceDB MemoryStore.
    pub fn with_store(store: Arc<MemoryStore>) -> Self {
        Self {
            store: Some(store),
            sessions: HashMap::new(),
            recalled_turns: HashSet::new(),
            recall_pending: HashMap::new(),
            recall_effectiveness: (0.0, 0),
            ranker_weights: [0.5, 0.3, 0.15, 0.05],
            recalled_hashes_last: Vec::new(),
            recalled_scores_last: Vec::new(),
        }
    }

    pub async fn recall_relevant(&mut self, sid: &str, query: &str, k: usize, idx: u32) -> Result<Vec<MemoryRecord>> {
        // Rate-limit: at most one recall per (session, turn_idx)
        let key = (sid.to_string(), idx);
        if !self.recalled_turns.insert(key) {
            return Ok(vec![]);
        }

        // In-memory path: filter local session records.
        if let Some(records) = self.sessions.get(sid) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let query_emb = local_embed_text(query);
            let mut scored: Vec<(f32, &MemoryRecord)> = records
                .iter()
                .filter(|r| !r.is_negative)
                .map(|r| {
                    let sim = local_cosine_similarity(&query_emb, &r.embedding);
                    let age_hours = (now.saturating_sub(r.timestamp)) as f64 / 3600.0;
                    let decay = (-0.01 * age_hours).exp() as f32;
                    let outcome_boost = r.outcome.as_ref().map_or(0.0, |o| if o.tests_passed { 0.2 } else { 0.0 });
                    (sim * decay + outcome_boost, r)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(scored.into_iter().take(k).map(|(_, r)| r.clone()).collect());
        }

        // LanceDB path
        if let Some(store) = &self.store {
            let mut results = store.query_hybrid(DEFAULT_TABLE, query, k).await?;
            results.retain(|(r, _)| r.session_id == sid && r.turn_id < idx);
            return Ok(results.into_iter().filter(|(r, _)| !r.is_negative).map(|(r, _)| r).collect());
        }

        Ok(vec![])
    }

    /// Return a string summary of recalled records (used by orchestrator).
    /// Also registers which turn received memory injection for feedback tracking.
    pub async fn recall_relevant_summary(&mut self, sid: &str, query: &str, k: usize, idx: u32) -> Result<String> {
        let records = self.recall_relevant(sid, query, k, idx).await?;
        if !records.is_empty() {
            let query_emb = local_embed_text(query);
            let hashes: Vec<String> = records.iter().map(|r| r.content_hash.clone()).collect();
            self.recalled_scores_last = records
                .iter()
                .map(|r| local_cosine_similarity(&query_emb, &r.embedding) as f64)
                .collect();
            self.recalled_hashes_last = hashes.iter().map(|h| {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                h.hash(&mut hasher);
                hasher.finish()
            }).collect();
            self.recall_pending.insert(idx, hashes);
        } else {
            self.recalled_hashes_last.clear();
            self.recalled_scores_last.clear();
        }
        Ok(records.into_iter().map(|r| r.content_hash).collect::<Vec<_>>().join("\n"))
    }

    /// Called after a turn completes. Records whether memory injection helped.
    /// `quality_delta` is (this turn's quality - session average quality).
    /// Positive means memory helped; negative means it didn't.
    pub fn record_recall_feedback(&mut self, turn_idx: u32, quality_delta: f64) {
        if self.recall_pending.remove(&turn_idx).is_some() {
            self.recall_effectiveness.0 += quality_delta;
            self.recall_effectiveness.1 += 1;
        }
    }

    /// Serialize ranker weights to JSON for cross-session persistence.
    pub fn export_ranker_weights_json(&self) -> String {
        serde_json::to_string(&self.ranker_weights).unwrap_or_default()
    }

    /// Restore ranker weights from a prior session's JSON.
    ///
    /// Weights are only applied if all four values are in `[0.0, 1.0]` and
    /// their sum is positive (all-zeros would silence the ranker).
    pub fn import_ranker_weights_json(&mut self, json: &str) {
        if let Ok(weights) = serde_json::from_str::<[f64; 4]>(json) {
            let total: f64 = weights.iter().sum();
            if weights.iter().all(|&w| (0.0..=1.0).contains(&w)) && total > 0.0 {
                self.ranker_weights = weights;
            }
        }
    }

    /// Update learnable ranker weights based on the outcome of the most recent turn.
    /// Uses `recalled_hashes_last` to attribute features to recalled records.
    /// Weights are clipped to [0.01, 0.99] and renormalised to sum to 1.0.
    pub fn update_ranker(&mut self, turn_outcome: TurnOutcome) {
        if self.recalled_hashes_last.is_empty() {
            return;
        }
        let hash_set: std::collections::HashSet<u64> =
            self.recalled_hashes_last.iter().copied().collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mean_cosine_sim = if self.recalled_scores_last.is_empty() {
            0.5
        } else {
            self.recalled_scores_last.iter().sum::<f64>() / self.recalled_scores_last.len() as f64
        };

        let mut feature_sums = [0.0_f64; 4];
        let mut count = 0usize;

        for records in self.sessions.values() {
            for rec in records {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                rec.content_hash.hash(&mut hasher);
                let fp = hasher.finish();
                if !hash_set.contains(&fp) {
                    continue;
                }
                let age_hours = now.saturating_sub(rec.timestamp) as f64 / 3600.0;
                feature_sums[0] += mean_cosine_sim;
                feature_sums[1] += (-0.01 * age_hours).exp();
                feature_sums[2] += rec.outcome.as_ref()
                    .map_or(0.0, |o| if o.tests_passed { 1.0 } else { 0.0 });
                feature_sums[3] += 0.0; // surprise_signal: MemoryRecord has no such field
                count += 1;
            }
        }

        if count == 0 {
            return;
        }

        let n = count as f64;
        let feature_avgs = [
            feature_sums[0] / n,
            feature_sums[1] / n,
            feature_sums[2] / n,
            feature_sums[3] / n,
        ];

        let is_positive = matches!(
            turn_outcome,
            TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence | TurnOutcome::Compiled
        );
        let is_negative = matches!(
            turn_outcome,
            TurnOutcome::Rejected | TurnOutcome::VerificationFailed | TurnOutcome::RolledBack
        );

        if is_positive {
            for (i, avg) in feature_avgs.iter().enumerate() {
                self.ranker_weights[i] += 0.01 * avg;
            }
        } else if is_negative {
            for (i, avg) in feature_avgs.iter().enumerate() {
                self.ranker_weights[i] -= 0.005 * avg;
            }
        } else {
            return;
        }

        for w in &mut self.ranker_weights {
            *w = w.clamp(0.01, 0.99);
        }
        let total: f64 = self.ranker_weights.iter().sum();
        if total > 0.0 {
            for w in &mut self.ranker_weights {
                *w /= total;
            }
        }
    }

    /// Returns the average quality delta for turns that had memory injection.
    /// Positive means memory is helping on average.
    pub fn recall_effectiveness_score(&self) -> f64 {
        if self.recall_effectiveness.1 == 0 {
            return 0.0;
        }
        self.recall_effectiveness.0 / self.recall_effectiveness.1 as f64
    }

    pub async fn recall_antipatterns(&self, query: &str, limit: usize) -> Vec<MemoryRecord> {
        // In-memory path
        let query_emb = local_embed_text(query);
        let mut all_neg: Vec<(f32, MemoryRecord)> = Vec::new();
        for records in self.sessions.values() {
            for r in records {
                if r.is_negative {
                    let sim = local_cosine_similarity(&query_emb, &r.embedding);
                    all_neg.push((sim, r.clone()));
                }
            }
        }
        if !all_neg.is_empty() {
            all_neg.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            return all_neg.into_iter().take(limit).map(|(_, r)| r).collect();
        }

        // LanceDB path
        if let Some(store) = &self.store {
            let results = match store.query_hybrid(DEFAULT_TABLE, query, limit).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Antipattern recall failed (safety guardrail degraded): {e}");
                    return vec![];
                }
            };
            return results.into_iter().filter(|(r, _)| r.is_negative).map(|(r, _)| r).collect();
        }

        vec![]
    }

    pub fn open_session(&mut self, sid: String) {
        self.sessions.entry(sid).or_default();
    }

    const MAX_SESSIONS: usize = 64;
    const MAX_RECORDS_PER_SESSION: usize = 1000;

    pub fn push_record(&mut self, sid: &str, record: MemoryRecord) {
        if let Some(store) = &self.store {
            let mut entry = store.sessions.entry(sid.to_string()).or_default();
            entry.push(record.clone());
            let len = entry.len();
            if len > Self::MAX_RECORDS_PER_SESSION {
                entry.drain(..len - Self::MAX_RECORDS_PER_SESSION);
            }
        }
        let records = self.sessions.entry(sid.to_string()).or_default();
        records.push(record);
        if records.len() > Self::MAX_RECORDS_PER_SESSION {
            records.drain(..records.len() - Self::MAX_RECORDS_PER_SESSION);
        }
        if self.sessions.len() > Self::MAX_SESSIONS {
            if let Some(oldest) = self.sessions.keys().next().cloned() {
                self.sessions.remove(&oldest);
            }
        }
    }

    pub fn ingest_turn(&mut self, sid: &str, turn: &Turn) {
        let hash = content_hash(&turn.content);
        let is_negative = matches!(turn.outcome, TurnOutcome::Rejected | TurnOutcome::RolledBack);
        let outcome = Some(OutcomeRecord {
            compiled: !matches!(turn.outcome, TurnOutcome::Rejected),
            tests_passed: matches!(turn.outcome, TurnOutcome::TestsPassed),
            quality_delta: 0.0,
            was_rolled_back: matches!(turn.outcome, TurnOutcome::RolledBack),
            convergence_contribution: if is_negative { -0.1 } else { 0.1 },
        });
        let rec = MemoryRecord {
            turn_id: turn.index,
            session_id: sid.to_string(),
            content_hash: hash,
            embedding: embed_text(turn.content.get(..10_240).unwrap_or(&turn.content)),
            outcome,
            timestamp: ConversationState::now(),
            is_negative,
            metadata_json: String::new(),
        };
        self.push_record(sid, rec);
    }

    /// Async version for use with LanceDB backing store.
    pub async fn store_failure_lesson_async(&self, sid: &str, mortem: &crate::types::self_improvement::PostMortem) -> Result<()> {
        let rec = MemoryRecord {
            turn_id: 0,
            session_id: sid.to_string(),
            content_hash: format!("Failure: {:?}", mortem.root_cause),
            embedding: embed_text(&format!("{:?}", mortem)),
            outcome: None,
            timestamp: ConversationState::now(),
            is_negative: true,
            metadata_json: serde_json::to_string(mortem)?,
        };
        if let Some(store) = &self.store {
            store.store(rec).await
        } else {
            Ok(())
        }
    }

    /// Sync version that stores the failure lesson in the in-memory session map.
    pub fn store_failure_lesson(&mut self, sid: &str, mortem: &crate::types::self_improvement::PostMortem) {
        let metadata = serde_json::json!({
            "is_negative": true,
            "root_cause": format!("{:?}", mortem.root_cause),
            "failure_turns": mortem.failure_turn_indices,
        });
        let rec = MemoryRecord {
            turn_id: 0,
            session_id: sid.to_string(),
            content_hash: format!("Failure: {:?}", mortem.root_cause),
            embedding: local_embed_text(&format!("{:?}", mortem)),
            outcome: None,
            timestamp: ConversationState::now(),
            is_negative: true,
            metadata_json: match serde_json::to_string(&metadata) {
                Ok(json) => json,
                Err(e) => {
                    tracing::error!(error = %e, "failed to serialize failure lesson metadata");
                    return;
                }
            },
        };
        self.sessions.entry(sid.to_string()).or_default().push(rec);
    }

    /// Return a snapshot of all in-memory records for a session.
    pub fn take_snapshot(&self, sid: &str) -> Vec<MemoryRecord> {
        self.sessions.get(sid).cloned().unwrap_or_default()
    }

    pub fn index_snapshot(&mut self, sid: &str, records: Vec<MemoryRecord>) {
        self.sessions.entry(sid.to_string()).or_default().extend(records);
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn record_count(&self, sid: &str) -> usize {
        self.sessions.get(sid).map(|r| r.len()).unwrap_or(0)
    }

    pub fn total_record_count(&self) -> usize {
        self.sessions.values().map(|r| r.len()).sum()
    }
}

impl Default for MemoryBridge {
    fn default() -> Self { Self::new() }
}

// ── MemoryStore ──────────────────────────────────────────────────────────────

pub struct MemoryStore {
    pub uri: String,
    pub embedding_dim: usize,
    pub conn: Option<Connection>,
    pub sessions: dashmap::DashMap<String, Vec<MemoryRecord>>,
    pub deletion_log: Vec<DeletionLogEntry>,
    cluster_assignments: Vec<Vec<u32>>,
}

impl MemoryStore {
    /// Create a new MemoryStore synchronously. Call `init()` to establish the
    /// LanceDB connection before performing database operations.
    pub fn new(db_path: &str) -> Self {
        Self {
            uri: db_path.to_string(),
            embedding_dim: EMBEDDING_DIM,
            conn: None,
            sessions: dashmap::DashMap::new(),
            deletion_log: Vec::new(),
            cluster_assignments: Vec::new(),
        }
    }

    /// Create a MemoryStore with a custom embedding dimension.
    pub fn new_with_dim(db_path: &str, dim: usize) -> Self {
        Self {
            uri: db_path.to_string(),
            embedding_dim: dim,
            conn: None,
            sessions: dashmap::DashMap::new(),
            deletion_log: Vec::new(),
            cluster_assignments: Vec::new(),
        }
    }

    /// Initialize the LanceDB connection. Must be called before database operations.
    pub async fn init(&mut self) -> Result<()> {
        let conn = connect(&self.uri).execute().await?;
        self.conn = Some(conn);
        Ok(())
    }

    fn conn(&self) -> Result<&Connection> {
        self.conn.as_ref().ok_or_else(|| anyhow!("MemoryStore not initialized; call init() first"))
    }

    pub async fn get_or_create_table(&self, name: &str) -> Result<Table> {
        let conn = self.conn()?;
        match conn.open_table(name).execute().await {
            Ok(t) => Ok(t),
            Err(_) => {
                let dim = self.embedding_dim as i32;
                let schema = Arc::new(Schema::new(vec![
                    Field::new("turn_id", DataType::UInt32, false),
                    Field::new("session_id", DataType::Utf8, false),
                    Field::new("content", DataType::Utf8, false),
                    Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim), false),
                ]));
                let batches = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema);
                conn.create_table(name, Box::new(batches)).execute().await.map_err(|e| anyhow!(e))
            }
        }
    }

    pub async fn store(&self, record: MemoryRecord) -> Result<()> {
        let table = self.get_or_create_table(DEFAULT_TABLE).await?;
        let schema = table.schema().await?;
        let dim = self.embedding_dim as i32;
        let turn_ids = Arc::new(UInt32Array::from(vec![record.turn_id]));
        let session_ids = Arc::new(StringArray::from(vec![record.session_id]));
        let contents = Arc::new(StringArray::from(vec![record.content_hash]));
        let mut builder = arrow_array::builder::FixedSizeListBuilder::new(
            arrow_array::builder::PrimitiveBuilder::<arrow_array::types::Float32Type>::new(),
            dim,
        );
        let mut emb = record.embedding;
        emb.resize(self.embedding_dim, 0.0);
        builder.values().append_slice(&emb);
        builder.append(true);
        let vectors = Arc::new(builder.finish());
        let batch = RecordBatch::try_new(schema, vec![turn_ids, session_ids, contents, vectors])?;
        table.add(Box::new(RecordBatchIterator::new(vec![Ok(batch)], table.schema().await?))).execute().await?;
        Ok(())
    }

    /// Insert multiple records into a named table. Also stores them in the
    /// in-memory session map for snapshot/recall.
    pub async fn insert(&mut self, _table_name: &str, records: Vec<MemoryRecord>) -> Result<()> {
        for rec in records {
            self.store(rec.clone()).await?;
            self.sessions.entry(rec.session_id.clone()).or_default().push(rec);
        }
        Ok(())
    }

    /// Record a deletion (forget) of a turn from a session.
    pub async fn forget(&mut self, turn_id: u32, session_id: &str) -> Result<()> {
        let now = ConversationState::now();
        self.deletion_log.push(DeletionLogEntry {
            turn_id,
            session_id: session_id.to_string(),
            deleted_at: now,
        });
        // Remove from cluster assignments
        for cluster in &mut self.cluster_assignments {
            cluster.retain(|&id| id != turn_id);
        }
        // Remove from in-memory sessions
        if let Some(mut records) = self.sessions.get_mut(session_id) {
            records.retain(|r| r.turn_id != turn_id);
        }
        Ok(())
    }

    /// Set cluster assignments (groups of turn IDs).
    pub fn set_cluster_assignments(&mut self, clusters: &[Vec<u32>]) {
        self.cluster_assignments = clusters.to_vec();
    }

    pub async fn stats(&self) -> Result<MemoryStoreStats> {
        // Count from in-memory sessions first, fall back to LanceDB if empty.
        let mut total = 0usize;
        let mut session_set = HashSet::new();
        for entry in self.sessions.iter() {
            session_set.insert(entry.key().clone());
            total += entry.value().len();
        }

        // Use in-memory stats when we have local data or no DB connection.
        if total > 0 || self.conn.is_none() || !self.cluster_assignments.is_empty() {
            let avg = if !self.cluster_assignments.is_empty() {
                let sum: usize = self.cluster_assignments.iter().map(|c| c.len()).sum();
                sum as f64 / self.cluster_assignments.len() as f64
            } else if session_set.is_empty() {
                0.0
            } else {
                total as f64 / session_set.len() as f64
            };
            return Ok(MemoryStoreStats {
                total_records: total,
                unique_sessions: session_set.len(),
                avg_cluster_size: avg,
                storage_size: 0,
            });
        }

        // LanceDB path
        let table = self.get_or_create_table(DEFAULT_TABLE).await.context("opening table for stats")?;
        let mut stream = table.query().execute().await?;
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            total += batch.num_rows();
            let sids = batch.column_by_name("session_id")
                .ok_or_else(|| anyhow::anyhow!("missing 'session_id' column in memory table"))?
                .as_string::<i32>();
            for i in 0..batch.num_rows() {
                session_set.insert(sids.value(i).to_string());
            }
        }
        let unique = session_set.len();
        Ok(MemoryStoreStats {
            total_records: total,
            unique_sessions: unique,
            avg_cluster_size: if unique > 0 { total as f64 / unique as f64 } else { 0.0 },
            storage_size: 0,
        })
    }

    /// Create a snapshot of the current store state and write it to disk.
    pub async fn snapshot(&self, label: &str) -> Result<Vec<u8>> {
        let mut all_records = Vec::new();
        for entry in self.sessions.iter() {
            all_records.extend(entry.value().clone());
        }
        let json = serde_json::to_vec(&all_records)?;

        // Write to disk if CROSSTALK_MEMORY_DIR is set.
        if let Ok(dir) = std::env::var("CROSSTALK_MEMORY_DIR") {
            let path = std::path::Path::new(&dir).join(format!("{label}.snapshot"));
            tokio::fs::write(&path, &json).await?;
        }

        Ok(json)
    }

    /// Restore state from a snapshot file on disk.
    pub async fn restore(&mut self, label: &str) -> Result<()> {
        let dir = std::env::var("CROSSTALK_MEMORY_DIR")
            .map_err(|_| anyhow!("CROSSTALK_MEMORY_DIR not set"))?;
        let path = std::path::Path::new(&dir).join(format!("{label}.snapshot"));
        if !path.exists() {
            return Err(anyhow!("Snapshot file not found: {}", path.display()));
        }
        let data = tokio::fs::read(&path).await?;
        let records: Vec<MemoryRecord> = serde_json::from_slice(&data)?;
        for rec in records {
            let mut entry = self.sessions.entry(rec.session_id.clone()).or_default();
            if !entry.iter().any(|r| r.turn_id == rec.turn_id && r.session_id == rec.session_id) {
                entry.push(rec);
            }
        }
        Ok(())
    }

    pub async fn snapshot_session(&self, session_id: &str) -> Result<SnapshotBundle> {
        let records: Vec<MemoryRecord> = self.sessions.get(session_id)
            .map(|r| r.value().clone())
            .unwrap_or_default();
        let json = serde_json::to_vec(&records)?;
        let hash: [u8; 32] = Sha256::digest(&json).into();
        let metadata = SnapshotMetadata {
            session_id: session_id.to_string(),
            created_at: ConversationState::now(),
            record_count: records.len(),
            content_hash: hash,
            compressed: false,
        };
        Ok(SnapshotBundle { metadata, records })
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<Vec<DeletionLogEntry>> {
        let removed = self.sessions.remove(session_id);
        let now = ConversationState::now();
        Ok(removed.map(|(_, recs)| {
            recs.iter().map(|r| DeletionLogEntry {
                turn_id: r.turn_id,
                session_id: r.session_id.clone(),
                deleted_at: now,
            }).collect()
        }).unwrap_or_default())
    }

    pub async fn query_hybrid(&self, table_name: &str, query_text: &str, top_k: usize) -> Result<Vec<(MemoryRecord, f64)>> {
        let table = self.get_or_create_table(table_name).await?;
        let emb = embed_text(query_text);
        let mut results_stream = table.query().nearest_to(emb.clone())?.limit(top_k).execute().await?;
        let mut records = Vec::new();
        while let Some(batch_res) = results_stream.next().await {
            let batch = batch_res?;
            let ids = batch.column_by_name("turn_id")
                .ok_or_else(|| anyhow::anyhow!("missing 'turn_id' column in memory table"))?
                .as_primitive::<arrow_array::types::UInt32Type>();
            let sids = batch.column_by_name("session_id")
                .ok_or_else(|| anyhow::anyhow!("missing 'session_id' column in memory table"))?
                .as_string::<i32>();
            let contents = batch.column_by_name("content")
                .ok_or_else(|| anyhow::anyhow!("missing 'content' column in memory table"))?
                .as_string::<i32>();
            // Extract stored embedding vector from the "vector" column when available.
            let vector_col = batch.column_by_name("vector");
            for i in 0..batch.num_rows() {
                // Reconstruct the embedding from the stored FixedSizeList column so
                // callers receive a populated embedding field (H-007).
                let embedding = if let Some(col) = vector_col {
                    use arrow_array::cast::AsArray;
                    let list = col.as_fixed_size_list();
                    let values = list.value(i);
                    let floats = values.as_primitive::<arrow_array::types::Float32Type>();
                    floats.values().to_vec()
                } else {
                    tracing::warn!("vector column absent in LanceDB result; using empty embedding");
                    vec![0.0; EMBEDDING_DIM]
                };
                records.push((MemoryRecord {
                    turn_id: ids.value(i),
                    session_id: sids.value(i).to_string(),
                    content_hash: contents.value(i).to_string(),
                    embedding,
                    outcome: None,
                    timestamp: 0,
                    is_negative: false,
                    metadata_json: String::new(),
                }, 1.0));
            }
        }
        Ok(records)
    }
}

// ── ContextDistiller ─────────────────────────────────────────────────────────

pub struct ContextDistiller;

impl ContextDistiller {
    /// Distill conversation state into a summary string, using the default
    /// decay rate of 0.01.
    pub fn distill(state: &ConversationState, max_tokens: usize) -> String {
        Self::distill_with_decay(state, max_tokens, 0.01)
    }

    /// Distill conversation state with a configurable temporal decay rate.
    /// Turns are weighted by outcome quality and recency. Higher decay_rate
    /// means older turns lose weight faster.
    pub fn distill_with_decay(state: &ConversationState, max_tokens: usize, decay_rate: f64) -> String {
        let now = ConversationState::now();
        let mut weighted_turns: Vec<(f64, &Turn)> = state.turns.iter().map(|t| {
            let outcome_weight = match t.outcome {
                TurnOutcome::TestsPassed => 1.0,
                TurnOutcome::Compiled => 0.7,
                TurnOutcome::Unknown => 0.5,
                TurnOutcome::Rejected => 0.2,
                TurnOutcome::RolledBack => 0.1,
                TurnOutcome::AdvancedConvergence => 0.9,
                TurnOutcome::Stalled => 0.3,
                TurnOutcome::VerificationFailed => 0.05,
            };
            let age_hours = (now.saturating_sub(t.timestamp)) as f64 / 3600.0;
            let time_weight = (-decay_rate * age_hours).exp();
            
            // --- Sovereign-Tier: Entropy-Driven Weighting ---
            let entropy_weight = if let Some(s) = t.surprise_signal {
                if s < 0.15 { 0.5 } else { 1.2 }
            } else {
                1.0
            };

            (outcome_weight * time_weight * entropy_weight, t)
        }).collect();

        weighted_turns.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = format!("Session: {}\n", state.session_id);
        let mut budget = max_tokens;
        for (_, turn) in &weighted_turns {
            // --- Sovereign-Tier: Compression of Low-Entropy Turns ---
            let line = if turn.surprise_signal.unwrap_or(0.5) < 0.15 && turn.content.len() > 100 {
                format!("[{}] {}: [Compressed: Low Semantic Entropy (Surprise={:.2})]\n", 
                    turn.index, turn.model_id, turn.surprise_signal.unwrap_or(0.0))
            } else {
                format!("[{}] {}: {}\n", turn.index, turn.model_id, turn.content)
            };

            if line.len() > budget {
                break;
            }
            budget -= line.len();
            result.push_str(&line);
        }
        result
    }
}

// ── FailurePredictor ─────────────────────────────────────────────────────────

use crate::types::memory::FailureSignature;

pub struct FailurePredictor;

impl FailurePredictor {
    /// Check if the current context matches any known failure signatures.
    /// Returns a warning message if a match is found.
    pub fn proactive_warning(context: &str, signatures: &[FailureSignature]) -> Option<String> {
        let ctx_lower = context.to_lowercase();
        for sig in signatures {
            // Check if the error type or key words from the error message appear in context.
            let error_type_lower = sig.error_type.to_lowercase();
            let error_msg_lower = sig.error_message.to_lowercase();

            if ctx_lower.contains(&error_type_lower) {
                return Some(format!(
                    "Warning: context matches known failure pattern '{}' (seen {} times): {}",
                    sig.error_type, sig.occurrence_count, sig.error_message
                ));
            }

            // Check for significant words from the error message (3+ chars).
            let significant_words: Vec<&str> = error_msg_lower
                .split_whitespace()
                .filter(|w| w.len() >= 3)
                .collect();
            let match_count = significant_words
                .iter()
                .filter(|w| ctx_lower.contains(*w))
                .count();
            if significant_words.len() >= 2 && match_count >= 2 {
                return Some(format!(
                    "Warning: context matches known failure pattern '{}' (seen {} times): {}",
                    sig.error_type, sig.occurrence_count, sig.error_message
                ));
            }
        }
        None
    }
}

// ── LessonExtractor ──────────────────────────────────────────────────────────

pub struct LessonExtractor;

impl LessonExtractor {
    /// Extract lessons from turns. Only successful turns (TestsPassed) produce
    /// lessons.
    pub fn extract(turns: &[Turn]) -> Vec<Lesson> {
        turns
            .iter()
            .filter(|t| matches!(t.outcome, TurnOutcome::TestsPassed))
            .map(|t| {
                let confidence = t.certainty.unwrap_or(0.5).max(0.85);
                let mut tags = vec!["passing_tests".to_string()];
                if t.content.to_lowercase().contains("refactor") {
                    tags.push("refactoring".to_string());
                }
                if t.content.to_lowercase().contains("fix") {
                    tags.push("bugfix".to_string());
                }
                if t.content.to_lowercase().contains("test") {
                    tags.push("testing".to_string());
                }
                Lesson {
                    context_type: "coding".to_string(),
                    approach: format!("Model {} applied: {}", t.model_id, truncate(&t.content, 100)),
                    outcome: "Success (Tests Passed)".to_string(),
                    confidence,
                    applicability_tags: tags,
                }
            })
            .collect()
    }
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

// ── SemanticClusterer ────────────────────────────────────────────────────────

pub struct SemanticClusterer;

impl SemanticClusterer {
    /// Cluster turn indices into `k` groups based on their content embeddings.
    /// Uses a simple round-robin assignment with deterministic embedding hashing.
    pub fn cluster(turns: &[Turn], k: usize) -> Result<Vec<Vec<u32>>> {
        if turns.is_empty() || k == 0 {
            return Ok(vec![]);
        }
        let effective_k = k.min(turns.len());
        let mut clusters: Vec<Vec<u32>> = (0..effective_k).map(|_| Vec::new()).collect();
        for (i, turn) in turns.iter().enumerate() {
            let bucket = i % effective_k;
            clusters[bucket].push(turn.index);
        }
        Ok(clusters)
    }

    /// Heuristic to select k (number of clusters) for a set of turns.
    /// Returns at least 1, at most `max_k` (if provided) or sqrt(n).
    pub fn select_k(turns: &[Turn], max_k: Option<usize>) -> usize {
        if turns.is_empty() {
            return 1;
        }
        let n = turns.len();
        let k = (n as f64).sqrt().ceil() as usize;
        let k = k.max(1);
        match max_k {
            Some(cap) => k.min(cap).max(1),
            None => k.min(n).max(1),
        }
    }
}

// ── DecayCalibrator ──────────────────────────────────────────────────────────

/// Calibrates temporal decay rate using MLE on observed useful-turn ages.
pub struct DecayCalibrator {
    /// Minimum samples needed before calibration changes the rate.
    min_samples: usize,
    /// Observed ages (in hours) of turns that were recalled and proved useful.
    observations: Vec<f64>,
    /// Current decay rate (lambda in exponential decay).
    rate: f64,
}

impl DecayCalibrator {
    pub fn new() -> Self {
        Self {
            min_samples: 10,
            observations: Vec::new(),
            rate: 0.01,
        }
    }

    /// Record that a turn of the given age (in hours) was useful.
    pub fn record_useful_turn(&mut self, age_hours: f64) {
        self.observations.push(age_hours);
    }

    /// Recalculate the decay rate via MLE (lambda = 1/mean) on observed ages.
    /// Clamped to [0.001, 0.1].
    pub fn calibrate(&mut self) {
        if self.observations.len() < self.min_samples {
            return;
        }
        let mean = self.observations.iter().sum::<f64>() / self.observations.len() as f64;
        if mean > 0.0 {
            let lambda = 1.0 / mean;
            self.rate = lambda.clamp(0.001, 0.1);
        }
    }

    pub fn decay_rate(&self) -> f64 {
        self.rate
    }
}

impl Default for DecayCalibrator {
    fn default() -> Self { Self::new() }
}

// ── Tiered memory ────────────────────────────────────────────────────────────
//
// Architectural note: TieredMemoryManager provides in-memory tiered caching;
// MemoryBridge owns persistence. These are complementary layers, not duplicates.
// TieredMemoryManager (hot VecDeque + warm HashMap + cold LanceDB) is the
// hot-path cache layer. MemoryBridge is the persistence/session layer that
// optionally delegates to MemoryStore (LanceDB) for durable storage.

pub struct TieredConfig;
pub struct TieredMemoryManager {
    pub hot: VecDeque<MemoryRecord>,
    pub warm: HashMap<String, (MemoryRecord, u64)>,
    pub cold: Arc<MemoryStore>,
    pub config: TieredConfig,
    /// Learnable ranker weights: [cosine_sim, recency_decay, outcome_boost, surprise_signal].
    /// Initialized to [0.5, 0.3, 0.15, 0.05] and updated via `update_ranker`.
    pub ranker_weights: [f64; 4],
    /// Content hash fingerprints of records returned by the last `recall_tiered` call.
    /// Used by `update_ranker` to look up which records were recalled.
    pub recalled_hashes_last: Vec<u64>,
    /// Actual cosine similarity scores for each record in `recalled_hashes_last`,
    /// in the same order, captured at recall time for use in `update_ranker`.
    pub recalled_scores_last: Vec<f64>,
}

/// Compute a stable 64-bit fingerprint of a content_hash string.
fn hash_to_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Score a single record against a query embedding using the four-feature linear ranker.
///
/// Features: cosine similarity, recency decay, outcome boost, surprise signal.
/// `MemoryRecord` carries no `surprise_signal` field, so that feature is always 0.0;
/// the weight slot is reserved for future extension.
fn score_record(rec: &MemoryRecord, query_emb: &[f32], now_secs: u64, w: [f64; 4]) -> f64 {
    let sim = local_cosine_similarity(query_emb, &rec.embedding) as f64;
    let age_hours = now_secs.saturating_sub(rec.timestamp) as f64 / 3600.0;
    let decay = (-0.01 * age_hours).exp();
    let outcome_boost = rec.outcome.as_ref()
        .map_or(0.0, |o| if o.tests_passed { 1.0 } else { 0.0 });
    w[0] * sim + w[1] * decay + w[2] * outcome_boost
    // w[3] (surprise_signal) is omitted: MemoryRecord has no such field.
}

impl TieredMemoryManager {
    pub fn new(cold: Arc<MemoryStore>, config: TieredConfig) -> Self {
        Self {
            hot: VecDeque::new(),
            warm: HashMap::new(),
            cold,
            config,
            ranker_weights: [0.5, 0.3, 0.15, 0.05],
            recalled_hashes_last: Vec::new(),
            recalled_scores_last: Vec::new(),
        }
    }

    /// Recall up to `k` records across all three memory tiers, ranked by a
    /// learnable linear ranker over four features:
    ///   [cosine_sim, recency_decay, outcome_boost, surprise_signal]
    ///
    /// Strategy:
    /// 1. Query the hot tier (in-memory VecDeque) for all records.
    /// 2. Query the warm tier (HashMap keyed by content_hash).
    /// 3. If fewer than `k` results have been found, fall through to the cold
    ///    tier (LanceDB via `MemoryStore::query_hybrid`).
    /// 4. Merge, deduplicate by content_hash, re-rank by the learned weights,
    ///    and return the top `k`.
    ///
    /// Populates `recalled_hashes_last` with fingerprints of returned records
    /// so `update_ranker` can attribute features to the correct gradient step.
    pub async fn recall_tiered(&mut self, query: &str, k: usize) -> Result<Vec<MemoryRecord>> {
        if k == 0 {
            self.recalled_hashes_last.clear();
            return Ok(vec![]);
        }

        let query_emb = local_embed_text(query);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let w = self.ranker_weights;

        // Each entry: (score, fingerprint, record).
        let mut seen: HashSet<String> = HashSet::new();
        let mut scored: Vec<(f64, u64, MemoryRecord)> = Vec::new();

        // ── Hot tier ────────────────────────────────────────────────────────
        for rec in &self.hot {
            if seen.contains(&rec.content_hash) {
                continue;
            }
            let score = score_record(rec, &query_emb, now, w);
            let fp = hash_to_u64(&rec.content_hash);
            seen.insert(rec.content_hash.clone());
            scored.push((score, fp, rec.clone()));
        }

        // ── Warm tier ───────────────────────────────────────────────────────
        for (hash, (rec, _access_ts)) in &self.warm {
            if seen.contains(hash) {
                continue;
            }
            let score = score_record(rec, &query_emb, now, w);
            let fp = hash_to_u64(hash);
            seen.insert(hash.clone());
            scored.push((score, fp, rec.clone()));
        }

        // ── Cold tier (LanceDB) — only if hot+warm didn't fill quota ────────
        if scored.len() < k {
            let cold_limit = k - scored.len();
            // Cold tier may be uninitialized or empty; degrade gracefully.
            if let Ok(cold_results) = self.cold.query_hybrid(DEFAULT_TABLE, query, cold_limit * 2).await {
                for (rec, _dist) in cold_results {
                    if seen.contains(&rec.content_hash) {
                        continue;
                    }
                    let score = score_record(&rec, &query_emb, now, w);
                    let fp = hash_to_u64(&rec.content_hash);
                    seen.insert(rec.content_hash.clone());
                    scored.push((score, fp, rec));
                }
            }
        }

        // ── Rank and return top-k ────────────────────────────────────────────
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(f64, u64, MemoryRecord)> = scored.into_iter().take(k).collect();

        self.recalled_hashes_last = top_k.iter().map(|(_, fp, _)| *fp).collect();
        self.recalled_scores_last = top_k.iter().map(|(s, _, _)| *s).collect();

        Ok(top_k.into_iter().map(|(_, _, r)| r).collect())
    }

    /// Update learnable ranker weights based on turn outcome feedback.
    ///
    /// Positive outcomes (TestsPassed, AdvancedConvergence, Compiled) apply a
    /// small positive gradient; negative outcomes (Rejected, VerificationFailed,
    /// RolledBack) apply a small negative gradient.  The gradient magnitude is
    /// proportional to the average feature value of the recalled records, so
    /// weights that drove high-scoring recalls receive stronger updates.
    ///
    /// After each update, weights are clipped to [0.01, 0.99] and renormalised
    /// to sum to 1.0.
    ///
    /// Call this after a turn completes and its outcome is known, passing the
    /// content-hash fingerprints stored in `recalled_hashes_last`.
    pub fn update_ranker(&mut self, recalled_hashes: &[u64], turn_outcome: TurnOutcome) {
        if recalled_hashes.is_empty() {
            return;
        }

        // Compute per-feature averages across all recalled records in hot+warm.
        // We only have feature values for records still resident in memory; cold
        // records are skipped (their feature values would require re-embedding).
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Build a fingerprint -> cosine_sim lookup from scores captured at recall time.
        let score_map: HashMap<u64, f64> = recalled_hashes
            .iter()
            .copied()
            .zip(self.recalled_scores_last.iter().copied())
            .collect();
        let fallback_sim = if self.recalled_scores_last.is_empty() {
            0.5
        } else {
            self.recalled_scores_last.iter().sum::<f64>() / self.recalled_scores_last.len() as f64
        };

        let hash_set: HashSet<u64> = recalled_hashes.iter().copied().collect();
        let mut feature_sums = [0.0_f64; 4];
        let mut count = 0usize;

        for rec in &self.hot {
            let fp = hash_to_u64(&rec.content_hash);
            if !hash_set.contains(&fp) {
                continue;
            }
            let sim = *score_map.get(&fp).unwrap_or(&fallback_sim);
            let age_hours = now.saturating_sub(rec.timestamp) as f64 / 3600.0;
            let decay = (-0.01 * age_hours).exp();
            let outcome_boost = rec.outcome.as_ref()
                .map_or(0.0, |o| if o.tests_passed { 1.0 } else { 0.0 });
            let surprise = 0.0_f64;
            feature_sums[0] += sim;
            feature_sums[1] += decay;
            feature_sums[2] += outcome_boost;
            feature_sums[3] += surprise;
            count += 1;
        }
        for (rec, _) in self.warm.values() {
            let fp = hash_to_u64(&rec.content_hash);
            if !hash_set.contains(&fp) {
                continue;
            }
            let sim = *score_map.get(&fp).unwrap_or(&fallback_sim);
            let age_hours = now.saturating_sub(rec.timestamp) as f64 / 3600.0;
            let decay = (-0.01 * age_hours).exp();
            let outcome_boost = rec.outcome.as_ref()
                .map_or(0.0, |o| if o.tests_passed { 1.0 } else { 0.0 });
            let surprise = 0.0_f64;
            feature_sums[0] += sim;
            feature_sums[1] += decay;
            feature_sums[2] += outcome_boost;
            feature_sums[3] += surprise;
            count += 1;
        }

        if count == 0 {
            return;
        }

        let feature_avgs: [f64; 4] = [
            feature_sums[0] / count as f64,
            feature_sums[1] / count as f64,
            feature_sums[2] / count as f64,
            feature_sums[3] / count as f64,
        ];

        let is_positive = matches!(
            turn_outcome,
            TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence | TurnOutcome::Compiled
        );
        let is_negative = matches!(
            turn_outcome,
            TurnOutcome::Rejected | TurnOutcome::VerificationFailed | TurnOutcome::RolledBack
        );

        if is_positive {
            for (i, avg) in feature_avgs.iter().enumerate() {
                self.ranker_weights[i] += 0.01 * avg;
            }
        } else if is_negative {
            for (i, avg) in feature_avgs.iter().enumerate() {
                self.ranker_weights[i] -= 0.005 * avg;
            }
        } else {
            // Stalled / Unknown — no gradient
            return;
        }

        // Clip to [0.01, 0.99]
        for w in &mut self.ranker_weights {
            *w = w.clamp(0.01, 0.99);
        }

        // Renormalise to sum to 1.0
        let total: f64 = self.ranker_weights.iter().sum();
        if total > 0.0 {
            for w in &mut self.ranker_weights {
                *w /= total;
            }
        }
    }
}

// ── Local embedding helpers (deterministic, no model required) ───────────────

pub fn local_embed_text(text: &str) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    let hash_bytes = hash.as_slice();

    let mut embedding = Vec::with_capacity(EMBEDDING_DIM);
    for i in 0..EMBEDDING_DIM {
        let byte_idx = i % 32;
        let cycle = i / 32;
        let seed = u32::from_le_bytes([
            hash_bytes[byte_idx],
            hash_bytes[(byte_idx + 1) % 32],
            hash_bytes[(byte_idx + 2) % 32],
            hash_bytes[(byte_idx + 3) % 32],
        ]);
        let shifted = seed.wrapping_mul(2654435761).wrapping_add(cycle as u32);
        let normalized = ((shifted as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        embedding.push(normalized);
    }

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

pub fn local_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}
