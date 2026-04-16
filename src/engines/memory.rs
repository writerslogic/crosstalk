use std::collections::VecDeque;
use crate::types::conversation::{ConversationState, Turn, TurnOutcome};
pub use crate::types::memory::OutcomeRecord;
use crate::types::memory::{
    DeletionLogEntry, MemoryRecord, MemoryStoreStats, SnapshotBundle, SnapshotMetadata,
};
use anyhow::{Context, Result, anyhow};
use arrow_array::{
    Array, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
    array::FixedSizeListArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use fastembed::{ExecutionProviderDispatch, InitOptions, TextEmbedding};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use futures::StreamExt;
use lancedb::{
    connect,
    connection::Connection,
    query::{ExecutableQuery, QueryBase},
    table::Table,
};
use ort::{CPUExecutionProvider, CoreMLExecutionProvider};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::panic;
use std::sync::Arc;
use std::sync::OnceLock;

const EMBEDDING_DIM: usize = 384;
const DEFAULT_TABLE: &str = "memory";
const GZIP_THRESHOLD: usize = 10 * 1024 * 1024;
const KMEANS_ITERATIONS: usize = 50;

fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec.to_vec();
    }
    vec.iter().map(|x| x / norm).collect()
}

static EMBEDDER: OnceLock<Option<TextEmbedding>> = OnceLock::new();

fn get_embedder() -> Option<&'static TextEmbedding> {
    EMBEDDER
        .get_or_init(|| {
            let result = panic::catch_unwind(|| {
                let options = InitOptions {
                    execution_providers: vec![
                        ExecutionProviderDispatch::from(CoreMLExecutionProvider::default()),
                        ExecutionProviderDispatch::from(CPUExecutionProvider::default()),
                    ],
                    ..Default::default()
                };
                TextEmbedding::try_new(options)
            });
            match result {
                Ok(Ok(model)) => Some(model),
                _ => None,
            }
        })
        .as_ref()
}

fn embed_text_hash(text: &str) -> Vec<f32> {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    let hash_bytes = hash.as_slice();

    let mut embedding = Vec::with_capacity(EMBEDDING_DIM);
    for i in 0..EMBEDDING_DIM {
        let byte_idx = i % 32;
        let seed = u32::from_le_bytes([
            hash_bytes[byte_idx],
            hash_bytes[(byte_idx + 1) % 32],
            hash_bytes[(byte_idx + 2) % 32],
            hash_bytes[(byte_idx + 3) % 32],
        ]);
        let rng_value = seed.wrapping_mul(1103515245).wrapping_add(12345) as f32 / u32::MAX as f32;
        embedding.push(rng_value * 2.0 - 1.0);
    }

    normalize_vector(&embedding)
}

fn embed_text(text: &str) -> Vec<f32> {
    if let Some(model) = get_embedder()
        && let Ok(mut vecs) = model.embed(vec![text], None)
        && let Some(v) = vecs.pop()
        && v.len() == EMBEDDING_DIM
    {
        return normalize_vector(&v);
    }
    embed_text_hash(text)
}

async fn embed_text_async(text: String) -> Vec<f32> {
    tokio::task::spawn_blocking(move || embed_text(&text))
        .await
        .unwrap_or_else(|_| embed_text_hash(""))
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dir_size(path: &str) -> u64 {
    dir_size_depth(path, 0)
}

fn dir_size_depth(path: &str, depth: usize) -> u64 {
    if depth > 20 {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir()
                    && let Some(sub) = entry.path().to_str()
                {
                    total += dir_size_depth(sub, depth + 1);
                }
            }
        }
    }
    total
}

fn record_weight(r: &MemoryRecord) -> f64 {
    match &r.outcome {
        Some(o) if o.tests_passed => 2.0,
        Some(o) if o.compiled => 1.5,
        Some(o) if o.was_rolled_back => 0.3,
        Some(_) => 1.0,
        None => 1.0,
    }
}

fn batch_to_records(batch: &RecordBatch, dim: usize) -> Result<Vec<MemoryRecord>> {
    let turn_ids = batch
        .column_by_name("turn_id")
        .ok_or_else(|| anyhow!("missing column: turn_id"))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| anyhow!("type error: turn_id"))?;
    let session_ids = batch
        .column_by_name("session_id")
        .ok_or_else(|| anyhow!("missing column: session_id"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("type error: session_id"))?;
    let content_hashes = batch
        .column_by_name("content_hash")
        .ok_or_else(|| anyhow!("missing column: content_hash"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("type error: content_hash"))?;
    let timestamps = batch
        .column_by_name("timestamp")
        .ok_or_else(|| anyhow!("missing column: timestamp"))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow!("type error: timestamp"))?;
    let metadata_col = batch
        .column_by_name("metadata")
        .ok_or_else(|| anyhow!("missing column: metadata"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("type error: metadata"))?;
    let vector_col = batch
        .column_by_name("vector")
        .ok_or_else(|| anyhow!("missing column: vector"))?
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("type error: vector"))?;

    (0..batch.num_rows())
        .map(|i| {
            let values = vector_col.value(i);
            let float_vals = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("type error: vector float elements"))?;
            if float_vals.len() < dim {
                return Err(anyhow!(
                    "embedding dimension mismatch: expected {}, got {}",
                    dim,
                    float_vals.len()
                ));
            }
            let embedding: Vec<f32> = (0..dim).map(|j| float_vals.value(j)).collect();
            let meta_str = metadata_col.value(i).to_string();
            let is_neg = meta_str.contains("\"is_negative\":true");
            Ok(MemoryRecord {
                turn_id: turn_ids.value(i),
                session_id: session_ids.value(i).to_string(),
                embedding,
                content_hash: content_hashes.value(i).to_string(),
                timestamp: timestamps.value(i),
                metadata_json: meta_str,
                outcome: None,
                is_negative: is_neg,
            })
        })
        .collect()
}

pub async fn embed_texts(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    if let Some(model) = get_embedder() {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        if let Ok(vecs) = model.embed(refs, None) {
            return Ok(vecs.into_iter().map(|v| normalize_vector(&v)).collect());
        }
    }
    Ok(texts.iter().map(|t| embed_text_hash(t)).collect())
}

fn outcome_weight(outcome: &TurnOutcome) -> f64 {
    match outcome {
        TurnOutcome::TestsPassed => 2.0,
        TurnOutcome::AdvancedConvergence => 1.8,
        TurnOutcome::Compiled => 1.5,
        TurnOutcome::Unknown => 1.0,
        TurnOutcome::Stalled => 0.5,
        TurnOutcome::RolledBack => 0.3,
        TurnOutcome::Rejected => 0.2,
    }
}

fn parse_outcome_from_metadata(meta: &str) -> TurnOutcome {
    if let Some(start) = meta.find("\"outcome\":\"") {
        let rest = &meta[start + 11..];
        if let Some(end) = rest.find('"') {
            return match &rest[..end] {
                "TestsPassed" => TurnOutcome::TestsPassed,
                "AdvancedConvergence" => TurnOutcome::AdvancedConvergence,
                "Compiled" => TurnOutcome::Compiled,
                "Stalled" => TurnOutcome::Stalled,
                "RolledBack" => TurnOutcome::RolledBack,
                "Rejected" => TurnOutcome::Rejected,
                _ => TurnOutcome::Unknown,
            };
        }
    }
    TurnOutcome::Unknown
}

pub struct MemoryStore {
    uri: String,
    conn: Option<Connection>,
    cluster_assignments: HashMap<u32, usize>,
    pub deletion_log: Vec<DeletionLogEntry>,
    pub embedding_dim: usize,
}

impl MemoryStore {
    #[must_use]
    pub fn new(uri: &str) -> Self {
        Self {
            uri: uri.to_string(),
            conn: None,
            cluster_assignments: HashMap::new(),
            deletion_log: Vec::new(),
            embedding_dim: EMBEDDING_DIM,
        }
    }

    #[must_use]
    pub fn new_with_dim(uri: &str, dim: usize) -> Self {
        Self {
            uri: uri.to_string(),
            conn: None,
            cluster_assignments: HashMap::new(),
            deletion_log: Vec::new(),
            embedding_dim: dim,
        }
    }

    pub async fn init(&mut self) -> Result<()> {
        self.conn = Some(connect(&self.uri).execute().await?);
        Ok(())
    }

    pub async fn get_or_create_table(&self, name: &str) -> Result<Table> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| anyhow!("Database not connected"))?;
        match conn.open_table(name).execute().await {
            Ok(t) => Ok(t),
            Err(_) => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new(
                        "vector",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            self.embedding_dim as i32,
                        ),
                        false,
                    ),
                    Field::new("turn_id", DataType::UInt32, false),
                    Field::new("session_id", DataType::Utf8, false),
                    Field::new("content_hash", DataType::Utf8, false),
                    Field::new("timestamp", DataType::UInt64, false),
                    Field::new("metadata", DataType::Utf8, false),
                ]));
                conn.create_table(
                    name,
                    RecordBatchIterator::new(
                        vec![] as Vec<Result<RecordBatch, ArrowError>>,
                        schema,
                    ),
                )
                .execute()
                .await
                .context("Failed to create LanceDB table")
            }
        }
    }

    pub async fn insert(&self, table_name: &str, mut records: Vec<MemoryRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let texts_to_embed: Vec<String> = records.iter().map(|r| r.content_hash.clone()).collect();
        let batch_embeddings = embed_texts(texts_to_embed).await?;
        if batch_embeddings.len() != records.len() {
            return Err(anyhow!(
                "embed_texts returned {} vectors for {} records",
                batch_embeddings.len(),
                records.len()
            ));
        }

        for (record, embedding) in records.iter_mut().zip(batch_embeddings) {
            record.embedding = embedding;
        }

        self.insert_raw(table_name, records).await
    }

    async fn insert_raw(&self, table_name: &str, records: Vec<MemoryRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let table = self.get_or_create_table(table_name).await?;

        let turn_ids = UInt32Array::from_iter_values(records.iter().map(|r| r.turn_id));
        let session_ids = StringArray::from_iter_values(records.iter().map(|r| &r.session_id));
        let content_hashes = StringArray::from_iter_values(records.iter().map(|r| &r.content_hash));
        let timestamps = UInt64Array::from_iter_values(records.iter().map(|r| r.timestamp));
        let metadata = StringArray::from_iter_values(records.iter().map(|r| &r.metadata_json));

        let flattened: Vec<f32> = records
            .iter()
            .flat_map(|r| r.embedding.iter().copied())
            .collect();
        let vector_values = Arc::new(Float32Array::from(flattened));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array =
            FixedSizeListArray::try_new(field, self.embedding_dim as i32, vector_values, None)?;

        let batch = RecordBatch::try_new(
            table.schema().await?,
            vec![
                Arc::new(vector_array),
                Arc::new(turn_ids),
                Arc::new(session_ids),
                Arc::new(content_hashes),
                Arc::new(timestamps),
                Arc::new(metadata),
            ],
        )?;

        table
            .add(RecordBatchIterator::new(
                vec![Ok(batch)],
                table.schema().await?,
            ))
            .execute()
            .await?;

        Ok(())
    }

    pub async fn query_weighted(
        &self,
        table_name: &str,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f64)>> {
        let mut embeddings = embed_texts(vec![query_text.to_string()]).await?;
        let query_vector = embeddings
            .pop()
            .ok_or_else(|| anyhow!("embed_texts returned no vectors"))?;

        let table = self.get_or_create_table(table_name).await?;

        let mut stream = table
            .vector_search(query_vector)?
            .limit(top_k * 2)
            .execute()
            .await?;

        let mut weighted_results = Vec::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;

            macro_rules! col {
                ($name:expr, $ty:ty) => {
                    batch
                        .column_by_name($name)
                        .ok_or_else(|| anyhow!("missing column: {}", $name))?
                        .as_any()
                        .downcast_ref::<$ty>()
                        .ok_or_else(|| anyhow!("column {} has wrong type", $name))?
                };
            }
            let turn_ids = col!("turn_id", UInt32Array);
            let session_ids = col!("session_id", StringArray);
            let content_hashes = col!("content_hash", StringArray);
            let timestamps = col!("timestamp", UInt64Array);
            let metadata = col!("metadata", StringArray);
            let distances = col!("_distance", Float32Array);

            for i in 0..batch.num_rows() {
                let similarity = 1.0 / (1.0 + distances.value(i) as f64);

                let meta_str = metadata.value(i).to_string();
                let is_neg = meta_str.contains("\"is_negative\":true");
                let parsed_outcome = parse_outcome_from_metadata(&meta_str);
                let ts = timestamps.value(i);
                let age_days = (now.saturating_sub(ts)) as f64 / 86400.0;
                let decay = (-age_days / 30.0).exp();
                let record = MemoryRecord {
                    turn_id: turn_ids.value(i),
                    session_id: session_ids.value(i).to_string(),
                    embedding: vec![],
                    content_hash: content_hashes.value(i).to_string(),
                    timestamp: ts,
                    metadata_json: meta_str,
                    outcome: None,
                    is_negative: is_neg,
                };

                let weight = outcome_weight(&parsed_outcome);
                weighted_results.push((record, similarity * weight * decay));
            }
        }

        weighted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(weighted_results.into_iter().take(top_k).collect())
    }
    /// Performs a hybrid ANN + Keyword search.
    /// Result Score = 0.7 * VectorSimilarity + 0.3 * KeywordMatchScore.
    pub async fn query_hybrid(
        &self,
        table_name: &str,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f64)>> {
        let weighted_results = self.query_weighted(table_name, query_text, top_k * 3).await?;
        
        let query_keywords: HashSet<String> = query_text
            .to_lowercase()
            .split_whitespace()
            .filter(|s| s.len() > 3)
            .map(|s| s.to_string())
            .collect();

        let mut hybrid_results = Vec::new();
        for (record, vector_score) in weighted_results {
            let record_text = record.content_hash.to_lowercase();
            let mut match_count = 0;
            for kw in &query_keywords {
                if record_text.contains(kw) {
                    match_count += 1;
                }
            }
            
            let keyword_score = if !query_keywords.is_empty() {
                match_count as f64 / query_keywords.len() as f64
            } else {
                0.0
            };

            let final_score = 0.7 * vector_score + 0.3 * keyword_score;
            hybrid_results.push((record, final_score));
        }

        hybrid_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(hybrid_results.into_iter().take(top_k).collect())
    }
}


    fn validate_session_id(session_id: &str) -> Result<()> {
        if session_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            Ok(())
        } else {
            Err(anyhow!("Invalid session_id: must match [a-zA-Z0-9_-]+"))
        }
    }

    fn session_filter(session_id: &str) -> Result<String> {
        Self::validate_session_id(session_id)?;
        Ok(format!("session_id = '{session_id}'"))
    }

    fn turn_session_filter(turn_id: u32, session_id: &str) -> Result<String> {
        Self::validate_session_id(session_id)?;
        Ok(format!(
            "turn_id = {turn_id} AND session_id = '{session_id}'"
        ))
    }

    fn turn_ids_filter(turn_ids: &[u32]) -> String {
        let ids_str = turn_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("turn_id IN ({ids_str})")
    }

    async fn scan_session(&self, table_name: &str, session_id: &str) -> Result<Vec<MemoryRecord>> {
        let filter = Self::session_filter(session_id)?;
        let table = self.get_or_create_table(table_name).await?;
        let mut stream = table.query().only_if(filter).execute().await?;

        let mut records = Vec::new();
        while let Some(batch_res) = stream.next().await {
            records.extend(batch_to_records(&batch_res?, self.embedding_dim)?);
        }
        Ok(records)
    }

    fn snapshot_path(session_id: &str) -> Result<std::path::PathBuf> {
        let base = std::env::var("CROSSTALK_MEMORY_DIR").unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                // fallback: HOME/USERPROFILE unset; /tmp is insecure on shared systems
                .unwrap_or_else(|_| "/tmp".to_string());
            format!("{home}/.crosstalk-memory")
        });
        Ok(std::path::Path::new(&base).join(format!("{session_id}.snapshot")))
    }

    pub async fn snapshot(&self, session_id: &str) -> Result<Vec<u8>> {
        let records = self.scan_session(DEFAULT_TABLE, session_id).await?;

        let records_bytes =
            bincode::serialize(&records).map_err(|e| anyhow!("Bincode serialize failed: {e}"))?;

        let content_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(&records_bytes);
            h.finalize().into()
        };

        let compressed = records_bytes.len() > GZIP_THRESHOLD;
        let metadata = SnapshotMetadata {
            session_id: session_id.to_string(),
            created_at: ConversationState::now(),
            record_count: records.len(),
            content_hash,
            compressed,
        };

        let bundle = SnapshotBundle { metadata, records };
        let bundle_bytes = bincode::serialize(&bundle)
            .map_err(|e| anyhow!("Bincode bundle serialize failed: {e}"))?;

        let final_bytes = if compressed {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&bundle_bytes)?;
            encoder.finish()?
        } else {
            bundle_bytes
        };

        let path = Self::snapshot_path(session_id)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, &final_bytes).await?;

        Ok(final_bytes)
    }

    pub async fn restore(&mut self, session_id: &str) -> Result<()> {
        let path = Self::snapshot_path(session_id)?;
        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("Snapshot not found: {}", path.display()))?;

        const MAX_SNAPSHOT_BYTES: u64 = 256 * 1024 * 1024;
        let bundle_bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
            let decoder = GzDecoder::new(&bytes[..]);
            let mut decompressed = Vec::new();
            decoder
                .take(MAX_SNAPSHOT_BYTES)
                .read_to_end(&mut decompressed)?;
            decompressed
        } else {
            bytes
        };

        let bundle: SnapshotBundle = bincode::deserialize(&bundle_bytes)
            .map_err(|e| anyhow!("Bincode deserialize failed: {e}"))?;

        let records_bytes = bincode::serialize(&bundle.records)
            .map_err(|e| anyhow!("Hash verification serialize failed: {e}"))?;
        let computed: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(&records_bytes);
            h.finalize().into()
        };
        if computed != bundle.metadata.content_hash {
            return Err(anyhow!("Snapshot integrity check failed: hash mismatch"));
        }

        if !bundle.records.is_empty() {
            let filter = Self::session_filter(session_id)?;
            let table = self.get_or_create_table(DEFAULT_TABLE).await?;
            table
                .delete(&filter)
                .await
                .context("Failed to clear existing records before restore")?;
            self.insert_raw(DEFAULT_TABLE, bundle.records).await?;
        }

        Ok(())
    }

    pub async fn forget(&mut self, turn_id: u32, session_id: &str) -> Result<()> {
        let filter = Self::turn_session_filter(turn_id, session_id)?;
        let table = self.get_or_create_table(DEFAULT_TABLE).await?;
        table
            .delete(&filter)
            .await
            .context("Failed to delete record")?;

        self.deletion_log.push(DeletionLogEntry {
            turn_id,
            session_id: session_id.to_string(),
            deleted_at: ConversationState::now(),
        });
        self.cluster_assignments.remove(&turn_id);

        Ok(())
    }

    pub fn set_cluster_assignments(&mut self, clusters: &[Vec<u32>]) {
        self.cluster_assignments.clear();
        for (cluster_id, turn_ids) in clusters.iter().enumerate() {
            for &turn_id in turn_ids {
                self.cluster_assignments.insert(turn_id, cluster_id);
            }
        }
    }

    pub async fn recall_by_cluster(&self, cluster_id: usize) -> Result<Vec<MemoryRecord>> {
        let turn_ids: Vec<u32> = self
            .cluster_assignments
            .iter()
            .filter(|(_, c)| **c == cluster_id)
            .map(|(&id, _)| id)
            .collect();

        if turn_ids.is_empty() {
            return Ok(vec![]);
        }

        let filter = Self::turn_ids_filter(&turn_ids);
        let table = self.get_or_create_table(DEFAULT_TABLE).await?;
        let mut stream = table.query().only_if(filter).execute().await?;

        let mut records = Vec::new();
        while let Some(batch_res) = stream.next().await {
            records.extend(batch_to_records(&batch_res?, self.embedding_dim)?);
        }

        records.sort_by(|a, b| {
            record_weight(b)
                .partial_cmp(&record_weight(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(records)
    }

    pub async fn stats(&self) -> Result<MemoryStoreStats> {
        let (total_records, unique_sessions) = if self.conn.is_some() {
            match self.get_or_create_table(DEFAULT_TABLE).await {
                Ok(table) => {
                    let mut stream = table.query().execute().await?;
                    let mut count = 0usize;
                    let mut sessions = HashSet::new();
                    while let Some(batch_res) = stream.next().await {
                        let batch = batch_res?;
                        count += batch.num_rows();
                        if let Some(col) = batch.column_by_name("session_id")
                            && let Some(arr) = col.as_any().downcast_ref::<StringArray>()
                        {
                            for i in 0..arr.len() {
                                sessions.insert(arr.value(i).to_string());
                            }
                        }
                    }
                    (count, sessions.len())
                }
                Err(e) => return Err(e),
            }
        } else {
            (0, 0)
        };

        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        for &c in self.cluster_assignments.values() {
            *cluster_counts.entry(c).or_default() += 1;
        }
        let avg_cluster_size = if cluster_counts.is_empty() {
            0.0
        } else {
            cluster_counts.values().map(|&c| c as f64).sum::<f64>() / cluster_counts.len() as f64
        };

        Ok(MemoryStoreStats {
            total_records,
            unique_sessions,
            avg_cluster_size,
            storage_size: dir_size(&self.uri),
        })
    }

pub struct ContextDistiller;

impl ContextDistiller {
    #[must_use]
    pub fn distill(sigma: &ConversationState, max_chars: usize) -> String {
        Self::distill_with_decay(sigma, max_chars, 0.01)
    }

    #[must_use]
    pub fn distill_with_decay(
        sigma: &ConversationState,
        max_chars: usize,
        decay_rate: f64,
    ) -> String {
        let now = ConversationState::now();
        let mut scored_turns: Vec<(&Turn, f64)> = sigma
            .turns
            .iter()
            .map(|t| {
                let age_hours = (now - t.timestamp) as f64 / 3600.0;
                let weight = outcome_weight(&t.outcome);
                (t, weight * (-decay_rate * age_hours).exp())
            })
            .collect();

        scored_turns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut distilled = format!("Session: {} (Distilled Context)\n", sigma.session_id);

        for (turn, _) in scored_turns.iter().take(15) {
            let entry = format!("i_{}: {}: {}\n", turn.index, turn.model_id, turn.content);

            if distilled.len() + entry.len() > max_chars {
                let remaining = max_chars.saturating_sub(distilled.len());
                let safe_entry: String = entry.chars().take(remaining).collect();
                distilled.push_str(&safe_entry);
                break;
            }
            distilled.push_str(&entry);
        }

        distilled
    }

    /// Distills context by clustering turns to ensure a representative sample of all discussion threads.
    #[must_use]
    pub fn distill_diverse(sigma: &ConversationState, max_chars: usize, k: usize) -> String {
        if sigma.turns.is_empty() {
            return format!("Session: {} (Empty Context)\n", sigma.session_id);
        }

        let k = k.clamp(1, sigma.turns.len());
        let clusters = SemanticClusterer::cluster(&sigma.turns, k).unwrap_or_default();

        let mut picked_indices = HashSet::new();
        for cluster in &clusters {
            // Pick the latest turn from each cluster as the representative.
            if let Some(&idx) = cluster.iter().max() {
                picked_indices.insert(idx);
            }
        }

        let mut distilled = format!(
            "Session: {} (Diverse Distilled Context)\n",
            sigma.session_id
        );
        let mut picked_turns: Vec<&Turn> = sigma
            .turns
            .iter()
            .filter(|t| picked_indices.contains(&t.index))
            .collect();

        // Sort by index to maintain temporal flow.
        picked_turns.sort_by_key(|t| t.index);

        for turn in picked_turns {
            let entry = format!("i_{}: {}: {}\n", turn.index, turn.model_id, turn.content);
            if distilled.len() + entry.len() > max_chars {
                let remaining = max_chars.saturating_sub(distilled.len());
                let safe_entry: String = entry.chars().take(remaining).collect();
                distilled.push_str(&safe_entry);
                break;
            }
            distilled.push_str(&entry);
        }
        distilled
    }
}

pub struct FailurePredictor;

impl FailurePredictor {
    /// Queries the memory bridge for matching antipatterns and returns a preventive warning.
    pub async fn proactive_avoidance(
        context: &str,
        bridge: &mut MemoryBridge,
    ) -> Option<String> {
        let antipatterns = bridge.recall_antipatterns(context, 1).await;
        
        if let Some(failure) = antipatterns.first() {
            // Calculate actual similarity
            let query_emb = embed_text_async(context.to_string()).await;
            let sim = cosine_sim(&query_emb, &failure.embedding);
            
            if sim > 0.8 {
                return Some(format!(
                    "[PROACTIVE AVOIDANCE] A similar approach previously failed (similarity: {:.2}). Antipattern detected: {}. Avoid this trajectory.",
                    sim, failure.content_hash
                ));
            }
        }
        None
    }
}

pub struct LessonExtractor;

impl LessonExtractor {
    /// Extracts structured lessons from a completed session.
    /// Identifies high-performing agent/task pairs and recurring failure modes.
    #[must_use]
    pub fn extract(turns: &[Turn]) -> Vec<crate::types::memory::Lesson> {
        let mut lessons = Vec::new();
        let mut agent_performance: HashMap<String, (u32, u32)> = HashMap::new(); // (success, total)

        for t in turns {
            let stats = agent_performance.entry(t.model_id.clone()).or_insert((0, 0));
            stats.1 += 1;
            if t.outcome == TurnOutcome::TestsPassed || t.outcome == TurnOutcome::AdvancedConvergence {
                stats.0 += 1;
            }
        }

        // 1. Extract Specialist Success Lessons
        for (agent_id, (success, total)) in &agent_performance {
            let rate = *success as f64 / *total as f64;
            if rate > 0.8 && *total >= 3 {
                lessons.push(crate::types::memory::Lesson {
                    context_type: "agent_specialization".to_string(),
                    approach: format!("Deploy {} for high-reliability tasks", agent_id),
                    outcome: format!("Success Rate: {:.0}% over {} turns", rate * 100.0, total),
                    confidence: rate,
                    applicability_tags: vec![agent_id.clone(), "reliable".to_string()],
                });
            }
        }

        // 2. Extract Task-Specific Successes
        for t in turns {
            if t.outcome == TurnOutcome::TestsPassed {
                lessons.push(crate::types::memory::Lesson {
                    context_type: "logic_pattern".to_string(),
                    approach: format!("Successful logic by {}: {}", t.model_id, t.content.chars().take(100).collect::<String>()),
                    outcome: "Verified via tests".to_string(),
                    confidence: 0.9,
                    applicability_tags: vec!["tested".to_string(), t.model_id.clone()],
                });
            }
        }

        lessons
    }
}

struct BridgeSessionContext {
    pub last_recall_turn: Option<u32>,
}

pub struct MemoryBridge {
    sessions: HashMap<String, BridgeSessionContext>,
    records: HashMap<String, Vec<MemoryRecord>>,
}

impl MemoryBridge {
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            records: HashMap::new(),
        }
    }

    pub fn open_session(&mut self, session_id: String) {
        self.sessions
            .entry(session_id)
            .or_insert_with(|| BridgeSessionContext {
                last_recall_turn: None,
            });
    }

    pub fn push_record(&mut self, session_id: &str, mut record: MemoryRecord) {
        if record.embedding.len() != EMBEDDING_DIM {
            record.embedding = embed_text(&record.content_hash);
        }
        self.records
            .entry(session_id.to_string())
            .or_default()
            .push(record);
    }

    pub async fn recall_relevant(
        &mut self,
        current_session_id: &str,
        query_text: &str,
        n_examples: usize,
        current_turn: u32,
    ) -> Result<Vec<MemoryRecord>> {
        if let Some(ctx) = self.sessions.get_mut(current_session_id) {
            if ctx.last_recall_turn == Some(current_turn) {
                return Ok(vec![]);
            }
            ctx.last_recall_turn = Some(current_turn);
        }

        let query_emb = embed_text_async(query_text.to_string()).await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut scored: Vec<(MemoryRecord, f64)> = self
            .records
            .values()
            .flat_map(|recs| recs.iter())
            .filter(|r| !r.is_negative)
            .map(|r| {
                let sim = cosine_sim(&query_emb, &r.embedding) as f64;
                let age_days = (now.saturating_sub(r.timestamp)) as f64 / 86400.0;
                let decay = (-age_days / 30.0).exp();
                (r.clone(), sim * record_weight(r) * decay)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(n_examples)
            .map(|(r, _)| r)
            .collect())
    }

    pub async fn recall_antipatterns(
        &mut self,
        query_text: &str,
        n_examples: usize,
    ) -> Vec<MemoryRecord> {
        let query_emb = embed_text_async(query_text.to_string()).await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut scored: Vec<(MemoryRecord, f64)> = self
            .records
            .values()
            .flat_map(|recs| recs.iter())
            .filter(|r| r.is_negative)
            .map(|r| {
                let sim = cosine_sim(&query_emb, &r.embedding) as f64;
                let age_days = (now.saturating_sub(r.timestamp)) as f64 / 86400.0;
                let decay = (-age_days / 30.0).exp();
                (r.clone(), sim * record_weight(r) * decay)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(n_examples)
            .map(|(r, _)| r)
            .collect()
    }

    pub fn store_failure_lesson(
        &mut self,
        session_id: &str,
        mortem: &crate::types::self_improvement::PostMortem,
    ) {
        let content = format!(
            "ANTIPATTERN: {:?} — {} failures in session {}",
            mortem.root_cause,
            mortem.failure_turn_indices.len(),
            mortem.session_id,
        );
        let embedding = embed_text(&content);
        let metadata = serde_json::json!({
            "is_negative": true,
            "root_cause": format!("{:?}", mortem.root_cause),
            "failure_count": mortem.failure_turn_indices.len(),
        });
        let record = MemoryRecord {
            turn_id: 0,
            session_id: session_id.to_string(),
            embedding,
            content_hash: content,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata_json: metadata.to_string(),
            outcome: Some(OutcomeRecord {
                compiled: false,
                tests_passed: false,
                quality_delta: -1.0,
                was_rolled_back: true,
                convergence_contribution: 0.0,
            }),
            is_negative: true,
        };
        self.records
            .entry(session_id.to_string())
            .or_default()
            .push(record);
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn record_count(&self, session_id: &str) -> usize {
        self.records.get(session_id).map_or(0, |v| v.len())
    }

    pub fn total_record_count(&self) -> usize {
        self.records.values().map(|v| v.len()).sum()
    }

    pub fn take_snapshot(&self, session_id: &str) -> Vec<MemoryRecord> {
        self.records.get(session_id).cloned().unwrap_or_default()
    }

    pub fn index_snapshot(&mut self, session_id: &str, records: Vec<MemoryRecord>) {
        let entry = self.records.entry(session_id.to_string()).or_default();
        entry.extend(records);
    }
}

impl Default for MemoryBridge {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DecayCalibrator {
    decay_rate: f64,
    useful_turn_ages: Vec<f64>,
}

impl DecayCalibrator {
    const DEFAULT_RATE: f64 = 0.01;
    const MIN_SAMPLES: usize = 10;

    #[must_use]
    pub fn new() -> Self {
        Self {
            decay_rate: Self::DEFAULT_RATE,
            useful_turn_ages: Vec::new(),
        }
    }

    /// Record the age (in hours) of a turn that proved useful at retrieval time.
    pub fn record_useful_turn(&mut self, age_hours: f64) {
        if age_hours >= 0.0 {
            self.useful_turn_ages.push(age_hours);
        }
    }

    /// Re-fit the exponential decay rate using MLE: λ = 1 / mean(ages).
    /// Requires at least MIN_SAMPLES observations; otherwise keeps current rate.
    pub fn calibrate(&mut self) {
        if self.useful_turn_ages.len() < Self::MIN_SAMPLES {
            return;
        }
        let mean_age =
            self.useful_turn_ages.iter().sum::<f64>() / self.useful_turn_ages.len() as f64;
        if mean_age > 0.0 {
            self.decay_rate = (1.0 / mean_age).clamp(0.001, 0.1);
        }
    }

    #[must_use]
    pub fn decay_rate(&self) -> f64 {
        self.decay_rate
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.useful_turn_ages.len()
    }
}

impl Default for DecayCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SemanticClusterer;

impl SemanticClusterer {
    /// Automatically select the number of clusters using the elbow method.
    /// Runs k-means for k = 1..=max_k and returns the k at the sharpest inertia drop.
    #[must_use]
    pub fn select_k(turns: &[Turn], max_k: Option<usize>) -> usize {
        let n = turns.len();
        if n <= 1 {
            return 1;
        }
        let max = max_k
            .unwrap_or_else(|| (n as f64).sqrt().ceil() as usize)
            .min(n)
            .max(1);

        if max == 1 {
            return 1;
        }

        let embeddings: Vec<Vec<f32>> = turns.iter().map(|t| embed_text(&t.content)).collect();

        let inertia: Vec<f64> = (1..=max)
            .map(|k| Self::kmeans_inertia(&embeddings, k))
            .collect();

        // Elbow via largest second derivative (curvature) of the inertia curve.
        if inertia.len() < 3 {
            return max;
        }
        let mut best_k = 1usize;
        let mut best_curv = f64::NEG_INFINITY;
        for i in 1..inertia.len() - 1 {
            let curv = inertia[i + 1] - 2.0 * inertia[i] + inertia[i - 1];
            if curv > best_curv {
                best_curv = curv;
                best_k = i + 1; // 1-indexed
            }
        }
        best_k
    }

    fn kmeans_inertia(embeddings: &[Vec<f32>], k: usize) -> f64 {
        if embeddings.is_empty() || k == 0 {
            return 0.0;
        }
        let k = k.min(embeddings.len());
        let dim = embeddings[0].len();

        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| embeddings[i * embeddings.len() / k].clone())
            .collect();
        let mut assignments = vec![0usize; embeddings.len()];

        for _ in 0..KMEANS_ITERATIONS {
            let mut changed = false;
            for (i, emb) in embeddings.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        cosine_sim(emb, a)
                            .partial_cmp(&cosine_sim(emb, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            for (ci, centroid) in centroids.iter_mut().enumerate() {
                let members: Vec<&Vec<f32>> = assignments
                    .iter()
                    .enumerate()
                    .filter(|&(_, &a)| a == ci)
                    .map(|(i, _)| &embeddings[i])
                    .collect();
                if members.is_empty() {
                    continue;
                }
                let mut c = vec![0.0f32; dim];
                for m in &members {
                    for (j, v) in m.iter().enumerate() {
                        c[j] += v;
                    }
                }
                let n = members.len() as f32;
                *centroid = normalize_vector(&c.iter().map(|v| v / n).collect::<Vec<_>>());
            }
        }

        // Inertia: sum of cosine distances (1 - sim) to assigned centroid.
        assignments
            .iter()
            .enumerate()
            .map(|(i, &c)| (1.0 - cosine_sim(&embeddings[i], &centroids[c])) as f64)
            .sum()
    }

    pub fn cluster(turns: &[Turn], k_clusters: usize) -> Result<Vec<Vec<u32>>> {
        if turns.is_empty() || k_clusters == 0 {
            return Ok(vec![]);
        }
        let k = k_clusters.min(turns.len());

        let embeddings: Vec<Vec<f32>> = turns.iter().map(|t| embed_text(&t.content)).collect();

        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| embeddings[i * turns.len() / k].clone())
            .collect();

        let mut assignments = vec![0usize; turns.len()];

        for _ in 0..KMEANS_ITERATIONS {
            let mut changed = false;

            for (i, emb) in embeddings.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        cosine_sim(emb, a)
                            .partial_cmp(&cosine_sim(emb, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            for (ci, centroid_slot) in centroids.iter_mut().enumerate() {
                let cluster_embs: Vec<&Vec<f32>> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| **a == ci)
                    .map(|(i, _)| &embeddings[i])
                    .collect();

                if cluster_embs.is_empty() {
                    continue;
                }

                let mut centroid = vec![0.0f32; EMBEDDING_DIM];
                for emb in &cluster_embs {
                    for (j, v) in emb.iter().enumerate() {
                        centroid[j] += v;
                    }
                }
                let n = cluster_embs.len() as f32;
                for v in centroid.iter_mut() {
                    *v /= n;
                }
                *centroid_slot = normalize_vector(&centroid);
            }
        }

        let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &cluster_id) in assignments.iter().enumerate() {
            clusters[cluster_id].push(turns[i].index);
        }

        Ok(clusters)
    }
}

#[derive(Debug, Clone)]
pub struct TieredConfig {
    pub warm_max_bytes: u64,
    pub warm_ttl_secs: u64,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            warm_max_bytes: 512 * 1024 * 1024,
            warm_ttl_secs: 3600,
        }
    }
}

pub struct TieredMemoryManager {
    hot: VecDeque<MemoryRecord>,
    warm: HashMap<String, (MemoryRecord, u64)>, // session_id -> (record, last_access)
    cold: Arc<MemoryStore>,
    config: TieredConfig,
}

impl TieredMemoryManager {
    pub fn new(cold: Arc<MemoryStore>, config: TieredConfig) -> Self {
        Self {
            hot: VecDeque::new(),
            warm: HashMap::new(),
            cold,
            config,
        }
    }

    pub fn promote_to_hot(&mut self, record: MemoryRecord) {
        if self.hot.len() >= 20 {
            self.hot.pop_front();
        }
        self.hot.push_back(record);
    }

    pub async fn recall_tiered(&mut self, query: &str, k: usize) -> Result<Vec<MemoryRecord>> {
        // 1. Check Hot (Active Context)
        let mut results = Vec::new();
        let query_emb = embed_text_async(query.to_string()).await;
        
        for r in &self.hot {
            let sim = cosine_sim(&query_emb, &r.embedding);
            if sim > 0.9 {
                results.push(r.clone());
            }
        }

        // 2. Check Warm (RAM Cache)
        for (r, _ts) in self.warm.values() {
            let sim = cosine_sim(&query_emb, &r.embedding);
            if sim > 0.85 {
                results.push(r.clone());
            }
        }

        // 3. Fallback to Cold (LanceDB)
        if results.len() < k {
            let cold_results = self.cold.query_weighted(DEFAULT_TABLE, query, k - results.len()).await?;
            for (r, _) in cold_results {
                results.push(r);
            }
        }

        Ok(results.into_iter().take(k).collect())
    }
}
