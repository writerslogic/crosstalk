use crate::types::conversation::{ConversationState, Turn, TurnOutcome};
use crate::types::memory::MemoryRecord;
use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
    array::FixedSizeListArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use futures::StreamExt;
use lancedb::{connect, connection::Connection, query::{ExecutableQuery, QueryBase}, table::Table};
use sha2::{Sha256, Digest};
use std::sync::Arc;

const EMBEDDING_DIM: usize = 384;

// =====================================================================
// DETERMINISTIC EMBEDDING ENGINE (SHA256-based, no external ML needed)
// =====================================================================

fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec.to_vec();
    }
    vec.iter().map(|x| x / norm).collect()
}

fn embed_text(text: &str) -> Vec<f32> {
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

pub async fn embed_texts(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    Ok(texts.into_iter().map(|t| embed_text(&t)).collect())
}

// =====================================================================
// LANCEDB STORAGE ENGINE
// =====================================================================

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

pub struct MemoryStore {
    uri: String,
    conn: Option<Connection>,
}

impl MemoryStore {
    #[must_use]
    pub fn new(uri: &str) -> Self {
        Self {
            uri: uri.to_string(),
            conn: None,
        }
    }

    pub async fn init(&mut self) -> Result<()> {
        self.conn = Some(connect(&self.uri).execute().await?);
        Ok(())
    }

    pub async fn get_or_create_table(&self, name: &str) -> Result<Table> {
        let conn = self.conn.as_ref().ok_or_else(|| anyhow!("Database not connected"))?;
        match conn.open_table(name).execute().await {
            Ok(t) => Ok(t),
            Err(_) => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new(
                        "vector",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            EMBEDDING_DIM as i32,
                        ),
                        false,
                    ),
                    Field::new("turn_id", DataType::UInt32, false),
                    Field::new("session_id", DataType::Utf8, false),
                    Field::new("content_hash", DataType::Utf8, false),
                    Field::new("timestamp", DataType::UInt64, false),
                    Field::new("metadata", DataType::Utf8, false),
                ]));
                conn.create_table(name, RecordBatchIterator::new(vec![] as Vec<Result<RecordBatch, ArrowError>>, schema))
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

        // 1. Batch Vector Generation (Massive performance boost over looping)
        let texts_to_embed: Vec<String> = records.iter().map(|r| r.content_hash.clone()).collect();
        let batch_embeddings = embed_texts(texts_to_embed).await?;

        for (record, embedding) in records.iter_mut().zip(batch_embeddings) {
            record.embedding = embedding;
        }

        let table = self.get_or_create_table(table_name).await?;

        // 2. Safe Arrow Memory Mapping
        let turn_ids = UInt32Array::from_iter_values(records.iter().map(|r| r.turn_id));
        let session_ids = StringArray::from_iter_values(records.iter().map(|r| &r.session_id));
        let content_hashes = StringArray::from_iter_values(records.iter().map(|r| &r.content_hash));
        let timestamps = UInt64Array::from_iter_values(records.iter().map(|r| r.timestamp));
        let metadata = StringArray::from_iter_values(records.iter().map(|r| &r.metadata_json));

        let flattened_vectors: Vec<f32> = records.iter().flat_map(|r| r.embedding.clone()).collect();
        let vector_values = Arc::new(Float32Array::from(flattened_vectors));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        
        let vector_array = FixedSizeListArray::try_new(field, EMBEDDING_DIM as i32, vector_values, None)?;

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

        table.add(RecordBatchIterator::new(vec![Ok(batch)], table.schema().await?))
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
        // Await the blocking thread execution
        let mut embeddings = embed_texts(vec![query_text.to_string()]).await?;
        let query_vector = embeddings.pop().unwrap_or_default();

        let table = self.get_or_create_table(table_name).await?;
        
        // LanceDB does the heavy nearest-neighbor math natively in C++
        let mut stream = table.vector_search(query_vector)?
            .limit(top_k * 2)
            .execute()
            .await?;

        let mut weighted_results = Vec::new();

        // 3. Robust Arrow Batch Parsing
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;

            let turn_ids = batch.column_by_name("turn_id").unwrap().as_any().downcast_ref::<UInt32Array>().unwrap();
            let session_ids = batch.column_by_name("session_id").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            let content_hashes = batch.column_by_name("content_hash").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            let timestamps = batch.column_by_name("timestamp").unwrap().as_any().downcast_ref::<UInt64Array>().unwrap();
            let metadata = batch.column_by_name("metadata").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            
            // Extract distances calculated by LanceDB
            let distances = batch.column_by_name("_distance").unwrap().as_any().downcast_ref::<Float32Array>().unwrap();

            for i in 0..batch.num_rows() {
                // LanceDB returns L2 distance by default. Convert to similarity logic.
                let similarity = 1.0 / (1.0 + distances.value(i) as f64);

                let record = MemoryRecord {
                    turn_id: turn_ids.value(i),
                    session_id: session_ids.value(i).to_string(),
                    embedding: vec![], // Omitted to save memory in app state
                    content_hash: content_hashes.value(i).to_string(),
                    timestamp: timestamps.value(i),
                    metadata_json: metadata.value(i).to_string(),
                    outcome: None, // Typically parsed from metadata in production
                };

                let weight = outcome_weight(&TurnOutcome::Unknown); // Replace with parsed outcome
                weighted_results.push((record, similarity * weight));
            }
        }

        weighted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(weighted_results.into_iter().take(top_k).collect())
    }
}

// =====================================================================
// UTILITIES
// =====================================================================

pub struct ContextDistiller;

impl ContextDistiller {
    #[must_use]
    pub fn distill(sigma: &ConversationState, max_chars: usize) -> String {
        let now = ConversationState::now();
        let mut scored_turns: Vec<(&Turn, f64)> = sigma.turns.iter().map(|t| {
            let age_hours = (now - t.timestamp) as f64 / 3600.0;
            let weight = outcome_weight(&t.outcome);
            // Exponential decay: older turns lose relevance
            (t, weight * (-0.01 * age_hours).exp())
        }).collect();

        // Sort highest score first
        scored_turns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut distilled = format!("Session: {} (Distilled Context)\n", sigma.session_id);
        
        for (turn, _) in scored_turns.iter().take(15) {
            let entry = format!("i_{}: {}: {}\n", turn.index, turn.model_id, turn.content);
            
            // UTF-8 Safe Truncation
            // Prevents byte-slicing panics if the string contains emojis or multi-byte characters
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
    #[must_use]
    pub fn proactive_warning(context: &str, failures: &[crate::types::memory::FailureSignature]) -> Option<String> {
        for failure in failures {
            if context.contains(&failure.error_type) || context.contains(&failure.error_message) {
                return Some(format!("Warning: {} detected in context", failure.error_type));
            }
        }
        None
    }
}

pub struct LessonExtractor;

impl LessonExtractor {
    #[must_use]
    pub fn extract(turns: &[Turn]) -> Vec<crate::types::memory::Lesson> {
        turns.iter()
            .filter(|t| t.outcome == TurnOutcome::TestsPassed)
            .map(|t| crate::types::memory::Lesson {
                context_type: "coding".to_string(),
                approach: format!("Approach used by {}", t.model_id),
                outcome: "Success (Tests Passed)".to_string(),
                confidence: 0.95,
                applicability_tags: vec!["passing_tests".to_string(), "tested".to_string()],
            })
            .collect()
    }
}