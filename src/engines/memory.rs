use crate::types::conversation::{ConversationState, Turn, TurnOutcome};
use crate::types::memory::{FailureSignature, Lesson, MemoryRecord};
use anyhow::{Result, anyhow};
use arrow_array::{
    Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lancedb::connect;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use sha2::{Sha256, Digest};
use std::sync::Arc;

const EMBEDDING_DIM: usize = 384;

fn embed_text(text: &str) -> Vec<f32> {
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

    normalize_vector(&embedding)
}

fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f32 {
    if vec_a.len() != vec_b.len() || vec_a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (a, b) in vec_a.iter().zip(vec_b.iter()) {
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot_product / denominator
    }
}

fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|x| x / norm).collect()
    }
}

fn outcome_weight(outcome: &TurnOutcome) -> f64 {
    match outcome {
        TurnOutcome::TestsPassed => 2.0,
        TurnOutcome::Compiled => 1.5,
        TurnOutcome::AdvancedConvergence => 1.8,
        TurnOutcome::RolledBack => 0.3,
        TurnOutcome::Rejected => 0.2,
        TurnOutcome::Stalled => 0.5,
        TurnOutcome::Unknown => 1.0,
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
        let conn = connect(&self.uri)
            .execute()
            .await
            .map_err(|e| anyhow!("Failed to connect to LanceDB: {:?}", e))?;
        self.conn = Some(conn);
        Ok(())
    }

    pub async fn get_or_create_table(&self, name: &str, dim: usize) -> Result<Table> {
        let conn = self.conn.as_ref().ok_or_else(|| anyhow!("Not connected"))?;
        match conn.open_table(name).execute().await {
            Ok(t) => Ok(t),
            Err(_) => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new(
                        "vector",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dim as i32,
                        ),
                        false,
                    ),
                    Field::new("turn_id", DataType::UInt32, false),
                    Field::new("session_id", DataType::Utf8, false),
                    Field::new("content_hash", DataType::Utf8, false),
                    Field::new("timestamp", DataType::UInt64, false),
                    Field::new("metadata", DataType::Utf8, false),
                ]));
                conn.create_table(name, RecordBatchIterator::new(vec![], schema))
                    .execute()
                    .await
                    .map_err(|e| anyhow!("Failed to create table {}: {:?}", name, e))
            }
        }
    }

    pub async fn insert(&self, table_name: &str, mut records: Vec<MemoryRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        for record in &mut records {
            if record.embedding.is_empty() || record.embedding.iter().all(|&x| x == 0.0) {
                record.embedding = embed_text(&record.content_hash);
            }
        }

        let dim = EMBEDDING_DIM;
        let table = self.get_or_create_table(table_name, dim).await?;

        let turn_ids = UInt32Array::from(records.iter().map(|r| r.turn_id).collect::<Vec<_>>());
        let session_ids = StringArray::from(
            records
                .iter()
                .map(|r| r.session_id.clone())
                .collect::<Vec<_>>(),
        );
        let content_hashes = StringArray::from(
            records
                .iter()
                .map(|r| r.content_hash.clone())
                .collect::<Vec<_>>(),
        );
        let timestamps = UInt64Array::from(records.iter().map(|r| r.timestamp).collect::<Vec<_>>());
        let metadata = StringArray::from(
            records
                .iter()
                .map(|r| r.metadata_json.clone())
                .collect::<Vec<_>>(),
        );

        // Use a more direct way to create the vector array if possible, or just values + offsets
        // For production simplicity in this turn, I'll use values and try_new
        let mut flattened_vectors = Vec::with_capacity(records.len() * dim);
        for r in &records {
            flattened_vectors.extend_from_slice(&r.embedding);
        }
        let vector_values = Arc::new(Float32Array::from(flattened_vectors));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array =
            arrow_array::array::FixedSizeListArray::try_new(field, dim as i32, vector_values, None)
                .map_err(|e| anyhow!("Failed to create vector array: {:?}", e))?;

        let batch = RecordBatch::try_from_iter(vec![
            (
                "vector",
                Arc::new(vector_array) as Arc<dyn arrow_array::Array>,
            ),
            ("turn_id", Arc::new(turn_ids) as Arc<dyn arrow_array::Array>),
            (
                "session_id",
                Arc::new(session_ids) as Arc<dyn arrow_array::Array>,
            ),
            (
                "content_hash",
                Arc::new(content_hashes) as Arc<dyn arrow_array::Array>,
            ),
            (
                "timestamp",
                Arc::new(timestamps) as Arc<dyn arrow_array::Array>,
            ),
            (
                "metadata",
                Arc::new(metadata) as Arc<dyn arrow_array::Array>,
            ),
        ])?;

        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(reader)
            .execute()
            .await
            .map_err(|e| anyhow!("Failed to add records: {:?}", e))?;

        Ok(())
    }

    pub async fn query_nearest(
        &self,
        table_name: &str,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<(MemoryRecord, f32)>> {
        let table = self.get_or_create_table(table_name, EMBEDDING_DIM).await?;

        let query = table.vector_search(query_vector.clone())?.limit(k * 2);

        let mut stream = query
            .execute()
            .await
            .map_err(|e| anyhow!("Search failed: {:?}", e))?;

        let mut scored_records = Vec::new();
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res.map_err(|e| anyhow!("Batch retrieval failed: {:?}", e))?;

            let vectors = batch
                .column_by_name("vector")
                .ok_or_else(|| anyhow!("Vector column missing"))?
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeListArray>()
                .ok_or_else(|| anyhow!("Vector column cast failed"))?;
            let turn_ids = batch
                .column_by_name("turn_id")
                .ok_or_else(|| anyhow!("Turn ID column missing"))?
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .ok_or_else(|| anyhow!("Turn ID column cast failed"))?;
            let session_ids = batch
                .column_by_name("session_id")
                .ok_or_else(|| anyhow!("Session ID column missing"))?
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("Session ID column cast failed"))?;
            let content_hashes = batch
                .column_by_name("content_hash")
                .ok_or_else(|| anyhow!("Content hash column missing"))?
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("Content hash column cast failed"))?;
            let timestamps = batch
                .column_by_name("timestamp")
                .ok_or_else(|| anyhow!("Timestamp column missing"))?
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or_else(|| anyhow!("Timestamp column cast failed"))?;
            let metadata = batch
                .column_by_name("metadata")
                .ok_or_else(|| anyhow!("Metadata column missing"))?
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("Metadata column cast failed"))?;

            for i in 0..batch.num_rows() {
                let vec_val = vectors.value(i);
                let vec_f32 = vec_val
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("Vector value cast failed"))?;
                let mut embedding = vec![0.0; vec_f32.len()];
                for (j, val) in embedding.iter_mut().enumerate() {
                    *val = vec_f32.value(j);
                }

                let similarity = cosine_similarity(&query_vector, &embedding);
                let record = MemoryRecord {
                    turn_id: turn_ids.value(i),
                    session_id: session_ids.value(i).to_string(),
                    embedding,
                    content_hash: content_hashes.value(i).to_string(),
                    timestamp: timestamps.value(i),
                    metadata_json: metadata.value(i).to_string(),
                    outcome: None,
                };
                scored_records.push((record, similarity));
            }
        }

        scored_records.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<(MemoryRecord, f32)> = scored_records.into_iter().take(k).collect();
        Ok(results)
    }

    pub async fn query_weighted(
        &self,
        table_name: &str,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f64)>> {
        let query_embedding = embed_text(query_text);
        let scored_records = self
            .query_nearest(table_name, query_embedding.clone(), top_k * 2)
            .await?;

        let mut weighted_results = Vec::new();
        for (record, similarity) in scored_records {
            let weight = record
                .outcome
                .as_ref()
                .map(|o| {
                    if o.tests_passed {
                        outcome_weight(&TurnOutcome::TestsPassed)
                    } else if o.compiled {
                        outcome_weight(&TurnOutcome::Compiled)
                    } else if o.was_rolled_back {
                        outcome_weight(&TurnOutcome::RolledBack)
                    } else {
                        outcome_weight(&TurnOutcome::Unknown)
                    }
                })
                .unwrap_or_else(|| outcome_weight(&TurnOutcome::Unknown));

            let weighted_score = (similarity as f64) * weight;
            weighted_results.push((record, weighted_score));
        }

        weighted_results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(weighted_results.into_iter().take(top_k).collect())
    }
}

pub struct LessonExtractor;

impl LessonExtractor {
    #[must_use]
    pub fn extract(turns: &[Turn]) -> Vec<Lesson> {
        let mut lessons = vec![];
        for turn in turns {
            if turn.outcome == TurnOutcome::TestsPassed {
                lessons.push(Lesson {
                    context_type: "coding".to_string(),
                    approach: format!("Model {} approach in turn {}", turn.model_id, turn.index),
                    outcome: "Success (Tests Passed)".to_string(),
                    confidence: 0.9,
                    applicability_tags: vec!["rust".to_string(), "passing_tests".to_string()],
                });
            }
        }
        lessons
    }
}

pub struct FailurePredictor;

impl FailurePredictor {
    #[must_use]
    pub fn proactive_warning(
        current_context: &str,
        failures: &[FailureSignature],
    ) -> Option<String> {
        for failure in failures {
            if current_context.contains(&failure.error_type)
                || current_context.contains(&failure.error_message)
            {
                return Some(format!(
                    "High probability of {} regression: {}",
                    failure.error_type, failure.error_message
                ));
            }
        }
        None
    }
}

pub struct ContextDistiller;

impl ContextDistiller {
    #[must_use]
    pub fn distill(sigma: &ConversationState, max_tokens: usize) -> String {
        let now = ConversationState::now();
        let mut scored_turns: Vec<(&Turn, f64)> = sigma
            .turns
            .iter()
            .map(|t| {
                let age_hours = (now - t.timestamp) as f64 / 3600.0;
                let outcome_weight = match t.outcome {
                    TurnOutcome::TestsPassed => 1.5,
                    TurnOutcome::Compiled => 1.2,
                    TurnOutcome::RolledBack | TurnOutcome::Rejected => 0.5,
                    _ => 1.0,
                };
                let score = outcome_weight * (-0.01 * age_hours).exp();
                (t, score)
            })
            .collect();

        scored_turns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut distilled = format!("Session: {} (Distilled Context)\n", sigma.session_id);
        let mut current_len = distilled.len();

        for (turn, _score) in scored_turns.iter().take(10) {
            let entry = format!("i_{}: {}: {}\n", turn.index, turn.model_id, turn.content);
            if current_len + entry.len() > max_tokens {
                break;
            }
            distilled.push_str(&entry);
            current_len += entry.len();
        }

        distilled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_text_produces_384_dims() {
        let embedding = embed_text("hello world");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_embed_text_normalized() {
        let embedding = embed_text("test string");
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "Vector should be normalized to unit length");
    }

    #[test]
    fn test_embed_text_deterministic() {
        let text = "deterministic test";
        let embedding1 = embed_text(text);
        let embedding2 = embed_text(text);
        assert_eq!(embedding1, embedding2);
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&vec, &vec);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&vec_a, &vec_b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_similar_texts() {
        let text1 = "the quick brown fox";
        let text2 = "the quick brown fox";
        let emb1 = embed_text(text1);
        let emb2 = embed_text(text2);
        let sim = cosine_similarity(&emb1, &emb2);
        assert!(
            sim > 0.99,
            "Identical texts should have similarity > 0.99, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_different_texts() {
        let text1 = "the quick brown fox jumps";
        let text2 = "completely different words here";
        let emb1 = embed_text(text1);
        let emb2 = embed_text(text2);
        let sim = cosine_similarity(&emb1, &emb2);
        assert!(
            sim < 0.5,
            "Different texts should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_normalize_vector() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize_vector(&vec);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
    }
}
