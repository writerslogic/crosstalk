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
use std::sync::Arc;

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

    pub async fn insert(&self, table_name: &str, records: Vec<MemoryRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let dim = records[0].embedding.len();
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
        vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let table = self.get_or_create_table(table_name, vector.len()).await?;

        let query = table.vector_search(vector)?.limit(k);

        let mut stream = query
            .execute()
            .await
            .map_err(|e| anyhow!("Search failed: {:?}", e))?;

        let mut records = Vec::new();
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

                records.push(MemoryRecord {
                    turn_id: turn_ids.value(i),
                    session_id: session_ids.value(i).to_string(),
                    embedding,
                    content_hash: content_hashes.value(i).to_string(),
                    timestamp: timestamps.value(i),
                    metadata_json: metadata.value(i).to_string(),
                    outcome: None,
                });
            }
        }
        Ok(records)
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
