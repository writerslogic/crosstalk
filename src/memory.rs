use lancedb::connect;
use lancedb::connection::Connection;
use lancedb::table::Table;
use anyhow::{Result, anyhow};
use crate::types::{MemoryRecord, Turn, TurnOutcome, Lesson, FailureSignature, ConversationState};

pub struct MemoryStore {
    uri: String,
    conn: Option<Connection>,
}

impl MemoryStore {
    pub fn new(uri: &str) -> Self {
        Self {
            uri: uri.to_string(),
            conn: None,
        }
    }

    pub async fn init(&mut self) -> Result<()> {
        let conn = connect(&self.uri).execute().await
            .map_err(|e| anyhow!("Failed to connect to LanceDB: {:?}", e))?;
        self.conn = Some(conn);
        Ok(())
    }

    pub async fn get_or_create_table(&self, name: &str, _dim: usize) -> Result<Table> {
        let conn = self.conn.as_ref().ok_or_else(|| anyhow!("Not connected"))?;
        conn.open_table(name).execute().await
            .map_err(|e| anyhow!("Failed to open table {}: {:?}", name, e))
    }

    pub async fn insert(&self, table_name: &str, _records: Vec<MemoryRecord>, _vectors: Vec<Vec<f32>>) -> Result<()> {
        let _table = self.get_or_create_table(table_name, 0).await?;
        Ok(())
    }

    pub async fn query_nearest(&self, table_name: &str, _vector: Vec<f32>, _k: usize) -> Result<Vec<MemoryRecord>> {
        let _table = self.get_or_create_table(table_name, 0).await?;
        Ok(vec![])
    }
}

pub struct LessonExtractor;

impl LessonExtractor {
    pub fn extract(turns: &[Turn]) -> Vec<Lesson> {
        let mut lessons = vec![];
        // Heuristic: identify successful turn patterns
        for turn in turns {
            if turn.outcome == TurnOutcome::TestsPassed {
                lessons.push(Lesson {
                    id: format!("lesson-{}", turn.index),
                    category: "success".to_string(),
                    description: format!("Model {} produced passing code for this context.", turn.model_id),
                    evidence_turn_ids: vec![turn.index],
                    confidence: 0.8,
                });
            }
        }
        lessons
    }
}

pub struct FailurePredictor;

impl FailurePredictor {
    pub fn proactive_warning(_current_context: &str, _failures: &[FailureSignature]) -> Option<String> {
        // In a real impl, we'd embed context and find similar failures.
        None
    }
}

pub struct ContextDistiller;

impl ContextDistiller {
    pub fn distill(sigma: &ConversationState, max_tokens: usize) -> String {
        // Simplified distillation: keep last N turns and all artifact summaries
        let mut distilled = format!("Session: {}\n", sigma.session_id);
        distilled.push_str("Recent History:\n");
        for turn in sigma.turns.iter().rev().take(5).rev() {
            distilled.push_str(&format!("i_{}: {}: {}\n", turn.index, turn.model_id, turn.content));
        }
        
        if distilled.len() > max_tokens {
            distilled.truncate(max_tokens);
        }
        distilled
    }
}
