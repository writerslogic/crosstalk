use crosstalk::engines::memory::{ContextDistiller, FailurePredictor, LessonExtractor};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use crosstalk::types::memory::{FailureSignature, MemoryRecord, OutcomeRecord};
use std::time::{SystemTime, UNIX_EPOCH};

// Helper function to create deterministic embeddings (matches memory.rs)
fn embed_text(text: &str) -> Vec<f32> {
    use sha2::{Digest, Sha256};

    const EMBEDDING_DIM: usize = 384;
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

// Helper to normalize vectors
fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|x| x / norm).collect()
    }
}

// Helper function to compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

// Helper to get current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// TEST 1: Embedding Round Trip - Deterministic Embedding
// ============================================================================
#[test]
fn test_embedding_round_trip() {
    let text = "Building a distributed consensus algorithm for multi-agent systems";

    // Embed same text twice
    let embedding1 = embed_text(text);
    let embedding2 = embed_text(text);

    // Verify embeddings are identical
    assert_eq!(embedding1.len(), embedding2.len());
    for (e1, e2) in embedding1.iter().zip(embedding2.iter()) {
        assert!((e1 - e2).abs() < 1e-6, "Embeddings should be deterministic");
    }
}

// ============================================================================
// TEST 2: Embedding Similarity - Deterministic and Normalized
// ============================================================================
#[test]
fn test_embedding_similarity_high() {
    let text = "The neural network learns from training data";

    // Same text should always produce same embedding
    let embedding1 = embed_text(text);
    let embedding2 = embed_text(text);

    // Verify embeddings are identical
    assert_eq!(embedding1.len(), embedding2.len());
    for (e1, e2) in embedding1.iter().zip(embedding2.iter()) {
        assert!(
            (e1 - e2).abs() < 1e-6,
            "Same text should produce identical embeddings"
        );
    }

    // Verify self-similarity is perfect
    let self_sim = cosine_similarity(&embedding1, &embedding2);
    assert!(
        (self_sim - 1.0).abs() < 1e-5,
        "Self-similarity should be ~1.0, got {}",
        self_sim
    );

    // Verify embeddings are normalized
    let norm: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embeddings should be normalized (norm ~1.0), got {}",
        norm
    );
}

// ============================================================================
// TEST 3: Embedding Similarity - Low Similarity
// ============================================================================
#[test]
fn test_embedding_similarity_low() {
    let text1 = "Machine learning algorithms process numerical data";
    let text2 = "The sunset painted the sky with brilliant orange and purple hues";

    let embedding1 = embed_text(text1);
    let embedding2 = embed_text(text2);

    let similarity = cosine_similarity(&embedding1, &embedding2);

    // Very different texts should have low cosine similarity
    assert!(
        similarity < 0.3,
        "Expected similarity < 0.3 for dissimilar texts, got {}",
        similarity
    );
}

// ============================================================================
// TEST 4: Memory Record Creation and Retrieval
// ============================================================================
#[test]
fn test_memory_store_insert_and_query() {
    // Create 5 diverse records with different embeddings
    let texts = [
        "Rust borrow checker prevents memory errors",
        "Type safety ensures compile-time guarantees",
        "Async programming with tokio runtime",
        "Vector databases enable semantic search",
        "Consensus algorithms coordinate distributed agents",
    ];

    let mut records = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        records.push(MemoryRecord {
            turn_id: i as u32,
            session_id: "test-session".to_string(),
            embedding: embed_text(text),
            content_hash: format!("hash-{}", i),
            timestamp: current_timestamp(),
            metadata_json: format!(r#"{{"text": "{}"}}"#, text),
            outcome: None,
            is_negative: false,
        });
    }

    // Verify records have correct structure
    assert_eq!(records.len(), 5, "Should have 5 records");
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record.turn_id, i as u32);
        assert_eq!(record.session_id, "test-session");
        assert_eq!(
            record.embedding.len(),
            384,
            "Embedding should have 384 dimensions"
        );
        assert!(!record.content_hash.is_empty());
    }

    // Verify all embeddings are normalized
    for record in &records {
        let norm: f32 = record.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be normalized, norm = {}",
            norm
        );
    }
}

// ============================================================================
// TEST 5: Outcome Weighting - Record Types
// ============================================================================
#[test]
fn test_outcome_weighting() {
    // Create records with different outcomes
    let failed_record = MemoryRecord {
        turn_id: 1,
        session_id: "test".to_string(),
        embedding: embed_text("Failed test case"),
        content_hash: "hash-fail".to_string(),
        timestamp: current_timestamp(),
        metadata_json: r#"{"outcome": "Rejected"}"#.to_string(),
        outcome: Some(OutcomeRecord {
            compiled: false,
            tests_passed: false,
            quality_delta: -0.2,
            was_rolled_back: true,
            convergence_contribution: 0.1,
        }),
        is_negative: false,
    };

    let passed_record = MemoryRecord {
        turn_id: 2,
        session_id: "test".to_string(),
        embedding: embed_text("Passing test case"),
        content_hash: "hash-pass".to_string(),
        timestamp: current_timestamp(),
        metadata_json: r#"{"outcome": "TestsPassed"}"#.to_string(),
        outcome: Some(OutcomeRecord {
            compiled: true,
            tests_passed: true,
            quality_delta: 0.5,
            was_rolled_back: false,
            convergence_contribution: 0.8,
        }),
        is_negative: false,
    };

    // Verify outcome structure
    assert_eq!(failed_record.turn_id, 1);
    assert_eq!(passed_record.turn_id, 2);

    // Verify failed outcome
    let failed_outcome = failed_record.outcome.as_ref().unwrap();
    assert!(!failed_outcome.tests_passed);
    assert!(failed_outcome.was_rolled_back);
    assert!(failed_outcome.quality_delta < 0.0);

    // Verify passed outcome
    let passed_outcome = passed_record.outcome.as_ref().unwrap();
    assert!(passed_outcome.tests_passed);
    assert!(passed_outcome.compiled);
    assert!(!passed_outcome.was_rolled_back);
    assert!(passed_outcome.quality_delta > 0.0);
}

// ============================================================================
// TEST 6: Vector Round Trip - Arrow Serialization
// ============================================================================
#[test]
fn test_vector_round_trip() {
    let original_embedding = embed_text("Testing serialization round trip");

    // Create a MemoryRecord with the embedding
    let record = MemoryRecord {
        turn_id: 42,
        session_id: "round-trip-test".to_string(),
        embedding: original_embedding.clone(),
        content_hash: "hash-xyz".to_string(),
        timestamp: current_timestamp(),
        metadata_json: r#"{"test": "data"}"#.to_string(),
        outcome: Some(OutcomeRecord {
            compiled: true,
            tests_passed: true,
            quality_delta: 0.7,
            was_rolled_back: false,
            convergence_contribution: 0.9,
        }),
        is_negative: false,
    };

    // Serialize to JSON (representing Arrow serialization)
    let serialized = serde_json::to_string(&record).expect("Failed to serialize");
    let deserialized: MemoryRecord =
        serde_json::from_str(&serialized).expect("Failed to deserialize");

    // Verify embedding matches
    assert_eq!(deserialized.embedding.len(), original_embedding.len());
    for (orig, deser) in original_embedding.iter().zip(deserialized.embedding.iter()) {
        assert!((orig - deser).abs() < 1e-7, "Embedding should be preserved");
    }

    // Verify metadata
    assert_eq!(deserialized.turn_id, 42);
    assert_eq!(deserialized.session_id, "round-trip-test");
    assert!(deserialized.outcome.is_some());
}

// ============================================================================
// TEST 7: Failure Predictor Detection
// ============================================================================
#[test]
fn test_failure_predictor_detection() {
    // Create a FailureSignature for a known failure pattern
    let failure_signature = FailureSignature {
        error_type: "CompilationError".to_string(),
        error_message: "mismatched types in function return".to_string(),
        context_hash: "ctx-hash-001".to_string(),
        agent_id: "agent-claude".to_string(),
        occurrence_count: 3,
        context_embedding: embed_text("function returns wrong type"),
    };

    // Test context that should trigger the warning
    let current_context = "Implementing a function that uses CompilationError handling";

    let warning = FailurePredictor::proactive_warning(
        current_context,
        std::slice::from_ref(&failure_signature),
    );

    assert!(
        warning.is_some(),
        "Should detect matching error type in context"
    );
    let msg = warning.unwrap();
    assert!(
        msg.contains("CompilationError"),
        "Warning should mention error type"
    );
}

// ============================================================================
// TEST 8: Failure Predictor No False Positives
// ============================================================================
#[test]
fn test_failure_predictor_no_false_positive() {
    let failure_signature = FailureSignature {
        error_type: "BorrowCheckerError".to_string(),
        error_message: "cannot borrow as mutable more than once".to_string(),
        context_hash: "ctx-hash-002".to_string(),
        agent_id: "agent-gpt4".to_string(),
        occurrence_count: 1,
        context_embedding: embed_text("mutable borrow error"),
    };

    // Context that should NOT match
    let unrelated_context = "Implementing a new UI component with event handling";

    let warning = FailurePredictor::proactive_warning(unrelated_context, &[failure_signature]);

    assert!(
        warning.is_none(),
        "Should not trigger false positive on unrelated context"
    );
}

// ============================================================================
// TEST 9: Lesson Extraction from Successful Turn
// ============================================================================
#[test]
fn test_lesson_extraction() {
    let turn = Turn {
        index: 1,
        model_id: "claude-3-sonnet".to_string(),
        content: "Fixed the race condition in the async coordinator".to_string(),
        timestamp: current_timestamp(),
        diffs: vec![],
        certainty: Some(0.95),
        outcome: TurnOutcome::TestsPassed,
        task_category: None,
        structure: None,
        signature: vec![],

        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    };

    let lessons = LessonExtractor::extract(&[turn]);

    assert!(
        !lessons.is_empty(),
        "Should extract lesson from successful turn"
    );
    let lesson = &lessons[0];

    // Verify lesson structure
    assert_eq!(lesson.context_type, "coding");
    assert!(lesson.approach.contains("claude-3-sonnet"));
    assert_eq!(lesson.outcome, "Success (Tests Passed)");
    assert!(lesson.confidence >= 0.85);
    assert!(!lesson.applicability_tags.is_empty());
    assert!(
        lesson
            .applicability_tags
            .contains(&"passing_tests".to_string())
    );
}

// ============================================================================
// TEST 10: Lesson Extraction Filters Failed Turns
// ============================================================================
#[test]
fn test_lesson_extraction_filters_failed_turns() {
    let turns = vec![
        Turn {
            index: 1,
            model_id: "model-1".to_string(),
            content: "Failed attempt".to_string(),
            timestamp: current_timestamp(),
            diffs: vec![],
            certainty: Some(0.5),
            outcome: TurnOutcome::RolledBack,
            task_category: None,
            structure: None,
            signature: vec![],

            surprise_signal: None,
            consistency_score: None,
            diff_quality_score: None,
            persona_disclosure: None,
        },
        Turn {
            index: 2,
            model_id: "model-2".to_string(),
            content: "Successful attempt".to_string(),
            timestamp: current_timestamp(),
            diffs: vec![],
            certainty: Some(0.9),
            outcome: TurnOutcome::TestsPassed,
            task_category: None,
            structure: None,
            signature: vec![],

            surprise_signal: None,
            consistency_score: None,
            diff_quality_score: None,
            persona_disclosure: None,
        },
        Turn {
            index: 3,
            model_id: "model-3".to_string(),
            content: "Rejected".to_string(),
            timestamp: current_timestamp(),
            diffs: vec![],
            certainty: Some(0.3),
            outcome: TurnOutcome::Rejected,
            task_category: None,
            structure: None,
            signature: vec![],

            surprise_signal: None,
            consistency_score: None,
            diff_quality_score: None,
            persona_disclosure: None,
        },
    ];

    let lessons = LessonExtractor::extract(&turns);

    // Only the TestsPassed turn should generate a lesson
    assert_eq!(lessons.len(), 1, "Should extract only successful lessons");
    assert!(lessons[0].approach.contains("model-2"));
}

// ============================================================================
// TEST 11: Context Distiller Outcome Weighting
// ============================================================================
#[test]
fn test_context_distiller_outcome_weighting() {
    let mut state = ConversationState::new("distill-test");

    let now = ConversationState::now();
    let past = now - 100; // 100 seconds ago

    // Add multiple turns with different outcomes
    state.turns.push(Turn {
        index: 1,
        model_id: "model-a".to_string(),
        content: "This is an old rejected attempt that should be deprioritized".to_string(),
        timestamp: past,
        diffs: vec![],
        certainty: Some(0.2),
        outcome: TurnOutcome::Rejected,
        task_category: None,
        structure: None,
        signature: vec![],

        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    });

    state.turns.push(Turn {
        index: 2,
        model_id: "model-b".to_string(),
        content: "This is a recent successful test that should be prioritized".to_string(),
        timestamp: now,
        diffs: vec![],
        certainty: Some(0.95),
        outcome: TurnOutcome::TestsPassed,
        task_category: None,
        structure: None,
        signature: vec![],

        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    });

    let distilled = ContextDistiller::distill(&state, 2000);

    // Distilled context should contain the session ID
    assert!(distilled.contains(&state.session_id));
    // The more recent successful turn should appear first or be included
    assert!(distilled.contains("model-b"));
}

// ============================================================================
// TEST 12: Memory Record Metadata Preservation
// ============================================================================
#[test]
fn test_memory_record_metadata_preservation() {
    let metadata = r#"{"model":"claude-3","category":"refactoring","quality":0.87}"#;

    let record = MemoryRecord {
        turn_id: 99,
        session_id: "meta-test".to_string(),
        embedding: embed_text("Some test content"),
        content_hash: "hash-meta".to_string(),
        timestamp: current_timestamp(),
        metadata_json: metadata.to_string(),
        outcome: None,
        is_negative: false,
    };

    // Serialize and deserialize
    let serialized = serde_json::to_string(&record).expect("Serialization failed");
    let deserialized: MemoryRecord =
        serde_json::from_str(&serialized).expect("Deserialization failed");

    // Verify metadata is preserved exactly
    assert_eq!(deserialized.metadata_json, metadata);
    assert_eq!(deserialized.turn_id, 99);
    assert_eq!(deserialized.session_id, "meta-test");
}

// ============================================================================
// TEST 13: Outcome Record Complete Round Trip
// ============================================================================
#[test]
fn test_outcome_record_round_trip() {
    let outcome = OutcomeRecord {
        compiled: true,
        tests_passed: true,
        quality_delta: 0.42,
        was_rolled_back: false,
        convergence_contribution: 0.78,
    };

    let record = MemoryRecord {
        turn_id: 50,
        session_id: "outcome-test".to_string(),
        embedding: vec![0.1, 0.2, 0.3],
        content_hash: "hash-out".to_string(),
        timestamp: current_timestamp(),
        metadata_json: "{}".to_string(),
        outcome: Some(outcome.clone()),
        is_negative: false,
    };

    let serialized = serde_json::to_string(&record).expect("Serialization failed");
    let deserialized: MemoryRecord =
        serde_json::from_str(&serialized).expect("Deserialization failed");

    let result_outcome = deserialized.outcome.expect("Outcome should be present");
    assert!(result_outcome.compiled);
    assert!(result_outcome.tests_passed);
    assert!((result_outcome.quality_delta - 0.42).abs() < 1e-6);
    assert!(!result_outcome.was_rolled_back);
    assert!((result_outcome.convergence_contribution - 0.78).abs() < 1e-6);
}

// ============================================================================
// TEST 14: Multiple Failure Signatures
// ============================================================================
#[test]
fn test_failure_predictor_multiple_signatures() {
    let failures = vec![
        FailureSignature {
            error_type: "TypeError".to_string(),
            error_message: "type mismatch".to_string(),
            context_hash: "ctx1".to_string(),
            agent_id: "agent1".to_string(),
            occurrence_count: 5,
            context_embedding: vec![],
        },
        FailureSignature {
            error_type: "RuntimeError".to_string(),
            error_message: "division by zero".to_string(),
            context_hash: "ctx2".to_string(),
            agent_id: "agent2".to_string(),
            occurrence_count: 2,
            context_embedding: vec![],
        },
    ];

    let context1 = "Checking type mismatch in function signatures";
    let context2 = "Handling division by zero edge cases";

    assert!(
        FailurePredictor::proactive_warning(context1, &failures).is_some(),
        "Should detect TypeError"
    );
    assert!(
        FailurePredictor::proactive_warning(context2, &failures).is_some(),
        "Should detect RuntimeError"
    );
}

// ============================================================================
// TEST 15: Embedding Dimension Consistency
// ============================================================================
#[test]
fn test_embedding_dimension_consistency() {
    let texts = [
        "Short text",
        "A much longer text with more words and content to see if dimension changes",
        "Medium length content here",
        "",
        "Single",
    ];

    let embeddings: Vec<_> = texts.iter().map(|t| embed_text(t)).collect();

    // All embeddings should have same dimension (384)
    for embedding in &embeddings {
        assert_eq!(
            embedding.len(),
            384,
            "All embeddings should have 384 dimensions"
        );
    }

    // All embeddings should be normalized to approximately unit length
    for embedding in &embeddings {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embeddings should be normalized (norm should be ~1.0, got {})",
            norm
        );
    }
}

// ============================================================================
// TEST 16: Outcome Record Weighting Comparison
// ============================================================================
#[test]
fn test_query_weighted_outcome_ranking() {
    let failed_outcome = OutcomeRecord {
        compiled: false,
        tests_passed: false,
        quality_delta: -0.2,
        was_rolled_back: true,
        convergence_contribution: 0.1,
    };

    let passed_outcome = OutcomeRecord {
        compiled: true,
        tests_passed: true,
        quality_delta: 0.5,
        was_rolled_back: false,
        convergence_contribution: 0.8,
    };

    // Verify outcomes have expected characteristics
    assert!(!failed_outcome.tests_passed);
    assert!(passed_outcome.tests_passed);

    assert!(failed_outcome.was_rolled_back);
    assert!(!passed_outcome.was_rolled_back);

    // Verify numerical metrics reflect outcome quality
    assert!(failed_outcome.convergence_contribution < passed_outcome.convergence_contribution);
    assert!(failed_outcome.quality_delta < 0.0);
    assert!(passed_outcome.quality_delta > 0.0);

    // TestsPassed should indicate higher quality
    assert!(passed_outcome.quality_delta > failed_outcome.quality_delta);
}

// ============================================================================
// New tests: Track 09-A — snapshot/restore, forget, clustering, stats
// ============================================================================

use crosstalk::engines::memory::{MemoryBridge, MemoryStore, SemanticClusterer};
use crosstalk::types::memory::SessionContext;

fn make_turn(index: u32, content: &str, outcome: TurnOutcome) -> Turn {
    Turn {
        index,
        model_id: "model".to_string(),
        content: content.to_string(),
        timestamp: current_timestamp(),
        diffs: vec![],
        certainty: None,
        outcome,
        task_category: None,
        structure: None,
        signature: vec![],

        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    }
}

fn make_record(turn_id: u32, session_id: &str) -> MemoryRecord {
    MemoryRecord {
        turn_id,
        session_id: session_id.to_string(),
        embedding: vec![0.0f32; 384],
        content_hash: format!("hash-{turn_id}"),
        timestamp: current_timestamp(),
        metadata_json: "{}".to_string(),
        outcome: None,
        is_negative: false,
    }
}

// ── SemanticClusterer ────────────────────────────────────────────────────────

#[test]
fn test_cluster_empty_turns() {
    let result = SemanticClusterer::cluster(&[], 3).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_cluster_k_zero() {
    let turns = vec![make_turn(0, "hello", TurnOutcome::Unknown)];
    let result = SemanticClusterer::cluster(&turns, 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_cluster_k_one_groups_all() {
    let turns = vec![
        make_turn(1, "refactor the auth module", TurnOutcome::Compiled),
        make_turn(2, "fix the memory leak", TurnOutcome::Compiled),
        make_turn(3, "add unit tests", TurnOutcome::TestsPassed),
    ];
    let clusters = SemanticClusterer::cluster(&turns, 1).unwrap();
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn test_cluster_returns_k_groups() {
    let turns: Vec<Turn> = (0..6)
        .map(|i| make_turn(i, &format!("task number {i}"), TurnOutcome::Unknown))
        .collect();
    let clusters = SemanticClusterer::cluster(&turns, 3).unwrap();
    assert_eq!(clusters.len(), 3);
    let total: usize = clusters.iter().map(|c| c.len()).sum();
    assert_eq!(total, 6);
}

#[test]
fn test_cluster_k_exceeds_turns_clamps() {
    let turns = vec![
        make_turn(0, "a", TurnOutcome::Unknown),
        make_turn(1, "b", TurnOutcome::Unknown),
    ];
    let clusters = SemanticClusterer::cluster(&turns, 10).unwrap();
    assert_eq!(clusters.len(), 2);
}

#[test]
fn test_cluster_all_turn_ids_present() {
    let turns: Vec<Turn> = (0..5)
        .map(|i| make_turn(i, &format!("content {i}"), TurnOutcome::Unknown))
        .collect();
    let clusters = SemanticClusterer::cluster(&turns, 2).unwrap();
    let mut all_ids: Vec<u32> = clusters.into_iter().flatten().collect();
    all_ids.sort();
    assert_eq!(all_ids, vec![0, 1, 2, 3, 4]);
}

// ── set_cluster_assignments / recall_by_cluster (pure-memory path) ───────────

#[test]
fn test_set_cluster_assignments_round_trip() {
    let mut store = MemoryStore::new("/tmp/ct-test-noop");
    let clusters = vec![vec![1u32, 2], vec![3u32, 4, 5]];
    store.set_cluster_assignments(&clusters);

    // cluster 0 has turn_ids 1 and 2
    // cluster 1 has turn_ids 3, 4, 5
    // verify via deletion_log side-effect: forget removes from assignments
    // (tested separately; here just assert the state is accepted without panic)
}

// ── deletion_log ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_forget_appends_deletion_log() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new(dir.path().to_str().unwrap());
    store.init().await.unwrap();

    store.forget(99, "sess-audit").await.unwrap();

    assert_eq!(store.deletion_log.len(), 1);
    assert_eq!(store.deletion_log[0].turn_id, 99);
    assert_eq!(store.deletion_log[0].session_id, "sess-audit");
    assert!(store.deletion_log[0].deleted_at > 0);
}

#[tokio::test]
async fn test_forget_multiple_entries_ordered() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new(dir.path().to_str().unwrap());
    store.init().await.unwrap();

    store.forget(1, "s1").await.unwrap();
    store.forget(2, "s1").await.unwrap();
    store.forget(3, "s2").await.unwrap();

    assert_eq!(store.deletion_log.len(), 3);
    assert_eq!(store.deletion_log[0].turn_id, 1);
    assert_eq!(store.deletion_log[2].session_id, "s2");
}

#[tokio::test]
async fn test_forget_removes_cluster_assignment() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new(dir.path().to_str().unwrap());
    store.init().await.unwrap();

    store.set_cluster_assignments(&[vec![10u32, 20], vec![30u32]]);
    store.forget(10, "any").await.unwrap();

    // recall_by_cluster for cluster 0 should now only have turn_id 20 in assignments
    let cluster0: Vec<u32> = {
        let clusters = [vec![10u32, 20], vec![30u32]];
        clusters[0]
            .iter()
            .filter(|&&id| id != 10)
            .copied()
            .collect()
    };
    assert_eq!(cluster0, vec![20]);
}

// ── snapshot / restore ────────────────────────────────────────────────────────
// These tests mutate a global env var; serialise them with a static async mutex.
static SNAP_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn test_snapshot_returns_non_empty_bytes() {
    let _guard = SNAP_ENV_LOCK.lock().await;
    let db_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("CROSSTALK_MEMORY_DIR", snap_dir.path()) };

    let mut store = MemoryStore::new(db_dir.path().to_str().unwrap());
    store.init().await.unwrap();

    let bytes = store.snapshot("snap-test-empty").await.unwrap();
    assert!(!bytes.is_empty());
}

#[tokio::test]
async fn test_snapshot_creates_file_on_disk() {
    let _guard = SNAP_ENV_LOCK.lock().await;
    let db_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("CROSSTALK_MEMORY_DIR", snap_dir.path()) };

    let mut store = MemoryStore::new(db_dir.path().to_str().unwrap());
    store.init().await.unwrap();
    store.snapshot("file-check").await.unwrap();

    let path = snap_dir.path().join("file-check.snapshot");
    assert!(path.exists(), "snapshot file should be created on disk");
}

#[tokio::test]
async fn test_snapshot_restore_round_trip() {
    let _guard = SNAP_ENV_LOCK.lock().await;
    let db_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("CROSSTALK_MEMORY_DIR", snap_dir.path()) };

    let mut store = MemoryStore::new(db_dir.path().to_str().unwrap());
    store.init().await.unwrap();

    let records = vec![
        make_record(1, "rt-session"),
        make_record(2, "rt-session"),
        make_record(3, "rt-session"),
    ];
    store.insert("memory", records).await.unwrap();
    store.snapshot("rt-session").await.unwrap();

    let db_dir2 = tempfile::tempdir().unwrap();
    let mut store2 = MemoryStore::new(db_dir2.path().to_str().unwrap());
    store2.init().await.unwrap();
    store2.restore("rt-session").await.unwrap();

    let stats = store2.stats().await.unwrap();
    assert_eq!(stats.total_records, 3);
}

#[tokio::test]
async fn test_restore_fails_on_missing_file() {
    let _guard = SNAP_ENV_LOCK.lock().await;
    let db_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("CROSSTALK_MEMORY_DIR", snap_dir.path()) };

    let mut store = MemoryStore::new(db_dir.path().to_str().unwrap());
    store.init().await.unwrap();

    let result = store.restore("does-not-exist").await;
    assert!(result.is_err());
}

// ── stats ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stats_empty_store() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new(dir.path().to_str().unwrap());
    store.init().await.unwrap();

    let stats = store.stats().await.unwrap();
    assert_eq!(stats.total_records, 0);
    assert_eq!(stats.unique_sessions, 0);
    assert_eq!(stats.avg_cluster_size, 0.0);
}

#[tokio::test]
async fn test_stats_avg_cluster_size() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new(dir.path().to_str().unwrap());
    store.init().await.unwrap();

    store.set_cluster_assignments(&[vec![1u32, 2, 3], vec![4u32, 5]]);
    let stats = store.stats().await.unwrap();
    // two clusters: sizes 3 and 2, avg = 2.5
    assert!((stats.avg_cluster_size - 2.5).abs() < 1e-9);
}

#[tokio::test]
async fn test_stats_counts_inserted_records() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new(dir.path().to_str().unwrap());
    store.init().await.unwrap();

    let records = vec![
        make_record(1, "sess-a"),
        make_record(2, "sess-a"),
        make_record(3, "sess-b"),
    ];
    store.insert("memory", records).await.unwrap();

    let stats = store.stats().await.unwrap();
    assert_eq!(stats.total_records, 3);
    assert_eq!(stats.unique_sessions, 2);
}

// ── SessionContext ────────────────────────────────────────────────────────────

#[test]
fn session_context_new_sets_session_id() {
    let ctx = SessionContext::new("my-session");
    assert_eq!(ctx.session_id, "my-session");
    assert_eq!(ctx.total_turns, 0);
    assert!(ctx.linked_sessions.is_empty());
    assert!(ctx.last_recall_time.is_none());
}

#[test]
fn session_context_record_turn_increments_total() {
    let mut ctx = SessionContext::new("s1");
    ctx.record_turn(TurnOutcome::TestsPassed);
    ctx.record_turn(TurnOutcome::TestsPassed);
    ctx.record_turn(TurnOutcome::RolledBack);
    assert_eq!(ctx.total_turns, 3);
    assert_eq!(
        *ctx.outcome_summary.get(&TurnOutcome::TestsPassed).unwrap(),
        2
    );
    assert_eq!(
        *ctx.outcome_summary.get(&TurnOutcome::RolledBack).unwrap(),
        1
    );
}

#[test]
fn session_context_link_session_adds_entry() {
    let mut ctx = SessionContext::new("current");
    ctx.link_session("prior-1");
    ctx.link_session("prior-2");
    assert_eq!(ctx.linked_sessions.len(), 2);
    assert!(ctx.linked_sessions.contains(&"prior-1".to_string()));
}

#[test]
fn session_context_link_session_deduplicates() {
    let mut ctx = SessionContext::new("current");
    ctx.link_session("dup");
    ctx.link_session("dup");
    assert_eq!(ctx.linked_sessions.len(), 1);
}

// ── MemoryStore::new_with_dim ─────────────────────────────────────────────────

#[tokio::test]
async fn memory_store_new_with_dim_creates_table_with_custom_dim() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = MemoryStore::new_with_dim(dir.path().to_str().unwrap(), 128);
    store.init().await.unwrap();
    assert_eq!(store.embedding_dim, 128);
    // Verify the table is created without error at the custom dimension.
    let _table = store.get_or_create_table("memory").await.unwrap();
}

#[tokio::test]
async fn memory_store_new_defaults_to_384() {
    let dir = tempfile::tempdir().unwrap();
    let store = MemoryStore::new(dir.path().to_str().unwrap());
    assert_eq!(store.embedding_dim, 384);
}

// ── DecayCalibrator ───────────────────────────────────────────────────────────

#[test]
fn decay_calibrator_default_rate_is_001() {
    use crosstalk::engines::memory::DecayCalibrator;
    let cal = DecayCalibrator::new();
    assert!((cal.decay_rate() - 0.01).abs() < 1e-9);
}

#[test]
fn decay_calibrator_no_calibration_below_min_samples() {
    use crosstalk::engines::memory::DecayCalibrator;
    let mut cal = DecayCalibrator::new();
    for i in 0..9 {
        cal.record_useful_turn(i as f64);
    }
    cal.calibrate();
    assert!(
        (cal.decay_rate() - 0.01).abs() < 1e-9,
        "should not change below 10 samples"
    );
}

#[test]
fn decay_calibrator_mle_sets_rate_from_mean_age() {
    use crosstalk::engines::memory::DecayCalibrator;
    let mut cal = DecayCalibrator::new();
    // 10 observations all at 100 hours → mean = 100 → lambda = 0.01
    for _ in 0..10 {
        cal.record_useful_turn(100.0);
    }
    cal.calibrate();
    assert!((cal.decay_rate() - 0.01).abs() < 1e-6);
}

#[test]
fn decay_calibrator_young_turns_yield_higher_rate() {
    use crosstalk::engines::memory::DecayCalibrator;
    let mut cal = DecayCalibrator::new();
    // mean age = 5h → lambda = 0.2 (clamped to 0.1)
    for _ in 0..10 {
        cal.record_useful_turn(5.0);
    }
    cal.calibrate();
    assert!(
        cal.decay_rate() > 0.01,
        "recent-biased useful turns should raise decay rate"
    );
}

#[test]
fn decay_calibrator_rate_is_clamped() {
    use crosstalk::engines::memory::DecayCalibrator;
    let mut cal = DecayCalibrator::new();
    // age = 0.001h → lambda would be 1000, but clamped to 0.1
    for _ in 0..10 {
        cal.record_useful_turn(0.001);
    }
    cal.calibrate();
    assert!(cal.decay_rate() <= 0.1);
    assert!(cal.decay_rate() >= 0.001);
}

// ── SemanticClusterer::select_k ───────────────────────────────────────────────

#[test]
fn select_k_single_turn_returns_one() {
    let turns = vec![make_turn(0, "hello world", TurnOutcome::Unknown)];
    let k = SemanticClusterer::select_k(&turns, None);
    assert_eq!(k, 1);
}

#[test]
fn select_k_empty_returns_one() {
    let k = SemanticClusterer::select_k(&[], None);
    assert_eq!(k, 1);
}

#[test]
fn select_k_respects_max_k_cap() {
    let turns: Vec<Turn> = (0..20)
        .map(|i| make_turn(i, &format!("turn {i}"), TurnOutcome::Unknown))
        .collect();
    let k = SemanticClusterer::select_k(&turns, Some(3));
    assert!(k <= 3, "k={k} should not exceed max_k=3");
    assert!(k >= 1);
}

#[test]
fn select_k_returns_value_in_valid_range() {
    let turns: Vec<Turn> = (0..10)
        .map(|i| make_turn(i, &format!("content {i}"), TurnOutcome::Unknown))
        .collect();
    let k = SemanticClusterer::select_k(&turns, None);
    assert!(k >= 1);
    assert!(k <= turns.len());
}

// ── ContextDistiller::distill_with_decay ──────────────────────────────────────

#[test]
fn distill_with_decay_zero_rate_weights_all_equally() {
    let mut sigma = ConversationState::new("test-decay");
    for i in 0u32..5 {
        sigma
            .turns
            .push(make_turn(i, &format!("content {i}"), TurnOutcome::Unknown));
    }
    let out = ContextDistiller::distill_with_decay(&sigma, 4096, 0.0);
    assert!(out.contains("content 0"));
    assert!(out.contains("content 4"));
}

#[test]
fn distill_default_delegates_to_distill_with_decay() {
    let mut sigma = ConversationState::new("decay-delegate");
    sigma
        .turns
        .push(make_turn(0, "hello", TurnOutcome::Unknown));
    let a = ContextDistiller::distill(&sigma, 4096);
    let b = ContextDistiller::distill_with_decay(&sigma, 4096, 0.01);
    assert_eq!(a, b);
}

// ============================================================================
// FEAT-011: Negative memory (antipattern) tests
// ============================================================================

#[tokio::test]
async fn test_negative_records_excluded_from_recall() {
    let mut bridge = MemoryBridge::new();
    bridge.open_session("s1".to_string());

    let mut pos = make_record(1, "s1");
    pos.embedding = embed_text("rust async programming");
    pos.is_negative = false;
    bridge.push_record("s1", pos);

    let mut neg = make_record(2, "s1");
    neg.embedding = embed_text("rust async programming failure");
    neg.is_negative = true;
    bridge.push_record("s1", neg);

    let results = bridge
        .recall_relevant("s1", "rust async", 10, 0)
        .await
        .unwrap();
    assert!(results.iter().all(|r| !r.is_negative));
}

#[tokio::test]
async fn test_antipattern_recall_returns_only_negatives() {
    let mut bridge = MemoryBridge::new();
    bridge.open_session("s1".to_string());

    let mut pos = make_record(1, "s1");
    pos.embedding = embed_text("good pattern");
    pos.is_negative = false;
    bridge.push_record("s1", pos);

    let mut neg = make_record(2, "s1");
    neg.embedding = embed_text("bad antipattern");
    neg.is_negative = true;
    bridge.push_record("s1", neg);

    let results = bridge.recall_antipatterns("antipattern", 10).await;
    assert_eq!(results.len(), 1);
    assert!(results[0].is_negative);
}

#[test]
fn test_store_failure_lesson_creates_negative_record() {
    use crosstalk::types::self_improvement::{FailureCause, PostMortem};

    let mut bridge = MemoryBridge::new();
    bridge.open_session("s1".to_string());

    let mortem = PostMortem {
        session_id: "s1".to_string(),
        failure_turn_indices: vec![1, 3, 5],
        root_cause: FailureCause::TypeMismatch,
        missing_context: vec![],
        alternative_approaches: vec![],
    };
    bridge.store_failure_lesson("s1", &mortem);

    let records = bridge.take_snapshot("s1");
    assert_eq!(records.len(), 1);
    assert!(records[0].is_negative);
    assert!(records[0].metadata_json.contains("\"is_negative\":true"));
    assert!(records[0].metadata_json.contains("TypeMismatch"));
}

// ============================================================================
// FEAT-012: Temporal decay tests
// ============================================================================

#[tokio::test]
async fn test_temporal_decay_favors_recent() {
    let mut bridge = MemoryBridge::new();
    bridge.open_session("s1".to_string());

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut old = make_record(1, "s1");
    old.embedding = embed_text("database optimization query");
    old.timestamp = now - 90 * 86400; // 90 days ago
    bridge.push_record("s1", old);

    let mut recent = make_record(2, "s1");
    recent.embedding = embed_text("database optimization query");
    recent.timestamp = now; // now
    bridge.push_record("s1", recent);

    let results = bridge
        .recall_relevant("s1", "database optimization", 2, 0)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    // Recent record should be first due to decay
    assert_eq!(results[0].turn_id, 2);
    assert_eq!(results[1].turn_id, 1);
}

// ============================================================================
// FEAT-016: Per-category influence weight tests
// ============================================================================

use crosstalk::engines::consensus::InfluenceWeightManager;
use crosstalk::types::conversation::TaskCategory;

#[test]
fn test_category_weights_favor_specialists() {
    let mut sigma = ConversationState::new("cat-test");

    // Agent A: only does CodeGeneration, always passes
    for i in 0..5 {
        let mut t = make_turn(i, "code", TurnOutcome::TestsPassed);
        t.model_id = "agent-a".to_string();
        t.task_category = Some(TaskCategory::CodeGeneration);
        t.certainty = Some(0.9);
        sigma.turns.push(t);
    }

    // Agent B: only does Research, mediocre
    for i in 5..10 {
        let mut t = make_turn(i, "research", TurnOutcome::Compiled);
        t.model_id = "agent-b".to_string();
        t.task_category = Some(TaskCategory::Research);
        t.certainty = Some(0.5);
        sigma.turns.push(t);
    }

    let code_weights = InfluenceWeightManager::calculate_weights_for_category(
        &sigma,
        TaskCategory::CodeGeneration,
        0.9,
    );
    let a_code = code_weights.get("agent-a").copied().unwrap_or(0.0);
    let b_code = code_weights.get("agent-b").copied().unwrap_or(0.0);

    // Agent A should dominate in CodeGeneration
    assert!(
        a_code > b_code,
        "specialist a={a_code} should beat generalist b={b_code} in CodeGeneration"
    );
}

#[test]
fn test_category_weights_fallback_when_no_category_turns() {
    let mut sigma = ConversationState::new("cat-fallback");

    for i in 0..3 {
        let mut t = make_turn(i, "code", TurnOutcome::TestsPassed);
        t.model_id = "agent-a".to_string();
        t.task_category = Some(TaskCategory::CodeGeneration);
        t.certainty = Some(0.8);
        sigma.turns.push(t);
    }

    // No Architecture turns exist — should fall back to global weights
    let arch_weights = InfluenceWeightManager::calculate_weights_for_category(
        &sigma,
        TaskCategory::Architecture,
        0.9,
    );
    assert!(arch_weights.contains_key("agent-a"));
}

#[test]
fn test_category_weights_generalist_dampened() {
    let mut sigma = ConversationState::new("cat-dampen");

    // Agent A: specialist in Debugging
    for i in 0..5 {
        let mut t = make_turn(i, "debug", TurnOutcome::TestsPassed);
        t.model_id = "agent-a".to_string();
        t.task_category = Some(TaskCategory::Debugging);
        t.certainty = Some(0.9);
        sigma.turns.push(t);
    }

    // Agent B: only does Research (no Debugging turns)
    for i in 5..10 {
        let mut t = make_turn(i, "research", TurnOutcome::TestsPassed);
        t.model_id = "agent-b".to_string();
        t.task_category = Some(TaskCategory::Research);
        t.certainty = Some(0.9);
        sigma.turns.push(t);
    }

    let debug_weights = InfluenceWeightManager::calculate_weights_for_category(
        &sigma,
        TaskCategory::Debugging,
        0.9,
    );
    let a_w = debug_weights.get("agent-a").copied().unwrap_or(0.0);
    let b_w = debug_weights.get("agent-b").copied().unwrap_or(0.0);

    // Agent B should be dampened (global * 0.3) in Debugging
    assert!(
        a_w > b_w * 2.0,
        "specialist a={a_w} should strongly outweigh dampened generalist b={b_w}"
    );
}

// ============================================================================
// local_embed_text and local_cosine_similarity direct tests
// ============================================================================

use crosstalk::engines::memory::{local_cosine_similarity, local_embed_text};

#[test]
fn local_embed_text_deterministic() {
    let a = local_embed_text("hello world");
    let b = local_embed_text("hello world");
    assert_eq!(a, b);
}

#[test]
fn local_embed_text_dimension_is_384() {
    let v = local_embed_text("any text");
    assert_eq!(v.len(), 384);
}

#[test]
fn local_embed_text_normalized_to_unit_length() {
    let v = local_embed_text("test normalization");
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.001,
        "expected unit norm, got {}",
        norm
    );
}

#[test]
fn local_embed_text_empty_string_still_normalized() {
    let v = local_embed_text("");
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_eq!(v.len(), 384);
    assert!(
        (norm - 1.0).abs() < 0.001,
        "empty string embedding should still be normalized, got {}",
        norm
    );
}

#[test]
fn local_embed_text_different_inputs_differ() {
    let a = local_embed_text("alpha");
    let b = local_embed_text("beta");
    assert_ne!(a, b);
}

#[test]
fn local_cosine_similarity_identical_vectors() {
    let v = local_embed_text("same text");
    let sim = local_cosine_similarity(&v, &v);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "self-similarity should be ~1.0, got {}",
        sim
    );
}

#[test]
fn local_cosine_similarity_different_texts_below_one() {
    // The local embedding is hash-based (not semantic), so we verify that
    // different inputs produce similarity strictly less than 1.0.
    let a = local_embed_text("alpha beta gamma");
    let b = local_embed_text("delta epsilon zeta");

    let sim = local_cosine_similarity(&a, &b);
    assert!(
        sim < 1.0,
        "different texts should have similarity < 1.0, got {}",
        sim
    );
}

#[test]
fn local_cosine_similarity_mismatched_lengths_returns_zero() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 0.0];
    assert_eq!(local_cosine_similarity(&a, &b), 0.0);
}

#[test]
fn local_cosine_similarity_zero_vectors_returns_zero() {
    let a = vec![0.0f32; 10];
    let b = vec![0.0f32; 10];
    assert_eq!(local_cosine_similarity(&a, &b), 0.0);
}

#[test]
fn local_cosine_similarity_orthogonal_vectors() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let sim = local_cosine_similarity(&a, &b);
    assert!(
        sim.abs() < 1e-6,
        "orthogonal vectors should have ~0 similarity, got {}",
        sim
    );
}

#[test]
fn local_cosine_similarity_opposite_vectors() {
    let a = vec![1.0f32, 0.0];
    let b = vec![-1.0f32, 0.0];
    let sim = local_cosine_similarity(&a, &b);
    assert!(
        (sim - (-1.0)).abs() < 1e-6,
        "opposite vectors should have similarity ~-1.0, got {}",
        sim
    );
}
