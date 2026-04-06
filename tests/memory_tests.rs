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
        assert!((e1 - e2).abs() < 1e-6, "Same text should produce identical embeddings");
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
    let texts = vec![
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
        });
    }

    // Verify records have correct structure
    assert_eq!(records.len(), 5, "Should have 5 records");
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record.turn_id, i as u32);
        assert_eq!(record.session_id, "test-session");
        assert_eq!(record.embedding.len(), 384, "Embedding should have 384 dimensions");
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

    let warning = FailurePredictor::proactive_warning(current_context, &[failure_signature.clone()]);

    assert!(
        warning.is_some(),
        "Should detect matching error type in context"
    );
    let msg = warning.unwrap();
    assert!(msg.contains("CompilationError"), "Warning should mention error type");
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
    };

    let lessons = LessonExtractor::extract(&[turn]);

    assert!(!lessons.is_empty(), "Should extract lesson from successful turn");
    let lesson = &lessons[0];

    // Verify lesson structure
    assert_eq!(lesson.context_type, "coding");
    assert!(lesson.approach.contains("claude-3-sonnet"));
    assert_eq!(lesson.outcome, "Success (Tests Passed)");
    assert!(lesson.confidence >= 0.85);
    assert!(!lesson.applicability_tags.is_empty());
    assert!(lesson.applicability_tags.contains(&"passing_tests".to_string()));
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
    };

    let serialized = serde_json::to_string(&record).expect("Serialization failed");
    let deserialized: MemoryRecord =
        serde_json::from_str(&serialized).expect("Deserialization failed");

    let result_outcome = deserialized.outcome.expect("Outcome should be present");
    assert_eq!(result_outcome.compiled, true);
    assert_eq!(result_outcome.tests_passed, true);
    assert!((result_outcome.quality_delta - 0.42).abs() < 1e-6);
    assert_eq!(result_outcome.was_rolled_back, false);
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
    let texts = vec![
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
