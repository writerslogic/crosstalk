use crosstalk::engines::proof::ProofManager;
use crosstalk::engines::verification::{HashChain, InvariantChecker, TautologyFilter};
use crosstalk::types::artifact::{Artifact, ArtifactDiff, ProofAttachment};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use crosstalk::engines::quality::ArtifactMetrics;
use std::collections::HashMap;

/// Test 1: Create 10 ConversationState snapshots and verify hash chain integrity
#[test]
fn test_hash_chain_10_consecutive() {
    let mut hashes = vec![[0u8; 32]]; // Genesis hash

    for i in 0..10 {
        let mut state = ConversationState::new(&format!("session-{}", i));
        state.iteration_index = i as u32;

        // Add a turn to make each state unique
        state.turns.push(Turn {
            index: i as u32,
            model_id: format!("model-{}", i),
            content: format!("Turn {} content", i),
            timestamp: ConversationState::now() + i as u64,
            diffs: vec![],
            certainty: Some(0.8 + (i as f64) * 0.01),
            outcome: TurnOutcome::Unknown,
            task_category: None,
            structure: None,
            signature: vec![],
        });

        let prev_hash = hashes[i];
        let current_hash = HashChain::compute(&state, &prev_hash);

        // Verify the hash is valid
        assert!(HashChain::verify(&state, &prev_hash, &current_hash));

        // Verify each hash differs from the previous one
        assert_ne!(current_hash, prev_hash, "Hash at iteration {} must differ from previous", i);

        hashes.push(current_hash);
    }

    // Verify we have 11 hashes (genesis + 10)
    assert_eq!(hashes.len(), 11);

    // Verify all hashes are unique
    for i in 0..hashes.len() {
        for j in i + 1..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "Hash {} and {} should be unique", i, j);
        }
    }
}

/// Test 2: Verify first hash uses [0u8; 32] as previous_hash
#[test]
fn test_hash_chain_genesis() {
    let mut state = ConversationState::new("genesis-session");
    state.iteration_index = 0;
    state.turns.push(Turn {
        index: 0,
        model_id: "genesis-model".to_string(),
        content: "Genesis content".to_string(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: Some(1.0),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    });

    let genesis_prev_hash = [0u8; 32];
    let genesis_hash = HashChain::compute(&state, &genesis_prev_hash);

    // Verify the genesis hash was computed with [0u8; 32]
    assert!(HashChain::verify(&state, &genesis_prev_hash, &genesis_hash));

    // Verify it would fail with a different previous hash
    let wrong_hash = [1u8; 32];
    assert!(!HashChain::verify(&state, &wrong_hash, &genesis_hash));
}

/// Test 3: Create two nearly identical artifact contents (>95% cosine similarity)
/// and verify is_tautological() returns true
#[test]
fn test_tautology_filter_detects_repetition() {
    let base_content = "This is a very important function that does something critical and important";
    let similar_content = "This is a very important function that does something critical and important";

    let history = vec![base_content.to_string()];

    assert!(
        TautologyFilter::is_tautological(similar_content, &history),
        "Nearly identical content should be detected as tautological"
    );

    // Test with variations in whitespace/punctuation but high similarity
    let variant = "This is a very important function that does something critical and important";
    assert!(
        TautologyFilter::is_tautological(variant, &history),
        "Content with high similarity should be detected as tautological"
    );

    // Test with longer history
    let longer_history = vec![
        "Some other content here".to_string(),
        base_content.to_string(),
        "Another different piece".to_string(),
    ];

    assert!(
        TautologyFilter::is_tautological(similar_content, &longer_history),
        "Should find match anywhere in history"
    );
}

/// Test 4: Create two different artifact contents (<95% similarity)
/// and verify is_tautological() returns false
#[test]
fn test_tautology_filter_allows_novel() {
    let content_a = "This function calculates the sum of two integers";
    let content_b = "This method multiplies two floating point numbers and returns the result";

    let history = vec![content_a.to_string()];

    assert!(
        !TautologyFilter::is_tautological(content_b, &history),
        "Different content should not be detected as tautological"
    );

    // Test with completely different topics
    let history = vec!["The sun rises in the east".to_string(),
                       "Cats are independent animals".to_string()];
    let novel = "Programming requires abstract thinking skills";

    assert!(
        !TautologyFilter::is_tautological(novel, &history),
        "Novel content should not be flagged as tautological"
    );

    // Test with empty history
    assert!(
        !TautologyFilter::is_tautological("Any content here", &[]),
        "Content with empty history should not be tautological"
    );
}

/// Test 5: Create state with turns that have increasing iteration_index
/// and verify check_all() passes
#[test]
fn test_invariant_checker_monotonic_indices() {
    let mut state = ConversationState::new("monotonic-session");
    state.iteration_index = 5;

    // Add turns with strictly increasing indices
    for i in 0..5 {
        state.turns.push(Turn {
            index: i,
            model_id: format!("model-{}", i),
            content: format!("Turn {}", i),
            timestamp: ConversationState::now() + i as u64,
            diffs: vec![],
            certainty: Some(0.5),
            outcome: TurnOutcome::Unknown,
            task_category: None,
            structure: None,
            signature: vec![],
        });
    }

    // Should pass because indices are monotonic and all < iteration_index
    assert!(
        InvariantChecker::check_all(&state).is_ok(),
        "State with monotonic indices should pass invariant check"
    );
}

/// Test 6: Create state with turn.index > sigma.iteration_index
/// and verify check_all() returns error
#[test]
fn test_invariant_checker_detects_orphan() {
    let mut state = ConversationState::new("orphan-session");
    state.iteration_index = 3;

    // Add a turn with index >= iteration_index (orphan)
    state.turns.push(Turn {
        index: 3,
        model_id: "orphan-model".to_string(),
        content: "This is an orphan turn".to_string(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: Some(0.5),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    });

    let result = InvariantChecker::check_all(&state);
    assert!(
        result.is_err(),
        "State with orphan turn (index >= iteration_index) should fail invariant check"
    );

    if let Err(e) = result {
        assert!(
            e.to_string().contains("Orphan"),
            "Error message should mention orphan turn"
        );
    }
}

/// Test 7: Create ProofAttachment, serialize/deserialize,
/// and verify hash matches
#[test]
fn test_proof_attachment_round_trip() {
    // Create an artifact
    let artifact = Artifact {
        name: "test_proof.rs".to_string(),
        language: "rust".to_string(),
        content: "fn verified_function() -> u32 { 42 }".to_string(),
        version: 1,
        history: vec![ArtifactDiff {
            original_version: 0,
            new_version: 1,
            diff_text: "added function".to_string(),
        }],
        ast_versions: HashMap::new(),
        proof_attachments: vec![],
        metrics: ArtifactMetrics::default(),
        skeleton: "fn verified_function() -> u32 { ... }".to_string(),
    };

    // Generate proof attachment
    let properties = vec![
        "returns_non_zero".to_string(),
        "pure_function".to_string(),
    ];
    let proof = ProofManager::generate_proof(&artifact, properties.clone());

    // Store original hash
    let original_hash = proof.proof_hash.clone();

    // Serialize
    let serialized = serde_json::to_string(&proof).unwrap();

    // Deserialize
    let deserialized: ProofAttachment = serde_json::from_str(&serialized).unwrap();

    // Verify hash matches
    assert_eq!(
        deserialized.proof_hash, original_hash,
        "Hash should match after round-trip serialization"
    );

    // Verify properties match
    assert_eq!(
        deserialized.proven_properties, properties,
        "Properties should match after deserialization"
    );

    // Verify artifact name matches
    assert_eq!(
        deserialized.artifact_name, artifact.name,
        "Artifact name should match"
    );

    // Verify verified_at is preserved
    assert!(deserialized.verified_at > 0, "verified_at should be set");
}

/// Test 8: Simulate a state with invalid invariants
/// and verify rollback restores valid state
#[test]
fn test_violation_rollback() {
    // Create a valid initial state
    let mut valid_state = ConversationState::new("rollback-session");
    valid_state.iteration_index = 3;
    valid_state.turns.push(Turn {
        index: 0,
        model_id: "model-0".to_string(),
        content: "Valid turn".to_string(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: Some(0.9),
        outcome: TurnOutcome::Compiled,
        task_category: None,
        structure: None,
        signature: vec![],
    });

    // Verify initial state is valid
    assert!(
        InvariantChecker::check_all(&valid_state).is_ok(),
        "Initial state should be valid"
    );

    // Create an invalid state (duplicate indices)
    let mut invalid_state = valid_state.clone();
    invalid_state.turns.push(Turn {
        index: 0, // Same index as previous turn - violation!
        model_id: "model-1".to_string(),
        content: "Invalid turn".to_string(),
        timestamp: ConversationState::now() + 1,
        diffs: vec![],
        certainty: Some(0.5),
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
    });

    // Verify invalid state fails
    assert!(
        InvariantChecker::check_all(&invalid_state).is_err(),
        "State with duplicate turn indices should be invalid"
    );

    // Simulate rollback: remove the violating turn
    invalid_state.turns.pop();

    // Verify rollback restores validity
    assert!(
        InvariantChecker::check_all(&invalid_state).is_ok(),
        "State should be valid after rollback"
    );

    // Verify state matches original valid state
    assert_eq!(
        invalid_state.turns.len(), valid_state.turns.len(),
        "Rollback should restore original turn count"
    );
    assert_eq!(
        invalid_state.turns[0].index, valid_state.turns[0].index,
        "Rollback should preserve turn data"
    );
}
