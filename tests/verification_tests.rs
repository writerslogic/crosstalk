use crosstalk::engines::proof::ProofManager;
use crosstalk::engines::quality::ArtifactMetrics;
use crosstalk::engines::verification::{HashChain, InvariantChecker, TautologyFilter};
use crosstalk::types::artifact::{Artifact, ArtifactDiff};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

#[test]
fn test_hash_chain_10_consecutive() {
    let mut hashes = vec![[0u8; 32]];

    for i in 0..10 {
        let mut state = ConversationState::new(&format!("session-{}", i));
        state.iteration_index = i as u32;

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
            surprise_signal: None,
            consistency_score: None,
            diff_quality_score: None,
            persona_disclosure: None,
        });

        let prev_hash = hashes[i];
        let current_hash = HashChain::compute(&state, &prev_hash).expect("Hash computation failed");

        assert!(HashChain::verify(&state, &prev_hash, &current_hash).expect("verify failed"));
        assert_ne!(current_hash, prev_hash);
        hashes.push(current_hash);
    }
    assert_eq!(hashes.len(), 11);
}

#[test]
fn test_invariant_checker_monotonic_indices() {
    let mut state = ConversationState::new("test");
    state.turns.push(Turn {
        index: 1,
        model_id: "m".to_string(),
        content: "c".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: None,
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    });
    state.turns.push(Turn {
        index: 0,
        model_id: "m".to_string(),
        content: "c".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: None,
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    });

    assert!(InvariantChecker::check_all(&state).is_err());
}

#[test]
fn test_tautology_filter_detects_identical() {
    let history = vec!["Hello world".to_string()];
    assert!(TautologyFilter::is_tautological("Hello world", &history));
}

#[test]
fn test_tautology_filter_detects_similar() {
    let history = vec!["This is a long sentence that repeats itself often and again and again in the same way over and over.".to_string()];
    assert!(TautologyFilter::is_tautological(
        "This is a long sentence that repeats itself often and again and again in the same way over and over!",
        &history
    ));
}

#[test]
fn test_hash_chain_determinism() {
    let mut s1 = ConversationState::new("test");
    s1.artifacts.insert(
        "a".to_string(),
        Arc::new(Artifact {
            name: "a".to_string(),
            content: "c".to_string(),
            language: "rust".to_string(),
            version: 0,
            history: vec![],
            ast_versions: BTreeMap::new(),
            proof_attachments: vec![],
            metrics: ArtifactMetrics::default(),
            skeleton: "".to_string(),
        }),
    );
    let s2 = s1.clone();

    let h1 = HashChain::compute(&s1, &[0u8; 32]).unwrap();
    let h2 = HashChain::compute(&s2, &[0u8; 32]).unwrap();
    assert_eq!(h1, h2);
}

#[test]
fn test_proof_attachment_round_trip() {
    let artifact = Artifact {
        name: "test.rs".to_string(),
        content: "fn main() {}".to_string(),
        language: "rust".to_string(),
        version: 1,
        history: vec![ArtifactDiff {
            original_version: 0,
            new_version: 1,
            diff_text: "added function".to_string(),
        }],
        ast_versions: BTreeMap::new(),
        proof_attachments: vec![],
        metrics: ArtifactMetrics::default(),
        skeleton: "fn verified_function() -> u32 { ... }".to_string(),
    };

    let proof = ProofManager::generate_proof(&artifact, vec!["type_safety".to_string()]);
    assert_eq!(proof.artifact_name, "test.rs");
    assert!(proof.proven_properties.contains(&"type_safety".to_string()));
}
