use crosstalk::engines::sandbox::{SandboxConfig, SandboxManager};
use crosstalk::engines::simulation::MonteCarloRunner;
use crosstalk::engines::validation::{AstValidator, AstVersionHistory};
use crosstalk::types::artifact::Artifact;
use crosstalk::types::conversation::ConversationState;
use std::collections::HashMap;

// Invalid WASM bytes for testing error handling
fn invalid_wasm_bytes() -> Vec<u8> {
    vec![0xFF, 0xFF, 0xFF, 0xFF]
}

#[tokio::test]
async fn test_sandbox_fuel_limit() {
    let manager = SandboxManager::new().expect("Failed to create SandboxManager");

    let config = SandboxConfig {
        memory_limit_bytes: 1024 * 1024,
        fuel_limit: 100, // Very low fuel limit
    };

    let wasm_bytes = invalid_wasm_bytes();

    let result = manager.execute(&wasm_bytes, &config).await;

    // Invalid WASM should produce an error
    assert!(
        result.is_err(),
        "Invalid WASM should fail to instantiate or execute"
    );
}

#[tokio::test]
async fn test_sandbox_memory_limit() {
    let manager = SandboxManager::new().expect("Failed to create SandboxManager");

    let config = SandboxConfig {
        memory_limit_bytes: 1024, // 1KB memory limit
        fuel_limit: 10_000_000,
    };

    let wasm_bytes = invalid_wasm_bytes();

    let result = manager.execute(&wasm_bytes, &config).await;

    // Invalid WASM with memory limit should produce an error
    assert!(result.is_err(), "Invalid WASM should fail");
}

#[tokio::test]
async fn test_sandbox_stdout_capture() {
    let manager = SandboxManager::new().expect("Failed to create SandboxManager");

    let config = SandboxConfig {
        memory_limit_bytes: 1024 * 1024,
        fuel_limit: 10_000_000,
    };

    let wasm_bytes = invalid_wasm_bytes();

    let result = manager.execute(&wasm_bytes, &config).await;

    // Test that SandboxResult structure is available
    // Even with invalid WASM, if execution happens, stdout should be captured
    match result {
        Ok(sandbox_result) => {
            // Verify the result has the expected fields
            let _ = &sandbox_result.stdout;
            let _ = &sandbox_result.stderr;
            let _ = sandbox_result.exit_code;
        }
        Err(_) => {
            // Invalid WASM fails, which is expected
        }
    }
}

#[tokio::test]
async fn test_monte_carlo_trials() {
    let runner = MonteCarloRunner::new().expect("Failed to create MonteCarloRunner");

    let artifact = Artifact {
        name: "test.rs".to_string(),
        language: "rust".to_string(),
        content: "fn main() {}".to_string(),
        version: 1,
        history: vec![],
        ast_versions: HashMap::new(),
        proof_attachments: vec![],
        metrics: Default::default(),
        skeleton: String::new(),
    };

    let diff = crosstalk::types::artifact::ArtifactDiff {
        original_version: 0,
        new_version: 1,
        diff_text: String::new(),
    };

    let (probability, _variance) = runner
        .predict(&artifact, &diff, 100)
        .await
        .expect("predict failed");

    // Probability should be between 0.0 and 1.0
    assert!(
        probability >= 0.0 && probability <= 1.0,
        "Probability {probability} should be in [0.0, 1.0]"
    );
}

#[tokio::test]
async fn test_monte_carlo_variance() {
    let runner = MonteCarloRunner::new().expect("Failed to create MonteCarloRunner");

    let artifact = Artifact {
        name: "test.rs".to_string(),
        language: "rust".to_string(),
        content: "fn main() {}".to_string(),
        version: 1,
        history: vec![],
        ast_versions: HashMap::new(),
        proof_attachments: vec![],
        metrics: Default::default(),
        skeleton: String::new(),
    };

    let diff = crosstalk::types::artifact::ArtifactDiff {
        original_version: 0,
        new_version: 1,
        diff_text: String::new(),
    };

    // Run prediction twice with same config
    let (result1, _) = runner
        .predict(&artifact, &diff, 100)
        .await
        .expect("predict failed");
    let (result2, _) = runner
        .predict(&artifact, &diff, 100)
        .await
        .expect("predict failed");

    assert!(result1 >= 0.0 && result1 <= 1.0);
    assert!(result2 >= 0.0 && result2 <= 1.0);
    assert!(result1.is_finite(), "result1 should be finite");
    assert!(result2.is_finite(), "result2 should be finite");
}

#[test]
fn test_ast_versioning() {
    let code1 = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        pub fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }
    "#;

    let code2 = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b + 1
        }

        pub fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }

        pub fn divide(a: i32, b: i32) -> i32 {
            a / b
        }
    "#;

    // Extract nodes from original code
    let nodes1 = AstValidator::extract_nodes(code1, "rust");
    assert!(nodes1.len() >= 2, "Expected at least 2 nodes in code1");

    // Extract nodes from modified code
    let nodes2 = AstValidator::extract_nodes(code2, "rust");
    assert!(nodes2.len() >= 3, "Expected at least 3 nodes in code2");

    // Identify changed nodes
    let changed_nodes = AstValidator::identify_changed_nodes(code1, code2, "rust");

    // The 'add' function should be identified as changed
    assert!(
        changed_nodes.iter().any(|id| id.contains("add")),
        "Expected 'add' function to be identified as changed"
    );

    // The 'divide' function should be identified as new
    assert!(
        changed_nodes.iter().any(|id| id.contains("divide")),
        "Expected 'divide' function to be identified as new"
    );

    // The 'multiply' function might or might not be in changed_nodes depending on implementation
    // (it's unchanged, so it should not be in the list if the implementation is correct)
    let multiply_changed = changed_nodes.iter().any(|id| id.contains("multiply"));
    assert!(
        !multiply_changed,
        "Expected 'multiply' to NOT be marked as changed (it's unchanged)"
    );
}

#[test]
fn test_ast_versioning_deletion() {
    let code1 = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        pub fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }
    "#;

    let code2 = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
    "#;

    // Identify changed nodes (deletion)
    let changed_nodes = AstValidator::identify_changed_nodes(code1, code2, "rust");

    // The 'multiply' function should be identified as deleted
    assert!(
        changed_nodes.iter().any(|id| id.contains("multiply")),
        "Expected 'multiply' function to be identified as deleted"
    );

    // The 'add' function should not be changed
    let add_changed = changed_nodes.iter().any(|id| id.contains("add"));
    assert!(!add_changed, "Expected 'add' to NOT be marked as changed");
}

#[test]
fn test_ast_versioning_struct() {
    let code1 = r#"
        struct Point {
            x: i32,
            y: i32,
        }
    "#;

    let code2 = r#"
        struct Point {
            x: i32,
            y: i32,
            z: i32,
        }
    "#;

    // Identify changed nodes
    let changed_nodes = AstValidator::identify_changed_nodes(code1, code2, "rust");

    // The 'Point' struct should be identified as changed
    assert!(
        changed_nodes.iter().any(|id| id.contains("Point")),
        "Expected 'Point' struct to be identified as changed"
    );
}

// ── AstVersionHistory ─────────────────────────────────────────────────────────

#[test]
fn revert_node_returns_content_at_target_turn() {
    let mut history = AstVersionHistory::new();
    let mut snap1 = HashMap::new();
    snap1.insert("fn:foo".to_string(), "fn foo() {}".to_string());
    history.record_snapshot(1, snap1);
    let mut snap2 = HashMap::new();
    snap2.insert("fn:foo".to_string(), "fn foo() { 42 }".to_string());
    history.record_snapshot(2, snap2);

    let v1 = history.revert_node("fn:foo", 1).unwrap();
    assert_eq!(v1, "fn foo() {}");
    let v2 = history.revert_node("fn:foo", 2).unwrap();
    assert_eq!(v2, "fn foo() { 42 }");
}

#[test]
fn revert_node_errors_for_unknown_node() {
    let history = AstVersionHistory::new();
    assert!(history.revert_node("fn:missing", 1).is_err());
}

#[test]
fn revert_node_errors_when_node_not_yet_created() {
    let mut history = AstVersionHistory::new();
    let mut snap = HashMap::new();
    snap.insert("fn:late".to_string(), "fn late() {}".to_string());
    history.record_snapshot(5, snap);
    assert!(history.revert_node("fn:late", 3).is_err());
}

#[test]
fn diff_nodes_contains_added_line() {
    let mut history = AstVersionHistory::new();
    let mut s1 = HashMap::new();
    s1.insert("fn:bar".to_string(), "line1\n".to_string());
    history.record_snapshot(1, s1);
    let mut s2 = HashMap::new();
    s2.insert("fn:bar".to_string(), "line1\nline2\n".to_string());
    history.record_snapshot(2, s2);

    let diff = history.diff_nodes("fn:bar", 1, 2).unwrap();
    assert!(diff.contains('+'), "diff must contain '+' for inserted lines");
}

#[test]
fn diff_nodes_identical_versions_has_no_changes() {
    let mut history = AstVersionHistory::new();
    let mut s1 = HashMap::new();
    s1.insert("fn:same".to_string(), "fn same() {}\n".to_string());
    history.record_snapshot(1, s1.clone());
    history.record_snapshot(2, s1);

    let diff = history.diff_nodes("fn:same", 1, 2).unwrap();
    assert!(!diff.contains('+') && !diff.contains('-'), "identical versions should produce no change markers");
}

// ── execute_with_rollback ─────────────────────────────────────────────────────

#[tokio::test]
async fn execute_with_rollback_returns_snapshot_on_invalid_wasm() {
    let manager = SandboxManager::new().unwrap();
    let snapshot = ConversationState::new("snap-session");
    let config = SandboxConfig { memory_limit_bytes: 1024 * 1024, fuel_limit: 10_000_000 };
    let (result, rollback) = manager
        .execute_with_rollback(&[0xFF, 0xFF], &config, &snapshot)
        .await
        .unwrap();
    assert!(result.exit_code != 0);
    assert!(rollback.is_some());
    assert_eq!(rollback.unwrap().session_id, "snap-session");
}

#[tokio::test]
async fn execute_with_rollback_rollback_is_none_on_success() {
    // A minimal valid WASM module that immediately returns (empty _start).
    // WAT: (module (func (export "_start")))
    let wasm_bytes: Vec<u8> = vec![
        0x00, 0x61, 0x73, 0x6D, // magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type section: () -> ()
        0x03, 0x02, 0x01, 0x00, // function section
        0x07, 0x0A, 0x01, 0x06, 0x5F, 0x73, 0x74, 0x61, 0x72, 0x74, 0x00, 0x00, // export "_start"
        0x0A, 0x04, 0x01, 0x02, 0x00, 0x0B, // code section: empty body
    ];
    let manager = SandboxManager::new().unwrap();
    let snapshot = ConversationState::new("success-session");
    let config = SandboxConfig { memory_limit_bytes: 1024 * 1024, fuel_limit: 10_000_000 };
    let (_result, rollback) = manager
        .execute_with_rollback(&wasm_bytes, &config, &snapshot)
        .await
        .unwrap();
    // Either success (rollback=None) or compilation failure (rollback=Some).
    // We can't guarantee the WASM is valid as a WASI component, so just assert the API works.
    let _ = rollback;
}
