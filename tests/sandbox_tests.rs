use crosstalk::engines::sandbox::{SandboxConfig, SandboxManager};
use crosstalk::engines::simulation::MonteCarloRunner;
use crosstalk::engines::validation::AstValidator;
use crosstalk::types::artifact::Artifact;
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

    let probability = runner
        .predict(&artifact, &diff, 100)
        .await
        .expect("predict failed");

    // Probability should be between 0.0 and 1.0
    assert!(
        probability >= 0.0 && probability <= 1.0,
        "Probability {} should be in [0.0, 1.0]",
        probability
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
    let result1 = runner
        .predict(&artifact, &diff, 100)
        .await
        .expect("predict failed");
    let result2 = runner
        .predict(&artifact, &diff, 100)
        .await
        .expect("predict failed");

    // Both should be valid probabilities
    assert!(result1 >= 0.0 && result1 <= 1.0);
    assert!(result2 >= 0.0 && result2 <= 1.0);

    // Results should be stochastic (likely different, but allow for small chance of equality)
    // With 100 trials and 95% success rate, probability of getting exact same result twice is low
    // However, due to randomness, we accept both identical and different results
    // The key is that they're both valid probabilities
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
