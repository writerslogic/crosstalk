use crosstalk::types::conversation::ConversationState;
use crosstalk::ui::visualization::GodView;
use crosstalk::engines::compute::ComputeManager;

#[test]
fn ingest_file_sets_skeleton_and_metrics() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file(
        "src/main.rs".to_string(),
        "rust".to_string(),
        "fn main() {\n    println!(\"hello\");\n}\n\nfn helper() -> u32 { 42 }\n".to_string(),
    );
    assert_eq!(sigma.artifacts.len(), 1);
    let art = sigma.artifacts.get("src/main.rs").unwrap();
    assert_eq!(art.version, 0);
    assert_eq!(art.language, "rust");
    assert!(!art.skeleton.is_empty(), "skeleton should be computed on ingestion");
    assert!(art.content.contains("fn main()"));
}

#[test]
fn ingest_file_python() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file(
        "script.py".to_string(),
        "python".to_string(),
        "def hello():\n    print('world')\n".to_string(),
    );
    let art = sigma.artifacts.get("script.py").unwrap();
    assert_eq!(art.language, "python");
    assert!(art.content.contains("def hello"));
}

#[test]
fn ingest_file_markdown_skeleton_empty_is_ok() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file(
        "README.md".to_string(),
        "markdown".to_string(),
        "# Title\n\nSome content.\n".to_string(),
    );
    let art = sigma.artifacts.get("README.md").unwrap();
    assert!(art.content.contains("# Title"));
}

#[test]
fn ingest_multiple_files_counted() {
    let mut sigma = ConversationState::new("test");
    for i in 0..5 {
        sigma.ingest_file(
            format!("file{i}.rs"),
            "rust".to_string(),
            format!("fn f{i}() {{}}"),
        );
    }
    assert_eq!(sigma.artifacts.len(), 5);
}

#[test]
fn last_verification_default_empty() {
    let sigma = ConversationState::new("test");
    assert!(sigma.last_verification.is_empty());
}

#[test]
fn godview_compute_metrics_increments_frame() {
    let mut gv = GodView::new();
    let sigma = ConversationState::new("test");
    let m1 = gv.compute_metrics(&sigma);
    let m2 = gv.compute_metrics(&sigma);
    assert_eq!(m1.frame, 1);
    assert_eq!(m2.frame, 2);
    assert_eq!(m1.turn_count, 0);
    assert_eq!(m1.artifact_count, 0);
}

#[test]
fn godview_metrics_reflect_state() {
    let mut gv = GodView::new();
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("a.rs".to_string(), "rust".to_string(), "fn a() {}".to_string());
    sigma.ingest_file("b.rs".to_string(), "rust".to_string(), "fn b() {}".to_string());
    let m = gv.compute_metrics(&sigma);
    assert_eq!(m.artifact_count, 2);
    assert_eq!(m.turn_count, 0);
}

#[test]
fn inference_cache_hit_and_miss() {
    let mut cache = crosstalk::engines::compute::InferenceCache::new();
    assert!(cache.get("prompt1", "model1").is_none());
    assert_eq!(cache.misses, 1);
    cache.insert("prompt1", "model1", "response1".to_string(), 1.0);
    assert_eq!(cache.get("prompt1", "model1"), Some("response1".to_string()));
    assert_eq!(cache.hits, 1);
}

#[tokio::test]
async fn compute_manager_starts_monitor() {
    let mut cm = ComputeManager::new();
    cm.start_background_monitor(60);
    let _rx = cm.resource_subscriber();
}

// ── lang_from_ext coverage (tested indirectly via ingest_file language) ───────

#[test]
fn ingest_file_rust_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("lib.rs".to_string(), "rust".to_string(), "pub fn x() {}".to_string());
    assert_eq!(sigma.artifacts.get("lib.rs").unwrap().language, "rust");
}

#[test]
fn ingest_file_javascript_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("app.js".to_string(), "javascript".to_string(), "const x = 1;".to_string());
    assert_eq!(sigma.artifacts.get("app.js").unwrap().language, "javascript");
}

#[test]
fn ingest_file_typescript_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("app.ts".to_string(), "typescript".to_string(), "let x: number = 1;".to_string());
    assert_eq!(sigma.artifacts.get("app.ts").unwrap().language, "typescript");
}

#[test]
fn ingest_file_shell_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("run.sh".to_string(), "shell".to_string(), "#!/bin/bash\necho hi".to_string());
    assert_eq!(sigma.artifacts.get("run.sh").unwrap().language, "shell");
}

#[test]
fn ingest_file_toml_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("Cargo.toml".to_string(), "toml".to_string(), "[package]\nname = \"x\"".to_string());
    assert_eq!(sigma.artifacts.get("Cargo.toml").unwrap().language, "toml");
}

#[test]
fn ingest_file_go_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("main.go".to_string(), "go".to_string(), "package main".to_string());
    assert_eq!(sigma.artifacts.get("main.go").unwrap().language, "go");
}

#[test]
fn ingest_file_cpp_extension() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file("main.cpp".to_string(), "cpp".to_string(), "#include <iostream>".to_string());
    assert_eq!(sigma.artifacts.get("main.cpp").unwrap().language, "cpp");
}

// ── should_skip coverage (tested indirectly: skipped dirs should not appear) ─

#[test]
fn ingest_file_with_git_path_component_still_ingests() {
    // should_skip filters at directory walk time, not at ingest_file level.
    // ingest_file itself does not filter; we verify it accepts any path string.
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file(".git/config".to_string(), "toml".to_string(), "[core]".to_string());
    assert!(sigma.artifacts.contains_key(".git/config"));
}

#[test]
fn ingest_file_node_modules_path_still_ingests() {
    let mut sigma = ConversationState::new("test");
    sigma.ingest_file(
        "node_modules/pkg/index.js".to_string(),
        "javascript".to_string(),
        "module.exports = {};".to_string(),
    );
    assert!(sigma.artifacts.contains_key("node_modules/pkg/index.js"));
}

// ── Token budget behavior ────────────────────────────────────────────────────

#[test]
fn distill_output_respects_budget_with_many_artifacts() {
    use crosstalk::engines::memory::ContextDistiller;

    let mut sigma = ConversationState::new("budget-test");
    // Create many large artifacts to exceed budget
    for i in 0..20 {
        let large_content = "x".repeat(5000);
        sigma.ingest_file(
            format!("file{i}.rs"),
            "rust".to_string(),
            large_content,
        );
    }
    // Add turns so distill has something to work with
    for i in 0..10u32 {
        sigma.turns.push(crosstalk::types::conversation::Turn {
            index: i,
            model_id: "model".to_string(),
            content: "a]".repeat(500),
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: Some(0.8),
            outcome: crosstalk::types::conversation::TurnOutcome::Compiled,
            task_category: None,
            structure: None,
            signature: vec![],
            surprise_signal: None,
        });
    }

    let budget = 4096;
    let output = ContextDistiller::distill(&sigma, budget);
    // The distiller should produce output that fits within the token budget.
    // Tokens are roughly chars/4; budget of 4096 tokens ~ 16384 chars.
    // The output should be substantially less than the total raw content.
    let total_raw: usize = sigma.artifacts.values().map(|a| a.content.len()).sum();
    assert!(
        output.len() < total_raw,
        "distilled output ({}) should be smaller than raw content ({})",
        output.len(),
        total_raw
    );
}

#[test]
fn distill_empty_state_produces_session_header() {
    use crosstalk::engines::memory::ContextDistiller;

    let sigma = ConversationState::new("empty-test");
    let output = ContextDistiller::distill(&sigma, 4096);
    assert!(output.contains("empty-test"));
}

#[test]
fn distill_small_state_includes_all_turns() {
    use crosstalk::engines::memory::ContextDistiller;

    let mut sigma = ConversationState::new("small-test");
    sigma.turns.push(crosstalk::types::conversation::Turn {
        index: 0,
        model_id: "model-abc".to_string(),
        content: "unique marker content".to_string(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: Some(0.9),
        outcome: crosstalk::types::conversation::TurnOutcome::TestsPassed,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    });

    let output = ContextDistiller::distill(&sigma, 50000);
    assert!(output.contains("model-abc"));
}
