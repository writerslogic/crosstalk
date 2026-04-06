use crosstalk::engines::compute::{BatchScheduler, ComputeManager, LatencyRouter, RateLimitManager, Urgency};
use crosstalk::engines::intelligence::IntelligenceEngine;
use crosstalk::engines::reasoning::{
    ArgumentNodeType, ArgumentParser, AssumptionExtractor, CrossExaminer, FallacyDetector,
    ReasoningScorer, ReportGenerator, StructureSelector, SynthesisEngine,
};
use crosstalk::engines::security::{
    AuditRunner, ExfilBlock, FuzzRunner, RiskLevel, SecretScanner, ShellSanity, TurnSigner,
    ZeroTrustPolicy,
};
use crosstalk::engines::validation::AstValidator;
use crosstalk::types::compute::{BudgetLedger, BudgetMode, CostEntry, ModelCapabilityMatrix, TokenUsage};
use crosstalk::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crosstalk::types::intelligence::ModelProfile;
use std::collections::HashMap;

#[test]
fn test_generate_skeleton() {
    let code = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
        struct Point { x: i32, y: i32 }
        impl Point {
            fn new() -> Self { Point { x: 0, y: 0 } }
        }
    "#;
    let skeleton = AstValidator::generate_skeleton(code, "rust");
    assert!(skeleton.contains("pub fn add(a: i32, b: i32) -> i32 { ... }"));
    assert!(skeleton.contains("struct Point { x: i32, y: i32 }"));
    assert!(skeleton.contains("impl Point {"));
    assert!(skeleton.contains("fn new() -> Self { ... }"));
    assert!(!skeleton.contains("a + b"));
}

#[test]
fn test_secret_scanner() {
    let content = "My key is AKIA1234567890ABCDEF";
    assert_eq!(SecretScanner::scan(content).len(), 1);
}

#[test]
fn test_shell_sanity() {
    assert!(ShellSanity::is_dangerous("rm -rf /"));
    assert!(!ShellSanity::is_dangerous("cargo test"));
}

#[test]
fn test_turn_signer() {
    let signer = TurnSigner::new();
    let data = b"turn data";
    let sig = signer.sign(data);
    assert!(signer.verify(data, &sig));
}

#[test]
fn test_detect_regression() {
    let mut engine = IntelligenceEngine::new();

    let baseline_score = 0.8;
    let mut profile = ModelProfile {
        model_id: "test-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: Default::default(),
    };

    let mut baseline_avg = crosstalk::types::intelligence::RunningAverage::default();
    for _ in 0..10 {
        baseline_avg.update(baseline_score);
    }
    profile.task_scores.insert(TaskCategory::CodeGeneration, baseline_avg);
    engine.profiles.insert("test-model".to_string(), profile);

    let mut recent_turns = Vec::new();
    let _low_score = baseline_score * 0.8;
    for i in 0..5 {
        let turn = Turn {
            index: i,
            model_id: "test-model".to_string(),
            content: "Low quality output".to_string(),
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: None,
            outcome: TurnOutcome::Unknown,
            task_category: Some(TaskCategory::CodeGeneration),
            structure: None,
            signature: vec![],

            surprise_signal: None,
        };
        recent_turns.push(turn);
    }

    let alert = engine.detect_regression("test-model", &recent_turns);
    assert!(alert.is_some(), "Expected regression to be detected");

    let alert = alert.unwrap();
    assert_eq!(alert.agent_id, "test-model");
    assert!(alert.recent_mean < alert.baseline_mean * 0.9);
}

#[test]
fn test_no_regression_when_above_threshold() {
    let mut engine = IntelligenceEngine::new();

    let baseline_score = 0.8;
    let mut profile = ModelProfile {
        model_id: "test-model".to_string(),
        task_scores: HashMap::new(),
        total_turns: 0,
        last_updated: ConversationState::now(),
        latency_ms: Default::default(),
    };

    let mut baseline_avg = crosstalk::types::intelligence::RunningAverage::default();
    for _ in 0..10 {
        baseline_avg.update(baseline_score);
    }
    profile.task_scores.insert(TaskCategory::CodeGeneration, baseline_avg);
    engine.profiles.insert("test-model".to_string(), profile);

    let mut recent_turns = Vec::new();
    for i in 0..5 {
        let turn = Turn {
            index: i,
            model_id: "test-model".to_string(),
            content: "Good quality output with evidence and code".to_string(),
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: None,
            outcome: TurnOutcome::TestsPassed,
            task_category: Some(TaskCategory::CodeGeneration),
            structure: None,
            signature: vec![],

            surprise_signal: None,
        };
        recent_turns.push(turn);
    }

    let alert = engine.detect_regression("test-model", &recent_turns);
    assert!(alert.is_none(), "Expected no regression when above threshold");
}

// ── Track 11: Compute ─────────────────────────────────────────────────────────

#[test]
fn test_budget_remaining_and_mode() {
    let mut ledger = BudgetLedger { session_budget: 1.0, spent: 0.85, entries: vec![] };
    assert!((ledger.remaining() - 0.15).abs() < 1e-9);
    assert_eq!(ledger.mode(), BudgetMode::CostReduction);
    ledger.spent = 0.97;
    assert_eq!(ledger.mode(), BudgetMode::Emergency);
    ledger.spent = 0.50;
    assert_eq!(ledger.mode(), BudgetMode::Normal);
}

#[test]
fn test_budget_burn_rate_zero_when_no_entries() {
    let ledger = BudgetLedger::default();
    assert_eq!(ledger.burn_rate(), 0.0);
}

#[test]
fn test_budget_summary_contains_fields() {
    let ledger = BudgetLedger { session_budget: 2.0, spent: 0.5, entries: vec![] };
    let s = ledger.summary();
    assert!(s.contains("spent="), "summary must include spent");
    assert!(s.contains("remaining="), "summary must include remaining");
    assert!(s.contains("mode="), "summary must include mode");
}

#[test]
fn test_manage_budget_returns_mode() {
    let mut sigma = ConversationState::new("budget-test");
    sigma.budget.session_budget = 1.0;
    let entry = CostEntry {
        turn_id: 1,
        model_id: "gpt".to_string(),
        usage: TokenUsage { input_tokens: 100, output_tokens: 50, total_tokens: 150 },
        cost_usd: 0.85,
        latency_ms: 200,
        timestamp: 0,
    };
    let mode = ComputeManager::manage_budget(&mut sigma, entry);
    assert_eq!(mode, BudgetMode::CostReduction);
}

#[test]
fn test_rate_limit_jitter_varies() {
    let mut mgr = RateLimitManager::new();
    mgr.report_429("model-a");
    let delays: Vec<u64> = (0..20).map(|_| mgr.get_delay("model-a").as_millis() as u64).collect();
    let min = *delays.iter().min().unwrap();
    let max = *delays.iter().max().unwrap();
    // With ±25% jitter on base 2s, range should be ~1500ms..2500ms
    assert!(min < max, "jitter should produce varying delays");
    assert!(min > 500, "delay should be > 500ms after one 429");
    assert!(max < 4000, "delay should be < 4000ms after one 429");
}

#[tokio::test]
async fn test_batch_scheduler_limits_concurrency() {
    let sched = BatchScheduler::new(2);
    assert_eq!(sched.available_permits(), 2);
    let _p1 = sched.acquire().await;
    assert_eq!(sched.available_permits(), 1);
    let _p2 = sched.acquire().await;
    assert_eq!(sched.available_permits(), 0);
    drop(_p1);
    assert_eq!(sched.available_permits(), 1);
}

#[test]
fn test_latency_router_filters_by_urgency() {
    let mut router = LatencyRouter::new();
    router.record("fast-model", 200);
    router.record("slow-model", 5000);
    let all = vec!["fast-model".to_string(), "slow-model".to_string()];

    let interactive = router.filter(&all, Urgency::Interactive);
    assert!(interactive.iter().any(|m| *m == "fast-model"));
    assert!(!interactive.iter().any(|m| *m == "slow-model"));

    let batch = router.filter(&all, Urgency::Batch);
    assert_eq!(batch.len(), 2);
}

#[test]
fn test_model_capability_matrix() {
    let mut matrix = ModelCapabilityMatrix::default();
    matrix.register("gpt4", "CodeGeneration", 0.9);
    assert!((matrix.score("gpt4", "CodeGeneration") - 0.9).abs() < 1e-9);
    assert_eq!(matrix.score("gpt4", "nonexistent"), 0.0);
    assert_eq!(matrix.score("unknown", "anything"), 0.0);
    matrix.register("gpt4", "clamp_test", 1.5);
    assert_eq!(matrix.score("gpt4", "clamp_test"), 1.0);
}

// ── Track 12: Reasoning ───────────────────────────────────────────────────────

#[test]
fn test_fallacy_detector_false_dichotomy() {
    let content = "There are only two options: you're with us or against us.";
    let reports = FallacyDetector::scan(content);
    assert!(
        reports.iter().any(|r| r.fallacy_type == "FalseDichotomy"),
        "should detect FalseDichotomy"
    );
}

#[test]
fn test_fallacy_detector_straw_man() {
    let content = "Their argument is that we should use Rust for everything. This is clearly wrong.";
    let reports = FallacyDetector::scan(content);
    assert!(
        reports.iter().any(|r| r.fallacy_type == "StrawMan"),
        "should detect StrawMan"
    );
}

#[test]
fn test_fallacy_detector_clean_content_returns_empty() {
    let content = "The CPU processes instructions in a fetch-decode-execute cycle. \
                   Modern pipelines improve throughput via parallel stage execution.";
    let reports = FallacyDetector::scan(content);
    assert!(reports.is_empty(), "clean factual content should produce no fallacies");
}

#[test]
fn test_assumption_extractor_finds_markers() {
    let content = "Assuming the API is stable, we can proceed. The server is ready.";
    let assumptions = AssumptionExtractor::extract(content);
    assert!(!assumptions.is_empty(), "should find at least one assumption");
    assert!(assumptions.iter().any(|a| a.to_lowercase().contains("assuming")));
}

#[test]
fn test_cross_examiner_generates_questions_for_causal_claim() {
    let argument = "Using Rust causes a 30% performance improvement over C++.";
    let questions = CrossExaminer::generate_questions(argument);
    assert!(!questions.is_empty());
    assert!(
        questions.iter().any(|q| q.to_lowercase().contains("evidence") || q.to_lowercase().contains("causal")),
        "should ask for causal evidence"
    );
}

#[test]
fn test_cross_examiner_questions_for_universal_claim() {
    let argument = "All developers always prefer type-safe languages.";
    let questions = CrossExaminer::generate_questions(argument);
    assert!(
        questions.iter().any(|q| q.to_lowercase().contains("counterexample")),
        "should challenge universal claim"
    );
}

#[test]
fn test_argument_parser_identifies_premise_and_conclusion() {
    let text = "Premise: Rust has no garbage collector.\nConclusion: Rust avoids GC pauses.";
    let graph = ArgumentParser::parse(text);
    assert!(graph.nodes.len() >= 2);
    assert!(graph.nodes.iter().any(|n| n.node_type == ArgumentNodeType::Premise));
    assert!(graph.nodes.iter().any(|n| n.node_type == ArgumentNodeType::Conclusion));
}

#[test]
fn test_structure_selector_recommends_highest_quality() {
    let mut sel = StructureSelector::new();
    for _ in 0..3 {
        sel.record_outcome(TaskCategory::CodeGeneration, "agent-a", TurnStructure::StepByStep, 0.9);
    }
    sel.record_outcome(TaskCategory::CodeGeneration, "agent-a", TurnStructure::FreeForm, 0.5);
    assert_eq!(
        sel.recommend(TaskCategory::CodeGeneration, "agent-a"),
        TurnStructure::StepByStep
    );
}

#[test]
fn test_structure_selector_falls_back_to_freeform() {
    let sel = StructureSelector::new();
    assert_eq!(
        sel.recommend(TaskCategory::CodeGeneration, "agent-a"),
        TurnStructure::FreeForm
    );
}

#[test]
fn test_report_generator_includes_all_sections() {
    use crosstalk::engines::reasoning::ReasoningEngine;
    let content = "decision: use caching\nIs this correct?";
    let signals = ReasoningEngine::extract_signals(content);
    let fallacies = FallacyDetector::scan(content);
    let assumptions = AssumptionExtractor::extract(content);
    let report = ReportGenerator::generate(&signals, &fallacies, &assumptions, 0.75);
    assert!(report.contains("## Reasoning Report"));
    assert!(report.contains("Decisions"));
    assert!(report.contains("Fallacies"));
    assert!(report.contains("Assumptions"));
    assert!(report.contains("0.75"));
}

#[test]
fn test_reasoning_scorer_rewards_structured_turns() {
    let turn = Turn {
        index: 1,
        model_id: "model".to_string(),
        content: "decision: use caching\n```rust\nfn cache() {}\n```".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: None,
        outcome: TurnOutcome::TestsPassed,
        task_category: Some(TaskCategory::CodeGeneration),
        structure: Some(TurnStructure::StepByStep),
        signature: vec![],
        surprise_signal: None,
    };
    assert!(ReasoningScorer::score(&turn) > 0.6, "structured turn with decisions+code should score > 0.6");
}

#[test]
fn test_reasoning_scorer_penalises_fallacies() {
    let turn = Turn {
        index: 1,
        model_id: "model".to_string(),
        content: "There are only two options: Rust or failure. Their argument is wrong. This is clearly wrong.".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: None,
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    };
    assert!(ReasoningScorer::score(&turn) < 0.6, "turn with multiple fallacies should score < 0.6");
}

// ── Track 17: Security acceptance criteria ────────────────────────────────────

fn make_signed_turn(signer: &TurnSigner, index: u32) -> Turn {
    let mut t = Turn {
        index,
        model_id: "model".to_string(),
        content: format!("content-{index}"),
        timestamp: index as u64,
        diffs: vec![],
        certainty: Some(0.8),
        outcome: TurnOutcome::Compiled,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    };
    let data = serde_json::to_vec(&t).unwrap();
    t.signature = signer.sign(&data);
    t
}

#[test]
fn secret_scanner_detects_aws_key_with_line_number() {
    let content = "line one\nkey=AKIAIOSFODNN7EXAMPLE1\nline three";
    let findings = SecretScanner::scan_text(content);
    assert!(!findings.is_empty(), "must detect the AWS key");
    assert_eq!(findings[0].line, 2);
    assert_eq!(findings[0].pattern_name, "AWS_ACCESS_KEY");
    assert!(findings[0].redacted_match.ends_with("***"));
}

#[test]
fn turn_signer_chain_all_valid() {
    let signer = TurnSigner::new();
    let turns: Vec<Turn> = (0..10).map(|i| make_signed_turn(&signer, i)).collect();
    assert!(signer.verify_chain(&turns), "all 10 turns must verify");
}

#[test]
fn turn_signer_chain_detects_tampered_turn() {
    let signer = TurnSigner::new();
    let mut turns: Vec<Turn> = (0..10).map(|i| make_signed_turn(&signer, i)).collect();
    turns[4].content.push('!');
    assert!(!signer.verify_chain(&turns), "tampered turn must fail verification");
}

#[test]
fn exfil_block_strips_proxy_vars() {
    let env: std::collections::HashMap<String, String> = [
        ("HTTP_PROXY".to_string(), "http://proxy:8080".to_string()),
        ("HTTPS_PROXY".to_string(), "http://proxy:8080".to_string()),
        ("PATH".to_string(), "/usr/bin".to_string()),
    ]
    .into_iter()
    .collect();
    let clean = ExfilBlock::sanitize_env(env);
    assert!(!clean.contains_key("HTTP_PROXY"));
    assert!(!clean.contains_key("HTTPS_PROXY"));
    assert!(clean.contains_key("PATH"), "non-proxy vars must be preserved");
}

#[test]
fn risk_level_rm_is_critical() {
    let policy = ZeroTrustPolicy::new();
    assert_eq!(policy.classify("rm", "-rf /"), RiskLevel::Critical);
}

#[test]
fn risk_level_cargo_test_is_low() {
    let policy = ZeroTrustPolicy::new();
    assert_eq!(policy.classify("cargo", "test"), RiskLevel::Low);
}

#[test]
fn risk_level_git_push_is_high() {
    let policy = ZeroTrustPolicy::new();
    assert_eq!(policy.classify("git", "push origin main"), RiskLevel::High);
}

#[test]
fn risk_level_confirmation_required_for_medium_and_above() {
    let policy = ZeroTrustPolicy::new();
    assert!(policy.requires_confirmation(&RiskLevel::Medium));
    assert!(policy.requires_confirmation(&RiskLevel::High));
    assert!(policy.requires_confirmation(&RiskLevel::Critical));
    assert!(!policy.requires_confirmation(&RiskLevel::Low));
}

#[test]
fn fuzz_runner_parse_output_detects_crash() {
    let output = "INFO: Running...\n10000 runs\nERROR: AddressSanitizer: heap-buffer-overflow\n  #0 0x123 in target.rs:5\n";
    let result = FuzzRunner::parse_output("my_target", output);
    assert_eq!(result.target, "my_target");
    assert!(!result.crashes.is_empty(), "must detect the ERROR line as a crash");
}

#[test]
fn fuzz_runner_parse_output_clean_run() {
    let output = "INFO: Running...\n50000 runs\nDone.\n";
    let result = FuzzRunner::parse_output("clean_target", output);
    assert!(result.crashes.is_empty(), "clean run must produce no crashes");
}

#[test]
fn audit_runner_parse_output_detects_vuln() {
    let output = "Crate:   openssl\nVersion: 0.10.0\nRUSTSEC-2021-0001: OpenSSL vulnerability\n";
    let result = AuditRunner::parse_output(output);
    assert!(!result.vulnerabilities.is_empty());
    assert!(!result.clean);
}

#[test]
fn audit_runner_parse_output_clean() {
    let output = "No vulnerabilities found\n";
    let result = AuditRunner::parse_output(output);
    assert!(result.clean);
}
