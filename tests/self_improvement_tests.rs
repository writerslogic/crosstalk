use crosstalk::engines::self_improvement::{
    AbTestManager, BenchmarkRegressionGuard, BenchmarkSuite, CalibrationAdjuster,
    CalibrationTracker, ContinuousLearner, DegradationHandler, ErrorBudgetLedger,
    EscalationContextBuilder, HypothesisGenerator, HypothesisPrioritizer, LearningEffectivenessMonitor,
    PerformanceTrendReport, PostMortemGenerator, PostMortemLearner, PromptEvolutionaryOptimizer,
    PromptLibrary, ProgressReporter, RuntimeParameterAdjuster, SafetyInterlock, SelfCodeModifier,
    SelfEvaluationTrendAnalyzer, SelfImprovementEngine, StrategyDatabase,
};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use crosstalk::types::self_improvement::{
    BenchmarkCategory, BenchmarkResult, CalibrationRecord, DegradationResponse,
    DegradationTrigger, EnforcementLevel, FailureCause, HypothesisStatus, ImprovementHypothesis,
    PostMortem, PromptTemplate, SessionEvaluation, StrategyEntry,
};
use std::collections::HashMap;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_eval(session_id: &str, metrics: &[(&str, f64)]) -> SessionEvaluation {
    SessionEvaluation {
        session_id: session_id.to_string(),
        metrics: metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        timestamp: 0,
    }
}

fn make_hypothesis(id: &str, impact: f64, confidence: f64, cost: f64) -> ImprovementHypothesis {
    ImprovementHypothesis {
        id: id.to_string(),
        description: format!("hypothesis {id}"),
        expected_impact: impact,
        confidence,
        estimated_cost: cost,
        status: HypothesisStatus::Queued,
    }
}

fn minimal_state(session_id: &str) -> ConversationState {
    ConversationState::new(session_id)
}

fn state_with_turns(session_id: &str, outcomes: &[TurnOutcome]) -> ConversationState {
    let mut sigma = ConversationState::new(session_id);
    for (i, &outcome) in outcomes.iter().enumerate() {
        sigma.turns.push(Turn {
            index: i as u32,
            model_id: "m".to_string(),
            content: String::new(),
            timestamp: 0,
            diffs: vec![],
            certainty: None,
            outcome,
            task_category: None,
            structure: None,
            signature: vec![],
            surprise_signal: None,
        });
    }
    sigma
}

// ── SelfImprovementEngine ─────────────────────────────────────────────────────

#[test]
fn evaluate_session_contains_required_metrics() {
    let sigma = minimal_state("s1");
    let eval = SelfImprovementEngine::evaluate_session(&sigma);
    assert!(eval.metrics.contains_key("turn_count"));
    assert!(eval.metrics.contains_key("convergence_p"));
    assert!(eval.metrics.contains_key("cost_spent"));
    assert!(eval.metrics.contains_key("failure_rate"));
    assert!(eval.metrics.contains_key("cost_efficiency"));
}

#[test]
fn evaluate_session_failure_rate_reflects_rejected_turns() {
    let sigma = state_with_turns(
        "s2",
        &[TurnOutcome::TestsPassed, TurnOutcome::Rejected, TurnOutcome::Rejected],
    );
    let eval = SelfImprovementEngine::evaluate_session(&sigma);
    let rate = eval.metrics["failure_rate"];
    assert!((rate - 2.0 / 3.0).abs() < 0.01, "expected ~0.67, got {rate}");
}

#[test]
fn evaluate_session_zero_failure_rate_when_all_pass() {
    let sigma = state_with_turns("s3", &[TurnOutcome::TestsPassed, TurnOutcome::Compiled]);
    let eval = SelfImprovementEngine::evaluate_session(&sigma);
    assert_eq!(eval.metrics["failure_rate"], 0.0);
}

// ── SelfEvaluationTrendAnalyzer ───────────────────────────────────────────────

#[test]
fn trend_analyzer_empty_returns_empty_report() {
    let report = SelfEvaluationTrendAnalyzer::analyze(&[]);
    assert!(report.ema.is_empty());
    assert!(report.improving.is_empty());
    assert!(report.degrading.is_empty());
}

#[test]
fn trend_analyzer_improving_metric_detected() {
    let evals = vec![
        make_eval("a", &[("quality", 0.1)]),
        make_eval("b", &[("quality", 0.5)]),
        make_eval("c", &[("quality", 0.9)]),
    ];
    let report = SelfEvaluationTrendAnalyzer::analyze(&evals);
    assert!(report.improving.contains(&"quality".to_string()));
}

#[test]
fn trend_analyzer_degrading_metric_detected() {
    let evals = vec![
        make_eval("a", &[("error_rate", 0.9)]),
        make_eval("b", &[("error_rate", 0.5)]),
        make_eval("c", &[("error_rate", 0.1)]),
    ];
    let report = SelfEvaluationTrendAnalyzer::analyze(&evals);
    assert!(report.degrading.contains(&"error_rate".to_string()));
}

#[test]
fn trend_analyzer_stable_metric_not_in_improving_or_degrading() {
    let evals = vec![
        make_eval("a", &[("cost", 1.0)]),
        make_eval("b", &[("cost", 1.0)]),
        make_eval("c", &[("cost", 1.0)]),
    ];
    let report = SelfEvaluationTrendAnalyzer::analyze(&evals);
    assert!(!report.improving.contains(&"cost".to_string()));
    assert!(!report.degrading.contains(&"cost".to_string()));
}

// ── HypothesisGenerator ───────────────────────────────────────────────────────

#[test]
fn hypothesis_generator_produces_one_per_degrading_metric() {
    let report = PerformanceTrendReport {
        ema: HashMap::new(),
        improving: vec![],
        degrading: vec!["alpha".to_string(), "beta".to_string()],
        stable: vec![],
    };
    let hyps = HypothesisGenerator::from_trend(&report);
    assert_eq!(hyps.len(), 2);
    assert!(hyps.iter().all(|h| h.status == HypothesisStatus::Queued));
}

#[test]
fn hypothesis_generator_empty_when_no_degradation() {
    let report = PerformanceTrendReport {
        ema: HashMap::new(),
        improving: vec!["quality".to_string()],
        degrading: vec![],
        stable: vec![],
    };
    assert!(HypothesisGenerator::from_trend(&report).is_empty());
}

// ── HypothesisPrioritizer ─────────────────────────────────────────────────────

#[test]
fn prioritizer_ranks_by_impact_times_confidence_over_cost() {
    let hyps = vec![
        make_hypothesis("low", 0.1, 0.5, 1.0),  // priority 0.05
        make_hypothesis("high", 0.9, 0.9, 1.0), // priority 0.81
        make_hypothesis("mid", 0.5, 0.5, 1.0),  // priority 0.25
    ];
    let ranked = HypothesisPrioritizer::rank(hyps);
    assert_eq!(ranked[0].id, "high");
    assert_eq!(ranked[2].id, "low");
}

#[test]
fn prioritizer_high_cost_lowers_rank() {
    let hyps = vec![
        make_hypothesis("cheap", 0.5, 0.8, 0.5), // priority 0.8
        make_hypothesis("expensive", 0.5, 0.8, 4.0), // priority 0.1
    ];
    let ranked = HypothesisPrioritizer::rank(hyps);
    assert_eq!(ranked[0].id, "cheap");
}

// ── AbTestManager ─────────────────────────────────────────────────────────────

#[test]
fn ab_test_significance_requires_minimum_samples() {
    let control: Vec<f64> = vec![1.0; 5];
    let test: Vec<f64> = vec![2.0; 5];
    assert!(!AbTestManager::check_significance(&control, &test));
}

#[test]
fn ab_test_detects_significant_improvement() {
    // Use linearly-spaced values so variance is non-zero.
    let control: Vec<f64> = (0..50).map(|i| 1.0 + i as f64 * 0.02).collect();
    let test: Vec<f64> = (0..50).map(|i| 2.0 + i as f64 * 0.02).collect();
    assert!(AbTestManager::check_significance(&control, &test));
}

#[test]
fn ab_test_rejects_negligible_effect() {
    let control: Vec<f64> = (0..50).map(|i| 1.0 + i as f64 * 0.02).collect();
    let test: Vec<f64> = (0..50).map(|i| 1.001 + i as f64 * 0.02).collect();
    assert!(!AbTestManager::check_significance(&control, &test));
}

#[test]
fn ab_test_report_adopted_when_significant_and_positive() {
    let control: Vec<f64> = (0..50).map(|i| 1.0 + i as f64 * 0.02).collect();
    let test: Vec<f64> = (0..50).map(|i| 2.0 + i as f64 * 0.02).collect();
    let report = AbTestManager::evaluate("h1", &control, &test);
    assert!(report.adopted);
    assert!(report.effect_size > 0.0);
}

#[test]
fn ab_test_report_not_adopted_when_test_worse() {
    let control: Vec<f64> = (0..50).map(|i| 2.0 + i as f64 * 0.02).collect();
    let test: Vec<f64> = (0..50).map(|i| 1.0 + i as f64 * 0.02).collect();
    let report = AbTestManager::evaluate("h2", &control, &test);
    assert!(!report.adopted);
}

#[test]
fn ab_test_confidence_interval_contains_mean() {
    let test: Vec<f64> = (0..30).map(|i| 1.0 + i as f64 * 0.01).collect();
    let report = AbTestManager::evaluate("h3", &[1.0; 30], &test);
    assert!(report.confidence_interval.0 <= report.test_mean);
    assert!(report.confidence_interval.1 >= report.test_mean);
}

// ── PromptLibrary ─────────────────────────────────────────────────────────────

fn make_template(id: &str, task_type: &str) -> PromptTemplate {
    PromptTemplate {
        id: id.to_string(),
        version: 1,
        content: format!("template for {id}"),
        task_types: vec![task_type.to_string()],
        performance_history: vec![],
    }
}

#[test]
fn prompt_library_insert_and_retrieve() {
    let mut lib = PromptLibrary::new();
    lib.insert(make_template("t1", "bug_fix"));
    assert!(lib.get("t1").is_some());
    assert!(lib.get("missing").is_none());
}

#[test]
fn prompt_library_best_for_task_returns_highest_mean_quality() {
    let mut lib = PromptLibrary::new();
    let mut low = make_template("low", "refactor");
    low.performance_history = vec![("s1".to_string(), 0.3)];
    let mut high = make_template("high", "refactor");
    high.performance_history = vec![("s2".to_string(), 0.9)];
    lib.insert(low);
    lib.insert(high);
    assert_eq!(lib.best_for_task("refactor").unwrap().id, "high");
}

#[test]
fn prompt_library_record_performance_updates_history() {
    let mut lib = PromptLibrary::new();
    lib.insert(make_template("t1", "gen"));
    lib.record_performance("t1", "sess", 0.75);
    assert_eq!(lib.get("t1").unwrap().performance_history.len(), 1);
}

// ── StrategyDatabase ──────────────────────────────────────────────────────────

fn make_strategy(id: &str, features: Vec<f64>) -> StrategyEntry {
    StrategyEntry {
        id: id.to_string(),
        task_features: features,
        approach: "approach".to_string(),
        steps: vec![],
        outcome_quality: 0.8,
        sessions_used: 1,
    }
}

#[test]
fn strategy_database_knn_returns_closest() {
    let mut db = StrategyDatabase::new();
    db.insert(make_strategy("far", vec![0.0, 0.0]));
    db.insert(make_strategy("near", vec![1.0, 1.0]));
    let results = db.knn(&[0.9, 0.9], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "near");
}

#[test]
fn strategy_database_knn_caps_at_k() {
    let mut db = StrategyDatabase::new();
    for i in 0..5 {
        db.insert(make_strategy(&i.to_string(), vec![i as f64]));
    }
    assert_eq!(db.knn(&[2.5], 2).len(), 2);
}

#[test]
fn strategy_database_empty_returns_empty() {
    let db = StrategyDatabase::new();
    assert!(db.knn(&[1.0], 3).is_empty());
}

// ── PostMortemGenerator ───────────────────────────────────────────────────────

#[test]
fn post_mortem_none_when_failure_rate_low() {
    let sigma = state_with_turns("s", &[TurnOutcome::TestsPassed; 10]);
    assert!(PostMortemGenerator::generate(&sigma).is_none());
}

#[test]
fn post_mortem_generated_when_failure_rate_high() {
    let mut outcomes = vec![TurnOutcome::TestsPassed; 5];
    outcomes.extend(vec![TurnOutcome::Rejected; 5]);
    let sigma = state_with_turns("s", &outcomes);
    assert!(PostMortemGenerator::generate(&sigma).is_some());
}

#[test]
fn post_mortem_detects_type_mismatch_cause() {
    let mut sigma = ConversationState::new("s");
    for i in 0..10 {
        let outcome = if i < 2 { TurnOutcome::Rejected } else { TurnOutcome::TestsPassed };
        let content = if i == 0 { "mismatched types".to_string() } else { String::new() };
        sigma.turns.push(Turn {
            index: i,
            model_id: "m".to_string(),
            content,
            timestamp: 0,
            diffs: vec![],
            certainty: None,
            outcome,
            task_category: None,
            structure: None,
            signature: vec![],
            surprise_signal: None,
        });
    }
    let pm = PostMortemGenerator::generate(&sigma).unwrap();
    assert_eq!(pm.root_cause, FailureCause::TypeMismatch);
}

// ── CalibrationTracker ────────────────────────────────────────────────────────

#[test]
fn calibration_ece_zero_when_no_records() {
    let tracker = CalibrationTracker::new();
    assert_eq!(tracker.ece(), 0.0);
}

#[test]
fn calibration_ece_nonzero_with_prediction_error() {
    let mut tracker = CalibrationTracker::new();
    tracker.record(CalibrationRecord {
        session_id: "s".to_string(),
        predicted_difficulty: 0.9,
        actual_difficulty: 0.1,
        predicted_outcome: 0.9,
        actual_outcome: 0.1,
    });
    assert!(tracker.ece() > 0.0);
}

#[test]
fn calibration_needs_recalibration_when_ece_above_threshold() {
    let mut tracker = CalibrationTracker::new();
    tracker.record(CalibrationRecord {
        session_id: "s".to_string(),
        predicted_difficulty: 1.0,
        actual_difficulty: 0.0,
        predicted_outcome: 1.0,
        actual_outcome: 0.0,
    });
    assert!(tracker.needs_recalibration());
}

#[test]
fn calibration_no_recalibration_needed_when_accurate() {
    let mut tracker = CalibrationTracker::new();
    tracker.record(CalibrationRecord {
        session_id: "s".to_string(),
        predicted_difficulty: 0.5,
        actual_difficulty: 0.5,
        predicted_outcome: 0.8,
        actual_outcome: 0.79,
    });
    assert!(!tracker.needs_recalibration());
}

// ── ErrorBudgetLedger ─────────────────────────────────────────────────────────

#[test]
fn error_budget_get_returns_none_before_set() {
    let ledger = ErrorBudgetLedger::new();
    assert!(ledger.get("unknown").is_none());
}

#[test]
fn error_budget_enforcement_strict_when_budget_low() {
    let mut ledger = ErrorBudgetLedger::new();
    ledger.set_budget("task", 0.05);
    for _ in 0..30 {
        ledger.record_outcome("task", true);
    }
    let budget = ledger.get("task").unwrap();
    assert_eq!(budget.enforcement_level, EnforcementLevel::Strict);
}

#[test]
fn error_budget_normal_when_few_failures() {
    let mut ledger = ErrorBudgetLedger::new();
    ledger.set_budget("task", 0.5);
    for _ in 0..5 {
        ledger.record_outcome("task", false);
    }
    let budget = ledger.get("task").unwrap();
    assert_eq!(budget.enforcement_level, EnforcementLevel::Normal);
}

// ── BenchmarkSuite ────────────────────────────────────────────────────────────

#[test]
fn benchmark_suite_has_twenty_standard_tasks() {
    let suite = BenchmarkSuite::with_standard_tasks();
    assert_eq!(suite.tasks().len(), 20);
}

#[test]
fn benchmark_suite_covers_all_categories() {
    let suite = BenchmarkSuite::with_standard_tasks();
    assert!(!suite.by_category(BenchmarkCategory::CodeGeneration).is_empty());
    assert!(!suite.by_category(BenchmarkCategory::BugFixing).is_empty());
    assert!(!suite.by_category(BenchmarkCategory::Refactoring).is_empty());
    assert!(!suite.by_category(BenchmarkCategory::ArchitectureDesign).is_empty());
    assert!(!suite.by_category(BenchmarkCategory::ResearchSynthesis).is_empty());
}

#[test]
fn benchmark_suite_all_tasks_have_valid_difficulty() {
    let suite = BenchmarkSuite::with_standard_tasks();
    for task in suite.tasks() {
        assert!((0.0..=1.0).contains(&task.difficulty), "difficulty out of range: {}", task.difficulty);
    }
}

// ── DegradationHandler ────────────────────────────────────────────────────────

#[test]
fn degradation_budget_exhausted_checkpoints() {
    let strategies = DegradationHandler::default_strategies();
    let resp = DegradationHandler::resolve(DegradationTrigger::BudgetExhausted, &strategies);
    assert_eq!(resp, DegradationResponse::Checkpoint);
}

#[test]
fn degradation_complexity_exceeded_tries_simpler_subgoal() {
    let strategies = DegradationHandler::default_strategies();
    let resp = DegradationHandler::resolve(DegradationTrigger::TaskComplexityExceeded, &strategies);
    assert_eq!(resp, DegradationResponse::AttemptSimplerSubGoal);
}

#[test]
fn degradation_all_models_failing_suggests_human() {
    let strategies = DegradationHandler::default_strategies();
    let resp = DegradationHandler::resolve(DegradationTrigger::AllModelsFailing, &strategies);
    assert_eq!(resp, DegradationResponse::SuggestHumanIntervention);
}

#[test]
fn degradation_unknown_trigger_falls_back_to_document() {
    let resp = DegradationHandler::resolve(DegradationTrigger::ConvergenceImpossible, &[]);
    assert_eq!(resp, DegradationResponse::DocumentBlocker);
}

// ── SafetyInterlock ───────────────────────────────────────────────────────────

#[test]
fn safety_interlock_blocks_protected_files() {
    assert!(!SafetyInterlock::is_modification_allowed("src/security.rs"));
    assert!(!SafetyInterlock::is_modification_allowed("src/verification.rs"));
    assert!(!SafetyInterlock::is_modification_allowed("src/self_improvement.rs"));
    assert!(!SafetyInterlock::is_modification_allowed("src/orchestrator.rs"));
}

#[test]
fn safety_interlock_allows_unprotected_files() {
    assert!(SafetyInterlock::is_modification_allowed("src/planning.rs"));
    assert!(SafetyInterlock::is_modification_allowed("src/consensus.rs"));
}

#[test]
fn safety_interlock_rejects_unparseable_path() {
    assert!(!SafetyInterlock::is_modification_allowed(""));
}

// ── SelfCodeModifier ──────────────────────────────────────────────────────────

#[test]
fn self_code_modifier_rejects_protected_file() {
    assert!(SelfCodeModifier::propose_improvement("src/security.rs", "fn foo() {}").is_err());
}

#[test]
fn self_code_modifier_applies_known_pattern() {
    let content = r#"let x = foo().unwrap_or_else(|_| panic!());"#;
    let result = SelfCodeModifier::propose_improvement("src/utils.rs", content).unwrap();
    assert!(result.contains(".unwrap()"));
    assert!(!result.contains("unwrap_or_else(|_| panic!())"));
}

#[test]
fn self_code_modifier_errors_when_no_pattern_matches() {
    let content = "fn clean_code() -> i32 { 42 }";
    assert!(SelfCodeModifier::propose_improvement("src/utils.rs", content).is_err());
}

// ── ContinuousLearner ─────────────────────────────────────────────────────────

#[test]
fn continuous_learner_records_calibration_entry() {
    let mut lib = PromptLibrary::new();
    let mut cal = CalibrationTracker::new();
    let mut ledger = ErrorBudgetLedger::new();
    ledger.set_budget("general", 0.2);

    let sigma = minimal_state("sess");
    let mut learner = ContinuousLearner {
        prompt_library: &mut lib,
        calibration: &mut cal,
        error_ledger: &mut ledger,
    };
    learner.run(&sigma, None, 0.5, 0.4);
    assert_eq!(cal.records.len(), 1);
}

#[test]
fn continuous_learner_records_prompt_performance_when_template_given() {
    let mut lib = PromptLibrary::new();
    lib.insert(make_template("t1", "gen"));
    let mut cal = CalibrationTracker::new();
    let mut ledger = ErrorBudgetLedger::new();
    ledger.set_budget("general", 0.2);

    let sigma = minimal_state("sess");
    let mut learner = ContinuousLearner {
        prompt_library: &mut lib,
        calibration: &mut cal,
        error_ledger: &mut ledger,
    };
    learner.run(&sigma, Some("t1"), 0.5, 0.5);
    assert_eq!(lib.get("t1").unwrap().performance_history.len(), 1);
}

// ── PromptEvolutionaryOptimizer ───────────────────────────────────────────────

#[test]
fn prompt_mutation_append_adds_content() {
    use crosstalk::engines::self_improvement::PromptMutation;
    let t = make_template("base", "code");
    let m = PromptEvolutionaryOptimizer::mutate(&t, PromptMutation::AppendEmphasis);
    assert!(m.content.contains("template for base"));
    assert!(m.content.contains("Be precise"));
    assert_eq!(m.version, 2);
}

#[test]
fn prompt_mutation_prepend_role() {
    use crosstalk::engines::self_improvement::PromptMutation;
    let t = make_template("base", "code");
    let m = PromptEvolutionaryOptimizer::mutate(&t, PromptMutation::PrependRole);
    assert!(m.content.starts_with("You are an expert"));
    assert!(m.content.contains("template for base"));
}

#[test]
fn prompt_mutation_trim_shortens_content() {
    use crosstalk::engines::self_improvement::PromptMutation;
    let t = PromptTemplate {
        id: "long".to_string(),
        version: 1,
        content: "a".repeat(100),
        task_types: vec!["code".to_string()],
        performance_history: vec![],
    };
    let m = PromptEvolutionaryOptimizer::mutate(&t, PromptMutation::TrimVerbose);
    assert!(m.content.len() < 100);
}

#[test]
fn generate_variants_produces_one_per_mutation() {
    let t = make_template("parent", "code");
    let variants = PromptEvolutionaryOptimizer::generate_variants(&t);
    assert!(!variants.is_empty());
    for v in &variants {
        assert!(v.version > t.version);
    }
}

#[test]
fn select_winner_returns_highest_mean_quality() {
    let mut a = make_template("a", "code");
    a.performance_history = vec![("s1".into(), 0.9), ("s2".into(), 0.8)];
    let mut b = make_template("b", "code");
    b.performance_history = vec![("s1".into(), 0.5)];
    let candidates = vec![a, b];
    let winner = PromptEvolutionaryOptimizer::select_winner(&candidates).unwrap();
    assert_eq!(winner.id, "a");
}

// ── CalibrationAdjuster ───────────────────────────────────────────────────────

#[test]
fn platt_identity_when_perfect_calibration() {
    let records: Vec<CalibrationRecord> = (0..10)
        .map(|i| CalibrationRecord {
            session_id: i.to_string(),
            predicted_difficulty: i as f64 * 0.1,
            actual_difficulty: i as f64 * 0.1,
            predicted_outcome: i as f64 * 0.1,
            actual_outcome: i as f64 * 0.1,
        })
        .collect();
    let (a, b) = CalibrationAdjuster::fit_platt(&records);
    // Perfect calibration: a ≈ 1, b ≈ 0
    assert!((a - 1.0).abs() < 0.1, "a={a}");
    assert!(b.abs() < 0.1, "b={b}");
}

#[test]
fn platt_apply_clamps_to_unit_interval() {
    let v = CalibrationAdjuster::apply(0.9, 2.0, 0.1);
    assert!(v <= 1.0);
    let v2 = CalibrationAdjuster::apply(0.0, -1.0, -0.5);
    assert!(v2 >= 0.0);
}

#[test]
fn platt_insufficient_data_returns_identity() {
    let (a, b) = CalibrationAdjuster::fit_platt(&[]);
    assert_eq!((a, b), (1.0, 0.0));
}

// ── BenchmarkRegressionGuard ──────────────────────────────────────────────────

#[test]
fn regression_guard_no_regression_passes() {
    let mut guard = BenchmarkRegressionGuard::new();
    guard.set_baseline("task-1", 0.8);
    let results = vec![BenchmarkResult { task_id: "task-1".into(), score: 0.82, timestamp: 0 }];
    assert!(guard.check(&results).is_ok());
}

#[test]
fn regression_guard_detects_drop() {
    let mut guard = BenchmarkRegressionGuard::new();
    guard.set_baseline("task-1", 0.8);
    let results = vec![BenchmarkResult { task_id: "task-1".into(), score: 0.7, timestamp: 0 }];
    let err = guard.check(&results).unwrap_err();
    assert!(err.to_string().contains("regression"));
}

#[test]
fn regression_guard_ignores_unknown_tasks() {
    let guard = BenchmarkRegressionGuard::new();
    let results = vec![BenchmarkResult { task_id: "unknown".into(), score: 0.0, timestamp: 0 }];
    assert!(guard.check(&results).is_ok());
}

// ── PostMortemLearner ─────────────────────────────────────────────────────────

#[test]
fn postmortem_learner_inserts_into_library() {
    let mortem = PostMortem {
        session_id: "s1".into(),
        failure_turn_indices: vec![0, 1],
        root_cause: FailureCause::TypeMismatch,
        missing_context: vec![],
        alternative_approaches: vec![],
    };
    let mut lib = PromptLibrary::new();
    let mut db = StrategyDatabase::new();
    PostMortemLearner::apply(&mortem, &mut lib, &mut db);
    assert!(lib.get("postmortem-s1").is_some());
}

#[test]
fn postmortem_learner_inserts_strategy() {
    let mortem = PostMortem {
        session_id: "s2".into(),
        failure_turn_indices: vec![2],
        root_cause: FailureCause::MissingContext,
        missing_context: vec![],
        alternative_approaches: vec![],
    };
    let mut lib = PromptLibrary::new();
    let mut db = StrategyDatabase::new();
    PostMortemLearner::apply(&mortem, &mut lib, &mut db);
    assert!(!db.knn(&[1.0], 5).is_empty());
}

// ── RuntimeParameterAdjuster ──────────────────────────────────────────────────

#[test]
fn parameter_adjuster_defaults_populated() {
    let adj = RuntimeParameterAdjuster::new();
    assert!(adj.get("min_quality_score").is_some());
    assert!(adj.get("convergence_threshold").is_some());
}

#[test]
fn parameter_adjuster_applies_on_significant_report() {
    let mut adj = RuntimeParameterAdjuster::new();
    let control: Vec<f64> = vec![0.5; 20];
    let test_vals: Vec<f64> = vec![0.8; 20];
    let report = AbTestManager::evaluate("h1", &control, &test_vals);
    let applied = adj.apply_if_significant("min_quality_score", 0.6, &report, "test");
    assert!(applied);
    assert!((adj.get("min_quality_score").unwrap() - 0.6).abs() < f64::EPSILON);
    assert_eq!(adj.history.len(), 1);
}

#[test]
fn parameter_adjuster_skips_non_significant() {
    let mut adj = RuntimeParameterAdjuster::new();
    let report = AbTestManager::evaluate("h2", &[0.5; 5], &[0.5; 5]);
    let applied = adj.apply_if_significant("min_quality_score", 0.9, &report, "test");
    assert!(!applied);
}

// ── ProgressReporter ──────────────────────────────────────────────────────────

#[test]
fn progress_report_zero_turns() {
    let sigma = ConversationState::new("p1");
    let report = ProgressReporter::report(&sigma, 20);
    assert_eq!(report.turns_completed, 0);
    assert_eq!(report.turns_expected, 20);
}

#[test]
fn progress_report_computes_eta() {
    let mut sigma = ConversationState::new("p2");
    sigma.completion_probability = 0.4;
    for i in 0u32..4 {
        sigma.turns.push(Turn {
            index: i, model_id: "m".into(), content: "x".into(),
            timestamp: 0, diffs: vec![], certainty: None,
            outcome: TurnOutcome::Compiled, task_category: None,
            structure: None, signature: vec![], surprise_signal: None,
        });
    }
    let report = ProgressReporter::report(&sigma, 10);
    assert_eq!(report.turns_completed, 4);
    assert!(report.estimated_turns_remaining.is_some());
}

// ── LearningEffectivenessMonitor ──────────────────────────────────────────────

#[test]
fn learning_monitor_mean_delta_empty() {
    let m = LearningEffectivenessMonitor::new();
    assert_eq!(m.mean_delta(), 0.0);
}

#[test]
fn learning_monitor_tracks_improvements() {
    let mut m = LearningEffectivenessMonitor::new();
    m.record("prompt-update", "quality", 0.5, 0.7);
    m.record("calibration", "ece", 0.15, 0.08);
    assert!((m.mean_delta() - 0.065).abs() < 1e-9);
    assert_eq!(m.effective_actions().len(), 1);
}

#[test]
fn learning_monitor_excludes_regressions_from_effective() {
    let mut m = LearningEffectivenessMonitor::new();
    m.record("bad-change", "quality", 0.8, 0.5);
    assert_eq!(m.effective_actions().len(), 0);
}

// ── EscalationContextBuilder ──────────────────────────────────────────────────

#[test]
fn escalation_builder_summary_contains_failure_count() {
    let mut sigma = ConversationState::new("esc-1");
    sigma.turns.push(Turn {
        index: 0, model_id: "m".into(), content: "x".into(),
        timestamp: 0, diffs: vec![], certainty: None,
        outcome: TurnOutcome::Rejected, task_category: None,
        structure: None, signature: vec![], surprise_signal: None,
    });
    let pkg = EscalationContextBuilder::build(&sigma, "all-models-failing", vec!["h1".into()]);
    assert!(pkg.failure_summary.contains("1 of 1"));
    assert_eq!(pkg.hypotheses_tried, vec!["h1"]);
}

#[test]
fn escalation_builder_identifies_last_successful_turn() {
    let mut sigma = ConversationState::new("esc-2");
    sigma.turns.push(Turn {
        index: 0, model_id: "m".into(), content: "ok".into(),
        timestamp: 0, diffs: vec![], certainty: None,
        outcome: TurnOutcome::TestsPassed, task_category: None,
        structure: None, signature: vec![], surprise_signal: None,
    });
    sigma.turns.push(Turn {
        index: 1, model_id: "m".into(), content: "fail".into(),
        timestamp: 1, diffs: vec![], certainty: None,
        outcome: TurnOutcome::Rejected, task_category: None,
        structure: None, signature: vec![], surprise_signal: None,
    });
    let pkg = EscalationContextBuilder::build(&sigma, "trigger", vec![]);
    assert_eq!(pkg.last_successful_turn, Some(0));
}

#[test]
fn escalation_builder_no_successful_turn_is_none() {
    let mut sigma = ConversationState::new("esc-3");
    sigma.turns.push(Turn {
        index: 0, model_id: "m".into(), content: "fail".into(),
        timestamp: 0, diffs: vec![], certainty: None,
        outcome: TurnOutcome::Rejected, task_category: None,
        structure: None, signature: vec![], surprise_signal: None,
    });
    let pkg = EscalationContextBuilder::build(&sigma, "t", vec![]);
    assert!(pkg.last_successful_turn.is_none());
}

// ── SelfCodeModifier::verify ──────────────────────────────────────────────────

#[test]
fn verify_protected_file_returns_error() {
    let err = SelfCodeModifier::verify("security.rs", "old", "new").unwrap_err();
    assert!(err.to_string().contains("protected"));
}

#[test]
fn verify_empty_proposed_returns_error() {
    let err = SelfCodeModifier::verify("lib.rs", "old", "").unwrap_err();
    assert!(err.to_string().contains("empty"));
}

#[test]
fn verify_valid_change_returns_delta() {
    let delta = SelfCodeModifier::verify("lib.rs", "abc", "abcde").unwrap();
    assert_eq!(delta, 2);
}
