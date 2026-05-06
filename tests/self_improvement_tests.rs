use crosstalk::engines::self_improvement::{
    AbTestManager, CalibrationAdjuster, PostMortemGenerator, ProgressReporter,
    PromptEvolutionaryOptimizer, RuntimeParameterAdjuster, SafetyInterlock, SelfCodeModifier,
    SelfEvaluationTrendAnalyzer, SelfImprovementEngine,
};
use crosstalk::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use crosstalk::types::intelligence::PromptTemplate;
use crosstalk::types::self_improvement::{CalibrationRecord, SessionEvaluation};
use std::collections::BTreeMap;

fn make_sigma(session_id: &str) -> ConversationState {
    ConversationState::new(session_id)
}

#[test]
fn evaluate_session_computes_basic_metrics() {
    let mut sigma = make_sigma("s1");
    sigma.completion_probability = 0.8;
    sigma.budget.spent = 0.5;
    sigma.turns.push(Turn {
        index: 0,
        model_id: "m1".to_string(),
        content: "c".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.9),
        outcome: TurnOutcome::Compiled,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    });

    let eval = SelfImprovementEngine::evaluate_session(&sigma);
    assert_eq!(eval.session_id, "s1");
    assert!(*eval.metrics.get("convergence_p").unwrap() > 0.7);
}

#[test]
fn trend_analyzer_detects_improving_metric() {
    let evals = vec![
        SessionEvaluation {
            session_id: "s1".to_string(),
            metrics: BTreeMap::from([("m1".to_string(), 0.5)]),
            timestamp: 0,
        },
        SessionEvaluation {
            session_id: "s2".to_string(),
            metrics: BTreeMap::from([("m1".to_string(), 0.7)]),
            timestamp: 1,
        },
    ];
    let report = SelfEvaluationTrendAnalyzer::analyze(&evals);
    assert!(report.improving.contains(&"m1".to_string()));
}

#[test]
fn trend_analyzer_detects_degrading_metric() {
    let evals = vec![
        SessionEvaluation {
            session_id: "s1".to_string(),
            metrics: BTreeMap::from([("m1".to_string(), 0.8)]),
            timestamp: 0,
        },
        SessionEvaluation {
            session_id: "s2".to_string(),
            metrics: BTreeMap::from([("m1".to_string(), 0.5)]),
            timestamp: 1,
        },
    ];
    let report = SelfEvaluationTrendAnalyzer::analyze(&evals);
    assert!(report.degrading.contains(&"m1".to_string()));
}

#[test]
fn ab_test_significance_requires_enough_data() {
    let control = vec![0.5; 5];
    let test = vec![0.8; 5];
    // Needs n >= 10
    assert!(!AbTestManager::check_significance(&control, &test));
}

#[test]
fn ab_test_significance_detected_with_enough_data() {
    let control = vec![0.5; 15];
    let test = vec![0.8; 15];
    assert!(AbTestManager::check_significance(&control, &test));
}

#[test]
fn prompt_optimizer_mutates_correctly() {
    let parent = PromptTemplate {
        id: "p1".to_string(),
        version: 1,
        template_text: "Hello".to_string(),
        task_category: TaskCategory::General,
        variables: vec![],
        tags: vec!["t1".to_string()],
        performance_history: vec![],
    };
    let variants = PromptEvolutionaryOptimizer::generate_variants(&parent);
    assert_eq!(variants.len(), 4);
    assert!(variants[0].id.contains("-m2"));
}

#[test]
fn platt_scaling_fit_returns_identity_on_empty() {
    let (a, b) = CalibrationAdjuster::fit_platt(&[]);
    assert_eq!(a, 1.0);
    assert_eq!(b, 0.0);
}

#[test]
fn platt_scaling_fit_computes_slopes() {
    let records = vec![
        CalibrationRecord {
            session_id: "s1".to_string(),
            predicted_difficulty: 0.5,
            actual_difficulty: 0.5,
            predicted_outcome: 0.2,
            actual_outcome: 0.4,
        },
        CalibrationRecord {
            session_id: "s2".to_string(),
            predicted_difficulty: 0.5,
            actual_difficulty: 0.5,
            predicted_outcome: 0.8,
            actual_outcome: 0.9,
        },
    ];
    let (a, b) = CalibrationAdjuster::fit_platt(&records);
    assert!(a > 0.0);
    let calibrated = CalibrationAdjuster::apply(0.5, a, b);
    assert!(calibrated > 0.5);
}

#[test]
fn post_mortem_generated_on_high_failure_rate() {
    let mut sigma = make_sigma("s1");
    for i in 0..10 {
        sigma.turns.push(Turn {
            index: i,
            model_id: "m".to_string(),
            content: "fail".to_string(),
            timestamp: 0,
            diffs: vec![],
            certainty: Some(0.1),
            outcome: TurnOutcome::Rejected,
            task_category: None,
            structure: None,
            signature: vec![],
            surprise_signal: None,
        });
    }
    let pm = PostMortemGenerator::generate(&sigma).expect("mortem should be generated");
    assert_eq!(pm.session_id, "s1");
    assert!(pm.failure_turn_indices.len() > 5);
}

#[test]
fn parameter_adjuster_applies_significant_changes() {
    let mut adj = RuntimeParameterAdjuster::new();
    let report = crosstalk::engines::self_improvement::AbTestReport {
        hypothesis_id: "h1".to_string(),
        control_mean: 0.5,
        test_mean: 0.8,
        effect_size: 0.6,
        significant: true,
        adopted: true,
        confidence_interval: (0.7, 0.9),
    };
    let changed = adj.apply_if_significant("min_quality_score", 0.7, &report, "better quality");
    assert!(changed);
    assert_eq!(adj.get("min_quality_score"), Some(0.7));
}

#[test]
fn safety_interlock_protects_core_files() {
    assert!(!SafetyInterlock::is_modification_allowed(
        "src/core/orchestrator.rs"
    ));
    assert!(SafetyInterlock::is_modification_allowed("src/ui/app.rs"));
}

#[test]
fn code_modifier_proposes_improvements() {
    let content = "let _ = opt.unwrap_or_else(|_| panic!());";
    let improved = SelfCodeModifier::propose_improvement("src/lib.rs", content).unwrap();
    assert!(improved.contains(".unwrap()"));
}

#[test]
fn progress_reporter_estimates_remaining_turns() {
    let mut sigma = make_sigma("s1");
    sigma.completion_probability = 0.4; // 40% after 2 turns
    sigma.turns.push(Turn {
        index: 0,
        model_id: "m".to_string(),
        content: "c".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.5),
        outcome: TurnOutcome::Compiled,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    });
    sigma.turns.push(Turn {
        index: 1,
        model_id: "m".to_string(),
        content: "c".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.5),
        outcome: TurnOutcome::Compiled,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    });

    let report = ProgressReporter::report(&sigma, 10);
    assert!(report.estimated_turns_remaining.is_some());
    // Exponential decay: -ln(1-0.4)/2 = 0.255. -ln(0.001)/0.255 = 27. 27-2 = 25.
    assert!(report.estimated_turns_remaining.unwrap() > 10);
}
