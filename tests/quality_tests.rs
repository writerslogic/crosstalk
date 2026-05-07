use crosstalk::engines::quality::{
    ArtifactMetrics, BlockerType, CoherenceChecker, CompletionScorer, DeadCodeDetector,
    DeadItemKind, DocChecker, IncoherenceKind, QualityTrend, QualityTrendAnalyzer,
    RegressionDetector, TournamentProposal, TournamentRunner, TrendClassification,
};
use crosstalk::types::artifact::{Artifact, ArtifactDiff, ProofAttachment};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

fn artifact_with(name: &str, content: &str) -> Artifact {
    Artifact {
        name: name.to_string(),
        content: content.to_string(),
        language: "rust".to_string(),
        version: 1,
        history: vec![ArtifactDiff {
            original_version: 0,
            new_version: 1,
            diff_text: content.to_string(),
        }],
        ast_versions: BTreeMap::new(),
        proof_attachments: vec![ProofAttachment {
            artifact_name: name.to_string(),
            proven_properties: vec!["test".to_string()],
            proof_hash: "hash".to_string(),
            verified_at: 0,
        }],
        metrics: ArtifactMetrics {
            cyclomatic_complexity: 1,
            coupling_factor: 0,
            comment_density: 0.1,
            line_count: content.lines().count() as u32,
            health_score: 0.9,
            visual_fidelity: 0.0,
        },
        skeleton: String::new(),
    }
}

// ── RegressionDetector ────────────────────────────────────────────────────────

#[test]
fn regression_detector_rejects_complexity_spike() {
    let old = ArtifactMetrics {
        cyclomatic_complexity: 2,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        cyclomatic_complexity: 8,
        ..Default::default()
    };
    assert!(RegressionDetector::is_regressive(&old, &new));
}

#[test]
fn regression_detector_rejects_comment_drop() {
    let old = ArtifactMetrics {
        comment_density: 0.3,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        comment_density: 0.1,
        ..Default::default()
    };
    assert!(RegressionDetector::is_regressive(&old, &new));
}

#[test]
fn regression_detector_accepts_stable_metrics() {
    let old = ArtifactMetrics {
        cyclomatic_complexity: 5,
        comment_density: 0.2,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        cyclomatic_complexity: 6,
        comment_density: 0.2,
        ..Default::default()
    };
    assert!(!RegressionDetector::is_regressive(&old, &new));
}

// ── QualityTrendAnalyzer ──────────────────────────────────────────────────────

#[test]
fn trend_analyzer_classifies_improving() {
    let trend = QualityTrend {
        artifact_id: "a1".to_string(),
        history: vec![
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 1,
                score: 0.5,
            },
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 2,
                score: 0.6,
            },
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 3,
                score: 0.7,
            },
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 4,
                score: 0.8,
            },
        ],
    };
    assert_eq!(
        QualityTrendAnalyzer::classify(&trend),
        TrendClassification::Improving
    );
}

#[test]
fn trend_analyzer_classifies_degrading() {
    let trend = QualityTrend {
        artifact_id: "a1".to_string(),
        history: vec![
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 1,
                score: 0.8,
            },
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 2,
                score: 0.7,
            },
            crosstalk::engines::quality::QualityTrendEntry {
                turn_id: 3,
                score: 0.6,
            },
        ],
    };
    assert_eq!(
        QualityTrendAnalyzer::classify(&trend),
        TrendClassification::Degrading
    );
}

// ── CoherenceChecker ──────────────────────────────────────────────────────────

#[test]
fn coherence_empty_artifacts_returns_no_reports() {
    assert!(CoherenceChecker::verify(&HashMap::new()).is_empty());
}

#[test]
fn coherence_stale_mod_declaration_flagged() {
    let mut arts = HashMap::new();
    arts.insert(
        "main.rs".to_string(),
        Arc::new(artifact_with("main.rs", "mod nonexistent_module;\n")),
    );
    let reports = CoherenceChecker::verify(&arts);
    assert!(
        reports
            .iter()
            .any(|r| r.symbol == "nonexistent_module" && r.kind == IncoherenceKind::StaleImport),
        "stale mod declaration must be flagged"
    );
}

#[test]
fn coherence_known_mod_not_flagged() {
    let mut arts = HashMap::new();
    arts.insert(
        "lib.rs".to_string(),
        Arc::new(artifact_with("lib.rs", "mod helper;\n")),
    );
    arts.insert(
        "helper.rs".to_string(),
        Arc::new(artifact_with("helper.rs", "pub fn foo() {}\n")),
    );
    let reports = CoherenceChecker::verify(&arts);
    assert!(
        !reports.iter().any(|r| r.symbol == "helper"),
        "mod declaration for an existing artifact must not be flagged"
    );
}

#[test]
fn coherence_undefined_symbol_in_use_flagged() {
    let mut arts = HashMap::new();
    arts.insert(
        "consumer.rs".to_string(),
        Arc::new(artifact_with(
            "consumer.rs",
            "use some_crate::GhostStruct;\n",
        )),
    );
    let reports = CoherenceChecker::verify(&arts);
    assert!(
        reports
            .iter()
            .any(|r| r.symbol == "GhostStruct" && r.kind == IncoherenceKind::UndefinedSymbol),
        "undefined PascalCase import must be flagged"
    );
}

// ── CompletionScorer ──────────────────────────────────────────────────────────

#[test]
fn completion_score_bounded() {
    let r = CompletionScorer::evaluate(0.5, 0.5, true);
    assert!((0.0..=1.0).contains(&r.score));
}

#[test]
fn completion_scorer_detects_stall() {
    let mut history = vec![];
    for _ in 0..5 {
        history.push(CompletionScorer::evaluate(0.6, 0.4, false));
    }
    let blocker = CompletionScorer::diagnose_blocker(&history).expect("stall should be detected");
    assert_eq!(blocker.blocker_type, BlockerType::FailingTest);
}

// ── TournamentRunner ──────────────────────────────────────────────────────────

#[test]
fn tournament_runner_picks_highest_score() {
    let proposals = vec![
        TournamentProposal {
            agent_id: "A".to_string(),
            quality_score: 0.9,
            compiled: true,
            tests_passed: false,
        },
        TournamentProposal {
            agent_id: "B".to_string(),
            quality_score: 0.7,
            compiled: true,
            tests_passed: true,
        },
    ];
    let result = TournamentRunner::run(&proposals).unwrap();
    assert_eq!(
        result.winner_agent_id, "B",
        "agent B wins due to passing tests"
    );
}

// ── DocChecker ────────────────────────────────────────────────────────────────

#[test]
fn doc_checker_detects_return_mismatch() {
    let content = "/// This function returns a Result.\npub fn foo() -> Option<u32> { None }";
    let issues = DocChecker::verify(content);
    assert!(
        !issues.is_empty(),
        "mismatch between doc 'Result' and sig 'Option' should be flagged"
    );
}

#[test]
fn doc_checker_detects_param_mismatch() {
    let content = "/// Takes missing_param:\npub fn foo(actual: u32) {}";
    let issues = DocChecker::verify(content);
    assert!(!issues.is_empty());
}

// ── DeadCodeDetector ──────────────────────────────────────────────────────────

#[test]
fn dead_code_detector_flags_unused_private_fn() {
    let content = "fn dead() {}\nfn main() {}";
    let report = DeadCodeDetector::scan("main.rs", content);
    assert!(
        report
            .dead_items
            .iter()
            .any(|i| i.name == "dead" && i.kind == DeadItemKind::UnusedFunction)
    );
}

#[test]
fn dead_code_detector_ignores_used_fn() {
    let content = "fn used() {}\nfn main() { used(); }";
    let report = DeadCodeDetector::scan("main.rs", content);
    assert!(report.dead_items.is_empty());
}

#[test]
fn dead_code_detector_flags_unused_import() {
    let content = "use std::collections::HashMap;\nfn main() {}";
    let report = DeadCodeDetector::scan("main.rs", content);
    assert!(
        report
            .dead_items
            .iter()
            .any(|i| i.name == "HashMap" && i.kind == DeadItemKind::UnusedImport)
    );
}

// ── RegressionDetector edge cases ────────────────────────────────────────────

#[test]
fn regression_detector_zero_old_complexity_no_regression() {
    let old = ArtifactMetrics {
        cyclomatic_complexity: 0,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        cyclomatic_complexity: 5,
        ..Default::default()
    };
    // When old complexity is 0, percentage increase is treated as 0.0
    assert!(!RegressionDetector::is_regressive(&old, &new));
}

#[test]
fn regression_detector_identical_metrics_not_regressive() {
    let m = ArtifactMetrics {
        cyclomatic_complexity: 10,
        coupling_factor: 3,
        comment_density: 0.25,
        line_count: 200,
        health_score: 0.85,
        visual_fidelity: 0.0,
    };
    assert!(!RegressionDetector::is_regressive(&m, &m));
}

#[test]
fn regression_detector_zero_to_zero_not_regressive() {
    let m = ArtifactMetrics::default();
    assert!(!RegressionDetector::is_regressive(&m, &m));
}

#[test]
fn regression_detector_health_score_drop_triggers() {
    let old = ArtifactMetrics {
        health_score: 0.9,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        health_score: 0.8,
        ..Default::default()
    };
    // 0.1 drop exceeds 0.05 threshold
    assert!(RegressionDetector::is_regressive(&old, &new));
}

#[test]
fn regression_detector_health_score_small_drop_ok() {
    let old = ArtifactMetrics {
        health_score: 0.9,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        health_score: 0.86,
        ..Default::default()
    };
    // 0.04 drop is below 0.05 threshold
    assert!(!RegressionDetector::is_regressive(&old, &new));
}

#[test]
fn regression_detector_comment_density_boundary() {
    let old = ArtifactMetrics {
        comment_density: 0.30,
        ..Default::default()
    };
    let just_below = ArtifactMetrics {
        comment_density: 0.16,
        ..Default::default()
    };
    let just_above = ArtifactMetrics {
        comment_density: 0.14,
        ..Default::default()
    };
    // 0.14 drop is < 0.15 threshold
    assert!(!RegressionDetector::is_regressive(&old, &just_below));
    // 0.16 drop exceeds 0.15 threshold
    assert!(RegressionDetector::is_regressive(&old, &just_above));
}

#[test]
fn regression_detector_complexity_boundary_20_percent() {
    let old = ArtifactMetrics {
        cyclomatic_complexity: 10,
        ..Default::default()
    };
    let at_20 = ArtifactMetrics {
        cyclomatic_complexity: 12,
        ..Default::default()
    };
    let over_20 = ArtifactMetrics {
        cyclomatic_complexity: 13,
        ..Default::default()
    };
    // 20% increase is at boundary (not strictly >)
    assert!(!RegressionDetector::is_regressive(&old, &at_20));
    // 30% increase exceeds 20% threshold
    assert!(RegressionDetector::is_regressive(&old, &over_20));
}

#[test]
fn regression_detector_improvement_not_regressive() {
    let old = ArtifactMetrics {
        cyclomatic_complexity: 20,
        comment_density: 0.1,
        health_score: 0.5,
        ..Default::default()
    };
    let new = ArtifactMetrics {
        cyclomatic_complexity: 10,
        comment_density: 0.3,
        health_score: 0.9,
        ..Default::default()
    };
    assert!(!RegressionDetector::is_regressive(&old, &new));
}
