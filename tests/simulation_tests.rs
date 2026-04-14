use crosstalk::engines::simulation::{
    Artifact, ArtifactDiff, DIVERGENCE_THRESHOLD, MonteCarloRunner,
};

// Helpers for common artifact profiles.

fn known_good() -> (Artifact, ArtifactDiff) {
    (
        Artifact {
            base_reliability: 0.99,
            complexity_score: 1.0,
        },
        ArtifactDiff {
            mutation_volatility: 0.0,
            structural_impact: 0,
        },
    )
}

fn known_bad() -> (Artifact, ArtifactDiff) {
    (
        Artifact {
            base_reliability: 0.01,
            complexity_score: 1.0,
        },
        ArtifactDiff {
            mutation_volatility: 0.99,
            structural_impact: 100,
        },
    )
}

// base_rel=0.60, vol=0.20, impact=1 gives p_fail ~0.49 (derived analytically).
fn flaky_50_50() -> (Artifact, ArtifactDiff) {
    (
        Artifact {
            base_reliability: 0.60,
            complexity_score: 1.0,
        },
        ArtifactDiff {
            mutation_volatility: 0.20,
            structural_impact: 1,
        },
    )
}

// ----------------------------------------------------------------
// 1. Known-good: mean p_fail is low.
// ----------------------------------------------------------------
#[test]
fn known_good_module_low_pfail() {
    let (art, diff) = known_good();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 200, 1);
    assert!(
        stats.mean_p_fail < 0.25,
        "expected low p_fail for stable artifact, got {:.3}",
        stats.mean_p_fail
    );
}

// ----------------------------------------------------------------
// 2. Known-bad: mean p_fail is high and prediction is stable.
// ----------------------------------------------------------------
#[test]
fn known_bad_module_high_pfail() {
    let (art, diff) = known_bad();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 200, 2);
    assert!(
        stats.mean_p_fail > 0.80,
        "expected high p_fail for broken artifact, got {:.3}",
        stats.mean_p_fail
    );
    // std_dev for p near 1.0 is small; prediction should be confident.
    assert!(
        stats.std_dev < DIVERGENCE_THRESHOLD,
        "stable bad prediction should not trigger divergence warning, std_dev={:.3}",
        stats.std_dev
    );
}

// ----------------------------------------------------------------
// 3. 50/50 flaky: mean converges near 0.5 over 100 trials.
// ----------------------------------------------------------------
#[test]
fn flaky_module_mean_near_half() {
    let (art, diff) = flaky_50_50();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 100, 3);
    assert!(
        (0.30..=0.70).contains(&stats.mean_p_fail),
        "50/50 module should have mean_p_fail near 0.5, got {:.3}",
        stats.mean_p_fail
    );
}

// ----------------------------------------------------------------
// 4. Divergence warning triggers for a flaky module.
// ----------------------------------------------------------------
#[test]
fn divergence_warning_triggered_for_flaky_module() {
    let (art, diff) = flaky_50_50();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 100, 4);
    assert!(
        stats.divergence_warning,
        "divergence_warning should be true for 50/50 module (std_dev={:.3})",
        stats.std_dev
    );
    assert!(stats.std_dev > DIVERGENCE_THRESHOLD);
}

// ----------------------------------------------------------------
// 5. Divergence warning absent for a clearly bad module.
// ----------------------------------------------------------------
#[test]
fn divergence_warning_absent_for_stable_prediction() {
    let (art, diff) = known_bad();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 200, 5);
    assert!(
        !stats.divergence_warning,
        "stable (always-fail) prediction must not raise divergence warning"
    );
}

// ----------------------------------------------------------------
// 6. 95% confidence interval contains the true mean.
// ----------------------------------------------------------------
#[test]
fn confidence_interval_contains_mean() {
    let (art, diff) = flaky_50_50();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 500, 6);
    let (lo, hi) = stats.confidence_interval_95;
    assert!(
        lo <= stats.mean_p_fail && stats.mean_p_fail <= hi,
        "mean {:.3} must lie within CI [{:.3}, {:.3}]",
        stats.mean_p_fail,
        lo,
        hi
    );
    assert!(lo >= 0.0 && hi <= 1.0, "CI must be clamped to [0,1]");
}

// ----------------------------------------------------------------
// 7. Determinism: same seed produces the same result.
// ----------------------------------------------------------------
#[test]
fn deterministic_with_same_seed() {
    let (art, diff) = flaky_50_50();
    let a = MonteCarloRunner::run_variance_trials(&art, &diff, 100, 99);
    let b = MonteCarloRunner::run_variance_trials(&art, &diff, 100, 99);
    assert_eq!(a.mean_p_fail, b.mean_p_fail);
    assert_eq!(a.std_dev, b.std_dev);
    assert_eq!(a.trials, 100);
}

// ----------------------------------------------------------------
// 8. Trial count is exact.
// ----------------------------------------------------------------
#[test]
fn trial_count_matches_requested() {
    let (art, diff) = known_good();
    for &n in &[1usize, 10, 50, 100, 333] {
        let stats = MonteCarloRunner::run_variance_trials(&art, &diff, n, 7);
        assert_eq!(stats.trials, n, "trials should equal requested count");
    }
}

// ----------------------------------------------------------------
// 9. Runtime for 100 trials is reasonable (< 2 s on any machine).
// ----------------------------------------------------------------
#[test]
fn performance_100_trials_under_2s() {
    let (art, diff) = flaky_50_50();
    let start = std::time::Instant::now();
    let _ = MonteCarloRunner::run_variance_trials(&art, &diff, 100, 8);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "100 trials took {:.2}s, must be < 2s",
        elapsed.as_secs_f64()
    );
}

// ----------------------------------------------------------------
// 10. is_divergent helper mirrors the divergence_warning field.
// ----------------------------------------------------------------
#[test]
fn is_divergent_helper_consistent_with_stats() {
    let (art, diff) = flaky_50_50();
    let stats = MonteCarloRunner::run_variance_trials(&art, &diff, 100, 9);
    assert_eq!(
        MonteCarloRunner::is_divergent(stats.std_dev),
        stats.divergence_warning
    );
}
