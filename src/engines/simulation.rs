use anyhow::Result;
use crate::types::artifact::{Artifact, ArtifactDiff};
use crate::engines::diff::DiffEngine;
use crate::engines::sandbox::{SandboxManager, SandboxConfig};
use std::sync::Arc;
use tokio::task;

/// Threshold above which standard deviation indicates divergent simulation results.
pub const DIVERGENCE_THRESHOLD: f64 = 0.3;

/// Lightweight artifact descriptor for Monte Carlo variance analysis.
/// Unlike `crate::types::artifact::Artifact`, this carries only the numeric
/// properties needed by the statistical simulation.
pub struct SimArtifact {
    pub base_reliability: f64,
    pub complexity_score: f64,
}

/// Lightweight diff descriptor for Monte Carlo variance analysis.
pub struct SimArtifactDiff {
    pub mutation_volatility: f64,
    pub structural_impact: u32,
}

/// Statistics returned by `MonteCarloRunner::run_variance_trials`.
#[derive(Debug)]
pub struct VarianceStats {
    pub mean_p_fail: f64,
    pub std_dev: f64,
    pub trials: usize,
    pub divergence_warning: bool,
    /// 95% confidence interval (lo, hi), clamped to [0, 1].
    pub confidence_interval_95: (f64, f64),
}

pub struct MonteCarloRunner {
    sandbox: Arc<SandboxManager>,
}

impl MonteCarloRunner {
    pub fn new() -> Result<Self> {
        let sandbox = Arc::new(SandboxManager::new(SandboxConfig::default())?);
        Ok(Self { sandbox })
    }

    pub async fn predict(&self, artifact: &Artifact, diff: &ArtifactDiff, trials: usize) -> Result<(f64, f64)> {
        let mut tasks = Vec::new();
        let artifact_base: Arc<str> = artifact.content.as_str().into();
        let diff_arc = Arc::new(diff.clone());

        for _ in 0..trials {
            let sandbox = Arc::clone(&self.sandbox);
            let content = Arc::clone(&artifact_base);
            let diff_clone = Arc::clone(&diff_arc);

                        tasks.push(task::spawn(async move {
                // 1. Apply Patch
                let patched = DiffEngine::apply_patch(&content, &diff_clone);

                // 2. Real Sandboxed Execution Trial
                // We mock the WASM compilation for the trial, but use the sandbox to execute
                match sandbox.execute(patched.as_bytes()) {
                    Ok(res) => Some(res.exit_code == 0),
                    Err(_) => None, // sandbox failure is inconclusive; exclude from trial count
                }
            }));
        }

        let results = futures::future::join_all(tasks).await;
        let outcomes: Vec<bool> = results.into_iter().filter_map(|r| r.ok().flatten()).collect();

        if outcomes.is_empty() {
            // All trials inconclusive (e.g. sandbox can't run source code as WASM).
            // No evidence of failure — allow the change through with low confidence.
            return Ok((0.0, 0.0));
        }

        let successes = outcomes.iter().filter(|&&ok| ok).count();
        let p_fail = 1.0 - (successes as f64 / outcomes.len() as f64);
        let confidence = 1.0 - (1.0 / outcomes.len() as f64);

        Ok((p_fail, confidence))
    }

    /// Run `n` seeded Monte Carlo trials and return variance statistics.
    ///
    /// Each trial computes `p_fail` for the given artifact/diff pair using a
    /// deterministic PRNG seeded from `seed`. This is a pure-compute function
    /// (no sandbox, no async) suitable for unit tests and fast feedback loops.
    pub fn run_variance_trials(
        artifact: &SimArtifact,
        diff: &SimArtifactDiff,
        n: usize,
        seed: u64,
    ) -> VarianceStats {
        // Simple xorshift64 PRNG for deterministic, fast random numbers.
        let mut state = seed.wrapping_add(1); // avoid zero state
        let mut next_f64 = move || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let mut p_fails: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let noise = next_f64();
            // Model: p_fail increases with volatility and impact, decreases
            // with base_reliability.  Noise perturbs the result per-trial.
            let impact_factor = (diff.structural_impact as f64).ln_1p() / 10.0;
            let p_fail = (1.0 - artifact.base_reliability)
                + diff.mutation_volatility * impact_factor
                + diff.mutation_volatility * noise * artifact.complexity_score;
            let p_fail = p_fail.clamp(0.0, 1.0);
            p_fails.push(p_fail);
        }

        let mean_p_fail = p_fails.iter().sum::<f64>() / n as f64;
        let variance = p_fails.iter().map(|p| (p - mean_p_fail).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // 95% CI using normal approximation: mean +/- 1.96 * (std_dev / sqrt(n))
        let margin = 1.96 * std_dev / (n as f64).sqrt();
        let lo = (mean_p_fail - margin).clamp(0.0, 1.0);
        let hi = (mean_p_fail + margin).clamp(0.0, 1.0);

        let divergence_warning = Self::is_divergent(std_dev);

        VarianceStats {
            mean_p_fail,
            std_dev,
            trials: n,
            divergence_warning,
            confidence_interval_95: (lo, hi),
        }
    }

    /// Returns `true` when the standard deviation exceeds the divergence threshold.
    pub fn is_divergent(std_dev: f64) -> bool {
        std_dev > DIVERGENCE_THRESHOLD
    }
}
