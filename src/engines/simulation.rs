use anyhow::{Context, Result};
use rand::{Rng, SeedableRng, rng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// =====================================================================
// CONSTANTS
// =====================================================================

/// Trials with std_dev above this are flagged as unreliable predictions.
pub const DIVERGENCE_THRESHOLD: f64 = 0.3;

// =====================================================================
// DOMAIN TYPES
// =====================================================================

#[derive(Debug, Clone)]
pub struct Artifact {
    /// 0.0 to 1.0 (e.g., 0.99 = highly stable, tested code)
    pub base_reliability: f64,
    /// 1.0+ multiplier (e.g., 1.5 means 50% larger state space)
    pub complexity_score: f64,
}

#[derive(Debug, Clone)]
pub struct ArtifactDiff {
    /// 0.0 to 1.0 (e.g., 0.8 = modifies critical infrastructure)
    pub mutation_volatility: f64,
    /// Number of discrete components altered
    pub structural_impact: usize,
}

// =====================================================================
// SIMULATION ENGINE TYPES
// =====================================================================

#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    pub max_trials: usize,
    pub batch_size: usize,
    pub target_error_margin: f64,
    pub base_seed: u64,
}

#[derive(Debug, Clone)]
pub struct SimulationTelemetry {
    pub trials_completed: usize,
    pub current_risk_factor: f64,
    pub standard_error: f64,
    pub converged: bool,
}

/// Full statistics from a variance-aware Monte Carlo run.
#[derive(Debug, Clone)]
pub struct PredictionStats {
    pub mean_p_fail: f64,
    pub std_dev: f64,
    /// 95% confidence interval (lower, upper), clamped to [0, 1].
    pub confidence_interval_95: (f64, f64),
    /// True when std_dev > DIVERGENCE_THRESHOLD — prediction is unreliable.
    pub divergence_warning: bool,
    pub trials: usize,
}

impl PredictionStats {
    fn compute(outcomes: &[f64]) -> Self {
        let n = outcomes.len();
        let mean_p_fail = outcomes.iter().sum::<f64>() / n as f64;
        // Population std dev of Bernoulli(p) = sqrt(p*(1-p)).
        // For finite samples the closed form equals the sample std dev
        // divided by a negligible correction factor at n >= 100.
        let std_dev = (mean_p_fail * (1.0 - mean_p_fail)).sqrt();
        let margin = 1.96 * std_dev / (n as f64).sqrt();
        let confidence_interval_95 = (
            (mean_p_fail - margin).max(0.0),
            (mean_p_fail + margin).min(1.0),
        );
        PredictionStats {
            mean_p_fail,
            std_dev,
            confidence_interval_95,
            divergence_warning: std_dev > DIVERGENCE_THRESHOLD,
            trials: n,
        }
    }
}

/// The environmental regime applied to each trial.
#[derive(Debug, Clone, Copy)]
enum TrialPhase {
    /// Single baseline trial (index 0): nominal fuel and memory.
    Baseline,
    /// Trials 1 ..= n/3: fuel jitter ±10% (simulates latency variance).
    FuelJitter,
    /// Trials n/3+1 ..= 2n/3: memory jitter ±15% (simulates contention).
    MemoryJitter,
    /// Trials 2n/3+1 ..= n: random startup delay 0–50 ms.
    StartupDelay,
}

fn trial_phase(i: usize, total: usize) -> TrialPhase {
    if i == 0 {
        TrialPhase::Baseline
    } else if i <= total / 3 {
        TrialPhase::FuelJitter
    } else if i <= 2 * total / 3 {
        TrialPhase::MemoryJitter
    } else {
        TrialPhase::StartupDelay
    }
}

// =====================================================================
// SANDBOX STUB (wired separately in production)
// =====================================================================

pub struct SandboxManager;
impl SandboxManager {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

// =====================================================================
// THE ENGINE
// =====================================================================

pub struct MonteCarloRunner {
    #[allow(dead_code)]
    sandbox: Arc<SandboxManager>,
}

impl MonteCarloRunner {
    pub fn new() -> Result<Self> {
        Ok(Self {
            sandbox: Arc::new(SandboxManager::new()?),
        })
    }

    /// Variance-aware prediction over `crate::types::artifact` types.
    /// Returns `(mean_p_fail, std_dev)`.
    pub async fn predict(
        &self,
        artifact: &crate::types::artifact::Artifact,
        diff: &crate::types::artifact::ArtifactDiff,
        trials: usize,
    ) -> Result<(f64, f64)> {
        let base_reliability = 0.85;
        let complexity_score =
            1.0 + (artifact.metrics.cyclomatic_complexity as f64 / 10.0).min(2.0);

        let sim_artifact = Artifact {
            base_reliability,
            complexity_score,
        };

        let structural_impact = diff.diff_text.len().max(1);
        let mutation_volatility = if artifact.content.is_empty() {
            0.1
        } else {
            (diff.diff_text.len() as f64 / artifact.content.len() as f64).min(1.0)
        };

        let sim_diff = ArtifactDiff {
            mutation_volatility,
            structural_impact,
        };

        let seed = rng().random::<u64>();
        let stats = tokio::task::spawn_blocking(move || {
            Self::run_variance_trials(&sim_artifact, &sim_diff, trials.max(1), seed)
        })
        .await
        .context("Monte Carlo thread panicked")?;

        Ok((stats.mean_p_fail, stats.std_dev))
    }

    /// Run `trials` variance-aware Monte Carlo trials and return full statistics.
    /// Pure, synchronous, and parallelised with Rayon — safe to call from
    /// `spawn_blocking` or tests.
    pub fn run_variance_trials(
        artifact: &Artifact,
        diff: &ArtifactDiff,
        trials: usize,
        seed: u64,
    ) -> PredictionStats {
        let base_rel = artifact.base_reliability;
        let complexity = artifact.complexity_score.max(1.0);
        let volatility = diff.mutation_volatility.clamp(0.0, 1.0);
        let impact = diff.structural_impact as f64;

        let outcomes: Vec<f64> = (0..trials)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));

                // --- A: Determine environmental regime ---
                let (fuel_factor, rel_jitter, delay_penalty) = match trial_phase(i, trials) {
                    TrialPhase::Baseline => (1.0_f64, 0.0_f64, 0.0_f64),
                    TrialPhase::FuelJitter => {
                        let jitter: f64 = rng.random_range(-0.10..=0.10);
                        (1.0 + jitter, 0.0, 0.0)
                    }
                    TrialPhase::MemoryJitter => {
                        let jitter: f64 = rng.random_range(-0.15..=0.15);
                        (1.0, jitter, 0.0)
                    }
                    TrialPhase::StartupDelay => {
                        let delay_ms: f64 = rng.random_range(0.0..50.0);
                        // 50 ms max delay maps to 1% reliability penalty.
                        (1.0, 0.0, delay_ms / 5000.0)
                    }
                };

                // --- B: Environmental noise (fuel headroom affects variance) ---
                let noise_bound = ((1000.0 * complexity) * fuel_factor.abs()) as i64;
                let noise_bound = noise_bound.max(1);
                let env_noise: i64 = rng.random_range(-noise_bound..=noise_bound);

                // --- C: Adjusted success probability ---
                let adjusted_rel =
                    (base_rel + rel_jitter - volatility * 0.15 - delay_penalty).clamp(0.01, 0.99);
                let success_roll = rng.random_bool(adjusted_rel);

                // --- D: Structural threshold check ---
                let critical_threshold = -800.0 + (impact * 15.0 * volatility);
                let success = env_noise as f64 >= critical_threshold && success_roll;

                // Return 1.0 for failure, 0.0 for success.
                if success { 0.0 } else { 1.0 }
            })
            .collect();

        PredictionStats::compute(&outcomes)
    }

    /// Returns true when the prediction std_dev exceeds the divergence threshold.
    pub fn is_divergent(std_dev: f64) -> bool {
        std_dev > DIVERGENCE_THRESHOLD
    }

    // ----------------------------------------------------------------
    // Legacy telemetry API (preserved for existing callers)
    // ----------------------------------------------------------------

    pub async fn predict_with_telemetry(
        &self,
        artifact: &Artifact,
        diff: &ArtifactDiff,
        config: MonteCarloConfig,
        cancel_token: CancellationToken,
        progress_tx: Option<mpsc::Sender<SimulationTelemetry>>,
    ) -> Result<SimulationTelemetry> {
        let base_rel = artifact.base_reliability;
        let complexity = artifact.complexity_score.max(1.0);
        let volatility = diff.mutation_volatility.clamp(0.0, 1.0);
        let impact = diff.structural_impact as f64;

        let result = tokio::task::spawn_blocking(move || {
            let mut total_successes = 0_usize;
            let mut trials_completed = 0_usize;
            let mut current_risk = 1.0;
            let mut standard_error = 1.0;

            let num_batches = (config
                .max_trials
                .saturating_add(config.batch_size)
                - 1)
                / config.batch_size;

            for batch_idx in 0..num_batches {
                if cancel_token.is_cancelled() {
                    break;
                }

                let start_idx = batch_idx * config.batch_size;
                let current_batch_size =
                    std::cmp::min(config.batch_size, config.max_trials - start_idx);

                let batch_successes: usize = (0..current_batch_size)
                    .into_par_iter()
                    .map(|i| {
                        let trial_id = (start_idx + i) as u64;
                        let mut rng =
                            SmallRng::seed_from_u64(config.base_seed.wrapping_add(trial_id));

                        let noise_bound = (1000.0 * complexity) as i64;
                        let env_noise: i64 = rng.random_range(-noise_bound..=noise_bound);
                        let critical_threshold = -800.0 + (impact * 15.0 * volatility);
                        let adjusted_probability =
                            (base_rel - (volatility * 0.15)).clamp(0.01, 0.99);
                        let success_roll = rng.random_bool(adjusted_probability);
                        usize::from(env_noise as f64 >= critical_threshold && success_roll)
                    })
                    .sum();

                total_successes += batch_successes;
                trials_completed += current_batch_size;

                let p = total_successes as f64 / trials_completed as f64;
                current_risk = 1.0 - p;
                standard_error = if trials_completed > 0 {
                    ((p * (1.0 - p)) / trials_completed as f64).sqrt()
                } else {
                    1.0
                };

                let is_converged =
                    standard_error <= config.target_error_margin && trials_completed >= 1000;

                if let Some(tx) = &progress_tx {
                    let _ = tx.try_send(SimulationTelemetry {
                        trials_completed,
                        current_risk_factor: current_risk,
                        standard_error,
                        converged: is_converged,
                    });
                }

                if is_converged {
                    break;
                }
            }

            SimulationTelemetry {
                trials_completed,
                current_risk_factor: current_risk,
                standard_error,
                converged: standard_error <= config.target_error_margin,
            }
        })
        .await
        .context("Monte Carlo thread pool panicked")?;

        Ok(result)
    }
}
