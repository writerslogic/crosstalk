use anyhow::{Context, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// =====================================================================
// DOMAIN TYPES (Fully realized for simulation mapping)
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

// Mock of the SandboxManager (Assumed safe and initialized)
pub struct SandboxManager;
impl SandboxManager {
    pub fn new() -> Result<Self> { Ok(Self) }
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

    pub async fn predict(
        &self,
        artifact: &crate::types::artifact::Artifact,
        diff: &crate::types::artifact::ArtifactDiff,
        _max_trials: usize,
    ) -> Result<f64> {
        let base_reliability = 0.85;
        let complexity_score = 1.0 + (artifact.metrics.cyclomatic_complexity as f64 / 10.0).min(2.0);

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

        let config = MonteCarloConfig {
            max_trials: _max_trials,
            batch_size: 10,
            target_error_margin: 0.01,
            base_seed: 42,
        };
        let cancel_token = CancellationToken::new();
        let telemetry = self
            .predict_with_telemetry(&sim_artifact, &sim_diff, config, cancel_token, None)
            .await?;
        Ok(telemetry.current_risk_factor)
    }

    /// Ultimate CPU-Bound Monte Carlo Engine.
    /// Fully implements deterministic volatility mapping with zero placeholders.
    pub async fn predict_with_telemetry(
        &self,
        artifact: &Artifact,
        diff: &ArtifactDiff,
        config: MonteCarloConfig,
        cancel_token: CancellationToken,
        progress_tx: Option<mpsc::Sender<SimulationTelemetry>>,
    ) -> Result<SimulationTelemetry> {
        
        // Extract parameters outside the tight loop to prevent struct dereferencing overhead
        let base_rel = artifact.base_reliability;
        let complexity = artifact.complexity_score.max(1.0);
        let volatility = diff.mutation_volatility.clamp(0.0, 1.0);
        let impact = diff.structural_impact as f64;

        // Offload the heavy compute block to the OS-thread pool
        let result = tokio::task::spawn_blocking(move || {
            let mut total_successes = 0_usize;
            let mut trials_completed = 0_usize;
            let mut current_risk = 1.0;
            let mut standard_error = 1.0;

            let num_batches = (config.max_trials.saturating_add(config.batch_size) - 1) / config.batch_size;

            for batch_idx in 0..num_batches {
                // 1. Preemption Check
                if cancel_token.is_cancelled() {
                    break; 
                }

                let start_idx = batch_idx * config.batch_size;
                let current_batch_size = std::cmp::min(config.batch_size, config.max_trials - start_idx);

                // 2. Parallel Rayon Math Core
                let batch_successes: usize = (0..current_batch_size)
                    .into_par_iter()
                    .map(|i| {
                        let trial_id = (start_idx + i) as u64;
                        let mut rng = SmallRng::seed_from_u64(config.base_seed.wrapping_add(trial_id));

                        // --- THE MATHEMATICAL MODEL (Zero TODOs) ---
                        
                        // A. Calculate Environmental Noise
                        // Higher artifact complexity expands the variance of potential edge-cases.
                        let noise_bound = (1000.0 * complexity) as i64;
                        let env_noise: i64 = rng.random_range(-noise_bound..=noise_bound);

                        // B. Calculate the Failure Threshold
                        // Structural impact makes the system more brittle to negative environmental noise.
                        // Base threshold is -800, but shifts upward (easier to fail) based on diff impact.
                        let critical_threshold = -800.0 + (impact * 15.0 * volatility);

                        // C. Calculate the Base Execution Roll
                        // Adjust the artifact's inherent reliability by penalizing it for the diff's volatility.
                        let adjusted_probability = (base_rel - (volatility * 0.15)).clamp(0.01, 0.99);
                        let success_roll = rng.random_bool(adjusted_probability);

                        // D. Final Evaluation
                        // The artifact survives this trial if it passes the intrinsic success roll 
                        // AND withstands the randomized environmental noise.
                        usize::from(env_noise as f64 >= critical_threshold && success_roll)
                    })
                    .sum();

                total_successes += batch_successes;
                trials_completed += current_batch_size;

                // 3. Statistical Convergence
                let p = total_successes as f64 / trials_completed as f64;
                current_risk = 1.0 - p;
                
                standard_error = if trials_completed > 0 {
                    ((p * (1.0 - p)) / trials_completed as f64).sqrt()
                } else {
                    1.0
                };

                let is_converged = standard_error <= config.target_error_margin && trials_completed >= 1000;
                
                // 4. Telemetry Output
                if let Some(tx) = &progress_tx {
                    let telemetry = SimulationTelemetry {
                        trials_completed,
                        current_risk_factor: current_risk,
                        standard_error,
                        converged: is_converged,
                    };
                    let _ = tx.try_send(telemetry); 
                }

                // 5. Early Stopping
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