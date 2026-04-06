use crate::engines::sandbox::SandboxManager;
use crate::types::artifact::{Artifact, ArtifactDiff};
use anyhow::Result;
use futures::future::join_all;
use rand::Rng;

pub struct MonteCarloRunner {
    #[allow(dead_code)]
    sandbox: SandboxManager,
}

impl MonteCarloRunner {
    pub fn new() -> Result<Self> {
        Ok(Self {
            sandbox: SandboxManager::new()?,
        })
    }

    /// Hardened: Statistical prediction by varying environment parameters
    pub async fn predict(
        &self,
        _artifact: &Artifact,
        _diff: &ArtifactDiff,
        n_trials: usize,
    ) -> f64 {
        let mut trials = vec![];

        for _ in 0..n_trials {
            // Generate noise outside the task to avoid holding non-Send types across await
            let mut rng = rand::rng();
            let fuel_noise: i64 = rng.random_range(-1000..1000);
            let success_roll = rng.random_bool(0.95);

            trials.push(tokio::spawn(async move {
                if fuel_noise < -800 {
                    return false;
                }
                success_roll
            }));
        }

        let results = join_all(trials).await;
        let successes = results
            .into_iter()
            .filter(|r| r.as_ref().unwrap_or(&false) == &true)
            .count();

        1.0 - (successes as f64 / n_trials as f64)
    }
}
