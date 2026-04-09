use crate::types::compute::{BudgetMode, CostEntry};
use crate::types::conversation::ConversationState;
use sha2::{Digest, Sha256};
use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use sysinfo::System;
use tokio::sync::{Semaphore, broadcast};

#[derive(Debug, Clone)]
pub struct ResourceEvent {
    pub rss_mb: u64,
    pub cpu_load: f32,
    pub disk_free_gb: u64,
    pub alert: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Urgency {
    RealTime,
    Interactive,
    Batch,
}

struct ResourceMonitorActor {
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
}

impl ResourceMonitorActor {
    fn spawn(tx: broadcast::Sender<ResourceEvent>, interval_secs: u64) -> Self {
        let handle = tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(Duration::from_secs(interval_secs));
            let mut sys = System::new_all();
            loop {
                ticker.tick().await;
                sys.refresh_all();
                let rss_mb = sys.used_memory() / 1024 / 1024;
                let cpu_load = sys.global_cpu_usage();
                let disks = sysinfo::Disks::new_with_refreshed_list();
                let disk_free_gb = disks
                    .iter()
                    .map(|d| d.available_space())
                    .sum::<u64>()
                    / 1024
                    / 1024
                    / 1024;
                let alert = if cpu_load > 90.0 {
                    Some("CPU Critical".to_string())
                } else if rss_mb > 16_000 {
                    Some("Memory Critical".to_string())
                } else {
                    None
                };
                let _ = tx.send(ResourceEvent { rss_mb, cpu_load, disk_free_gb, alert });
            }
        });
        Self { handle }
    }
}

pub struct ComputeManager {
    sys: System,
    resource_tx: broadcast::Sender<ResourceEvent>,
    pub cache: InferenceCache,
    pub rate_limits: RateLimitManager,
    monitor: Option<ResourceMonitorActor>,
}

impl ComputeManager {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            sys: System::new_all(),
            resource_tx: tx,
            cache: InferenceCache::new(),
            rate_limits: RateLimitManager::new(),
            monitor: None,
        }
    }

    pub fn start_background_monitor(&mut self, interval_secs: u64) {
        self.monitor = Some(ResourceMonitorActor::spawn(
            self.resource_tx.clone(),
            interval_secs,
        ));
    }

    pub fn manage_budget(sigma: &mut ConversationState, entry: CostEntry) -> BudgetMode {
        sigma.budget.spent += entry.cost_usd;
        sigma.budget.entries.push(entry);
        sigma.budget.mode()
    }

    pub fn monitor_resources(&mut self) -> ResourceEvent {
        self.sys.refresh_all();
        let rss_mb = self.sys.used_memory() / 1024 / 1024;
        let cpu_load = self.sys.global_cpu_usage();
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let disk_free_gb =
            disks.iter().map(|d| d.available_space()).sum::<u64>() / 1024 / 1024 / 1024;

        let alert = if cpu_load > 90.0 {
            Some("CPU Critical".to_string())
        } else if rss_mb > 16000 {
            Some("Memory Critical".to_string())
        } else {
            None
        };

        let event = ResourceEvent { rss_mb, cpu_load, disk_free_gb, alert };
        let _ = self.resource_tx.send(event.clone());
        event
    }

    pub fn resource_subscriber(&self) -> broadcast::Receiver<ResourceEvent> {
        self.resource_tx.subscribe()
    }
}

impl Default for ComputeManager {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ParallelInference;

impl ParallelInference {
    /// Runs inference across multiple models in parallel.
    /// Fails if any model fails.
    pub async fn run<F, Fut>(
        prompt: String,
        models: Vec<String>,
        mut f: F,
    ) -> Result<Vec<(String, String)>>
    where
        F: FnMut(String, String) -> Fut,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
    {
        let mut set: tokio::task::JoinSet<(String, Result<String>)> =
            tokio::task::JoinSet::new();

        for model in models {
            let fut = f(prompt.clone(), model.clone());
            set.spawn(async move { (model, fut.await) });
        }

        let mut results = Vec::new();
        while let Some(res) = set.join_next().await {
            match res {
                Ok((model, output)) => results.push((model, output?)),
                Err(e) => return Err(anyhow::anyhow!("Task join error: {:?}", e)),
            }
        }
        Ok(results)
    }

    /// Runs inference across multiple models in parallel with a timeout.
    /// Returns only the successes within the timeout.
    pub async fn run_robust<F, Fut>(
        prompt: String,
        models: Vec<String>,
        timeout: Duration,
        mut f: F,
    ) -> Vec<(String, String)>
    where
        F: FnMut(String, String) -> Fut,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
    {
        let mut set: tokio::task::JoinSet<(String, Result<String>)> =
            tokio::task::JoinSet::new();

        for model in models {
            let fut = f(prompt.clone(), model.clone());
            set.spawn(async move { (model, fut.await) });
        }

        let mut results = Vec::new();
        let timeout_fut = tokio::time::sleep(timeout);
        tokio::pin!(timeout_fut);

        loop {
            tokio::select! {
                res = set.join_next() => {
                    if let Some(join_res) = res {
                        if let Ok((model, Ok(output))) = join_res {
                            results.push((model, output));
                        }
                    } else {
                        break;
                    }
                }
                _ = &mut timeout_fut => {
                    break;
                }
            }
        }
        results
    }
}

pub struct FallbackChain {
    pub chain: Vec<(String, u32, f64)>, // (ModelId, MaxRetries, QualityFloor)
}

impl FallbackChain {
    pub fn next(&self, current_index: usize) -> Option<&(String, u32, f64)> {
        self.chain.get(current_index + 1)
    }
}

#[derive(Default)]
pub struct InferenceCache {
    pub entries: HashMap<String, (String, f64)>,
    pub hits: u64,
    pub misses: u64,
}

impl InferenceCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&mut self, prompt: &str, model_id: &str) -> Option<String> {
        let key = Self::cache_key(prompt, model_id);
        if let Some((res, _)) = self.entries.get(&key) {
            self.hits += 1;
            Some(res.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, prompt: &str, model_id: &str, response: String, quality: f64) {
        let key = Self::cache_key(prompt, model_id);
        self.entries.insert(key, (response, quality));
    }

    fn cache_key(prompt: &str, model_id: &str) -> String {
        let mut h = Sha256::new();
        h.update(prompt.as_bytes());
        h.update(b":");
        h.update(model_id.as_bytes());
        format!("{:x}", h.finalize())
    }

    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

#[derive(Default)]
pub struct RateLimitManager {
    pub backoffs: HashMap<String, u32>,
}

impl RateLimitManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_delay(&self, model_id: &str) -> Duration {
        let attempts = self.backoffs.get(model_id).copied().unwrap_or(0);
        if attempts == 0 {
            return Duration::ZERO;
        }
        let base_secs = 2u64.pow(attempts).min(60) as f64;
        let jitter: f64 = rand::rng().random_range(-0.25..=0.25);
        let secs = (base_secs * (1.0 + jitter)).max(0.1);
        Duration::from_secs_f64(secs)
    }

    pub fn report_success(&mut self, model_id: &str) {
        self.backoffs.remove(model_id);
    }

    pub fn report_429(&mut self, model_id: &str) {
        *self.backoffs.entry(model_id.to_string()).or_insert(0) += 1;
    }
}

pub struct LatencyRouter {
    pub thresholds_ms: HashMap<String, u64>,
}

impl LatencyRouter {
    pub fn new() -> Self {
        Self { thresholds_ms: HashMap::new() }
    }

    pub fn record(&mut self, model_id: &str, latency_ms: u64) {
        let entry = self.thresholds_ms.entry(model_id.to_string()).or_insert(0);
        *entry = (*entry).max(latency_ms);
    }

    #[must_use]
    pub fn filter<'a>(&self, candidates: &'a [String], urgency: Urgency) -> Vec<&'a String> {
        let limit_ms: u64 = match urgency {
            Urgency::RealTime => 500,
            Urgency::Interactive => 3_000,
            Urgency::Batch => u64::MAX,
        };
        candidates
            .iter()
            .filter(|id| self.thresholds_ms.get(*id).copied().unwrap_or(0) <= limit_ms)
            .collect()
    }

    /// Returns the single fastest model that satisfies the urgency constraint.
    #[must_use]
    pub fn select<'a>(&self, candidates: &'a [String], urgency: Urgency) -> Option<&'a String> {
        self.filter(candidates, urgency)
            .into_iter()
            .min_by_key(|id| self.thresholds_ms.get(*id).copied().unwrap_or(u64::MAX))
    }
}

impl Default for LatencyRouter {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BatchScheduler {
    semaphore: Arc<Semaphore>,
    pub max_concurrent: usize,
}

impl BatchScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("semaphore closed")
    }

    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}
