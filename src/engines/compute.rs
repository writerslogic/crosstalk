use crate::types::compute::CostEntry;
use crate::types::conversation::ConversationState;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use sysinfo::System;
use tokio::sync::broadcast;

#[derive(Debug, Clone)]
pub struct ResourceEvent {
    pub rss_mb: u64,
    pub cpu_load: f32,
    pub disk_free_gb: u64,
    pub alert: Option<String>,
}

pub struct ComputeManager {
    sys: System,
    resource_tx: broadcast::Sender<ResourceEvent>,
    pub cache: InferenceCache,
    pub rate_limits: RateLimitManager,
}

impl ComputeManager {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            sys: System::new_all(),
            resource_tx: tx,
            cache: InferenceCache::new(),
            rate_limits: RateLimitManager::new(),
        }
    }

    pub fn manage_budget(sigma: &mut ConversationState, entry: CostEntry) {
        sigma.budget.spent += entry.cost_usd;
        sigma.budget.entries.push(entry);

        let remaining = sigma.budget.session_budget - sigma.budget.spent;
        if remaining < sigma.budget.session_budget * 0.2 {
            println!("[compute] Warning: Budget < 20% remaining. Cost-reduction mode active.");
        }
    }

    pub fn monitor_resources(&mut self) -> ResourceEvent {
        self.sys.refresh_all();
        let rss_mb = self.sys.used_memory() / 1024 / 1024;
        let cpu_load = self.sys.global_cpu_usage();
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let disk_free_gb =
            disks.iter().map(|d| d.available_space()).sum::<u64>() / 1024 / 1024 / 1024;

        let mut alert = None;
        if cpu_load > 90.0 {
            alert = Some("CPU Critical".to_string());
        }
        if rss_mb > 16000 {
            alert = Some("Memory Critical".to_string());
        }

        let event = ResourceEvent {
            rss_mb,
            cpu_load,
            disk_free_gb,
            alert,
        };
        let _ = self.resource_tx.send(event.clone());
        event
    }

    pub fn resource_subscriber(&self) -> broadcast::Receiver<ResourceEvent> {
        self.resource_tx.subscribe()
    }
}

pub struct ParallelInference;

impl ParallelInference {
    pub async fn run<F, Fut>(
        prompt: String,
        models: Vec<String>,
        mut f: F,
    ) -> Result<Vec<(String, String)>>
    where
        F: FnMut(String, String) -> Fut,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
    {
        let mut set: tokio::task::JoinSet<(String, Result<String>)> = tokio::task::JoinSet::new();

        for model in models {
            let fut = f(prompt.clone(), model.clone());
            set.spawn(async move {
                let res = fut.await;
                (model, res)
            });
        }

        let mut results = Vec::new();
        while let Some(res) = set.join_next().await {
            match res {
                Ok((model, output)) => {
                    results.push((model, output?));
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Task join error: {:?}", e));
                }
            }
        }

        Ok(results)
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

pub struct InferenceCache {
    pub entries: HashMap<String, (String, f64)>, // hash -> (response, quality)
    pub hits: u64,
    pub misses: u64,
}

impl InferenceCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, prompt: &str, model_id: &str) -> Option<String> {
        let key = format!("{}:{}", prompt, model_id); // Simple key for now
        if let Some((res, _)) = self.entries.get(&key) {
            self.hits += 1;
            Some(res.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, prompt: &str, model_id: &str, response: String, quality: f64) {
        let key = format!("{}:{}", prompt, model_id);
        self.entries.insert(key, (response, quality));
    }
}

pub struct RateLimitManager {
    pub backoffs: HashMap<String, u32>, // model_id -> consecutive_429s
}

impl RateLimitManager {
    pub fn new() -> Self {
        Self {
            backoffs: HashMap::new(),
        }
    }

    pub fn get_delay(&self, model_id: &str) -> Duration {
        let attempts = self.backoffs.get(model_id).unwrap_or(&0);
        if *attempts == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs(2u64.pow(*attempts).min(60))
    }

    pub fn report_success(&mut self, model_id: &str) {
        self.backoffs.remove(model_id);
    }

    pub fn report_429(&mut self, model_id: &str) {
        let entry = self.backoffs.entry(model_id.to_string()).or_insert(0);
        *entry += 1;
    }
}

impl Default for ComputeManager {
    fn default() -> Self {
        Self::new()
    }
}
