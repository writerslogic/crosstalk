use crate::types::compute::{BudgetMode, CostEntry};
use crate::types::conversation::ConversationState;
use anyhow::Result;
use rand::Rng;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{Disks, System};
use tokio::sync::{Semaphore, broadcast};

/// Memory usage threshold above which a "Memory Critical" alert is emitted.
const MEMORY_CRITICAL_MB: u64 = 16_000;

/// CPU load percentage above which a "CPU Critical" alert is emitted.
const CPU_CRITICAL_THRESHOLD: f32 = 90.0;

/// Divisor to convert bytes (u64) to gibibytes (GiB).
const BYTES_TO_GIB: u64 = 1024 * 1024 * 1024;

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
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for ResourceMonitorActor {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Sums available space across all disks and converts to GiB.
fn disk_free_gib(disks: &Disks) -> u64 {
    disks.iter().map(|d| d.available_space()).sum::<u64>() / BYTES_TO_GIB
}

/// Returns an alert string when CPU or memory exceeds critical thresholds.
fn resource_alert(cpu_load: f32, rss_mb: u64) -> Option<String> {
    if cpu_load > CPU_CRITICAL_THRESHOLD {
        Some("CPU Critical".to_string())
    } else if rss_mb > MEMORY_CRITICAL_MB {
        Some("Memory Critical".to_string())
    } else {
        None
    }
}

impl ResourceMonitorActor {
    fn spawn(tx: broadcast::Sender<ResourceEvent>, interval_secs: u64) -> Self {
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
            let mut sys = System::new_all();
            let mut disks = Disks::new_with_refreshed_list();
            loop {
                ticker.tick().await;
                sys.refresh_all();
                disks.refresh(false);
                let rss_mb = sys.used_memory() / 1024 / 1024;
                let cpu_load = sys.global_cpu_usage();
                let disk_free_gb = disk_free_gib(&disks);
                let alert = resource_alert(cpu_load, rss_mb);
                if tx.send(ResourceEvent {
                    rss_mb,
                    cpu_load,
                    disk_free_gb,
                    alert,
                }).is_err() {
                    break;
                }
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
        let disks = Disks::new_with_refreshed_list();
        let disk_free_gb = disk_free_gib(&disks);
        let alert = resource_alert(cpu_load, rss_mb);

        let event = ResourceEvent {
            rss_mb,
            cpu_load,
            disk_free_gb,
            alert,
        };
        crate::log_warn!(self.resource_tx.send(event.clone()), "Failed to broadcast resource event");
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
    /// Runs parallel inference and returns the best response based on quality scores.
    pub async fn select_best<F, Fut, S>(
        prompt: String,
        models: Vec<String>,
        mut inference_fn: F,
        scorer: S,
    ) -> Result<(String, String, f64)>
    where
        F: FnMut(String, String) -> Fut,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
        S: Fn(&str) -> f64,
    {
        let mut set = tokio::task::JoinSet::new();
        for model in models {
            let fut = inference_fn(prompt.clone(), model.clone());
            set.spawn(async move { (model, fut.await) });
        }

        let mut candidates = Vec::new();
        while let Some(res) = set.join_next().await {
            if let Ok((model, Ok(text))) = res {
                let score = scorer(&text);
                candidates.push((model, text, score));
            }
        }

        candidates
            .into_iter()
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .ok_or_else(|| anyhow::anyhow!("All parallel inference tasks failed"))
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
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

thread_local! {
    static RNG: std::cell::RefCell<rand::rngs::ThreadRng> = std::cell::RefCell::new(rand::rng());
}

fn thread_local_jitter() -> f64 {
    RNG.with(|r| r.borrow_mut().random_range(-0.25..=0.25))
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
        let base_secs = 2u64.saturating_pow(attempts).min(60) as f64;
        let jitter: f64 = thread_local_jitter();
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
        Self {
            thresholds_ms: HashMap::new(),
        }
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

    pub async fn acquire(&self) -> Result<tokio::sync::SemaphorePermit<'_>, anyhow::Error> {
        self.semaphore
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("inference semaphore closed"))
    }

    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

pub struct RequestRateLimiter {
    requests_per_minute: u32,
    timestamps: tokio::sync::Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl RequestRateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            timestamps: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    pub async fn wait_for_permit(&self, model_id: &str) {
        let window = Duration::from_secs(60);
        let wait_duration = {
            let mut guard = self.timestamps.lock().await;
            let entries = guard.entry(model_id.to_string()).or_default();
            let now = Instant::now();
            while entries.front().is_some_and(|t| now.duration_since(*t) >= window) {
                entries.pop_front();
            }
            if entries.len() >= self.requests_per_minute as usize {
                entries
                    .front()
                    .map(|t| window.saturating_sub(now.duration_since(*t)))
                    .unwrap_or(Duration::ZERO)
            } else {
                entries.push_back(now);
                Duration::ZERO
            }
        };
        if !wait_duration.is_zero() {
            tokio::time::sleep(wait_duration).await;
            let mut guard = self.timestamps.lock().await;
            let entries = guard.entry(model_id.to_string()).or_default();
            let now = Instant::now();
            while entries.front().is_some_and(|t| now.duration_since(*t) >= window) {
                entries.pop_front();
            }
            entries.push_back(now);
        }
    }
}

pub struct LocalInference;

impl LocalInference {
    /// Placeholder for llama-cpp-rs integration.
    /// In emergency mode, this performs low-cost local synthesis.
    pub fn generate_emergency(prompt: &str) -> String {
        format!("[EMERGENCY LOCAL INFERENCE] Fallback active for prompt: {}

Note: Switched to offline synthesis to preserve final 5% budget.", 
                prompt.chars().take(100).collect::<String>())
    }
}
