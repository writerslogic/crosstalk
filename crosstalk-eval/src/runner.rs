//! Production execution pipeline for live SWE-bench evaluation.
//!
//! Manages concurrent Docker-backed instance runs with semaphore throttling,
//! incremental JSONL checkpointing, graceful Ctrl-C teardown, and live
//! progress logging — suitable for an unattended 10-hour batch run.
//!
//! ## Concurrency model
//!
//! A `tokio::sync::Semaphore` caps the number of simultaneously active
//! containers. Each acquired permit is held for exactly one instance's
//! lifecycle: container spawn → run → checkpoint → container stop → permit
//! release. This prevents Docker daemon thrash and keeps LLM API token
//! throughput within rate-limit budgets.
//!
//! ## Checkpointing
//!
//! Results are flushed to `checkpoint_path` (JSONL, one record per line)
//! immediately after each instance completes. A crash or rate-limit
//! permanent failure at instance N costs at most one partially-written
//! record, not the entire run.
//!
//! ## Graceful shutdown
//!
//! A background task listens for Ctrl-C. On signal it:
//! 1. Sets a cancellation flag so no new instances start.
//! 2. Runs `docker rm -f` on every active container, which causes all
//!    in-flight `docker exec` calls to fail immediately, unblocking the
//!    task futures so they can drain and release their semaphore permits.
//! 3. The main dispatch loop sees the flag and exits after the drain.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::process::Command as DockerCmd;
use tokio::sync::{Mutex, Semaphore};

use crate::docker_env::SwebenchHarnessEnv;
use crate::swe_bench_runner::{
    SweBenchInstance, SweBenchResult, SweBenchRunner, SweBenchRunnerConfig,
};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Execution mode selected by the CLI flag.
#[derive(Debug, Clone)]
pub enum RunMode {
    /// Run `count` instances sequentially for a quick sanity check.
    SmokeTest { count: usize },
    /// Run the full dataset with the given concurrency limit.
    FullRun { concurrency: usize },
}

/// Configuration for [`run_live`].
#[derive(Debug, Clone)]
pub struct LiveRunConfig {
    pub mode: RunMode,
    pub max_turns: u32,
    pub seed: u64,
    /// Path for incremental JSONL output. Appended to if it already exists,
    /// allowing a resumed run to continue from where it left off.
    pub checkpoint_path: PathBuf,
    /// Docker image name prefix (e.g. `"sweb.eval.x86_64"`).
    pub image_prefix: String,
    /// SWE-bench harness version embedded in image tags (e.g. `"1776"`).
    /// The full image name is:
    /// `swebench/<prefix>.<org>_<version>_<repo>-<issue>:latest`
    /// where the instance ID `{org}__{repo}-{issue}` is split on `__`.
    pub image_version: String,
    /// API key for the LLM provider. `None` = mock simulation.
    pub api_key: Option<String>,
    /// Model ID (e.g. `"meta-llama/llama-3.3-70b-instruct:free"`).
    pub model: String,
    /// API base URL. `None` = Anthropic. `Some` = OpenAI-compatible.
    pub api_base: Option<String>,
    /// Anthropic API key used as Haiku fallback when all free OpenRouter
    /// models are rate-limited. Only used when `api_base` is `Some`.
    pub fallback_anthropic_key: Option<String>,
    /// Reasoning-tier model for swarm escalation (default: Sonnet).
    pub reasoning_model: String,
}

impl Default for LiveRunConfig {
    fn default() -> Self {
        Self {
            mode: RunMode::FullRun { concurrency: 10 },
            max_turns: 10,
            seed: 42,
            checkpoint_path: PathBuf::from("results/live_run_checkpoint.jsonl"),
            image_prefix: "sweb.eval.x86_64".to_string(),
            image_version: "1776".to_string(),
            api_key: None,
            model: crate::claude_agent::DEFAULT_MODEL.to_string(),
            api_base: Some(crate::claude_agent::OPENROUTER_BASE.to_string()),
            fallback_anthropic_key: None,
            reasoning_model: crate::claude_agent::OPUS_MODEL.to_string(),
        }
    }
}

// ─── Shared state ─────────────────────────────────────────────────────────────

struct RunState {
    completed: AtomicU64,
    resolved: AtomicU64,
    failed: AtomicU64,
    /// Cost stored as integer microdollars (USD × 10⁶) for lock-free atomics.
    cost_ucu: AtomicU64,
    total: usize,
    start: Instant,
    cancelled: AtomicBool,
    /// Container IDs currently active; used by the Ctrl-C teardown handler.
    active_containers: Mutex<Vec<String>>,
}

impl RunState {
    fn new(total: usize) -> Self {
        Self {
            completed: AtomicU64::new(0),
            resolved: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cost_ucu: AtomicU64::new(0),
            total,
            start: Instant::now(),
            cancelled: AtomicBool::new(false),
            active_containers: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, result: &SweBenchResult) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        if result.metrics.patch_resolved {
            self.resolved.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed.fetch_add(1, Ordering::Relaxed);
        }
        let ucu = (result.metrics.total_cost_usd * 1_000_000.0) as u64;
        self.cost_ucu.fetch_add(ucu, Ordering::Relaxed);
    }

    fn log_progress(&self) {
        let done = self.completed.load(Ordering::Relaxed);
        let resolved = self.resolved.load(Ordering::Relaxed);
        let cost_usd = self.cost_ucu.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let avg_cost = if done > 0 {
            cost_usd / done as f64
        } else {
            0.0
        };

        let elapsed = self.start.elapsed().as_secs_f64();
        let eta = if done > 0 {
            let rate = done as f64 / elapsed;
            let rem = (self.total as u64).saturating_sub(done) as f64;
            format_eta(rem / rate)
        } else {
            "—".to_string()
        };

        let resolve_pct = if done > 0 {
            100.0 * resolved as f64 / done as f64
        } else {
            0.0
        };

        tracing::info!(
            "[{done}/{total}] Resolved: {resolved} ({resolve_pct:.1}%) | \
             Cost: ${cost_usd:.2} total / ${avg_cost:.4} avg | ETA: {eta}",
            total = self.total,
        );
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Run the SWE-bench dataset with live Docker containers.
///
/// Returns all completed [`SweBenchResult`]s, including partial results
/// collected before a Ctrl-C cancellation.
pub async fn run_live(
    instances: &[SweBenchInstance],
    config: LiveRunConfig,
) -> Result<Vec<SweBenchResult>> {
    let (instances_to_run, concurrency) = match &config.mode {
        RunMode::SmokeTest { count } => (&instances[..instances.len().min(*count)], 1),
        RunMode::FullRun { concurrency } => (instances, *concurrency),
    };

    let total = instances_to_run.len();
    tracing::info!(
        total,
        concurrency,
        checkpoint = %config.checkpoint_path.display(),
        "Starting live SWE-bench run"
    );

    // Open checkpoint file (append so a resume doesn't overwrite prior results).
    std::fs::create_dir_all(
        config
            .checkpoint_path
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.checkpoint_path)
        .with_context(|| {
            format!(
                "Cannot open checkpoint: {}",
                config.checkpoint_path.display()
            )
        })?;
    let checkpoint = Arc::new(Mutex::new(BufWriter::new(file)));

    let state = Arc::new(RunState::new(total));
    let sem = Arc::new(Semaphore::new(concurrency));
    let config = Arc::new(config);

    // ── Ctrl-C handler ────────────────────────────────────────────────────────
    let state_ctrlc = state.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
        tracing::warn!("Ctrl-C received — stopping after active instances drain");
        state_ctrlc.cancelled.store(true, Ordering::Relaxed);
        teardown_containers(&state_ctrlc).await;
    });

    // ── Dispatch loop ─────────────────────────────────────────────────────────
    let mut handles = Vec::with_capacity(total);

    for inst in instances_to_run {
        if state.cancelled.load(Ordering::Relaxed) {
            break;
        }

        // acquire_owned: the permit is moved into the spawned task and released
        // when the task completes (or when teardown kills its container).
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .context("semaphore closed")?;

        if state.cancelled.load(Ordering::Relaxed) {
            // Cancellation arrived while we were waiting for a permit.
            drop(permit);
            break;
        }

        let inst = inst.clone();
        let config = config.clone();
        let state = state.clone();
        let checkpoint = checkpoint.clone();

        handles.push(tokio::spawn(async move {
            let result = run_one(&inst, config, state, checkpoint).await;
            drop(permit);
            result
        }));
    }

    // ── Drain ─────────────────────────────────────────────────────────────────
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(Some(r)) => results.push(r),
            Ok(None) => {} // instance skipped (image missing, container error)
            Err(e) => tracing::error!("Task panicked: {e}"),
        }
    }

    // Final flush.
    checkpoint.lock().await.flush().ok();

    tracing::info!(
        completed = results.len(),
        "Run complete. Checkpoint: {}",
        config.checkpoint_path.display()
    );

    Ok(results)
}

// ─── Per-instance execution ───────────────────────────────────────────────────

/// Spawn a container, run one instance, checkpoint the result, stop the container.
///
/// Returns `None` if the container could not be started (image not found) or
/// if the run was cancelled. Non-fatal errors are logged and counted as failed.
async fn run_one(
    inst: &SweBenchInstance,
    config: Arc<LiveRunConfig>,
    state: Arc<RunState>,
    checkpoint: Arc<Mutex<BufWriter<std::fs::File>>>,
) -> Option<SweBenchResult> {
    if state.cancelled.load(Ordering::Relaxed) {
        return None;
    }

    let image = image_name(
        &config.image_prefix,
        &config.image_version,
        &inst.instance_id,
    );
    tracing::debug!(instance = %inst.instance_id, %image, "Spawning container");

    let env = match SwebenchHarnessEnv::spawn(&image, &inst.instance_id).await {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(instance = %inst.instance_id, "Container spawn failed: {e:#}");
            return None;
        }
    };

    let container_id = env.container_id().to_owned();
    state
        .active_containers
        .lock()
        .await
        .push(container_id.clone());

    let runner_cfg = SweBenchRunnerConfig {
        max_turns: config.max_turns,
        session_budget_usd: 5.00,
        seed: config.seed,
        verbose: false,
        api_key: config.api_key.clone(),
        model: config.model.clone(),
        api_base: config.api_base.clone(),
        fallback_anthropic_key: config.fallback_anthropic_key.clone(),
        reasoning_model: config.reasoning_model.clone(),
    };
    let mut runner = SweBenchRunner::with_environment(runner_cfg, env);

    let result = runner.run_instance(inst).await;

    // Unregister before stopping so teardown doesn't double-remove.
    state
        .active_containers
        .lock()
        .await
        .retain(|id| id != &container_id);

    // Best-effort container removal; log but don't propagate errors.
    let _ = DockerCmd::new("docker")
        .args(["rm", "-f", &container_id])
        .output()
        .await;

    match result {
        Ok(r) => {
            state.record(&r);
            state.log_progress();
            if let Err(e) = write_checkpoint(&checkpoint, &r).await {
                tracing::error!(instance = %inst.instance_id, "Checkpoint write failed: {e:#}");
            }
            Some(r)
        }
        Err(e) => {
            tracing::error!(instance = %inst.instance_id, "run_instance error: {e:#}");
            state.failed.fetch_add(1, Ordering::Relaxed);
            state.completed.fetch_add(1, Ordering::Relaxed);
            state.log_progress();
            None
        }
    }
}

// ─── Checkpointing ────────────────────────────────────────────────────────────

async fn write_checkpoint(
    checkpoint: &Arc<Mutex<BufWriter<std::fs::File>>>,
    result: &SweBenchResult,
) -> Result<()> {
    let line = serde_json::to_string(result).context("serialize result")? + "\n";
    let mut w = checkpoint.lock().await;
    w.write_all(line.as_bytes()).context("write checkpoint")?;
    w.flush().context("flush checkpoint")
}

// ─── Teardown ─────────────────────────────────────────────────────────────────

/// Force-remove all containers currently tracked in `state.active_containers`.
///
/// Called by the Ctrl-C handler. Killing the containers causes any in-flight
/// `docker exec` futures to return an error immediately, which unblocks the
/// task futures and lets them release their semaphore permits so the dispatch
/// loop can drain cleanly.
async fn teardown_containers(state: &RunState) {
    let ids = state.active_containers.lock().await.clone();
    if ids.is_empty() {
        return;
    }
    tracing::warn!("Tearing down {} active container(s)", ids.len());
    for id in &ids {
        match DockerCmd::new("docker")
            .args(["rm", "-f", id])
            .output()
            .await
        {
            Ok(_) => tracing::info!(container = %id, "Container removed"),
            Err(e) => tracing::warn!(container = %id, "docker rm failed: {e}"),
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Derive the SWE-bench Docker image name from the instance ID.
///
/// SWE-bench v2 naming convention:
/// `swebench/<prefix>.<org>_<version>_<repo>-<issue>:latest`
///
/// Instance IDs use `__` as the org/repo separator, e.g.
/// `astropy__astropy-12907` → `astropy_1776_astropy-12907`.
fn image_name(prefix: &str, version: &str, instance_id: &str) -> String {
    let body = if let Some((org, rest)) = instance_id.split_once("__") {
        format!("{org}_{version}_{rest}")
    } else {
        instance_id.to_owned()
    };
    format!("swebench/{prefix}.{body}:latest")
}

/// Format a duration in seconds as a human-readable ETA string.
fn format_eta(secs: f64) -> String {
    if secs.is_nan() || secs.is_infinite() || secs < 0.0 {
        return "—".to_string();
    }
    let s = secs as u64;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let s = s % 60;
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

// ─── Summary ─────────────────────────────────────────────────────────────────

/// Print a final aggregate summary to stdout after the run completes.
pub fn print_live_summary(results: &[SweBenchResult]) {
    if results.is_empty() {
        println!("No results collected.");
        return;
    }
    let n = results.len() as f64;
    let resolved = results.iter().filter(|r| r.metrics.patch_resolved).count();
    let cost = results
        .iter()
        .map(|r| r.metrics.total_cost_usd)
        .sum::<f64>();
    let latency = results
        .iter()
        .map(|r| r.metrics.total_latency_ms)
        .sum::<f64>()
        / n;
    let avg_turns = results
        .iter()
        .map(|r| r.metrics.turns_executed as f64)
        .sum::<f64>()
        / n;

    println!("\n=== Live SWE-bench Run Summary ===");
    println!("  Instances      : {}", results.len());
    println!(
        "  Resolved       : {resolved} ({:.1}%)",
        100.0 * resolved as f64 / n
    );
    println!("  Total cost     : ${cost:.4}");
    println!("  Avg cost       : ${:.4}", cost / n);
    println!("  Avg latency    : {latency:.0} ms");
    println!("  Avg turns      : {avg_turns:.1}");
    println!("==================================\n");
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_name_sanitizes_instance_id() {
        assert_eq!(
            image_name("sweb.eval.x86_64", "1776", "django__django-15790"),
            "swebench/sweb.eval.x86_64.django_1776_django-15790:latest"
        );
        assert_eq!(
            image_name("sweb.eval.arm64", "1776", "sympy__sympy-24213"),
            "swebench/sweb.eval.arm64.sympy_1776_sympy-24213:latest"
        );
        // Instance IDs without __ pass through unchanged.
        assert_eq!(
            image_name("sweb.eval.x86_64", "1776", "no-separator"),
            "swebench/sweb.eval.x86_64.no-separator:latest"
        );
    }

    #[test]
    fn format_eta_values() {
        assert_eq!(format_eta(0.0), "0s");
        assert_eq!(format_eta(59.0), "59s");
        assert_eq!(format_eta(90.0), "1m30s");
        assert_eq!(format_eta(3661.0), "1h01m");
        assert_eq!(format_eta(29580.0), "8h13m");
        assert_eq!(format_eta(f64::NAN), "—");
    }

    #[test]
    fn run_state_record_accumulates() {
        use crate::swe_bench_runner::SweBenchMetrics;
        let state = RunState::new(10);
        let result = SweBenchResult {
            instance_id: "test__test-1".to_string(),
            repo: "test/test".to_string(),
            difficulty: 0.5,
            produced_patch: Some("patch".to_string()),
            metrics: SweBenchMetrics {
                total_cost_usd: 0.42,
                total_latency_ms: 8_000.0,
                topology_sequence: vec![],
                patch_generated: true,
                patch_resolved: true,
                turns_executed: 5,
                total_tool_calls: 12,
            },
        };
        state.record(&result);
        assert_eq!(state.completed.load(Ordering::Relaxed), 1);
        assert_eq!(state.resolved.load(Ordering::Relaxed), 1);
        let cost_ucu = state.cost_ucu.load(Ordering::Relaxed);
        let cost_usd = cost_ucu as f64 / 1_000_000.0;
        assert!((cost_usd - 0.42).abs() < 1e-4);
    }

    #[test]
    fn run_state_failed_not_counted_as_resolved() {
        use crate::swe_bench_runner::SweBenchMetrics;
        let state = RunState::new(10);
        let result = SweBenchResult {
            instance_id: "test__test-2".to_string(),
            repo: "test/test".to_string(),
            difficulty: 0.3,
            produced_patch: None,
            metrics: SweBenchMetrics {
                total_cost_usd: 0.10,
                total_latency_ms: 5_000.0,
                topology_sequence: vec![],
                patch_generated: false,
                patch_resolved: false,
                turns_executed: 10,
                total_tool_calls: 20,
            },
        };
        state.record(&result);
        assert_eq!(state.resolved.load(Ordering::Relaxed), 0);
        assert_eq!(state.failed.load(Ordering::Relaxed), 1);
    }
}
