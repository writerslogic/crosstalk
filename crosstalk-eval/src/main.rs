mod claude_agent;
mod dataset;
mod docker_env;
mod harness;
mod report;
mod runner;
mod scenarios;
mod swe_bench_runner;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// Load `~/.env` (KEY=value lines) into the process environment so that
/// clap's `env` attributes pick them up. Existing env vars are not overwritten.
fn load_dotenv() {
    let path = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".env"))
        .unwrap_or_default();
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((key, val)) = line.split_once('=') {
            let key = key.trim();
            let val = val.trim().trim_matches('"').trim_matches('\'');
            if std::env::var(key).is_err() {
                // SAFETY: single-threaded at this point (before tokio runtime starts).
                unsafe { std::env::set_var(key, val) };
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "crosstalk-eval",
    about = "Benchmarking harness for Crosstalk UCB1 topology selection (arXiv companion)"
)]
struct Args {
    /// Path to GSM8K JSONL dataset file (falls back to synthetic problems if absent)
    #[arg(short, long, default_value = "data/gsm8k_test.jsonl")]
    dataset: PathBuf,

    /// Output directory for CSV reports
    #[arg(short, long, default_value = "results")]
    output: PathBuf,

    /// Random seed for reproducibility
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Run only Scenario 1 (budget pressure test)
    #[arg(long, conflicts_with = "scenario2_only")]
    scenario1_only: bool,

    /// Run only Scenario 2 (UCB1 convergence test)
    #[arg(long, conflicts_with = "scenario1_only")]
    scenario2_only: bool,

    /// Path to SWE-bench (or SWE-bench Lite) JSONL dataset; enables Scenario 3
    #[arg(long)]
    swe_bench: Option<std::path::PathBuf>,

    /// Maximum turns per SWE-bench instance
    #[arg(long, default_value_t = 10)]
    swe_max_turns: u32,

    /// Run SWE-bench with live Docker containers (requires --swe-bench)
    #[arg(long, conflicts_with = "smoke_test")]
    live_run: bool,

    /// Run N instances sequentially as a smoke test (implies --live-run)
    #[arg(long, conflicts_with = "live_run")]
    smoke_test: bool,

    /// Number of instances for --smoke-test (default: 5)
    #[arg(long, default_value_t = 5)]
    count: usize,

    /// Max concurrent Docker containers for --live-run
    #[arg(long, default_value_t = 10)]
    concurrency: usize,

    /// Docker image prefix for SWE-bench containers
    #[arg(long, default_value = "sweb.eval.x86_64")]
    image_prefix: String,

    /// SWE-bench harness version embedded in image tags (e.g. 1776)
    #[arg(long, default_value = "1776")]
    image_version: String,

    /// Path for incremental JSONL checkpoint (appended on resume)
    #[arg(long, default_value = "results/live_run_checkpoint.jsonl")]
    checkpoint: PathBuf,

    /// OpenRouter API key (reads OPENROUTER_API_KEY env / ~/.env)
    #[arg(long, env = "OPENROUTER_API_KEY")]
    openrouter_key: Option<String>,

    /// Anthropic API key — use this to switch to Claude models
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    api_key: Option<String>,

    /// LLM model ID (default: Haiku — Fast tier)
    #[arg(long, default_value = claude_agent::DEFAULT_MODEL)]
    model: String,

    /// Reasoning-tier model used for complex topologies and temporal escalation
    #[arg(long, default_value = claude_agent::OPUS_MODEL)]
    reasoning_model: String,

    /// API base URL for OpenAI-compatible providers; omit to use Anthropic
    #[arg(long, default_value = claude_agent::OPENROUTER_BASE)]
    api_base: String,
}

impl Args {
    /// Resolve the effective (api_key, api_base) pair.
    ///
    /// Priority: --api-key → Anthropic, --openrouter-key → OpenRouter.
    /// Anthropic wins so that ANTHROPIC_API_KEY in ~/.env is always used for
    /// the single-model rigorous benchmark path when both keys are present.
    fn effective_agent(&self) -> (Option<String>, Option<String>) {
        if let Some(k) = &self.api_key {
            (Some(k.clone()), None) // None api_base → Anthropic
        } else if let Some(k) = &self.openrouter_key {
            (Some(k.clone()), Some(self.api_base.clone()))
        } else {
            (None, None)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    load_dotenv();

    tracing_subscriber::fmt()
        .with_env_filter("crosstalk_eval=info,warn")
        .init();

    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;

    let (api_key, api_base) = args.effective_agent();

    if args.live_run || args.smoke_test {
        let swe_path = args.swe_bench.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--swe-bench is required with --live-run / --smoke-test")
        })?;
        let instances = swe_bench_runner::load_swe_bench(swe_path)?;
        let mode = if args.smoke_test {
            runner::RunMode::SmokeTest { count: args.count }
        } else {
            runner::RunMode::FullRun {
                concurrency: args.concurrency,
            }
        };
        // When using OpenRouter (api_base is Some), the Anthropic key becomes
        // a Haiku fallback for when all free models are rate-limited.
        let fallback_anthropic_key = if api_base.is_some() {
            args.api_key.clone()
        } else {
            None
        };
        let cfg = runner::LiveRunConfig {
            mode,
            max_turns: args.swe_max_turns,
            seed: args.seed,
            checkpoint_path: args.checkpoint.clone(),
            image_prefix: args.image_prefix.clone(),
            image_version: args.image_version.clone(),
            api_key,
            model: args.model.clone(),
            api_base,
            fallback_anthropic_key,
            reasoning_model: args.reasoning_model.clone(),
        };
        let results = runner::run_live(&instances, cfg).await?;
        runner::print_live_summary(&results);
        return Ok(());
    }

    let questions = dataset::load_gsm8k(&args.dataset).unwrap_or_else(|e| {
        tracing::warn!("GSM8K dataset unavailable ({e}); using synthetic problems");
        dataset::synthetic_math_questions(200)
    });

    tracing::info!("Loaded {} problems", questions.len());

    if !args.scenario2_only {
        tracing::info!("=== Scenario 1: Budget Pressure Test ===");
        let records = scenarios::run_budget_pressure(&questions, args.seed)?;
        let path = args.output.join("topology_distribution.csv");
        report::write_budget_pressure_csv(&path, &records)?;
        tracing::info!("Written: {}", path.display());
        report::print_budget_pressure_summary(&records);
    }

    if !args.scenario1_only {
        tracing::info!("=== Scenario 2: UCB1 Convergence Test ===");
        let records = scenarios::run_ucb1_convergence(&questions, args.seed)?;
        let path = args.output.join("ucb1_convergence.csv");
        report::write_ucb1_convergence_csv(&path, &records)?;
        tracing::info!("Written: {}", path.display());
        report::print_ucb1_convergence_summary(&records);
    }

    if let Some(swe_path) = args.swe_bench {
        tracing::info!("=== Scenario 3: SWE-bench Evaluation ===");
        let instances = swe_bench_runner::load_swe_bench(&swe_path).unwrap_or_else(|e| {
            tracing::warn!("SWE-bench dataset unavailable ({e}); using synthetic instances");
            swe_bench_runner::synthetic_swe_instances(20)
        });
        tracing::info!("Loaded {} SWE-bench instances", instances.len());
        let fallback_anthropic_key = if api_base.is_some() {
            args.api_key.clone()
        } else {
            None
        };
        let cfg = swe_bench_runner::SweBenchRunnerConfig {
            max_turns: args.swe_max_turns,
            seed: args.seed,
            api_key,
            model: args.model.clone(),
            api_base,
            fallback_anthropic_key,
            reasoning_model: args.reasoning_model.clone(),
            ..Default::default()
        };
        let env = swe_bench_runner::MockSweBenchEnvironment::new(args.seed);
        let mut runner = swe_bench_runner::SweBenchRunner::with_environment(cfg, env);
        let results = runner.run_dataset(&instances).await?;
        let path = args.output.join("swe_bench_results.csv");
        swe_bench_runner::write_swe_bench_csv(&path, &results)?;
        tracing::info!("Written: {}", path.display());
        swe_bench_runner::print_swe_bench_summary(&results);
    }

    Ok(())
}
