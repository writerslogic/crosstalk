//! SWE-bench Runner — evaluation harness for Crosstalk on software-engineering tasks.
//!
//! ## Overview
//!
//! [`SweBenchRunner`] feeds SWE-bench task instances through Crosstalk's UCB1
//! topology-selection engine, measures patch generation quality, and records
//! the per-instance metrics needed for the paper (Table 3 / Appendix B).
//!
//! ## Dataset format
//!
//! Expects a JSONL file where each line is a JSON object with at minimum:
//!
//! | Field               | Type   | Description                              |
//! |---------------------|--------|------------------------------------------|
//! | `instance_id`       | string | Unique task identifier                   |
//! | `repo`              | string | GitHub slug (e.g. `django/django`)        |
//! | `base_commit`       | string | SHA of the commit to patch               |
//! | `problem_statement` | string | The bug report / issue body              |
//! | `test_patch`        | string | Patch adding the regression test(s)      |
//!
//! Optional but used when present: `hints_text`, `patch` (gold standard),
//! `FAIL_TO_PASS`, `PASS_TO_PASS`, `version`.
//!
//! Download SWE-bench Lite (300 instances):
//! <https://github.com/princeton-nlp/SWE-bench>
//! Expected file: `data/swe_bench_lite.jsonl`
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   SweBenchRunner<E>                      │
//! │                                                          │
//! │  problem_statement                                       │
//! │       │                                                  │
//! │  ┌────▼──────────────────────────────────────────────┐  │
//! │  │            Multi-Turn Evaluation Loop              │  │
//! │  │                                                    │  │
//! │  │  for turn in 0..max_turns:                         │  │
//! │  │    topology  ←── CrosstalkHarness (UCB1)           │  │
//! │  │    output    ←── simulate_agent_turn(...)           │  │
//! │  │    tools     ←── parse_tool_directives(output)     │  │
//! │  │    responses ←── E::shell_exec / file_read / …     │  │
//! │  │    patch?    ←── extract_patch(output)             │  │
//! │  └────────────────────────────────────────────────────┘  │
//! │                                                          │
//! │  SweBenchMetrics { cost, latency, topology_seq, … }      │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Wasm / MCP Boundary
//!
//! In production, tool calls are dispatched through `McpGateway::dispatch`,
//! which can route `shell_exec` into a Wasmtime sandbox (`SandboxManager::execute`)
//! for untrusted code. This harness abstracts that boundary behind
//! [`SweBenchEnvironment`]. The provided [`MockSweBenchEnvironment`] simulates
//! realistic latency and output without containers, so UCB1 routing is testable
//! offline. To wire in a real environment, implement the trait and pass it to
//! [`SweBenchRunner::with_environment`].

use anyhow::{Context, Result};
use rand::{
    Rng, SeedableRng,
    distr::{Bernoulli, Distribution},
    rngs::StdRng,
};
use regex::Regex;
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    sync::OnceLock,
};

use crate::claude_agent::{ClaudeAgent, ModelTier};
use crate::harness::{BudgetMode, CrosstalkHarness, DebateTopology};

// ─── Dataset Types ────────────────────────────────────────────────────────────

/// A normalized SWE-bench task instance.
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields are public API consumed by real environment implementations
pub struct SweBenchInstance {
    /// Unique identifier (e.g. `"django__django-12345"`).
    pub instance_id: String,
    /// GitHub repository slug (e.g. `"django/django"`).
    pub repo: String,
    /// Base git commit SHA that the agent must patch.
    pub base_commit: String,
    /// The issue / bug report the agent must resolve.
    pub problem_statement: String,
    /// Patch that adds the regression test(s) (applied before evaluation).
    pub test_patch: String,
    /// Optional human-written hints from the dataset.
    pub hints_text: Option<String>,
    /// Gold-standard patch for reference (not shown to the agent).
    pub gold_patch: Option<String>,
    /// Tests that must flip from FAIL → PASS for a resolved instance.
    pub fail_to_pass: Vec<String>,
    /// Tests that must remain PASS after the agent's patch.
    pub pass_to_pass: Vec<String>,
    /// Library version string (e.g. `"3.2"`).
    pub version: Option<String>,
    /// Estimated difficulty ∈ [0.0, 1.0]. Higher = harder.
    pub difficulty: f64,
}

/// Raw JSONL record — superset of SWE-bench and SWE-bench Lite fields.
#[derive(Deserialize)]
struct SweBenchRecord {
    instance_id: String,
    repo: String,
    base_commit: String,
    problem_statement: String,
    #[serde(default)]
    test_patch: String,
    #[serde(default)]
    hints_text: Option<String>,
    #[serde(default)]
    patch: Option<String>,
    /// May be a JSON array or a JSON-encoded-string containing an array.
    #[serde(rename = "FAIL_TO_PASS", default)]
    fail_to_pass: serde_json::Value,
    #[serde(rename = "PASS_TO_PASS", default)]
    pass_to_pass: serde_json::Value,
    #[serde(default)]
    version: Option<String>,
}

/// Load and normalize a SWE-bench (or SWE-bench Lite) JSONL dataset.
///
/// Lines that fail JSON parsing are skipped with a WARN log. Returns an error
/// only if the file cannot be opened or yields zero valid instances.
pub fn load_swe_bench(path: &Path) -> Result<Vec<SweBenchInstance>> {
    let file = File::open(path)
        .with_context(|| format!("Cannot open SWE-bench dataset: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut instances = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("IO error on line {line_no}"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_swe_record(line) {
            Ok(inst) => instances.push(inst),
            Err(e) => tracing::warn!("Skipping SWE-bench line {line_no}: {e}"),
        }
    }

    anyhow::ensure!(
        !instances.is_empty(),
        "No valid SWE-bench instances found in {}",
        path.display()
    );
    tracing::info!(
        "Loaded {} SWE-bench instances from {}",
        instances.len(),
        path.display()
    );
    Ok(instances)
}

/// Generate synthetic SWE-bench–style instances for offline testing.
pub fn synthetic_swe_instances(n: usize) -> Vec<SweBenchInstance> {
    const REPOS: &[(&str, &str)] = &[
        ("requests/requests", "3.0"),
        ("pallets/flask", "2.3"),
        ("django/django", "4.2"),
        ("numpy/numpy", "1.25"),
        ("psf/black", "23.7"),
        ("sympy/sympy", "1.12"),
        ("pytest-dev/pytest", "7.4"),
        ("scikit-learn/scikit-learn", "1.3"),
    ];

    (0..n)
        .map(|i| {
            let (repo, version) = REPOS[i % REPOS.len()];
            let short = repo.split('/').next_back().unwrap_or("lib");
            let a = i + 3;
            let difficulty = 0.25 + (i % 6) as f64 * 0.09; // 0.25 – 0.70
            SweBenchInstance {
                instance_id: format!("{}__{i:04}", repo.replace('/', "__")),
                repo: repo.to_string(),
                base_commit: format!("c0ffee{i:06x}"),
                problem_statement: format!(
                    "Bug #{i}: Calling `{short}.process([])` raises `ValueError: empty sequence` \
                     instead of returning an empty list. This regression was introduced in \
                     commit `{i:06x}`. The guard clause at line ~{} does not account for \
                     zero-length inputs when the fast-path optimisation is active.",
                    40 + i * 3
                ),
                test_patch: format!(
                    "diff --git a/tests/test_{short}.py b/tests/test_{short}.py\n\
                     --- a/tests/test_{short}.py\n+++ b/tests/test_{short}.py\n\
                     @@ -1,0 +1,4 @@\n\
                     +def test_empty_input_{i}():\n\
                     +    from {short} import process\n\
                     +    assert process([]) == []\n"
                ),
                hints_text: Some(format!("The guard in `{short}/core.py` near line {a}")),
                gold_patch: None,
                fail_to_pass: vec![format!("tests/test_{short}.py::test_empty_input_{i}")],
                pass_to_pass: vec![format!("tests/test_{short}.py::test_basic")],
                version: Some(version.to_string()),
                difficulty,
            }
        })
        .collect()
}

fn parse_swe_record(line: &str) -> Result<SweBenchInstance> {
    let rec: SweBenchRecord = serde_json::from_str(line).context("Invalid JSON")?;
    let fail_to_pass = parse_test_list(&rec.fail_to_pass);
    let pass_to_pass = parse_test_list(&rec.pass_to_pass);
    let difficulty = estimate_difficulty(&rec, &fail_to_pass);

    Ok(SweBenchInstance {
        instance_id: rec.instance_id,
        repo: rec.repo,
        base_commit: rec.base_commit,
        problem_statement: rec.problem_statement,
        test_patch: rec.test_patch,
        hints_text: rec.hints_text,
        gold_patch: rec.patch,
        fail_to_pass,
        pass_to_pass,
        version: rec.version,
        difficulty,
    })
}

/// Normalise `FAIL_TO_PASS` / `PASS_TO_PASS`, which may be:
/// - A JSON array: `["a::b", "c::d"]`
/// - A JSON-encoded string: `"[\"a::b\"]"` (seen in some dataset versions)
/// - Absent / null → empty vec
fn parse_test_list(val: &serde_json::Value) -> Vec<String> {
    match val {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        serde_json::Value::String(s) => serde_json::from_str::<Vec<String>>(s).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Estimate task difficulty ∈ [0.0, 1.0] from heuristic signals.
///
/// - Longer problem statements → more complex root cause
/// - More `FAIL_TO_PASS` tests → wider blast radius
/// - Repository identity (sympy, matplotlib are structurally harder)
fn estimate_difficulty(rec: &SweBenchRecord, fail_to_pass: &[String]) -> f64 {
    let stmt_factor = (rec.problem_statement.len() as f64 / 2_000.0).min(1.0) * 0.40;
    let test_factor = ((fail_to_pass.len() as f64).ln_1p() / (10.0_f64).ln_1p()) * 0.30;
    let repo_factor = match rec.repo.as_str() {
        "sympy/sympy" | "matplotlib/matplotlib" | "scikit-learn/scikit-learn" => 0.30,
        "django/django" | "sphinx-doc/sphinx" | "astropy/astropy" => 0.20,
        "pylint-dev/pylint" | "pydata/xarray" => 0.15,
        _ => 0.10,
    };
    (stmt_factor + test_factor + repo_factor).min(1.0)
}

// ─── Tool Directive Parsing ───────────────────────────────────────────────────

/// A parsed tool call emitted by an agent in the form `[TOOL: name(args)]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolDirective {
    /// `[TOOL: shell_exec(git diff HEAD)]`
    ShellExec(String),
    /// `[TOOL: file_read(src/lib.rs)]`
    FileRead(String),
    /// `[TOOL: file_write(src/lib.rs, content …)]`
    FileWrite { path: String, content: String },
    /// `[TOOL: apply_patch(--- a/file …)]`
    ApplyPatch(String),
    /// `[TOOL: run_tests()]`
    RunTests,
    /// Any unrecognised directive — captured for logging.
    Unknown(String),
}

/// Parse all `[TOOL: name(args)]` directives from agent output text.
pub fn parse_tool_directives(text: &str) -> Vec<ToolDirective> {
    static TOOL_RE: OnceLock<Regex> = OnceLock::new();
    let re = TOOL_RE.get_or_init(|| {
        Regex::new(r"\[TOOL:\s*(\w+)\(([\s\S]*?)\)\]").expect("static regex is valid")
    });

    re.captures_iter(text)
        .map(|cap| {
            let name = cap[1].trim();
            let args = cap[2].trim();
            match name {
                "shell_exec" => ToolDirective::ShellExec(args.to_owned()),
                "file_read" => ToolDirective::FileRead(args.to_owned()),
                "run_tests" => ToolDirective::RunTests,
                "apply_patch" => ToolDirective::ApplyPatch(args.to_owned()),
                "file_write" => {
                    // Convention: first comma separates path from content.
                    if let Some((path, content)) = args.split_once(',') {
                        ToolDirective::FileWrite {
                            path: path.trim().to_owned(),
                            content: content.trim().to_owned(),
                        }
                    } else {
                        ToolDirective::Unknown(format!("file_write({args})"))
                    }
                }
                other => ToolDirective::Unknown(format!("{other}({args})")),
            }
        })
        .collect()
}

/// Extract the last patch block from agent output.
///
/// Recognises three formats (last one wins when multiple appear):
/// 1. `[PATCH]…[/PATCH]`  — preferred explicit marker
/// 2. ` ```diff … ``` `   — fenced diff block
/// 3. ` ```patch … ``` `  — alternate fenced marker
///
/// Candidates are validated to contain at least one unified-diff hunk
/// marker (`--- ` or `+++ `). When both formats appear, the one at the
/// highest byte offset wins (i.e. the most recent revision).
pub fn extract_patch(text: &str) -> Option<String> {
    static EXPLICIT_RE: OnceLock<Regex> = OnceLock::new();
    static FENCED_RE: OnceLock<Regex> = OnceLock::new();

    let explicit = EXPLICIT_RE
        .get_or_init(|| Regex::new(r"(?s)\[PATCH\](.*?)\[/PATCH\]").expect("static regex"));
    let fenced = FENCED_RE
        .get_or_init(|| Regex::new(r"(?s)```(?:diff|patch)\n(.*?)```").expect("static regex"));

    let is_diff = |s: &str| s.contains("--- ") || s.contains("+++ ");

    let mut best: Option<(usize, String)> = None;
    let mut consider = |pos: usize, body: String, require_diff: bool| {
        if (!require_diff || is_diff(&body)) && best.as_ref().is_none_or(|(p, _)| pos > *p) {
            best = Some((pos, body));
        }
    };

    for cap in explicit.captures_iter(text) {
        // Explicit [PATCH] markers are always trusted.
        consider(
            cap.get(0).map_or(0, |m| m.start()),
            cap[1].trim().to_owned(),
            false,
        );
    }
    for cap in fenced.captures_iter(text) {
        // Fenced blocks must look like a diff to avoid false positives.
        consider(
            cap.get(0).map_or(0, |m| m.start()),
            cap[1].trim().to_owned(),
            true,
        );
    }

    best.map(|(_, body)| body)
}

// ─── Environment Abstraction ──────────────────────────────────────────────────

/// Return value of a single tool call.
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields are public API consumed by real environment implementations
pub struct EnvResponse {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    /// Simulated or measured wall-clock duration.
    pub elapsed_ms: u64,
}

/// Result of running the instance's test suite.
#[derive(Debug, Clone)]
pub struct TestRunResult {
    /// All `fail_to_pass` tests pass AND all `pass_to_pass` tests still pass.
    pub resolved: bool,
    pub passed: usize,
    pub failed: usize,
    pub elapsed_ms: u64,
}

/// Abstract interface for the SWE-bench execution environment.
///
/// In production this wraps Docker containers via `CliBridge::invoke_with_timeout`
/// (or Wasmtime for sandboxed code via `SandboxManager::execute`).
/// This harness keeps that boundary here so the UCB1 routing logic is
/// testable without any container infrastructure.
///
/// ## Implementing for a real Docker environment
///
/// ```ignore
/// struct DockerEnv { container_id: String }
///
/// impl SweBenchEnvironment for DockerEnv {
///     async fn shell_exec(&mut self, cmd: &str) -> Result<EnvResponse> {
///         // Route through Crosstalk's MCP gateway or CliBridge directly:
///         // CliBridge::invoke_with_timeout("docker",
///         //   vec!["exec", &self.container_id, "bash", "-c", cmd], None, 60)
///         todo!()
///     }
///     // … remaining methods …
/// }
/// ```
pub trait SweBenchEnvironment: Send + Sync {
    async fn shell_exec(&mut self, cmd: &str) -> Result<EnvResponse>;
    async fn file_read(&mut self, path: &str) -> Result<EnvResponse>;
    async fn file_write(&mut self, path: &str, content: &str) -> Result<EnvResponse>;
    async fn apply_patch(&mut self, patch: &str) -> Result<EnvResponse>;
    async fn run_tests(&mut self, fail_to_pass: &[String]) -> Result<TestRunResult>;
    async fn reset(&mut self) -> Result<()>;
}

/// Simulated environment for offline harness evaluation.
///
/// Tool calls return plausible mock output after a configurable base latency
/// with ±20 % jitter. No real filesystem, git, or container access occurs.
///
/// State tracked across calls:
/// - Whether a well-formed patch has been applied (gates `run_tests` outcome)
/// - Accumulated `file_write` operations (path, byte count)
pub struct MockSweBenchEnvironment {
    rng: StdRng,
    /// Simulated base latency per tool call (ms).
    base_latency: u64,
    /// Set to `true` by `apply_patch` when the patch looks like a valid unified diff.
    patch_applied: bool,
    writes: Vec<(String, usize)>,
}

impl MockSweBenchEnvironment {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            base_latency: 200,
            patch_applied: false,
            writes: Vec::new(),
        }
    }

    fn jittered_latency(&mut self) -> u64 {
        let jitter = self.rng.random::<f64>() * 0.4 - 0.20; // ±20 %
        (self.base_latency as f64 * (1.0 + jitter)).max(1.0) as u64
    }
}

impl SweBenchEnvironment for MockSweBenchEnvironment {
    async fn shell_exec(&mut self, cmd: &str) -> Result<EnvResponse> {
        let elapsed = self.jittered_latency();
        let stdout = if cmd.starts_with("git diff") {
            "diff --git a/src/core.py b/src/core.py\n\
             index 1a2b3c4..5d6e7f8 100644\n\
             --- a/src/core.py\n+++ b/src/core.py\n\
             @@ -42,6 +42,7 @@ def process(data):\n"
                .to_string()
        } else if cmd.starts_with("grep") {
            "src/core.py:43:    if not data:\nsrc/core.py:47:    return [x * 2 for x in data]\n"
                .to_string()
        } else if cmd.starts_with("find") || cmd.starts_with("ls") {
            "src/\n  core.py\n  utils.py\ntests/\n  test_core.py\n".to_string()
        } else if cmd.contains("pytest") || cmd.contains("python -m") {
            "collected 12 items\n............                                  [100%]\n\
             12 passed in 0.41s\n"
                .to_string()
        } else if cmd.starts_with("git log") {
            "c0ffee1 Fix typo\nc0ffee2 Add feature\nc0ffee3 Initial commit\n".to_string()
        } else {
            format!("$ {cmd}\n[mock output]\n")
        };
        Ok(EnvResponse {
            success: true,
            stdout,
            stderr: String::new(),
            elapsed_ms: elapsed,
        })
    }

    async fn file_read(&mut self, path: &str) -> Result<EnvResponse> {
        let elapsed = self.jittered_latency();
        let content = format!(
            "# {path} (mock)\ndef process(data):\n    # BUG: missing empty-input guard\n\
                 result = []\n    for x in data:\n        result.append(x * 2)\n    return result\n"
        );
        Ok(EnvResponse {
            success: true,
            stdout: content,
            stderr: String::new(),
            elapsed_ms: elapsed,
        })
    }

    async fn file_write(&mut self, path: &str, content: &str) -> Result<EnvResponse> {
        let elapsed = self.jittered_latency();
        self.writes.push((path.to_owned(), content.len()));
        Ok(EnvResponse {
            success: true,
            stdout: format!("Wrote {} bytes to {path}", content.len()),
            stderr: String::new(),
            elapsed_ms: elapsed,
        })
    }

    async fn apply_patch(&mut self, patch: &str) -> Result<EnvResponse> {
        let elapsed = self.jittered_latency() * 2;
        // Accept any patch that looks like a minimal unified diff.
        let valid = patch.contains("---") && patch.contains("+++") && patch.contains("@@");
        self.patch_applied = valid;
        Ok(EnvResponse {
            success: valid,
            stdout: if valid {
                "Applied patch cleanly.\n".to_string()
            } else {
                String::new()
            },
            stderr: if valid {
                String::new()
            } else {
                "error: malformed diff\n".to_string()
            },
            elapsed_ms: elapsed,
        })
    }

    async fn run_tests(&mut self, fail_to_pass: &[String]) -> Result<TestRunResult> {
        let elapsed = self.jittered_latency() * 6; // test runs are slow
        let resolved = self.patch_applied;
        let passed = if resolved { fail_to_pass.len() } else { 0 };
        let failed = fail_to_pass.len() - passed;
        Ok(TestRunResult {
            resolved,
            passed,
            failed,
            elapsed_ms: elapsed,
        })
    }

    async fn reset(&mut self) -> Result<()> {
        self.patch_applied = false;
        self.writes.clear();
        Ok(())
    }
}

// ─── SWE-bench Topology Parameters ───────────────────────────────────────────

/// SWE-bench–specific performance parameters for each topology.
///
/// These differ from the GSM8K parameters in `harness.rs`:
/// - Overall solve rates are 10–30 % (vs. 68–85 % for math reasoning).
/// - TreeOfThoughts has greater advantage because hypothesis-branch exploration
///   maps naturally to "root-cause localisation" in multi-file codebases.
/// - Costs are ~4× higher than GSM8K due to longer context and tool-call overhead.
trait SweBenchTopologyExt {
    /// Probability that the task is ultimately resolved (all FAIL_TO_PASS pass).
    fn swe_quality_prob(self) -> f64;
    /// Mean number of tool calls emitted per agent turn.
    fn swe_tool_calls_per_turn(self) -> f64;
    /// Mean API cost per turn (USD), including multi-agent overhead.
    fn swe_mean_cost_usd(self) -> f64;
    /// Mean wall-clock latency per turn (ms).
    fn swe_mean_latency_ms(self) -> f64;
}

impl SweBenchTopologyExt for DebateTopology {
    fn swe_quality_prob(self) -> f64 {
        match self {
            Self::RoundRobin => 0.14,
            Self::Adversarial => 0.19,
            Self::Ensemble => 0.22,
            Self::TreeOfThoughts => 0.28,
            Self::Mediated => 0.17,
            Self::Critique => 0.16,
        }
    }

    fn swe_tool_calls_per_turn(self) -> f64 {
        match self {
            Self::TreeOfThoughts => 4.5,
            Self::Ensemble => 3.5,
            Self::Adversarial => 3.0,
            Self::Mediated => 2.5,
            Self::RoundRobin => 2.0,
            Self::Critique => 2.0,
        }
    }

    fn swe_mean_cost_usd(self) -> f64 {
        match self {
            Self::RoundRobin => 0.045,
            Self::Adversarial => 0.090,
            Self::Ensemble => 0.110,
            Self::TreeOfThoughts => 0.180,
            Self::Mediated => 0.100,
            Self::Critique => 0.055,
        }
    }

    fn swe_mean_latency_ms(self) -> f64 {
        match self {
            Self::RoundRobin => 8_000.0,
            Self::Adversarial => 14_000.0,
            Self::Ensemble => 15_000.0,
            Self::TreeOfThoughts => 28_000.0,
            Self::Mediated => 16_000.0,
            Self::Critique => 9_000.0,
        }
    }
}

// ─── Metrics & Result Types ───────────────────────────────────────────────────

/// One topology selection event within a multi-turn session.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopologyHop {
    /// Zero-based turn index at which this topology was selected.
    pub turn_index: u32,
    /// Topology name (mirrors `DebateTopology::name()`).
    pub topology: String,
    /// UCB1 efficiency score at the moment of selection.
    pub efficiency: f64,
    /// API cost incurred during this turn (USD).
    pub cost_usd: f64,
    /// Wall-clock latency for this turn (ms).
    pub latency_ms: f64,
    /// Model tier used for this turn ("Fast" or "Reasoning").
    pub model_tier: String,
}

/// Aggregate metrics for a single SWE-bench instance evaluation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweBenchMetrics {
    /// Cumulative API cost across all turns (USD).
    pub total_cost_usd: f64,
    /// Cumulative wall-clock latency across all turns (ms).
    pub total_latency_ms: f64,
    /// Ordered sequence of topology selections — one entry per turn.
    pub topology_sequence: Vec<TopologyHop>,
    /// True iff the agent emitted a `[PATCH]` block.
    pub patch_generated: bool,
    /// True iff the test suite passed after applying the agent's patch.
    pub patch_resolved: bool,
    /// Number of turns actually executed (≤ `max_turns`).
    pub turns_executed: u32,
    /// Total number of tool calls dispatched across all turns.
    pub total_tool_calls: u32,
}

/// Complete result for a single evaluated SWE-bench instance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweBenchResult {
    pub instance_id: String,
    pub repo: String,
    pub difficulty: f64,
    pub metrics: SweBenchMetrics,
    /// The patch text produced by the agent, if any.
    pub produced_patch: Option<String>,
}

// ─── Runner Configuration ─────────────────────────────────────────────────────

/// Configuration for [`SweBenchRunner`].
#[derive(Debug, Clone)]
pub struct SweBenchRunnerConfig {
    /// Maximum turns per instance before declaring failure.
    pub max_turns: u32,
    /// Total session budget (USD). Controls `BudgetMode` switching.
    pub session_budget_usd: f64,
    /// Random seed for simulation reproducibility.
    pub seed: u64,
    /// Log per-turn progress at DEBUG level.
    pub verbose: bool,
    /// When `Some`, use a real LLM API instead of simulation.
    pub api_key: Option<String>,
    /// Model ID passed to the API (default: [`claude_agent::DEFAULT_MODEL`]).
    pub model: String,
    /// API base URL. `None` → Anthropic. `Some(url)` → OpenAI-compatible
    /// (OpenRouter, Groq, Together AI, etc.).
    pub api_base: Option<String>,
    /// Anthropic API key used as Haiku fallback when all free OpenRouter
    /// models are rate-limited. Only applies when `api_base` is `Some`.
    pub fallback_anthropic_key: Option<String>,
    /// Reasoning-tier model for swarm escalation (default: Sonnet).
    /// Empty string disables escalation (all turns use the Fast tier).
    pub reasoning_model: String,
}

impl Default for SweBenchRunnerConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            session_budget_usd: 2.00,
            seed: 42,
            verbose: false,
            api_key: None,
            model: crate::claude_agent::DEFAULT_MODEL.to_string(),
            api_base: Some(crate::claude_agent::OPENROUTER_BASE.to_string()),
            fallback_anthropic_key: None,
            reasoning_model: crate::claude_agent::OPUS_MODEL.to_string(),
        }
    }
}

// ─── Runner ───────────────────────────────────────────────────────────────────

/// Async evaluation runner connecting SWE-bench instances to Crosstalk's UCB1
/// topology engine.
///
/// # Generic parameter
///
/// `E` implements [`SweBenchEnvironment`]. Use [`MockSweBenchEnvironment`] for
/// offline evaluation or supply a Docker-backed implementation for live runs.
pub struct SweBenchRunner<E: SweBenchEnvironment> {
    harness: CrosstalkHarness,
    config: SweBenchRunnerConfig,
    env: E,
    rng: StdRng,
    agent: Option<ClaudeAgent>,
}

/// Select the model tier for a given turn.
///
/// Complex topologies and temporal escalation (turn ≥ 4 with no patch yet)
/// use the Reasoning tier (Sonnet); all others use the Fast tier (Haiku).
fn tier_for_turn(topology: DebateTopology, turn: u32, no_patch_yet: bool) -> ModelTier {
    if turn >= 4 && no_patch_yet {
        return ModelTier::Reasoning;
    }
    match topology {
        DebateTopology::Adversarial | DebateTopology::TreeOfThoughts | DebateTopology::Mediated => {
            ModelTier::Reasoning
        }
        _ => ModelTier::Fast,
    }
}

impl<E: SweBenchEnvironment> SweBenchRunner<E> {
    /// Create a runner backed by `env`.
    pub fn with_environment(config: SweBenchRunnerConfig, env: E) -> Self {
        let harness = CrosstalkHarness::new(config.seed);
        let rng = StdRng::seed_from_u64(config.seed.wrapping_add(1));
        let agent = config.api_key.clone().map(|key| match &config.api_base {
            Some(base) => {
                let models: Vec<String> = crate::claude_agent::DEFAULT_MODELS
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let agent = ClaudeAgent::new_openai_compat(key, models, base.clone());
                if let Some(fk) = config.fallback_anthropic_key.clone() {
                    agent.with_haiku_fallback(fk)
                } else {
                    agent
                }
            }
            None => ClaudeAgent::new(key, config.model.clone())
                .with_reasoning_model(config.reasoning_model.clone()),
        });
        Self {
            harness,
            config,
            env,
            rng,
            agent,
        }
    }

    /// Evaluate a single SWE-bench instance and return per-instance metrics.
    ///
    /// Uses the real Claude API when `config.api_key` is set; otherwise falls
    /// back to the probabilistic mock simulation.
    pub async fn run_instance(&mut self, inst: &SweBenchInstance) -> Result<SweBenchResult> {
        if let Some(mut agent) = self.agent.take() {
            let result = self.run_instance_real(inst, &mut agent).await;
            self.agent = Some(agent);
            result
        } else {
            self.run_instance_mock(inst).await
        }
    }

    /// Real agent loop: calls the Claude API each turn and feeds tool results
    /// back so the model can reason across turns.
    async fn run_instance_real(
        &mut self,
        inst: &SweBenchInstance,
        agent: &mut ClaudeAgent,
    ) -> Result<SweBenchResult> {
        self.env.reset().await.context("environment reset failed")?;
        agent.reset();

        // Apply the test patch so the agent can run the failing tests.
        if !inst.test_patch.is_empty() {
            let r = self
                .env
                .apply_patch(&inst.test_patch)
                .await
                .unwrap_or(EnvResponse {
                    success: false,
                    stdout: String::new(),
                    stderr: "test patch apply failed".into(),
                    elapsed_ms: 0,
                });
            if !r.success {
                tracing::warn!(instance = %inst.instance_id, "Test patch did not apply: {}", r.stderr);
            }
        }

        let mut total_cost_usd = 0.0_f64;
        let mut total_latency_ms = 0.0_f64;
        let mut topology_sequence: Vec<TopologyHop> = Vec::new();
        let mut total_tool_calls = 0_u32;
        let mut produced_patch = None::<String>;
        let mut patch_resolved = false;
        let mut prev_tier = ModelTier::Fast;

        let hints_section = inst
            .hints_text
            .as_deref()
            .map(|h| format!("\nHints:\n{h}\n"))
            .unwrap_or_default();
        let tests_list = inst.fail_to_pass.join("\n- ");

        let mut next_user_msg = format!(
            "Repository: {repo}\nProblem statement:\n{stmt}\n\n\
             Tests that must change FAIL → PASS:\n- {tests}{hints}",
            repo = inst.repo,
            stmt = inst.problem_statement,
            tests = tests_list,
            hints = hints_section,
        );

        'turns: for turn in 0..self.config.max_turns {
            let budget_mode = self.budget_mode(total_cost_usd);
            let run = self.harness.run(budget_mode);
            let topology = run.winning_topology;

            let tier = tier_for_turn(topology, turn, produced_patch.is_none());
            agent.set_tier(tier);

            // On the first escalation to Reasoning tier, prepend a synthesis
            // directive so Sonnet gets a clean framing of what Haiku found,
            // rather than inheriting its exploratory noise raw.
            if tier == ModelTier::Reasoning && prev_tier == ModelTier::Fast && turn > 0 {
                next_user_msg = format!(
                    "You are now the primary reasoning agent. Review the conversation \
                     history above and synthesize: (1) which file and function is most \
                     likely responsible for the bug, (2) what the correct fix is. \
                     Then emit a [PATCH] immediately — do not explore further unless \
                     the patch fails tests.\n\n{}",
                    next_user_msg
                );
            }
            prev_tier = tier;

            // Mechanical enforcement — cannot be overridden by prompt instructions.
            // Turn 2: no patch yet → force one.
            if turn == 2 && produced_patch.is_none() {
                next_user_msg = "You have used 2 turns exploring. You MUST emit a \
                    [PATCH] block in this response. Make your best guess at the fix \
                    based on what you have found. Do not call any more tools — \
                    emit [PATCH] only."
                    .to_string();
            }
            // Turn 5: still no patch applied → no more budget, exit.
            if turn >= 5 && produced_patch.is_none() {
                tracing::info!(instance = %inst.instance_id, turn, "Hard stop: no patch after turn 5");
                break 'turns;
            }

            let t0 = std::time::Instant::now();
            let (output, cost) = match agent.send(next_user_msg.clone()).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(turn, "Claude API error: {e:#}");
                    break 'turns;
                }
            };
            let latency = t0.elapsed().as_millis() as f64;

            total_cost_usd += cost;
            total_latency_ms += latency;

            let tier_name = match tier {
                ModelTier::Fast => "Fast",
                ModelTier::Reasoning => "Reasoning",
            };
            topology_sequence.push(TopologyHop {
                turn_index: turn,
                topology: topology.name().to_owned(),
                efficiency: run.efficiency,
                cost_usd: cost,
                latency_ms: latency,
                model_tier: tier_name.to_owned(),
            });

            if self.config.verbose {
                tracing::debug!(
                    turn,
                    topology = topology.name(),
                    tier = tier_name,
                    cost,
                    latency,
                    "Agent turn"
                );
            }

            // Dispatch all tool calls; collect formatted results for next turn.
            let directives = parse_tool_directives(&output);
            total_tool_calls += directives.len() as u32;

            let mut result_parts: Vec<String> = Vec::new();
            for directive in &directives {
                let formatted = self.dispatch_and_format(directive, inst).await;
                result_parts.push(formatted);
            }

            next_user_msg = if result_parts.is_empty() {
                "No tool calls detected. Use [TOOL: ...] directives to explore and fix the bug."
                    .to_string()
            } else {
                result_parts.join("\n\n")
            };

            // Check for a patch; apply and test it.
            if let Some(patch) = extract_patch(&output) {
                let apply_resp = self.env.apply_patch(&patch).await.unwrap_or(EnvResponse {
                    success: false,
                    stdout: String::new(),
                    stderr: "apply_patch error".into(),
                    elapsed_ms: 0,
                });

                if apply_resp.success {
                    let test =
                        self.env
                            .run_tests(&inst.fail_to_pass)
                            .await
                            .unwrap_or(TestRunResult {
                                resolved: false,
                                passed: 0,
                                failed: inst.fail_to_pass.len(),
                                elapsed_ms: 0,
                            });

                    total_latency_ms += (apply_resp.elapsed_ms + test.elapsed_ms) as f64;
                    patch_resolved = test.resolved;
                    produced_patch = Some(patch);

                    tracing::info!(
                        instance = %inst.instance_id, turn, resolved = patch_resolved,
                        "Patch generated and tested"
                    );

                    if patch_resolved {
                        break 'turns;
                    }

                    next_user_msg = format!(
                        "Patch applied but tests still failing ({} passed, {} failed). \
                         Please investigate and try a different fix.\n\n{}",
                        test.passed,
                        test.failed,
                        result_parts.join("\n\n")
                    );
                } else {
                    next_user_msg = format!(
                        "git apply failed: {}\n\nPlease check paths and hunk offsets.\n\n{}",
                        apply_resp.stderr,
                        result_parts.join("\n\n")
                    );
                }
            }
        }

        let patch_generated = produced_patch.is_some();
        let turns_executed = topology_sequence.len() as u32;

        Ok(SweBenchResult {
            instance_id: inst.instance_id.clone(),
            repo: inst.repo.clone(),
            difficulty: inst.difficulty,
            produced_patch,
            metrics: SweBenchMetrics {
                total_cost_usd,
                total_latency_ms,
                topology_sequence,
                patch_generated,
                patch_resolved,
                turns_executed,
                total_tool_calls,
            },
        })
    }

    /// Mock simulation loop (no API calls; used when `config.api_key` is None).
    async fn run_instance_mock(&mut self, inst: &SweBenchInstance) -> Result<SweBenchResult> {
        self.env.reset().await.context("environment reset failed")?;

        let mut total_cost_usd = 0.0_f64;
        let mut total_latency_ms = 0.0_f64;
        let mut topology_sequence: Vec<TopologyHop> = Vec::new();
        let mut total_tool_calls = 0_u32;
        let mut produced_patch = None::<String>;
        let mut patch_resolved = false;

        tracing::debug!(
            instance = %inst.instance_id,
            difficulty = inst.difficulty,
            "Starting SWE-bench evaluation"
        );

        'turns: for turn in 0..self.config.max_turns {
            let budget_mode = self.budget_mode(total_cost_usd);

            let run = self.harness.run(budget_mode);
            let topology = run.winning_topology;

            let (agent_output, sim_cost, sim_latency) =
                self.simulate_agent_turn(topology, turn, inst);

            total_cost_usd += sim_cost;
            total_latency_ms += sim_latency;

            topology_sequence.push(TopologyHop {
                turn_index: turn,
                topology: topology.name().to_owned(),
                efficiency: run.efficiency,
                cost_usd: sim_cost,
                latency_ms: sim_latency,
                model_tier: "Fast".to_owned(),
            });

            if self.config.verbose {
                tracing::debug!(
                    turn,
                    topology = topology.name(),
                    cost = sim_cost,
                    latency = sim_latency,
                    "Agent turn"
                );
            }

            let directives = parse_tool_directives(&agent_output);
            total_tool_calls += directives.len() as u32;
            for directive in directives {
                if let Err(e) = self.dispatch_tool(directive, inst).await {
                    tracing::warn!(turn, "Tool call error: {e}");
                }
            }

            if let Some(patch) = extract_patch(&agent_output) {
                let apply_resp = self.env.apply_patch(&patch).await.unwrap_or(EnvResponse {
                    success: false,
                    stdout: String::new(),
                    stderr: "apply_patch error".into(),
                    elapsed_ms: 0,
                });

                if apply_resp.success {
                    let test =
                        self.env
                            .run_tests(&inst.fail_to_pass)
                            .await
                            .unwrap_or(TestRunResult {
                                resolved: false,
                                passed: 0,
                                failed: inst.fail_to_pass.len(),
                                elapsed_ms: 0,
                            });

                    total_latency_ms += (apply_resp.elapsed_ms + test.elapsed_ms) as f64;
                    patch_resolved = test.resolved;
                    produced_patch = Some(patch);

                    tracing::info!(
                        instance  = %inst.instance_id,
                        turn,
                        resolved  = patch_resolved,
                        "Patch generated and tested"
                    );
                    break 'turns;
                }
            }
        }

        let patch_generated = produced_patch.is_some();
        let turns_executed = topology_sequence.len() as u32;

        Ok(SweBenchResult {
            instance_id: inst.instance_id.clone(),
            repo: inst.repo.clone(),
            difficulty: inst.difficulty,
            produced_patch,
            metrics: SweBenchMetrics {
                total_cost_usd,
                total_latency_ms,
                topology_sequence,
                patch_generated,
                patch_resolved,
                turns_executed,
                total_tool_calls,
            },
        })
    }

    /// Execute a tool directive and return a formatted result string for Claude.
    async fn dispatch_and_format(
        &mut self,
        directive: &ToolDirective,
        _inst: &SweBenchInstance,
    ) -> String {
        /// Hard cap on tool output fed back into the conversation context.
        /// Prevents large grep/cat outputs from flooding the context window.
        const MAX_TOOL_OUTPUT: usize = 3_000;

        fn truncate(s: String) -> String {
            if s.len() <= MAX_TOOL_OUTPUT {
                return s;
            }
            let head = &s[..MAX_TOOL_OUTPUT];
            format!(
                "{head}\n[... output truncated at {MAX_TOOL_OUTPUT} chars — use more targeted commands ...]"
            )
        }

        match directive {
            ToolDirective::ShellExec(cmd) => match self.env.shell_exec(cmd).await {
                Ok(r) => {
                    let out = if r.stdout.is_empty() {
                        r.stderr
                    } else {
                        r.stdout
                    };
                    let out = truncate(out);
                    format!(
                        "[shell_exec({cmd}) exit={}]\n{out}",
                        if r.success { 0 } else { 1 }
                    )
                }
                Err(e) => format!("[shell_exec({cmd}) error]\n{e}"),
            },
            ToolDirective::FileRead(path) => match self.env.file_read(path).await {
                Ok(r) => format!("[file_read({path})]\n{}", truncate(r.stdout)),
                Err(e) => format!("[file_read({path}) error]\n{e}"),
            },
            ToolDirective::FileWrite { path, content } => {
                match self.env.file_write(path, content).await {
                    Ok(r) => format!(
                        "[file_write({path})]\n{}",
                        if r.success { "OK" } else { &r.stderr }
                    ),
                    Err(e) => format!("[file_write({path}) error]\n{e}"),
                }
            }
            ToolDirective::ApplyPatch(patch) => match self.env.apply_patch(patch).await {
                Ok(r) => format!(
                    "[apply_patch]\n{}",
                    if r.success { "OK" } else { &r.stderr }
                ),
                Err(e) => format!("[apply_patch error]\n{e}"),
            },
            ToolDirective::RunTests => {
                // In real-agent mode, test runs are triggered automatically
                // after a [PATCH] block is applied. Running pytest speculatively
                // every turn is expensive (~60s each) and rarely informative
                // without a patch applied first.
                "[run_tests]\nTests will run automatically after you submit a [PATCH] block. \
                 Use shell_exec to run pytest manually if you need interim results."
                    .to_string()
            }
            ToolDirective::Unknown(raw) => {
                format!("[unknown tool: {raw}]\nThis tool is not available.")
            }
        }
    }

    /// Evaluate all instances sequentially.
    ///
    /// UCB1 state is preserved across instances so the bandit can learn
    /// cross-instance topology preferences — matching production Crosstalk
    /// behaviour where a single session may span many tasks.
    pub async fn run_dataset(
        &mut self,
        instances: &[SweBenchInstance],
    ) -> Result<Vec<SweBenchResult>> {
        let mut results = Vec::with_capacity(instances.len());
        for (idx, inst) in instances.iter().enumerate() {
            tracing::info!(
                "[{}/{}] Evaluating {}",
                idx + 1,
                instances.len(),
                inst.instance_id
            );
            let result = self
                .run_instance(inst)
                .await
                .with_context(|| format!("instance {} failed", inst.instance_id))?;
            results.push(result);
        }
        Ok(results)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn budget_mode(&self, spent_usd: f64) -> BudgetMode {
        let remaining = (self.config.session_budget_usd - spent_usd).max(0.0)
            / self.config.session_budget_usd.max(f64::EPSILON);
        if remaining > 0.20 {
            BudgetMode::Normal
        } else if remaining > 0.05 {
            BudgetMode::CostReduction
        } else {
            BudgetMode::Emergency
        }
    }

    /// Produce one agent turn's output string and its simulated cost/latency.
    ///
    /// Whether a `[PATCH]` block is included is determined probabilistically:
    /// - p_patch = topology_quality × (1 − difficulty×0.5) × turn_ramp(turn)
    /// - `turn_ramp` is a bell-shaped ramp peaking at ≈ 50 % of the turn budget,
    ///   so agents explore first and patch mid-session — matching observed
    ///   SWE-bench agent trajectories.
    fn simulate_agent_turn(
        &mut self,
        topology: DebateTopology,
        turn: u32,
        inst: &SweBenchInstance,
    ) -> (String, f64, f64) {
        let cost = sample_lognormal(&mut self.rng, topology.swe_mean_cost_usd(), 0.20);
        let latency = sample_lognormal(&mut self.rng, topology.swe_mean_latency_ms(), 0.25);

        let n_tools = topology.swe_tool_calls_per_turn().round() as usize;
        let tool_block = build_tool_block(n_tools, turn, self.config.max_turns, inst);

        let ramp = turn_patch_ramp(turn, self.config.max_turns);
        let p_patch =
            (topology.swe_quality_prob() * (1.0 - inst.difficulty * 0.5) * ramp).clamp(0.0, 1.0);

        let emit_patch = Bernoulli::new(p_patch)
            .expect("p_patch clamped to [0,1]")
            .sample(&mut self.rng);

        let short = inst.repo.split('/').next_back().unwrap_or("lib");
        let patch_block = if emit_patch {
            format!(
                "\n\n[PATCH]\n\
                 --- a/src/{short}/core.py\n\
                 +++ b/src/{short}/core.py\n\
                 @@ -42,4 +42,7 @@ def process(data):\n\
                 +    if not data:\n\
                 +        return []\n\
                      result = []\n\
                      for x in data:\n\
                          result.append(x * 2)\n\
                 [/PATCH]"
            )
        } else {
            String::new()
        };

        let stmt_preview = &inst.problem_statement[..inst.problem_statement.len().min(140)];
        let output = format!(
            "[Turn {turn} | {topo}] Instance: {id}\nProblem: {stmt}…\n\n{tools}{patch}",
            topo = topology.name(),
            id = inst.instance_id,
            stmt = stmt_preview,
            tools = tool_block,
            patch = patch_block,
        );

        (output, cost, latency)
    }

    async fn dispatch_tool(
        &mut self,
        directive: ToolDirective,
        inst: &SweBenchInstance,
    ) -> Result<EnvResponse> {
        match directive {
            ToolDirective::ShellExec(cmd) => self.env.shell_exec(&cmd).await,
            ToolDirective::FileRead(path) => self.env.file_read(&path).await,
            ToolDirective::FileWrite { path, content } => {
                self.env.file_write(&path, &content).await
            }
            ToolDirective::ApplyPatch(patch) => self.env.apply_patch(&patch).await,
            ToolDirective::RunTests => {
                self.env
                    .run_tests(&inst.fail_to_pass)
                    .await
                    .map(|r| EnvResponse {
                        success: r.resolved,
                        stdout: format!("{} passed, {} failed", r.passed, r.failed),
                        stderr: String::new(),
                        elapsed_ms: r.elapsed_ms,
                    })
            }
            ToolDirective::Unknown(raw) => {
                tracing::warn!("Unknown tool directive: {raw}");
                Ok(EnvResponse {
                    success: false,
                    stdout: String::new(),
                    stderr: format!("unknown tool: {raw}"),
                    elapsed_ms: 0,
                })
            }
        }
    }
}

// ─── Simulation Helpers ───────────────────────────────────────────────────────

/// Patch-generation ramp: bell-shaped, peaking at ≈ 50 % of the turn budget.
///
/// `f(progress) = 4·p·(1−p)` gives 0 at the endpoints and 1.0 at p = 0.5.
/// Agents explore in early turns and are most likely to emit a patch mid-way.
fn turn_patch_ramp(turn: u32, max_turns: u32) -> f64 {
    if max_turns == 0 {
        return 0.0;
    }
    let p = turn as f64 / max_turns as f64;
    4.0 * p * (1.0 - p)
}

/// Build a representative `[TOOL: ...]` block for the given turn.
///
/// Early turns are exploration-heavy (git diff, file reads, grep).
/// Mid turns shift to writes. Late turns focus on test verification.
fn build_tool_block(n_tools: usize, turn: u32, max_turns: u32, inst: &SweBenchInstance) -> String {
    let short = inst.repo.split('/').next_back().unwrap_or("lib");
    let progress = if max_turns > 0 {
        turn as f64 / max_turns as f64
    } else {
        0.5
    };

    (0..n_tools)
        .map(|i| {
            if progress < 0.30 || (i == 0 && progress < 0.55) {
                // Exploration phase
                match i % 3 {
                    0 => "[TOOL: shell_exec(git diff HEAD)]".to_string(),
                    1 => format!("[TOOL: file_read(src/{short}/core.py)]"),
                    _ => format!("[TOOL: shell_exec(grep -n 'def process' src/{short}/core.py)]"),
                }
            } else if progress < 0.70 {
                // Patch-attempt phase
                if i % 2 == 0 {
                    format!("[TOOL: file_write(src/{short}/core.py, fixed content)]")
                } else {
                    "[TOOL: run_tests()]".to_string()
                }
            } else {
                // Verification phase
                "[TOOL: run_tests()]".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Log-normal sampler using the Box-Muller transform.
///
/// Parametrisation matches `harness.rs`:
/// - `σ²_ln = ln(CV² + 1)`
/// - `μ_ln  = ln(mean) − σ²_ln / 2`
fn sample_lognormal(rng: &mut StdRng, mean: f64, cv: f64) -> f64 {
    let sigma_sq = (cv * cv + 1.0).ln();
    let mu = mean.ln() - sigma_sq / 2.0;
    let sigma = sigma_sq.sqrt();
    let u1: f64 = rng.random::<f64>().max(f64::EPSILON);
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (mu + sigma * z).exp()
}

// ─── CSV Report ───────────────────────────────────────────────────────────────

/// Write per-instance SWE-bench results to a CSV file.
///
/// Columns: `instance_id`, `repo`, `difficulty`, `total_cost_usd`,
/// `total_latency_ms`, `turns_executed`, `total_tool_calls`,
/// `patch_generated`, `patch_resolved`, `topology_sequence`
pub fn write_swe_bench_csv(path: &Path, results: &[SweBenchResult]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)
        .with_context(|| format!("Cannot create CSV: {}", path.display()))?;

    wtr.write_record([
        "instance_id",
        "repo",
        "difficulty",
        "total_cost_usd",
        "total_latency_ms",
        "turns_executed",
        "total_tool_calls",
        "patch_generated",
        "patch_resolved",
        "topology_sequence",
    ])?;

    for r in results {
        let topo_seq = r
            .metrics
            .topology_sequence
            .iter()
            .map(|h| h.topology.as_str())
            .collect::<Vec<_>>()
            .join(";");

        wtr.write_record([
            &r.instance_id,
            &r.repo,
            &format!("{:.3}", r.difficulty),
            &format!("{:.6}", r.metrics.total_cost_usd),
            &format!("{:.1}", r.metrics.total_latency_ms),
            &r.metrics.turns_executed.to_string(),
            &r.metrics.total_tool_calls.to_string(),
            &r.metrics.patch_generated.to_string(),
            &r.metrics.patch_resolved.to_string(),
            &topo_seq,
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Print an aggregate summary of SWE-bench results to stdout.
pub fn print_swe_bench_summary(results: &[SweBenchResult]) {
    if results.is_empty() {
        println!("No SWE-bench results.");
        return;
    }
    let n = results.len() as f64;
    let generated = results.iter().filter(|r| r.metrics.patch_generated).count();
    let resolved = results.iter().filter(|r| r.metrics.patch_resolved).count();
    let avg_cost = results
        .iter()
        .map(|r| r.metrics.total_cost_usd)
        .sum::<f64>()
        / n;
    let avg_lat = results
        .iter()
        .map(|r| r.metrics.total_latency_ms)
        .sum::<f64>()
        / n;
    let avg_turns = results
        .iter()
        .map(|r| r.metrics.turns_executed as f64)
        .sum::<f64>()
        / n;
    let avg_tools = results
        .iter()
        .map(|r| r.metrics.total_tool_calls as f64)
        .sum::<f64>()
        / n;

    let mut topo_counts: HashMap<&str, usize> = HashMap::new();
    for r in results {
        for hop in &r.metrics.topology_sequence {
            *topo_counts.entry(hop.topology.as_str()).or_default() += 1;
        }
    }
    let total_hops: usize = topo_counts.values().sum();
    let mut topo_sorted: Vec<_> = topo_counts.iter().collect();
    topo_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("\n=== SWE-bench Evaluation Summary ===");
    println!("  Instances        : {}", results.len());
    println!(
        "  Patch generated  : {} ({:.1}%)",
        generated,
        100.0 * generated as f64 / n
    );
    println!(
        "  Resolved         : {} ({:.1}%)",
        resolved,
        100.0 * resolved as f64 / n
    );
    println!("  Avg cost (USD)   : ${avg_cost:.4}");
    println!("  Avg latency (ms) : {avg_lat:.0}");
    println!("  Avg turns        : {avg_turns:.1}");
    println!("  Avg tool calls   : {avg_tools:.1}");
    println!("  Topology usage   :");
    for (topo, count) in &topo_sorted {
        let pct = 100.0 * **count as f64 / total_hops.max(1) as f64;
        println!("    {:>16} : {:4}  ({pct:.1}%)", topo, count);
    }
    println!("====================================\n");
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_directives_roundtrip() {
        let text = "Let me look at the code.\n\
                    [TOOL: shell_exec(git diff HEAD)]\n\
                    [TOOL: file_read(src/core.py)]\n\
                    [TOOL: run_tests()]";
        let dirs = parse_tool_directives(text);
        assert_eq!(dirs.len(), 3);
        assert_eq!(
            dirs[0],
            ToolDirective::ShellExec("git diff HEAD".to_owned())
        );
        assert_eq!(dirs[1], ToolDirective::FileRead("src/core.py".to_owned()));
        assert_eq!(dirs[2], ToolDirective::RunTests);
    }

    #[test]
    fn extract_patch_last_block_wins() {
        let text = "[PATCH]\nfirst patch\n[/PATCH]\nRevised:\n[PATCH]\nsecond patch\n[/PATCH]";
        assert_eq!(extract_patch(text).as_deref(), Some("second patch"));
    }

    #[test]
    fn extract_patch_none_when_absent() {
        assert!(extract_patch("No patch here.").is_none());
    }

    #[test]
    fn turn_patch_ramp_shape() {
        // Ramp should be zero at turn 0 (progress=0) and peak somewhere mid-range.
        assert_eq!(turn_patch_ramp(0, 10), 0.0);
        // Peak at turn 5 of 10 (progress=0.5): 4*0.5*0.5 = 1.0
        assert!((turn_patch_ramp(5, 10) - 1.0).abs() < 1e-9);
        // Zero when max_turns is 0.
        assert_eq!(turn_patch_ramp(0, 0), 0.0);
    }

    #[test]
    fn synthetic_instances_count_and_difficulty() {
        let instances = synthetic_swe_instances(16);
        assert_eq!(instances.len(), 16);
        for inst in &instances {
            assert!((0.0..=1.0).contains(&inst.difficulty));
            assert!(!inst.fail_to_pass.is_empty());
        }
    }

    #[tokio::test]
    async fn mock_env_apply_and_test_lifecycle() {
        let mut env = MockSweBenchEnvironment::new(99);

        // Before patch: run_tests returns unresolved.
        let before = env.run_tests(&["tests::t1".to_string()]).await.unwrap();
        assert!(!before.resolved);

        // Apply a valid-looking patch.
        let patch = "--- a/core.py\n+++ b/core.py\n@@ -1,1 +1,2 @@\n+fix\n";
        let resp = env.apply_patch(patch).await.unwrap();
        assert!(resp.success);

        // After patch: run_tests should resolve.
        let after = env.run_tests(&["tests::t1".to_string()]).await.unwrap();
        assert!(after.resolved);
        assert_eq!(after.passed, 1);

        // Reset clears patch state.
        env.reset().await.unwrap();
        let post_reset = env.run_tests(&["tests::t1".to_string()]).await.unwrap();
        assert!(!post_reset.resolved);
    }

    #[tokio::test]
    async fn runner_produces_metrics_for_synthetic_instance() {
        let inst = synthetic_swe_instances(1).remove(0);
        let env = MockSweBenchEnvironment::new(7);
        let cfg = SweBenchRunnerConfig {
            max_turns: 8,
            verbose: false,
            ..Default::default()
        };

        let mut runner = SweBenchRunner::with_environment(cfg, env);
        let result = runner.run_instance(&inst).await.unwrap();

        assert_eq!(result.instance_id, inst.instance_id);
        assert!(result.metrics.turns_executed > 0);
        assert!(result.metrics.turns_executed <= 8);
        assert!(result.metrics.total_cost_usd > 0.0);
        assert!(result.metrics.total_latency_ms > 0.0);
        assert!(!result.metrics.topology_sequence.is_empty());
        // topology_sequence length == turns_executed
        assert_eq!(
            result.metrics.topology_sequence.len() as u32,
            result.metrics.turns_executed
        );
    }

    #[tokio::test]
    async fn dataset_run_preserves_ucb1_state_across_instances() {
        let instances = synthetic_swe_instances(4);
        let env = MockSweBenchEnvironment::new(11);
        let cfg = SweBenchRunnerConfig {
            max_turns: 5,
            ..Default::default()
        };

        let mut runner = SweBenchRunner::with_environment(cfg, env);
        let results = runner.run_dataset(&instances).await.unwrap();

        assert_eq!(results.len(), 4);

        // After 4 instances × up to 5 turns, UCB1 should have explored all 6
        // topology arms (the harness forces exploration of unvisited arms first).
        let selection_counts = runner.harness.selection_counts();
        let explored = selection_counts.values().filter(|&&c| c > 0).count();
        // At least 4 distinct topologies must have been selected.
        assert!(
            explored >= 4,
            "UCB1 should have explored ≥4 arms, got {explored}"
        );
    }
}
