use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use crate::types::self_improvement::*;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;

#[derive(Debug, Clone)]
pub enum WriteOutcome {
    Written(PathBuf),
    Skipped(String),
    VerificationFailed(String),
}

pub struct SelfImprovementEngine;

impl SelfImprovementEngine {
    pub fn evaluate_session(sigma: &ConversationState) -> SessionEvaluation {
        let mut metrics = BTreeMap::new();
        metrics.insert("convergence_p".to_string(), sigma.completion_probability);
        let failure_rate = if sigma.turns.is_empty() {
            0.0
        } else {
            sigma
                .turns
                .iter()
                .filter(|t| matches!(t.outcome, TurnOutcome::Rejected | TurnOutcome::RolledBack))
                .count() as f64
                / sigma.turns.len() as f64
        };
        metrics.insert("failure_rate".to_string(), failure_rate);

        SessionEvaluation {
            session_id: sigma.session_id.clone(),
            metrics,
            timestamp: ConversationState::now(),
        }
    }

    pub fn evaluate_turn(turn: &Turn, category: TaskCategory) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        let base = match turn.outcome {
            TurnOutcome::TestsPassed => 1.0,
            TurnOutcome::Compiled => 0.7,
            TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.1,
            _ => 0.5,
        };
        scores.insert("base_score".to_string(), base);
        scores.insert(
            "category_weight".to_string(),
            match category {
                TaskCategory::CodeGeneration => 1.0,
                TaskCategory::Debugging => 0.9,
                TaskCategory::Testing => 0.9,
                TaskCategory::Research => 0.6,
                TaskCategory::Architecture | TaskCategory::Refactoring => 0.7,
                TaskCategory::General => 0.5,
            },
        );
        scores
    }
}

pub struct PostMortemGenerator;

impl PostMortemGenerator {
    pub fn generate(sigma: &ConversationState) -> Option<PostMortem> {
        let failures: Vec<u32> = sigma
            .turns
            .iter()
            .filter(|t| {
                matches!(
                    t.outcome,
                    TurnOutcome::Rejected
                        | TurnOutcome::RolledBack
                        | TurnOutcome::VerificationFailed
                )
            })
            .map(|t| t.index)
            .collect();

        if failures.len() < 2 {
            return None;
        }

        let failed_turns: Vec<&Turn> = sigma
            .turns
            .iter()
            .filter(|t| {
                matches!(
                    t.outcome,
                    TurnOutcome::Rejected
                        | TurnOutcome::RolledBack
                        | TurnOutcome::VerificationFailed
                )
            })
            .collect();

        let root_cause = Self::diagnose_root_cause(&failed_turns, sigma);
        let missing_context = Self::identify_missing_context(&failed_turns, sigma);
        let alternative_approaches = Self::suggest_alternatives(&root_cause, sigma);

        Some(PostMortem {
            session_id: sigma.session_id.clone(),
            failure_turn_indices: failures,
            root_cause,
            missing_context,
            alternative_approaches,
        })
    }

    fn diagnose_root_cause(failed_turns: &[&Turn], sigma: &ConversationState) -> FailureCause {
        let total_turns = sigma.turns.len().max(1);
        let failure_rate = failed_turns.len() as f64 / total_turns as f64;

        // Check if a single agent is responsible for most failures
        let mut agent_failures: HashMap<&str, u32> = HashMap::new();
        for t in failed_turns {
            *agent_failures.entry(&t.model_id).or_default() += 1;
        }
        if let Some((_, &count)) = agent_failures.iter().max_by_key(|(_, c)| **c)
            && count as f64 / failed_turns.len() as f64 > 0.7
        {
            return FailureCause::AgentCapabilityLimit;
        }

        // Check for verification failures (type/syntax errors)
        let verification_fails = failed_turns
            .iter()
            .filter(|t| t.outcome == TurnOutcome::VerificationFailed)
            .count();
        if verification_fails as f64 / failed_turns.len().max(1) as f64 > 0.5 {
            return FailureCause::TypeMismatch;
        }

        // Check for increasing complexity (artifact sizes growing but quality dropping)
        if sigma
            .artifacts
            .values()
            .any(|a| a.metrics.cyclomatic_complexity > 50)
        {
            return FailureCause::ComplexityExceeded;
        }

        // High failure rate with low convergence suggests missing context
        if failure_rate > 0.4 && sigma.completion_probability < 0.3 {
            return FailureCause::MissingContext;
        }

        // Budget exhaustion
        if sigma.budget.mode() == crate::types::compute::BudgetMode::Emergency {
            return FailureCause::InsufficientBudget;
        }

        FailureCause::Unknown
    }

    fn identify_missing_context(failed_turns: &[&Turn], sigma: &ConversationState) -> Vec<String> {
        let mut missing = Vec::new();

        // Check if failures correlate with no artifacts loaded
        if sigma.artifacts.is_empty() {
            missing.push("No workspace artifacts loaded; agents may lack code context".to_string());
        }

        // Check if the initial task is vague
        if let Some(first) = sigma.turns.first()
            && first.content.split_whitespace().count() < 20
        {
            missing
                .push("Task description is very brief; consider providing more detail".to_string());
        }

        // Check for repeated similar failures
        let mut seen_errors: HashMap<String, u32> = HashMap::new();
        for t in failed_turns {
            let key = t.content.chars().take(100).collect::<String>();
            *seen_errors.entry(key).or_default() += 1;
        }
        if seen_errors.values().any(|&c| c >= 3) {
            missing.push(
                "Agents are repeating similar failures; the task may need decomposition"
                    .to_string(),
            );
        }

        missing
    }

    fn suggest_alternatives(cause: &FailureCause, sigma: &ConversationState) -> Vec<String> {
        let mut suggestions = Vec::new();
        match cause {
            FailureCause::TypeMismatch => {
                suggestions
                    .push("Enable linter feedback loop to catch type errors earlier".to_string());
                suggestions
                    .push("Use agents with stronger typed-language capabilities".to_string());
            }
            FailureCause::MissingContext => {
                suggestions.push(
                    "Load additional workspace files with --files or --workspace".to_string(),
                );
                suggestions.push("Provide more detailed task description".to_string());
            }
            FailureCause::AgentCapabilityLimit => {
                suggestions.push("Add a stronger model to the agent pool".to_string());
                if sigma
                    .turns
                    .iter()
                    .filter(|t| t.model_id != "User")
                    .map(|t| &t.model_id)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    < 2
                {
                    suggestions
                        .push("Use multiple diverse agents for cross-validation".to_string());
                }
            }
            FailureCause::ComplexityExceeded => {
                suggestions.push("Decompose the task into smaller sub-tasks".to_string());
                suggestions.push(
                    "Reduce artifact complexity before attempting further changes".to_string(),
                );
            }
            FailureCause::InsufficientBudget => {
                suggestions
                    .push("Increase budget limit or use more cost-effective models".to_string());
            }
            FailureCause::Unknown => {
                suggestions.push("Try a different topology (e.g. mediated debate)".to_string());
                if sigma.iteration_index > 10 {
                    suggestions.push(
                        "Session may be stuck; consider restarting with refined task".to_string(),
                    );
                }
            }
        }
        suggestions
    }
}

pub struct ContinuousLearner<'a> {
    pub prompt_library: &'a mut Vec<crate::types::intelligence::PromptTemplate>,
    pub calibration: &'a mut Vec<CalibrationRecord>,
}

impl<'a> ContinuousLearner<'a> {
    pub fn run(
        &mut self,
        sigma: &ConversationState,
        mortem: Option<PostMortem>,
        base_perf: f64,
        current_perf: f64,
    ) {
        if let Some(pm) = mortem {
            let remediation = pm.alternative_approaches.join("; ");
            self.prompt_library
                .push(crate::types::intelligence::PromptTemplate {
                    id: format!("remediation-{}", pm.session_id),
                    version: 1,
                    template_text: format!("[Learned correction] {}", remediation),
                    task_category: TaskCategory::General,
                    variables: vec![],
                    tags: vec!["general".to_string()],
                    performance_history: vec![],
                });
        }
        self.calibration.push(CalibrationRecord {
            session_id: sigma.session_id.clone(),
            predicted_difficulty: 0.5,
            actual_difficulty: 0.5,
            predicted_outcome: base_perf,
            actual_outcome: current_perf,
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementProposal {
    pub file_path: String,
    pub line: String,
    pub description: String,
}

pub struct SelfCodeModifier;

impl SelfCodeModifier {
    pub fn identify_improvements(file_path: &str, content: &str) -> Vec<(String, String)> {
        let mut improvements = Vec::new();
        if !SafetyInterlock::is_modification_allowed(file_path) {
            return improvements;
        }
        for (i, line) in content.lines().enumerate() {
            if line.contains("TODO") || line.contains("FIXME") {
                improvements.push((format!("L{}", i + 1), line.trim().to_string()));
            }
        }
        improvements
    }

    pub fn propose_improvement(file_path: &str, current_content: &str) -> Result<String> {
        if !SafetyInterlock::is_modification_allowed(file_path) {
            return Err(anyhow!("Protected file: {}", file_path));
        }
        let old_panic = ".unwrap_or_else(|_| panic!())";
        if current_content.contains(old_panic) {
            Ok(current_content.replace(old_panic, ".unwrap()"))
        } else {
            Err(anyhow!("No improvements found"))
        }
    }

    pub fn check_file(file_path: &str) -> Result<Vec<ImprovementProposal>> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(anyhow!("File not found: {}", file_path)).context("check_file");
        }
        let canonical = path.canonicalize().context("check_file: canonicalize")?;
        let canonical_str = canonical
            .to_str()
            .ok_or_else(|| anyhow!("check_file: path is not valid UTF-8"))?;
        let content =
            std::fs::read_to_string(&canonical).context("reading file for improvement scan")?;
        Ok(Self::identify_improvements(canonical_str, &content)
            .into_iter()
            .map(|(line, desc)| ImprovementProposal {
                file_path: file_path.to_string(),
                line,
                description: desc,
            })
            .collect())
    }

    pub fn run_format_check(workspace_root: &str) -> Result<bool> {
        let output = Command::new("cargo")
            .args(["fmt", "--check"])
            .current_dir(workspace_root)
            .output()
            .context("running cargo fmt --check")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                tracing::warn!(workspace = %workspace_root, %stderr, "cargo fmt --check failed");
            }
        }
        Ok(output.status.success())
    }
}

// ── SelfEvaluationTrendAnalyzer ──────────────────────────────────────────────

pub struct TrendReport {
    pub improving: Vec<String>,
    pub degrading: Vec<String>,
    pub stable: Vec<String>,
}

pub struct SelfEvaluationTrendAnalyzer;

impl SelfEvaluationTrendAnalyzer {
    /// Analyze a sequence of session evaluations and classify each metric as
    /// improving, degrading, or stable based on first-vs-last comparison.
    pub fn analyze(evals: &[SessionEvaluation]) -> TrendReport {
        let mut improving = Vec::new();
        let mut degrading = Vec::new();
        let mut stable = Vec::new();

        if evals.len() < 2 {
            return TrendReport {
                improving,
                degrading,
                stable,
            };
        }

        // Collect all metric names across evaluations.
        let mut metric_names = std::collections::BTreeSet::new();
        for e in evals {
            for k in e.metrics.keys() {
                metric_names.insert(k.clone());
            }
        }

        for name in metric_names {
            let first = evals.first().and_then(|e| e.metrics.get(&name).copied());
            let last = evals.last().and_then(|e| e.metrics.get(&name).copied());
            match (first, last) {
                (Some(f), Some(l)) if l > f + 0.01 => improving.push(name),
                (Some(f), Some(l)) if l < f - 0.01 => degrading.push(name),
                _ => stable.push(name),
            }
        }

        TrendReport {
            improving,
            degrading,
            stable,
        }
    }
}

// ── AbTestManager ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestReport {
    pub hypothesis_id: String,
    pub control_mean: f64,
    pub test_mean: f64,
    pub effect_size: f64,
    pub significant: bool,
    pub adopted: bool,
    pub confidence_interval: (f64, f64),
}

pub struct AbTestManager;

impl AbTestManager {
    /// Simple significance check: requires n >= 10 per arm and a sufficiently
    /// large effect size (Cohen's d >= 0.5) relative to pooled standard
    /// deviation.
    pub fn check_significance(control: &[f64], test: &[f64]) -> bool {
        let n_min = 10;
        if control.len() < n_min || test.len() < n_min {
            return false;
        }
        let mean_c = control.iter().sum::<f64>() / control.len() as f64;
        let mean_t = test.iter().sum::<f64>() / test.len() as f64;
        let var_c =
            control.iter().map(|x| (x - mean_c).powi(2)).sum::<f64>() / control.len() as f64;
        let var_t = test.iter().map(|x| (x - mean_t).powi(2)).sum::<f64>() / test.len() as f64;
        let pooled_sd = ((var_c + var_t) / 2.0).sqrt();
        if pooled_sd < f64::EPSILON {
            return (mean_t - mean_c).abs() > f64::EPSILON;
        }
        let d = (mean_t - mean_c).abs() / pooled_sd;
        d >= 0.5
    }
}

// ── PromptEvolutionaryOptimizer ──────────────────────────────────────────────

type MutationEntry = (&'static str, fn(&str) -> String);

pub struct PromptEvolutionaryOptimizer;

impl PromptEvolutionaryOptimizer {
    /// Produce 4 mutated variants of a parent prompt template.
    /// Each variant gets an id suffixed with `-m{i}` and an incremented version.
    pub fn generate_variants(
        parent: &crate::types::intelligence::PromptTemplate,
    ) -> Vec<crate::types::intelligence::PromptTemplate> {
        let mutations: &[MutationEntry] = &[
            ("-m2", |c: &str| format!("{} [Concise]", c)),
            ("-m3", |c: &str| format!("{} [Detailed]", c)),
            ("-m4", |c: &str| format!("{} [Step-by-step]", c)),
            ("-m5", |c: &str| format!("{} [Contrarian]", c)),
        ];
        mutations
            .iter()
            .map(
                |(suffix, mutate)| crate::types::intelligence::PromptTemplate {
                    id: format!("{}{}", parent.id, suffix),
                    version: parent.version + 1,
                    template_text: mutate(&parent.template_text),
                    task_category: parent.task_category,
                    variables: parent.variables.clone(),
                    tags: parent.tags.clone(),
                    performance_history: vec![],
                },
            )
            .collect()
    }
}

// ── CalibrationAdjuster ──────────────────────────────────────────────────────

pub struct CalibrationAdjuster;

impl CalibrationAdjuster {
    /// Fit Platt scaling parameters (a, b) such that calibrated = 1/(1+exp(-(a*x + b))).
    /// Returns (1.0, 0.0) (identity in logit space) when no data is provided.
    pub fn fit_platt(records: &[CalibrationRecord]) -> (f64, f64) {
        if records.is_empty() {
            return (1.0, 0.0);
        }
        // Simple linear regression in logit space: logit(actual) ~ a * predicted + b
        // For numerical safety, clamp actuals away from 0 and 1.
        let eps = 1e-6;
        let n = records.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        for r in records {
            let x = r.predicted_outcome;
            let clamped = r.actual_outcome.clamp(eps, 1.0 - eps);
            let y = (clamped / (1.0 - clamped)).ln(); // logit
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < eps {
            return (1.0, 0.0);
        }
        let a = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - a * sum_x) / n;
        (a, b)
    }

    /// Apply Platt scaling: sigmoid(a * x + b).
    pub fn apply(x: f64, a: f64, b: f64) -> f64 {
        1.0 / (1.0 + (-(a * x + b)).exp())
    }
}

// ── RuntimeParameterAdjuster ─────────────────────────────────────────────────

pub struct RuntimeParameterAdjuster {
    params: HashMap<String, f64>,
}

impl RuntimeParameterAdjuster {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Apply a parameter change only when the A/B test report is significant
    /// and the test arm was adopted. Returns `true` if the change was applied.
    pub fn apply_if_significant(
        &mut self,
        name: &str,
        value: f64,
        report: &AbTestReport,
        _rationale: &str,
    ) -> bool {
        if report.significant && report.adopted {
            self.params.insert(name.to_string(), value);
            true
        } else {
            false
        }
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.params.get(name).copied()
    }
}

impl Default for RuntimeParameterAdjuster {
    fn default() -> Self {
        Self::new()
    }
}

// ── ProgressReporter ─────────────────────────────────────────────────────────

pub struct ProgressReporter;

impl ProgressReporter {
    /// Build a progress report for a session. `turns_expected` is the budget cap.
    /// Remaining turns estimated via exponential-decay model:
    /// rate = -ln(1 - p) / turns_done; remaining = -ln(threshold) / rate - turns_done.
    pub fn report(
        sigma: &ConversationState,
        turns_expected: u32,
    ) -> crate::types::self_improvement::ProgressReport {
        let turns_done = sigma.turns.len() as u32;
        let p = sigma
            .completion_probability
            .clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        let threshold = 0.999; // target probability
        let estimated = if turns_done > 0 && p > 0.0 && p < 1.0 {
            let rate = -(1.0 - p).ln() / turns_done as f64;
            if rate > f64::EPSILON {
                let total = -(1.0_f64 - threshold).ln() / rate;
                let remaining = (total - turns_done as f64).max(0.0).ceil() as u32;
                Some(remaining)
            } else {
                None
            }
        } else {
            None
        };

        crate::types::self_improvement::ProgressReport {
            session_id: sigma.session_id.clone(),
            turns_completed: turns_done,
            turns_expected,
            completion_probability: p,
            estimated_turns_remaining: estimated,
            success_probability: p,
        }
    }
}

pub struct SafetyInterlock;
impl SafetyInterlock {
    const PROTECTED: &[&str] = &[
        "src/core/orchestrator.rs",
        "src/engines/security.rs",
        "src/engines/verification.rs",
        "Cargo.toml",
        "Cargo.lock",
    ];

    /// Directory prefixes that must never be written into, regardless of filename.
    const PROTECTED_DIRS: &[&str] = &[
        ".git/", ".cargo/", ".config/", ".ssh/", ".gnupg/", ".github/",
    ];

    /// Files that must never be written.
    const PROTECTED_FILES: &[&str] = &[
        ".env",
        ".env.local",
        ".env.production",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    ];

    pub fn is_modification_allowed(path: &str) -> bool {
        let normalized = path.replace('\\', "/");
        if Self::PROTECTED
            .iter()
            .any(|&p| normalized == p || normalized.ends_with(&format!("/{p}")))
        {
            return false;
        }
        if Self::PROTECTED_DIRS
            .iter()
            .any(|&d| normalized.starts_with(d) || normalized.contains(&format!("/{d}")))
        {
            return false;
        }
        if Self::PROTECTED_FILES
            .iter()
            .any(|&f| normalized == f || normalized.ends_with(&format!("/{f}")))
        {
            return false;
        }
        true
    }
}

pub struct FileWriter {
    pub root: PathBuf,
    canonical_root: PathBuf,
}

impl FileWriter {
    pub fn new(root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root).context("creating FileWriter root")?;
        let canonical_root = root
            .canonicalize()
            .context("canonicalizing FileWriter root")?;
        Ok(Self {
            root,
            canonical_root,
        })
    }

    pub fn from_env() -> Result<Self> {
        let root = std::env::var("CROSSTALK_PROJECT_ROOT").unwrap_or_else(|_| ".".to_string());
        Self::new(PathBuf::from(root))
    }

    pub async fn write_artifact_with_proof(
        &self,
        artifact: &crate::types::artifact::Artifact,
    ) -> Result<WriteOutcome> {
        let outcome = self
            .write_artifact(&artifact.name, &artifact.content)
            .await?;
        if let WriteOutcome::Written(ref path) = outcome
            && let Some(proof) = artifact.proof_attachments.last()
        {
            let proof_path = path.with_extension(format!(
                "{}.proof",
                path.extension().and_then(|e| e.to_str()).unwrap_or("txt")
            ));
            let proof_json = serde_json::to_string_pretty(proof)?;
            tokio::fs::write(proof_path, proof_json).await?;
        }
        Ok(outcome)
    }

    pub async fn write_artifact(&self, name: &str, content: &str) -> Result<WriteOutcome> {
        if name.contains("..") || name.starts_with('/') || name.starts_with('\\') {
            return Ok(WriteOutcome::Skipped("path traversal rejected".to_string()));
        }
        if !SafetyInterlock::is_modification_allowed(name) {
            return Ok(WriteOutcome::Skipped(
                "SafetyInterlock: protected file".to_string(),
            ));
        }
        let abs_path = self.root.join(name);
        if let Some(parent) = abs_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        // Canonicalize the target's parent after create_dir_all; symlinks are
        // resolved against the pre-resolved canonical_root stored at construction.
        // Fail closed: if canonicalize fails (permission denied, symlink loop, etc.)
        // we reject the write rather than falling back to the non-canonical path.
        let canonical_parent = match abs_path.parent() {
            Some(p) => p.canonicalize().map_err(|e| {
                anyhow::anyhow!(
                    "failed to canonicalize parent of '{}': {}",
                    abs_path.display(),
                    e
                )
            })?,
            None => {
                return Ok(WriteOutcome::Skipped(
                    "artifact path has no parent directory".to_string(),
                ));
            }
        };
        let canonical_abs = canonical_parent.join(abs_path.file_name().unwrap_or_default());
        if !canonical_abs.starts_with(&self.canonical_root) {
            tracing::warn!(
                path = %abs_path.display(),
                root = %self.canonical_root.display(),
                "Artifact write blocked: path escapes project root"
            );
            return Ok(WriteOutcome::Skipped(
                "path escapes project root".to_string(),
            ));
        }
        let tmp_path = abs_path.with_extension(format!(
            "{}.tmp",
            abs_path.extension().unwrap_or_default().to_string_lossy()
        ));
        fs::write(&tmp_path, content).await?;
        fs::rename(&tmp_path, &abs_path).await?;
        Ok(WriteOutcome::Written(abs_path))
    }
}
