use anyhow::{Context, Result, anyhow};
use crate::types::conversation::{ConversationState, Turn, TurnOutcome, TaskCategory};
use crate::types::self_improvement::*;
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
        let failure_rate = if sigma.turns.is_empty() { 0.0 } else {
            sigma.turns.iter().filter(|t| matches!(t.outcome, TurnOutcome::Rejected | TurnOutcome::RolledBack)).count() as f64 / sigma.turns.len() as f64
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
        scores.insert("category_weight".to_string(), match category {
            TaskCategory::CodeGeneration => 1.0,
            TaskCategory::Debugging => 0.9,
            TaskCategory::Testing => 0.9,
            TaskCategory::Research => 0.6,
            TaskCategory::Architecture | TaskCategory::Refactoring => 0.7,
            TaskCategory::General => 0.5,
        });
        scores
    }
}

pub struct PostMortemGenerator;

impl PostMortemGenerator {
    pub fn generate(sigma: &ConversationState) -> Option<PostMortem> {
        let failures: Vec<u32> = sigma.turns.iter()
            .filter(|t| matches!(t.outcome, TurnOutcome::Rejected | TurnOutcome::RolledBack))
            .map(|t| t.index)
            .collect();
        
        if failures.len() >= 3 {
            Some(PostMortem {
                session_id: sigma.session_id.clone(),
                failure_turn_indices: failures,
                root_cause: FailureCause::Unknown,
                missing_context: vec![],
                alternative_approaches: vec!["Increase context window".to_string()],
            })
        } else {
            None
        }
    }
}

pub struct ContinuousLearner<'a> {
    pub prompt_library: &'a mut Vec<crate::types::intelligence::PromptTemplate>,
    pub calibration: &'a mut Vec<CalibrationRecord>,
}

impl<'a> ContinuousLearner<'a> {
    pub fn run(&mut self, sigma: &ConversationState, mortem: Option<PostMortem>, base_perf: f64, current_perf: f64) {
        if let Some(pm) = mortem {
            let remediation = pm.alternative_approaches.join("; ");
            self.prompt_library.push(crate::types::intelligence::PromptTemplate {
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
        if !SafetyInterlock::is_modification_allowed(file_path) { return improvements; }
        for (i, line) in content.lines().enumerate() {
            if line.contains("TODO") || line.contains("FIXME") {
                improvements.push((format!("L{}", i+1), line.trim().to_string()));
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
        let content = std::fs::read_to_string(path).context("reading file for improvement scan")?;
        Ok(Self::identify_improvements(file_path, &content)
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
            return TrendReport { improving, degrading, stable };
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

        TrendReport { improving, degrading, stable }
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
        let var_c = control.iter().map(|x| (x - mean_c).powi(2)).sum::<f64>() / control.len() as f64;
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
    pub fn generate_variants(parent: &crate::types::intelligence::PromptTemplate) -> Vec<crate::types::intelligence::PromptTemplate> {
        let mutations: &[MutationEntry] = &[
            ("-m2", |c: &str| format!("{} [Concise]", c)),
            ("-m3", |c: &str| format!("{} [Detailed]", c)),
            ("-m4", |c: &str| format!("{} [Step-by-step]", c)),
            ("-m5", |c: &str| format!("{} [Contrarian]", c)),
        ];
        mutations
            .iter()
            .map(|(suffix, mutate)| crate::types::intelligence::PromptTemplate {
                id: format!("{}{}", parent.id, suffix),
                version: parent.version + 1,
                template_text: mutate(&parent.template_text),
                task_category: parent.task_category,
                variables: parent.variables.clone(),
                tags: parent.tags.clone(),
                performance_history: vec![],
            })
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
        Self { params: HashMap::new() }
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
    fn default() -> Self { Self::new() }
}

// ── ProgressReporter ─────────────────────────────────────────────────────────

pub struct ProgressReporter;

impl ProgressReporter {
    /// Build a progress report for a session. `turns_expected` is the budget cap.
    /// Remaining turns estimated via exponential-decay model:
    /// rate = -ln(1 - p) / turns_done; remaining = -ln(threshold) / rate - turns_done.
    pub fn report(sigma: &ConversationState, turns_expected: u32) -> crate::types::self_improvement::ProgressReport {
        let turns_done = sigma.turns.len() as u32;
        let p = sigma.completion_probability;
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

    pub fn is_modification_allowed(path: &str) -> bool {
        let normalized = path.replace('\\', "/");
        !Self::PROTECTED.iter().any(|&p| normalized == p || normalized.ends_with(&format!("/{p}")))
    }
}

pub struct FileWriter {
    pub root: PathBuf,
}

impl FileWriter {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn from_env() -> Result<Self> {
        let root = std::env::var("CROSSTALK_PROJECT_ROOT").unwrap_or_else(|_| ".".to_string());
        Ok(Self { root: PathBuf::from(root) })
    }

    pub async fn write_artifact_with_proof(&self, artifact: &crate::types::artifact::Artifact) -> Result<WriteOutcome> {
        let outcome = self.write_artifact(&artifact.name, &artifact.content).await?;
        if let WriteOutcome::Written(ref path) = outcome
            && let Some(proof) = artifact.proof_attachments.last()
        {
            let proof_path = path.with_extension(format!("{}.proof", path.extension().and_then(|e| e.to_str()).unwrap_or("txt")));
            let proof_json = serde_json::to_string_pretty(proof)?;
            tokio::fs::write(proof_path, proof_json).await?;
        }
        Ok(outcome)
    }

    pub async fn write_artifact(&self, name: &str, content: &str) -> Result<WriteOutcome> {
        if name.contains("..") || name.starts_with('/') || name.starts_with('\\') {
            return Ok(WriteOutcome::Skipped("path traversal rejected".to_string()));
        }
        let abs_path = self.root.join(name);
        let canonical_root = self.root.canonicalize().unwrap_or_else(|_| self.root.clone());
        if !abs_path.starts_with(&self.root) {
            return Ok(WriteOutcome::Skipped("path escapes project root".to_string()));
        }
        if !SafetyInterlock::is_modification_allowed(name) {
            return Ok(WriteOutcome::Skipped("SafetyInterlock: protected file".to_string()));
        }
        if let Some(parent) = abs_path.parent() {
            fs::create_dir_all(parent).await?;
            if let Ok(canonical_parent) = parent.canonicalize()
                && !canonical_parent.starts_with(&canonical_root)
            {
                crate::log_warn!(fs::remove_dir(parent).await, "Failed to remove directory after root escape detection");
                return Ok(WriteOutcome::Skipped("path escapes project root".to_string()));
            }
        }
        fs::write(&abs_path, content).await?;
        Ok(WriteOutcome::Written(abs_path))
    }
}
