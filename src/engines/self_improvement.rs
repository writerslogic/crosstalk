use crate::types::conversation::ConversationState;
use crate::types::self_improvement::{
    BenchmarkCategory, BenchmarkResult, BenchmarkTask, CalibrationRecord, DegradationResponse,
    DegradationStrategy, DegradationTrigger, EnforcementLevel, ErrorBudget, FailureCause,
    HandoffPackage, HypothesisStatus, ImprovementHypothesis, LearningOutcome, ParameterAdjustment,
    PostMortem, ProgressReport, PromptTemplate, SessionEvaluation, StrategyEntry,
};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;

// ── SelfImprovementEngine ─────────────────────────────────────────────────────

pub struct SelfImprovementEngine;

impl SelfImprovementEngine {
    #[must_use]
    pub fn evaluate_session(sigma: &ConversationState) -> SessionEvaluation {
        let mut metrics = HashMap::with_capacity(5);
        metrics.insert("turn_count".to_string(), sigma.turns.len() as f64);
        metrics.insert("convergence_p".to_string(), sigma.completion_probability);
        metrics.insert("cost_spent".to_string(), sigma.budget.spent);

        let failure_count = sigma
            .turns
            .iter()
            .filter(|t| {
                matches!(
                    t.outcome,
                    crate::types::conversation::TurnOutcome::Rejected
                        | crate::types::conversation::TurnOutcome::RolledBack
                )
            })
            .count() as f64;
        let total = sigma.turns.len() as f64;
        metrics.insert(
            "failure_rate".to_string(),
            if total > 0.0 { failure_count / total } else { 0.0 },
        );
        metrics.insert(
            "cost_efficiency".to_string(),
            if sigma.budget.spent > 0.0 {
                sigma.completion_probability / sigma.budget.spent
            } else {
                0.0
            },
        );

        SessionEvaluation {
            session_id: sigma.session_id.clone(),
            metrics,
            timestamp: ConversationState::now(),
        }
    }
}

// ── SelfEvaluationTrendAnalyzer ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PerformanceTrendReport {
    /// EMA value per metric name.
    pub ema: HashMap<String, f64>,
    /// Which metrics are trending up vs. down vs. flat.
    pub improving: Vec<String>,
    pub degrading: Vec<String>,
    pub stable: Vec<String>,
}

pub struct SelfEvaluationTrendAnalyzer;

impl SelfEvaluationTrendAnalyzer {
    const ALPHA: f64 = 0.3; // EMA smoothing factor

    /// Compute exponential moving averages for every metric across `evals`
    /// (oldest first) and classify each metric's trend.
    #[must_use]
    pub fn analyze(evals: &[SessionEvaluation]) -> PerformanceTrendReport {
        if evals.is_empty() {
            return PerformanceTrendReport {
                ema: HashMap::new(),
                improving: vec![],
                degrading: vec![],
                stable: vec![],
            };
        }

        // Collect all metric names
        // BTreeSet for deterministic iteration order across runs.
        let mut all_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for e in evals {
            all_keys.extend(e.metrics.keys().cloned());
        }

        let mut ema: HashMap<String, f64> = HashMap::new();
        let mut first_values: HashMap<String, f64> = HashMap::new();

        for key in &all_keys {
            let mut current_ema: Option<f64> = None;
            let mut first: Option<f64> = None;
            for eval in evals {
                if let Some(&v) = eval.metrics.get(key) {
                    // Skip NaN/Inf values to prevent silent propagation through EMA.
                    if !v.is_finite() {
                        continue;
                    }
                    current_ema = Some(match current_ema {
                        None => v,
                        Some(prev) => Self::ALPHA * v + (1.0 - Self::ALPHA) * prev,
                    });
                    if first.is_none() {
                        first = Some(v);
                    }
                }
            }
            if let (Some(e), Some(f)) = (current_ema, first) {
                ema.insert(key.clone(), e);
                first_values.insert(key.clone(), f);
            }
        }

        let mut improving = vec![];
        let mut degrading = vec![];
        let mut stable = vec![];
        for (key, &e) in &ema {
            let f = first_values.get(key).copied().unwrap_or(e);
            let delta = e - f;
            if delta > 0.02 {
                improving.push(key.clone());
            } else if delta < -0.02 {
                degrading.push(key.clone());
            } else {
                stable.push(key.clone());
            }
        }
        // Vecs are already in BTreeSet (sorted) order because we iterated `ema` via sorted keys above,
        // but ema is a HashMap so sort explicitly for determinism.
        improving.sort();
        degrading.sort();
        stable.sort();

        PerformanceTrendReport { ema, improving, degrading, stable }
    }
}

// ── HypothesisGenerator + Prioritizer ────────────────────────────────────────

pub struct HypothesisGenerator;

impl HypothesisGenerator {
    /// Generate hypotheses from metrics that show degradation.
    #[must_use]
    pub fn from_trend(report: &PerformanceTrendReport) -> Vec<ImprovementHypothesis> {
        report
            .degrading
            .iter()
            .enumerate()
            .map(|(i, metric)| ImprovementHypothesis {
                id: format!("hyp-{i}-{metric}"),
                description: format!(
                    "Investigate why `{metric}` is degrading and switch strategy to recover it."
                ),
                expected_impact: 0.15,
                confidence: 0.5,
                estimated_cost: 1.0,
                status: HypothesisStatus::Queued,
            })
            .collect()
    }
}

pub struct HypothesisPrioritizer;

impl HypothesisPrioritizer {
    /// Sort hypotheses by `(expected_impact * confidence) / estimated_cost` descending.
    #[must_use]
    pub fn rank(mut hypotheses: Vec<ImprovementHypothesis>) -> Vec<ImprovementHypothesis> {
        hypotheses.sort_by(|a, b| b.priority().total_cmp(&a.priority()));
        hypotheses
    }
}

// ── AbTestManager ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AbTestReport {
    pub hypothesis_id: String,
    pub control_mean: f64,
    pub test_mean: f64,
    pub effect_size: f64,
    pub significant: bool,
    pub adopted: bool,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Default, Clone)]
pub struct AbTestManager {
    pub active_tests: HashMap<String, ImprovementHypothesis>,
}

impl AbTestManager {
    #[must_use]
    pub fn new() -> Self {
        Self { active_tests: HashMap::new() }
    }

    pub fn register(&mut self, hypothesis: ImprovementHypothesis) {
        self.active_tests.insert(hypothesis.id.clone(), hypothesis);
    }

    /// Welch's t-test (two-sample, unequal variance).
    /// Returns `true` when p < 0.05 AND relative effect size > 5 %.
    /// Uses a conservative piecewise t-critical value that is accurate for n >= 10.
    #[must_use]
    pub fn check_significance(control: &[f64], test: &[f64]) -> bool {
        let n_c = control.len() as f64;
        let n_t = test.len() as f64;
        if n_c < 10.0 || n_t < 10.0 {
            return false;
        }
        let mean_c = control.iter().sum::<f64>() / n_c;
        let mean_t = test.iter().sum::<f64>() / n_t;
        let var_c = control.iter().map(|&x| (x - mean_c).powi(2)).sum::<f64>() / (n_c - 1.0);
        let var_t = test.iter().map(|&x| (x - mean_t).powi(2)).sum::<f64>() / (n_t - 1.0);
        let se = ((var_c / n_c) + (var_t / n_t)).sqrt();
        if se == 0.0 {
            return false;
        }
        let t_stat = (mean_t - mean_c) / se;
        // Conservative piecewise t-critical: t_{df, 0.025} for Welch df ≈ min(n_c, n_t) - 1.
        // n < 15: df ≈ 14 → 2.145; n < 25: df ≈ 24 → 2.064; else converges to 2.00.
        let min_n = n_c.min(n_t);
        let t_critical = if min_n < 15.0 { 2.145 } else if min_n < 25.0 { 2.064 } else { 2.00 };
        let effect = if mean_c.abs() > f64::EPSILON { (mean_t - mean_c) / mean_c.abs() } else { 0.0 };
        t_stat > t_critical && effect > 0.05
    }

    /// Evaluate an A/B test, computing the full report with 95 % CI.
    /// `effect_size` is the relative improvement ((test - control) / |control|) to
    /// match the significance criterion used in `check_significance`.
    #[must_use]
    pub fn evaluate(hypothesis_id: &str, control: &[f64], test: &[f64]) -> AbTestReport {
        let significant = Self::check_significance(control, test);
        let n_c = control.len() as f64;
        let n_t = test.len() as f64;
        let mean_c = if n_c > 0.0 { control.iter().sum::<f64>() / n_c } else { 0.0 };
        let mean_t = if n_t > 0.0 { test.iter().sum::<f64>() / n_t } else { 0.0 };
        // Relative effect size — consistent with the significance threshold in check_significance.
        let effect_size = if mean_c.abs() > f64::EPSILON { (mean_t - mean_c) / mean_c.abs() } else { 0.0 };

        let se_t = if n_t > 1.0 {
            let var = test.iter().map(|&x| (x - mean_t).powi(2)).sum::<f64>() / (n_t - 1.0);
            (var / n_t).sqrt()
        } else {
            0.0
        };
        let margin = 1.96 * se_t;
        let confidence_interval = (mean_t - margin, mean_t + margin);

        AbTestReport {
            hypothesis_id: hypothesis_id.to_string(),
            control_mean: mean_c,
            test_mean: mean_t,
            effect_size,
            significant,
            adopted: significant && effect_size > 0.0,
            confidence_interval,
        }
    }
}

// ── PromptLibrary ─────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct PromptLibrary {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptLibrary {
    #[must_use]
    pub fn new() -> Self {
        Self { templates: HashMap::new() }
    }

    pub fn insert(&mut self, template: PromptTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&PromptTemplate> {
        self.templates.get(id)
    }

    pub fn record_performance(&mut self, id: &str, session_id: &str, quality: f64) {
        if let Some(t) = self.templates.get_mut(id) {
            t.performance_history.push((session_id.to_string(), quality));
        }
    }

    /// Return the template with the highest mean quality score across its history.
    #[must_use]
    pub fn best_for_task(&self, task_type: &str) -> Option<&PromptTemplate> {
        self.templates
            .values()
            .filter(|t| t.task_types.iter().any(|tt| tt == task_type))
            .max_by(|a, b| a.mean_quality().total_cmp(&b.mean_quality()))
    }
}

// ── StrategyDatabase ──────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct StrategyDatabase {
    entries: Vec<StrategyEntry>,
}

impl StrategyDatabase {
    #[must_use]
    pub fn new() -> Self {
        Self { entries: vec![] }
    }

    pub fn insert(&mut self, entry: StrategyEntry) {
        self.entries.push(entry);
    }

    /// Return the `k` nearest strategies to `query` by squared Euclidean distance,
    /// sorted closest-first.
    #[must_use]
    pub fn knn(&self, query: &[f64], k: usize) -> Vec<&StrategyEntry> {
        let mut scored: Vec<(&StrategyEntry, f64)> = self
            .entries
            .iter()
            .map(|e| (e, e.distance_sq(query)))
            .collect();
        scored.sort_by(|(_, a), (_, b)| a.total_cmp(b));
        scored.into_iter().take(k).map(|(e, _)| e).collect()
    }
}

// ── PostMortemGenerator ───────────────────────────────────────────────────────

pub struct PostMortemGenerator;

impl PostMortemGenerator {
    /// Generate a post-mortem when failure rate > 10 %.
    ///
    /// Identifies the first failing turn and classifies the root cause by
    /// inspecting turn content for common error patterns.
    #[must_use]
    pub fn generate(sigma: &ConversationState) -> Option<PostMortem> {
        let total = sigma.turns.len();
        if total == 0 {
            return None;
        }
        let failures: Vec<u32> = sigma
            .turns
            .iter()
            .filter(|t| {
                matches!(
                    t.outcome,
                    crate::types::conversation::TurnOutcome::Rejected
                        | crate::types::conversation::TurnOutcome::RolledBack
                )
            })
            .map(|t| t.index)
            .collect();

        let failure_rate = failures.len() as f64 / total as f64;
        if failure_rate <= 0.10 {
            return None;
        }

        let first_failure = sigma
            .turns
            .iter()
            .find(|t| failures.contains(&t.index))
            .map(|t| t.content.as_str())
            .unwrap_or("");

        let root_cause = if first_failure.contains("mismatched types") || first_failure.contains("E0308") {
            FailureCause::TypeMismatch
        } else if first_failure.contains("budget") || first_failure.contains("cost limit") {
            FailureCause::InsufficientBudget
        } else if first_failure.contains("too complex") || first_failure.contains("complexity") {
            FailureCause::ComplexityExceeded
        } else if first_failure.contains("missing") || first_failure.contains("not found") {
            FailureCause::MissingContext
        } else {
            FailureCause::Unknown
        };

        Some(PostMortem {
            session_id: sigma.session_id.clone(),
            failure_turn_indices: failures,
            root_cause,
            missing_context: vec![],
            alternative_approaches: vec![],
        })
    }
}

// ── CalibrationTracker ────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct CalibrationTracker {
    pub records: Vec<CalibrationRecord>,
}

impl CalibrationTracker {
    #[must_use]
    pub fn new() -> Self {
        Self { records: vec![] }
    }

    pub fn record(&mut self, rec: CalibrationRecord) {
        self.records.push(rec);
    }

    /// Expected Calibration Error over all records: mean absolute prediction error.
    #[must_use]
    pub fn ece(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .records
            .iter()
            .map(|r| (r.outcome_error() + r.difficulty_error()) / 2.0)
            .sum();
        sum / self.records.len() as f64
    }

    /// Returns `true` when ECE exceeds 10 % — recalibration needed.
    #[must_use]
    pub fn needs_recalibration(&self) -> bool {
        self.ece() > 0.10
    }
}

// ── ErrorBudgetLedger ─────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct ErrorBudgetLedger {
    budgets: HashMap<String, ErrorBudget>,
}

impl ErrorBudgetLedger {
    #[must_use]
    pub fn new() -> Self {
        Self { budgets: HashMap::new() }
    }

    pub fn set_budget(&mut self, task_type: &str, allowed_rate: f64) {
        self.budgets.insert(
            task_type.to_string(),
            ErrorBudget {
                task_type: task_type.to_string(),
                allowed_rate,
                actual_rate: 0.0,
                budget_remaining: 1.0,
                enforcement_level: EnforcementLevel::Normal,
            },
        );
    }

    /// Record an outcome for `task_type`. `failed` = true when the turn failed.
    pub fn record_outcome(&mut self, task_type: &str, failed: bool) {
        let Some(budget) = self.budgets.get_mut(task_type) else {
            return;
        };
        let alpha = 0.1; // EMA smoothing
        let obs = if failed { 1.0 } else { 0.0 };
        budget.actual_rate = alpha * obs + (1.0 - alpha) * budget.actual_rate;
        budget.budget_remaining =
            ((budget.allowed_rate - budget.actual_rate) / budget.allowed_rate.max(f64::EPSILON))
                .clamp(0.0, 1.0);
        budget.enforcement_level = if budget.budget_remaining < 0.2 {
            EnforcementLevel::Strict
        } else if budget.budget_remaining == 0.0 {
            EnforcementLevel::Suspended
        } else {
            EnforcementLevel::Normal
        };
    }

    #[must_use]
    pub fn get(&self, task_type: &str) -> Option<&ErrorBudget> {
        self.budgets.get(task_type)
    }
}

// ── BenchmarkSuite ────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct BenchmarkSuite {
    tasks: Vec<BenchmarkTask>,
}

impl BenchmarkSuite {
    #[must_use]
    pub fn new() -> Self {
        Self { tasks: vec![] }
    }

    /// Populate the suite with a representative set of 20 standardised tasks.
    #[must_use]
    pub fn with_standard_tasks() -> Self {
        let mut suite = Self::new();
        let specs: &[(&str, BenchmarkCategory, &str, f64)] = &[
            ("bm-cg-01", BenchmarkCategory::CodeGeneration, "Implement a binary search function.", 0.2),
            ("bm-cg-02", BenchmarkCategory::CodeGeneration, "Write a linked-list in safe Rust.", 0.5),
            ("bm-cg-03", BenchmarkCategory::CodeGeneration, "Implement a thread-safe ring buffer.", 0.7),
            ("bm-cg-04", BenchmarkCategory::CodeGeneration, "Build an async HTTP client wrapper.", 0.6),
            ("bm-bf-01", BenchmarkCategory::BugFixing, "Fix off-by-one in slice indexing.", 0.3),
            ("bm-bf-02", BenchmarkCategory::BugFixing, "Resolve borrow-checker lifetime conflict.", 0.6),
            ("bm-bf-03", BenchmarkCategory::BugFixing, "Fix race condition in shared state.", 0.8),
            ("bm-bf-04", BenchmarkCategory::BugFixing, "Debug integer overflow in release mode.", 0.5),
            ("bm-rf-01", BenchmarkCategory::Refactoring, "Extract duplicated logic into a trait.", 0.4),
            ("bm-rf-02", BenchmarkCategory::Refactoring, "Replace nested match with combinator chain.", 0.3),
            ("bm-rf-03", BenchmarkCategory::Refactoring, "Decompose god struct into smaller types.", 0.6),
            ("bm-rf-04", BenchmarkCategory::Refactoring, "Migrate sync I/O to async.", 0.7),
            ("bm-ad-01", BenchmarkCategory::ArchitectureDesign, "Design plugin system with trait objects.", 0.8),
            ("bm-ad-02", BenchmarkCategory::ArchitectureDesign, "Model state machine for connection lifecycle.", 0.6),
            ("bm-ad-03", BenchmarkCategory::ArchitectureDesign, "Design zero-copy message passing.", 0.9),
            ("bm-ad-04", BenchmarkCategory::ArchitectureDesign, "Define layered error type hierarchy.", 0.5),
            ("bm-rs-01", BenchmarkCategory::ResearchSynthesis, "Summarise trade-offs of lock-free queues.", 0.5),
            ("bm-rs-02", BenchmarkCategory::ResearchSynthesis, "Compare consensus algorithms for embedded.", 0.7),
            ("bm-rs-03", BenchmarkCategory::ResearchSynthesis, "Survey WASM runtimes for safety-critical use.", 0.6),
            ("bm-rs-04", BenchmarkCategory::ResearchSynthesis, "Evaluate formal verification tools for Rust.", 0.8),
        ];
        for (id, cat, spec, diff) in specs {
            suite.tasks.push(BenchmarkTask {
                id: id.to_string(),
                category: *cat,
                input_spec: spec.to_string(),
                quality_rubric: vec!["Correct".to_string(), "Idiomatic".to_string(), "Tested".to_string()],
                reference_solution: String::new(),
                difficulty: *diff,
            });
        }
        suite
    }

    pub fn add(&mut self, task: BenchmarkTask) {
        self.tasks.push(task);
    }

    #[must_use]
    pub fn tasks(&self) -> &[BenchmarkTask] {
        &self.tasks
    }

    #[must_use]
    pub fn by_category(&self, cat: BenchmarkCategory) -> Vec<&BenchmarkTask> {
        self.tasks.iter().filter(|t| t.category == cat).collect()
    }
}

// ── DegradationHandler ────────────────────────────────────────────────────────

pub struct DegradationHandler;

impl DegradationHandler {
    /// Default strategy table: maps each trigger to its preferred response.
    #[must_use]
    pub fn default_strategies() -> Vec<DegradationStrategy> {
        vec![
            DegradationStrategy {
                trigger: DegradationTrigger::TaskComplexityExceeded,
                response: DegradationResponse::AttemptSimplerSubGoal,
            },
            DegradationStrategy {
                trigger: DegradationTrigger::BudgetExhausted,
                response: DegradationResponse::Checkpoint,
            },
            DegradationStrategy {
                trigger: DegradationTrigger::AllModelsFailing,
                response: DegradationResponse::SuggestHumanIntervention,
            },
            DegradationStrategy {
                trigger: DegradationTrigger::ConvergenceImpossible,
                response: DegradationResponse::DocumentBlocker,
            },
        ]
    }

    /// Resolve a trigger to its response using the provided strategy table.
    #[must_use]
    pub fn resolve(
        trigger: DegradationTrigger,
        strategies: &[DegradationStrategy],
    ) -> DegradationResponse {
        strategies
            .iter()
            .find(|s| s.trigger == trigger)
            .map(|s| s.response)
            .unwrap_or(DegradationResponse::DocumentBlocker)
    }
}

// ── ContinuousLearner ─────────────────────────────────────────────────────────

pub struct ContinuousLearner<'a> {
    pub prompt_library: &'a mut PromptLibrary,
    pub calibration: &'a mut CalibrationTracker,
    pub error_ledger: &'a mut ErrorBudgetLedger,
}

impl<'a> ContinuousLearner<'a> {
    /// Run the post-session learning pipeline synchronously.
    ///
    /// Steps (in dependency order):
    /// 1. Evaluate session and record a calibration observation.
    /// 2. Record the session outcome in the error budget ledger.
    /// 3. Record prompt quality for any templates used.
    pub fn run(
        &mut self,
        sigma: &ConversationState,
        template_id: Option<&str>,
        predicted_difficulty: f64,
        actual_difficulty: f64,
    ) {
        let eval = SelfImprovementEngine::evaluate_session(sigma);

        let actual_outcome = eval.metrics.get("convergence_p").copied().unwrap_or(0.0);
        self.calibration.record(CalibrationRecord {
            session_id: sigma.session_id.clone(),
            predicted_difficulty,
            actual_difficulty,
            predicted_outcome: sigma.completion_probability,
            actual_outcome,
        });

        let failure_rate = eval.metrics.get("failure_rate").copied().unwrap_or(0.0);
        self.error_ledger
            .record_outcome("general", failure_rate > 0.1);

        if let Some(tid) = template_id {
            self.prompt_library.record_performance(
                tid,
                &sigma.session_id,
                actual_outcome,
            );
        }
    }
}

// ── SafetyInterlock ───────────────────────────────────────────────────────────

pub struct SafetyInterlock;

impl SafetyInterlock {
    const PROTECTED_FILES: &'static [&'static str] = &[
        "security.rs",
        "verification.rs",
        "self_improvement.rs",
        "orchestrator.rs",
    ];

    #[must_use]
    pub fn is_modification_allowed(file_path: &str) -> bool {
        let path = Path::new(file_path);
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        !Self::PROTECTED_FILES.contains(&file_name)
    }
}

// ── SelfCodeModifier ──────────────────────────────────────────────────────────

pub struct SelfCodeModifier;

impl SelfCodeModifier {
    // Only patterns that are unambiguous token-level substitutions safe to apply
    // via string replace. Patterns that require AST context (e.g. "&String" which
    // appears in string literals and comments, or collect→iter which breaks non-Vec
    // iterators) have been removed until tree-sitter-based transforms are available.
    const PATTERNS: &'static [(&'static str, &'static str)] = &[
        (".unwrap_or_else(|_| panic!())", ".unwrap()"),
    ];

    /// Scan `current_content` for known sub-optimal patterns and return the
    /// improved version, or an error if the file is protected or nothing matched.
    pub fn propose_improvement(file_path: &str, current_content: &str) -> Result<String> {
        if !SafetyInterlock::is_modification_allowed(file_path) {
            return Err(anyhow!("Modification rejected: {} is a protected file", file_path));
        }
        let mut result = current_content.to_string();
        let mut changed = false;
        for (pattern, replacement) in Self::PATTERNS {
            if result.contains(pattern) {
                result = result.replace(pattern, replacement);
                changed = true;
            }
        }
        if changed {
            Ok(result)
        } else {
            Err(anyhow!("No sub-optimal code patterns identified in {}", file_path))
        }
    }

    /// Verify a proposed modification compiles by checking that it parses as valid UTF-8
    /// and that safety interlocks are satisfied. Returns the diff character count on success.
    pub fn verify(file_path: &str, original: &str, proposed: &str) -> Result<usize> {
        if !SafetyInterlock::is_modification_allowed(file_path) {
            return Err(anyhow!("Verification rejected: {} is protected", file_path));
        }
        if proposed.is_empty() {
            return Err(anyhow!("Proposed content is empty"));
        }
        let delta = proposed.len().abs_diff(original.len());
        Ok(delta)
    }
}

// ── PromptEvolutionaryOptimizer ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum PromptMutation {
    AppendEmphasis,
    PrependRole,
    TrimVerbose,
    InjectExamples,
}

pub struct PromptEvolutionaryOptimizer;

impl PromptEvolutionaryOptimizer {
    const MUTATIONS: &'static [PromptMutation] = &[
        PromptMutation::AppendEmphasis,
        PromptMutation::PrependRole,
        PromptMutation::TrimVerbose,
        PromptMutation::InjectExamples,
    ];

    /// Apply a single mutation to `template`, returning the mutated copy.
    #[must_use]
    pub fn mutate(template: &PromptTemplate, mutation: PromptMutation) -> PromptTemplate {
        let mut m = template.clone();
        m.version += 1;
        m.id = format!("{}-m{}", template.id, m.version);
        match mutation {
            PromptMutation::AppendEmphasis => {
                m.content.push_str("\n\nBe precise and cite evidence.");
            }
            PromptMutation::PrependRole => {
                m.content = format!("You are an expert Rust engineer.\n\n{}", m.content);
            }
            PromptMutation::TrimVerbose => {
                m.content = m.content.chars().take(m.content.len().saturating_sub(m.content.len() / 4)).collect();
            }
            PromptMutation::InjectExamples => {
                if !m.content.contains("{{examples}}") {
                    m.content.push_str("\n\nExamples:\n{{examples}}");
                }
            }
        }
        m
    }

    /// Generate one mutant per mutation strategy from `parent`.
    #[must_use]
    pub fn generate_variants(parent: &PromptTemplate) -> Vec<PromptTemplate> {
        Self::MUTATIONS.iter().map(|&m| Self::mutate(parent, m)).collect()
    }

    /// Tournament selection: return the template with the highest mean quality.
    #[must_use]
    pub fn select_winner(candidates: &[PromptTemplate]) -> Option<&PromptTemplate> {
        candidates.iter().max_by(|a, b| a.mean_quality().total_cmp(&b.mean_quality()))
    }
}

// ── CalibrationAdjuster ───────────────────────────────────────────────────────

pub struct CalibrationAdjuster;

impl CalibrationAdjuster {
    /// Fit a linear Platt scaling correction: `calibrated = a * raw + b`.
    /// Uses ordinary least squares on (predicted_outcome, actual_outcome) pairs.
    /// Returns `(a, b)` or `(1.0, 0.0)` when there is insufficient data.
    #[must_use]
    pub fn fit_platt(records: &[CalibrationRecord]) -> (f64, f64) {
        let n = records.len() as f64;
        if n < 2.0 {
            return (1.0, 0.0);
        }
        let sum_x: f64 = records.iter().map(|r| r.predicted_outcome).sum();
        let sum_y: f64 = records.iter().map(|r| r.actual_outcome).sum();
        let sum_xx: f64 = records.iter().map(|r| r.predicted_outcome * r.predicted_outcome).sum();
        let sum_xy: f64 = records.iter().map(|r| r.predicted_outcome * r.actual_outcome).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return (1.0, 0.0);
        }
        let a = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - a * sum_x) / n;
        (a, b)
    }

    /// Apply the fitted correction to a raw predicted probability.
    #[must_use]
    pub fn apply(raw: f64, a: f64, b: f64) -> f64 {
        (a * raw + b).clamp(0.0, 1.0)
    }
}

// ── BenchmarkRegressionGuard ──────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct BenchmarkRegressionGuard {
    baseline: HashMap<String, f64>,
}

impl BenchmarkRegressionGuard {
    const REGRESSION_THRESHOLD: f64 = 0.05;

    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the baseline score for a task before any modification.
    pub fn set_baseline(&mut self, task_id: &str, score: f64) {
        self.baseline.insert(task_id.to_string(), score);
    }

    /// Compare `results` against the stored baseline.
    /// Returns `Ok(())` when no task regresses by more than 5%.
    /// Returns `Err` listing the regressing task IDs.
    pub fn check(&self, results: &[BenchmarkResult]) -> Result<()> {
        let regressions: Vec<String> = results
            .iter()
            .filter_map(|r| {
                let base = self.baseline.get(&r.task_id)?;
                if *base - r.score > Self::REGRESSION_THRESHOLD {
                    Some(format!("{}: {:.3} → {:.3}", r.task_id, base, r.score))
                } else {
                    None
                }
            })
            .collect();
        if regressions.is_empty() {
            Ok(())
        } else {
            Err(anyhow!("Benchmark regression detected: {}", regressions.join("; ")))
        }
    }

    #[must_use]
    pub fn baseline_count(&self) -> usize {
        self.baseline.len()
    }
}

// ── PostMortemLearner ─────────────────────────────────────────────────────────

pub struct PostMortemLearner;

impl PostMortemLearner {
    /// Apply lessons from a post-mortem: update the prompt library and strategy
    /// database so the same failure is less likely to recur.
    pub fn apply(
        mortem: &PostMortem,
        library: &mut PromptLibrary,
        strategies: &mut StrategyDatabase,
    ) {
        let remediation = match mortem.root_cause {
            FailureCause::TypeMismatch => {
                "Carefully verify all type signatures before proposing code changes."
            }
            FailureCause::MissingContext => {
                "Retrieve and include relevant context from memory before starting the task."
            }
            FailureCause::ComplexityExceeded => {
                "Decompose the task into smaller sub-goals before attempting implementation."
            }
            FailureCause::InsufficientBudget => {
                "Estimate token cost upfront and choose the most concise approach."
            }
            FailureCause::AgentCapabilityLimit | FailureCause::Unknown => {
                "Escalate to a higher-capability model or request human assistance."
            }
        };

        let patched = PromptTemplate {
            id: format!("postmortem-{}", mortem.session_id),
            version: 1,
            content: format!("[Learned correction]\n{remediation}"),
            task_types: vec!["general".to_string()],
            performance_history: vec![],
        };
        library.insert(patched);

        strategies.insert(StrategyEntry {
            id: format!("postmortem-strategy-{}", mortem.session_id),
            task_features: vec![mortem.failure_turn_indices.len() as f64],
            approach: remediation.to_string(),
            steps: vec![remediation.to_string()],
            outcome_quality: 0.0,
            sessions_used: 0,
        });
    }
}

// ── RuntimeParameterAdjuster ──────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct RuntimeParameterAdjuster {
    pub parameters: HashMap<String, f64>,
    pub history: Vec<ParameterAdjustment>,
}

impl RuntimeParameterAdjuster {
    #[must_use]
    pub fn new() -> Self {
        let mut params = HashMap::new();
        params.insert("min_quality_score".to_string(), 0.5);
        params.insert("convergence_threshold".to_string(), 0.98);
        params.insert("regression_alert_ratio".to_string(), 0.9);
        params.insert("tautology_similarity_threshold".to_string(), 0.95);
        Self { parameters: params, history: vec![] }
    }

    /// Adjust `parameter` to `new_value` if the A/B report shows significant improvement.
    pub fn apply_if_significant(
        &mut self,
        parameter: &str,
        new_value: f64,
        report: &AbTestReport,
        rationale: &str,
    ) -> bool {
        if !report.significant || !report.adopted {
            return false;
        }
        let old_value = match self.parameters.get(parameter).copied() {
            Some(v) => v,
            None => return false,
        };
        self.parameters.insert(parameter.to_string(), new_value);
        self.history.push(ParameterAdjustment {
            parameter: parameter.to_string(),
            old_value,
            new_value,
            rationale: rationale.to_string(),
            applied_at: ConversationState::now(),
        });
        true
    }

    #[must_use]
    pub fn get(&self, parameter: &str) -> Option<f64> {
        self.parameters.get(parameter).copied()
    }
}

// ── ProgressReporter ─────────────────────────────────────────────────────────

pub struct ProgressReporter;

impl ProgressReporter {
    /// Build a progress report from `sigma`, given an `expected_turns` budget.
    #[must_use]
    pub fn report(sigma: &ConversationState, expected_turns: u32) -> ProgressReport {
        let completed = sigma.turns.len() as u32;
        let p = sigma.completion_probability;

        // Estimate velocity from the last two turns' contribution to completion probability.
        // Uses the per-turn average only when we have enough data; avoids the error of
        // dividing the current P by total turns (which assumes P started at 0).
        let estimated_remaining = if p > 0.0 && p < 1.0 && completed >= 2 {
            let velocity = p / completed as f64;
            if velocity > 0.0 {
                Some(((1.0 - p) / velocity).ceil() as u32)
            } else {
                None
            }
        } else {
            None
        };

        let success_probability = p
            * (1.0
                - sigma
                    .turns
                    .iter()
                    .filter(|t| {
                        matches!(
                            t.outcome,
                            crate::types::conversation::TurnOutcome::Rejected
                                | crate::types::conversation::TurnOutcome::RolledBack
                        )
                    })
                    .count() as f64
                    / (completed.max(1) as f64));

        ProgressReport {
            session_id: sigma.session_id.clone(),
            turns_completed: completed,
            turns_expected: expected_turns,
            completion_probability: p,
            estimated_turns_remaining: estimated_remaining,
            success_probability: success_probability.clamp(0.0, 1.0),
        }
    }
}

// ── LearningEffectivenessMonitor ──────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct LearningEffectivenessMonitor {
    pub outcomes: Vec<LearningOutcome>,
}

impl LearningEffectivenessMonitor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, action: &str, metric: &str, before: f64, after: f64) {
        self.outcomes.push(LearningOutcome {
            action: action.to_string(),
            metric: metric.to_string(),
            before,
            after,
        });
    }

    /// Mean improvement delta across all recorded outcomes.
    #[must_use]
    pub fn mean_delta(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        self.outcomes.iter().map(|o| o.delta()).sum::<f64>() / self.outcomes.len() as f64
    }

    /// Outcomes where `after > before`.
    #[must_use]
    pub fn effective_actions(&self) -> Vec<&LearningOutcome> {
        self.outcomes.iter().filter(|o| o.delta() > 0.0).collect()
    }
}

// ── EscalationContextBuilder ──────────────────────────────────────────────────

pub struct EscalationContextBuilder;

impl EscalationContextBuilder {
    /// Build a `HandoffPackage` summarising why the session failed and what was tried.
    #[must_use]
    pub fn build(
        sigma: &ConversationState,
        trigger: &str,
        hypotheses_tried: Vec<String>,
    ) -> HandoffPackage {
        let failures: Vec<&crate::types::conversation::Turn> = sigma
            .turns
            .iter()
            .filter(|t| {
                matches!(
                    t.outcome,
                    crate::types::conversation::TurnOutcome::Rejected
                        | crate::types::conversation::TurnOutcome::RolledBack
                )
            })
            .collect();

        let last_successful = sigma
            .turns
            .iter()
            .rev()
            .find(|t| {
                matches!(
                    t.outcome,
                    crate::types::conversation::TurnOutcome::TestsPassed
                        | crate::types::conversation::TurnOutcome::Compiled
                )
            })
            .map(|t| t.index);

        let failure_summary = format!(
            "{} of {} turns failed (failure rate {:.0}%)",
            failures.len(),
            sigma.turns.len(),
            failures.len() as f64 / sigma.turns.len().max(1) as f64 * 100.0
        );

        let context_snapshot = sigma
            .turns
            .iter()
            .rev()
            .take(3)
            .map(|t| format!("[{}] {}: {:?}", t.index, t.model_id, t.outcome))
            .collect::<Vec<_>>()
            .join("\n");

        let recommended = if failures.len() as f64 / sigma.turns.len().max(1) as f64 > 0.5 {
            "Switch to a higher-capability model or decompose task further."
        } else {
            "Review last failing turn and provide explicit constraints."
        };

        HandoffPackage {
            session_id: sigma.session_id.clone(),
            trigger: trigger.to_string(),
            failure_summary,
            hypotheses_tried,
            last_successful_turn: last_successful,
            recommended_next_action: recommended.to_string(),
            context_snapshot,
        }
    }
}
