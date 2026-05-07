use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use crate::types::intelligence::{ModelProfile, PromptTemplate, RegressionAlert};
use crate::types::self_improvement::CalibrationRecord;
use anyhow::{Context, Result, anyhow};
use sled;
use dashmap::DashMap;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{RwLock, mpsc};
use tokio::time::{Duration, interval};

pub struct CheckpointService {
    flush_tx: Option<mpsc::Sender<()>>,
    pub(crate) handle: Option<tokio::task::JoinHandle<()>>,
}

impl CheckpointService {
    fn new() -> Self {
        Self {
            flush_tx: None,
            handle: None,
        }
    }

    fn spawn(
        path: String,
        profiles: Arc<DashMap<String, ModelProfile>>,
        templates: Arc<RwLock<Vec<PromptTemplate>>>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel(1);
        let handle = tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    Some(_) = rx.recv() => {
                        ticker.tick().await;
                        while rx.try_recv().is_ok() {}
                        let profiles_map: BTreeMap<String, ModelProfile> = profiles
                            .iter()
                            .map(|entry| (entry.key().clone(), entry.value().clone()))
                            .collect();
                        let templates_guard = templates.read().await;
                        let data = serde_json::json!({
                            "profiles": profiles_map,
                            "templates": &*templates_guard,
                        });
                        drop(templates_guard);
                        if let Ok(content) = serde_json::to_string_pretty(&data) {
                            let temp_path = format!("{}.tmp", path);
                            if fs::write(&temp_path, &content).await.is_ok()
                                && fs::rename(&temp_path, &path).await.is_err()
                            {
                                crate::log_warn!(fs::remove_file(&temp_path).await, "Failed to remove temp file during checkpoint");
                            }
                        }
                    }
                    else => break,
                }
            }
        });
        Self {
            flush_tx: Some(tx),
            handle: Some(handle),
        }
    }

    fn trigger(&self) -> Result<()> {
        if let Some(tx) = &self.flush_tx {
            match tx.try_send(()) {
                Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(anyhow!(
                        "checkpoint flush channel closed; flusher task is dead"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn save_all(&self) -> Result<()> {
        self.trigger()
    }
}

pub struct IntelligenceEngine {
    profiles: Arc<DashMap<String, ModelProfile>>,
    templates: Arc<RwLock<Vec<PromptTemplate>>>,
    calibration: Arc<RwLock<Vec<CalibrationRecord>>>,
    storage_path: Option<String>,
    latency_predictor: LatencyPredictor,
    #[allow(dead_code)]
    failures: Arc<FailurePatternStore>,
    #[allow(dead_code)]
    learner: StrategyLearner,
    checkpoint: CheckpointService,
    /// Per-agent diff quality score (default 1.0 for unknown agents).
    /// Clean diffs raise it; regressive or conflicted diffs lower it.
    /// Used by routing to downweight agents with poor diff history.
    diff_quality: Arc<DashMap<String, f64>>,
}

impl Default for IntelligenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IntelligenceEngine {
    fn drop(&mut self) {
        if let Some(h) = self.checkpoint.handle.take() {
            h.abort();
        }
    }
}

impl IntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(DashMap::new()),
            templates: Arc::new(RwLock::new(Vec::new())),
            calibration: Arc::new(RwLock::new(Vec::new())),
            storage_path: None,
            latency_predictor: LatencyPredictor::new(),
            failures: {
                let id = format!("{}-{:x}", std::process::id(), rand::random::<u64>());
                Arc::new(FailurePatternStore::new(&format!("{}/crosstalk-intel-{}", std::env::temp_dir().display(), id))
                    .unwrap_or_else(|_| FailurePatternStore::ephemeral()))
            },
            learner: StrategyLearner::new(),
            checkpoint: CheckpointService::new(),
            diff_quality: Arc::new(DashMap::new()),
        }
    }

    /// Initializes the engine, loads state asynchronously, and spawns the Checkpoint Background Actor.
    pub async fn with_storage(path: &str) -> Result<Self> {
        let mut engine = Self {
            profiles: Arc::new(DashMap::new()),
            templates: Arc::new(RwLock::new(Vec::new())),
            calibration: Arc::new(RwLock::new(Vec::new())),
            storage_path: Some(path.to_string()),
            latency_predictor: LatencyPredictor::new(),
            failures: {
                let id = format!("{}-{:x}", std::process::id(), rand::random::<u64>());
                Arc::new(FailurePatternStore::new(&format!("{}/crosstalk-intel-{}", std::env::temp_dir().display(), id))
                    .unwrap_or_else(|_| FailurePatternStore::ephemeral()))
            },
            learner: StrategyLearner::new(),
            checkpoint: CheckpointService::new(),
            diff_quality: Arc::new(DashMap::new()),
        };

        engine.load_profiles().await?;
        engine.spawn_checkpoint_actor();

        Ok(engine)
    }

    /// Asynchronous, non-blocking file read.
    pub async fn load_profiles(&self) -> Result<()> {
        if let Some(path) = &self.storage_path
            && match tokio::fs::try_exists(path).await {
                Ok(v) => v,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
                Err(e) => return Err(anyhow::anyhow!(e)),
            }
        {
            let content = fs::read_to_string(path)
                .await
                .context("Failed to read intelligence data")?;
            let data: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(profiles) = data.get("profiles") {
                let parsed: BTreeMap<String, ModelProfile> =
                    serde_json::from_value(profiles.clone())?;
                for (k, v) in parsed {
                    self.profiles.insert(k, v);
                }
            }
            if let Some(templates) = data.get("templates") {
                *self.templates.write().await = serde_json::from_value(templates.clone())?;
            }
        }
        Ok(())
    }

    fn spawn_checkpoint_actor(&mut self) {
        if let Some(path) = self.storage_path.clone() {
            self.checkpoint = CheckpointService::spawn(
                path,
                Arc::clone(&self.profiles),
                Arc::clone(&self.templates),
            );
        }
    }

    pub fn update_profile(&self, turn: &Turn, quality_score: f64) {
        let mut profile = self
            .profiles
            .entry(turn.model_id.clone())
            .or_insert_with(|| ModelProfile {
                model_id: turn.model_id.clone(),
                task_scores: BTreeMap::new(),
                total_turns: 0,
                last_updated: ConversationState::now(),
                latency_ms: Default::default(),
            });

        if let Some(cat) = turn.task_category {
            profile
                .task_scores
                .entry(cat)
                .or_default()
                .update(quality_score);
        }

        profile.total_turns += 1;
        profile.last_updated = ConversationState::now();

        self.trigger_save();
    }

    pub fn update_profile_with_latency(&self, turn: &Turn, quality_score: f64, latency_ms: u64) {
        self.latency_predictor.record(&turn.model_id, latency_ms);
        let predicted = self.latency_predictor.predict_latency(&turn.model_id) as f64;

        let mut profile = self
            .profiles
            .entry(turn.model_id.clone())
            .or_insert_with(|| ModelProfile {
                model_id: turn.model_id.clone(),
                task_scores: BTreeMap::new(),
                total_turns: 0,
                last_updated: ConversationState::now(),
                latency_ms: Default::default(),
            });

        if let Some(cat) = turn.task_category {
            profile.task_scores.entry(cat).or_default().update(quality_score);
        }
        profile.total_turns += 1;
        profile.last_updated = ConversationState::now();
        profile.latency_ms.update(predicted);

        drop(profile);
        self.trigger_save();
    }

    fn trigger_save(&self) {
        if let Err(e) = self.checkpoint.trigger() {
            tracing::warn!(error = %e, "checkpoint trigger failed");
        }
    }

    pub fn save_all(&self) -> Result<()> {
        self.checkpoint.save_all()
    }

    /// Returns a shared handle to the templates store (Arc clone, not a data copy).
    pub fn templates(&self) -> Arc<RwLock<Vec<PromptTemplate>>> {
        Arc::clone(&self.templates)
    }

    /// Returns a shared handle to the calibration store (Arc clone, not a data copy).
    pub fn calibration(&self) -> Arc<RwLock<Vec<CalibrationRecord>>> {
        Arc::clone(&self.calibration)
    }

    /// Update the diff quality score for `agent_id` based on the outcome of its
    /// most recent artifact change.
    ///
    /// - `clean`: diff applied without conflict or validation failure.
    /// - `regressive`: `RegressionDetector::is_regressive` returned true for
    ///   at least one artifact in this turn.
    ///
    /// If both `clean` and `regressive` are false the diff was conflicted or
    /// failed to apply (e.g. patch rejected), which also penalises the score.
    pub fn update_diff_quality(&self, agent_id: &str, clean: bool, regressive: bool) {
        let mut score = self.diff_quality
            .entry(agent_id.to_string())
            .or_insert(1.0);
        if regressive {
            *score = (*score - 0.2).max(0.0);
        } else if clean {
            *score = (*score + 0.05).min(1.0);
        } else {
            // conflicted / failed diff
            *score = (*score - 0.15).max(0.0);
        }
    }

    /// Returns the diff quality score for `agent_id`, or 1.0 if no history
    /// exists yet.
    #[must_use]
    pub fn diff_quality_score(&self, agent_id: &str) -> f64 {
        self.diff_quality
            .get(agent_id)
            .map(|v| *v)
            .unwrap_or(1.0)
    }

    pub fn detect_regression(
        &self,
        model_id: &str,
        recent_turns: &[Turn],
    ) -> Option<RegressionAlert> {
        if recent_turns.is_empty() {
            return None;
        }

        let profile = self.profiles.get(model_id)?;
        let mut recent_quality_sum = 0.0;
        let mut valid_turns = 0;
        let mut task_category = TaskCategory::Research;

        for turn in recent_turns {
            if turn.model_id == model_id {
                let score = QualityScorer::score(turn);
                recent_quality_sum += score;
                valid_turns += 1;
                if let Some(cat) = turn.task_category {
                    task_category = cat;
                }
            }
        }

        if valid_turns == 0 {
            return None;
        }
        let recent_avg = recent_quality_sum / valid_turns as f64;

        let baseline = profile
            .task_scores
            .get(&task_category)
            .map(|s| s.mean)
            .unwrap_or(0.5);

        if recent_avg < baseline * 0.9 {
            return Some(RegressionAlert {
                agent_id: model_id.to_string(),
                task_category,
                baseline_mean: baseline,
                recent_mean: recent_avg,
                severity: (baseline - recent_avg) / baseline,
                timestamp: ConversationState::now(),
            });
        }
        None
    }

    /// Thompson Sampling router. Samples from Beta(alpha, beta) per agent to balance
    /// exploration (try underused agents) with exploitation (prefer proven performers).
    pub fn route_task_constrained(
        &self,
        category: TaskCategory,
        available_models: &[String],
        budget: u32,
        latency_ms: u64,
        blacklist: &[String],
    ) -> Result<String, String> {
        self.route_task_internal(category, available_models, Some((budget, latency_ms, blacklist)))
    }

    /// Sample from Beta(alpha, beta) using the Joehnk method.
    fn beta_sample(alpha: f64, beta: f64) -> f64 {
        let mut rng = rand::rng();
        let u1: f64 = rand::Rng::random_range(&mut rng, 0.001..1.0);
        let u2: f64 = rand::Rng::random_range(&mut rng, 0.001..1.0);
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        let sum = x + y;
        if sum <= 1.0 {
            x / sum
        } else {
            // Rejection: fall back to mean estimate
            alpha / (alpha + beta)
        }
    }

    fn generate_routing_diagnostics(
        &self,
        category: TaskCategory,
        available_models: &[String],
        budget: u32,
        latency_ms: u64,
        blacklist: &[String],
    ) -> String {
        let mut issues = Vec::new();

        if Self::estimate_tokens(category) > budget {
            issues.push(format!(
                "Category {:?} needs {} tokens but budget is {}",
                category,
                Self::estimate_tokens(category),
                budget
            ));
        }

        for model in available_models {
            if blacklist.contains(model) {
                issues.push(format!("{} is blacklisted", model));
            }
            if let Some(profile) = self.profiles.get(model)
                && profile.latency_ms.mean > latency_ms as f64
            {
                issues.push(format!(
                    "{} latency {}ms exceeds {}ms limit",
                    model, profile.latency_ms.mean as u64, latency_ms
                ));
            }
        }

        if issues.is_empty() {
            "No specific routing issues identified.".to_string()
        } else {
            format!("Issues: {}", issues.join("; "))
        }
    }

    /// Shared routing logic used by both `route_task` and `route_task_constrained`.
    ///
    /// When `constraints` is `None`, all models are eligible and Thompson Sampling
    /// is replaced by plain mean-score selection (deterministic, no budget check).
    /// When `constraints` is `Some((budget, latency_ms, blacklist))`, the full
    /// constrained Thompson-Sampling path runs.
    fn route_task_internal(
        &self,
        category: TaskCategory,
        available_models: &[String],
        constraints: Option<(u32, u64, &[String])>,
    ) -> Result<String, String> {
        if available_models.is_empty() {
            return Ok(String::new());
        }

        if let Some((budget, latency_ms, blacklist)) = constraints {
            let estimated_tokens = Self::estimate_tokens(category);
            if estimated_tokens > budget {
                return Err(format!(
                    "Estimated tokens {} exceeds budget {}",
                    estimated_tokens, budget
                ));
            }

            let blacklist_set: HashSet<&str> = blacklist.iter().map(|s| s.as_str()).collect();
            let mut best_candidate: Option<&String> = None;
            let mut highest_sample = -1.0_f64;

            for model_id in available_models {
                if blacklist_set.contains(model_id.as_str()) {
                    continue;
                }
                if let Some(profile) = self.profiles.get(model_id) {
                    let predicted = self.latency_predictor.predict_latency(model_id);
                    let effective_latency = if predicted > 0 {
                        predicted as f64
                    } else {
                        profile.latency_ms.mean
                    };
                    if effective_latency > latency_ms as f64 {
                        continue;
                    }
                    let mean = profile.task_scores.get(&category).map_or(0.5, |ra| ra.mean);
                    let n = profile.task_scores.get(&category).map_or(1, |ra| ra.count);
                    // Beta distribution parameters from observed mean and sample count.
                    // More observations → tighter distribution (less exploration).
                    // Fewer observations → wider (more exploration of unknown agents).
                    let alpha = mean * n as f64 + 1.0;
                    let beta = (1.0 - mean) * n as f64 + 1.0;
                    let sample = Self::beta_sample(alpha, beta) * self.diff_quality_score(model_id);
                    if sample > highest_sample {
                        highest_sample = sample;
                        best_candidate = Some(model_id);
                    }
                }
            }

            match best_candidate {
                Some(model) => Ok(model.clone()),
                None => {
                    let diag = self.generate_routing_diagnostics(
                        category,
                        available_models,
                        budget,
                        latency_ms,
                        blacklist,
                    );
                    Err(format!("No models satisfy constraints. {}", diag))
                }
            }
        } else {
            // Unconstrained: pick the highest mean score weighted by diff quality.
            let mut best_model = available_models[0].clone();
            let mut best_score = -1.0_f64;
            for model_id in available_models {
                if let Some(profile) = self.profiles.get(model_id) {
                    let score = profile
                        .task_scores
                        .get(&category)
                        .map(|ra| ra.mean)
                        .unwrap_or(0.5)
                        * self.diff_quality_score(model_id);
                    if score > best_score {
                        best_score = score;
                        best_model = model_id.clone();
                    }
                }
            }
            Ok(best_model)
        }
    }

    pub fn route_task(&self, category: TaskCategory, available_models: &[String]) -> String {
        // Unconstrained routing never errors; unwrap is safe.
        self.route_task_internal(category, available_models, None)
            .unwrap_or_default()
    }

    pub async fn evolve_prompts(&self, category: TaskCategory) -> Result<()> {
        let mut templates = self.templates.write().await;
        let mut top_performers: Vec<_> = templates.iter()
            .filter(|t| t.category() == category)
            .collect();

        top_performers.sort_by(|a, b| b.mean_performance().total_cmp(&a.mean_performance()));

        if let Some(best) = top_performers.first()
            && best.mean_performance() > 0.8
        {
            let mutation = crate::types::intelligence::MutationStrategy::Append(
                "Be precise and follow all project invariants strictly.".to_string()
            );
            let new_version = best.mutate(mutation);
            templates.push(new_version);
        }
        Ok(())
    }

    pub fn estimate_tokens(category: TaskCategory) -> u32 {
        category.token_estimate()
    }
}

pub struct QualityScorer;

impl QualityScorer {
    /// Semantic quality scoring using embedding-based cosine similarity.
    ///
    /// Replaces shallow keyword heuristics ("evidence", backtick counting) with
    /// cosine similarity between the turn content embedding and the task
    /// description embedding.  When `task_description` is empty the content is
    /// compared to itself (similarity 1.0), yielding a neutral bonus so that
    /// outcome and certainty weights remain dominant.
    ///
    /// When `observer` is `Some((obs, agent_id))` the final score is multiplied
    /// by the agent's Beta-posterior calibration score to downweight agents
    /// whose claimed certainty is poorly calibrated against verified outcomes.
    #[must_use]
    pub fn score_with_context(
        turn: &Turn,
        task_description: &str,
        observer: Option<(&crate::engines::metacognition::MetacognitiveObserver, &str)>,
    ) -> f64 {
        let mut score = 0.5;

        // Outcome weighting (unchanged from original).
        score += match turn.outcome {
            TurnOutcome::TestsPassed => 0.4,
            TurnOutcome::Compiled => 0.2,
            TurnOutcome::AdvancedConvergence => 0.35,
            TurnOutcome::Unknown => 0.0,
            TurnOutcome::Stalled => -0.2,
            TurnOutcome::RolledBack => -0.4,
            TurnOutcome::Rejected => -0.4,
            TurnOutcome::VerificationFailed => -0.4,
        };

        // Semantic relevance: embedding cosine similarity replaces keyword heuristics.
        // embed_text falls back to local_embed_text when the ort-embeddings feature
        // is absent, so this never makes external API calls.
        let content_emb = crate::engines::memory::embed_text(&turn.content);
        let task_emb = if task_description.is_empty() {
            content_emb.clone()
        } else {
            crate::engines::memory::embed_text(task_description)
        };
        // cosine_sim returns a value in [-1, 1]; map to [0, 0.1] additive bonus.
        let relevance = crate::engines::memory::cosine_sim(&content_emb, &task_emb) as f64;
        score += ((relevance + 1.0) / 2.0) * 0.1;

        // Certainty weighting (unchanged from original).
        if let Some(certainty) = turn.certainty {
            score += certainty * 0.1;
        }

        score = score.clamp(0.0, 1.0);

        // Consistency score (unchanged from original).
        if let Some(cs) = turn.consistency_score {
            score *= 0.7 + 0.3 * cs;
        }

        let score = score.clamp(0.0, 1.0);

        // Calibration multiplier: downweight agents whose high-certainty claims
        // are not borne out by verified outcomes.
        if let Some((obs, agent_id)) = observer {
            (score * obs.calibration_score(agent_id)).clamp(0.0, 1.0)
        } else {
            score
        }
    }

    /// Backward-compatible entry point used by all existing call sites.
    /// Delegates to `score_with_context` with no task description and no observer.
    #[must_use]
    pub fn score(turn: &Turn) -> f64 {
        Self::score_with_context(turn, "", None)
    }
}

pub struct ConsistencyScorer;

impl ConsistencyScorer {
    const STOPWORDS: &'static [&'static str] = &[
        "fn", "let", "mut", "pub", "use", "mod", "impl", "self", "true", "false",
        "else", "enum", "struct", "trait", "type", "where", "match", "loop", "while",
        "for", "return", "async", "await", "move", "ref", "const", "static",
    ];

    #[must_use]
    pub fn score(explanation: &str, diff_text: &str) -> f64 {
        if explanation.is_empty() || diff_text.is_empty() {
            return 0.5;
        }

        let diff_terms = Self::extract_terms(diff_text);
        let expl_terms = Self::extract_terms(explanation);

        if diff_terms.is_empty() || expl_terms.is_empty() {
            return 0.5;
        }

        let intersection = diff_terms.iter().filter(|t| expl_terms.contains(*t)).count();
        let union = {
            let mut all: std::collections::HashSet<&str> =
                diff_terms.iter().map(|s| s.as_str()).collect();
            for t in &expl_terms {
                all.insert(t.as_str());
            }
            all.len()
        };

        if union == 0 {
            return 0.5;
        }

        (intersection as f64 / union as f64).clamp(0.0, 1.0)
    }

    fn extract_terms(text: &str) -> std::collections::HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|tok| tok.len() > 4)
            .map(|tok| tok.to_lowercase())
            .filter(|tok| !Self::STOPWORDS.contains(&tok.as_str()))
            .collect()
    }
}

pub struct ConvergenceMonitor;

impl ConvergenceMonitor {
    pub fn monitor_convergence(sigma: &ConversationState) -> crate::types::intelligence::IterationDecision {
        if sigma.turns.len() < 3 {
            return crate::types::intelligence::IterationDecision::Continue;
        }

        let last_p = sigma.completion_probability;
        let p_history: Vec<f64> = sigma.turns.iter()
            .map(|t| t.certainty.unwrap_or(0.0))
            .collect();
        
        if last_p > 0.98 { return crate::types::intelligence::IterationDecision::StopEarly; }

        let window = 3;
        if p_history.len() >= window {
            let recent = &p_history[p_history.len()-window..];
            let velocity = recent[window-1] - recent[0];
            let acceleration = if p_history.len() > window {
                let prev_recent = &p_history[p_history.len()-window-1..p_history.len()-1];
                let prev_velocity = prev_recent[window-1] - prev_recent[0];
                velocity - prev_velocity
            } else { 0.0 };

            if velocity < 0.005 && sigma.turns.len() > 10 {
                return crate::types::intelligence::IterationDecision::StopEarly;
            }
            if acceleration > 0.1 {
                return crate::types::intelligence::IterationDecision::Extend;
            }
        }

        crate::types::intelligence::IterationDecision::Continue
    }
}

pub struct ContextBudgeter;

impl ContextBudgeter {
    /// Allocates token budget based on information density:
    /// Score = (unique_decisions + unresolved_conflicts) / token_count
    #[must_use]
    pub fn allocate_by_density(available_tokens: usize, segments: &[(&str, usize)]) -> Vec<usize> {
        let mut density_weights = Vec::new();
        for (text, token_count) in segments {
            let decisions = text.matches("[decision]").count() + text.matches("###").count();
            let conflicts = text.matches("TODO").count() + text.matches("FIXME").count();
            let density = if *token_count > 0 {
                (decisions + conflicts) as f64 / (*token_count as f64)
            } else {
                0.0
            };
            density_weights.push(density.max(0.1)); // Minimum weight to prevent starvation
        }

        let total_weight: f64 = density_weights.iter().sum();
        let mut allocation = Vec::new();
        for w in density_weights {
            allocation.push(((w / total_weight) * available_tokens as f64) as usize);
        }

        let allocated_total: usize = allocation.iter().sum();
        let remainder = available_tokens.saturating_sub(allocated_total);
        if remainder > 0 && !allocation.is_empty() {
            allocation[0] += remainder;
        }

        allocation
    }
}

#[derive(Debug, Clone)]
pub enum VotingStrategy {
    Majority,
    WeightedConsensus,
    MaxConfidence,
}

pub struct ModelEnsemble {
    pub models: Vec<String>,
    pub voting_strategy: VotingStrategy,
    scores: DashMap<String, f64>,
}

impl ModelEnsemble {
    pub fn new(models: Vec<String>, voting_strategy: VotingStrategy) -> Self {
        Self {
            models,
            voting_strategy,
            scores: DashMap::new(),
        }
    }

    pub fn update_scores(&self, engine: &IntelligenceEngine, category: TaskCategory) {
        for model_id in &self.models {
            let score = engine
                .profiles
                .get(model_id)
                .map(|p| p.task_scores.get(&category).map_or(0.5, |ra| ra.mean))
                .unwrap_or(0.5);
            self.scores.insert(model_id.clone(), score);
        }
    }

    pub fn route_ensemble(
        &self,
        category: TaskCategory,
        available: &[String],
    ) -> Result<Vec<(String, f64)>> {
        let candidates: Vec<(String, f64)> = available
            .iter()
            .filter(|m| self.models.contains(m))
            .map(|m| (m.clone(), self.scores.get(m).map(|s| *s).unwrap_or(0.5)))
            .collect();

        if candidates.is_empty() {
            return Err(anyhow!(
                "No ensemble candidates available for {:?}",
                category
            ));
        }

        if Self::is_safety_critical(category) {
            let high_confidence = candidates.iter().filter(|(_, s)| *s > 0.8).count();
            if high_confidence < 3 {
                return Err(anyhow!(
                    "Safety-critical task requires 3 models with confidence > 0.8, got {}",
                    high_confidence
                ));
            }
        }

        let mut ranked: Vec<(String, f64)> = match self.voting_strategy {
            VotingStrategy::MaxConfidence => candidates,
            VotingStrategy::Majority => candidates.into_iter().filter(|(_, s)| *s >= 0.5).collect(),
            VotingStrategy::WeightedConsensus => {
                let total: f64 = candidates.iter().map(|(_, s)| s).sum();
                if total == 0.0 {
                    candidates
                } else {
                    candidates
                        .into_iter()
                        .map(|(m, s)| (m, s / total))
                        .collect()
                }
            }
        };

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(ranked)
    }

    pub fn route_ensemble_with_fallback(
        &self,
        category: TaskCategory,
        available: &[String],
        fallback: &str,
    ) -> Vec<(String, f64)> {
        match self.route_ensemble(category, available) {
            Ok(candidates) if !candidates.is_empty() => candidates,
            _ => vec![(fallback.to_string(), 0.5)],
        }
    }

    fn is_safety_critical(category: TaskCategory) -> bool {
        matches!(category, TaskCategory::Architecture)
    }
}

pub struct PromptComposer;

impl PromptComposer {
    pub fn compose(
        template: &PromptTemplate,
        base_task: &str,
        context_turns: &[&Turn],
        profile: &ModelProfile,
    ) -> Result<String> {
        let context_str = context_turns
            .iter()
            .take(3)
            .map(|t| format!("[Turn {}|{}] {}", t.index, t.model_id, t.content))
            .collect::<Vec<_>>()
            .join("\n");

        let cat = template.category();
        let profile_summary = if profile.task_scores.is_empty() {
            format!("{} (no history)", profile.model_id)
        } else {
            let avg = profile
                .task_scores
                .get(&cat)
                .map(|ra| ra.mean)
                .unwrap_or(0.5);
            format!(
                "{} | {:?} mean: {:.2} | {} turns",
                profile.model_id, cat, avg, profile.total_turns
            )
        };

        let mut vars = BTreeMap::new();
        vars.insert("task".to_string(), base_task.to_string());
        vars.insert("context".to_string(), context_str);
        vars.insert("profile_summary".to_string(), profile_summary);

        template.render(&vars)
    }

    pub fn select_template(
        templates: &[PromptTemplate],
        category: TaskCategory,
        is_in_regression: bool,
    ) -> Option<&PromptTemplate> {
        let matching: Vec<&PromptTemplate> = templates
            .iter()
            .filter(|t| t.category() == category)
            .collect();

        if is_in_regression {
            matching
                .iter()
                .find(|t| t.is_corrective())
                .copied()
                .or_else(|| matching.first().copied())
        } else {
            matching
                .iter()
                .find(|t| !t.is_corrective())
                .copied()
                .or_else(|| matching.first().copied())
        }
    }
}

pub struct RegressionFeedbackHandler;

impl RegressionFeedbackHandler {
    pub fn compose_corrective_prompt(
        alert: &RegressionAlert,
        base_prompt: &str,
        examples: &[String],
    ) -> String {
        let mut out = format!(
            "[Corrective: {:.0}% quality drop on {:?} — baseline {:.2}, recent {:.2}]\n",
            alert.severity * 100.0,
            alert.task_category,
            alert.baseline_mean,
            alert.recent_mean,
        );

        if !examples.is_empty() {
            out.push_str("Counter-examples (successful turns):\n");
            for (i, ex) in examples.iter().take(3).enumerate() {
                out.push_str(&format!("  {}. {}\n", i + 1, ex));
            }
        }

        out.push_str(base_prompt);
        out
    }

    pub fn counter_examples(turns: &[Turn], category: TaskCategory) -> Vec<String> {
        turns
            .iter()
            .filter(|t| {
                t.task_category == Some(category)
                    && matches!(
                        t.outcome,
                        TurnOutcome::TestsPassed
                            | TurnOutcome::Compiled
                            | TurnOutcome::AdvancedConvergence
                    )
            })
            .rev()
            .take(3)
            .map(|t| {
                let preview: String = t.content.chars().take(80).collect();
                format!("[Turn {}|{}] {}", t.index, t.model_id, preview)
            })
            .collect()
    }
}

pub struct LatencyPredictor {
    history: DashMap<String, VecDeque<u64>>,
    ema: DashMap<String, f64>,
}

impl LatencyPredictor {
    const ALPHA: f64 = 0.3;
    const WINDOW: usize = 20;

    pub fn new() -> Self {
        Self {
            history: DashMap::new(),
            ema: DashMap::new(),
        }
    }

    pub fn record(&self, model_id: &str, latency_ms: u64) {
        let mut hist = self.history.entry(model_id.to_string()).or_default();
        if hist.len() >= Self::WINDOW {
            hist.pop_front();
        }
        hist.push_back(latency_ms);
        drop(hist);

        let sample = latency_ms as f64;
        let mut ema = self.ema.entry(model_id.to_string()).or_insert(sample);
        *ema = Self::ALPHA * sample + (1.0 - Self::ALPHA) * *ema;
    }

    pub fn predict_latency(&self, model_id: &str) -> u64 {
        self.ema
            .get(model_id)
            .map(|v| if v.is_finite() { *v as u64 } else { 0u64 })
            .unwrap_or(0)
    }

    pub fn is_high_variance(&self, model_id: &str) -> bool {
        let hist = match self.history.get(model_id) {
            Some(h) => h,
            None => return false,
        };
        if hist.len() < 2 {
            return false;
        }
        let mean = hist.iter().sum::<u64>() as f64 / hist.len() as f64;
        if mean == 0.0 {
            return false;
        }
        let variance = hist
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / hist.len() as f64;
        variance.sqrt() > mean * 0.5
    }
}

impl Default for LatencyPredictor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ConvergenceVelocityTracker {
    p_history: VecDeque<f64>,
    velocity_history: VecDeque<f64>,
    window: usize,
}

impl ConvergenceVelocityTracker {
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self {
            p_history: VecDeque::new(),
            velocity_history: VecDeque::new(),
            window: window.max(2),
        }
    }

    pub fn record(&mut self, completion_probability: f64) {
        if let Some(&prev) = self.p_history.back() {
            let v = completion_probability - prev;
            if self.velocity_history.len() >= self.window {
                self.velocity_history.pop_front();
            }
            self.velocity_history.push_back(v);
        }
        if self.p_history.len() >= self.window {
            self.p_history.pop_front();
        }
        self.p_history.push_back(completion_probability);
    }

    #[must_use]
    pub fn current_velocity(&self) -> f64 {
        self.velocity_history.back().copied().unwrap_or(0.0)
    }

    #[must_use]
    pub fn mean_velocity(&self) -> f64 {
        if self.velocity_history.is_empty() {
            return 0.0;
        }
        self.velocity_history.iter().sum::<f64>() / self.velocity_history.len() as f64
    }

    #[must_use]
    pub fn acceleration(&self) -> f64 {
        if self.velocity_history.len() < 2 {
            return 0.0;
        }
        let mut it = self.velocity_history.iter().rev();
        let latest = *it.next().unwrap();
        let Some(prior) = it.next() else { return 0.0 };
        latest - *prior
    }

    #[must_use]
    pub fn is_stalled(&self) -> bool {
        self.velocity_history.len() >= 3
            && self
                .velocity_history
                .iter()
                .rev()
                .take(3)
                .all(|&v| v.abs() < 0.005)
    }

    /// Estimated turns remaining until completion_probability reaches 1.0.
    /// Returns `None` if velocity is zero or negative.
    #[must_use]
    pub fn predict_turns_to_completion(&self, current_p: f64) -> Option<u32> {
        let v = self.mean_velocity();
        if v <= 0.0 {
            return None;
        }
        let remaining = (1.0 - current_p).max(0.0);
        Some((remaining / v).ceil() as u32)
    }
}

impl Default for ConvergenceVelocityTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub model_id: String,
    pub quality: f64,
    pub cost_tokens: u32,
}

pub struct ParetoOptimizer;

impl ParetoOptimizer {
    /// Returns the non-dominated subset: no other point has both higher quality
    /// and lower (or equal) cost. Result is sorted by quality descending.
    ///
    /// O(n²) dominance check is intentional and acceptable: the input set is
    /// bounded by the number of active model endpoints (always <100 in practice).
    #[must_use]
    pub fn compute_frontier(points: Vec<ParetoPoint>) -> Vec<ParetoPoint> {
        let mut frontier: Vec<ParetoPoint> = points
            .iter()
            .filter(|candidate| {
                !points.iter().any(|other| {
                    other.model_id != candidate.model_id
                        && other.quality >= candidate.quality
                        && other.cost_tokens <= candidate.cost_tokens
                        && (other.quality > candidate.quality
                            || other.cost_tokens < candidate.cost_tokens)
                })
            })
            .cloned()
            .collect();
        frontier.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        frontier
    }

    /// From the frontier, select the cheapest point meeting both constraints.
    #[must_use]
    pub fn select(
        frontier: &[ParetoPoint],
        min_quality: f64,
        max_tokens: u32,
    ) -> Option<&ParetoPoint> {
        frontier
            .iter()
            .filter(|p| p.quality >= min_quality && p.cost_tokens <= max_tokens)
            .min_by_key(|p| p.cost_tokens)
    }

    /// Build `ParetoPoint`s from live engine profiles for a given category.
    #[must_use]
    pub fn from_profiles(
        engine: &IntelligenceEngine,
        category: TaskCategory,
        available_models: &[String],
    ) -> Vec<ParetoPoint> {
        available_models
            .iter()
            .filter_map(|model_id| {
                let profile = engine.profiles.get(model_id)?;
                let quality = profile.task_scores.get(&category).map_or(0.5, |ra| ra.mean);
                Some(ParetoPoint {
                    model_id: model_id.clone(),
                    quality,
                    cost_tokens: IntelligenceEngine::estimate_tokens(category),
                })
            })
            .collect()
    }
}

pub struct FailurePatternStore {
    db: Arc<sled::Db>,
}

impl FailurePatternStore {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(format!("{}/failures", path))?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn ephemeral() -> Self {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .unwrap_or_else(|e| {
                tracing::warn!(err = ?e, "failure pattern store unavailable, retrying with default config");
                sled::open(std::env::temp_dir().join("crosstalk_failures_fallback"))
                    .expect("cannot open any sled db")
            });
        Self { db: Arc::new(db) }
    }

    pub fn record_failure(&self, pattern: crate::types::intelligence::FailurePattern) -> Result<()> {
        let key = format!("failure:{}", pattern.pattern_id);
        let encoded = serde_json::to_vec(&pattern)?;
        self.db.insert(key, encoded)?;
        Ok(())
    }

    pub fn find_matches(&self, context_emb: &[f32], threshold: f32) -> Vec<crate::types::intelligence::FailurePattern> {
        let mut matches = Vec::new();
        for (_, v) in self.db.scan_prefix("failure:").flatten() {
            if let Ok(p) = serde_json::from_slice::<crate::types::intelligence::FailurePattern>(&v) {
                let sim = self.cosine_similarity(context_emb, &p.context_signature);
                if sim > threshold {
                    matches.push(p);
                }
            }
        }
        matches
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() { return 0.0; }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
        dot / (mag_a * mag_b)
    }
}

pub struct StrategyLearner {
    strategies: Arc<DashMap<TaskCategory, Vec<crate::types::intelligence::TurnStrategy>>>,
}

impl StrategyLearner {
    pub fn new() -> Self {
        Self {
            strategies: Arc::new(DashMap::new()),
        }
    }

    pub fn record_session_sequence(&self, category: TaskCategory, sequence: Vec<String>, quality: f64) {
        let mut list = self.strategies.entry(category).or_default();
        if let Some(existing) = list.iter_mut().find(|s| s.agent_sequence == sequence) {
            let total = existing.avg_quality * f64::from(existing.sample_size) + quality;
            existing.sample_size += 1;
            existing.avg_quality = total / f64::from(existing.sample_size);
        } else {
            list.push(crate::types::intelligence::TurnStrategy {
                task_category: category,
                agent_sequence: sequence,
                avg_quality: quality,
                sample_size: 1,
            });
        }
    }

    pub fn get_best_strategy(&self, category: TaskCategory) -> Option<Vec<String>> {
        let list = self.strategies.get(&category)?;
        list.iter()
            .max_by(|a, b| a.avg_quality.total_cmp(&b.avg_quality))
            .map(|s| s.agent_sequence.clone())
    }
}

impl Default for StrategyLearner {
    fn default() -> Self { Self::new() }
}
