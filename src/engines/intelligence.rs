use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use crate::types::intelligence::{ModelProfile, PromptTemplate, RegressionAlert};
use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};

pub struct CheckpointService {
    flush_tx: Option<mpsc::Sender<()>>,
    #[allow(dead_code)]
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl CheckpointService {
    fn new() -> Self {
        Self { flush_tx: None, handle: None }
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
                        let data = serde_json::json!({
                            "profiles": profiles_map,
                            "templates": &*templates.read().await,
                        });
                        if let Ok(content) = serde_json::to_string_pretty(&data) {
                            let temp_path = format!("{}.tmp", path);
                            if fs::write(&temp_path, &content).await.is_ok()
                                && fs::rename(&temp_path, &path).await.is_err()
                            {
                                let _ = fs::remove_file(&temp_path).await;
                            }
                        }
                    }
                    else => break,
                }
            }
        });
        Self { flush_tx: Some(tx), handle: Some(handle) }
    }

    fn trigger(&self) {
        if let Some(tx) = &self.flush_tx {
            let _ = tx.try_send(());
        }
    }

    pub fn save_all(&self) -> Result<()> {
        self.trigger();
        Ok(())
    }
}

pub struct IntelligenceEngine {
    pub profiles: Arc<DashMap<String, ModelProfile>>,
    pub templates: Arc<RwLock<Vec<PromptTemplate>>>,
    pub storage_path: Option<String>,
    pub latency_predictor: LatencyPredictor,
    checkpoint: CheckpointService,
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
            storage_path: None,
            latency_predictor: LatencyPredictor::new(),
            checkpoint: CheckpointService::new(),
        }
    }

    /// Initializes the engine, loads state asynchronously, and spawns the Checkpoint Background Actor.
    pub async fn with_storage(path: &str) -> Result<Self> {
        let mut engine = Self {
            profiles: Arc::new(DashMap::new()),
            templates: Arc::new(RwLock::new(Vec::new())),
            storage_path: Some(path.to_string()),
            latency_predictor: LatencyPredictor::new(),
            checkpoint: CheckpointService::new(),
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
            let content = fs::read_to_string(path).await.context("Failed to read intelligence data")?;
            let data: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(profiles) = data.get("profiles") {
                let parsed: BTreeMap<String, ModelProfile> = serde_json::from_value(profiles.clone())?;
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
        let mut profile = self.profiles.entry(turn.model_id.clone()).or_insert_with(|| ModelProfile {
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

        self.trigger_save();
    }

    pub fn update_profile_with_latency(&self, turn: &Turn, quality_score: f64, latency_ms: u64) {
        // C-009: there is a TOCTOU window between update_profile (which drops the DashMap guard)
        // and the get_mut below. Collapsing both into a single entry() closure would require
        // LatencyPredictor to be callable without a separate record() step; deferred for now.
        self.update_profile(turn, quality_score);
        self.latency_predictor.record(&turn.model_id, latency_ms);
        if let Some(mut profile) = self.profiles.get_mut(&turn.model_id) {
            profile.latency_ms.update(self.latency_predictor.predict_latency(&turn.model_id) as f64);
        }
    }

    fn trigger_save(&self) {
        self.checkpoint.trigger();
    }

    pub fn save_all(&self) -> Result<()> {
        self.checkpoint.save_all()
    }

    pub fn detect_regression(&self, model_id: &str, recent_turns: &[Turn]) -> Option<RegressionAlert> {
        if recent_turns.is_empty() { return None; }

        let profile = self.profiles.get(model_id)?;
        let mut recent_quality_sum = 0.0;
        let mut valid_turns = 0;
        let mut category_counts: std::collections::HashMap<TaskCategory, usize> = std::collections::HashMap::new();

        for turn in recent_turns {
            if turn.model_id == model_id {
                let score = QualityScorer::score(turn);
                recent_quality_sum += score;
                valid_turns += 1;
                if let Some(cat) = turn.task_category {
                    *category_counts.entry(cat).or_insert(0) += 1;
                }
            }
        }

        if valid_turns == 0 { return None; }
        let recent_avg = recent_quality_sum / valid_turns as f64;

        let baseline = if profile.task_scores.is_empty() {
            0.5
        } else {
            profile.task_scores.values().map(|avg| avg.mean).sum::<f64>() / profile.task_scores.len() as f64
        };

        if recent_avg < baseline * 0.9 {
            let task_category = category_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(cat, _)| cat)
                .unwrap_or(TaskCategory::Research);
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

    /// O(N) Routing algorithm. Finds the absolute best model in a single pass without allocating arrays or sorting.
    pub fn route_task_constrained(
        &self,
        category: TaskCategory,
        available_models: &[String],
        budget: u32,
        latency_ms: u64,
        blacklist: &[String],
    ) -> Result<String, String> {
        let estimated_tokens = Self::estimate_tokens(category);
        if estimated_tokens > budget {
            return Err(format!("Estimated tokens {} exceeds budget {}", estimated_tokens, budget));
        }

        let mut best_candidate: Option<&String> = None;
        let mut highest_score = -1.0;

        for model_id in available_models {
            if blacklist.contains(model_id) { continue; }

            if let Some(profile) = self.profiles.get(model_id) {
                let predicted = self.latency_predictor.predict_latency(model_id);
                let effective_latency = if predicted > 0 { predicted as f64 } else { profile.latency_ms.mean };
                if effective_latency > latency_ms as f64 { continue; }

                let score = profile.task_scores.get(&category).map_or(0.5, |ra| ra.mean);

                if score > highest_score {
                    highest_score = score;
                    best_candidate = Some(model_id);
                }
            }
        }

        match best_candidate {
            Some(model) => Ok(model.clone()),
            None => {
                let diag = self.generate_routing_diagnostics(category, available_models, budget, latency_ms, blacklist);
                Err(format!("No models satisfy constraints. {}", diag))
            }
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
            issues.push(format!("Category {:?} needs {} tokens but budget is {}", category, Self::estimate_tokens(category), budget));
        }

        for model in available_models {
            if blacklist.contains(model) {
                issues.push(format!("{} is blacklisted", model));
            }
            if let Some(profile) = self.profiles.get(model)
                && profile.latency_ms.mean > latency_ms as f64
            {
                issues.push(format!("{} latency {}ms exceeds {}ms limit", model, profile.latency_ms.mean as u64, latency_ms));
            }
        }

        if issues.is_empty() {
            "No specific routing issues identified.".to_string()
        } else {
            format!("Issues: {}", issues.join("; "))
        }
    }

    pub fn route_task(&self, category: TaskCategory, available_models: &[String]) -> String {
        if available_models.is_empty() {
            return String::new();
        }
        let mut best_model = available_models[0].clone();
        let mut best_score = -1.0;

        for model_id in available_models {
            if let Some(profile) = self.profiles.get(model_id) {
                let score = profile
                    .task_scores
                    .get(&category)
                    .map(|ra| ra.mean)
                    .unwrap_or(0.5);
                if score > best_score {
                    best_score = score;
                    best_model = model_id.clone();
                }
            }
        }
        best_model
    }

    #[must_use]
    pub fn estimate_tokens(category: TaskCategory) -> u32 {
        category.token_estimate()
    }
}

pub struct QualityScorer;

impl QualityScorer {
    #[must_use]
    pub fn score(turn: &Turn) -> f64 {
        let mut score = 0.5;

        score += match turn.outcome {
            TurnOutcome::TestsPassed => 0.4,
            TurnOutcome::Compiled => 0.2,
            TurnOutcome::AdvancedConvergence => 0.35,
            TurnOutcome::Unknown => 0.0,
            TurnOutcome::Stalled => -0.2,
            TurnOutcome::RolledBack => -0.4,
            TurnOutcome::Rejected => -0.4,
        };

        if turn.content.matches("```").count() >= 2 {
            score += 0.05;
        }

        let has_evidence = turn.content.split_whitespace().any(|w| {
            let word = w.trim_matches(|c: char| !c.is_alphanumeric());
            word.eq_ignore_ascii_case("evidence") || word.eq_ignore_ascii_case("proof")
        });
        if has_evidence {
            score += 0.05;
        }

        if let Some(certainty) = turn.certainty {
            score += certainty * 0.1;
        }

        score.clamp(0.0, 1.0)
    }
}

pub struct ConvergenceMonitor;

impl ConvergenceMonitor {
    pub fn should_continue(sigma: &ConversationState) -> bool {
        if sigma.turns.len() < 3 {
            return true;
        }

        let recent_p: Vec<f64> = sigma
            .turns
            .iter()
            .rev()
            .take(3)
            .map(|_| sigma.completion_probability)
            .collect();
        let velocity = recent_p[0] - recent_p[recent_p.len() - 1];

        if sigma.completion_probability > 0.98 {
            return false;
        }
        if velocity < 0.01 && sigma.turns.len() > 10 {
            return false;
        }

        true
    }
}

pub struct ContextBudgeter;

impl ContextBudgeter {
    #[must_use]
    pub fn allocate(available_tokens: usize, segments: &[(&str, usize)]) -> Vec<usize> {
        let total_weight: usize = segments.iter().map(|s| s.1).sum();
        if total_weight == 0 {
            let n = segments.len().max(1);
            return vec![available_tokens / n; segments.len()];
        }

        let mut allocation: Vec<usize> = segments
            .iter()
            .map(|s| (s.1 * available_tokens) / total_weight)
            .collect();

        let allocated_total: usize = allocation.iter().sum();
        let remainder = available_tokens - allocated_total;
        if remainder > 0 && !allocation.is_empty() {
            let last_idx = allocation.len() - 1;
            allocation[last_idx] += remainder;
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
            return Err(anyhow!("No ensemble candidates available for {:?}", category));
        }

        if Self::is_safety_critical(category) {
            let high_confidence = candidates
                .iter()
                .filter(|(_, s)| *s > 0.8)
                .count();
            if high_confidence < 3 {
                return Err(anyhow!(
                    "Safety-critical task requires 3 models with confidence > 0.8, got {}",
                    high_confidence
                ));
            }
        }

        let mut ranked: Vec<(String, f64)> = match self.voting_strategy {
            VotingStrategy::MaxConfidence => candidates,
            VotingStrategy::Majority => candidates
                .into_iter()
                .filter(|(_, s)| *s >= 0.5)
                .collect(),
            VotingStrategy::WeightedConsensus => {
                let total: f64 = candidates.iter().map(|(_, s)| s).sum();
                if total == 0.0 {
                    candidates
                } else {
                    candidates.into_iter().map(|(m, s)| (m, s / total)).collect()
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
            let avg = profile.task_scores.get(&cat).map(|ra| ra.mean).unwrap_or(0.5);
            format!("{} | {:?} mean: {:.2} | {} turns", profile.model_id, cat, avg, profile.total_turns)
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
            matching.iter().find(|t| t.is_corrective()).copied()
                .or_else(|| matching.first().copied())
        } else {
            matching.iter().find(|t| !t.is_corrective()).copied()
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
        self.ema.get(model_id).map(|v| if v.is_finite() { *v as u64 } else { 0u64 }).unwrap_or(0)
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
        let variance = hist.iter().map(|&x| {
            let d = x as f64 - mean;
            d * d
        }).sum::<f64>() / hist.len() as f64;
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
        let prior = *it.next().unwrap();
        latest - prior
    }

    #[must_use]
    pub fn is_stalled(&self) -> bool {
        self.velocity_history.len() >= 3
            && self.velocity_history.iter().rev().take(3).all(|&v| v.abs() < 0.005)
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
        frontier.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));
        frontier
    }

    /// From the frontier, select the cheapest point meeting both constraints.
    #[must_use]
    pub fn select(frontier: &[ParetoPoint], min_quality: f64, max_tokens: u32) -> Option<&ParetoPoint> {
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
