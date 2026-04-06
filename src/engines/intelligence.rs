use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome};
use crate::types::intelligence::{ModelProfile, PromptTemplate, RegressionAlert};
use anyhow::{Context, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};

pub struct IntelligenceEngine {
    /// Upgraded to DashMap for lock-free concurrent updates across the Swarm
    pub profiles: Arc<DashMap<String, ModelProfile>>,
    /// Templates rarely change at runtime, RwLock optimizes for heavy concurrent reads
    pub templates: Arc<RwLock<Vec<PromptTemplate>>>,
    pub storage_path: Option<String>,
    /// Background channel for non-blocking disk writes
    flush_tx: Option<mpsc::Sender<()>>,
}

impl IntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(DashMap::new()),
            templates: Arc::new(RwLock::new(Vec::new())),
            storage_path: None,
            flush_tx: None,
        }
    }

    /// Initializes the engine, loads state asynchronously, and spawns the Checkpoint Background Actor.
    pub async fn with_storage(path: &str) -> Result<Self> {
        let mut engine = Self {
            profiles: Arc::new(DashMap::new()),
            templates: Arc::new(RwLock::new(Vec::new())),
            storage_path: Some(path.to_string()),
            flush_tx: None,
        };

        engine.load_profiles().await?;
        engine.spawn_checkpoint_actor();

        Ok(engine)
    }

    /// Asynchronous, non-blocking file read.
    pub async fn load_profiles(&self) -> Result<()> {
        if let Some(path) = &self.storage_path {
            if tokio::fs::try_exists(path).await.unwrap_or(false) {
                let content = fs::read_to_string(path).await.context("Failed to read intelligence data")?;
                let data: serde_json::Value = serde_json::from_str(&content)?;
                
                if let Some(profiles) = data.get("profiles") {
                    let parsed: HashMap<String, ModelProfile> = serde_json::from_value(profiles.clone())?;
                    for (k, v) in parsed {
                        self.profiles.insert(k, v);
                    }
                }
                if let Some(templates) = data.get("templates") {
                    *self.templates.write().await = serde_json::from_value(templates.clone())?;
                }
            }
        }
        Ok(())
    }

    /// Spawns a background actor that "debounces" disk writes.
    /// Instead of writing to disk 1,000 times a second, it wakes up periodically,
    /// checks if a write was requested, and dumps the state exactly once.
    fn spawn_checkpoint_actor(&mut self) {
        if let Some(path) = self.storage_path.clone() {
            let (tx, mut rx) = mpsc::channel(1);
            self.flush_tx = Some(tx);
            let profiles_ref = Arc::clone(&self.profiles);
            let templates_ref = Arc::clone(&self.templates);

            tokio::spawn(async move {
                let mut ticker = interval(Duration::from_secs(5)); // Debounce window

                loop {
                    tokio::select! {
                        Some(_) = rx.recv() => {
                            // A write was requested. Wait for the tick to batch changes.
                            ticker.tick().await;
                            
                            // Drain any subsequent requests that piled up during the wait
                            while rx.try_recv().is_ok() {}

                            let profiles_map: HashMap<String, ModelProfile> = profiles_ref
                                .iter()
                                .map(|entry| (entry.key().clone(), entry.value().clone()))
                                .collect();
                            let data = serde_json::json!({
                                "profiles": profiles_map,
                                "templates": &*templates_ref.read().await,
                            });
                            
                            if let Ok(content) = serde_json::to_string_pretty(&data) {
                                // Write to a temp file and rename to prevent corruption on crash
                                let temp_path = format!("{}.tmp", path);
                                if fs::write(&temp_path, content).await.is_ok() {
                                    let _ = fs::rename(&temp_path, &path).await;
                                }
                            }
                        }
                        else => break,
                    }
                }
            });
        }
    }

    /// Thread-safe, non-blocking profile update.
    pub fn update_profile(&self, turn: &Turn, quality_score: f64) {
        let mut profile = self.profiles.entry(turn.model_id.clone()).or_insert_with(|| ModelProfile {
            model_id: turn.model_id.clone(),
            task_scores: HashMap::new(),
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
        self.update_profile(turn, quality_score);
        if let Some(mut profile) = self.profiles.get_mut(&turn.model_id) {
            profile.latency_ms.update(latency_ms as f64);
        }
    }

    /// Non-blocking signal to the background writer.
    fn trigger_save(&self) {
        if let Some(tx) = &self.flush_tx {
            let _ = tx.try_send(()); // try_send ignores if the channel is full (already queued)
        }
    }

    pub fn detect_regression(&self, model_id: &str, recent_turns: &[Turn]) -> Option<RegressionAlert> {
        if recent_turns.is_empty() { return None; }

        let profile = self.profiles.get(model_id)?;
        let mut recent_quality_sum = 0.0;
        let mut valid_turns = 0;

        for turn in recent_turns {
            if turn.model_id == model_id {
                let score = QualityScorer::score(turn, &ConversationState::new("temp"));
                recent_quality_sum += score;
                valid_turns += 1;
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
            return Some(RegressionAlert {
                agent_id: model_id.to_string(),
                task_category: TaskCategory::CodeGeneration,
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
                if profile.latency_ms.mean > latency_ms as f64 { continue; }

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
            if let Some(profile) = self.profiles.get(model) {
                if profile.latency_ms.mean > latency_ms as f64 {
                    issues.push(format!("{} latency {}ms exceeds {}ms limit", model, profile.latency_ms.mean as u64, latency_ms));
                }
            }
        }

        if issues.is_empty() {
            "No specific routing issues identified.".to_string()
        } else {
            format!("Issues: {}", issues.join("; "))
        }
    }

    #[must_use]
    pub fn estimate_tokens(category: TaskCategory) -> u32 {
        match category {
            TaskCategory::Architecture => 2500,
            TaskCategory::Research => 2200,
            TaskCategory::CodeGeneration => 2000,
            TaskCategory::Refactoring => 1800,
            TaskCategory::Debugging | TaskCategory::Testing => 1500,
        }
    }
}

pub struct QualityScorer;

impl QualityScorer {
    #[must_use]
    pub fn score(turn: &Turn, _sigma: &ConversationState) -> f64 {
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

        if turn.content.contains("```") {
            score += 0.05;
        }

        if turn.content.contains("evidence") || turn.content.contains("proof") {
            score += 0.05;
        }

        if let Some(certainty) = turn.certainty {
            score += certainty * 0.1;
        }

        score.max(0.0).min(1.0)
    }
}