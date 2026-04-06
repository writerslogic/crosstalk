use crate::types::conversation::TaskCategory;
use crate::types::conversation::{ConversationState, Turn, TurnOutcome};
use crate::types::intelligence::{ModelProfile, PromptTemplate, RegressionAlert};
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Default)]
pub struct IntelligenceEngine {
    pub profiles: HashMap<String, ModelProfile>,
    pub templates: Vec<PromptTemplate>,
    pub storage_path: Option<String>,
}

impl IntelligenceEngine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_storage(path: &str) -> Self {
        let mut engine = Self::new();
        engine.storage_path = Some(path.to_string());
        let _ = engine.load_profiles();
        engine
    }

    pub fn load_profiles(&mut self) -> Result<()> {
        if let Some(path) = &self.storage_path
            && Path::new(path).exists()
        {
            let content = fs::read_to_string(path)?;
            let data: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(profiles) = data.get("profiles") {
                self.profiles = serde_json::from_value(profiles.clone())?;
            }
            if let Some(templates) = data.get("templates") {
                self.templates = serde_json::from_value(templates.clone())?;
            }
        }
        Ok(())
    }

    pub fn save_all(&self) -> Result<()> {
        if let Some(path) = &self.storage_path {
            let data = serde_json::json!({
                "profiles": self.profiles,
                "templates": self.templates,
            });
            let content = serde_json::to_string_pretty(&data)?;
            fs::write(path, content)?;
        }
        Ok(())
    }

    pub fn update_profile(&mut self, turn: &Turn, quality_score: f64) {
        let profile = self
            .profiles
            .entry(turn.model_id.clone())
            .or_insert(ModelProfile {
                model_id: turn.model_id.clone(),
                task_scores: HashMap::new(),
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

        let _ = self.save_all();
    }

    pub fn update_profile_with_latency(
        &mut self,
        turn: &Turn,
        quality_score: f64,
        latency_ms: u64,
    ) {
        self.update_profile(turn, quality_score);
        if let Some(profile) = self.profiles.get_mut(&turn.model_id) {
            profile.latency_ms.update(latency_ms as f64);
        }
        let _ = self.save_all();
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
        for turn in recent_turns {
            if turn.model_id == model_id {
                let score = QualityScorer::score(turn, &ConversationState::new("temp"));
                recent_quality_sum += score;
                valid_turns += 1;
            }
        }

        if valid_turns == 0 {
            return None;
        }

        let recent_avg = recent_quality_sum / valid_turns as f64;

        let baseline = if profile.task_scores.is_empty() {
            0.5
        } else {
            let mut baseline_sum = 0.0;
            for (_, avg) in &profile.task_scores {
                baseline_sum += avg.mean;
            }
            baseline_sum / profile.task_scores.len() as f64
        };

        if recent_avg < baseline * 0.9 {
            let alert = RegressionAlert {
                agent_id: model_id.to_string(),
                task_category: TaskCategory::CodeGeneration,
                baseline_mean: baseline,
                recent_mean: recent_avg,
                severity: (baseline - recent_avg) / baseline,
                timestamp: ConversationState::now(),
            };
            return Some(alert);
        }

        None
    }

    #[must_use]
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

    pub fn route_task_constrained(
        &self,
        category: TaskCategory,
        available_models: &[String],
        budget: u32,
        latency_ms: u64,
        blacklist: &[String],
    ) -> Result<String, String> {
        if available_models.is_empty() {
            return Err("No models available for routing".to_string());
        }

        let estimated_tokens = Self::estimate_tokens(category);
        if estimated_tokens > budget {
            return Err(format!(
                "Estimated tokens {} exceeds budget {}",
                estimated_tokens, budget
            ));
        }

        let mut candidates: Vec<(String, f64)> = Vec::new();

        for model_id in available_models {
            if blacklist.contains(model_id) {
                continue;
            }

            if let Some(profile) = self.profiles.get(model_id) {
                if profile.latency_ms.mean > latency_ms as f64 {
                    continue;
                }

                let score = profile
                    .task_scores
                    .get(&category)
                    .map(|ra| ra.mean)
                    .unwrap_or(0.5);

                candidates.push((model_id.clone(), score));
            }
        }

        if candidates.is_empty() {
            let diagnostics = self.generate_routing_diagnostics(
                category,
                available_models,
                budget,
                latency_ms,
                blacklist,
            );
            return Err(format!("No models satisfy constraints. {}", diagnostics));
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates[0].0.clone())
    }

    fn generate_routing_diagnostics(
        &self,
        category: TaskCategory,
        available_models: &[String],
        budget: u32,
        latency_ms: u64,
        blacklist: &[String],
    ) -> String {
        let estimated_tokens = Self::estimate_tokens(category);
        let mut diagnostics = format!(
            "Task category: {:?}, Estimated tokens: {}, Budget: {}, Latency limit: {}ms. ",
            category, estimated_tokens, budget, latency_ms
        );

        diagnostics.push_str("Models: ");
        for model_id in available_models {
            if blacklist.contains(model_id) {
                diagnostics.push_str(&format!("{} (blacklisted), ", model_id));
            } else if let Some(profile) = self.profiles.get(model_id) {
                let latency_str = if profile.latency_ms.mean > latency_ms as f64 {
                    format!("{}ms > limit", profile.latency_ms.mean as u64)
                } else {
                    format!("{}ms OK", profile.latency_ms.mean as u64)
                };
                let quality = profile
                    .task_scores
                    .get(&category)
                    .map(|ra| ra.mean)
                    .unwrap_or(0.5);
                diagnostics.push_str(&format!("{} (quality: {:.2}, latency: {}), ", model_id, quality, latency_str));
            } else {
                diagnostics.push_str(&format!("{} (no profile), ", model_id));
            }
        }

        diagnostics
    }

    #[must_use]
    pub fn estimate_tokens(category: TaskCategory) -> u32 {
        match category {
            TaskCategory::CodeGeneration => 2000,
            TaskCategory::Debugging => 1500,
            TaskCategory::Architecture => 2500,
            TaskCategory::Refactoring => 1800,
            TaskCategory::Research => 2200,
            TaskCategory::Testing => 1500,
        }
    }
}

pub struct QualityScorer;

impl QualityScorer {
    #[must_use]
    pub fn score(turn: &Turn, sigma: &ConversationState) -> f64 {
        let mut score: f64 = 0.5;

        match turn.outcome {
            TurnOutcome::TestsPassed => score += 0.4,
            TurnOutcome::Compiled => score += 0.2,
            TurnOutcome::AdvancedConvergence => score += 0.1,
            TurnOutcome::RolledBack | TurnOutcome::Rejected => score -= 0.4,
            TurnOutcome::Stalled => score -= 0.1,
            TurnOutcome::Unknown => {}
        }

        let mut max_similarity = 0.0;
        for prev in &sigma.turns {
            if prev.index < turn.index {
                let sim = Self::content_similarity(&turn.content, &prev.content);
                if sim > max_similarity {
                    max_similarity = sim;
                }
            }
        }
        score -= (max_similarity - 0.8).max(0.0);

        if turn.content.contains("```") {
            score += 0.05;
        }
        if turn.content.contains("because") || turn.content.contains("evidence") {
            score += 0.05;
        }

        score.clamp(0.0, 1.0)
    }

    fn content_similarity(a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }
        let words_a: std::collections::HashSet<_> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<_> = b.split_whitespace().collect();
        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }
        let intersect = words_a.intersection(&words_b).count();
        intersect as f64 / words_a.len().max(words_b.len()) as f64
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
