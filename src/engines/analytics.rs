use crate::types::analytics::{
    AgentPerformanceReport, AnalyticsReport, ConvergenceDiagnostic, MetaLearningInsight,
    QualityTrend, Recommendation,
};
use crate::types::conversation::{ConversationState, TurnOutcome};
use anyhow::Result;
use sled::Db;
use std::collections::HashMap;

pub struct AnalyticsEngine;

impl AnalyticsEngine {
    pub fn generate_report(sigma: &ConversationState) -> AnalyticsReport {
        let convergence = Self::analyze_convergence(sigma);
        let agent_performances = Self::profile_agents(sigma);
        let recommendations = StrategyRecommender::recommend(sigma);
        AnalyticsReport {
            session_id: sigma.session_id.clone(),
            convergence,
            agent_performances,
            recommendations,
            timestamp: ConversationState::now(),
        }
    }

    fn analyze_convergence(sigma: &ConversationState) -> ConvergenceDiagnostic {
        let n = sigma.turns.len();
        let mut velocity = 0.0;
        let mut quality_trend = 0.0;

        if n >= 5 {
            let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
            let y: Vec<f64> = sigma
                .turns
                .iter()
                .rev()
                .take(5)
                .map(|t| t.certainty.unwrap_or(sigma.completion_probability))
                .collect();
            velocity = Self::linear_regression_slope(&x, &y);
            quality_trend = velocity * 0.8;
        } else if n >= 2 {
            velocity = sigma.completion_probability - 0.5;
        }

        // Δα magnitude trend from artifact diff sizes
        let delta_trend = Self::compute_delta_trend(sigma);
        let semantic_delta = SemanticConvergenceDetector::calculate_semantic_delta(sigma);

        let mut blockers = vec![];
        // Stuck goal: high iterations, low convergence
        if sigma.completion_probability < 0.5 && n > 10 {
            blockers.push("StuckGoal: low convergence velocity after 10+ turns".to_string());
        }
        if semantic_delta > 0.4 && n > 5 {
            blockers.push(format!(
                "SemanticDivergence: delta={:.2} exceeds threshold",
                semantic_delta
            ));
        }
        // Capability mismatch: one agent dominates failures
        let failure_agents = Self::dominant_failure_agent(sigma);
        if let Some(agent) = failure_agents {
            blockers.push(format!(
                "CapabilityMismatch: agent '{agent}' drives most failures"
            ));
        }
        // Conflicting proposals: many RolledBack turns
        let rollback_rate = sigma
            .turns
            .iter()
            .filter(|t| t.outcome == TurnOutcome::RolledBack)
            .count() as f64
            / n.max(1) as f64;
        if rollback_rate > 0.3 {
            blockers.push(format!(
                "ConflictingProposals: {:.0}% of turns rolled back",
                rollback_rate * 100.0
            ));
        }
        if n > 20 && sigma.completion_probability < 0.9 {
            blockers
                .push("HighIterationCount: session has not converged after 20+ turns".to_string());
        }

        ConvergenceDiagnostic {
            velocity,
            delta_trend,
            quality_trend,
            blockers,
        }
    }

    fn compute_delta_trend(sigma: &ConversationState) -> f64 {
        let sizes: Vec<f64> = sigma
            .turns
            .iter()
            .rev()
            .take(10)
            .flat_map(|t| t.diffs.iter().map(|(_, d)| d.diff_text.len() as f64))
            .collect();
        if sizes.len() < 2 {
            return 0.0;
        }
        let x: Vec<f64> = (0..sizes.len()).map(|i| i as f64).collect();
        Self::linear_regression_slope(&x, &sizes)
    }

    fn dominant_failure_agent(sigma: &ConversationState) -> Option<String> {
        let mut failures: HashMap<&str, u32> = HashMap::new();
        let mut total_failures = 0u32;
        for turn in &sigma.turns {
            if matches!(
                turn.outcome,
                TurnOutcome::RolledBack | TurnOutcome::Rejected
            ) {
                *failures.entry(&turn.model_id).or_insert(0) += 1;
                total_failures += 1;
            }
        }
        if total_failures < 3 {
            return None;
        }
        failures
            .into_iter()
            .find(|(_, count)| *count as f64 / total_failures as f64 > 0.6)
            .map(|(id, _)| id.to_string())
    }

    fn profile_agents(sigma: &ConversationState) -> Vec<AgentPerformanceReport> {
        let mut stats: HashMap<String, (u32, u32, Vec<f64>)> = HashMap::new();

        for turn in &sigma.turns {
            let entry = stats.entry(turn.model_id.clone()).or_insert((0, 0, vec![]));
            entry.0 += 1;
            if turn.outcome == TurnOutcome::TestsPassed || turn.outcome == TurnOutcome::Compiled {
                entry.1 += 1;
            }
            entry.2.push(turn.certainty.unwrap_or(0.5));
        }

        stats
            .into_iter()
            .map(|(id, (total, success, qualities))| {
                let avg_quality = if total > 0 {
                    qualities.iter().sum::<f64>() / total as f64
                } else {
                    0.0
                };
                let trend = if qualities.len() >= 3 {
                    let x: Vec<f64> = (0..qualities.len()).map(|i| i as f64).collect();
                    Self::linear_regression_slope(&x, &qualities)
                } else {
                    0.0
                };
                AgentPerformanceReport {
                    agent_id: id,
                    success_rate: if total > 0 {
                        f64::from(success) / f64::from(total)
                    } else {
                        0.0
                    },
                    avg_quality,
                    cost_per_turn: 0.01,
                    improvement_slope: trend,
                }
            })
            .collect()
    }

    fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
        let sum_xx = x.iter().map(|xi| xi * xi).sum::<f64>();
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator == 0.0 {
            return 0.0;
        }
        (n * sum_xy - sum_x * sum_y) / denominator
    }
}

pub struct QualityTrendDetector;

impl QualityTrendDetector {
    #[must_use]
    pub fn detect(scores: &[f64]) -> QualityTrend {
        if scores.len() < 3 {
            return QualityTrend::Plateau;
        }
        let x: Vec<f64> = (0..scores.len()).map(|i| i as f64).collect();
        let slope = AnalyticsEngine::linear_regression_slope(&x, scores);
        if slope > 0.01 {
            QualityTrend::Improving
        } else if slope < -0.01 {
            QualityTrend::Regressing
        } else {
            QualityTrend::Plateau
        }
    }
}

pub struct DecisionReplay;

impl DecisionReplay {
    #[must_use]
    pub fn reconstruct(sigma: &ConversationState, turn_index: u32) -> Option<String> {
        let turn = sigma.turns.iter().find(|t| t.index == turn_index)?;
        Some(format!(
            "Turn {}: agent={} outcome={:?} certainty={:.2} diffs={} context_turns={}",
            turn.index,
            turn.model_id,
            turn.outcome,
            turn.certainty.unwrap_or(0.0),
            turn.diffs.len(),
            sigma.turns.iter().filter(|t| t.index < turn_index).count(),
        ))
    }
}

pub struct StrategyRecommender;

impl StrategyRecommender {
    #[must_use]
    pub fn recommend(sigma: &ConversationState) -> Vec<Recommendation> {
        let mut recs = vec![];
        let n = sigma.turns.len();

        if n == 0 {
            return recs;
        }

        // Only count completed turns (not Unknown/dropped) to avoid credit errors and
        // agent drops inflating the failure rate before we have meaningful signal.
        let completed = sigma
            .turns
            .iter()
            .filter(|t| !matches!(t.outcome, TurnOutcome::Unknown))
            .count();

        if completed >= 3 {
            let success_rate = sigma
                .turns
                .iter()
                .filter(|t| matches!(t.outcome, TurnOutcome::TestsPassed | TurnOutcome::Compiled))
                .count() as f64
                / completed as f64;

            if success_rate < 0.4 {
                recs.push(Recommendation {
                    action: "switch_to_critique_protocol".to_string(),
                    expected_impact: 0.25,
                    confidence: 0.7,
                });
            }
        }

        if n > 15 && sigma.completion_probability < 0.6 {
            recs.push(Recommendation {
                action: "increase_context_window".to_string(),
                expected_impact: 0.15,
                confidence: 0.65,
            });
        }

        if sigma.budget.burn_rate() > 0.05 {
            recs.push(Recommendation {
                action: "reduce_parallel_inference".to_string(),
                expected_impact: 0.10,
                confidence: 0.80,
            });
        }

        recs
    }
}

pub struct MetaLearningEngine;

impl MetaLearningEngine {
    /// Computes multi-session insights, specifically tracking the Intelligence Growth Rate.
    /// Growth is defined by sessions becoming faster (fewer turns) and cheaper (fewer tokens) over time.
    #[must_use]
    pub fn compute_insight(sessions: &[&ConversationState]) -> MetaLearningInsight {
        let session_count = sessions.len();
        // Best model identification (works with any number of sessions)
        let mut model_wins: HashMap<String, u32> = HashMap::new();
        for s in sessions {
            for t in &s.turns {
                if t.outcome == TurnOutcome::TestsPassed {
                    *model_wins.entry(t.model_id.clone()).or_insert(0) += 1;
                }
            }
        }
        let best_model = model_wins
            .into_iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, _)| k);

        if session_count < 2 {
            return MetaLearningInsight {
                session_count,
                avg_turns_to_convergence: sessions
                    .first()
                    .map(|s| s.turns.len() as f64)
                    .unwrap_or(0.0),
                quality_growth_rate: 0.0,
                best_model,
            };
        }

        // 1. Turns trend (Velocity of learning)
        let turns_y: Vec<f64> = sessions.iter().map(|s| s.turns.len() as f64).collect();
        let x: Vec<f64> = (0..session_count).map(|i| i as f64).collect();
        let turns_slope = AnalyticsEngine::linear_regression_slope(&x, &turns_y);

        // 2. Quality growth
        let quality_y: Vec<f64> = sessions.iter().map(|s| s.completion_probability).collect();
        let quality_growth = AnalyticsEngine::linear_regression_slope(&x, &quality_y);

        MetaLearningInsight {
            session_count,
            avg_turns_to_convergence: turns_y.iter().sum::<f64>() / session_count as f64,
            quality_growth_rate: quality_growth - (turns_slope * 0.1), // Penalize slow convergence
            best_model,
        }
    }
}

pub struct SemanticConvergenceDetector;

impl SemanticConvergenceDetector {
    #[must_use]
    pub fn calculate_semantic_delta(sigma: &ConversationState) -> f64 {
        if sigma.turns.len() < 2 {
            return 1.0;
        }
        let current_turn = &sigma.turns[sigma.turns.len() - 1];
        let prev_turn = &sigma.turns[sigma.turns.len() - 2];

        let current_vec = crate::engines::memory::embed_text(&current_turn.content);
        let prev_vec = crate::engines::memory::embed_text(&prev_turn.content);

        let sim = crate::engines::memory::cosine_sim(&current_vec, &prev_vec);
        (1.0 - f64::from(sim)).clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn is_converged(sigma: &ConversationState, threshold: f64) -> bool {
        Self::calculate_semantic_delta(sigma) < threshold
    }
}

pub struct FailureTaxonomy;

impl FailureTaxonomy {
    pub fn categorize(error: &str) -> String {
        let err_lower = error.to_lowercase();
        if err_lower.contains("mismatched types") || err_lower.contains("expected") {
            "TypeError".to_string()
        } else if err_lower.contains("timeout") {
            "Timeout".to_string()
        } else if err_lower.contains("panic") {
            "Panic".to_string()
        } else if err_lower.contains("circular") || err_lower.contains("tautology") {
            "CircularReasoning".to_string()
        } else if err_lower.contains("regression") {
            "QualityRegression".to_string()
        } else {
            "LogicError".to_string()
        }
    }
}

pub struct FailureStore {
    db: Db,
}

impl FailureStore {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn record(&self, error: &str) -> Result<()> {
        let category = FailureTaxonomy::categorize(error);
        let tree = self.db.open_tree("failure_counts")?;
        let current: u64 = tree
            .get(category.as_bytes())?
            .and_then(|v| v.as_ref().try_into().ok().map(u64::from_be_bytes))
            .unwrap_or(0);
        tree.insert(category.as_bytes(), &(current + 1).to_be_bytes())?;
        self.db.flush()?;
        Ok(())
    }

    pub fn rates(&self) -> Result<HashMap<String, u64>> {
        let tree = self.db.open_tree("failure_counts")?;
        let mut out = HashMap::new();
        for item in tree.iter() {
            let (k, v) = item?;
            let key = String::from_utf8(k.to_vec())?;
            let count = v.as_ref().try_into().map(u64::from_be_bytes).unwrap_or(0);
            out.insert(key, count);
        }
        Ok(out)
    }
}
