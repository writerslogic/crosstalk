use crate::types::analytics::{AgentPerformanceReport, AnalyticsReport, ConvergenceDiagnostic};
use crate::types::conversation::{ConversationState, TurnOutcome};
use std::collections::HashMap;

pub struct AnalyticsEngine;

impl AnalyticsEngine {
    pub fn generate_report(sigma: &ConversationState) -> AnalyticsReport {
        let convergence = Self::analyze_convergence(sigma);
        let agent_performances = Self::profile_agents(sigma);

        AnalyticsReport {
            session_id: sigma.session_id.clone(),
            convergence,
            agent_performances,
            timestamp: ConversationState::now(),
        }
    }

    fn analyze_convergence(sigma: &ConversationState) -> ConvergenceDiagnostic {
        let n = sigma.turns.len();
        let mut velocity = 0.0;
        let mut quality_trend = 0.0;

        if n >= 5 {
            // Simple linear regression for P(C) velocity over last 5 turns
            let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
            let y: Vec<f64> = sigma
                .turns
                .iter()
                .rev()
                .take(5)
                .map(|_| sigma.completion_probability)
                .collect();
            velocity = Self::linear_regression_slope(&x, &y);

            // Placeholder for quality trend
            quality_trend = velocity * 0.8;
        } else if n >= 2 {
            velocity = sigma.completion_probability - 0.5; // Simplified initial velocity
        }

        let mut blockers = vec![];
        if sigma.completion_probability < 0.5 && n > 10 {
            blockers.push("Low convergence velocity: session may be stuck".to_string());
        }
        if n > 20 && sigma.completion_probability < 0.9 {
            blockers.push("High iteration count without convergence".to_string());
        }

        ConvergenceDiagnostic {
            velocity,
            delta_trend: 0.0, // Needs Δα magnitude tracking
            quality_trend,
            blockers,
        }
    }

    fn profile_agents(sigma: &ConversationState) -> Vec<AgentPerformanceReport> {
        let mut stats: HashMap<String, (u32, u32, Vec<f64>)> = HashMap::new();

        for turn in &sigma.turns {
            let entry = stats.entry(turn.model_id.clone()).or_insert((0, 0, vec![]));
            entry.0 += 1;
            if turn.outcome == TurnOutcome::TestsPassed || turn.outcome == TurnOutcome::Compiled {
                entry.1 += 1;
            }
            // Use certainty as a proxy for quality if not explicitly stored
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
                    cost_per_turn: 0.01, // Default cost
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
