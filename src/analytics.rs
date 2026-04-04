use crate::types::{ConversationState, Turn, TurnOutcome, ConvergenceDiagnostic, AgentPerformanceReport, AnalyticsReport};
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
        let mut velocity = 0.0;
        if sigma.turns.len() >= 2 {
            let p_new = sigma.completion_probability;
            let p_old = sigma.turns[sigma.turns.len()-2].certainty.unwrap_or(0.0);
            velocity = p_new - p_old;
        }

        let mut blockers = vec![];
        if sigma.completion_probability < 0.5 && sigma.turns.len() > 10 {
            blockers.push("Low convergence velocity".to_string());
        }

        ConvergenceDiagnostic {
            velocity,
            delta_trend: 0.0, // Simplified
            quality_trend: 0.0,
            blockers,
        }
    }

    fn profile_agents(sigma: &ConversationState) -> Vec<AgentPerformanceReport> {
        let mut stats: HashMap<String, (u32, u32, f64)> = HashMap::new();

        for turn in &sigma.turns {
            let entry = stats.entry(turn.model_id.clone()).or_insert((0, 0, 0.0));
            entry.0 += 1;
            if turn.outcome == TurnOutcome::TestsPassed || turn.outcome == TurnOutcome::Compiled {
                entry.1 += 1;
            }
            // Quality score from Phase 10 logic (mocked here)
            entry.2 += 0.8; 
        }

        stats.into_iter().map(|(id, (total, success, quality))| {
            AgentPerformanceReport {
                agent_id: id,
                success_rate: if total > 0 { f64::from(success) / f64::from(total) } else { 0.0 },
                avg_quality: if total > 0 { quality / f64::from(total) } else { 0.0 },
                cost_per_turn: 0.01,
                improvement_slope: 0.0,
            }
        }).collect()
    }
}

pub struct FailureTaxonomy;

impl FailureTaxonomy {
    pub fn categorize(error: &str) -> String {
        if error.contains("mismatched types") { "TypeError".to_string() }
        else if error.contains("timeout") { "Timeout".to_string() }
        else if error.contains("panic") { "Panic".to_string() }
        else { "Unknown".to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_velocity() {
        let mut sigma = ConversationState::new("test");
        sigma.completion_probability = 0.8;
        // Mock a turn to allow velocity calc
        sigma.turns.push(Turn {
            index: 0,
            model_id: "m".into(),
            content: "c".into(),
            timestamp: 0,
            diffs: vec![],
            certainty: Some(0.5),
            outcome: TurnOutcome::Compiled,
            task_category: None,
            structure: None,
            signature: vec![],
        });
        sigma.turns.push(Turn {
            index: 1,
            model_id: "m".into(),
            content: "c".into(),
            timestamp: 0,
            diffs: vec![],
            certainty: Some(0.8),
            outcome: TurnOutcome::Compiled,
            task_category: None,
            structure: None,
            signature: vec![],
        });
        
        let report = AnalyticsEngine::generate_report(&sigma);
        assert!(report.convergence.velocity > 0.2);
    }
}
