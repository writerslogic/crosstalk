use crate::types::{TurnStructure, TaskCategory, ArtifactDiff};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FallacyReport {
    pub fallacy_type: String,
    pub evidence_span: String,
    pub confidence: f64,
}

pub struct ReasoningEngine;

impl ReasoningEngine {
    pub fn select_structure(_task: TaskCategory, _agent_id: &str) -> TurnStructure {
        // Simple default: StepByStep for complex tasks
        TurnStructure::StepByStep
    }
}

pub struct FallacyDetector;

impl FallacyDetector {
    pub fn scan(content: &str) -> Vec<FallacyReport> {
        let mut reports = vec![];
        let content_lower = content.to_lowercase();

        // 1. Circular Reasoning (Mock regex/similarity)
        if content_lower.contains("therefore") && content_lower.len() < 100 {
            reports.push(FallacyReport {
                fallacy_type: "CircularReasoning".to_string(),
                evidence_span: content.to_string(),
                confidence: 0.7,
            });
        }

        // 2. Appeal to Authority
        if content_lower.contains("as the documentation says") && !content_lower.contains("```") {
            reports.push(FallacyReport {
                fallacy_type: "AppealToAuthority".to_string(),
                evidence_span: "as the documentation says".to_string(),
                confidence: 0.8,
            });
        }

        reports
    }
}

pub struct SignalExtractor;

impl SignalExtractor {
    pub fn extract_decisions(content: &str) -> Vec<String> {
        let mut decisions = vec![];
        for line in content.lines() {
            if line.to_lowercase().starts_with("decision:") || line.to_lowercase().starts_with("decide:") {
                decisions.push(line[9..].trim().to_string());
            }
        }
        decisions
    }
}

pub struct SynthesisEngine;

impl SynthesisEngine {
    pub fn merge(proposals: Vec<ArtifactDiff>) -> Option<ArtifactDiff> {
        if proposals.is_empty() { return None; }
        if proposals.len() == 1 { return Some(proposals[0].clone()); }

        // Simplified synthesis: take the longest diff as "best" for now
        let mut best = &proposals[0];
        for p in &proposals {
            if p.diff_text.len() > best.diff_text.len() {
                best = p;
            }
        }
        Some(best.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallacy_detection() {
        let content = "As the documentation says, this is correct.";
        let reports = FallacyDetector::scan(content);
        assert!(!reports.is_empty());
        assert_eq!(reports[0].fallacy_type, "AppealToAuthority");
    }

    #[test]
    fn test_signal_extraction() {
        let content = "Decision: Use SHA256 for hashing\nDecide: Implement LanceDB";
        let decisions = SignalExtractor::extract_decisions(content);
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0], "Use SHA256 for hashing");
    }
}
