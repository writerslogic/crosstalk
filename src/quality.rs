use crate::types::Artifact;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use petgraph::graph::DiGraph;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ArtifactMetrics {
    pub cyclomatic_complexity: u32,
    pub coupling: u32,
    pub cohesion: f64,
    pub line_count: u32,
    pub comment_density: f64,
}

pub struct QualityEngine;

impl QualityEngine {
    pub fn analyze_artifact(artifact: &Artifact) -> ArtifactMetrics {
        let lines: Vec<&str> = artifact.content.lines().collect();
        let line_count = lines.len() as u32;
        
        let comment_lines = lines.iter().filter(|l| l.trim().starts_with("//") || l.trim().starts_with("/*")).count();
        let comment_density = if line_count > 0 { comment_lines as f64 / line_count as f64 } else { 0.0 };

        // Simplified complexity: count branch keywords
        let branch_keywords = ["if", "for", "while", "match", "&&", "||"];
        let mut complexity = 1;
        for line in &lines {
            for kw in branch_keywords {
                if line.contains(kw) { complexity += 1; }
            }
        }

        ArtifactMetrics {
            cyclomatic_complexity: complexity,
            coupling: 0,
            cohesion: 1.0,
            line_count,
            comment_density,
        }
    }

    pub fn detect_duplication(new_content: &str, existing_artifacts: &HashMap<String, Artifact>) -> Vec<(String, f64)> {
        let mut duplicates = vec![];
        for (name, art) in existing_artifacts {
            let common = new_content.chars().filter(|c| art.content.contains(*c)).count();
            let ratio = if !new_content.is_empty() { common as f64 / new_content.len() as f64 } else { 0.0 };
            if ratio > 0.8 {
                duplicates.push((name.clone(), ratio));
            }
        }
        duplicates
    }

    pub fn build_dependency_graph(artifacts: &HashMap<String, Artifact>) -> DiGraph<String, ()> {
        let mut graph = DiGraph::new();
        let mut nodes = HashMap::new();

        for name in artifacts.keys() {
            let idx = graph.add_node(name.clone());
            nodes.insert(name.clone(), idx);
        }

        for (name, art) in artifacts {
            for other_name in artifacts.keys() {
                if name != other_name && art.content.contains(other_name) {
                    if let (Some(&u), Some(&v)) = (nodes.get(name), nodes.get(other_name)) {
                        graph.add_edge(u, v, ());
                    }
                }
            }
        }
        graph
    }
}

pub struct RegressionDetector;

impl RegressionDetector {
    pub fn is_regressive(old: &ArtifactMetrics, new: &ArtifactMetrics) -> bool {
        if new.cyclomatic_complexity > old.cyclomatic_complexity + 5 { return true; }
        if new.comment_density < old.comment_density - 0.1 { return true; }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_complexity_counting() {
        let art = Artifact {
            name: "test.rs".to_string(),
            language: "rust".to_string(),
            content: "fn main() { if true { if false {} } }".to_string(),
            version: 1,
            history: vec![],
            ast_versions: HashMap::new(),
            proof_attachments: vec![],
            metrics: ArtifactMetrics::default(),
        };
        let metrics = QualityEngine::analyze_artifact(&art);
        assert!(metrics.cyclomatic_complexity >= 3);
    }

    #[test]
    fn test_regression_detection() {
        let old = ArtifactMetrics { cyclomatic_complexity: 5, comment_density: 0.2, ..Default::default() };
        let mut new = old.clone();
        new.cyclomatic_complexity = 15;
        assert!(RegressionDetector::is_regressive(&old, &new));
    }
}
