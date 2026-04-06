use crate::types::artifact::Artifact;
use anyhow::{Result, anyhow};
use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use tree_sitter::Parser;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ArtifactMetrics {
    pub line_count: u32,
    pub cyclomatic_complexity: u32,
    pub comment_density: f64,
    pub coupling: u32,
}

pub struct QualityEngine;

impl QualityEngine {
    pub fn analyze_artifact(
        artifact: &Artifact,
        other_artifact_names: &[String],
    ) -> ArtifactMetrics {
        let lines: Vec<&str> = artifact.content.lines().collect();
        let line_count = lines.len() as u32;

        let comment_lines = lines
            .iter()
            .filter(|l: &&&str| l.trim().starts_with("//") || l.trim().starts_with("/*"))
            .count();
        let comment_density = if line_count > 0 {
            comment_lines as f64 / line_count as f64
        } else {
            0.0
        };

        let mut complexity = 1;
        if artifact.language.to_lowercase() == "rust" || artifact.language.to_lowercase() == "rs" {
            complexity = Self::calculate_rust_complexity(&artifact.content);
        }

        let coupling = Self::calculate_coupling(artifact, other_artifact_names);

        ArtifactMetrics {
            line_count,
            cyclomatic_complexity: complexity,
            comment_density,
            coupling,
        }
    }

    fn calculate_coupling(artifact: &Artifact, other_artifact_names: &[String]) -> u32 {
        let mut coupling = 0;
        let mut seen = std::collections::HashSet::new();
        for line in artifact.content.lines() {
            let trimmed = line.trim();
            // Look for imports or direct module references
            if (trimmed.starts_with("use ")
                || trimmed.starts_with("mod ")
                || trimmed.contains("::"))
                && !trimmed.starts_with("//")
            {
                for other in other_artifact_names {
                    let other_mod = other
                        .trim_end_matches(".rs")
                        .split('/')
                        .next_back()
                        .unwrap_or(other);
                    if other != &artifact.name
                        && trimmed.contains(other_mod)
                        && !seen.contains(other)
                    {
                        coupling += 1;
                        seen.insert(other.clone());
                    }
                }
            }
        }
        coupling
    }

    fn calculate_rust_complexity(content: &str) -> u32 {
        let mut parser = Parser::new();
        let _ = parser.set_language(&tree_sitter_rust::LANGUAGE.into());
        let tree = match parser.parse(content, None) {
            Some(t) => t,
            None => return 1,
        };

        let mut complexity = 1;
        let mut cursor = tree.root_node().walk();
        let mut stack = vec![tree.root_node()];

        while let Some(node) = stack.pop() {
            let kind = node.kind();
            if matches!(
                kind,
                "if_expression"
                    | "for_expression"
                    | "while_expression"
                    | "match_arm"
                    | "if_let_expression"
                    | "while_let_expression"
            ) {
                complexity += 1;
            }

            for child in node.children(&mut cursor) {
                stack.push(child);
            }
        }
        complexity
    }

    pub fn verify_compilation() -> Result<bool> {
        let output = Command::new("cargo").arg("check").output()?;

        Ok(output.status.success())
    }

    pub fn build_dependency_graph(artifacts: &HashMap<String, Artifact>) -> DiGraph<String, ()> {
        let mut graph = DiGraph::new();
        let mut nodes = HashMap::new();

        for name in artifacts.keys() {
            let idx = graph.add_node(name.clone());
            nodes.insert(name.clone(), idx);
        }

        for (name, artifact) in artifacts {
            let idx = nodes[name];
            for line in artifact.content.lines() {
                if line.trim().starts_with("use ") || line.trim().starts_with("mod ") {
                    for other_name in artifacts.keys() {
                        if name != other_name && line.contains(other_name.trim_end_matches(".rs"))
                            && let Some(&other_idx) = nodes.get(other_name)
                        {
                            graph.add_edge(idx, other_idx, ());
                        }
                    }
                }
            }
        }
        graph
    }
    pub fn detect_duplication(
        new_content: &str,
        existing: &HashMap<String, Artifact>,
    ) -> Vec<String> {
        let mut duplicates = vec![];
        for (name, artifact) in existing {
            if artifact.content.contains(new_content) && !new_content.is_empty() {
                duplicates.push(name.clone());
            }
        }
        duplicates
    }
}

pub struct ValidatorRegistry;

impl ValidatorRegistry {
    pub fn validate(content: &str, language: &str) -> Result<()> {
        match language.to_lowercase().as_str() {
            "json" => serde_json::from_str::<serde_json::Value>(content)
                .map(|_| ())
                .map_err(|e| anyhow!("JSON error: {}", e)),
            "toml" => toml::from_str::<toml::Value>(content)
                .map(|_| ())
                .map_err(|e| anyhow!("TOML error: {}", e)),
            "yaml" | "yml" => serde_yaml::from_str::<serde_yaml::Value>(content)
                .map(|_| ())
                .map_err(|e| anyhow!("YAML error: {}", e)),
            _ => Ok(()),
        }
    }
}

pub struct RegressionDetector;

impl RegressionDetector {
    #[must_use]
    pub fn is_regressive(old: &ArtifactMetrics, new: &ArtifactMetrics) -> bool {
        if new.cyclomatic_complexity > old.cyclomatic_complexity + 5 {
            return true;
        }
        if new.comment_density < old.comment_density - 0.1 {
            return true;
        }
        false
    }
}
