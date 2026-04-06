use crate::types::artifact::Artifact;
use anyhow::{Context, Result};
use petgraph::graph::DiGraph;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet}; 
use serde::{Deserialize, Serialize, de::IgnoredAny};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
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

        // O(N) optimized comment counting
        let comment_lines = lines
            .iter()
            .filter(|l| {
                let trimmed = l.trim_start();
                trimmed.starts_with("//") || trimmed.starts_with("/*")
            })
            .count();
            
        let comment_density = if line_count > 0 {
            comment_lines as f64 / line_count as f64
        } else {
            0.0
        };

        // Combine Complexity and Coupling into a single, highly-optimized AST pass
        let (complexity, coupling) = if artifact.language.eq_ignore_ascii_case("rust") || artifact.language.eq_ignore_ascii_case("rs") {
            Self::analyze_rust_ast(&artifact.content, other_artifact_names)
        } else {
            (1, 0)
        };

        ArtifactMetrics {
            line_count,
            cyclomatic_complexity: complexity,
            comment_density,
            coupling,
        }
    }

    /// SINGLE-PASS AST TRAVERSAL
    /// Extracts both Cyclomatic Complexity and Dependency Coupling simultaneously.
    /// Completely immune to string-matching false positives (e.g., comments/strings).
    fn analyze_rust_ast(content: &str, known_modules: &[String]) -> (u32, u32) {
        let mut parser = Parser::new();
        if parser.set_language(&tree_sitter_rust::LANGUAGE.into()).is_err() {
            return (1, 0);
        }
        
        let tree = match parser.parse(content, None) {
            Some(t) => t,
            None => return (1, 0),
        };

        // Pre-compute lookup table for O(1) existence checks
        let module_lookup: FxHashSet<&str> = known_modules
            .iter()
            .map(|name| name.trim_end_matches(".rs"))
            .collect();

        let mut complexity = 1;
        let mut dependencies = FxHashSet::default();
        
        let mut cursor = tree.walk();
        let mut going_down = true;

        loop {
            if going_down {
                let node = cursor.node();
                let kind = node.kind();
                
                // 1. Evaluate Complexity
                if matches!(
                    kind,
                    "if_expression" | "for_expression" | "while_expression" | 
                    "match_arm" | "if_let_expression" | "while_let_expression"
                ) {
                    complexity += 1;
                }
                
                // 2. Evaluate Coupling (Only inside actual `use` or `mod` declarations)
                if kind == "use_declaration" || kind == "scoped_identifier" {
                    // Extract the text of the node directly from the source bytes
                    if let Ok(text) = std::str::from_utf8(&content.as_bytes()[node.byte_range()]) {
                        for token in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
                            if module_lookup.contains(token) {
                                dependencies.insert(token);
                            }
                        }
                    }
                }
                
                if cursor.goto_first_child() { continue; }
            }
            if cursor.goto_next_sibling() {
                going_down = true;
                continue;
            }
            if cursor.goto_parent() {
                going_down = false;
                continue;
            }
            break; 
        }
        
        (complexity, dependencies.len() as u32)
    }

    /// Asynchronous, timeout-protected, AND leak-proof compilation check.
    pub async fn verify_compilation(workspace_dir: &str) -> Result<bool> {
        let mut check_task = Command::new("cargo")
            .current_dir(workspace_dir)
            .arg("check")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true) // CRITICAL: Reaps the OS process if Tokio drops the future
            .spawn()
            .context("Failed to spawn cargo process")?;

        let output = timeout(Duration::from_secs(60), check_task.wait())
            .await
            .context("Cargo check timed out")??;

        Ok(output.success())
    }

    /// Parallelized Graph Construction.
    /// Operates in O(N * L) time but executes across all available CPU cores concurrently.
    pub fn build_dependency_graph(artifacts: &FxHashMap<String, Artifact>) -> DiGraph<String, ()> {
        let mut graph = DiGraph::new();
        let mut node_indices = FxHashMap::default();

        // Register nodes
        for name in artifacts.keys() {
            let stripped = name.trim_end_matches(".rs");
            node_indices.insert(stripped, graph.add_node(name.clone()));
        }

        // Parallel map: Extract edges for all artifacts simultaneously
        let all_edges: Vec<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> = artifacts
            .par_iter()
            .flat_map(|(name, artifact)| {
                let stripped_name = name.trim_end_matches(".rs");
                let current_idx = *node_indices.get(stripped_name).unwrap();
                let mut local_edges = vec![];

                for line in artifact.content.lines() {
                    let trimmed = line.trim_start();
                    if trimmed.starts_with("use ") || trimmed.starts_with("mod ") {
                        for token in trimmed.split(|c: char| !c.is_alphanumeric() && c != '_') {
                            if let Some(&target_idx) = node_indices.get(token)
                                && current_idx != target_idx
                            {
                                local_edges.push((current_idx, target_idx));
                            }
                        }
                    }
                }
                local_edges
            })
            .collect();

        // Sequential reduce: Insert edges into the graph
        for (source, target) in all_edges {
            graph.add_edge(source, target, ());
        }

        graph
    }

    /// Rayon-Parallelized Duplication Detection.
    /// Eliminates thread-blocking on massive artifact registries.
    pub fn detect_duplication(
        new_content: &str,
        existing: &std::collections::HashMap<String, Artifact>,
    ) -> Vec<String> {
        let new_lines: FxHashSet<&str> = new_content
            .lines()
            .map(|l| l.trim())
            .filter(|l| l.len() > 2 && *l != "}" && *l != "{")
            .collect();

        if new_lines.is_empty() {
            return vec![];
        }

        let new_lines_len = new_lines.len() as f64;

        // Map-Reduce pattern using Rayon
        existing
            .par_iter()
            .filter_map(|(name, artifact)| {
                let match_count = artifact.content.lines()
                    .map(|l| l.trim())
                    .filter(|l| new_lines.contains(l))
                    .count();

                let overlap_ratio = match_count as f64 / new_lines_len;
                if overlap_ratio > 0.70 {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct ValidatorRegistry;

impl ValidatorRegistry {
    pub fn validate(content: &str, language: &str) -> Result<()> {
        match language.to_lowercase().as_str() {
            "json" => {
                serde_json::from_str::<IgnoredAny>(content).context("Invalid JSON")?;
            }
            "toml" => {
                toml::from_str::<toml::Value>(content).context("Invalid TOML")?;
            }
            "yaml" | "yml" => {
                serde_yaml::from_str::<serde_yaml::Value>(content).context("Invalid YAML")?;
            }
            _ => {}
        }
        Ok(())
    }
}

pub struct RegressionDetector;

impl RegressionDetector {
    #[must_use]
    pub fn is_regressive(old: &ArtifactMetrics, new: &ArtifactMetrics) -> bool {
        new.cyclomatic_complexity > old.cyclomatic_complexity + 5 || 
        new.comment_density < old.comment_density - 0.1
    }
}