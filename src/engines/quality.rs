use crate::types::artifact::Artifact;
use anyhow::{Context, Result};
use petgraph::graph::DiGraph;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize, de::IgnoredAny};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tree_sitter::Parser;

thread_local! {
    static PARSER: RefCell<Parser> = RefCell::new({
        let mut p = Parser::new();
        p.set_language(&tree_sitter_rust::LANGUAGE.into()).ok();
        p
    });
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArtifactMetrics {
    pub cyclomatic_complexity: u32,
    pub coupling_factor: u32,
    pub comment_density: f64,
    pub line_count: u32,
    pub health_score: f64,
}

pub struct QualityEngine;

impl QualityEngine {
    #[must_use]
    pub fn analyze_artifact(artifact: &Artifact, workspace_modules: &[String]) -> ArtifactMetrics {
        let total_lines = artifact.content.lines().count() as u32;
        let comment_lines = artifact
            .content
            .lines()
            .filter(|l| l.trim().starts_with("//") || l.trim().starts_with("/*"))
            .count() as u32;

        let (complexity, coupling) =
            Self::compute_ast_metrics(&artifact.content, workspace_modules);

        let comment_density = if total_lines > 0 {
            f64::from(comment_lines) / f64::from(total_lines)
        } else {
            0.0
        };

        let health_score =
            1.0 - (f64::from(complexity) / 100.0) - (f64::from(coupling) / 20.0) + comment_density;

        ArtifactMetrics {
            cyclomatic_complexity: complexity,
            coupling_factor: coupling,
            comment_density,
            line_count: total_lines,
            health_score: health_score.clamp(0.0, 1.0),
        }
    }

    fn compute_ast_metrics(content: &str, module_lookup: &[String]) -> (u32, u32) {
        if content.len() > 10_000_000 {
            return (1, 0);
        }
        let mut complexity = 1u32;
        let mut dependencies = HashSet::new();

        let tree = PARSER.with(|p| p.borrow_mut().parse(content, None));
        let Some(t) = tree else { return (1, 0) };

        let mut cursor = t.walk();
        let mut going_down = true;

        loop {
            if going_down {
                let node = cursor.node();
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

                if (kind == "use_declaration" || kind == "scoped_identifier")
                    && let Some(slice) = content.as_bytes().get(node.byte_range())
                    && let Ok(text) = std::str::from_utf8(slice)
                {
                    for token in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
                        if module_lookup.contains(&token.to_string()) {
                            dependencies.insert(token);
                        }
                    }
                }

                if cursor.goto_first_child() {
                    continue;
                }
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

    pub async fn verify_compilation(workspace_dir: &str) -> Result<(bool, String)> {
        let workspace = workspace_dir.to_string();
        let out = tokio::task::spawn_blocking(move || {
            std::process::Command::new("cargo")
                .current_dir(workspace)
                .arg("check")
                .output()
        })
        .await??;

        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        Ok((out.status.success(), stderr))
    }

    pub fn build_dependency_graph(
        artifacts: &HashMap<String, Arc<Artifact>>,
    ) -> DiGraph<String, ()> {
        let mut graph = DiGraph::new();
        let mut node_indices = FxHashMap::default();

        for name in artifacts.keys() {
            let stripped = name.trim_end_matches(".rs");
            node_indices.insert(stripped, graph.add_node(name.clone()));
        }

        let all_edges: Vec<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> = artifacts
            .par_iter()
            .flat_map(|(name, artifact)| {
                let stripped_name = name.trim_end_matches(".rs");
                let current_idx = match node_indices.get(stripped_name) {
                    Some(&idx) => idx,
                    None => return vec![],
                };
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

        for (source, target) in all_edges {
            graph.add_edge(source, target, ());
        }

        graph
    }

    pub fn detect_duplication(
        new_content: &str,
        existing: &BTreeMap<String, Arc<Artifact>>,
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

        existing
            .par_iter()
            .filter_map(|(name, artifact)| {
                let match_count = artifact
                    .content
                    .lines()
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
        new.cyclomatic_complexity > old.cyclomatic_complexity + 5
            || new.comment_density < old.comment_density - 0.1
    }
}

// ── QualityTrend ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QualityTrendEntry {
    pub turn_id: u32,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct QualityTrend {
    pub artifact_id: String,
    pub history: Vec<QualityTrendEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendClassification {
    Improving,
    Stable,
    Degrading,
}

pub struct QualityTrendAnalyzer;

impl QualityTrendAnalyzer {
    #[must_use]
    pub fn velocity(trend: &QualityTrend) -> Option<f64> {
        let h = &trend.history;
        if h.len() < 2 {
            return None;
        }
        let first = &h[0];
        let last = h.last().unwrap();
        let turns = (last.turn_id as f64 - first.turn_id as f64).max(1.0);
        Some((last.score - first.score) / turns)
    }

    #[must_use]
    pub fn classify(trend: &QualityTrend) -> TrendClassification {
        let h = &trend.history;
        if h.len() < 2 {
            return TrendClassification::Stable;
        }
        let deltas: Vec<f64> = h.windows(2).map(|w| w[1].score - w[0].score).collect();
        let pos_run = deltas.iter().rev().take_while(|&&d| d > 0.0).count();
        let neg_run = deltas.iter().rev().take_while(|&&d| d < 0.0).count();
        if pos_run >= 3 {
            TrendClassification::Improving
        } else if neg_run >= 2 {
            TrendClassification::Degrading
        } else {
            TrendClassification::Stable
        }
    }
}

// ── CompileError ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub error_code: String,
    pub message: String,
    pub suggestion: String,
}

pub struct CompileErrorParser;

impl CompileErrorParser {
    #[must_use]
    pub fn parse(output: &str) -> Vec<CompileError> {
        let mut errors: Vec<CompileError> = Vec::new();
        let mut pending_code = String::new();
        let mut pending_msg = String::new();
        let mut file = String::new();
        let mut line_num = 0u32;
        let mut col_num = 0u32;
        let mut suggestion = String::new();

        let flush = |errors: &mut Vec<CompileError>,
                     code: &mut String,
                     msg: &mut String,
                     file: &mut String,
                     ln: &mut u32,
                     col: &mut u32,
                     sug: &mut String| {
            if !msg.is_empty() {
                errors.push(CompileError {
                    file: std::mem::take(file),
                    line: std::mem::replace(ln, 0),
                    column: std::mem::replace(col, 0),
                    error_code: std::mem::take(code),
                    message: std::mem::take(msg),
                    suggestion: std::mem::take(sug),
                });
            }
        };

        for raw in output.lines() {
            let t = raw.trim();
            if let Some(rest) = t.strip_prefix("error[") {
                flush(
                    &mut errors,
                    &mut pending_code,
                    &mut pending_msg,
                    &mut file,
                    &mut line_num,
                    &mut col_num,
                    &mut suggestion,
                );
                if let Some((code, msg)) = rest.split_once("]: ") {
                    pending_code = code.to_string();
                    pending_msg = msg.to_string();
                }
            } else if let Some(msg) = t.strip_prefix("error: ") {
                flush(
                    &mut errors,
                    &mut pending_code,
                    &mut pending_msg,
                    &mut file,
                    &mut line_num,
                    &mut col_num,
                    &mut suggestion,
                );
                pending_msg = msg.to_string();
            } else if let Some(loc) = t.strip_prefix("--> ") {
                let parts: Vec<&str> = loc.splitn(3, ':').collect();
                if parts.len() == 3 {
                    file = parts[0].to_string();
                    line_num = parts[1].parse().unwrap_or(0);
                    col_num = parts[2].parse().unwrap_or(0);
                }
            } else if suggestion.is_empty() {
                for pfx in &["= help: ", "= note: ", "help: "] {
                    if let Some(hint) = t.strip_prefix(pfx) {
                        suggestion = hint.to_string();
                        break;
                    }
                }
            }
        }
        flush(
            &mut errors,
            &mut pending_code,
            &mut pending_msg,
            &mut file,
            &mut line_num,
            &mut col_num,
            &mut suggestion,
        );
        errors
    }
}

// ── CoherenceChecker ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncoherenceKind {
    UndefinedSymbol,
    StaleImport,
}

#[derive(Debug, Clone)]
pub struct IncoherenceReport {
    pub artifact_from: String,
    pub symbol: String,
    pub kind: IncoherenceKind,
}

pub struct CoherenceChecker;

impl CoherenceChecker {
    #[must_use]
    pub fn verify(artifacts: &HashMap<String, Arc<Artifact>>) -> Vec<IncoherenceReport> {
        let stems: HashSet<String> = artifacts
            .keys()
            .map(|k| k.trim_end_matches(".rs").to_string())
            .collect();

        let mut defined: HashSet<String> = HashSet::new();
        let def_prefixes = [
            "pub fn ",
            "fn ",
            "pub struct ",
            "struct ",
            "pub enum ",
            "enum ",
            "pub type ",
            "type ",
            "pub trait ",
            "trait ",
        ];
        for artifact in artifacts.values() {
            for line in artifact.content.lines() {
                let t = line.trim();
                for pfx in &def_prefixes {
                    if let Some(rest) = t.strip_prefix(pfx) {
                        let name = rest
                            .split(|c: char| !c.is_alphanumeric() && c != '_')
                            .next()
                            .unwrap_or("");
                        if !name.is_empty() {
                            defined.insert(name.to_string());
                        }
                        break;
                    }
                }
            }
        }

        let mut reports = Vec::new();
        for (art_name, artifact) in artifacts {
            for line in artifact.content.lines() {
                let t = line.trim();

                if !t.contains('{')
                    && let Some(mod_name) = t
                        .strip_prefix("mod ")
                        .map(|r| r.trim_end_matches(';').trim())
                    && !stems.contains(mod_name)
                {
                    reports.push(IncoherenceReport {
                        artifact_from: art_name.clone(),
                        symbol: mod_name.to_string(),
                        kind: IncoherenceKind::StaleImport,
                    });
                }

                if let Some(rest) = t.strip_prefix("use ").map(|r| r.trim_end_matches(';')) {
                    let last = rest.split("::").last().unwrap_or("");
                    let candidates: Vec<&str> = if last.starts_with('{') {
                        last.trim_matches(|c| c == '{' || c == '}')
                            .split(',')
                            .map(str::trim)
                            .collect()
                    } else {
                        vec![last]
                    };
                    for sym in candidates {
                        let sym = sym.trim();
                        if sym.is_empty() || sym == "*" || sym == "self" {
                            continue;
                        }
                        if sym.starts_with(|c: char| c.is_uppercase()) && !defined.contains(sym) {
                            reports.push(IncoherenceReport {
                                artifact_from: art_name.clone(),
                                symbol: sym.to_string(),
                                kind: IncoherenceKind::UndefinedSymbol,
                            });
                        }
                    }
                }
            }
        }
        reports
    }
}

// ── CompletionScorer ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompletionReport {
    pub score: f64,
    pub requirements_met: f64,
    pub tests_passing: f64,
    pub quality_floor_met: bool,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockerType {
    FailingTest,
    QualityBelowThreshold,
    MissingArtifact,
    UnresolvedIncoherence,
}

#[derive(Debug, Clone)]
pub struct BlockerReport {
    pub blocker_type: BlockerType,
    pub affected_artifacts: Vec<String>,
    pub suggested_action: String,
}

pub struct CompletionScorer;

impl CompletionScorer {
    #[must_use]
    pub fn evaluate(
        requirements_met: f64,
        tests_passing: f64,
        quality_floor_met: bool,
    ) -> CompletionReport {
        let qf = if quality_floor_met { 1.0 } else { 0.0 };
        let score = (0.5 * requirements_met + 0.3 * tests_passing + 0.2 * qf).clamp(0.0, 1.0);
        CompletionReport {
            score,
            requirements_met: requirements_met.clamp(0.0, 1.0),
            tests_passing: tests_passing.clamp(0.0, 1.0),
            quality_floor_met,
            converged: score > 0.95,
        }
    }

    #[must_use]
    pub fn diagnose_blocker(recent: &[CompletionReport]) -> Option<BlockerReport> {
        if recent.len() < 3 {
            return None;
        }
        let stalled = recent
            .windows(2)
            .rev()
            .take(3)
            .all(|w| (w[1].score - w[0].score).abs() < 0.01);
        if !stalled {
            return None;
        }
        let last = recent.last().unwrap();
        let (blocker_type, suggested_action) = if last.tests_passing < 0.5 {
            (
                BlockerType::FailingTest,
                "Fix the failing tests before continuing.".to_string(),
            )
        } else if !last.quality_floor_met {
            (
                BlockerType::QualityBelowThreshold,
                "Reduce cyclomatic complexity or improve comment density.".to_string(),
            )
        } else {
            (
                BlockerType::UnresolvedIncoherence,
                "Resolve cross-artifact symbol incoherences.".to_string(),
            )
        };
        Some(BlockerReport {
            blocker_type,
            affected_artifacts: vec![],
            suggested_action,
        })
    }
}

// ── TournamentRunner ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TournamentProposal {
    pub agent_id: String,
    pub quality_score: f64,
    pub compiled: bool,
    pub tests_passed: bool,
}

#[derive(Debug, Clone)]
pub struct TournamentResult {
    pub proposals: Vec<TournamentProposal>,
    pub winner_agent_id: String,
}

pub struct TournamentRunner;

impl TournamentRunner {
    #[must_use]
    pub fn run(proposals: &[TournamentProposal]) -> Option<TournamentResult> {
        if proposals.is_empty() {
            return None;
        }
        let scored: Vec<(&TournamentProposal, f64)> = proposals
            .iter()
            .map(|p| {
                let s = p.compiled as u8 as f64 * 0.3
                    + p.tests_passed as u8 as f64 * 0.4
                    + p.quality_score * 0.3;
                (p, s)
            })
            .collect();
        let winner = scored
            .iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(p, _)| p.agent_id.clone())
            .unwrap_or_default();
        Some(TournamentResult {
            proposals: proposals.to_vec(),
            winner_agent_id: winner,
        })
    }
}

// ── DuplicationDetector ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DuplicationReport {
    pub source_artifact: String,
    pub overlap_ratio: f64,
}

pub struct DuplicationDetector;

impl DuplicationDetector {
    const NGRAM_SIZE: usize = 10;
    const OVERLAP_THRESHOLD: f64 = 0.30;

    fn tokenize(content: &str) -> Vec<String> {
        content
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect()
    }

    fn ngram_set(tokens: &[String], n: usize) -> HashSet<Vec<String>> {
        if tokens.len() < n {
            return HashSet::new();
        }
        tokens.windows(n).map(|w| w.to_vec()).collect()
    }

    #[must_use]
    pub fn scan<'a>(
        new_content: &str,
        existing: impl Iterator<Item = (&'a str, &'a str)>,
    ) -> Vec<DuplicationReport> {
        let new_tokens = Self::tokenize(new_content);
        let new_grams = Self::ngram_set(&new_tokens, Self::NGRAM_SIZE);
        if new_grams.is_empty() {
            return vec![];
        }

        let mut reports = Vec::new();
        for (name, content) in existing {
            let existing_tokens = Self::tokenize(content);
            let existing_grams = Self::ngram_set(&existing_tokens, Self::NGRAM_SIZE);
            let overlap = new_grams.intersection(&existing_grams).count();
            let ratio = overlap as f64 / new_grams.len() as f64;
            if ratio > Self::OVERLAP_THRESHOLD {
                reports.push(DuplicationReport {
                    source_artifact: name.to_string(),
                    overlap_ratio: ratio,
                });
            }
        }
        reports
    }
}

// ── DocChecker ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DocInconsistency {
    pub function_name: String,
    pub issue: String,
}

pub struct DocChecker;

impl DocChecker {
    #[must_use]
    pub fn verify(content: &str) -> Vec<DocInconsistency> {
        let mut inconsistencies = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        let mut doc_lines: Vec<String> = Vec::new();
        for (i, &line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.starts_with("///") || t.starts_with("//!") {
                doc_lines.push(t.trim_start_matches('/').trim().to_string());
            } else {
                if !doc_lines.is_empty()
                    && let Some(fn_line) = lines.get(i)
                {
                    let fn_t = fn_line.trim();
                    if fn_t.contains("fn ") {
                        let fn_name = fn_t
                            .split("fn ")
                            .nth(1)
                            .and_then(|r| r.split('(').next())
                            .unwrap_or("")
                            .trim()
                            .to_string();

                        let doc_text = doc_lines.join(" ");

                        let doc_returns_option = doc_text.contains("Option");
                        let doc_returns_result = doc_text.contains("Result");
                        let sig_returns_option = fn_t.contains("-> Option");
                        let sig_returns_result = fn_t.contains("-> Result");

                        if doc_returns_option && sig_returns_result {
                            inconsistencies.push(DocInconsistency {
                                function_name: fn_name.clone(),
                                issue: "Doc says returns Option but signature returns Result."
                                    .to_string(),
                            });
                        } else if doc_returns_result && sig_returns_option {
                            inconsistencies.push(DocInconsistency {
                                function_name: fn_name.clone(),
                                issue: "Doc says returns Result but signature returns Option."
                                    .to_string(),
                            });
                        }

                        let sig_params = fn_t
                            .split('(')
                            .nth(1)
                            .and_then(|s| s.split(')').next())
                            .unwrap_or("");
                        for word in doc_text.split_whitespace() {
                            let candidate = word.trim_end_matches(':');
                            if candidate.ends_with("_param")
                                || (candidate.len() > 2
                                    && candidate.starts_with(|c: char| c.is_lowercase())
                                    && word.ends_with(':')
                                    && !sig_params.contains(candidate))
                            {
                                inconsistencies.push(DocInconsistency {
                                        function_name: fn_name.clone(),
                                        issue: format!(
                                            "Doc references parameter `{candidate}` not found in signature."
                                        ),
                                    });
                            }
                        }
                    }
                }
                doc_lines.clear();
            }
        }
        inconsistencies
    }
}

// ── DeadCodeDetector ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeadItemKind {
    UnusedFunction,
    UnusedImport,
}

#[derive(Debug, Clone)]
pub struct DeadItem {
    pub name: String,
    pub kind: DeadItemKind,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct DeadCodeReport {
    pub artifact_name: String,
    pub dead_items: Vec<DeadItem>,
    pub dead_code_ratio: f64,
}

pub struct DeadCodeDetector;

impl DeadCodeDetector {
    #[must_use]
    pub fn scan(artifact_name: &str, content: &str) -> DeadCodeReport {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len() as f64;
        let mut dead_items: Vec<DeadItem> = Vec::new();
        let mut dead_lines = 0usize;

        let mut defined_fns: Vec<(String, u32)> = Vec::new();
        let mut prev_attrs: Vec<&str> = Vec::new();

        for (i, &line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.starts_with("#[") {
                prev_attrs.push(t);
            } else {
                if let Some(rest) = t.strip_prefix("fn ").or_else(|| {
                    if t.starts_with("async fn ") {
                        t.strip_prefix("async fn ")
                    } else {
                        None
                    }
                }) {
                    let fn_name = rest
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("");
                    let has_test = prev_attrs.iter().any(|a| a.contains("test"));
                    let has_allow = prev_attrs
                        .iter()
                        .any(|a| a.contains("allow") && a.contains("dead_code"));
                    if !fn_name.is_empty() && fn_name != "main" && !has_test && !has_allow {
                        defined_fns.push((fn_name.to_string(), i as u32 + 1));
                    }
                }
                if !t.starts_with('#') {
                    prev_attrs.clear();
                }
            }
        }

        for (fn_name, ln) in &defined_fns {
            let def_pattern = format!("fn {fn_name}");
            let call_pattern = format!("{fn_name}(");
            let call_count = lines
                .iter()
                .filter(|&&l| l.contains(&call_pattern) && !l.contains(&def_pattern))
                .count();
            if call_count == 0 {
                dead_items.push(DeadItem {
                    name: fn_name.clone(),
                    kind: DeadItemKind::UnusedFunction,
                    line: *ln,
                });
                dead_lines += 1;
            }
        }

        for (i, &line) in lines.iter().enumerate() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("use ").map(|r| r.trim_end_matches(';')) {
                let last = rest.split("::").last().unwrap_or("");
                let candidates: Vec<&str> = if last.starts_with('{') {
                    last.trim_matches(|c| c == '{' || c == '}')
                        .split(',')
                        .map(str::trim)
                        .collect()
                } else {
                    vec![last.trim()]
                };
                for sym in candidates {
                    let sym = sym.trim().trim_start_matches("self").trim();
                    if sym.is_empty() || sym == "*" {
                        continue;
                    }
                    let used_elsewhere = lines
                        .iter()
                        .enumerate()
                        .any(|(j, &l)| j != i && l.contains(sym) && !l.trim().starts_with("use "));
                    if !used_elsewhere {
                        dead_items.push(DeadItem {
                            name: sym.to_string(),
                            kind: DeadItemKind::UnusedImport,
                            line: i as u32 + 1,
                        });
                        dead_lines += 1;
                    }
                }
            }
        }

        let dead_code_ratio = if total_lines > 0.0 {
            (dead_lines as f64 / total_lines).clamp(0.0, 1.0)
        } else {
            0.0
        };

        DeadCodeReport {
            artifact_name: artifact_name.to_string(),
            dead_items,
            dead_code_ratio,
        }
    }
}
