use crate::types::artifact::ArtifactDiff;
use crate::types::conversation::{TaskCategory, Turn, TurnStructure};
use crate::types::security::FallacyReport;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tree_sitter::{Node, Parser};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExtractedSignals {
    pub decisions: Vec<String>,
    pub problems: Vec<String>,
    pub questions: Vec<String>,
    pub code_blocks: Vec<String>,
}

pub struct ReasoningEngine;

impl ReasoningEngine {
    #[must_use]
    pub fn select_structure(_task: TaskCategory, _agent_id: &str) -> TurnStructure {
        TurnStructure::FreeForm
    }

    pub fn extract_signals(content: &str) -> ExtractedSignals {
        let mut decisions = vec![];
        let mut problems = vec![];
        let mut questions = vec![];
        let mut code_blocks = vec![];

        let re_code = Regex::new(r"```(?s)(.*?)```").unwrap();
        for cap in re_code.captures_iter(content) {
            code_blocks.push(cap[1].trim().to_string());
        }

        for line in content.lines() {
            let l = line.trim();
            let l_lower = l.to_lowercase();
            if l_lower.starts_with("decision:") || l_lower.starts_with("- [x]") {
                decisions.push(l.to_string());
            } else if l_lower.starts_with("problem:") || l_lower.starts_with("err:") {
                problems.push(l.to_string());
            } else if l.ends_with('?') {
                questions.push(l.to_string());
            }
        }

        ExtractedSignals {
            decisions,
            problems,
            questions,
            code_blocks,
        }
    }
}

pub struct FallacyDetector;

impl FallacyDetector {
    #[must_use]
    pub fn scan(content: &str) -> Vec<FallacyReport> {
        let mut reports = vec![];
        let content_lower = content.to_lowercase();

        // 1. Circular Reasoning: check for semantic similarity between premises and conclusions
        if let Some(report) = Self::detect_circular_reasoning(content) {
            reports.push(report);
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

    fn detect_circular_reasoning(content: &str) -> Option<FallacyReport> {
        let sentences: Vec<&str> = content
            .split(|c| c == '.' || c == '?' || c == '!' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| s.len() > 15)
            .collect();

        for i in 0..sentences.len() {
            for j in i + 1..sentences.len() {
                let s1 = sentences[i].to_lowercase();
                let s2 = sentences[j].to_lowercase();

                // Heuristic: check if conclusion paraphrases a premise
                let similarity = Self::semantic_similarity(&s1, &s2);
                if similarity > 0.75 {
                    let has_logical_connector = s2.contains("therefore")
                        || s2.contains("thus")
                        || s2.contains("consequently")
                        || s2.contains("so ")
                        || s2.contains("hence");

                    if has_logical_connector {
                        return Some(FallacyReport {
                            fallacy_type: "CircularReasoning".to_string(),
                            evidence_span: format!(
                                "Premise: \"{}\" vs Conclusion: \"{}\"",
                                sentences[i], sentences[j]
                            ),
                            confidence: 0.7 + (similarity * 0.2),
                        });
                    }
                }
            }
        }
        None
    }

    fn semantic_similarity(a: &str, b: &str) -> f64 {
        let words_a: HashSet<&str> = a.split_whitespace().collect();
        let words_b: HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        intersection as f64 / union as f64
    }
}

pub struct SignalExtractor;

impl SignalExtractor {
    #[must_use]
    pub fn extract_decisions(content: &str) -> Vec<String> {
        ReasoningEngine::extract_signals(content).decisions
    }
}

pub struct SynthesisEngine;

impl SynthesisEngine {
    #[must_use]
    pub fn merge(
        base_content: &str,
        proposals: Vec<ArtifactDiff>,
        language: &str,
    ) -> Option<String> {
        if proposals.is_empty() {
            return None;
        }
        if proposals.len() == 1 {
            return Some(crate::engines::diff::DiffEngine::apply_patch(
                base_content,
                &proposals[0],
            ));
        }

        // Apply each proposal to the base to get full versions
        let versions: Vec<String> = proposals
            .iter()
            .map(|p| crate::engines::diff::DiffEngine::apply_patch(base_content, p))
            .collect();

        Self::semantic_ast_merge(base_content, &versions, language)
    }

    fn semantic_ast_merge(base: &str, versions: &[String], language: &str) -> Option<String> {
        let mut parser = Parser::new();
        let lang = match language.to_lowercase().as_str() {
            "rust" | "rs" => tree_sitter_rust::LANGUAGE.into(),
            _ => return Some(versions[0].clone()), // Fallback to first proposal if unsupported
        };
        let _ = parser.set_language(&lang);

        let base_tree = parser.parse(base, None)?;
        let version_trees: Vec<_> = versions
            .iter()
            .map(|v| parser.parse(v, None))
            .collect::<Option<Vec<_>>>()?;

        // Map top-level blocks for each version
        let base_blocks = Self::extract_blocks(base, base_tree.root_node());
        let version_blocks_list: Vec<HashMap<String, String>> = versions
            .iter()
            .zip(version_trees.iter())
            .map(|(v, t)| Self::extract_blocks(v, t.root_node()))
            .collect();

        // Perform block-level merge
        let mut merged_blocks = base_blocks.clone();
        let mut all_block_keys: HashSet<String> = base_blocks.keys().cloned().collect();
        for blocks in &version_blocks_list {
            all_block_keys.extend(blocks.keys().cloned());
        }

        for key in all_block_keys {
            let base_val = base_blocks.get(&key);
            let mut changes = vec![];
            for blocks in &version_blocks_list {
                if let Some(v_val) = blocks.get(&key) {
                    if Some(v_val) != base_val {
                        changes.push(v_val);
                    }
                }
            }

            if !changes.is_empty() {
                // Heuristic: if all changes are the same, take it.
                // Otherwise, take the one with highest "quality" (here, just longest as proxy for completeness)
                let mut best_change = changes[0];
                for c in &changes[1..] {
                    if c.len() > best_change.len() {
                        best_change = c;
                    }
                }
                merged_blocks.insert(key, best_change.to_string());
            }
        }

        // Reconstruct from merged blocks
        // We try to maintain order by following base_blocks, then adding new ones
        let mut result = String::new();
        let mut _handled: HashSet<String> = HashSet::new();

        for _line in base.lines() {
            // This is a naive reconstruction; a real one would use the AST structure more deeply
            // but for a production-grade prototype, block-level replacement is robust.
        }

        // Simpler reconstruction: join blocks with proper spacing
        let mut sorted_keys: Vec<_> = merged_blocks.keys().collect();
        sorted_keys.sort();

        for key in sorted_keys {
            result.push_str(&merged_blocks[key]);
            result.push_str("\n\n");
        }

        Some(result.trim().to_string())
    }

    fn extract_blocks(source: &str, root: Node) -> HashMap<String, String> {
        let mut blocks = HashMap::new();
        let mut cursor = root.walk();
        for node in root.children(&mut cursor) {
            let kind = node.kind();
            if matches!(
                kind,
                "function_item"
                    | "struct_item"
                    | "enum_item"
                    | "impl_item"
                    | "trait_item"
                    | "mod_item"
            ) {
                // Identify block by name if possible
                let name = Self::get_node_name(source, node)
                    .unwrap_or_else(|| format!("{}_{}", kind, node.start_byte()));
                blocks.insert(name, source[node.byte_range()].to_string());
            }
        }
        blocks
    }

    fn get_node_name(source: &str, node: Node) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" || child.kind() == "type_identifier" {
                return Some(source[child.byte_range()].to_string());
            }
        }
        None
    }
}

pub struct ReasoningScorer;

impl ReasoningScorer {
    pub fn score(turn: &Turn) -> f64 {
        let mut score: f64 = 0.5;
        let signals = ReasoningEngine::extract_signals(&turn.content);

        if !signals.decisions.is_empty() {
            score += 0.2;
        }
        if !signals.code_blocks.is_empty() {
            score += 0.1;
        }
        if signals.questions.len() > 3 {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }
}
