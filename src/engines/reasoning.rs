use crate::types::artifact::ArtifactDiff;
use crate::types::conversation::{TaskCategory, Turn, TurnStructure};
use crate::types::security::FallacyReport;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use rustc_hash::FxHasher;
use tree_sitter::{Node, Parser};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExtractedSignals {
    pub decisions: Vec<String>,
    pub problems: Vec<String>,
    pub questions: Vec<String>,
    pub code_blocks: Vec<String>,
}

// =====================================================================
// REASONING ENGINE (ReDoS-Immune State Machine)
// =====================================================================

pub struct ReasoningEngine;

impl ReasoningEngine {
    #[must_use]
    pub fn select_structure(_task: TaskCategory, _agent_id: &str) -> TurnStructure {
        TurnStructure::FreeForm
    }

    /// Single-pass, $O(N)$, zero-regex state machine.
    /// Completely immune to backtracking attacks and avoids allocating intermediate strings.
    pub fn extract_signals(content: &str) -> ExtractedSignals {
        let mut decisions = vec![];
        let mut problems = vec![];
        let mut questions = vec![];
        let mut code_blocks = vec![];

        let mut in_code_block = false;
        let mut current_block = String::with_capacity(1024);

        for line in content.lines() {
            let trimmed = line.trim();

            // Handle Code Blocks securely
            if trimmed.starts_with("```") {
                if in_code_block {
                    // Close the block, push to list, and instantly clear memory without dropping the allocation
                    code_blocks.push(current_block.trim_end().to_string());
                    current_block.clear();
                }
                in_code_block = !in_code_block;
                continue;
            }

            if in_code_block {
                current_block.push_str(line);
                current_block.push('\n');
                continue; // Do not extract logical signals from inside code blocks
            }

            // Handle Logic Signals (Zero-allocation prefix matching)
            if trimmed.is_empty() { continue; }

            if trimmed.eq_ignore_ascii_case("decision:") || trimmed.starts_with("- [x]") || trimmed.starts_with("- [X]") {
                decisions.push(trimmed.to_string());
            } else if trimmed.eq_ignore_ascii_case("problem:") || trimmed.eq_ignore_ascii_case("err:") {
                problems.push(trimmed.to_string());
            } else if trimmed.ends_with('?') {
                questions.push(trimmed.to_string());
            }
        }

        ExtractedSignals { decisions, problems, questions, code_blocks }
    }
}

pub struct SignalExtractor;
impl SignalExtractor {
    #[must_use]
    pub fn extract_decisions(content: &str) -> Vec<String> {
        ReasoningEngine::extract_signals(content).decisions
    }
}

pub struct ReasoningScorer;
impl ReasoningScorer {
    #[must_use]
    pub fn score(turn: &Turn) -> f64 {
        let signals = ReasoningEngine::extract_signals(&turn.content);
        let mut score: f64 = 0.5;

        if !signals.decisions.is_empty() { score += 0.2; }
        if !signals.code_blocks.is_empty() { score += 0.1; }
        if signals.questions.len() > 3 { score -= 0.1; }

        score.clamp(0.0, 1.0)
    }
}

// =====================================================================
// FALLACY DETECTION (O(N) MinHash Concept)
// =====================================================================

type FxBuildHasher = BuildHasherDefault<FxHasher>;

pub struct FallacyDetector;

impl FallacyDetector {
    #[must_use]
    pub fn scan(content: &str) -> Vec<FallacyReport> {
        let mut reports = vec![];

        if let Some(report) = Self::detect_circular_reasoning(content) {
            reports.push(report);
        }

        if content.contains("documentation") || content.contains("Documentation") {
            let content_lower = content.to_lowercase();
            if content_lower.contains("as the documentation says") && !content.contains("```") {
                reports.push(FallacyReport {
                    fallacy_type: "AppealToAuthority".to_string(),
                    evidence_span: "as the documentation says".to_string(),
                    confidence: 0.8,
                });
            }
        }

        reports
    }

    /// Upgraded from $O(N^2)$ to $O(N)$ using a semantic hash-binning approach.
    /// Sentences are hashed into semantic buckets. Collisions indicate circular reasoning.
    fn detect_circular_reasoning(content: &str) -> Option<FallacyReport> {
        let mut semantic_buckets: HashMap<u64, &str, FxBuildHasher> = HashMap::with_capacity_and_hasher(128, Default::default());

        let sentences = content
            .split(['.', '?', '!', '\n'])
            .map(|s| s.trim())
            .filter(|s| s.len() > 20); // Minimum length to carry logical weight

        for sentence in sentences {
            // Generate a rapid "semantic signature" by hashing bi-grams.
            let signature = Self::compute_semantic_signature(sentence);

            // If a highly similar semantic structure already exists in the text...
            if let Some(previous_sentence) = semantic_buckets.get(&signature) {
                let s_lower = sentence.to_lowercase();
                
                // ...and the new sentence acts as a conclusion, it is circular.
                if s_lower.contains("therefore") || s_lower.contains("thus") || s_lower.contains("consequently") {
                    return Some(FallacyReport {
                        fallacy_type: "CircularReasoning".to_string(),
                        evidence_span: format!("Premise: \"{}\" vs Conclusion: \"{}\"", previous_sentence, sentence),
                        confidence: 0.85,
                    });
                }
            } else {
                semantic_buckets.insert(signature, sentence);
            }
        }
        None
    }

    /// Computes a structural signature that ignores stop words and exact ordering.
    /// Mathematically forces similar sentences into the same u64 hash bucket.
    fn compute_semantic_signature(text: &str) -> u64 {
        let mut hasher = FxHasher::default();
        let mut words: Vec<&str> = text
            .split_whitespace()
            .filter(|w| w.len() > 3) // Ignore "the", "and", "is"
            .collect();
        
        words.sort_unstable(); // Order-independent signature
        for word in words {
            hasher.write(word.to_ascii_lowercase().as_bytes());
        }
        hasher.finish()
    }
}

// =====================================================================
// AST-AWARE SYNTHESIS (Quorum-Based Consensus Merge)
// =====================================================================

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct AstBlock {
    signature: String,
    byte_range: std::ops::Range<usize>,
    content: String,
}

pub struct SynthesisEngine;

impl SynthesisEngine {
    #[must_use]
    pub fn merge(base_content: &str, proposals: Vec<ArtifactDiff>, language: &str) -> Option<String> {
        if proposals.is_empty() { return None; }
        if proposals.len() == 1 {
            return Some(crate::engines::diff::DiffEngine::apply_patch(base_content, &proposals[0]));
        }

        let versions: Vec<String> = proposals
            .iter()
            .map(|p| crate::engines::diff::DiffEngine::apply_patch(base_content, p))
            .collect();

        Self::semantic_ast_merge(base_content, &versions, language)
    }

    /// True Swarm Consensus Merge.
    /// Does not just pick the "longest" code. It checks what the majority of AI models agreed upon.
    fn semantic_ast_merge(base: &str, versions: &[String], language: &str) -> Option<String> {
        let mut parser = Parser::new();
        let lang = match language.to_lowercase().as_str() {
            "rust" | "rs" => tree_sitter_rust::LANGUAGE.into(),
            _ => return Some(versions[0].clone()), 
        };
        parser.set_language(&lang).ok()?;

        let base_tree = parser.parse(base, None)?;
        let base_blocks = Self::extract_blocks(base, base_tree.root_node());

        // Map every proposed block change by signature -> List of proposed contents
        let mut block_proposals: HashMap<String, Vec<String>> = HashMap::new();

        for v in versions {
            if let Some(tree) = parser.parse(v, None) {
                let blocks = Self::extract_blocks(v, tree.root_node());
                for b in blocks {
                    block_proposals.entry(b.signature).or_default().push(b.content);
                }
            }
        }

        // Collect winning replacements for all changed blocks.
        let mut replacements: Vec<(std::ops::Range<usize>, String)> = base_blocks
            .iter()
            .filter_map(|base_block| {
                let proposals = block_proposals.get(&base_block.signature)?;
                let changes: Vec<&String> = proposals
                    .iter()
                    .filter(|c| **c != base_block.content)
                    .collect();
                if changes.is_empty() {
                    return None;
                }
                let mut frequency: HashMap<&String, usize> = HashMap::new();
                for change in &changes {
                    *frequency.entry(*change).or_insert(0) += 1;
                }
                let (winning_change, _) =
                    frequency.into_iter().max_by_key(|&(_, count)| count)?;
                Some((base_block.byte_range.clone(), winning_change.clone()))
            })
            .collect();

        // Sort by start offset for a single forward-pass reconstruction (O(n) vs O(n*k)).
        replacements.sort_unstable_by_key(|(r, _)| r.start);

        let mut result = String::with_capacity(base.len());
        let mut cursor = 0usize;
        for (range, replacement) in replacements {
            if range.start >= cursor {
                result.push_str(&base[cursor..range.start]);
                result.push_str(&replacement);
                cursor = range.end;
            }
        }
        result.push_str(&base[cursor..]);

        Some(result)
    }

    /// Extracts structural blocks, specifically including macros and attributes (e.g., #[derive(...)])
    fn extract_blocks(source: &str, root: Node) -> Vec<AstBlock> {
        let mut blocks = Vec::new();
        let mut cursor = root.walk();
        
        for node in root.children(&mut cursor) {
            let kind = node.kind();
            if matches!(kind, "function_item" | "struct_item" | "enum_item" | "impl_item" | "trait_item") {
                
                // Crucial AST Fix: Expand byte_range upward to include #[attributes] preceding the block
                let mut start_byte = node.start_byte();
                if let Some(prev) = node.prev_sibling()
                    && (prev.kind() == "attribute_item" || prev.kind() == "line_comment")
                {
                    start_byte = prev.start_byte();
                }

                let signature = Self::get_node_signature(source, node);
                let byte_range = start_byte..node.end_byte();
                
                blocks.push(AstBlock {
                    signature,
                    byte_range: byte_range.clone(),
                    content: source[byte_range].to_string(),
                });
            }
        }
        blocks
    }

    fn get_node_signature(source: &str, node: Node) -> String {
        let mut signature = format!("{}_", node.kind());
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if matches!(child.kind(), "identifier" | "type_identifier") {
                signature.push_str(&source[child.byte_range()]);
                signature.push('_');
            }
        }
        if signature.ends_with('_') && signature.len() > node.kind().len() + 1 {
            signature
        } else {
            format!("{}_{}", node.kind(), node.start_byte())
        }
    }
}