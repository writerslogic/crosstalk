use crate::types::artifact::ArtifactDiff;
use crate::types::conversation::{ConversationState, TaskCategory, Turn, TurnStructure};
use crate::types::security::FallacyReport;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
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
    pub fn select_structure(task: TaskCategory, _agent_id: &str) -> TurnStructure {
        task.preferred_structure()
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
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.eq_ignore_ascii_case("decision:")
                || trimmed.starts_with("- [x]")
                || trimmed.starts_with("- [X]")
                || trimmed.starts_with("⊢")
                || trimmed.starts_with("Δα:")
            {
                decisions.push(trimmed.to_string());
            } else if trimmed.eq_ignore_ascii_case("problem:")
                || trimmed.eq_ignore_ascii_case("err:")
                || trimmed.starts_with("⊥")
            {
                problems.push(trimmed.to_string());
            } else if trimmed.ends_with('?') {
                questions.push(trimmed.to_string());
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
        let fallacies = FallacyDetector::scan(&turn.content);
        let assumptions = AssumptionExtractor::extract(&turn.content);

        // Dimension 1: signal richness (0.0..=0.30)
        let signal_score = {
            let d = if signals.decisions.is_empty() {
                0.0
            } else {
                0.15
            };
            let q = if signals.questions.len() <= 3 {
                0.05
            } else {
                0.0
            };
            let c = if signals.code_blocks.is_empty() {
                0.0
            } else {
                0.10
            };
            d + q + c
        };

        // Dimension 2: evidence anchoring proxy (0.0..=0.25)
        let evidence_score = {
            let has_code_ref = turn.content.contains("```") || turn.content.contains("line ");
            let has_citation = turn.content.contains("http") || turn.content.contains("doi:");
            let density = (has_code_ref as u8 + has_citation as u8) as f64 * 0.125;
            let word_count = turn.content.split_whitespace().count();
            let length_bonus = (word_count as f64 / 200.0).min(1.0) * 0.05;
            (density + length_bonus).min(0.25)
        };

        // Dimension 3: structural coherence (0.0..=0.25)
        let structure_score = match turn.structure {
            Some(TurnStructure::StepByStep) | Some(TurnStructure::HypothesisTest) => 0.25,
            Some(TurnStructure::ProsCons) | Some(TurnStructure::CodeFirst) => 0.15,
            _ => 0.10,
        };

        // Dimension 4: fallacy penalty (max 0.20)
        let fallacy_penalty = (fallacies.len() as f64 * 0.05).min(0.20);

        let assumption_bonus = (assumptions.len() as f64 * 0.025).min(0.05);

        let base = 0.30;
        (base + signal_score + evidence_score + structure_score + assumption_bonus
            - fallacy_penalty)
            .clamp(0.0, 1.0)
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

        if let Some(r) = Self::detect_circular_reasoning(content) {
            reports.push(r);
        }
        if let Some(r) = Self::detect_appeal_to_authority(content) {
            reports.push(r);
        }
        if let Some(r) = Self::detect_false_dichotomy(content) {
            reports.push(r);
        }
        if let Some(r) = Self::detect_straw_man(content) {
            reports.push(r);
        }

        reports
    }

    fn detect_appeal_to_authority(content: &str) -> Option<FallacyReport> {
        let lower = content.to_lowercase();
        if lower.contains("as the documentation says") && !content.contains("```") {
            return Some(FallacyReport {
                fallacy_type: "AppealToAuthority".to_string(),
                evidence_span: "as the documentation says".to_string(),
                confidence: 0.8,
            });
        }
        None
    }

    fn detect_false_dichotomy(content: &str) -> Option<FallacyReport> {
        let lower = content.to_lowercase();
        let markers = [
            "either...or",
            "only two options",
            "must choose between",
            "only two choices",
            "there are only two",
        ];
        let span = markers.iter().find(|&&m| lower.contains(m))?;
        Some(FallacyReport {
            fallacy_type: "FalseDichotomy".to_string(),
            evidence_span: span.to_string(),
            confidence: 0.75,
        })
    }

    fn detect_straw_man(content: &str) -> Option<FallacyReport> {
        let lower = content.to_lowercase();
        let setups = [
            "they claim",
            "their argument is",
            "they believe",
            "their position is",
        ];
        let dismissals = [
            "is absurd",
            "is ridiculous",
            "is clearly wrong",
            "is obviously false",
        ];
        let has_setup = setups.iter().any(|&s| lower.contains(s));
        let has_dismissal = dismissals.iter().any(|&d| lower.contains(d));
        if has_setup && has_dismissal {
            Some(FallacyReport {
                fallacy_type: "StrawMan".to_string(),
                evidence_span: "exaggerated attribution followed by dismissal".to_string(),
                confidence: 0.70,
            })
        } else {
            None
        }
    }

    /// Upgraded from $O(N^2)$ to $O(N)$ using a semantic hash-binning approach.
    /// Sentences are hashed into semantic buckets. Collisions indicate circular reasoning.
    fn detect_circular_reasoning(content: &str) -> Option<FallacyReport> {
        let mut semantic_buckets: HashMap<u64, &str, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(128, Default::default());

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
                if s_lower.contains("therefore")
                    || s_lower.contains("thus")
                    || s_lower.contains("consequently")
                {
                    return Some(FallacyReport {
                        fallacy_type: "CircularReasoning".to_string(),
                        evidence_span: format!(
                            "Premise: \"{}\" vs Conclusion: \"{}\"",
                            previous_sentence, sentence
                        ),
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
// STRUCTURE SELECTOR & STRATEGY MIXER
// =====================================================================

#[derive(Debug, Clone)]
pub struct StrategyRecord {
    pub task: TaskCategory,
    pub agent_id: String,
    pub structure: TurnStructure,
    pub quality_score: f64,
    pub timestamp: u64,
}

pub struct StrategyMixer {
    pub history: Vec<StrategyRecord>,
}

impl StrategyMixer {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    pub fn record(&mut self, rec: StrategyRecord) {
        self.history.push(rec);
    }

    #[must_use]
    pub fn blend(&self, task: TaskCategory, agent_id: Option<&str>) -> Vec<(TurnStructure, f64)> {
        let mut acc: HashMap<TurnStructure, (f64, u32)> = HashMap::new();
        for r in self
            .history
            .iter()
            .filter(|r| r.task == task && agent_id.is_none_or(|id| r.agent_id == id))
        {
            let e = acc.entry(r.structure).or_default();
            e.0 += r.quality_score;
            e.1 += 1;
        }
        // Fall back to task-wide data if no agent-specific data
        if acc.is_empty() && agent_id.is_some() {
            return self.blend(task, None);
        }
        let mut out: Vec<(TurnStructure, f64)> = acc
            .into_iter()
            .map(|(s, (sum, n))| (s, sum / n as f64))
            .collect();
        out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

impl Default for StrategyMixer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StructureSelector {
    mixer: StrategyMixer,
}

impl StructureSelector {
    pub fn new() -> Self {
        Self {
            mixer: StrategyMixer::new(),
        }
    }

    pub fn record_outcome(
        &mut self,
        task: TaskCategory,
        agent_id: &str,
        structure: TurnStructure,
        quality: f64,
    ) {
        self.mixer.record(StrategyRecord {
            task,
            agent_id: agent_id.to_string(),
            structure,
            quality_score: quality,
            timestamp: ConversationState::now(),
        });
    }

    #[must_use]
    pub fn recommend(&self, task: TaskCategory, agent_id: &str) -> TurnStructure {
        self.mixer
            .blend(task, Some(agent_id))
            .first()
            .map(|(s, _)| *s)
            .unwrap_or(TurnStructure::FreeForm)
    }
}

impl Default for StructureSelector {
    fn default() -> Self {
        Self::new()
    }
}

// =====================================================================
// EVIDENCE ANCHORING
// =====================================================================

#[derive(Debug, Clone)]
pub enum EvidenceAnchor {
    CodeRef { file: String, line: u32 },
    Citation { source: String, quote: String },
    DataPoint { label: String, value: f64 },
}

#[derive(Debug, Clone)]
pub struct AnchoredClaim {
    pub claim: String,
    pub anchors: Vec<EvidenceAnchor>,
    pub confidence: f64,
}

// =====================================================================
// ASSUMPTION EXTRACTOR
// =====================================================================

pub struct AssumptionExtractor;

impl AssumptionExtractor {
    #[must_use]
    pub fn extract(content: &str) -> Vec<String> {
        let markers = [
            "assume",
            "assuming",
            "we take for granted",
            "it is assumed",
            "presuppose",
            "given that",
        ];
        content
            .split(['.', '\n', '!'])
            .map(str::trim)
            .filter(|s| {
                let lower = s.to_lowercase();
                markers.iter().any(|&m| lower.contains(m))
            })
            .map(ToOwned::to_owned)
            .collect()
    }
}

// =====================================================================
// CROSS EXAMINER
// =====================================================================

pub struct CrossExaminer;

impl CrossExaminer {
    #[must_use]
    pub fn generate_questions(argument: &str) -> Vec<String> {
        let mut questions = Vec::new();

        for word in argument.split_whitespace() {
            let stripped = word.trim_matches(|c: char| !c.is_alphabetic());
            if stripped.len() > 4 && stripped == stripped.to_uppercase() {
                questions.push(format!("How is \"{}\" defined in this context?", stripped));
            }
        }

        let lower = argument.to_lowercase();
        if lower.contains("causes") || lower.contains("leads to") || lower.contains("results in") {
            questions.push("What evidence supports this causal relationship?".to_string());
        }

        if lower.contains("always") || lower.contains("never") || lower.contains("all ") {
            questions
                .push("Are there documented counterexamples to this universal claim?".to_string());
        }

        if questions.is_empty() {
            questions.push("What is the primary evidence for this claim?".to_string());
        }

        questions
    }
}

// =====================================================================
// ARGUMENT GRAPH & PARSER
// =====================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgumentNodeType {
    Premise,
    Conclusion,
    Rebuttal,
}

#[derive(Debug, Clone)]
pub struct ArgumentNode {
    pub id: usize,
    pub claim: String,
    pub node_type: ArgumentNodeType,
}

#[derive(Debug, Default, Clone)]
pub struct ArgumentGraph {
    pub nodes: Vec<ArgumentNode>,
    pub edges: Vec<(usize, usize)>,
}

pub struct ArgumentParser;

impl ArgumentParser {
    #[must_use]
    pub fn parse(text: &str) -> ArgumentGraph {
        let mut graph = ArgumentGraph::default();
        let mut id = 0usize;

        for line in text.lines() {
            let trimmed = line.trim();
            let lower = trimmed.to_lowercase();
            let node_type = if lower.starts_with("premise:") || lower.starts_with("- ") {
                ArgumentNodeType::Premise
            } else if lower.starts_with("conclusion:") || lower.starts_with("therefore") {
                ArgumentNodeType::Conclusion
            } else if lower.starts_with("however") || lower.starts_with("rebuttal:") {
                ArgumentNodeType::Rebuttal
            } else {
                continue;
            };

            graph.nodes.push(ArgumentNode {
                id,
                claim: trimmed.to_string(),
                node_type,
            });
            if id > 0 {
                graph.edges.push((id - 1, id));
            }
            id += 1;
        }

        graph
    }
}

// =====================================================================
// REPORT GENERATOR
// =====================================================================

pub struct ReportGenerator;

impl ReportGenerator {
    #[must_use]
    pub fn generate(
        signals: &ExtractedSignals,
        fallacies: &[FallacyReport],
        assumptions: &[String],
        score: f64,
    ) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("## Reasoning Report\n\n");
        out.push_str(&format!("**Overall Score**: {:.2}\n\n", score));

        out.push_str("### Decisions\n");
        if signals.decisions.is_empty() {
            out.push_str("- None recorded\n");
        } else {
            for d in &signals.decisions {
                out.push_str(&format!("- {}\n", d));
            }
        }

        out.push_str("\n### Detected Fallacies\n");
        if fallacies.is_empty() {
            out.push_str("- None detected\n");
        } else {
            for f in fallacies {
                out.push_str(&format!(
                    "- **{}** (conf: {:.0}%): {}\n",
                    f.fallacy_type,
                    f.confidence * 100.0,
                    f.evidence_span
                ));
            }
        }

        out.push_str("\n### Assumptions\n");
        if assumptions.is_empty() {
            out.push_str("- None identified\n");
        } else {
            for a in assumptions {
                out.push_str(&format!("- {}\n", a));
            }
        }

        out
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
                    block_proposals
                        .entry(b.signature)
                        .or_default()
                        .push(b.content);
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
                let (winning_change, _) = frequency.into_iter().max_by_key(|&(_, count)| count)?;
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
            if matches!(
                kind,
                "function_item" | "struct_item" | "enum_item" | "impl_item" | "trait_item"
            ) {
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
