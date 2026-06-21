use crate::types::artifact::{Artifact, ArtifactDiff};
use crate::types::compute::BudgetLedger;
use crate::types::fiduciary::PersonaDisclosure;
use crate::types::planning::GoalTree;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// The observable result of a single model turn.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TurnOutcome {
    /// The generated artifact compiled successfully.
    Compiled,
    /// All tests passed after applying the turn's changes.
    TestsPassed,
    /// The turn moved the session closer to the goal.
    AdvancedConvergence,
    /// The turn was reverted; prior state restored.
    RolledBack,
    /// The turn was rejected by the consensus engine.
    Rejected,
    /// The artifact failed formal verification (Verus).
    VerificationFailed,
    /// No meaningful progress was made.
    Stalled,
    /// Outcome has not been evaluated yet.
    Unknown,
}

/// The prompt layout strategy used when generating a turn.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TurnStructure {
    /// Unconstrained free-form output.
    FreeForm,
    /// Numbered step-by-step reasoning.
    StepByStep,
    /// Explicit pros/cons enumeration.
    ProsCons,
    /// State a hypothesis then validate it.
    HypothesisTest,
    /// Lead with code, follow with explanation.
    CodeFirst,
    /// Symbolic logic and mathematical notation.
    Symbolic,
}

/// High-level category used for routing, analytics, and prompt selection.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TaskCategory {
    CodeGeneration,
    Debugging,
    Architecture,
    Refactoring,
    Research,
    Testing,
    General,
}

impl TaskCategory {
    pub fn preferred_structure(self) -> TurnStructure {
        match self {
            TaskCategory::Research => TurnStructure::Symbolic,
            TaskCategory::CodeGeneration => TurnStructure::CodeFirst,
            TaskCategory::General => TurnStructure::FreeForm,
            _ => TurnStructure::StepByStep,
        }
    }

    pub fn token_estimate(self) -> u32 {
        match self {
            TaskCategory::Architecture => 2500,
            TaskCategory::Research => 2200,
            TaskCategory::CodeGeneration => 2000,
            TaskCategory::Refactoring => 1800,
            TaskCategory::General => 1000,
            TaskCategory::Debugging | TaskCategory::Testing => 1500,
        }
    }
}

/// A single model response within a session, including its diff, metadata, and
/// cryptographic signature.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Turn {
    pub index: u32,
    pub model_id: String,
    pub content: String,
    pub timestamp: u64,
    pub diffs: Vec<(String, ArtifactDiff)>,
    #[serde(default)]
    pub certainty: Option<f64>,
    #[serde(default = "default_outcome")]
    pub outcome: TurnOutcome,
    pub task_category: Option<TaskCategory>,
    pub structure: Option<TurnStructure>,
    #[serde(default)]
    pub signature: Vec<u8>,
    #[serde(default)]
    pub surprise_signal: Option<f64>,
    #[serde(default)]
    pub consistency_score: Option<f64>,
    /// Per-agent diff quality score at the time this turn was committed. Stored
    /// here for observability; the live score lives in IntelligenceEngine.
    #[serde(default)]
    pub diff_quality_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona_disclosure: Option<PersonaDisclosure>,
}

fn default_outcome() -> TurnOutcome {
    TurnOutcome::Unknown
}

/// SHA-256 over a turn's canonical serialization (content, metadata, and
/// signature). Field order is fixed by the struct, so the digest is stable.
fn turn_content_hash(turn: &Turn) -> [u8; 32] {
    let bytes = serde_json::to_vec(turn).unwrap_or_default();
    sha2::Sha256::digest(bytes).into()
}

/// Full mutable state of a running session, persisted to Sled on every checkpoint.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ConversationState {
    pub session_id: String,
    pub iteration_index: u32,
    pub turns: Vec<Turn>,
    pub artifacts: BTreeMap<String, Arc<Artifact>>,
    #[serde(default)]
    pub agent_weights: BTreeMap<String, f64>,
    #[serde(default)]
    pub completion_probability: f64,
    #[serde(default)]
    pub state_hash: [u8; 32],
    #[serde(default)]
    pub budget: BudgetLedger,
    #[serde(default)]
    pub goal_tree: GoalTree,
    #[serde(default)]
    pub node_consensus: BTreeMap<String, f64>,
    #[serde(default)]
    pub last_verification: Vec<(String, String, bool)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal_id: Option<String>,
    #[serde(default)]
    pub mode_library: crate::types::mode::ModeLibrary,
    #[serde(default)]
    pub novel_signal: Option<String>,
    #[serde(default)]
    pub last_tool_outputs: Vec<(String, String)>,
    #[serde(default)]
    pub rejection_loop_active: bool,
    #[serde(default)]
    pub mode_active_turns: u32,
    /// Running hash chain over `turns`: `turn_hashes[i]` commits to
    /// `turn_hashes[i-1]` and the content of `turns[i]`, so any edit,
    /// reorder, insertion, or deletion within the retained window is
    /// detectable without a secret key. Maintained in lockstep by `push_turn`.
    #[serde(default)]
    pub turn_hashes: Vec<Vec<u8>>,
}

impl ConversationState {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            iteration_index: 0,
            turns: vec![],
            artifacts: BTreeMap::new(),
            agent_weights: BTreeMap::new(),
            completion_probability: 0.0,
            state_hash: [0u8; 32],
            budget: BudgetLedger::default(),
            goal_tree: GoalTree::default(),
            node_consensus: BTreeMap::new(),
            last_verification: Vec::new(),
            principal_id: None,
            mode_library: crate::types::mode::ModeLibrary::new(),
            novel_signal: None,
            last_tool_outputs: Vec::new(),
            rejection_loop_active: false,
            mode_active_turns: 0,
            turn_hashes: Vec::new(),
        }
    }

    pub fn ingest_file(&mut self, name: String, language: String, content: String) {
        const MAX_FILE_BYTES: usize = 10_000_000;
        if content.len() > MAX_FILE_BYTES {
            tracing::warn!(file = %name, bytes = content.len(), "ingest_file: file exceeds 10 MB limit; skipping");
            return;
        }
        use crate::engines::validation::AstValidator;
        use crate::types::artifact::Artifact;
        let skeleton = AstValidator::generate_skeleton(&content, &language);
        let artifact = Artifact {
            name: name.clone(),
            language,
            content,
            version: 0,
            history: vec![],
            ast_versions: std::collections::BTreeMap::new(),
            proof_attachments: vec![],
            metrics: crate::engines::quality::ArtifactMetrics::default(),
            skeleton,
        };
        let all_names: Vec<String> = self.artifacts.keys().cloned().collect();
        let metrics =
            crate::engines::quality::QualityEngine::analyze_artifact(&artifact, &all_names);
        self.artifacts.insert(
            name,
            std::sync::Arc::new(Artifact {
                metrics,
                ..artifact
            }),
        );
    }

    pub fn push_turn(&mut self, turn: Turn) {
        const MAX_TURNS: usize = 200;
        // Extend the tamper-evident hash chain before appending the turn.
        let mut hasher = sha2::Sha256::new();
        if let Some(prev) = self.turn_hashes.last() {
            hasher.update(prev);
        }
        hasher.update(turn_content_hash(&turn));
        self.turn_hashes.push(hasher.finalize().to_vec());
        self.turns.push(turn);
        if self.turns.len() > MAX_TURNS {
            let excess = self.turns.len() - MAX_TURNS;
            self.turns.drain(..excess);
            // Keep the chain aligned with the retained turns.
            if self.turn_hashes.len() >= excess {
                self.turn_hashes.drain(..excess);
            }
        }
    }

    /// The current head of the turn hash chain (hex), suitable for anchoring in
    /// an external append-only log (e.g. a git commit message). Empty when no
    /// turns have been recorded.
    #[must_use]
    pub fn chain_head_hex(&self) -> String {
        match self.turn_hashes.last() {
            Some(h) => h.iter().map(|b| format!("{b:02x}")).collect(),
            None => String::new(),
        }
    }

    /// Verify the internal consistency of the turn hash chain over the retained
    /// window. Returns the index of the first turn that fails to chain, or
    /// `None` if the chain is intact (or absent, for legacy states predating it).
    #[must_use]
    pub fn verify_chain(&self) -> Option<usize> {
        if self.turn_hashes.is_empty() {
            return None; // legacy state with no recorded chain
        }
        if self.turn_hashes.len() != self.turns.len() {
            return Some(0); // chain/turn count diverged → tampering
        }
        for i in 1..self.turns.len() {
            let mut hasher = sha2::Sha256::new();
            hasher.update(&self.turn_hashes[i - 1]);
            hasher.update(turn_content_hash(&self.turns[i]));
            if self.turn_hashes[i].as_slice() != hasher.finalize().as_slice() {
                return Some(i);
            }
        }
        None
    }

    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
