use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum ContextDistribution {
    #[default]
    Shared,
    Divergent,
    RoleFiltered,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum ConvergenceDirection {
    #[default]
    TowardAgreement,
    TowardDivergence,
    TowardTradeoffMap,
    TowardNovelty,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum SurpriseHandling {
    Amplify,
    #[default]
    Suppress,
    Neutral,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum TerminationCondition {
    #[default]
    OptimalSignal,
    Exhaustion {
        max_turns: u32,
    },
    RejectionCycles {
        n: u32,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum RoleAssignment {
    #[default]
    Homogeneous,
    AdversarialPairs,
    Specialized,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub enum LoopStructure {
    #[default]
    Linear,
    RejectionLoop,
    TreeSearch,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModeDefinition {
    pub name: String,
    pub description: String,
    pub context_distribution: ContextDistribution,
    pub convergence_direction: ConvergenceDirection,
    pub surprise_handling: SurpriseHandling,
    pub termination: TerminationCondition,
    pub role_assignment: RoleAssignment,
    pub loop_structure: LoopStructure,
    pub prompt_prefix: String,
    /// true if synthesized by the system at runtime
    pub synthesized: bool,
    pub synthesis_reason: Option<String>,
}

impl Default for ModeDefinition {
    fn default() -> Self {
        Self::convergence()
    }
}

impl ModeDefinition {
    pub fn convergence() -> Self {
        Self {
            name: "Convergence".to_string(),
            description: "Find the single best answer and build consensus.".to_string(),
            context_distribution: ContextDistribution::Shared,
            convergence_direction: ConvergenceDirection::TowardAgreement,
            surprise_handling: SurpriseHandling::Suppress,
            termination: TerminationCondition::OptimalSignal,
            role_assignment: RoleAssignment::Homogeneous,
            loop_structure: LoopStructure::Linear,
            prompt_prefix: "[MODE: CONVERGENCE] Identify the single best answer and build consensus. Tag your conclusion with OPTIMAL when you believe it is correct.".to_string(),
            synthesized: false,
            synthesis_reason: None,
        }
    }

    pub fn generative() -> Self {
        Self {
            name: "Generative".to_string(),
            description: "Produce novel ideas that do not yet exist.".to_string(),
            context_distribution: ContextDistribution::Divergent,
            convergence_direction: ConvergenceDirection::TowardNovelty,
            surprise_handling: SurpriseHandling::Amplify,
            termination: TerminationCondition::RejectionCycles { n: 3 },
            role_assignment: RoleAssignment::Specialized,
            loop_structure: LoopStructure::RejectionLoop,
            prompt_prefix: "[MODE: GENERATIVE] Prioritise novel, divergent ideas. Do NOT converge prematurely. Propose at least three structurally distinct approaches. Avoid repeating any framing already in the conversation. Label proposals [PROPOSAL].".to_string(),
            synthesized: false,
            synthesis_reason: None,
        }
    }

    pub fn stress_test() -> Self {
        Self {
            name: "StressTest".to_string(),
            description: "Adversarially attack the current proposal.".to_string(),
            context_distribution: ContextDistribution::Shared,
            convergence_direction: ConvergenceDirection::TowardDivergence,
            surprise_handling: SurpriseHandling::Neutral,
            termination: TerminationCondition::Exhaustion { max_turns: 6 },
            role_assignment: RoleAssignment::AdversarialPairs,
            loop_structure: LoopStructure::Linear,
            prompt_prefix: "[MODE: STRESS-TEST] Your role is adversarial. Find failure modes, edge cases, and hidden assumptions. Tag each attack SEVERITY: LOW|MED|HIGH|CRITICAL. Do not defend or propose fixes.".to_string(),
            synthesized: false,
            synthesis_reason: None,
        }
    }

    pub fn decision() -> Self {
        Self {
            name: "Decision".to_string(),
            description: "Map trade-offs without collapsing to one answer.".to_string(),
            context_distribution: ContextDistribution::Shared,
            convergence_direction: ConvergenceDirection::TowardTradeoffMap,
            surprise_handling: SurpriseHandling::Neutral,
            termination: TerminationCondition::OptimalSignal,
            role_assignment: RoleAssignment::Specialized,
            loop_structure: LoopStructure::Linear,
            prompt_prefix: "[MODE: DECISION] Map the trade-off space. For each option: PROS, CONS, RISK, REVERSIBILITY. Do not recommend unless explicitly asked.".to_string(),
            synthesized: false,
            synthesis_reason: None,
        }
    }

    pub fn synthesis() -> Self {
        Self {
            name: "Synthesis".to_string(),
            description: "Build unified understanding from disparate sources.".to_string(),
            context_distribution: ContextDistribution::RoleFiltered,
            convergence_direction: ConvergenceDirection::TowardAgreement,
            surprise_handling: SurpriseHandling::Neutral,
            termination: TerminationCondition::OptimalSignal,
            role_assignment: RoleAssignment::Homogeneous,
            loop_structure: LoopStructure::Linear,
            prompt_prefix: "[MODE: SYNTHESIS] Build a unified model from all provided sources. Identify contradictions and gaps. End your response with §GAP-LIST of unresolved unknowns.".to_string(),
            synthesized: false,
            synthesis_reason: None,
        }
    }

    pub fn socratic() -> Self {
        Self {
            name: "Socratic".to_string(),
            description: "Guide the user to articulate what they actually want.".to_string(),
            context_distribution: ContextDistribution::Shared,
            convergence_direction: ConvergenceDirection::TowardAgreement,
            surprise_handling: SurpriseHandling::Neutral,
            termination: TerminationCondition::OptimalSignal,
            role_assignment: RoleAssignment::Homogeneous,
            loop_structure: LoopStructure::Linear,
            prompt_prefix: "[MODE: SOCRATIC] Do not answer the question directly. Ask clarifying questions to help the user articulate what they actually want. Surface hidden assumptions in the task statement.".to_string(),
            synthesized: false,
            synthesis_reason: None,
        }
    }

    pub fn presets() -> Vec<Self> {
        vec![
            Self::convergence(),
            Self::generative(),
            Self::stress_test(),
            Self::decision(),
            Self::synthesis(),
            Self::socratic(),
        ]
    }

    /// Detect the best preset for a task string. Returns index into `presets()`.
    pub fn detect_preset_index(task: &str) -> usize {
        let lower = task.to_lowercase();
        if lower.contains("brainstorm")
            || lower.contains("novel")
            || lower.contains("invent")
            || lower.contains("generate ideas")
            || lower.contains("explore")
            || lower.contains("new ideas")
            || lower.contains("beyond")
            || lower.contains("better than")
        {
            1 // Generative
        } else if lower.contains("attack")
            || lower.contains("critique")
            || lower.contains("stress test")
            || lower.contains("find flaws")
            || lower.contains("what could go wrong")
            || lower.contains("weaknesses")
            || lower.contains("vulnerabilities")
        {
            2 // StressTest
        } else if lower.contains("trade-off")
            || lower.contains("tradeoff")
            || lower.contains("compare")
            || lower.contains("pros and cons")
            || lower.contains("options")
            || lower.contains("versus")
        {
            3 // Decision
        } else if lower.contains("synthesize")
            || lower.contains("synthesise")
            || lower.contains("summarize")
            || lower.contains("summarise")
            || lower.contains("understand")
            || lower.contains("integrate")
        {
            4 // Synthesis
        } else if lower.contains("help me figure out")
            || lower.contains("not sure what")
            || lower.contains("clarify")
            || lower.contains("what should i")
        {
            5 // Socratic
        } else {
            0 // Convergence (default)
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModeLibrary {
    pub modes: Vec<ModeDefinition>,
    pub current_index: usize,
}

impl Default for ModeLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl ModeLibrary {
    pub fn new() -> Self {
        Self {
            modes: ModeDefinition::presets(),
            current_index: 0,
        }
    }

    pub fn current(&self) -> &ModeDefinition {
        &self.modes[self.current_index.min(self.modes.len().saturating_sub(1))]
    }

    pub fn current_name(&self) -> &str {
        self.current().name.as_str()
    }

    /// Add or replace a mode by name. Returns the index.
    pub fn upsert(&mut self, mode: ModeDefinition) -> usize {
        if let Some(i) = self.modes.iter().position(|m| m.name == mode.name) {
            self.modes[i] = mode;
            i
        } else {
            self.modes.push(mode);
            self.modes.len() - 1
        }
    }

    /// Switch to a mode by name. Returns true if found.
    pub fn switch_to_name(&mut self, name: &str) -> bool {
        if let Some(i) = self.modes.iter().position(|m| m.name == name) {
            self.current_index = i;
            true
        } else {
            false
        }
    }

    pub fn switch_to_index(&mut self, i: usize) -> bool {
        if i < self.modes.len() {
            self.current_index = i;
            true
        } else {
            false
        }
    }

    /// Cycle to the next mode (wraps around).
    pub fn cycle_next(&mut self) -> &ModeDefinition {
        if self.modes.is_empty() {
            return self.current();
        }
        self.current_index = (self.current_index + 1) % self.modes.len();
        self.current()
    }

    /// Try to parse a synthesized ModeDefinition from agent JSON output.
    /// Looks for a JSON block starting with `{"name":` in the text.
    pub fn try_parse_synthesized(text: &str, reason: String) -> Option<ModeDefinition> {
        let start = text.find("{\"name\":")?;
        let fragment = &text[start..];
        let end = fragment
            .find("\n}\n")
            .or_else(|| fragment.find("\n}"))
            .map(|i| i + 2)
            .unwrap_or(fragment.len());
        let json = &fragment[..end];
        let mut def: ModeDefinition = serde_json::from_str(json).ok()?;
        def.synthesized = true;
        def.synthesis_reason = Some(reason);
        Some(def)
    }
}
