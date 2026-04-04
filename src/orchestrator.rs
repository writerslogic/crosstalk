use crate::agent_trait::PromptAgent;
use crate::diff::DiffEngine;
use crate::state::StateManager;
use crate::types::{Artifact, ConversationState, Turn};
use crate::validation::AstValidator;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fmt::Write;

pub struct Orchestrator {
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
}

impl Orchestrator {
    #[must_use]
    pub fn new(state_manager: StateManager, agents: Vec<Box<dyn PromptAgent>>) -> Self {
        Self {
            agents,
            state_manager,
        }
    }

    /// # Errors
    /// Returns error if agent failure or state persistence fails.
    pub async fn run_turn(&self, sigma: &mut ConversationState) -> Result<bool> {
        // Ensure initial state is checkpointed if it's the first turn
        if sigma.iteration_index == 0 && sigma.turns.is_empty() {
            self.state_manager.checkpoint(sigma)?;
        }

        let agent_idx = (sigma.iteration_index as usize) % self.agents.len();
        let agent = &self.agents[agent_idx];
        let model_id = agent.name();

        println!(
            "--- Turn {} | Model: {} ---",
            sigma.iteration_index, model_id
        );

        let prompt = self.build_prompt(sigma);
        let response = agent
            .prompt(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Agent failure: {e:?}"))?;

        // Δα Capture: Extract artifacts from response
        let proposed_artifacts = Self::parse_artifacts(&response);
        let mut turn_diffs = vec![];

        for (name, (lang, new_content)) in proposed_artifacts {
            // AST Validation
            if let Err(e) = AstValidator::validate(&new_content, &lang) {
                println!("[diff] artifact \"{name}\" rejected: validation failed for {lang}: {e}");
                continue;
            }

            let current_artifact =
                sigma
                    .artifacts
                    .entry(name.clone())
                    .or_insert_with(|| Artifact {
                        name: name.clone(),
                        language: lang.clone(),
                        content: String::new(),
                        version: 0,
                        history: vec![],
                    });

            if current_artifact.content != new_content {
                let delta = DiffEngine::generate_delta(
                    &current_artifact.content,
                    &new_content,
                    current_artifact.version,
                );

                current_artifact.history.push(delta.clone());
                current_artifact.content = new_content;
                current_artifact.version += 1;
                current_artifact.language = lang; // Update language if it changed

                turn_diffs.push((name, delta));
            }
        }

        let turn = Turn {
            index: sigma.iteration_index,
            model_id: model_id.to_string(),
            content: response.clone(),
            timestamp: ConversationState::now(),
            diffs: turn_diffs,
        };
        sigma.turns.push(turn);
        sigma.iteration_index += 1;

        self.state_manager.checkpoint(sigma)?;

        let is_optimal = response.contains("OPTIMAL") || response.contains("CONVERGED");
        Ok(is_optimal)
    }

    /// Parses the response for artifacts.
    /// Format: ```<lang>:<filename>\ncontent\n```
    /// Returns `HashMap`<Filename, (Language, Content)>
    fn parse_artifacts(response: &str) -> HashMap<String, (String, String)> {
        let mut artifacts = HashMap::new();
        let mut lines = response.lines();

        while let Some(line) = lines.next() {
            if line.starts_with("```") && line.contains(':') {
                let parts: Vec<&str> = line.trim_start_matches('`').split(':').collect();
                if parts.len() >= 2 {
                    let lang = parts[0].trim();
                    let name = parts[1].trim();
                    let mut content = String::new();
                    for inner_line in lines.by_ref() {
                        if inner_line.starts_with("```") {
                            break;
                        }
                        content.push_str(inner_line);
                        content.push('\n');
                    }
                    artifacts.insert(
                        name.to_string(),
                        (lang.to_string(), content.trim_end().to_string()),
                    );
                }
            }
        }
        artifacts
    }

    /// Rewind :: `σ_t` ← σ_{t-k}
    /// # Errors
    /// Returns error if restore fails.
    pub fn rewind(&self, index: u32) -> Result<ConversationState> {
        self.state_manager
            .restore(index)?
            .context(format!("Failed to rewind to index {index}"))
    }

    /// Resume :: Continue from the latest or specified state
    /// # Errors
    /// Returns error if rewind fails.
    pub fn resume(&self, index: u32) -> Result<ConversationState> {
        self.rewind(index)
    }

    fn build_prompt(&self, sigma: &ConversationState) -> String {
        let mut p = format!("Project Context: {}\n\nHistory:\n", sigma.session_id);
        for t in sigma.turns.iter().rev().take(10).rev() {
            let _ = writeln!(p, "{}: {}", t.model_id, t.content);
        }
        p.push_str("\nRefine artifacts or debate the solution. Use ```lang:filename to propose changes. Tag completion with 'OPTIMAL'.");
        p
    }
}
