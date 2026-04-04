use crate::agent_trait::PromptAgent;
use crate::diff::DiffEngine;
use crate::state::StateManager;
use crate::types::{Artifact, ConversationState, Turn};
use anyhow::{Context, Result};
use std::collections::HashMap;

pub struct Orchestrator {
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
}

impl Orchestrator {
    pub fn new(state_manager: StateManager, agents: Vec<Box<dyn PromptAgent>>) -> Self {
        Self {
            agents,
            state_manager,
        }
    }

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
            .map_err(|e| anyhow::anyhow!("Agent failure: {:?}", e))?;

        // Δα Capture: Extract artifacts from response
        let proposed_artifacts = self.parse_artifacts(&response);
        let mut turn_diffs = vec![];

        for (name, new_content) in proposed_artifacts {
            let current_artifact = sigma.artifacts.entry(name.clone()).or_insert(Artifact {
                name: name.clone(),
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
    fn parse_artifacts(&self, response: &str) -> HashMap<String, String> {
        let mut artifacts = HashMap::new();
        let mut lines = response.lines().peekable();

        while let Some(line) = lines.next() {
            if line.starts_with("```") && line.contains(':') {
                let parts: Vec<&str> = line.trim_start_matches('`').split(':').collect();
                if parts.len() >= 2 {
                    let name = parts[1].trim();
                    let mut content = String::new();
                    while let Some(inner_line) = lines.next() {
                        if inner_line.starts_with("```") {
                            break;
                        }
                        content.push_str(inner_line);
                        content.push('\n');
                    }
                    artifacts.insert(name.to_string(), content.trim_end().to_string());
                }
            }
        }
        artifacts
    }

    /// Rewind :: σ_t ← σ_{t-k}
    pub fn rewind(&self, index: u32) -> Result<ConversationState> {
        self.state_manager
            .restore(index)?
            .context(format!("Failed to rewind to index {}", index))
    }

    /// Resume :: Continue from the latest or specified state
    pub fn resume(&self, index: u32) -> Result<ConversationState> {
        self.rewind(index)
    }

    fn build_prompt(&self, sigma: &ConversationState) -> String {
        let mut p = format!("Project Context: {}\n\nHistory:\n", sigma.session_id);
        for t in sigma.turns.iter().rev().take(10).rev() {
            p.push_str(&format!("{}: {}\n", t.model_id, t.content));
        }
        p.push_str("\nRefine artifacts or debate the solution. Use ```lang:filename to propose changes. Tag completion with 'OPTIMAL'.");
        p
    }
}
