use crate::types::{ConversationState, Turn};
use crate::state::StateManager;
use crate::agent_trait::PromptAgent;
use anyhow::{Result, Context};

pub struct Orchestrator {
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
}

impl Orchestrator {
    pub fn new(state_manager: StateManager, agents: Vec<Box<dyn PromptAgent>>) -> Self {
        Self { agents, state_manager }
    }

    pub async fn run_turn(&self, sigma: &mut ConversationState) -> Result<bool> {
        // Ensure initial state is checkpointed if it's the first turn
        if sigma.iteration_index == 0 && sigma.turns.is_empty() {
             self.state_manager.checkpoint(sigma)?;
        }

        let agent_idx = (sigma.iteration_index as usize) % self.agents.len();
        let agent = &self.agents[agent_idx];
        let model_id = agent.name();

        println!("--- Turn {} | Model: {} ---", sigma.iteration_index, model_id);

        let prompt = self.build_prompt(sigma);
        let response = agent.prompt(&prompt).await.map_err(|e| anyhow::anyhow!("Agent failure: {:?}", e))?;

        let turn = Turn {
            index: sigma.iteration_index,
            model_id: model_id.to_string(),
            content: response.clone(),
            timestamp: ConversationState::now(),
        };
        sigma.turns.push(turn);
        sigma.iteration_index += 1;

        self.state_manager.checkpoint(sigma)?;

        let is_optimal = response.contains("OPTIMAL") || response.contains("CONVERGED");
        Ok(is_optimal)
    }

    /// Rewind :: σ_t ← σ_{t-k}
    pub fn rewind(&self, index: u32) -> Result<ConversationState> {
        self.state_manager.restore(index)?
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
        p.push_str("\nRefine artifacts or debate the solution. Tag completion with 'OPTIMAL'.");
        p
    }
}
