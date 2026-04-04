use crate::agent_trait::PromptAgent;
use crate::diff::DiffEngine;
use crate::state::StateManager;
use crate::types::{Artifact, ControlSignal, ConversationState, StreamEvent, Turn};
use crate::validation::AstValidator;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fmt::Write;
use tokio::sync::mpsc;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Orchestrator {
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
    event_tx: mpsc::Sender<StreamEvent>,
    control_rx: Mutex<mpsc::Receiver<ControlSignal>>,
}

impl Orchestrator {
    #[must_use]
    pub fn new(
        state_manager: StateManager,
        agents: Vec<Box<dyn PromptAgent>>,
        event_tx: mpsc::Sender<StreamEvent>,
        control_rx: mpsc::Receiver<ControlSignal>,
    ) -> Self {
        Self {
            agents,
            state_manager,
            event_tx,
            control_rx: Mutex::new(control_rx),
        }
    }

    /// # Errors
    /// Returns error if agent failure or state persistence fails.
    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        let (iteration_index, prompt) = {
            let s = sigma_lock.lock().await;
            // Ensure initial state is checkpointed if it's the first turn
            if s.iteration_index == 0 && s.turns.is_empty() {
                self.state_manager.checkpoint(&s)?;
            }
            (s.iteration_index, self.build_prompt(&s))
        };

        let agent_idx = (iteration_index as usize) % self.agents.len();
        let agent = &self.agents[agent_idx];
        let model_id = agent.name();

        println!(
            "--- Turn {} | Model: {} ---",
            iteration_index, model_id
        );

        // Real-time Token Piping (Ghost-Stream)
        let mut stream = agent
            .stream_prompt(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Agent failure: {e:?}"))?;

        let mut response = String::new();
        let mut paused = false;

        loop {
            // 1. Handle Control Signals
            {
                let mut rx = self.control_rx.lock().await;
                while let Ok(signal) = rx.try_recv() {
                    match signal {
                        ControlSignal::Pause => paused = true,
                        ControlSignal::Resume => paused = false,
                        ControlSignal::Shutdown => return Ok(false),
                        ControlSignal::Inject(text) => {
                            response.push_str(&text);
                            let _ = self.event_tx.send(StreamEvent::TokenReceived(text)).await;
                        }
                        _ => {}
                    }
                }
            }

            if paused {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                continue;
            }

            // 2. Process Model Stream
            match stream.next().await {
                Some(chunk_res) => {
                    let chunk = chunk_res.map_err(|e| anyhow::anyhow!("Stream error: {e:?}"))?;
                    response.push_str(&chunk);
                    let _ = self.event_tx.send(StreamEvent::TokenReceived(chunk)).await;
                }
                None => break, // Stream finished
            }
        }

        // Δα Capture: Extract artifacts from response
        let proposed_artifacts = Self::parse_artifacts(&response);
        let mut turn_diffs = vec![];

        {
            let mut sigma = sigma_lock.lock().await;

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
            sigma.turns.push(turn.clone());
            sigma.iteration_index += 1;

            self.state_manager.checkpoint(&sigma)?;
            
            // Signal turn completion to UI
            let _ = self.event_tx.send(StreamEvent::TokenReceived("\n[Turn Complete]\n".to_string())).await;
            let _ = self.event_tx.send(StreamEvent::TurnComplete(turn)).await;
        }

        let is_optimal = response.contains("OPTIMAL") || response.contains("CONVERGED");
        Ok(is_optimal)
    }

    /// Parses the response for artifacts. 
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

    pub fn rewind(&self, index: u32) -> Result<ConversationState> {
        self.state_manager
            .restore(index)?
            .context(format!("Failed to rewind to index {index}"))
    }

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
