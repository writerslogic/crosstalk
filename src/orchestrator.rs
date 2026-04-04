use crate::agent_trait::PromptAgent;
use crate::diff::DiffEngine;
use crate::state::StateManager;
use crate::types::{Artifact, ControlSignal, ConversationState, StreamEvent, Turn, TurnOutcome, TaskCategory, CostEntry, TokenUsage, TurnStructure};
use crate::validation::AstValidator;
use crate::consensus::{CertaintyAnalyzer, KalmanConvergence, InfluenceWeightManager};
use crate::sandbox::SandboxManager;
use crate::simulation::MonteCarloRunner;
use crate::mcp::McpGateway;
use crate::environment::ToolDiscovery;
use crate::verification::{HashChain, TautologyFilter};
use crate::proof::ProofManager;
use crate::memory::{MemoryStore, ContextDistiller};
use crate::intelligence::{IntelligenceEngine, QualityScorer};
use crate::compute::ComputeManager;
use crate::reasoning::{ReasoningEngine, FallacyDetector};
use crate::quality::{QualityEngine, RegressionDetector};
use crate::self_improvement::SelfImprovementEngine;
use crate::swarm::SwarmController;
use crate::planning::PlanningEngine;
use crate::security::{SecretScanner, TurnSigner};
use crate::analytics::AnalyticsEngine;
use crate::release::{ReleaseManager, ConvergenceReport};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fmt::Write;
use tokio::sync::mpsc;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Instant;

pub struct Orchestrator {
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
    event_tx: mpsc::Sender<StreamEvent>,
    control_rx: Mutex<mpsc::Receiver<ControlSignal>>,
    sandbox: SandboxManager,
    mc_runner: MonteCarloRunner,
    pub mcp_gateway: McpGateway,
    pub memory_store: MemoryStore,
    pub intelligence: Mutex<IntelligenceEngine>,
    pub compute: Mutex<ComputeManager>,
    pub reasoning: ReasoningEngine,
    pub self_improve: SelfImprovementEngine,
    pub swarm: SwarmController,
    pub planning: PlanningEngine,
    pub signer: TurnSigner,
    pub analytics: AnalyticsEngine,
    pub release: ReleaseManager,
}

impl Orchestrator {
    #[must_use]
    pub fn new(
        state_manager: StateManager,
        agents: Vec<Box<dyn PromptAgent>>,
        event_tx: mpsc::Sender<StreamEvent>,
        control_rx: mpsc::Receiver<ControlSignal>,
    ) -> Self {
        let mut mcp_gateway = McpGateway::new();
        let tools = ToolDiscovery::scan();
        for tool in tools {
            mcp_gateway.register_tool(tool);
        }

        Self {
            agents,
            state_manager,
            event_tx,
            control_rx: Mutex::new(control_rx),
            sandbox: SandboxManager::new().expect("Failed to init sandbox"),
            mc_runner: MonteCarloRunner::new().expect("Failed to init simulation"),
            mcp_gateway,
            memory_store: MemoryStore::new("/tmp/crosstalk-memory"),
            intelligence: Mutex::new(IntelligenceEngine::new()),
            compute: Mutex::new(ComputeManager::new()),
            reasoning: ReasoningEngine,
            self_improve: SelfImprovementEngine,
            swarm: SwarmController::new(),
            planning: PlanningEngine,
            signer: TurnSigner::new(),
            analytics: AnalyticsEngine,
            release: ReleaseManager,
        }
    }

    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        let (iteration_index, prompt, history_contents, agent_id) = {
            let s = sigma_lock.lock().await;
            if s.iteration_index == 0 && s.turns.is_empty() {
                self.state_manager.checkpoint(&s)?;
            }
            let contents: Vec<String> = s.turns.iter().map(|t| t.content.clone()).collect();
            let distilled_prompt = ContextDistiller::distill(&s, 2000);
            
            let agent_idx = (s.iteration_index as usize) % self.agents.len();
            let model_id = self.agents[agent_idx].name().to_string();
            
            let structure = ReasoningEngine::select_structure(TaskCategory::CodeGeneration, &model_id);
            let mut final_prompt = distilled_prompt;
            match structure {
                TurnStructure::StepByStep => final_prompt.push_str("\nStructure your response with numbered reasoning steps."),
                TurnStructure::ProsCons => final_prompt.push_str("\nExplicitly analyze tradeoffs (Pros vs Cons)."),
                TurnStructure::CodeFirst => final_prompt.push_str("\nProvide the code delta (Δα) before any explanation."),
                _ => {}
            }

            (s.iteration_index, final_prompt, contents, model_id)
        };

        println!("--- Turn {} | Model: {} ---", iteration_index, agent_id);

        let agent_idx = (iteration_index as usize) % self.agents.len();
        let agent = &self.agents[agent_idx];

        let start_time = Instant::now();
        let mut stream = agent
            .stream_prompt(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Agent failure: {e:?}"))?;

        let mut response = String::new();
        let mut paused = false;

        loop {
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

            match stream.next().await {
                Some(chunk_res) => {
                    let chunk = chunk_res.map_err(|e| anyhow::anyhow!("Stream error: {e:?}"))?;
                    response.push_str(&chunk);
                    let _ = self.event_tx.send(StreamEvent::TokenReceived(chunk)).await;
                }
                None => break,
            }
        }

        let latency_ms = start_time.elapsed().as_millis() as u64;

        let secrets = SecretScanner::scan(&response);
        if !secrets.is_empty() {
            println!("[security] turn {iteration_index} blocked: detected secrets {:?}", secrets);
            let _ = self.event_tx.send(StreamEvent::TokenReceived("\n[Blocked: Security Violation]\n".to_string())).await;
            return Ok(false);
        }

        if TautologyFilter::is_tautological(&response, &history_contents) {
            println!("[verification] turn {iteration_index} pruned: tautological reasoning detected");
            let _ = self.event_tx.send(StreamEvent::TokenReceived("\n[Pruned: Tautology]\n".to_string())).await;
            return Ok(false);
        }

        let fallacies = FallacyDetector::scan(&response);
        if !fallacies.is_empty() {
            println!("[reasoning] fallacies detected: {:?}", fallacies);
            let _ = self.event_tx.send(StreamEvent::TokenReceived(format!("\n[Warning: {} fallacies detected]\n", fallacies.len()))).await;
        }

        let proposed_artifacts = Self::parse_artifacts(&response);
        let mut turn_diffs = vec![];
        let certainty = CertaintyAnalyzer::compute(&response);
        let mut outcome = TurnOutcome::Unknown;

        {
            let mut sigma = sigma_lock.lock().await;
            let sigma_snapshot = sigma.clone();
            let current_i = sigma.iteration_index;

            let mut all_valid = true;
            for (name, (lang, new_content)) in proposed_artifacts {
                if let Err(e) = AstValidator::validate(&new_content, &lang) {
                    println!("[diff] artifact \"{name}\" rejected: AST validation failed: {e}");
                    all_valid = false;
                    outcome = TurnOutcome::Rejected;
                    break;
                }

                let dups = QualityEngine::detect_duplication(&new_content, &sigma.artifacts);
                if !dups.is_empty() {
                    println!("[quality] duplication detected for \"{name}\": {:?}", dups);
                }

                let current_artifact = sigma.artifacts.entry(name.clone()).or_insert_with(|| Artifact {
                    name: name.clone(),
                    language: lang.clone(),
                    content: String::new(),
                    version: 0,
                    history: vec![],
                    ast_versions: HashMap::new(),
                    proof_attachments: vec![],
                    metrics: crate::quality::ArtifactMetrics::default(),
                });

                if current_artifact.content != new_content {
                    let delta = DiffEngine::generate_delta(&current_artifact.content, &new_content, current_artifact.version);
                    let p_fail = self.mc_runner.predict(current_artifact, &delta, 5).await;
                    if p_fail > 0.5 {
                        println!("[sandbox] artifact \"{name}\" rejected: MC P(fail) = {p_fail}");
                        all_valid = false;
                        outcome = TurnOutcome::RolledBack;
                        break;
                    }

                    let new_metrics = QualityEngine::analyze_artifact(&Artifact {
                        content: new_content.clone(),
                        ..current_artifact.clone()
                    });
                    if RegressionDetector::is_regressive(&current_artifact.metrics, &new_metrics) {
                        println!("[quality] regression blocked for \"{name}\"");
                        all_valid = false;
                        outcome = TurnOutcome::Rejected;
                        break;
                    }

                    current_artifact.history.push(delta.clone());
                    current_artifact.content = new_content.clone();
                    current_artifact.version += 1;
                    current_artifact.language = lang.clone();
                    current_artifact.metrics = new_metrics;

                    let nodes = AstValidator::extract_nodes(&new_content, &lang);
                    for (node_id, content) in nodes {
                        current_artifact.ast_versions.entry(node_id).or_default().push((current_i, content));
                    }

                    let proof = ProofManager::generate_proof(current_artifact, vec!["ast_valid".to_string(), "mc_safe".to_string(), "quality_checked".to_string()]);
                    current_artifact.proof_attachments.push(proof);

                    turn_diffs.push((name, delta));
                    outcome = TurnOutcome::Compiled;
                }
            }

            if !all_valid {
                println!("[sandbox] Rollback triggered: validation pipeline failed.");
                *sigma = sigma_snapshot;
                return Ok(false);
            }

            let mut turn = Turn {
                index: sigma.iteration_index,
                model_id: agent_id.clone(),
                content: response.clone(),
                timestamp: ConversationState::now(),
                diffs: turn_diffs,
                certainty: Some(certainty),
                outcome,
                task_category: Some(TaskCategory::CodeGeneration),
                structure: Some(ReasoningEngine::select_structure(TaskCategory::CodeGeneration, &agent_id)),
                signature: vec![],
            };
            
            let serialized = serde_json::to_vec(&turn).unwrap();
            turn.signature = self.signer.sign(&serialized);

            let quality_score = QualityScorer::score(&turn);
            {
                let mut intell = self.intelligence.lock().await;
                intell.update_profile(&turn, quality_score);
            }

            let cost_entry = CostEntry {
                turn_id: turn.index,
                model_id: agent_id.clone(),
                usage: TokenUsage {
                    input_tokens: prompt.len() as u32 / 4,
                    output_tokens: response.len() as u32 / 4,
                    total_tokens: (prompt.len() + response.len()) as u32 / 4,
                },
                cost_usd: 0.01,
                latency_ms,
                timestamp: turn.timestamp,
            };
            ComputeManager::manage_budget(&mut sigma, cost_entry);

            sigma.turns.push(turn.clone());
            sigma.iteration_index += 1;

            let prev_hash = sigma.state_hash;
            sigma.state_hash = HashChain::compute(&sigma, &prev_hash);

            if sigma.iteration_index <= current_i {
                println!("[verification] Rollback: Non-monotonic index violation.");
                *sigma = sigma_snapshot;
                return Ok(false);
            }

            sigma.agent_weights = InfluenceWeightManager::calculate_weights(&sigma);
            let mut kalman = KalmanConvergence { p_c: sigma.completion_probability, variance: 0.1 };
            let measurement = if response.contains("OPTIMAL") || response.contains("CONVERGED") { 1.0 } else { certainty * 0.8 };
            sigma.completion_probability = kalman.update(measurement);

            self.state_manager.checkpoint(&sigma)?;
            
            self.swarm.broadcast_turn(turn.clone());

            if let Some(ref mut root) = sigma.goal_tree.root {
                PlanningEngine::update_goal_status(root);
            }

            let _ = self.event_tx.send(StreamEvent::TokenReceived(format!("\n[Turn Complete | P(C): {:.2} | Hash: {:02x?}]\n", sigma.completion_probability, &sigma.state_hash[..4]))).await;
            let _ = self.event_tx.send(StreamEvent::TurnComplete(turn)).await;

            let is_converged = sigma.completion_probability > 0.95;
            
            if is_converged {
                let eval = SelfImprovementEngine::evaluate_session(&sigma);
                println!("[self-improve] Session evaluation: {:?}", eval);
                
                let report = AnalyticsEngine::generate_report(&sigma);
                println!("[analytics] Session report: {:?}", report);

                // Release: Final Report
                let exec_summary = ConvergenceReport::generate(&sigma);
                println!("[release] Convergence Report:\n{}", exec_summary);
            }

            Ok(is_converged)
        }
    }

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
                        if inner_line.starts_with("```") { break; }
                        content.push_str(inner_line);
                        content.push('\n');
                    }
                    artifacts.insert(name.to_string(), (lang.to_string(), content.trim_end().to_string()));
                }
            }
        }
        artifacts
    }

    pub fn rewind(&self, index: u32) -> Result<ConversationState> {
        self.state_manager.restore(index)?.context(format!("Failed to rewind to index {index}"))
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
