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
use crate::collective_intelligence::CollectiveIntelligenceEngine;
use crate::visualization::GodView;
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
    pub collective: Mutex<CollectiveIntelligenceEngine>,
    pub viz: Mutex<GodView>,
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
            collective: Mutex::new(CollectiveIntelligenceEngine::new()),
            viz: Mutex::new(GodView::new()),
        }
    }

    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        let (iteration_index, prompt, history_contents, agent_id) = {
            let s = sigma_lock.lock().await;
            if s.iteration_index == 0 && s.turns.is_empty() {
                self.state_manager.checkpoint(&s)?;
            }
            let contents: Vec<String> = s.turns.iter().map(|t| t.content.clone()).collect();
            let distilled_prompt = self.build_differential_prompt(&s);
            
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

        let start_time = Instant::now();
        let mut stream = agent_id.clone(); // Placeholder for agent retrieval
        let agent_ptr = &self.agents[(iteration_index as usize) % self.agents.len()];
        
        let mut stream = agent_ptr
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
            let _ = self.event_tx.send(StreamEvent::TokenReceived("\n[Blocked: Security Violation]\n".to_string())).await;
            return Ok(false);
        }

        if TautologyFilter::is_tautological(&response, &history_contents) {
            let _ = self.event_tx.send(StreamEvent::TokenReceived("\n[Pruned: Tautology]\n".to_string())).await;
            return Ok(false);
        }

        let fallacies = FallacyDetector::scan(&response);
        if !fallacies.is_empty() {
            let _ = self.event_tx.send(StreamEvent::TokenReceived(format!("\n[Warning: {} fallacies detected]\n", fallacies.len()))).await;
        }

        let proposed_artifacts = Self::parse_artifacts(&response);
        let mut turn_diffs = vec![];
        let mut turn_outcome = TurnOutcome::Unknown;

        {
            let mut sigma = sigma_lock.lock().await;
            let sigma_snapshot = sigma.clone();
            let current_i = sigma.iteration_index;

            let mut all_valid = true;
            for (name, (lang, new_content)) in proposed_artifacts {
                if let Err(e) = AstValidator::validate(&new_content, &lang) {
                    println!("[diff] artifact \"{name}\" rejected: AST validation failed: {e}");
                    all_valid = false;
                    turn_outcome = TurnOutcome::Rejected;
                    break;
                }

                let dups = QualityEngine::detect_duplication(&new_content, &sigma.artifacts);
                if !dups.is_empty() {
                    println!("[quality] duplication detected for \"{name}\": {:?}", dups);
                }

                let (delta, node_updates) = {
                    let current_artifact = sigma.artifacts.entry(name.clone()).or_insert_with(|| Artifact {
                        name: name.clone(),
                        language: lang.clone(),
                        content: String::new(),
                        version: 0,
                        history: vec![],
                        ast_versions: HashMap::new(),
                        proof_attachments: vec![],
                        metrics: crate::quality::ArtifactMetrics::default(),
                        skeleton: String::new(),
                    });

                    if current_artifact.content != new_content {
                        let delta = DiffEngine::generate_delta(&current_artifact.content, &new_content, current_artifact.version);
                        let p_fail = self.mc_runner.predict(current_artifact, &delta, 5).await;
                        if p_fail > 0.5 {
                            all_valid = false;
                            turn_outcome = TurnOutcome::RolledBack;
                            break;
                        }

                        let new_metrics = QualityEngine::analyze_artifact(&Artifact {
                            content: new_content.clone(),
                            ..current_artifact.clone()
                        });
                        if RegressionDetector::is_regressive(&current_artifact.metrics, &new_metrics) {
                            all_valid = false;
                            turn_outcome = TurnOutcome::Rejected;
                            break;
                        }

                        current_artifact.history.push(delta.clone());
                        current_artifact.content = new_content.clone();
                        current_artifact.version += 1;
                        current_artifact.language = lang.clone();
                        current_artifact.metrics = new_metrics;
                        current_artifact.skeleton = AstValidator::generate_skeleton(&new_content, &lang);

                        let nodes = AstValidator::extract_nodes(&new_content, &lang);
                        let mut updates = vec![];
                        for (node_id, content) in nodes {
                            current_artifact.ast_versions.entry(node_id.clone()).or_default().push((current_i, content.clone()));
                            updates.push((node_id, content));
                        }

                        let proof = ProofManager::generate_proof(current_artifact, vec!["ast_valid".to_string(), "mc_safe".to_string(), "quality_checked".to_string()]);
                        current_artifact.proof_attachments.push(proof);

                        (Some(delta), updates)
                    } else {
                        (None, vec![])
                    }
                };

                if let Some(d) = delta {
                    turn_diffs.push((name.clone(), d));
                    turn_outcome = TurnOutcome::Compiled;
                }

                for (node_id, _content) in node_updates {
                    if let Some(history) = sigma.artifacts.get(&name).and_then(|a| a.ast_versions.get(&node_id)) {
                        if history.len() >= 3 {
                            let v1 = &history[history.len()-1].1;
                            let v2 = &history[history.len()-2].1;
                            let v3 = &history[history.len()-3].1;
                            if v1 == v3 && v1 != v2 {
                                let matrix = [[(0.8, 0.2), (0.1, 0.1)], [(0.1, 0.1), (0.2, 0.8)]];
                                let _eq = crate::consensus::NashSolver::solve_2x2_pure(&matrix);
                            }
                        }
                    }

                    let node_p = sigma.node_consensus.entry(node_id).or_insert(0.1);
                    let mut kalman = KalmanConvergence::new(*node_p);
                    *node_p = kalman.update(0.8);
                }
            }

            if !all_valid {
                println!("[sandbox] Rollback triggered.");
                *sigma = sigma_snapshot;
                return Ok(false);
            }

            let volatility = 0.1;
            let certainty = CertaintyAnalyzer::compute(&response, volatility);

            let mut turn = Turn {
                index: sigma.iteration_index,
                model_id: agent_id.clone(),
                content: response.clone(),
                timestamp: ConversationState::now(),
                diffs: turn_diffs,
                certainty: Some(certainty),
                outcome: turn_outcome,
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

            {
                let mut coll = self.collective.lock().await;
                coll.update_specialization(&turn);
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
                *sigma = sigma_snapshot;
                return Ok(false);
            }

            sigma.agent_weights = InfluenceWeightManager::calculate_weights(&sigma);
            let mut kalman = KalmanConvergence::new(sigma.completion_probability);
            let measurement = if response.contains("OPTIMAL") || response.contains("CONVERGED") { 1.0 } else { certainty * 0.8 };
            sigma.completion_probability = kalman.update(measurement);

            self.state_manager.checkpoint(&sigma)?;
            self.swarm.broadcast_turn(turn.clone());

            if let Some(ref mut root) = sigma.goal_tree.root {
                PlanningEngine::update_goal_status(root);
            }

            {
                let mut viz = self.viz.lock().await;
                let _ = viz.render_frame(&sigma).await;
            }

            let _ = self.event_tx.send(StreamEvent::TokenReceived(format!("\n[Turn Complete | P(C): {:.2} | Hash: {:02x?}]\n", sigma.completion_probability, &sigma.state_hash[..4]))).await;
            let _ = self.event_tx.send(StreamEvent::TurnComplete(turn)).await;

            let is_converged = sigma.completion_probability > 0.95;
            if is_converged {
                let eval = SelfImprovementEngine::evaluate_session(&sigma);
                println!("[self-improve] Session evaluation: {:?}", eval);
                let report = AnalyticsEngine::generate_report(&sigma);
                println!("[analytics] Session report: {:?}", report);
                let exec_summary = ConvergenceReport::generate(&sigma);
                println!("[release] Convergence Report:\n{}", exec_summary);
            }

            Ok(is_converged)
        }
    }

    fn build_differential_prompt(&self, sigma: &ConversationState) -> String {
        let mut p = format!("Project Context: {}\n\n", sigma.session_id);
        p.push_str("Artifacts (Semantic Skeleton + Active Nodes):\n");
        for artifact in sigma.artifacts.values() {
            p.push_str(&format!("--- Artifact: {} ---\n", artifact.name));
            p.push_str("Skeleton:\n");
            p.push_str(&artifact.skeleton);
            p.push_str("\nActive Nodes (Full Content):\n");
            let mut active_node_ids = std::collections::HashSet::new();
            for turn in sigma.turns.iter().rev().take(2) {
                for (name, _diff) in &turn.diffs {
                    if name == &artifact.name {
                        let changed_nodes = AstValidator::identify_changed_nodes("", &artifact.content, &artifact.language);
                        for id in changed_nodes { active_node_ids.insert(id); }
                    }
                }
            }
            let nodes = AstValidator::extract_nodes(&artifact.content, &artifact.language);
            for id in active_node_ids {
                if let Some(content) = nodes.get(&id) { p.push_str(&format!("Node {}:\n{}\n", id, content)); }
            }
            if let Some(last_diff) = artifact.history.last() {
                p.push_str("\nMost Recent Δα:\n");
                p.push_str(&last_diff.diff_text);
            }
            p.push_str("\n");
        }
        p.push_str("\nRecent History (Last 5 turns):\n");
        for t in sigma.turns.iter().rev().take(5).rev() {
            let _ = writeln!(p, "{}: {}", t.model_id, t.content);
        }
        p.push_str("\nRefine artifacts or debate the solution. Use ```lang:filename to propose changes. Tag completion with 'OPTIMAL'.");
        p
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
}
