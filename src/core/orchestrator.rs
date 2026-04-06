use crate::core::agent_trait::PromptAgent;
use crate::core::state::StateManager;
use crate::engines::analytics::AnalyticsEngine;
use crate::engines::collective_intelligence::CollectiveIntelligenceEngine;
use crate::engines::compute::ComputeManager;
use crate::engines::consensus::{
    CertaintyAnalyzer, InfluenceWeightManager, KalmanConvergence, NashSolver,
};
use crate::engines::diff::DiffEngine;
use crate::engines::intelligence::{
    IntelligenceEngine, PromptComposer, QualityScorer, RegressionFeedbackHandler,
};
use crate::types::intelligence::PromptTemplate;
use crate::engines::linter::LinterGuard;
use crate::engines::memory::{MemoryBridge, MemoryStore};
use crate::types::memory::{MemoryRecord, OutcomeRecord};
use crate::engines::planning::PlanningEngine;
use crate::engines::proof::ProofManager;
use crate::engines::quality::{ArtifactMetrics, QualityEngine, RegressionDetector};
use crate::engines::reasoning::{FallacyDetector, ReasoningEngine};
use crate::engines::release::ConvergenceReport;
use crate::engines::sandbox::{SandboxManager, SandboxResult};
use crate::engines::surprise::SurpriseEngine;
use crate::engines::security::{SecretScanner, TurnSigner};
use crate::engines::self_improvement::SelfImprovementEngine;
use crate::engines::simulation::MonteCarloRunner;
use crate::engines::swarm::SwarmController;
use crate::engines::validation::AstValidator;
use crate::engines::verification::{AuditAlert, ContinuousAuditor, HashChain, InvariantChecker, TautologyFilter};
use crate::mcp::bridge::ToolDiscovery;
use crate::mcp::gateway::McpGateway;
use crate::types::artifact::{Artifact, ArtifactDiff, ProofAttachment};
use crate::types::compute::{CostEntry, TokenUsage};
use crate::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crate::types::events::{ControlSignal, StreamEvent};
use crate::ui::visualization::GodView;
use anyhow::{Context, Result};
use futures::StreamExt;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};
use tokio::sync::mpsc;

/// All computed data for one artifact change, assembled outside the sigma lock.
struct PreparedArtifactChange {
    name: String,
    lang: String,
    new_content: String,
    delta: ArtifactDiff,
    new_metrics: ArtifactMetrics,
    node_updates: Vec<(String, String)>,
    proof: ProofAttachment,
    skeleton: String,
}

pub struct Orchestrator {
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
    event_tx: mpsc::Sender<StreamEvent>,
    control_rx: Mutex<mpsc::Receiver<ControlSignal>>,
    #[allow(dead_code)]
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
    pub collective: Mutex<CollectiveIntelligenceEngine>,
    pub viz: Mutex<GodView>,
    pub auditor_tx: Option<mpsc::Sender<ConversationState>>,
    pub audit_rx: Mutex<mpsc::Receiver<AuditAlert>>,
    pub surprise_engine: Mutex<SurpriseEngine>,
    pub template_cache: Arc<RwLock<HashMap<String, PromptTemplate>>>,
    pub session_memory_map: Mutex<HashMap<String, Arc<MemoryStore>>>,
    pub memory_bridge: Mutex<MemoryBridge>,
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

        let (alert_tx, alert_rx) = mpsc::channel::<AuditAlert>(32);
        let auditor_tx = Some(ContinuousAuditor::spawn(alert_tx));

        Self {
            agents,
            state_manager,
            event_tx,
            control_rx: Mutex::new(control_rx),
            sandbox: SandboxManager::new().expect("Failed to init sandbox"),
            mc_runner: MonteCarloRunner::new().expect("Failed to init simulation"),
            mcp_gateway,
            memory_store: MemoryStore::new(
                &std::env::var("XDG_DATA_HOME")
                    .map(|d| format!("{d}/crosstalk"))
                    .unwrap_or_else(|_| {
                        std::env::var("HOME")
                            .map(|h| format!("{h}/.local/share/crosstalk"))
                            .unwrap_or_else(|_| "/tmp/crosstalk-memory".to_string())
                    }),
            ),
            intelligence: Mutex::new(IntelligenceEngine::new()),
            compute: Mutex::new(ComputeManager::new()),
            reasoning: ReasoningEngine,
            self_improve: SelfImprovementEngine,
            swarm: SwarmController::new(),
            planning: PlanningEngine,
            signer: TurnSigner::new(),
            analytics: AnalyticsEngine,
            collective: Mutex::new(CollectiveIntelligenceEngine::new()),
            viz: Mutex::new(GodView::new()),
            auditor_tx,
            audit_rx: Mutex::new(alert_rx),
            surprise_engine: Mutex::new(SurpriseEngine::new()),
            session_memory_map: Mutex::new(HashMap::new()),
            memory_bridge: Mutex::new(MemoryBridge::new()),
            template_cache: Arc::new(RwLock::new({
                let mut m = HashMap::new();
                m.insert(
                    "base".to_string(),
                    PromptTemplate {
                        id: "base".to_string(),
                        version: 1,
                        template_text: "Analyze and improve the codebase collaboratively."
                            .to_string(),
                        task_category: TaskCategory::CodeGeneration,
                        variables: vec!["context".to_string()],
                        performance_history: vec![],
                    },
                );
                m.insert(
                    "corrective".to_string(),
                    PromptTemplate {
                        id: "corrective".to_string(),
                        version: 1,
                        template_text:
                            "Quality regression detected. Prioritize correctness."
                                .to_string(),
                        task_category: TaskCategory::CodeGeneration,
                        variables: vec!["baseline".to_string(), "recent".to_string()],
                        performance_history: vec![],
                    },
                );
                m
            })),
        }
    }

    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        // Phase 0: Recall cross-session memory examples before acquiring the main lock.
        let (pre_session_id, pre_turn_idx) = {
            let s = sigma_lock.lock().await;
            (s.session_id.clone(), s.iteration_index)
        };
        let memory_examples = {
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(pre_session_id.clone());
            bridge
                .recall_relevant(&pre_session_id, &pre_session_id, 5, pre_turn_idx)
                .unwrap_or_default()
        };

        // Phase 1: Brief lock to snapshot state needed for streaming and processing.
        let (iteration_index, prompt, history_contents, agent_id, artifacts_snapshot) = {
            let s = sigma_lock.lock().await;
            if s.iteration_index == 0 && s.turns.is_empty() {
                self.state_manager.checkpoint(&s)?;
            }
            let contents: Vec<String> = s.turns.iter().rev().take(10).map(|t| t.content.clone()).collect();
            let distilled_prompt = self.build_differential_prompt(&s);

            let agent_idx = (s.iteration_index as usize) % self.agents.len();
            let model_id = self.agents[agent_idx].name().to_string();

            let structure =
                ReasoningEngine::select_structure(TaskCategory::CodeGeneration, &model_id);
            let mut final_prompt = distilled_prompt;
            match structure {
                TurnStructure::StepByStep => final_prompt
                    .push_str("\nStructure your response with numbered reasoning steps."),
                TurnStructure::ProsCons => {
                    final_prompt.push_str("\nExplicitly analyze tradeoffs (Pros vs Cons).")
                }
                TurnStructure::CodeFirst => {
                    final_prompt
                        .push_str("\nProvide the code delta (Δα) before any explanation.")
                }
                _ => {}
            }

            if !memory_examples.is_empty() {
                final_prompt.push_str("\n\nSuccessful examples from similar tasks:\n");
                for ex in memory_examples.iter().take(5) {
                    let _ = writeln!(
                        final_prompt,
                        "- [Session {}] Turn {}: {}",
                        ex.session_id, ex.turn_id, ex.metadata_json
                    );
                }
            }

            let artifacts_snapshot = s.artifacts.clone();
            (
                s.iteration_index,
                final_prompt,
                contents,
                model_id,
                artifacts_snapshot,
            )
        };
        // sigma_lock released here.

        // Phase 2: Stream response — no lock held.
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
                    let chunk =
                        chunk_res.map_err(|e| anyhow::anyhow!("Stream error: {e:?}"))?;
                    response.push_str(&chunk);
                    let _ = self.event_tx.send(StreamEvent::TokenReceived(chunk)).await;
                }
                None => break,
            }
        }

        let latency_ms = start_time.elapsed().as_millis() as u64;

        // Phase 3: Stateless checks — no lock held.
        let secrets = SecretScanner::scan(&response);
        if !secrets.is_empty() {
            let _ = self
                .event_tx
                .send(StreamEvent::TokenReceived(
                    "\n[Blocked: Security Violation]\n".to_string(),
                ))
                .await;
            return Ok(false);
        }

        if TautologyFilter::is_tautological(&response, &history_contents) {
            let _ = self
                .event_tx
                .send(StreamEvent::TokenReceived(
                    "\n[Pruned: Tautology]\n".to_string(),
                ))
                .await;
            return Ok(false);
        }

        let fallacies = FallacyDetector::scan(&response);
        if !fallacies.is_empty() {
            let _ = self
                .event_tx
                .send(StreamEvent::TokenReceived(format!(
                    "\n[Warning: {} fallacies detected]\n",
                    fallacies.len()
                )))
                .await;
        }

        // Phase 4: Validate and prepare artifact changes — no lock held.
        // mc_runner.predict() and LinterGuard::check() run freely here.
        let proposed_artifacts = Self::parse_artifacts(&response);
        let Some((changes, turn_outcome)) = self
            .process_proposed_artifacts(proposed_artifacts, &artifacts_snapshot)
            .await?
        else {
            return Ok(false);
        };

        // Phase 5: Commit under lock — only fast, CPU-bound work remains.
        let mut sigma = sigma_lock.lock().await;
        self.commit_turn(
            &mut sigma,
            changes,
            turn_outcome,
            &agent_id,
            &response,
            &prompt,
            latency_ms,
        )
        .await
    }

    /// Validate proposed artifacts and compute all changes without holding the sigma lock.
    /// Returns None if any artifact fails validation (caller should return Ok(false)).
    async fn process_proposed_artifacts(
        &self,
        proposed: HashMap<String, (String, String)>,
        snapshot: &HashMap<String, Artifact>,
    ) -> Result<Option<(Vec<PreparedArtifactChange>, TurnOutcome)>> {
        let all_names: Vec<String> = snapshot.keys().cloned().collect();
        let mut changes = Vec::new();
        let mut turn_outcome = TurnOutcome::Unknown;

        for (name, (lang, new_content)) in proposed {
            if let Err(e) = AstValidator::validate(&new_content, &lang) {
                let _ = self
                    .event_tx
                    .send(StreamEvent::TokenReceived(format!(
                        "[diff] artifact \"{name}\" rejected: AST validation failed: {e}"
                    )))
                    .await;
                return Ok(None);
            }

            let dups = QualityEngine::detect_duplication(&new_content, snapshot);
            if !dups.is_empty() {
                let _ = self
                    .event_tx
                    .send(StreamEvent::TokenReceived(format!(
                        "[quality] duplication detected for \"{name}\": {:?}",
                        dups
                    )))
                    .await;
            }

            let default_artifact = Artifact {
                name: name.clone(),
                language: lang.clone(),
                content: String::new(),
                version: 0,
                history: vec![],
                ast_versions: HashMap::new(),
                proof_attachments: vec![],
                metrics: ArtifactMetrics::default(),
                skeleton: String::new(),
            };
            let current = snapshot.get(&name).unwrap_or(&default_artifact);

            if current.content == new_content {
                continue;
            }

            let delta =
                DiffEngine::generate_delta(&current.content, &new_content, current.version);

            let p_fail = self
                .mc_runner
                .predict(current, &delta, 10)
                .await
                .map(|(mean, _)| mean)
                .unwrap_or(0.5);
            if p_fail > 0.5 {
                return Ok(None);
            }

            let new_metrics = QualityEngine::analyze_artifact(
                &Artifact {
                    content: new_content.clone(),
                    ..current.clone()
                },
                &all_names,
            );
            if RegressionDetector::is_regressive(&current.metrics, &new_metrics) {
                return Ok(None);
            }

            let node_updates: Vec<(String, String)> =
                AstValidator::extract_nodes(&new_content, &lang)
                    .into_iter()
                    .collect();

            let proof = ProofManager::generate_proof(
                &Artifact {
                    name: name.clone(),
                    content: new_content.clone(),
                    language: lang.clone(),
                    version: current.version + 1,
                    history: vec![],
                    ast_versions: HashMap::new(),
                    proof_attachments: vec![],
                    metrics: new_metrics.clone(),
                    skeleton: String::new(),
                },
                vec![
                    "ast_valid".to_string(),
                    "mc_safe".to_string(),
                    "quality_checked".to_string(),
                ],
            );

            if lang.to_lowercase() == "rust" || lang.to_lowercase() == "rs" {
                let sandbox_result = SandboxResult {
                    exit_code: 0,
                    stdout: new_content.clone(),
                    stderr: String::new(),
                };
                match LinterGuard::check(&sandbox_result, "/tmp").await {
                    Ok(report) if !report.passed => return Ok(None),
                    Err(_) => return Ok(None),
                    _ => {}
                }
            }

            let skeleton = AstValidator::generate_skeleton(&new_content, &lang);

            changes.push(PreparedArtifactChange {
                name,
                lang,
                new_content,
                delta,
                new_metrics,
                node_updates,
                proof,
                skeleton,
            });
            turn_outcome = TurnOutcome::Compiled;
        }

        Ok(Some((changes, turn_outcome)))
    }

    /// Apply validated changes to sigma and persist the turn. Called under the sigma lock.
    async fn commit_turn(
        &self,
        sigma: &mut ConversationState,
        changes: Vec<PreparedArtifactChange>,
        turn_outcome: TurnOutcome,
        agent_id: &str,
        response: &str,
        prompt: &str,
        latency_ms: u64,
    ) -> Result<bool> {
        let current_i = sigma.iteration_index;

        // Apply artifact changes and update node consensus.
        let mut turn_diffs = Vec::new();
        for change in changes {
            let artifact = sigma
                .artifacts
                .entry(change.name.clone())
                .or_insert_with(|| Artifact {
                    name: change.name.clone(),
                    language: change.lang.clone(),
                    content: String::new(),
                    version: 0,
                    history: vec![],
                    ast_versions: HashMap::new(),
                    proof_attachments: vec![],
                    metrics: ArtifactMetrics::default(),
                    skeleton: String::new(),
                });

            artifact.history.push(change.delta.clone());
            artifact.content = change.new_content;
            artifact.version += 1;
            artifact.language = change.lang;
            artifact.metrics = change.new_metrics;
            artifact.skeleton = change.skeleton;
            artifact.proof_attachments.push(change.proof);

            for (node_id, content) in &change.node_updates {
                artifact
                    .ast_versions
                    .entry(node_id.clone())
                    .or_default()
                    .push((current_i, content.clone()));
            }

            for (node_id, _) in &change.node_updates {
                if let Some(history) = artifact.ast_versions.get(node_id)
                    && history.len() >= 3
                {
                    let v1 = &history[history.len() - 1].1;
                    let v2 = &history[history.len() - 2].1;
                    let v3 = &history[history.len() - 3].1;
                    if v1 == v3 && v1 != v2 {
                        let matrix = [[(0.8, 0.2), (0.1, 0.1)], [(0.1, 0.1), (0.2, 0.8)]];
                        let _eq = NashSolver::solve_2x2_pure(&matrix);
                    }
                }

                let node_p = sigma.node_consensus.entry(node_id.clone()).or_insert(0.1);
                let mut kalman = KalmanConvergence::new(*node_p);
                *node_p = kalman.update(0.8);
            }

            turn_diffs.push((change.name, change.delta));
        }

        let volatility = 0.1;
        let certainty = CertaintyAnalyzer::compute(response, volatility);

        let surprise = {
            let mut se = self.surprise_engine.lock().await;
            se.record_prediction(agent_id, certainty);
            let s = se.compute_surprise(agent_id, turn_outcome);
            if s > 0.5 {
                let _ = self
                    .event_tx
                    .send(StreamEvent::TokenReceived(format!(
                        "[sandbox] High Surprise detected: {:.2}",
                        s
                    )))
                    .await;
            }
            let current_w = sigma.agent_weights.get(agent_id).copied().unwrap_or(1.0);
            let new_w = se.calibrate_weight(agent_id, current_w);
            sigma.agent_weights.insert(agent_id.to_string(), new_w);
            s
        };

        let mut turn = Turn {
            index: sigma.iteration_index,
            model_id: agent_id.to_string(),
            content: response.to_string(),
            timestamp: ConversationState::now(),
            diffs: turn_diffs,
            certainty: Some(certainty),
            outcome: turn_outcome,
            task_category: Some(TaskCategory::CodeGeneration),
            structure: Some(ReasoningEngine::select_structure(
                TaskCategory::CodeGeneration,
                agent_id,
            )),
            signature: vec![],
            surprise_signal: Some(surprise),
        };

        let serialized = serde_json::to_vec(&turn)?;
        turn.signature = self.signer.sign(&serialized);

        {
            let intell = self.intelligence.lock().await;
            intell.update_profile(&turn, QualityScorer::score(&turn, sigma));

            let recent_turns: Vec<Turn> = sigma
                .turns
                .iter()
                .rev()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if let Some(alert) = intell.detect_regression(agent_id, &recent_turns) {
                let _ = self
                    .event_tx
                    .send(StreamEvent::TokenReceived(format!(
                        "[intelligence] Regression detected for {}: {:.2} -> {:.2}",
                        alert.agent_id, alert.baseline_mean, alert.recent_mean
                    )))
                    .await;
            }
        }

        {
            let mut coll = self.collective.lock().await;
            coll.update_specialization(&turn);
        }

        let cost_entry = CostEntry {
            turn_id: turn.index,
            model_id: agent_id.to_string(),
            usage: TokenUsage {
                input_tokens: prompt.len() as u32 / 4,
                output_tokens: response.len() as u32 / 4,
                total_tokens: (prompt.len() + response.len()) as u32 / 4,
            },
            cost_usd: 0.01,
            latency_ms,
            timestamp: turn.timestamp,
        };
        ComputeManager::manage_budget(sigma, cost_entry);

        sigma.turns.push(turn.clone());
        sigma.iteration_index += 1;

        let prev_hash = sigma.state_hash;
        sigma.state_hash = HashChain::compute(sigma, &prev_hash)?;

        if sigma.iteration_index <= current_i {
            return Ok(false);
        }

        sigma.agent_weights = InfluenceWeightManager::calculate_weights(sigma);
        let mut kalman = KalmanConvergence::new(sigma.completion_probability);
        let measurement = if response.contains("OPTIMAL") || response.contains("CONVERGED") {
            1.0
        } else {
            certainty * 0.8
        };
        sigma.completion_probability = kalman.update(measurement);

        InvariantChecker::check_all(sigma)?;
        self.state_manager.checkpoint(sigma)?;
        let _ = self.swarm.broadcast_turn(turn.clone());

        if let Some(ref auditor_tx) = self.auditor_tx {
            let _ = auditor_tx.send(sigma.clone()).await;
        }

        if let Some(ref mut root) = sigma.goal_tree.root {
            PlanningEngine::update_goal_status(root);
        }

        {
            let mut viz = self.viz.lock().await;
            let _ = viz.render_frame(sigma).await;
        }

        let _ = self
            .event_tx
            .send(StreamEvent::TokenReceived(format!(
                "\n[Turn Complete | P(C): {:.2} | Hash: {:02x?}]\n",
                sigma.completion_probability,
                &sigma.state_hash[..4]
            )))
            .await;
        let _ = self.event_tx.send(StreamEvent::TurnComplete(turn.clone())).await;

        {
            let mut audit_rx = self.audit_rx.lock().await;
            while let Ok(alert) = audit_rx.try_recv() {
                let _ = self
                    .event_tx
                    .send(StreamEvent::TokenReceived(format!(
                        "[audit] Hash mismatch at iteration {}: expected {:02x?}, got {:02x?}",
                        alert.iteration_index,
                        &alert.expected_hash[..4],
                        &alert.actual_hash[..4]
                    )))
                    .await;
                if let Ok(Some(safe_state)) = self
                    .state_manager
                    .restore(alert.iteration_index.saturating_sub(1))
                {
                    *sigma = safe_state;
                }
            }
        }

        // Record the committed turn in the memory bridge for cross-session recall.
        {
            let session_id = sigma.session_id.clone();
            let compiled = matches!(
                turn.outcome,
                TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
            );
            let tests_passed = turn.outcome == TurnOutcome::TestsPassed;
            let convergence_contribution = sigma.completion_probability;
            let content_key = response[..response.len().min(500)].to_string();
            let preview = response[..response.len().min(200)].to_string();
            let outcome_str = format!("{:?}", turn.outcome);
            let metadata = serde_json::json!({
                "content": preview,
                "outcome": outcome_str,
                "agent": agent_id,
            })
            .to_string();
            let record = MemoryRecord {
                turn_id: turn.index,
                session_id: session_id.clone(),
                embedding: vec![],
                content_hash: content_key,
                timestamp: turn.timestamp,
                metadata_json: metadata,
                outcome: Some(OutcomeRecord {
                    compiled,
                    tests_passed,
                    quality_delta: 0.0,
                    was_rolled_back: false,
                    convergence_contribution,
                }),
            };
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(session_id.clone());
            bridge.push_record(&session_id, record);
        }

        let is_converged = sigma.completion_probability > 0.95;
        if is_converged {
            let eval = SelfImprovementEngine::evaluate_session(sigma);
            let report = AnalyticsEngine::generate_report(sigma);
            let exec_summary = ConvergenceReport::generate(sigma);
            let _ = self
                .event_tx
                .send(StreamEvent::TokenReceived(format!(
                    "[self-improve] {:?} | [analytics] {:?} | [release] {}",
                    eval, report, exec_summary
                )))
                .await;
            // Register session in map to mark it as indexed for cross-session search.
            self.session_memory_map
                .lock()
                .await
                .entry(sigma.session_id.clone())
                .or_insert_with(|| {
                    Arc::new(MemoryStore::new(&format!(
                        "/tmp/crosstalk-{}",
                        sigma.session_id
                    )))
                });
        }

        Ok(is_converged)
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
                        let changed_nodes = AstValidator::identify_changed_nodes(
                            "",
                            &artifact.content,
                            &artifact.language,
                        );
                        for id in changed_nodes {
                            active_node_ids.insert(id);
                        }
                    }
                }
            }
            let nodes = AstValidator::extract_nodes(&artifact.content, &artifact.language);
            for id in active_node_ids {
                if let Some(content) = nodes.get(&id) {
                    p.push_str(&format!("Node {}:\n{}\n", id, content));
                }
            }
            if let Some(last_diff) = artifact.history.last() {
                p.push_str("\nMost Recent Δα:\n");
                p.push_str(&last_diff.diff_text);
            }
            p.push('\n');
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
}
