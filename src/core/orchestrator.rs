use crate::core::agent_trait::PromptAgent;
use crate::core::environment::NixManager;
use crate::core::state::StateManager;
use crate::engines::analytics::AnalyticsEngine;
use crate::engines::collective_intelligence::{CollectiveIntelligenceEngine, EnsembleEngine};
use crate::engines::compute::{ComputeManager, RequestRateLimiter};
use crate::engines::consensus::{
    CertaintyAnalyzer, InfluenceWeightManager, KalmanConvergence,
};
use crate::engines::diff::DiffEngine;
use crate::engines::intelligence::{
    IntelligenceEngine, QualityScorer, RegressionFeedbackHandler,
};
use crate::engines::memory::{MemoryBridge, MemoryStore};
use crate::engines::planning::PlanningEngine;
use crate::engines::quality::{ArtifactMetrics, QualityEngine, RegressionDetector};
use crate::engines::reasoning::{ReasoningEngine, SynthesisEngine};
use crate::engines::release::ConvergenceReport;
use crate::engines::simulation::MonteCarloRunner;
use crate::engines::sandbox::SandboxResult;
use crate::engines::verification::{AuditAlert, ContinuousAuditor, HashChain, InvariantChecker, TautologyFilter};
use crate::engines::security::SecretScanner;
use crate::engines::FallacyDetector;
use crate::engines::linter::LinterGuard;
use crate::engines::self_improvement::{SelfImprovementEngine, FileWriter, WriteOutcome};
use crate::engines::surprise::SurpriseEngine;
use crate::engines::validation::AstValidator;
use crate::mcp::bridge::ToolDiscovery;
use crate::mcp::gateway::McpGateway;
use crate::types::artifact::{Artifact, ArtifactDiff, ProofAttachment};
use crate::types::compute::{CostEntry, TokenUsage};
use crate::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crate::types::events::{ControlSignal, StreamEvent};
use crate::types::intelligence::PromptTemplate;
use crate::types::memory::{MemoryRecord, OutcomeRecord, SessionContext};
use crate::engines::proof::ProofManager;
use crate::engines::swarm::SwarmController;
use crate::engines::security::TurnSigner;
use crate::ui::visualization::GodView;
use anyhow::{Context, Result};
use futures::StreamExt;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::{mpsc, Mutex, RwLock};

const MAX_SESSION_TURNS: usize = 1000;

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
    pub file_writer: FileWriter,
    pub auditor_tx: Option<mpsc::Sender<ConversationState>>,
    pub audit_rx: Mutex<mpsc::UnboundedReceiver<AuditAlert>>,
    pub surprise_engine: Mutex<SurpriseEngine>,
    pub template_cache: Arc<RwLock<BTreeMap<String, PromptTemplate>>>,
    pub session_memory_map: Mutex<BTreeMap<String, Arc<MemoryStore>>>,
    pub memory_bridge: Mutex<MemoryBridge>,
    pub completion_probability: Arc<AtomicU64>,
    session_ctx: Mutex<SessionContext>,
    rollback_counters: Mutex<BTreeMap<String, u32>>,
    skip_until: Mutex<BTreeMap<String, u32>>,
    rate_limiter: Arc<RequestRateLimiter>,
    nix_env: Option<HashMap<String, String>>,
}

fn resolve_memory_store_path() -> Result<String> {
    use std::path::{Component, PathBuf};
    let base: PathBuf = if let Ok(d) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(d).join("crosstalk")
    } else if let Ok(h) = std::env::var("HOME") {
        PathBuf::from(h).join(".local/share/crosstalk")
    } else {
        anyhow::bail!(
            "neither XDG_DATA_HOME nor HOME is set; cannot resolve memory store path"
        );
    };
    if base.components().any(|c| matches!(c, Component::ParentDir)) {
        anyhow::bail!(
            "memory store path contains parent-dir traversal: {}",
            base.display()
        );
    }
    base.into_os_string()
        .into_string()
        .map_err(|p| anyhow::anyhow!("memory store path is not valid UTF-8: {:?}", p))
}

fn is_rate_limited(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    s.contains("429") || s.contains("Too Many Requests") || s.contains("rate_limit") || s.contains("RateLimit")
}

impl Orchestrator {
    pub async fn new(
        state_manager: StateManager,
        agents: Vec<Box<dyn PromptAgent>>,
        event_tx: mpsc::Sender<StreamEvent>,
        control_rx: mpsc::Receiver<ControlSignal>,
    ) -> Result<Self> {
        let mut mcp_gateway = McpGateway::new();
        let tools = tokio::task::spawn_blocking(ToolDiscovery::scan).await.unwrap_or_default();
        for tool in tools {
            mcp_gateway.register_tool(tool);
        }

        let nix_env = tokio::task::spawn_blocking(|| {
            if which::which("nix").is_err() {
                return None;
            }
            let deps: Vec<String> = ["rustc", "cargo", "git", "gcc", "pkg-config"]
                .iter()
                .filter(|d| which::which(d).is_ok())
                .map(|d| d.to_string())
                .collect();
            NixManager::new(deps)
                .ok()
                .and_then(|mgr| mgr.synthesize().ok())
        })
        .await
        .unwrap_or(None);

        mcp_gateway.set_nix_env(nix_env.clone());

        let (alert_tx, alert_rx) = mpsc::unbounded_channel::<AuditAlert>();
        let auditor_tx = Some(ContinuousAuditor::spawn(alert_tx));

        Ok(Self {
            agents,
            state_manager,
            event_tx,
            control_rx: Mutex::new(control_rx),
            mc_runner: MonteCarloRunner::new().context("Failed to init simulation")?,
            mcp_gateway,
            memory_store: MemoryStore::new(&resolve_memory_store_path()?),
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
            file_writer: FileWriter::from_env()?,
            auditor_tx,
            audit_rx: Mutex::new(alert_rx),
            surprise_engine: Mutex::new(SurpriseEngine::new()),
            session_memory_map: Mutex::new(BTreeMap::new()),
            memory_bridge: Mutex::new(MemoryBridge::new()),
            completion_probability: Arc::new(AtomicU64::new(0.0f64.to_bits())),
            session_ctx: Mutex::new(SessionContext::new("pending")),
            rollback_counters: Mutex::new(BTreeMap::new()),
            skip_until: Mutex::new(BTreeMap::new()),
            template_cache: Arc::new(RwLock::new({
                let mut m = BTreeMap::new();
                m.insert(
                    "base".to_string(),
                    PromptTemplate {
                        id: "base".to_string(),
                        version: 1,
                        template_text: "Execute turn in Symbolic Swarm Mode (SSM). Minimize prose. Use Δα for code changes, σ for state, μ for agent consensus. Prefer symbolic logic and mathematical notation for reasoning trajectories.".to_string(),
                        task_category: TaskCategory::Research,
                        variables: vec!["task".to_string()],
                        performance_history: vec![],
                    },
                );
                m.insert(
                    "corrective".to_string(),
                    PromptTemplate {
                        id: "corrective".to_string(),
                        version: 1,
                        template_text: "The previous turn regressed. Please fix: {{task}}".to_string(),
                        task_category: TaskCategory::Research,
                        variables: vec!["task".to_string()],
                        performance_history: vec![],
                    },
                );
                m
            })),
            rate_limiter: Arc::new(RequestRateLimiter::new(
                std::env::var("CROSSTALK_RPM")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60),
            )),
            nix_env,
        })
    }

    pub async fn shutdown(&self) {
        self.swarm.shutdown().await;
    }

    async fn emit(&self, event: StreamEvent) -> Result<()> {
        self.event_tx
            .send(event)
            .await
            .map_err(|_| anyhow::anyhow!("event channel closed"))
    }

    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        let (pre_session_id, pre_turn_idx, pre_recent_turns) = {
            let s = sigma_lock.lock().await;
            let recent: Vec<Turn> = s.turns.iter().rev().take(5).cloned().collect();
            (s.session_id.clone(), s.iteration_index, recent)
        };

        let mut memory_examples = vec![];
        let mut regression_prefix = String::new();

        {
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(pre_session_id.clone());
            memory_examples = bridge
                .recall_relevant(&pre_session_id, "latest turn context", 3, pre_turn_idx)
                .await
                .unwrap_or_default();
        }

        {
            let intell = self.intelligence.lock().await;
            if let Some(alert) = intell.detect_regression("swarm", &pre_recent_turns) {
                regression_prefix = RegressionFeedbackHandler::compose_corrective_prompt(
                    &alert,
                    "",
                    &RegressionFeedbackHandler::counter_examples(
                        &pre_recent_turns,
                        TaskCategory::Research,
                    ),
                );
            }
        }

        let (prompt, history_contents, active_agents, artifacts_snapshot) = {
            let s = sigma_lock.lock().await;
            if s.iteration_index == 0 && s.turns.is_empty() {
                self.state_manager.checkpoint(&s)?;
            }
            let contents: Vec<String> = s.turns.iter().rev().take(10).map(|t| t.content.clone()).collect();
            let distilled_prompt = self.build_differential_prompt(&s);

            let skips = self.skip_until.lock().await;
            let mut active = Vec::new();
            for (idx, agent) in self.agents.iter().enumerate() {
                let until = skips.get(agent.name()).copied().unwrap_or(0);
                if s.iteration_index >= until {
                    active.push((idx, agent.name().to_string()));
                }
            }
            if active.is_empty() {
                active.push((0, self.agents[0].name().to_string()));
            }

            {
                let intell = self.intelligence.lock().await;
                let names: Vec<String> = active.iter().map(|(_, n)| n.clone()).collect();
                if let Ok(best) = intell.route_task_constrained(
                    TaskCategory::Research,
                    &names,
                    u32::MAX,
                    u64::MAX,
                    &[],
                ) {
                    if let Some(pos) = active.iter().position(|(_, n)| *n == best) {
                        active.swap(0, pos);
                    }
                }
            }

            let mut final_prompt = distilled_prompt;
            let structure = ReasoningEngine::select_structure(TaskCategory::Research, &active[0].1);
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
                TurnStructure::Symbolic => {
                    final_prompt.push_str("\nUse SSM (Symbolic Swarm Mode): use ∀, ∃, ⊢, ⊥, Δα, σ, μ. Minimize prose.")
                }
                _ => {}
            }

            if !regression_prefix.is_empty() {
                final_prompt = format!("{regression_prefix}\n{final_prompt}");
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
                final_prompt,
                contents,
                active,
                artifacts_snapshot,
            )
        };

        let start_time = Instant::now();
        let (paused_tx, paused_rx) = tokio::sync::watch::channel(false);

        let mut tasks = Vec::new();
        for (idx, name) in &active_agents {
            let agent = &self.agents[*idx];
            let agent_id = name.clone();
            let prompt = prompt.clone();
            let event_tx = self.event_tx.clone();
            let mut p_rx = paused_rx.clone();
            let rate_limiter = Arc::clone(&self.rate_limiter);

            tasks.push(async move {
                let mut delay_ms = 1_000u64;
                for attempt in 0u32..4 {
                    if attempt > 0 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                        delay_ms = (delay_ms * 2).min(30_000);
                    }

                    rate_limiter.wait_for_permit(&agent_id).await;
                    let mut stream = match agent.stream_prompt(&prompt).await {
                        Ok(s) => s,
                        Err(e) => {
                            let e = anyhow::anyhow!("Agent {agent_id} failure: {e:?}");
                            if is_rate_limited(&e) && attempt < 3 {
                                event_tx.send(StreamEvent::TokenReceived { agent_id: agent_id.clone(), token: format!("\n[{agent_id}] rate-limited, retrying in {}s...\n", delay_ms / 1000) }).await?;
                                continue;
                            }
                            return Err(e);
                        }
                    };

                    let mut response = String::new();
                    let mut hit_rate_limit = false;
                    loop {
                        if *p_rx.borrow() { let _ = p_rx.changed().await; continue; }
                        match stream.next().await {
                            Some(Ok(chunk)) => {
                                response.push_str(&chunk);
                                event_tx.send(StreamEvent::TokenReceived { agent_id: agent_id.clone(), token: chunk }).await?;
                            }
                            Some(Err(e)) => {
                                let e = anyhow::anyhow!("Agent {agent_id} stream error: {e:?}");
                                if is_rate_limited(&e) && attempt < 3 {
                                    hit_rate_limit = true;
                                    event_tx.send(StreamEvent::TokenReceived { agent_id: agent_id.clone(), token: format!("\n[{agent_id}] rate-limited mid-stream, retrying in {}s...\n", delay_ms / 1000) }).await?;
                                } else {
                                    return Err(e);
                                }
                                break;
                            }
                            None => break,
                        }
                    }
                    if !hit_rate_limit {
                        return Ok((agent_id, response));
                    }
                }
                Err(anyhow::anyhow!("Agent {agent_id} exhausted rate-limit retries"))
            });
        }

        let mut results_fut = futures::future::join_all(tasks);
        let mut final_results = Vec::new();
        let control_rx = &self.control_rx;
        let mut ctrl_open = true;

        loop {
            tokio::select! {
                res = &mut results_fut => {
                    for r in res {
                        match r {
                            Ok(val) => final_results.push(val),
                            Err(e) => {
                                self.emit(StreamEvent::Error(format!("Agent dropped: {}", e))).await?;
                            }
                        }
                    }
                    if final_results.is_empty() {
                        return Err(anyhow::anyhow!("All agents in swarm failed to respond."));
                    }
                    break;
                }
                signal = async { control_rx.lock().await.recv().await }, if ctrl_open => {
                    match signal {
                        Some(ControlSignal::Pause) => { let _ = paused_tx.send(true); },
                        Some(ControlSignal::Resume) => { let _ = paused_tx.send(false); },
                        Some(ControlSignal::Shutdown) => return Ok(false),
                        Some(ControlSignal::Inject(text)) => {
                            self.emit(StreamEvent::TokenReceived { agent_id: "User".to_string(), token: text }).await?;
                        }
                        Some(ControlSignal::Rewind(index)) => {
                            if let Ok(Some(restored)) = self.state_manager.restore_async(index).await {
                                let mut s = sigma_lock.lock().await;
                                *s = restored;
                                self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("\n[Rewound to iteration {}]\n", index) }).await?;
                                return Ok(true);
                            }
                        }
                        None => { ctrl_open = false; },
                    }
                }
            }
        }

        let weights = InfluenceWeightManager::calculate_weights_with_recency(&*sigma_lock.lock().await, 0.9);

        // Outlier detection: compute pairwise word-overlap similarity
        let mut outlier_penalty: HashMap<&str, f64> = HashMap::new();
        if final_results.len() >= 2 {
            let word_sets: Vec<(&str, std::collections::HashSet<&str>)> = final_results
                .iter()
                .map(|(id, text)| (id.as_str(), text.split_whitespace().collect()))
                .collect();
            for (i, (id, set_i)) in word_sets.iter().enumerate() {
                let mut sim_sum = 0.0;
                let mut count = 0;
                for (j, (_, set_j)) in word_sets.iter().enumerate() {
                    if i == j { continue; }
                    let intersection = set_i.intersection(set_j).count() as f64;
                    let union = set_i.union(set_j).count().max(1) as f64;
                    sim_sum += intersection / union;
                    count += 1;
                }
                let mean_sim = if count > 0 { sim_sum / count as f64 } else { 1.0 };
                if mean_sim < 0.3 {
                    outlier_penalty.insert(id, 0.1);
                }
            }
        }

        // 1. Collective Text Synthesis (surprise + certainty + outlier calibrated)
        let se = self.surprise_engine.lock().await;
        let text_proposals: Vec<(String, String, f64)> = final_results.iter()
            .map(|(id, text)| {
                let base_w = weights.get(id).copied().unwrap_or(1.0);
                let surprise_w = se.calibrate_weight(id, base_w);
                let certainty = CertaintyAnalyzer::compute(text, 0.1);
                let outlier_w = outlier_penalty.get(id.as_str()).copied().unwrap_or(1.0);
                (id.clone(), text.clone(), surprise_w * certainty * outlier_w)
            })
            .collect();
        drop(se);
        let synthesized_text = EnsembleEngine::merge_proposals(text_proposals);

        // 2. Collective Artifact Synthesis
        let mut artifact_proposals: BTreeMap<String, (String, Vec<ArtifactDiff>)> = BTreeMap::new();
        for (_id, text) in &final_results {
            let parsed = Self::parse_artifacts(text);
            for (name, (lang, content)) in parsed {
                let default_art = Arc::new(Artifact::default());
                let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                let diff = DiffEngine::generate_delta(&current.content, &content, current.version);
                
                let entry = artifact_proposals.entry(name).or_insert((lang, vec![]));
                entry.1.push(diff);
            }
        }

        let mut synthesized_artifacts = String::new();
        for (name, (lang, diffs)) in artifact_proposals {
            let default_art = Arc::new(Artifact::default());
            let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
            if let Some(merged_content) = SynthesisEngine::merge(&current.content, diffs, &lang) {
                let _ = writeln!(synthesized_artifacts, "\n```{}:{}\n{}\n```", lang, name, merged_content);
            }
        }

        let response = format!("{}\n{}", synthesized_text, synthesized_artifacts);
        let winner_id = "Collective Swarm".to_string();

        let latency_ms = start_time.elapsed().as_millis() as u64;

        let secrets = SecretScanner::scan(&response);
        if !secrets.is_empty() {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Blocked: Security Violation]\n".to_string(),
            })
            .await?;
            return Ok(false);
        }

        if TautologyFilter::is_tautological(&response, &history_contents) {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Pruned: Tautology]\n".to_string(),
            })
            .await?;
            return Ok(false);
        }

        let fallacies = FallacyDetector::scan(&response);
        if !fallacies.is_empty() {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("\n[Warning: {} fallacies detected]\n", fallacies.len()),
            })
            .await?;
        }

        // Phase 4: Validate and prepare artifact changes — no lock held.
        let mut retry_count = 0;
        let mut current_response = response;
        let mut final_prepared = None;

        while retry_count < 3 {
            let proposed_artifacts = Self::parse_artifacts(&current_response);
            match self.process_proposed_artifacts(proposed_artifacts, &artifacts_snapshot).await? {
                Some((changes, turn_outcome)) => {
                    final_prepared = Some((changes, turn_outcome, current_response));
                    break;
                }
                None => {
                    retry_count += 1;
                    self.emit(StreamEvent::TokenReceived { 
                        agent_id: "System".to_string(), 
                        token: format!("\n[Self-Healing] Synthesis failed validation. Attempting hot-patch cycle {}/3...\n", retry_count) 
                    }).await?;

                    // Re-dispatch to swarm with error context
                    let corrective_prompt = format!(
                        "{}\n\n[CRITICAL: Validation Failed]\nThe previous collective synthesis failed quality/safety gates. Re-implement the code blocks ensuring strict adherence to Rust safety and project invariants.",
                        prompt
                    );
                    
                    let mut tasks = Vec::new();
                    for (idx, name) in &active_agents {
                        let agent = &self.agents[*idx];
                        let agent_id = name.clone();
                        let p = corrective_prompt.clone();
                        let event_tx = self.event_tx.clone();
                        let mut p_rx = paused_rx.clone();
                        let rate_limiter = Arc::clone(&self.rate_limiter);

                        tasks.push(async move {
                            rate_limiter.wait_for_permit(&agent_id).await;
                            let mut stream = agent.stream_prompt(&p).await.map_err(|e| anyhow::anyhow!("Agent {agent_id} failure: {e:?}"))?;
                            let mut resp = String::new();
                            loop {
                                if *p_rx.borrow() { let _ = p_rx.changed().await; continue; }
                                match stream.next().await {
                                    Some(Ok(chunk)) => {
                                        resp.push_str(&chunk);
                                        event_tx.send(StreamEvent::TokenReceived { agent_id: agent_id.clone(), token: chunk }).await?;
                                    }
                                    _ => break,
                                }
                            }
                            Ok::<(String, String), anyhow::Error>((agent_id, resp))
                        });
                    }
                    let results = futures::future::join_all(tasks).await;
                    let mut new_proposals = Vec::new();
                    for res in results.into_iter().flatten() { new_proposals.push(res); }
                    
                    if new_proposals.is_empty() { break; }

                    let text_proposals: Vec<(String, String, f64)> = new_proposals.iter()
                        .map(|(id, text)| (id.clone(), text.clone(), weights.get(id).copied().unwrap_or(1.0)))
                        .collect();
                    let syn_text = EnsembleEngine::merge_proposals(text_proposals);
                    
                    let mut art_proposals: BTreeMap<String, (String, Vec<ArtifactDiff>)> = BTreeMap::new();
                    for (_id, text) in &new_proposals {
                        for (name, (lang, content)) in Self::parse_artifacts(text) {
                            let default_art = Arc::new(Artifact::default());
                            let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                            let entry = art_proposals.entry(name).or_insert((lang, vec![]));
                            entry.1.push(DiffEngine::generate_delta(&current.content, &content, current.version));
                        }
                    }
                    let mut syn_arts = String::new();
                    for (name, (lang, diffs)) in art_proposals {
                        let default_art = Arc::new(Artifact::default());
                        let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                        if let Some(merged) = SynthesisEngine::merge(&current.content, diffs, &lang) {
                            let _ = writeln!(syn_arts, "\n```{}:{}\n{}\n```", lang, name, merged);
                        }
                    }
                    current_response = format!("{}\n{}", syn_text, syn_arts);
                }
            }
        }

        let Some((changes, turn_outcome, final_response)) = final_prepared else {
            self.emit(StreamEvent::TokenReceived { 
                agent_id: "System".to_string(), 
                token: "\n[Self-Healing Failed] Could not converge on a valid synthesis after 3 attempts. Aborting turn.\n".to_string() 
            }).await?;
            return Ok(false);
        };

        // Phase 5: Commit under lock.
        let mut sigma = sigma_lock.lock().await;
        
        {
            let intell = self.intelligence.lock().await;
            for (id, text) in &final_results {
                let p_turn = Turn {
                    index: sigma.iteration_index,
                    model_id: id.clone(),
                    content: text.clone(),
                    timestamp: ConversationState::now(),
                    diffs: vec![],
                    certainty: Some(CertaintyAnalyzer::compute(text, 0.1)),
                    outcome: if id == &winner_id { turn_outcome } else { TurnOutcome::Unknown },
                    task_category: Some(TaskCategory::Research),
                    structure: Some(TurnStructure::FreeForm),
                    signature: vec![],
                    surprise_signal: None,
                };
                intell.update_profile_with_latency(&p_turn, 0.7, latency_ms);
            }
        }

        self.commit_turn(
            &mut sigma,
            changes,
            turn_outcome,
            &winner_id,
            &final_response,
            &prompt,
            latency_ms,
        )
        .await
    }

    async fn process_proposed_artifacts(
        &self,
        proposed: HashMap<String, (String, String)>,
        snapshot: &BTreeMap<String, Arc<Artifact>>,
    ) -> Result<Option<(Vec<PreparedArtifactChange>, TurnOutcome)>> {
        let all_names: Vec<String> = snapshot.keys().cloned().collect();
        let mut changes = Vec::new();
        let mut turn_outcome = TurnOutcome::Unknown;

        for (name, (lang, new_content)) in proposed {
            if let Err(e) = AstValidator::validate(&new_content, &lang) {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[diff] artifact \"{name}\" rejected: AST validation failed: {e}"),
                })
                .await?;
                return Ok(None);
            }

            let dups = QualityEngine::detect_duplication(&new_content, snapshot);
            if !dups.is_empty() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[quality] duplication detected for \"{name}\": {:?}", dups),
                })
                .await?;
            }

            let default_artifact = Arc::new(Artifact {
                name: name.clone(),
                language: lang.clone(),
                content: String::new(),
                version: 0,
                history: vec![],
                ast_versions: BTreeMap::new(),
                proof_attachments: vec![],
                metrics: ArtifactMetrics::default(),
                skeleton: String::new(),
            });
            let current = snapshot.get(&name).unwrap_or(&default_artifact);

            if current.content == new_content {
                continue;
            }

            let delta =
                DiffEngine::generate_delta(&current.content, &new_content, current.version);

            let p_fail: f64 = self
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
                    ..(**current).clone()
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
                    ast_versions: BTreeMap::new(),
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
                let tmp = std::env::temp_dir();
                match LinterGuard::check(&sandbox_result, tmp.to_str().unwrap_or("/tmp"), self.nix_env.as_ref()).await {
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

    #[allow(clippy::too_many_arguments)]
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
        let artifact_snapshot = sigma.artifacts.clone();

        let mut turn_diffs = Vec::new();
        for change in changes {
            let artifact_arc = sigma
                .artifacts
                .entry(change.name.clone())
                .or_insert_with(|| Arc::new(Artifact {
                    name: change.name.clone(),
                    language: change.lang.clone(),
                    content: String::new(),
                    version: 0,
                    history: vec![],
                    ast_versions: BTreeMap::new(),
                    proof_attachments: vec![],
                    metrics: ArtifactMetrics::default(),
                    skeleton: String::new(),
                }));

            let mut artifact = (**artifact_arc).clone();

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

            *artifact_arc = Arc::new(artifact);

            for (node_id, _) in &change.node_updates {
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
            let current_w = sigma.agent_weights.get(agent_id).copied().unwrap_or(1.0);
            let new_w = se.calibrate_weight(agent_id, current_w);
            sigma.agent_weights.insert(agent_id.to_string(), new_w);
            s
        };
        if surprise > 0.5 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[sandbox] High Surprise detected: {:.2}", surprise),
            })
            .await?;
        }

        let mut turn = Turn {
            index: sigma.iteration_index,
            model_id: agent_id.to_string(),
            content: response.to_string(),
            timestamp: ConversationState::now(),
            diffs: turn_diffs,
            certainty: Some(certainty),
            outcome: turn_outcome,
            task_category: Some(TaskCategory::Research),
            structure: Some(ReasoningEngine::select_structure(
                TaskCategory::Research,
                agent_id,
            )),
            signature: vec![],
            surprise_signal: Some(surprise),
        };

        let serialized = serde_json::to_vec(&turn)?;
        turn.signature = self.signer.sign(&serialized);

        let quality_score = {
            let base = QualityScorer::score(&turn);
            let surprise_penalty = (surprise - 0.5).max(0.0) * 0.6;
            let artifact_health = if !sigma.artifacts.is_empty() {
                sigma.artifacts.values().map(|a| a.metrics.health_score).sum::<f64>()
                    / sigma.artifacts.len() as f64
            } else {
                1.0
            };
            ((base - surprise_penalty) * artifact_health).max(0.0)
        };
        {
            let alert_info = {
                let intell = self.intelligence.lock().await;
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
                intell.detect_regression(agent_id, &recent_turns)
            };
            if let Some(alert) = alert_info {
                let severity = if alert.baseline_mean > 0.0 {
                    (alert.baseline_mean - alert.recent_mean) / alert.baseline_mean
                } else {
                    0.0
                };
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[intelligence] Regression detected for {}: {:.2} -> {:.2} (severity {:.0}%)",
                        alert.agent_id, alert.baseline_mean, alert.recent_mean, severity * 100.0
                    ),
                })
                .await?;
                if severity > 0.3 {
                    let mut skips = self.skip_until.lock().await;
                    skips.insert(
                        alert.agent_id.clone(),
                        sigma.iteration_index + 2,
                    );
                }
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

        if sigma.turns.len() >= MAX_SESSION_TURNS {
            return Err(anyhow::anyhow!("Session turn limit ({}) exceeded", MAX_SESSION_TURNS));
        }

        sigma.turns.push(turn.clone());
        sigma.iteration_index += 1;

        let prev_hash = sigma.state_hash;
        sigma.state_hash = HashChain::compute(sigma, &prev_hash)?;

        sigma.agent_weights = InfluenceWeightManager::calculate_weights(sigma).into_iter().collect();
        
        let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
        let mut kalman = KalmanConvergence::new(current_p);
        let measurement = if response.contains("OPTIMAL") || response.contains("CONVERGED") {
            1.0
        } else {
            certainty * 0.8
        };
        let next_p = kalman.update(measurement);
        self.completion_probability.store(next_p.to_bits(), Ordering::Release);
        sigma.completion_probability = next_p;
        self.emit(StreamEvent::ConvergenceUpdated { p: next_p, certainty }).await?;

        if let Err(e) = InvariantChecker::check_all(sigma) {
            sigma.artifacts = artifact_snapshot;
            if let Some(t) = sigma.turns.last_mut() {
                t.outcome = TurnOutcome::RolledBack;
            }
            let should_skip = {
                let mut counters = self.rollback_counters.lock().await;
                let count = counters.entry(agent_id.to_string()).or_insert(0);
                *count += 1;
                let exceeded = *count >= 3;
                if exceeded {
                    *count = 0;
                }
                exceeded
            };
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[rollback] Invariant violation: {e}"),
            })
            .await?;
            if should_skip {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[rollback] Agent {agent_id} exceeded consecutive rollbacks"),
                })
                .await?;
                let mut skips = self.skip_until.lock().await;
                skips.insert(agent_id.to_string(), sigma.iteration_index + 1);
            }
            turn.outcome = TurnOutcome::RolledBack;
            {
                let intell = self.intelligence.lock().await;
                intell.update_profile_with_latency(&turn, 0.0, latency_ms);
            }
            {
                let mut ctx = self.session_ctx.lock().await;
                ctx.record_turn(TurnOutcome::RolledBack);
            }
            return Ok(false);
        }
        {
            let mut counters = self.rollback_counters.lock().await;
            counters.insert(agent_id.to_string(), 0);
        }
        self.state_manager.checkpoint_async(sigma).await?;
        self.emit(StreamEvent::CheckpointWritten(current_i)).await?;

        for (name, _) in &turn.diffs {
            if let Some(artifact) = sigma.artifacts.get(name) {
                match self.file_writer.write_artifact(&artifact.name, &artifact.content).await {
                    Ok(WriteOutcome::Written(path)) => {
                        self.emit(StreamEvent::TokenReceived {
                            agent_id: "System".to_string(),
                            token: format!("[write] {}\n", path.display()),
                        }).await?;
                    }
                    Ok(WriteOutcome::Skipped(_)) => {}
                    Ok(WriteOutcome::VerificationFailed(stderr)) => {
                        self.emit(StreamEvent::TokenReceived {
                            agent_id: "System".to_string(),
                            token: format!("[write] {name}: verification failed, original restored\n{stderr}"),
                        }).await?;
                    }
                    Err(e) => {
                        self.emit(StreamEvent::TokenReceived {
                            agent_id: "System".to_string(),
                            token: format!("[write] error writing {name}: {e}\n"),
                        }).await?;
                    }
                }
            }
        }

        {
            let intell = self.intelligence.lock().await;
            intell.update_profile_with_latency(&turn, quality_score, latency_ms);
        }
        {
            let mut ctx = self.session_ctx.lock().await;
            ctx.record_turn(turn.outcome);
        }
        self.swarm.broadcast_turn(turn.clone())?;

        if let Some(ref auditor_tx) = self.auditor_tx
            && !auditor_tx.is_closed()
                && let Err(e) = auditor_tx.send(sigma.clone()).await {
                    self.emit(StreamEvent::Error(format!(
                        "auditor send failed, auditor task dead: {e}"
                    )))
                    .await?;
                }

        if let Some(ref mut root) = sigma.goal_tree.root {
            PlanningEngine::update_goal_status(root);
        }

        {
            let mut viz = self.viz.lock().await;
            let _ = viz.render_frame(sigma).await;
        }

        let artifact_snapshot: Vec<(String, String)> = sigma
            .artifacts
            .iter()
            .map(|(name, a)| (name.clone(), a.skeleton.clone()))
            .collect();
        self.emit(StreamEvent::ArtifactsUpdated(artifact_snapshot)).await?;

        self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "\n[Turn Complete | P(C): {:.2} | Hash: {:02x?}]\n",
                    next_p, &sigma.state_hash[..4]
                ),
            })
            .await?;
        self.emit(StreamEvent::TurnComplete(turn.clone())).await?;

        {
            let mut audit_rx: MutexGuard<'_, mpsc::UnboundedReceiver<AuditAlert>> = self.audit_rx.lock().await;
            while let Ok(alert) = audit_rx.try_recv() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[audit] Hash mismatch at iteration {}: expected {:02x?}, got {:02x?}",
                        alert.iteration_index,
                        &alert.expected_hash[..4],
                        &alert.actual_hash[..4]
                    ),
                }).await?;
                if let Ok(Some(safe_state)) = self
                    .state_manager
                    .restore(alert.iteration_index.saturating_sub(1))
                {
                    *sigma = safe_state;
                }
            }
        }

        {
            let session_id = sigma.session_id.clone();
            let compiled = matches!(
                turn.outcome,
                TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
            );
            let tests_passed = turn.outcome == TurnOutcome::TestsPassed;
            let convergence_contribution = next_p;
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

        if next_p > 0.70 && next_p <= 0.85 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[convergence] p={next_p:.2}, moderate confidence, continuing refinement"),
            }).await?;
        } else if next_p > 0.85 && next_p <= 0.95 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[convergence] p={next_p:.2}, high confidence, final polish"),
            }).await?;
        }

        let is_converged = next_p > 0.95;
        if is_converged {
            let eval = SelfImprovementEngine::evaluate_session(sigma);
            let report = AnalyticsEngine::generate_report(sigma);
            let exec_summary = ConvergenceReport::generate(sigma);
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[self-improve] {:?} | [analytics] {:?} | [release] {}",
                    eval, report, exec_summary
                ),
            }).await?;
            self.session_memory_map
                .lock()
                .await
                .insert(sigma.session_id.clone(), Arc::new(MemoryStore::new(&format!(
                        "/tmp/crosstalk-{}",
                        sigma.session_id
                    ))));
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
        p.push_str("\nRecent History (compressed):\n");
        for t in sigma.turns.iter().rev().take(5).rev() {
            let signals = ReasoningEngine::extract_signals(&t.content);
            let outcome_tag = format!("{:?}", t.outcome);
            let decisions = if signals.decisions.is_empty() {
                String::new()
            } else {
                format!(" decisions=[{}]", signals.decisions.join("; "))
            };
            let problems = if signals.problems.is_empty() {
                String::new()
            } else {
                format!(" problems=[{}]", signals.problems.join("; "))
            };
            let code_count = signals.code_blocks.len();
            let code_tag = if code_count > 0 {
                format!(" code_blocks={code_count}")
            } else {
                String::new()
            };
            let _ = writeln!(
                p,
                "Turn {} by {} ({}){}{}{}: {}",
                t.index,
                t.model_id,
                outcome_tag,
                decisions,
                problems,
                code_tag,
                &t.content[..t.content.len().min(150)],
            );
        }
        p.push_str("\nRefine artifacts or debate the solution. Use ```lang:filename to propose changes. Tag completion with 'OPTIMAL'.");
        p
    }

    fn lang_to_ext(lang: &str) -> &'static str {
        match lang.to_lowercase().as_str() {
            "rust" | "rs" => "rs",
            "python" | "py" => "py",
            "javascript" | "js" => "js",
            "typescript" | "ts" => "ts",
            "go" => "go",
            "java" => "java",
            "c" => "c",
            "cpp" | "c++" => "cpp",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            "toml" => "toml",
            "markdown" | "md" => "md",
            "html" => "html",
            "css" => "css",
            "sql" => "sql",
            "bash" | "sh" => "sh",
            _ => "txt",
        }
    }

    fn ext_to_lang(ext: &str) -> &'static str {
        match ext.to_lowercase().as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" => "javascript",
            "ts" => "typescript",
            "go" => "go",
            "java" => "java",
            "c" => "c",
            "cpp" | "cc" | "cxx" => "cpp",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            "toml" => "toml",
            "md" => "markdown",
            "html" | "htm" => "html",
            "css" => "css",
            "sql" => "sql",
            "sh" | "bash" => "bash",
            _ => "",
        }
    }

    /// Resolve (lang, name) from a fence annotation string (everything after the backticks).
    fn parse_fence_hint(hint: &str) -> (String, String) {
        // Strip outer debug-format quotes: "rust" → rust
        let hint = hint.trim().trim_matches('"');

        // Colon separator: lang:name or "lang":name
        if let Some(pos) = hint.find(':') {
            let l = hint[..pos].trim().trim_matches('"').to_string();
            let n = hint[pos + 1..].trim().to_string();
            if !l.is_empty() {
                return (l, n);
            }
        }

        // Looks like a bare filename with extension (e.g. `main.rs`, `src/lib.rs`)
        if hint.contains('.') && !hint.contains(' ') {
            let ext = hint.rsplit('.').next().unwrap_or("");
            let lang = Self::ext_to_lang(ext).to_string();
            return (lang, hint.to_string());
        }

        // Space separator: lang name
        if let Some(pos) = hint.find(' ') {
            let l = hint[..pos].trim().to_string();
            let n = hint[pos + 1..].trim().to_string();
            if !l.is_empty() && !n.is_empty() {
                return (l, n);
            }
        }

        // Bare lang token — name resolved later
        (hint.to_string(), String::new())
    }

    /// Check whether a line immediately preceding a fence looks like a filename hint.
    /// Matches: `### foo.rs`, `**foo.rs**`, `File: foo.rs`, `> foo.rs`
    fn extract_pre_fence_name(line: &str) -> Option<&str> {
        let s = line
            .trim()
            .trim_start_matches('#')
            .trim_start_matches('>')
            .trim_start_matches('*')
            .trim_end_matches('*')
            .trim_end_matches(':')
            .trim();
        let s = s
            .strip_prefix("File:")
            .or_else(|| s.strip_prefix("file:"))
            .or_else(|| s.strip_prefix("Filename:"))
            .or_else(|| s.strip_prefix("filename:"))
            .map(str::trim)
            .unwrap_or(s);
        // Accept only if it looks like a filename: has extension, no whitespace
        if s.contains('.') && !s.contains(' ') && s.len() < 128 {
            Some(s)
        } else {
            None
        }
    }

    /// Try to pull a filename out of a first-line comment inside a code block.
    /// Handles: `// filename: foo.rs`, `# foo.py`, `-- name: query.sql`, `/* foo.c */`
    fn extract_comment_filename(line: &str) -> Option<String> {
        let t = line.trim();
        let rest = if let Some(r) = t.strip_prefix("//") {
            r
        } else if let Some(r) = t.strip_prefix('#') {
            r
        } else if let Some(r) = t.strip_prefix("--") {
            r
        } else if let Some(r) = t.strip_prefix("/*").and_then(|r| r.strip_suffix("*/")) {
            r
        } else {
            return None;
        };
        let rest = rest.trim();
        let candidate = rest
            .strip_prefix("filename:")
            .or_else(|| rest.strip_prefix("file:"))
            .or_else(|| rest.strip_prefix("path:"))
            .or_else(|| rest.strip_prefix("name:"))
            .map(str::trim)
            .unwrap_or(rest);
        if candidate.contains('.') && !candidate.contains(' ') && !candidate.is_empty() && candidate.len() < 200 {
            Some(candidate.trim_start_matches("./").trim_start_matches('/').to_string())
        } else {
            None
        }
    }

    fn parse_artifacts(response: &str) -> HashMap<String, (String, String)> {
        let mut artifacts = HashMap::new();
        let all_lines: Vec<&str> = response.lines().collect();
        let mut i = 0usize;
        let mut unnamed_count = 0usize;

        while i < all_lines.len() {
            let trimmed = all_lines[i].trim();

            if !trimmed.starts_with("```") && !trimmed.starts_with("Δα:") {
                i += 1;
                continue;
            }

            // Resolve (lang, name) from the fence line itself
            let (mut lang, mut name) = if trimmed.starts_with("Δα:") {
                let parts: Vec<&str> = trimmed.splitn(3, ':').collect();
                if parts.len() < 2 { i += 1; continue; }
                let l = parts[1].trim().to_string();
                let n = if parts.len() >= 3 { parts[2].trim().to_string() } else { String::new() };
                (l, n)
            } else {
                let rest = trimmed.trim_start_matches('`').trim();
                if rest.is_empty() { i += 1; continue; }
                Self::parse_fence_hint(rest)
            };

            // Pre-fence hint: check the nearest non-empty line above for a filename
            if name.is_empty() {
                let hint = all_lines[..i].iter().rev()
                    .find(|l| !l.trim().is_empty())
                    .and_then(|l| Self::extract_pre_fence_name(l));
                if let Some(h) = hint {
                    name = h.to_string();
                }
            }

            // Collect content lines until closing fence
            i += 1;
            let content_start = i;
            while i < all_lines.len() && !all_lines[i].trim().starts_with("```") {
                i += 1;
            }
            let mut content_lines: Vec<&str> = all_lines[content_start..i].to_vec();
            if i < all_lines.len() { i += 1; } // consume closing fence

            if content_lines.is_empty() { continue; }

            // First-line comment may carry the filename
            if name.is_empty()
                && let Some(fname) = Self::extract_comment_filename(content_lines[0]) {
                    name = fname;
                    content_lines.remove(0);
                    // drop optional blank separator line
                    if content_lines.first().map(|l| l.trim().is_empty()).unwrap_or(false) {
                        content_lines.remove(0);
                    }
                }

            // Infer lang from filename extension if still unknown
            if (lang.is_empty() || lang == "text" || lang == "plaintext") && !name.is_empty()
                && let Some(ext) = name.rsplit('.').next() {
                    let inferred = Self::ext_to_lang(ext);
                    if !inferred.is_empty() { lang = inferred.to_string(); }
                }

            if lang.is_empty() { continue; }

            // Synthesize a name if none found
            if name.is_empty() {
                unnamed_count += 1;
                name = format!("artifact_{}.{}", unnamed_count, Self::lang_to_ext(&lang));
            }

            // Normalize path separators
            let name = name.trim_start_matches("./").trim_start_matches('/').to_string();

            let content = content_lines.join("\n").trim_end().to_string();
            if !content.is_empty() {
                // Last write wins — later refinements override earlier ones
                artifacts.insert(name, (lang, content));
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

    pub fn get_completion_probability(&self) -> f64 {
        f64::from_bits(self.completion_probability.load(Ordering::Acquire))
    }
}

use tokio::sync::MutexGuard;
