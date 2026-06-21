use crate::core::agent_trait::PromptAgent;
use crate::core::environment::NixManager;
use crate::core::environment::ToolDiscovery;
use crate::core::state::StateManager;
use crate::engines::FallacyDetector;
use crate::engines::analytics::AnalyticsEngine;
use crate::engines::collective_intelligence::{
    AdaptiveSelection, CollectiveIntelligenceEngine, EnsembleEngine, MetaStrategy,
};
use crate::engines::compute::{ComputeManager, RequestRateLimiter};
use crate::engines::consensus::{CertaintyAnalyzer, KalmanConvergence, StallDetector};
use crate::engines::diff::DiffEngine;
use crate::engines::intelligence::{
    ConsistencyScorer, IntelligenceEngine, QualityScorer, RegressionFeedbackHandler, RewardVector,
};
use crate::engines::linter::LinterGuard;
use crate::engines::memory::{MemoryBridge, MemoryStore};
use crate::engines::metacognition::MetacognitiveObserver;
use crate::engines::planning::PlanningEngine;
use crate::engines::prompt_evolution::PromptEvolver;
use crate::engines::proof::ProofManager;
use crate::engines::quality::{ArtifactMetrics, QualityEngine, RegressionDetector};
use crate::engines::reasoning::{ReasoningEngine, SynthesisEngine};
use crate::engines::release::ConvergenceReport;
use crate::engines::sandbox::SandboxResult;
use crate::engines::security::SecretScanner;
use crate::engines::security::TurnSigner;
use crate::engines::self_improvement::{
    FileWriter, PostMortemGenerator, SelfImprovementEngine, WriteOutcome,
};
use crate::engines::simulation::MonteCarloRunner;
use crate::engines::surprise::SurpriseEngine;
use crate::engines::swarm::SwarmController;
use crate::engines::topology::TopologyManager;
use crate::engines::validation::AstValidator;
use crate::engines::verification::DecisionLedger;
use crate::engines::verification::{
    AuditAlert, ContinuousAuditor, HashChain, InvariantChecker, TautologyFilter,
};
use crate::mcp::gateway::McpGateway;
use crate::types::artifact::{Artifact, ArtifactDiff, ProofAttachment};
use crate::types::compute::{BudgetMode, CostEntry, TokenUsage};
use crate::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crate::types::events::{ControlSignal, StreamEvent};
use crate::types::fiduciary::{FiduciaryDutyEvent, PersonaDisclosure};
use crate::types::intelligence::PromptTemplate;
use crate::types::memory::{MemoryRecord, OutcomeRecord, SessionContext};
use crate::types::principal::{AutonomyLevel, Principal};
use crate::types::self_improvement::SessionLesson;
use crate::ui::visualization::GodView;
use anyhow::{Context, Result};
use futures::StreamExt;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::{Mutex, MutexGuard, RwLock, mpsc};
use tracing::{instrument, warn};

const MAX_SESSION_TURNS: usize = 1000;

/// Return value of `process_proposed_artifacts`.
enum ArtifactProcessOutcome {
    /// All artifacts passed validation; ready to commit.
    Ready(Vec<PreparedArtifactChange>, TurnOutcome),
    /// One or more artifacts were detected as regressive (quality regression).
    Regressive,
    /// Validation failed for a reason other than regression (AST, MC, lint, …).
    Invalid,
}

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

/// Central orchestrator for the Crosstalk multi-agent swarm.
///
/// # Field Organisation
///
/// Fields are grouped into five logical categories:
///
/// ## Core I/O
/// Agent list, state manager, event/control channels, file writer, and the
/// MCP gateway used to dispatch tool calls.
///
/// ## Intelligence & Routing
/// Engines that decide *which* agents run, *how* their outputs are merged, and
/// *whether* a turn should be retried: `IntelligenceEngine`, `CollectiveIntelligenceEngine`,
/// `ReasoningEngine`, `TopologyManager`, `SurpriseEngine`, `MetacognitiveObserver`,
/// `PromptEvolver`, and the template cache.
///
/// ## Memory
/// Persistent and in-session memory: `MemoryStore`, `MemoryBridge`,
/// `session_memory_map`, and the gold-state snapshot used for regression detection.
///
/// ## Observability & Telemetry
/// Engines that produce signals without mutating core state: `AnalyticsEngine`,
/// `GodView`, the continuous auditor channel pair, and the `StallDetector`.
///
/// ## Concurrency & Session
/// Per-session bookkeeping that is shared safely across async boundaries:
/// `completion_probability` (atomic), rate limiter, rollback/skip counters,
/// verification-failure map, `session_ctx`, and the turn broadcast channel.
pub struct Orchestrator {
    // ── Core I/O ─────────────────────────────────────────────────────────────
    agents: Vec<Box<dyn PromptAgent>>,
    state_manager: StateManager,
    event_tx: mpsc::Sender<StreamEvent>,
    control_rx: Mutex<mpsc::Receiver<ControlSignal>>,
    mc_runner: MonteCarloRunner,
    pub mcp_gateway: Mutex<McpGateway>,
    pub file_writer: FileWriter,
    nix_env: Option<HashMap<String, String>>,

    // ── Intelligence & Routing ────────────────────────────────────────────────
    pub intelligence: Mutex<IntelligenceEngine>,
    pub reasoning: ReasoningEngine,
    pub self_improve: SelfImprovementEngine,
    pub swarm: SwarmController,
    pub planning: PlanningEngine,
    pub collective: Mutex<CollectiveIntelligenceEngine>,
    pub surprise_engine: Mutex<SurpriseEngine>,
    pub observer: Mutex<MetacognitiveObserver>,
    pub pending_interventions: Mutex<Vec<crate::engines::metacognition::Intervention>>,
    pub prompt_evolver: Mutex<PromptEvolver>,
    pub topology: Mutex<TopologyManager>,
    active_topology_directive: Mutex<Option<crate::engines::topology::TopologyDirective>>,
    pub template_cache: Arc<RwLock<BTreeMap<String, PromptTemplate>>>,

    // ── Memory ────────────────────────────────────────────────────────────────
    pub memory_store: Arc<MemoryStore>,
    pub memory_bridge: Mutex<MemoryBridge>,
    pub session_memory_map: Mutex<BTreeMap<String, Arc<MemoryStore>>>,
    pub gold_state: Mutex<Option<ConversationState>>,

    // ── Observability & Telemetry ─────────────────────────────────────────────
    pub analytics: AnalyticsEngine,
    pub viz: Mutex<GodView>,
    pub auditor_tx: Option<mpsc::Sender<ConversationState>>,
    pub audit_rx: Mutex<mpsc::UnboundedReceiver<AuditAlert>>,
    stall_detector: Mutex<StallDetector>,

    // ── Concurrency & Session ─────────────────────────────────────────────────
    pub signer: TurnSigner,
    pub compute: Mutex<ComputeManager>,
    pub completion_probability: Arc<AtomicU64>,
    rate_limiter: Arc<RequestRateLimiter>,
    resource_rx: Mutex<tokio::sync::broadcast::Receiver<crate::engines::compute::ResourceEvent>>,
    session_ctx: Mutex<SessionContext>,
    rollback_counters: Mutex<BTreeMap<String, u32>>,
    skip_until: Mutex<BTreeMap<String, u32>>,
    /// Tracks per-agent verification failure counts for the current task.
    pub verification_failures: Mutex<HashMap<String, u32>>,
    /// Broadcast channel: every committed Turn is sent here so swarm nodes can react.
    turn_tx: tokio::sync::broadcast::Sender<Turn>,
    pub principal: Mutex<Principal>,

    // ── Closed-Loop Feedback (C-002) ──────────────────────────────────────────
    /// Quality samples from turns where critique protocol was OFF (A/B control arm).
    ab_control_quality: Mutex<Vec<f64>>,
    /// Quality samples from turns where critique protocol was ON (A/B test arm).
    ab_test_quality: Mutex<Vec<f64>>,
    /// Applies auto-adopted parameter changes when an A/B test is significant.
    runtime_adjuster: Mutex<crate::engines::self_improvement::RuntimeParameterAdjuster>,
    /// High-impact analytics recommendations queued for injection into the next prompt.
    pending_planning_hints: Mutex<Vec<String>>,
    /// ID of the prompt template last rendered in build_differential_prompt (for feedback loop).
    last_rendered_template_id: Mutex<Option<String>>,
    /// EMA of recent turn quality (f64 bits stored atomically) for adaptive evolution rate.
    recent_quality_ema: Arc<std::sync::atomic::AtomicU64>,
    /// Lessons distilled from previous sessions, injected on the first turn.
    prior_lessons: Mutex<Vec<SessionLesson>>,
}

fn resolve_memory_store_path() -> Result<String> {
    use std::path::{Component, PathBuf};
    let base: PathBuf = if let Ok(d) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(d).join("crosstalk")
    } else if let Ok(h) = std::env::var("HOME") {
        PathBuf::from(h).join(".local/share/crosstalk")
    } else {
        anyhow::bail!("neither XDG_DATA_HOME nor HOME is set; cannot resolve memory store path");
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
    s.contains("429")
        || s.contains("Too Many Requests")
        || s.contains("rate_limit")
        || s.contains("RateLimit")
}

fn is_fatal_auth_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    s.contains("credit balance")
        || s.contains("insufficient_quota")
        || s.contains("Insufficient Balance")
        || s.contains("Payment Required")
        || s.contains("401")
        || s.contains("402")
        || s.contains("403")
}

impl Orchestrator {
    fn truncate_str(s: &str, max_bytes: usize) -> &str {
        let mut end = s.len().min(max_bytes);
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }

    pub async fn new(
        state_manager: StateManager,
        agents: Vec<Box<dyn PromptAgent>>,
        event_tx: mpsc::Sender<StreamEvent>,
        control_rx: mpsc::Receiver<ControlSignal>,
        workspace_root: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let agent_count = agents.len();
        let file_writer = match workspace_root {
            Some(root) => FileWriter::new(root)?,
            None => FileWriter::from_env()?,
        };
        let mut mcp_gateway = McpGateway::with_workspace(file_writer.root.display().to_string());
        let tools = match tokio::task::spawn_blocking(ToolDiscovery::scan).await {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "tool discovery task failed");
                vec![]
            }
        };
        for tool in tools {
            mcp_gateway.register_tool(tool);
        }
        mcp_gateway.permissions.tiers.insert(
            "orchestrator".to_string(),
            crate::types::mcp::PermissionTier::Full,
        );

        let nix_env = match tokio::task::spawn_blocking(|| {
            if which::which("nix").is_err() {
                return None;
            }
            let deps: Vec<String> = ["rustc", "cargo", "git", "gcc", "pkg-config"]
                .iter()
                .filter(|d| which::which(d).is_ok())
                .map(|d| d.to_string())
                .collect();
            Some(NixManager::generate_flake_static(&deps))
        })
        .await
        {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "nix environment setup task failed");
                None
            }
        };

        if let Some(env) = &nix_env {
            mcp_gateway.set_nix_env(Some(std::collections::HashMap::from([(
                "NIX_ENV".to_string(),
                env.clone(),
            )])));
        }

        let (alert_tx, alert_rx) = mpsc::unbounded_channel::<AuditAlert>();
        let auditor_tx = Some(ContinuousAuditor::spawn(alert_tx));
        let mut memory_store_inner = MemoryStore::new(&resolve_memory_store_path()?);
        memory_store_inner.init().await?;
        let memory_store = Arc::new(memory_store_inner);

        let mut compute = ComputeManager::new();
        compute.start_background_monitor(10);
        let resource_rx = compute.resource_subscriber();

        // Load prior Elo ratings before memory_store is moved into Self.
        let prior_elo_json: Option<String> = memory_store
            .sessions
            .get("elo_ratings")
            .and_then(|records| records.last().map(|r| r.metadata_json.clone()));

        let prior_prompt_pop_json: Option<String> = memory_store
            .sessions
            .get("prompt_population")
            .and_then(|records| records.last().map(|r| r.metadata_json.clone()));

        let prior_topology_scores_json: Option<String> = memory_store
            .sessions
            .get("topology_scores")
            .and_then(|records| records.last().map(|r| r.metadata_json.clone()));

        let prior_collective_json: Option<String> = memory_store
            .sessions
            .get("collective_profiles")
            .and_then(|records| records.last().map(|r| r.metadata_json.clone()));

        let prior_ranker_weights_json: Option<String> = memory_store
            .sessions
            .get("ranker_weights")
            .and_then(|records| records.last().map(|r| r.metadata_json.clone()));

        let prior_lessons_loaded: Vec<SessionLesson> = memory_store
            .sessions
            .get("session_lessons")
            .map(|records| {
                records
                    .iter()
                    .rev()
                    .take(3)
                    .filter_map(|r| serde_json::from_str::<SessionLesson>(&r.metadata_json).ok())
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            agents,
            state_manager,
            event_tx,
            control_rx: Mutex::new(control_rx),
            mc_runner: MonteCarloRunner::new().context("Failed to init simulation")?,
            mcp_gateway: Mutex::new(mcp_gateway),
            memory_bridge: Mutex::new({
                let mut bridge = MemoryBridge::with_store(Arc::clone(&memory_store));
                if let Some(json) = &prior_ranker_weights_json {
                    bridge.import_ranker_weights_json(json);
                }
                bridge
            }),
            memory_store,
            intelligence: Mutex::new(IntelligenceEngine::new()),
            compute: Mutex::new(compute),
            reasoning: ReasoningEngine,
            self_improve: SelfImprovementEngine,
            swarm: SwarmController::new(),
            planning: PlanningEngine,
            signer: TurnSigner::new(),
            analytics: AnalyticsEngine,
            collective: Mutex::new({
                let mut coll = CollectiveIntelligenceEngine::new();
                if let Some(json) = &prior_collective_json {
                    coll.import_state_json(json);
                }
                coll
            }),
            viz: Mutex::new(GodView::new()),
            file_writer,
            auditor_tx,
            audit_rx: Mutex::new(alert_rx),
            surprise_engine: Mutex::new(SurpriseEngine::new()),
            observer: Mutex::new({
                let mut obs = MetacognitiveObserver::new();
                if let Some(json) = &prior_elo_json {
                    obs.import_elo_ratings(json);
                }
                obs
            }),
            pending_interventions: Mutex::new(Vec::new()),
            prompt_evolver: Mutex::new({
                let mut evolver = PromptEvolver::new();
                if let Some(json) = &prior_prompt_pop_json {
                    evolver.import_state_json(json);
                }
                evolver
            }),
            topology: Mutex::new({
                let mut topo = TopologyManager::new(agent_count);
                if let Some(json) = &prior_topology_scores_json {
                    topo.import_scores_json(json);
                }
                topo
            }),
            session_memory_map: Mutex::new(BTreeMap::new()),
            completion_probability: Arc::new(AtomicU64::new(0.0f64.to_bits())),
            active_topology_directive: Mutex::new(None),
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
                        tags: vec![],
                        performance_history: vec![],
                    },
                );
                m.insert(
                    "corrective".to_string(),
                    PromptTemplate {
                        id: "corrective".to_string(),
                        version: 1,
                        template_text: "The previous turn regressed. Please fix: {{task}}"
                            .to_string(),
                        task_category: TaskCategory::Research,
                        variables: vec!["task".to_string()],
                        tags: vec!["corrective".to_string()],
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
            nix_env: None,
            resource_rx: Mutex::new(resource_rx),
            gold_state: Mutex::new(None),
            turn_tx: tokio::sync::broadcast::channel::<Turn>(256).0,
            verification_failures: Mutex::new(HashMap::new()),
            stall_detector: Mutex::new(StallDetector::new()),
            principal: Mutex::new(Principal::anonymous("pending")),
            ab_control_quality: Mutex::new(Vec::new()),
            ab_test_quality: Mutex::new(Vec::new()),
            runtime_adjuster: Mutex::new(
                crate::engines::self_improvement::RuntimeParameterAdjuster::new(),
            ),
            pending_planning_hints: Mutex::new(Vec::new()),
            last_rendered_template_id: Mutex::new(None),
            recent_quality_ema: Arc::new(std::sync::atomic::AtomicU64::new(0.5f64.to_bits())),
            prior_lessons: Mutex::new(prior_lessons_loaded),
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

    pub async fn tool_call(
        &self,
        agent_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let categories = {
            let p = self.principal.lock().await;
            p.constraints.allowed_tool_categories.clone()
        };
        // Authorize under the gateway lock (fast, in-memory), then release the
        // lock before running the tool so concurrent tool calls do not
        // serialize on the gateway (H-019).
        let authorized = {
            let mut gw = self.mcp_gateway.lock().await;
            gw.set_principal_allowed_categories(categories);
            gw.authorize_tool_call(agent_id, name, &args)?
        };
        McpGateway::execute_authorized(authorized).await
    }

    #[instrument(skip_all, fields(session, turn))]
    pub fn subscribe_all(&self) -> tokio::sync::broadcast::Receiver<Turn> {
        self.turn_tx.subscribe()
    }

    /// Phase 1: Load session metadata, recall memory context, and detect regressions.
    /// Returns `(session_id, turn_idx, recent_turns, memory_examples, antipatterns, regression_prefix)`.
    async fn prepare_context_from_memory(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
    ) -> Result<(
        String,
        u32,
        Vec<Turn>,
        Vec<String>,
        Vec<MemoryRecord>,
        String,
    )> {
        let (session_id, turn_idx, recent_turns) = {
            let s = sigma_lock.lock().await;
            let recent: Vec<Turn> = s.turns.iter().rev().take(5).cloned().collect();
            (s.session_id.clone(), s.iteration_index, recent)
        };

        let recall_query = recent_turns
            .first()
            .map(|t| t.content.chars().take(200).collect::<String>())
            .unwrap_or_else(|| "latest turn context".to_string());

        let (memory_examples, antipatterns) = {
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(session_id.clone());
            let examples = vec![
                bridge
                    .recall_relevant_summary(&session_id, &recall_query, 3, turn_idx)
                    .await
                    .unwrap_or_default(),
            ];
            let anti = bridge.recall_antipatterns(&recall_query, 2).await;
            (examples, anti)
        };

        {
            let mut rx = self.resource_rx.lock().await;
            while let Ok(event) = rx.try_recv() {
                if let Some(alert) = &event.alert {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[compute] Resource alert: {}\n", alert),
                    })
                    .await?;
                }
            }
        }

        let regression_prefix = {
            let intell = self.intelligence.lock().await;
            if let Some(alert) = intell.detect_regression("swarm", &recent_turns) {
                RegressionFeedbackHandler::compose_corrective_prompt(
                    &alert,
                    "",
                    &RegressionFeedbackHandler::counter_examples(
                        &recent_turns,
                        TaskCategory::Research,
                    ),
                )
            } else {
                String::new()
            }
        };

        Ok((
            session_id,
            turn_idx,
            recent_turns,
            memory_examples,
            antipatterns,
            regression_prefix,
        ))
    }

    /// Phase 2: Apply analytics strategy recommendations and select active agents.
    /// Returns `(strategy_critique, strategy_reduce_agents, adaptive_selection, state_clone)`.
    async fn analyze_strategy_and_select_agents(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
    ) -> Result<(bool, bool, Option<AdaptiveSelection>, ConversationState)> {
        let mut strategy_critique = false;
        let mut strategy_reduce_agents = false;
        let adaptive_selection;

        {
            let s = sigma_lock.lock().await;
            let sub_swarms = crate::engines::planning::SubSwarmGenerator::identify_sub_swarms(&s);
            for task in sub_swarms {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[Swarm] Spawning sub-orchestrator for complex task: {}\n",
                        task.description
                    ),
                })
                .await?;
                self.swarm.spawn_node(&task.id, self.turn_tx.subscribe());
            }

            let recs = crate::engines::analytics::StrategyRecommender::recommend(&s);
            for rec in &recs {
                if rec.confidence > 0.5 {
                    match rec.action.as_str() {
                        "switch_to_critique_protocol" => {
                            strategy_critique = true;
                            self.emit(StreamEvent::TokenReceived {
                                agent_id: "System".to_string(),
                                token:
                                    "Low success rate detected — switching to critique protocol\n"
                                        .to_string(),
                            })
                            .await?;
                        }
                        "reduce_parallel_inference" => {
                            strategy_reduce_agents = true;
                        }
                        _ => {}
                    }
                }
            }
            // Route high-impact recommendations to the planning layer.
            {
                let high_impact: Vec<String> = recs
                    .iter()
                    .filter(|r| r.expected_impact > 0.7 && r.confidence > 0.7)
                    .map(|r| r.action.clone())
                    .collect();
                if !high_impact.is_empty() {
                    let mut hints = self.pending_planning_hints.lock().await;
                    hints.extend(high_impact);
                    let excess = hints.len().saturating_sub(10);
                    if excess > 0 {
                        hints.drain(..excess);
                    }
                }
            }

            adaptive_selection = {
                let obs = self.observer.lock().await;
                let coll = self.collective.lock().await;
                Some(coll.select_strategy_adaptive(&s, &obs))
            };
        }

        let state_clone = {
            let guard = sigma_lock.lock().await;
            if guard.iteration_index == 0 && guard.turns.is_empty() {
                self.state_manager.checkpoint(&guard)?;
            }
            guard.clone()
        };

        Ok((
            strategy_critique,
            strategy_reduce_agents,
            adaptive_selection,
            state_clone,
        ))
    }

    /// Phase 3: Build the final prompt string including memory, metacognition, and topology.
    /// Returns `(prompt, history_contents, active_agents, artifacts_snapshot)`.
    async fn build_prompt(
        &self,
        s: &ConversationState,
        strategy_critique: bool,
        adaptive_selection: &Option<AdaptiveSelection>,
        memory_examples: &[String],
        antipatterns: &[MemoryRecord],
        regression_prefix: &str,
    ) -> Result<(
        String,
        Vec<String>,
        Vec<(usize, String)>,
        BTreeMap<String, Arc<Artifact>>,
    )> {
        let history_contents: Vec<String> = s
            .turns
            .iter()
            .rev()
            .take(10)
            .map(|t| t.content.clone())
            .collect();

        let mut distilled_prompt = self.build_differential_prompt(s).await;

        if strategy_critique {
            distilled_prompt.push_str(
                "\n\nCRITICAL: Previous attempts had high failure rate. \
                 Before proposing changes, critique your own approach. \
                 Identify what could go wrong. Only proceed with the safest path.\n",
            );
        }

        if let Some(sel) = adaptive_selection {
            match sel.strategy {
                MetaStrategy::DebateAndCritique => {
                    distilled_prompt.push_str(
                        "\n\n[META-STRATEGY: DebateAndCritique] \
                         High variance in recent certainty scores detected. \
                         Critically examine all proposals and surface disagreements before converging.\n",
                    );
                }
                MetaStrategy::DirectImplementation => {
                    if let Some(ref agent) = sel.preferred_agent {
                        distilled_prompt.push_str(&format!(
                            "\n\n[META-STRATEGY: DirectImplementation] \
                             Dominant specialist {} detected (Elo > 1600). \
                             Prefer this agent's proposal for final synthesis.\n",
                            agent
                        ));
                    }
                }
                MetaStrategy::MemoryInjection => {
                    distilled_prompt.push_str(
                        "\n\n[META-STRATEGY: MemoryInjection] \
                         Convergence probability is low after multiple turns. \
                         Retrieve and apply relevant prior session lessons before proceeding.\n",
                    );
                }
                _ => {}
            }
        }

        let mut active = self.select_active_agents(s, adaptive_selection).await;

        // Apply topology-driven agent grouping
        {
            use crate::engines::topology::AgentGrouping;
            let directive = {
                let stored = self.active_topology_directive.lock().await;
                match stored.as_ref() {
                    Some(d) => d.clone(),
                    None => {
                        let topo = self.topology.lock().await;
                        topo.current_directive()
                    }
                }
            };
            match directive.agent_grouping {
                AgentGrouping::Single(idx) => {
                    if idx < active.len() {
                        active = vec![active[idx].clone()];
                    }
                }
                AgentGrouping::Pairs(ref pairs) => {
                    if let Some(&(a, b)) = pairs.first() {
                        let mut subset = Vec::new();
                        if a < active.len() {
                            subset.push(active[a].clone());
                        }
                        if b < active.len() {
                            subset.push(active[b].clone());
                        }
                        if !subset.is_empty() {
                            active = subset;
                        }
                    }
                }
                AgentGrouping::Branches(ref branches) => {
                    let branch_idx = (s.iteration_index as usize) % branches.len().max(1);
                    if let Some(branch) = branches.get(branch_idx) {
                        let subset: Vec<_> = branch
                            .iter()
                            .filter_map(|&i| active.get(i).cloned())
                            .collect();
                        if !subset.is_empty() {
                            active = subset;
                        }
                    }
                }
                AgentGrouping::All => {}
            }
            if let Some(modifier) = &directive.prompt_modifier {
                distilled_prompt.push('\n');
                distilled_prompt.push_str(modifier);
                distilled_prompt.push('\n');
            }
        }

        let structure = ReasoningEngine::select_structure(TaskCategory::Research, &active[0].1);
        match structure {
            TurnStructure::StepByStep => distilled_prompt
                .push_str("\nStructure your response with numbered reasoning steps."),
            TurnStructure::ProsCons => {
                distilled_prompt.push_str("\nExplicitly analyze tradeoffs (Pros vs Cons).")
            }
            TurnStructure::CodeFirst => {
                distilled_prompt.push_str("\nProvide the code delta (Δα) before any explanation.")
            }
            TurnStructure::Symbolic => distilled_prompt.push_str(
                "\nUse SSM (Symbolic Swarm Mode): use ∀, ∃, ⊢, ⊥, Δα, σ, μ. Minimize prose.",
            ),
            _ => {}
        }

        if !regression_prefix.is_empty() {
            distilled_prompt = format!("{regression_prefix}\n{distilled_prompt}");
        }
        if !memory_examples.is_empty() {
            distilled_prompt.push_str("\n\nSuccessful examples from similar tasks:\n");
            for (i, _ex) in memory_examples.iter().take(5).enumerate() {
                crate::log_warn!(
                    writeln!(
                        distilled_prompt,
                        "- [Example {}] (recalled from memory)",
                        i + 1
                    ),
                    "Failed to write example to prompt"
                );
            }
        }
        if !antipatterns.is_empty() {
            distilled_prompt.push_str("\n\nAntipatterns to AVOID (failed in similar tasks):\n");
            for ap in antipatterns.iter().take(3) {
                crate::log_warn!(
                    writeln!(
                        distilled_prompt,
                        "- [Session {}] {}",
                        ap.session_id, ap.metadata_json
                    ),
                    "Failed to write antipattern to prompt"
                );
            }
        }

        // Inject known failure patterns from intelligence store (Task 6)
        {
            let intell = self.intelligence.lock().await;
            let task_cat = s
                .turns
                .last()
                .and_then(|t| t.task_category)
                .unwrap_or(TaskCategory::Research);
            let patterns = intell.top_failure_patterns(task_cat, 3);
            if !patterns.is_empty() {
                distilled_prompt.push_str("\n\n[KNOWN FAILURE MODES — avoid these]:\n");
                for p in &patterns {
                    crate::log_warn!(
                        writeln!(distilled_prompt, "- {p}"),
                        "Failed to write failure pattern to prompt"
                    );
                }
            }
        }

        // Inject metacognitive observer feedback from prior turn
        {
            let obs = self.observer.lock().await;
            for (_, name) in &active {
                if let Some(state) = obs.epistemic_state(name)
                    && state.confidence < 0.5
                    && !state.defeated.is_empty()
                {
                    crate::log_warn!(
                        writeln!(
                            distilled_prompt,
                            "\n[EPISTEMIC UPDATE for {name}] Confidence: {:.0}%. \
                             Defeated assumptions: {:?}. Adjust your reasoning accordingly.",
                            state.confidence * 100.0,
                            state.defeated
                        ),
                        "Failed to write epistemic update to prompt"
                    );
                }
            }
        }

        // Inject pending interventions from prior turn's observer
        {
            let mut pending = self.pending_interventions.lock().await;
            if !pending.is_empty() {
                for (_, name) in &active {
                    if let Some(block) = MetacognitiveObserver::format_interventions(&pending, name)
                    {
                        distilled_prompt.push_str(&block);
                    }
                }
                pending.clear();
            }
        }

        // Inject high-impact analytics recommendations as planning hints.
        {
            let hints = self.pending_planning_hints.lock().await;
            if !hints.is_empty() {
                distilled_prompt.push_str("\n\n[PLANNING HINTS]:\n");
                for hint in hints.iter() {
                    crate::log_warn!(
                        writeln!(distilled_prompt, "- {hint}"),
                        "Failed to write planning hint to prompt"
                    );
                }
                // Hints are cleared in run_turn() only after successful commit.
            }
        }

        let artifacts_snapshot = s.artifacts.clone();
        Ok((
            distilled_prompt,
            history_contents,
            active,
            artifacts_snapshot,
        ))
    }

    /// Selects and orders the active agent list for a turn.
    async fn select_active_agents(
        &self,
        s: &ConversationState,
        adaptive_selection: &Option<AdaptiveSelection>,
    ) -> Vec<(usize, String)> {
        let mut active = Vec::new();
        {
            let skips = self.skip_until.lock().await;
            for (idx, agent) in self.agents.iter().enumerate() {
                let until = skips.get(agent.name()).copied().unwrap_or(0);
                if s.iteration_index >= until {
                    active.push((idx, agent.name().to_string()));
                }
            }
        }
        if active.is_empty() {
            active.push((0, self.agents[0].name().to_string()));
        }

        {
            let vf = self.verification_failures.lock().await;
            active.retain(|(_, n)| vf.get(n).copied().unwrap_or(0) <= 3);
            if active.is_empty() {
                active.push((0, self.agents[0].name().to_string()));
            }
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
            ) && let Some(pos) = active.iter().position(|(_, n)| *n == best)
            {
                active.swap(0, pos);
            }
        }

        {
            let obs = self.observer.lock().await;
            let task_cat = s
                .turns
                .last()
                .and_then(|t| t.task_category)
                .unwrap_or(TaskCategory::Research);
            if !obs.elo_ratings.is_empty() {
                active.sort_by(|a, b| {
                    let elo_a = obs.elo_for_category(&a.1, task_cat);
                    let elo_b = obs.elo_for_category(&b.1, task_cat);
                    elo_b
                        .partial_cmp(&elo_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        if let Some(AdaptiveSelection {
            preferred_agent: Some(preferred),
            ..
        }) = adaptive_selection
            && let Some(pos) = active.iter().position(|(_, n)| n == preferred)
        {
            active.swap(0, pos);
        }

        active
    }

    /// Phase 4: Call agents (with caching, rate-limiting, streaming, and control signal handling).
    /// Returns the collected `(agent_id, response_text)` pairs.
    async fn call_agents(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
        prompt: Arc<str>,
        mut active_agents: Vec<(usize, String)>,
        strategy_reduce_agents: bool,
    ) -> Result<Vec<(String, String)>> {
        let (paused_tx, paused_rx) = tokio::sync::watch::channel(false);

        if strategy_reduce_agents && active_agents.len() > 1 {
            active_agents.truncate(1);
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "Budget burn rate high — reducing to single agent\n".to_string(),
            })
            .await?;
        }
        {
            let s = sigma_lock.lock().await;
            if s.budget.mode() == BudgetMode::Emergency && active_agents.len() > 1 {
                drop(s);
                active_agents.truncate(1);
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: "[compute] Emergency budget mode: single agent only\n".to_string(),
                })
                .await?;
            }
        }

        let mut cached_results = Vec::new();
        let mut uncached_agents = Vec::new();
        {
            let mut compute = self.compute.lock().await;
            for entry in &active_agents {
                if let Some(cached) = compute.cache.get(&prompt, &entry.1) {
                    cached_results.push((entry.1.clone(), cached));
                } else {
                    uncached_agents.push(entry.clone());
                }
            }
        }

        {
            let agent_names: Vec<&str> = uncached_agents.iter().map(|(_, n)| n.as_str()).collect();
            if !agent_names.is_empty() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("\nAsking {} for their take...\n", agent_names.join(" and ")),
                })
                .await?;
            }
            if !cached_results.is_empty() {
                let cached_names: Vec<&str> =
                    cached_results.iter().map(|(n, _)| n.as_str()).collect();
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("Reusing cached response from {}\n", cached_names.join(", ")),
                })
                .await?;
            }
        }

        let (is_divergent, artifacts_for_divergent) = {
            let s = sigma_lock.lock().await;
            let divergent = s.mode_library.current().context_distribution
                == crate::types::mode::ContextDistribution::Divergent;
            let arts: std::collections::HashMap<
                String,
                std::sync::Arc<crate::types::artifact::Artifact>,
            > = s
                .artifacts
                .iter()
                .map(|(k, v)| (k.clone(), std::sync::Arc::clone(v)))
                .collect();
            (divergent, arts)
        };

        let mut tasks = Vec::new();
        for (idx, name) in &uncached_agents {
            let agent = &self.agents[*idx];
            let agent_id = name.clone();
            let prompt = Arc::clone(&prompt);
            let event_tx = self.event_tx.clone();
            let mut p_rx = paused_rx.clone();
            let rate_limiter = Arc::clone(&self.rate_limiter);

            let divergent_supplement = if is_divergent {
                if let Some(hash_pos) = agent_id.find('#') {
                    let role = &agent_id[hash_pos + 1..];
                    Self::divergent_context_for_role(role, &artifacts_for_divergent)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let feedback = {
                let collective = self.collective.lock().await;
                collective
                    .profiles
                    .get(&agent_id)
                    .and_then(crate::engines::prompt_evolution::ClosedLoopFeedback::generate_corrective_directive)
            };

            tasks.push(async move {
                let mut agent_prompt = (*prompt).to_string();
                if let Some(f) = feedback {
                    agent_prompt = format!("{}\n\n{}", f, agent_prompt);
                }
                if !divergent_supplement.is_empty() {
                    agent_prompt.push_str(&divergent_supplement);
                }

                let mut delay_ms = 1_000u64;
                for attempt in 0u32..4 {
                    if attempt > 0 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                        delay_ms = (delay_ms * 2).min(30_000);
                    }

                    rate_limiter.wait_for_permit(&agent_id).await;
                    let mut stream = match agent.stream_prompt(&agent_prompt).await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::info!(agent = %agent_id, "local inference failed, attempting remote MCP fallback");
                            if let Ok(remote_res) = McpGateway::remote_sampling(&agent_prompt, &agent_id).await {
                                return Ok((agent_id, remote_res));
                            }
                            let e = anyhow::anyhow!("Agent {agent_id} failure: {e:?}");
                            if is_fatal_auth_error(&e) {
                                return Err(e);
                            }
                            if is_rate_limited(&e) && attempt < 3 {
                                event_tx
                                    .send(StreamEvent::TokenReceived {
                                        agent_id: agent_id.clone(),
                                        token: format!("\n[{agent_id}] rate-limited, retrying in {}s...\n", delay_ms / 1000),
                                    })
                                    .await?;
                                continue;
                            }
                            return Err(e);
                        }
                    };

                    let mut response = String::new();
                    let mut hit_rate_limit = false;
                    loop {
                        if *p_rx.borrow() {
                            crate::log_warn!(p_rx.changed().await, "Failed to wait for pause state change");
                            continue;
                        }
                        match tokio::time::timeout(std::time::Duration::from_secs(120), stream.next()).await {
                            Err(_) => return Err(anyhow::anyhow!("Agent {agent_id} timed out waiting for response")),
                            Ok(Some(Ok(chunk))) => {
                                response.push_str(&chunk);
                                event_tx
                                    .send(StreamEvent::TokenReceived { agent_id: agent_id.clone(), token: chunk })
                                    .await?;
                            }
                            Ok(Some(Err(e))) => {
                                let e = anyhow::anyhow!("Agent {agent_id} stream error: {e:?}");
                                if is_fatal_auth_error(&e) {
                                    return Err(e);
                                }
                                if is_rate_limited(&e) && attempt < 3 {
                                    hit_rate_limit = true;
                                    event_tx
                                        .send(StreamEvent::TokenReceived {
                                            agent_id: agent_id.clone(),
                                            token: format!("\n[{agent_id}] rate-limited mid-stream, retrying in {}s...\n", delay_ms / 1000),
                                        })
                                        .await?;
                                } else {
                                    return Err(e);
                                }
                                break;
                            }
                            Ok(None) => break,
                        }
                    }
                    if !hit_rate_limit {
                        tracing::info!(agent = %agent_id, response_len = response.len(), "agent responded");
                        return Ok((agent_id, response));
                    }
                }
                Err(anyhow::anyhow!("Agent {agent_id} exhausted rate-limit retries"))
            });
        }

        let mut results_fut = futures::future::join_all(tasks);
        let mut final_results = Vec::new();
        let mut ctrl_guard = self.control_rx.lock().await;
        let mut ctrl_open = true;

        loop {
            tokio::select! {
                res = &mut results_fut => {
                    for r in res {
                        match r {
                            Ok(val) => final_results.push(val),
                            Err(e) => {
                                let msg = e.to_string();
                                if msg.contains("timed out")
                                    && let Some(name) = msg.strip_prefix("Agent ").and_then(|s| s.split_whitespace().next()) {
                                        let turn = sigma_lock.lock().await.iteration_index;
                                        self.skip_until.lock().await.insert(name.to_string(), turn + 3);
                                        self.emit(StreamEvent::Error(format!("Agent {} timed out, skipping for 3 turns", name))).await?;
                                        continue;
                                    }
                                self.emit(StreamEvent::Error(format!("Agent dropped: {}", e))).await?;
                            }
                        }
                    }
                    final_results.append(&mut cached_results);
                    if final_results.is_empty() {
                        return Err(anyhow::anyhow!("All agents in swarm failed to respond."));
                    }
                    {
                        let mut compute = self.compute.lock().await;
                        for (id, text) in &final_results {
                            compute.cache.insert(&prompt, id, text.clone(), 1.0);
                        }
                    }
                    break;
                }
                signal = ctrl_guard.recv(), if ctrl_open => {
                    match signal {
                        Some(ControlSignal::Pause) => { crate::log_warn!(paused_tx.send(true), "Failed to send pause signal"); }
                        Some(ControlSignal::Resume) => { crate::log_warn!(paused_tx.send(false), "Failed to send resume signal"); }
                        Some(ControlSignal::Shutdown) => return Ok(vec![]),
                        Some(ControlSignal::LockCode(name)) => {
                            let mut sigma = sigma_lock.lock().await;
                            if let Some(r) = sigma.goal_tree.root.as_mut() {
                                r.title = format!("{} [LOCKED: {}]", r.title, name);
                                r.status = crate::types::planning::GoalStatus::Complete;
                            }
                            self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("[Steer] Locked artifact: {}\n", name) }).await?;
                        }
                        Some(ControlSignal::MuteAgent(id)) => {
                            let mut sigma = sigma_lock.lock().await;
                            sigma.agent_weights.retain(|k, _| k != &id);
                            self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("[Steer] Muted agent: {}\n", id) }).await?;
                        }
                        Some(ControlSignal::DampenSwarm(factor)) => {
                            let mut sigma = sigma_lock.lock().await;
                            for w in sigma.agent_weights.values_mut() {
                                *w *= factor;
                            }
                            self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("[Steer] Dampened swarm by factor: {:.2}\n", factor) }).await?;
                        }
                        Some(ControlSignal::Inject(text)) => {
                            self.emit(StreamEvent::TokenReceived { agent_id: "User".to_string(), token: format!("\n[Neural Intercept] Injecting: {}\n", text) }).await?;
                            let mut sigma = sigma_lock.lock().await;
                            let user_turn = Turn {
                                index: sigma.iteration_index,
                                model_id: "User".to_string(),
                                content: text.clone(),
                                timestamp: ConversationState::now(),
                                diffs: vec![],
                                certainty: Some(1.0),
                                outcome: TurnOutcome::Unknown,
                                task_category: Some(TaskCategory::Research),
                                structure: Some(TurnStructure::FreeForm),
                                signature: vec![],
                                surprise_signal: None,
                                consistency_score: None,
                                diff_quality_score: None,
                                persona_disclosure: None,
                            };
                            sigma.push_turn(user_turn);
                            sigma.iteration_index += 1;
                            return Ok(vec![]);
                        }
                        Some(ControlSignal::Rewind(index)) => {
                            if let Ok(Some(restored)) = self.state_manager.restore_async(index).await {
                                let mut s = sigma_lock.lock().await;
                                *s = restored;
                                self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("\n[Rewound to iteration {}]\n", index) }).await?;
                                return Ok(vec![]);
                            }
                        }
                        Some(ControlSignal::CycleMode) => {
                            let mut s = sigma_lock.lock().await;
                            let old = s.mode_library.current_name().to_string();
                            s.mode_library.cycle_next();
                            let new_name = s.mode_library.current_name().to_string();
                            drop(s);
                            let _ = self.emit(StreamEvent::ModeTransition {
                                from_name: old,
                                to_name: new_name,
                                reason: "User cycle".to_string(),
                                synthesized: false,
                            }).await;
                        }
                        Some(ControlSignal::SetModeByName(name)) => {
                            let mut s = sigma_lock.lock().await;
                            let old = s.mode_library.current_name().to_string();
                            if s.mode_library.switch_to_name(&name) {
                                let new_name = name.clone();
                                drop(s);
                                let _ = self.emit(StreamEvent::ModeTransition {
                                    from_name: old,
                                    to_name: new_name,
                                    reason: "User override".to_string(),
                                    synthesized: false,
                                }).await;
                            }
                        }
                        None => { ctrl_open = false; }
                    }
                }
            }
        }

        Ok(final_results)
    }

    /// Phase 5: Synthesise raw agent responses into a single text + artifact block.
    /// Returns `(response, nash_weight_updates, stall_risk)`.
    async fn synthesize_responses(
        &self,
        final_results: &[(String, String)],
        artifacts_snapshot: &BTreeMap<String, Arc<Artifact>>,
        weights: &BTreeMap<String, f64>,
    ) -> Result<(String, BTreeMap<String, f64>, f64)> {
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
                    if i == j {
                        continue;
                    }
                    let intersection = set_i.intersection(set_j).count() as f64;
                    let union = set_i.union(set_j).count().max(1) as f64;
                    sim_sum += intersection / union;
                    count += 1;
                }
                let mean_sim = if count > 0 {
                    sim_sum / count as f64
                } else {
                    1.0
                };
                if mean_sim < 0.3 {
                    outlier_penalty.insert(id, 0.1);
                }
            }

            // Entropy mapping (disagreement heatmap)
            let mut entropy_entries = Vec::new();
            let mut agent_artifact_proposals: std::collections::HashMap<
                String,
                std::collections::HashMap<String, String>,
            > = std::collections::HashMap::new();
            for (id, text) in final_results {
                for (art_name, (_, content)) in Self::parse_artifacts(text) {
                    agent_artifact_proposals
                        .entry(art_name)
                        .or_default()
                        .insert(id.clone(), content);
                }
            }
            for (art_name, proposals) in agent_artifact_proposals {
                let mut scores = Vec::new();
                let agents: Vec<(&String, &String)> = proposals.iter().collect();
                for (id_i, content_i) in &agents {
                    let mut dist_sum = 0.0;
                    let mut count = 0;
                    for (id_j, content_j) in &agents {
                        if id_i == id_j {
                            continue;
                        }
                        let diff =
                            similar::TextDiff::from_lines(content_i.as_str(), content_j.as_str());
                        dist_sum += 1.0 - diff.ratio();
                        count += 1;
                    }
                    let score: f64 = if count > 0 {
                        dist_sum as f64 / (count as f64).max(1.0)
                    } else {
                        0.0
                    };
                    scores.push(((*id_i).clone(), score));
                }
                entropy_entries.push(crate::types::events::EntropyEntry {
                    artifact_name: art_name,
                    scores,
                });
            }
            self.emit(crate::types::events::StreamEvent::EntropyUpdated(
                entropy_entries,
            ))
            .await?;
        }

        if final_results.len() > 1 {
            let outlier_names: Vec<&str> = outlier_penalty.keys().copied().collect();
            if outlier_names.is_empty() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "All {} agents broadly agree. Merging into consensus...\n",
                        final_results.len()
                    ),
                })
                .await?;
            } else {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "{} diverged from the group — downweighting outlier during synthesis\n",
                        outlier_names.join(", ")
                    ),
                })
                .await?;
            }
        }

        // Collective text synthesis (surprise + certainty + outlier calibrated)
        let synthesized_text = {
            let se = self.surprise_engine.lock().await;
            let text_proposals: Vec<(String, String, f64)> = final_results
                .iter()
                .map(|(id, text)| {
                    let base_w = weights.get(id).copied().unwrap_or(1.0);
                    let surprise_w = se.calibrate_weight(id, base_w);
                    let certainty = CertaintyAnalyzer::compute(text, 0.1);
                    let outlier_w = outlier_penalty.get(id.as_str()).copied().unwrap_or(1.0);
                    (id.clone(), text.clone(), surprise_w * certainty * outlier_w)
                })
                .collect();
            EnsembleEngine::merge_proposals(text_proposals, TaskCategory::Research, "")
        };

        // Collective artifact synthesis (Nash equilibrium resolution)
        let mut synthesized_artifacts = String::new();
        let mut nash_score_acc: std::collections::BTreeMap<String, (f64, u32)> =
            std::collections::BTreeMap::new();
        let artifact_proposals = Self::parse_artifacts(&synthesized_text);
        for (name, (lang, _content)) in artifact_proposals {
            let default_art = Arc::new(Artifact::default());
            let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
            let mut proposals_for_nash = Vec::new();
            for (agent_id, text) in final_results {
                let parsed = Self::parse_artifacts(text);
                if let Some((_, proposal_content)) = parsed.get(&name) {
                    let mut temp_art = (**current).clone();
                    temp_art.content = proposal_content.clone();
                    proposals_for_nash.push((agent_id.as_str(), temp_art, TurnOutcome::Compiled));
                }
            }
            if proposals_for_nash.is_empty() {
                continue;
            }
            let nash_refs: Vec<(&str, &Artifact, TurnOutcome)> = proposals_for_nash
                .iter()
                .map(|(id, art, out)| (*id, art, *out))
                .collect();
            for (agent_id, score) in
                crate::engines::consensus::NashSolver::compute_nash_scores(&nash_refs)
            {
                let e = nash_score_acc.entry(agent_id).or_insert((0.0_f64, 0_u32));
                e.0 += score;
                e.1 += 1;
            }
            let winning_content =
                crate::engines::consensus::NashSolver::resolve_with_synthesis(&nash_refs, current);
            crate::log_warn!(
                writeln!(
                    synthesized_artifacts,
                    "\n```{}:{}\n{}\n```",
                    lang, name, winning_content
                ),
                "Failed to write synthesized artifact"
            );
        }

        let nash_weight_updates: std::collections::BTreeMap<String, f64> = nash_score_acc
            .into_iter()
            .map(|(id, (sum, count))| (id, if count > 0 { sum / count as f64 } else { 0.5 }))
            .collect();

        // Stall detection
        let stall_risk = {
            let proposals_map: HashMap<String, String> = final_results
                .iter()
                .map(|(id, text)| {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    text.hash(&mut h);
                    (id.clone(), format!("{:x}", h.finish()))
                })
                .collect();
            let turn_entropy = if final_results.len() >= 2 {
                let word_sets: Vec<std::collections::HashSet<&str>> = final_results
                    .iter()
                    .map(|(_, t)| t.split_whitespace().collect())
                    .collect();
                let union_size = word_sets
                    .iter()
                    .flat_map(|s| s.iter())
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                let avg_size: f64 =
                    word_sets.iter().map(|s| s.len() as f64).sum::<f64>() / word_sets.len() as f64;
                if union_size > 0 {
                    1.0 - avg_size / union_size as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let mut sd = self.stall_detector.lock().await;
            sd.push_turn(proposals_map, turn_entropy)
        };
        if stall_risk > 0.6 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[stall] High stall risk detected: {:.2}\n", stall_risk),
            })
            .await?;
        }

        let response = format!("{}\n{}", synthesized_text, synthesized_artifacts);
        Ok((response, nash_weight_updates, stall_risk))
    }

    /// Phase 6: Run security, tautology, and fallacy filters on the synthesised response.
    /// Returns `Ok(false)` when the response must be dropped, `Ok(true)` when it passes.
    async fn filter_response(&self, response: &str, history_contents: &[String]) -> Result<bool> {
        let secrets = SecretScanner::scan(response);
        if !secrets.is_empty() {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Blocked: Security Violation]\n".to_string(),
            })
            .await?;
            return Ok(false);
        }

        let current_p = f64::from_bits(
            self.completion_probability
                .load(std::sync::atomic::Ordering::Acquire),
        );
        if current_p < 0.5 && TautologyFilter::is_tautological(response, history_contents) {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Pruned: Tautology]\n".to_string(),
            })
            .await?;
            return Ok(false);
        }

        let fallacies = FallacyDetector::scan(response, &[]);
        if !fallacies.is_empty() {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("\n[Warning: {} fallacies detected]\n", fallacies.len()),
            })
            .await?;
        }

        Ok(true)
    }

    /// Phase 7: Validate artifacts, self-heal up to 3 times, then reward/penalise agents.
    /// Returns `Some((changes, turn_outcome, final_response))` on success, `None` on total failure.
    #[allow(clippy::too_many_arguments)]
    async fn validate_and_heal(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
        response: String,
        artifacts_snapshot: &BTreeMap<String, Arc<Artifact>>,
        active_agents: &[(usize, String)],
        weights: &BTreeMap<String, f64>,
        final_results: &[(String, String)],
        prompt: &Arc<str>,
        pre_turn_idx: u32,
        paused_rx: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Option<(Vec<PreparedArtifactChange>, TurnOutcome, String)>> {
        let mut retry_count = 0u32;
        let mut current_response = response;
        let mut final_prepared = None;
        let mut last_failure_regressive = false;

        while retry_count < 3 {
            let proposed_artifacts = Self::parse_artifacts(&current_response);
            match self
                .process_proposed_artifacts(proposed_artifacts, artifacts_snapshot)
                .await?
            {
                ArtifactProcessOutcome::Ready(changes, turn_outcome) => {
                    final_prepared = Some((changes, turn_outcome, current_response));
                    break;
                }
                outcome => {
                    last_failure_regressive = matches!(outcome, ArtifactProcessOutcome::Regressive);
                    retry_count += 1;
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "\n[Self-Healing] Synthesis failed validation. Attempting hot-patch cycle {}/3...\n",
                            retry_count
                        ),
                    })
                    .await?;

                    if retry_count == 2 {
                        let sigma = sigma_lock.lock().await;
                        let mut topo = self.topology.lock().await;
                        let retry_cat = sigma
                            .turns
                            .last()
                            .and_then(|t| t.task_category)
                            .unwrap_or(TaskCategory::Research);
                        let directive = topo.maybe_shift(&sigma, sigma.iteration_index, retry_cat);
                        drop(sigma);
                        if directive.is_none() {
                            topo.shift_to(
                                crate::engines::topology::DebateTopology::Mediated,
                                pre_turn_idx,
                                crate::engines::topology::TopologyReason::Deadlock,
                            );
                        }
                    }

                    let corrective_prompt = format!(
                        "{}\n\n[CRITICAL: Validation Failed]\nThe previous collective synthesis failed quality/safety gates. Re-implement the code blocks ensuring strict adherence to Rust safety and project invariants.",
                        prompt
                    );

                    let mut tasks = Vec::new();
                    for (idx, name) in active_agents {
                        let agent = &self.agents[*idx];
                        let agent_id = name.clone();
                        let p = corrective_prompt.clone();
                        let event_tx = self.event_tx.clone();
                        let mut p_rx = paused_rx.clone();
                        let rate_limiter = Arc::clone(&self.rate_limiter);

                        tasks.push(async move {
                            rate_limiter.wait_for_permit(&agent_id).await;
                            let mut stream = agent
                                .stream_prompt(&p)
                                .await
                                .map_err(|e| anyhow::anyhow!("Agent {agent_id} failure: {e:?}"))?;
                            let mut resp = String::new();
                            loop {
                                if *p_rx.borrow() {
                                    crate::log_warn!(
                                        p_rx.changed().await,
                                        "Failed to wait for pause state change during retry"
                                    );
                                    continue;
                                }
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(120),
                                    stream.next(),
                                )
                                .await
                                {
                                    Ok(Some(Ok(chunk))) => {
                                        resp.push_str(&chunk);
                                        event_tx
                                            .send(StreamEvent::TokenReceived {
                                                agent_id: agent_id.clone(),
                                                token: chunk,
                                            })
                                            .await?;
                                    }
                                    Err(_) => {
                                        return Err(anyhow::anyhow!(
                                            "Agent {agent_id} timed out waiting for response"
                                        ));
                                    }
                                    _ => break,
                                }
                            }
                            Ok::<(String, String), anyhow::Error>((agent_id, resp))
                        });
                    }

                    let new_proposals: Vec<(String, String)> = futures::future::join_all(tasks)
                        .await
                        .into_iter()
                        .flatten()
                        .collect();

                    if new_proposals.is_empty() {
                        break;
                    }

                    let text_proposals: Vec<(String, String, f64)> = new_proposals
                        .iter()
                        .map(|(id, text)| {
                            (
                                id.clone(),
                                text.clone(),
                                weights.get(id).copied().unwrap_or(1.0),
                            )
                        })
                        .collect();
                    let syn_text =
                        EnsembleEngine::merge_proposals(text_proposals, TaskCategory::Research, "");

                    let mut art_proposals: BTreeMap<String, (String, Vec<ArtifactDiff>)> =
                        BTreeMap::new();
                    for (_id, text) in &new_proposals {
                        for (name, (lang, content)) in Self::parse_artifacts(text) {
                            let default_art = Arc::new(Artifact::default());
                            let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                            let entry = art_proposals.entry(name).or_insert((lang, vec![]));
                            entry.1.push(DiffEngine::generate_delta(
                                &current.content,
                                &content,
                                current.version,
                            ));
                        }
                    }
                    let mut syn_arts = String::new();
                    for (name, (lang, diffs)) in art_proposals {
                        let default_art = Arc::new(Artifact::default());
                        let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                        if let Some(merged) = SynthesisEngine::merge(
                            &current.content,
                            diffs
                                .into_iter()
                                .map(|d| ("Anonymous".to_string(), d))
                                .collect(),
                            &lang,
                        ) {
                            crate::log_warn!(
                                writeln!(syn_arts, "\n```{}:{}\n{}\n```", lang, name, merged),
                                "Failed to write merged artifact"
                            );
                        }
                    }
                    current_response = format!("{}\n{}", syn_text, syn_arts);
                }
            }
        }

        if final_prepared.is_none() {
            {
                let intell = self.intelligence.lock().await;
                for (id, _) in final_results {
                    intell.update_diff_quality(id, false, last_failure_regressive);
                }
            }
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Self-Healing Failed] Could not converge on a valid synthesis after 3 attempts. Aborting turn.\n".to_string(),
            })
            .await?;
        }

        Ok(final_prepared)
    }

    /// Phase 8: Update agent intelligence profiles and metacognitive state after a successful turn.
    #[allow(clippy::too_many_arguments)]
    async fn update_agent_profiles(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
        final_results: &[(String, String)],
        winner_id: &str,
        turn_outcome: TurnOutcome,
        final_response: &str,
        pre_turn_idx: u32,
        pre_recent_turns: &[Turn],
        latency_ms: u64,
    ) -> Result<()> {
        // Reward agents that produced good diffs
        {
            let intell = self.intelligence.lock().await;
            for (id, _) in final_results {
                intell.update_diff_quality(id, true, false);
            }
        }

        if !final_results.is_empty() {
            let names: Vec<&str> = final_results.iter().map(|(id, _)| id.as_str()).collect();
            // Notify which artifacts will be written — done upstream, so just profile updates here.
            let sigma = sigma_lock.lock().await;
            let intell = self.intelligence.lock().await;
            for (id, text) in final_results {
                let p_turn = Turn {
                    index: sigma.iteration_index,
                    model_id: id.clone(),
                    content: text.clone(),
                    timestamp: ConversationState::now(),
                    diffs: vec![],
                    certainty: Some(CertaintyAnalyzer::compute(text, 0.1)),
                    outcome: if id == winner_id {
                        turn_outcome
                    } else {
                        TurnOutcome::Unknown
                    },
                    task_category: Some(TaskCategory::Research),
                    structure: Some(TurnStructure::FreeForm),
                    signature: vec![],
                    surprise_signal: None,
                    consistency_score: None,
                    diff_quality_score: None,
                    persona_disclosure: None,
                };
                intell.update_profile_with_latency(&p_turn, 0.7, latency_ms);
            }
            drop(intell);
            drop(sigma);
            let _ = names; // consumed above
        }

        // Metacognitive observation
        {
            let mut obs = self.observer.lock().await;
            let mut surprise = self.surprise_engine.lock().await;
            let mut interventions = Vec::new();
            for (agent_id_obs, response_text) in final_results {
                let agent_turn = Turn {
                    index: pre_turn_idx,
                    model_id: agent_id_obs.clone(),
                    content: response_text.clone(),
                    timestamp: ConversationState::now(),
                    diffs: vec![],
                    certainty: Some(CertaintyAnalyzer::compute(response_text, 0.1)),
                    outcome: if agent_id_obs == winner_id {
                        turn_outcome
                    } else {
                        TurnOutcome::Unknown
                    },
                    task_category: Some(TaskCategory::Research),
                    structure: None,
                    signature: vec![],
                    surprise_signal: None,
                    consistency_score: None,
                    diff_quality_score: None,
                    persona_disclosure: None,
                };
                let agent_interventions =
                    obs.observe_turn(&agent_turn, pre_recent_turns, &mut surprise);
                interventions.extend(agent_interventions);
            }
            drop(surprise);
            for intervention in &interventions {
                if let Some(block) = MetacognitiveObserver::format_interventions(
                    std::slice::from_ref(intervention),
                    &intervention.target_agent,
                ) {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "Observer".to_string(),
                        token: block,
                    })
                    .await?;
                }
            }

            let winner_turn = Turn {
                index: pre_turn_idx,
                model_id: winner_id.to_string(),
                content: final_response.to_string(),
                timestamp: ConversationState::now(),
                diffs: vec![],
                certainty: Some(CertaintyAnalyzer::compute(final_response, 0.1)),
                outcome: turn_outcome,
                task_category: Some(TaskCategory::Research),
                structure: None,
                signature: vec![],
                surprise_signal: None,
                consistency_score: None,
                diff_quality_score: None,
                persona_disclosure: None,
            };
            let current_quality = QualityScorer::score(&winner_turn);
            let prior_quality = pre_recent_turns
                .first()
                .map(QualityScorer::score)
                .unwrap_or(0.5);
            let improved = current_quality > prior_quality;
            {
                let pending = self.pending_interventions.lock().await;
                for intervention in pending.iter().filter(|i| i.target_agent == winner_id) {
                    obs.record_intervention_outcome(winner_id, intervention.source, improved);
                }
            }
            if !interventions.is_empty() {
                let mut pending = self.pending_interventions.lock().await;
                *pending = interventions;
            }

            if obs.should_eliminate(winner_id, 10) {
                let mut skips = self.skip_until.lock().await;
                skips.insert(winner_id.to_string(), pre_turn_idx + 100);
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "Observer".to_string(),
                    token: format!(
                        "[ELIMINATION] Agent {} removed from pool (Elo < 1200)\n",
                        winner_id
                    ),
                })
                .await?;
            }
        }

        // Topology: record outcome and check for shifts
        {
            let sigma = sigma_lock.lock().await;
            let topo_turn = Turn {
                index: pre_turn_idx,
                model_id: winner_id.to_string(),
                content: final_response.to_string(),
                timestamp: ConversationState::now(),
                diffs: vec![],
                certainty: None,
                outcome: turn_outcome,
                task_category: Some(TaskCategory::Research),
                structure: None,
                signature: vec![],
                surprise_signal: None,
                consistency_score: None,
                diff_quality_score: None,
                persona_disclosure: None,
            };
            let topo_cat = topo_turn.task_category.unwrap_or(TaskCategory::Research);
            let quality = RewardVector::from_turn(&topo_turn).weighted_score(topo_cat);
            // Use last ledger entry as a cost proxy; apply_turn_to_state hasn't run yet.
            let topo_cost = sigma
                .budget
                .entries
                .last()
                .map(|e| e.cost_usd)
                .unwrap_or(0.01);
            let mut topo = self.topology.lock().await;
            topo.record_turn_outcome(turn_outcome, quality, topo_cat, topo_cost, latency_ms);
            if let Some(directive) = topo.maybe_shift(&sigma, sigma.iteration_index, topo_cat) {
                if let Some(modifier) = &directive.prompt_modifier {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "Topology".to_string(),
                        token: format!("[TOPOLOGY SHIFT → {:?}] {modifier}\n", directive.topology),
                    })
                    .await?;
                }
                let mut stored = self.active_topology_directive.lock().await;
                *stored = Some(directive);
            }
        }

        Ok(())
    }

    #[instrument(skip_all, fields(session, turn))]
    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        // Phase 1: load memory context and detect prior regressions.
        let (
            pre_session_id,
            pre_turn_idx,
            pre_recent_turns,
            memory_examples,
            antipatterns,
            regression_prefix,
        ) = self.prepare_context_from_memory(&sigma_lock).await?;
        tracing::Span::current().record("session", pre_session_id.as_str());
        tracing::Span::current().record("turn", pre_turn_idx);
        tracing::info!(turn = pre_turn_idx, "turn starting");

        // Early convergence: skip agent calls when P(C) is high and last turn had no changes.
        {
            let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
            let last_turn_had_changes = pre_recent_turns
                .first()
                .map(|t| !t.diffs.is_empty())
                .unwrap_or(true);
            if current_p > 0.85 && !last_turn_had_changes && pre_turn_idx > 2 {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[convergence] P(C)={current_p:.2}, no artifact changes, skipping turn\n"
                    ),
                })
                .await?;
                return Ok(true);
            }
        }

        // Phase 2: analytics strategy + adaptive agent selection.
        let (strategy_critique, strategy_reduce_agents, adaptive_selection, s) =
            self.analyze_strategy_and_select_agents(&sigma_lock).await?;

        // Phase 3: build final prompt.
        let (raw_prompt, history_contents, active_agents, artifacts_snapshot) = self
            .build_prompt(
                &s,
                strategy_critique,
                &adaptive_selection,
                &memory_examples,
                &antipatterns,
                &regression_prefix,
            )
            .await?;

        const PROMPT_CHAR_CAP: usize = 24_000;
        let prompt_str = if raw_prompt.len() > PROMPT_CHAR_CAP {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[prompt] Truncated to {}K chars (was {} chars)\n",
                    PROMPT_CHAR_CAP / 1000,
                    raw_prompt.len()
                ),
            })
            .await?;
            let mut truncated = raw_prompt[..PROMPT_CHAR_CAP].to_string();
            truncated.push_str("\n... [truncated]");
            truncated
        } else {
            raw_prompt
        };

        tracing::info!(
            turn = pre_turn_idx,
            agents = active_agents.len(),
            prompt_len = prompt_str.len(),
            "calling agents"
        );
        let start_time = Instant::now();
        let prompt_arc: Arc<str> = Arc::from(prompt_str);

        // Phase 4: call agents (streaming, caching, control signal handling).
        let final_results = self
            .call_agents(
                &sigma_lock,
                Arc::clone(&prompt_arc),
                active_agents.clone(),
                strategy_reduce_agents,
            )
            .await?;

        // An empty result vec signals a control-signal early exit (Shutdown/Inject/Rewind).
        if final_results.is_empty() {
            // Determine the correct return value from the current state.
            let s = sigma_lock.lock().await;
            return Ok(s.iteration_index > pre_turn_idx);
        }

        let weights =
            crate::engines::consensus::InfluenceWeightManager::calculate_weights_with_recency(
                &*sigma_lock.lock().await,
                0.9,
            );

        // Phase 5: synthesise responses into a single text + artifact block.
        let (response, nash_weight_updates, stall_risk) = self
            .synthesize_responses(&final_results, &artifacts_snapshot, &weights)
            .await?;

        let latency_ms = start_time.elapsed().as_millis() as u64;
        tracing::info!(
            turn = pre_turn_idx,
            latency_ms,
            responses = final_results.len(),
            "agents responded"
        );
        let winner_id = "Collective Swarm".to_string();

        // Phase 6: security / tautology / fallacy filters.
        if !self.filter_response(&response, &history_contents).await? {
            return Ok(false);
        }

        // Phase 7: artifact validation with self-healing retry loop.
        let (_paused_tx, paused_rx) = tokio::sync::watch::channel(false);
        let Some((changes, turn_outcome, final_response)) = self
            .validate_and_heal(
                &sigma_lock,
                response,
                &artifacts_snapshot,
                &active_agents,
                &weights,
                &final_results,
                &prompt_arc,
                pre_turn_idx,
                paused_rx,
            )
            .await?
        else {
            return Ok(false);
        };

        if !changes.is_empty() {
            let names: Vec<&str> = changes.iter().map(|c| c.name.as_str()).collect();
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("Writing changes to {}\n", names.join(", ")),
            })
            .await?;
        }

        // Execute any [TOOL: ...] directives embedded in the final response (Task 9)
        {
            let directives = Self::parse_tool_directives(&final_response);
            let mut tool_outputs = Vec::new();
            for (tool_name, args) in &directives {
                let output = self
                    .execute_tool_directive(tool_name, args, &sigma_lock)
                    .await;
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "Tool".to_string(),
                    token: format!("{output}\n"),
                })
                .await?;
                tool_outputs.push((tool_name.clone(), output));
            }
            if !tool_outputs.is_empty() {
                sigma_lock.lock().await.last_tool_outputs = tool_outputs;
            }
        }

        // Phase 8: update agent profiles and metacognitive state.
        self.update_agent_profiles(
            &sigma_lock,
            &final_results,
            &winner_id,
            turn_outcome,
            &final_response,
            pre_turn_idx,
            &pre_recent_turns,
            latency_ms,
        )
        .await?;

        // A/B test: accumulate quality by critique arm; auto-adopt when significant.
        {
            let quality = match turn_outcome {
                TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence => 1.0,
                TurnOutcome::RolledBack
                | TurnOutcome::Rejected
                | TurnOutcome::VerificationFailed => 0.0,
                _ => 0.5,
            };
            if strategy_critique {
                self.ab_test_quality.lock().await.push(quality);
            } else {
                self.ab_control_quality.lock().await.push(quality);
            }
            let (control, test) = {
                let mut c = self.ab_control_quality.lock().await;
                let mut t = self.ab_test_quality.lock().await;
                if c.len() >= 10 && t.len() >= 10 {
                    (std::mem::take(&mut *c), std::mem::take(&mut *t))
                } else {
                    (vec![], vec![])
                }
            };
            if !control.is_empty() && !test.is_empty() {
                let significant =
                    crate::engines::self_improvement::AbTestManager::check_significance(
                        &control, &test,
                    );
                let control_mean = control.iter().sum::<f64>() / control.len() as f64;
                let test_mean = test.iter().sum::<f64>() / test.len() as f64;
                let adopted = significant && test_mean > control_mean;
                let report = crate::engines::self_improvement::AbTestReport {
                    hypothesis_id: "critique_protocol".to_string(),
                    control_mean,
                    test_mean,
                    effect_size: (test_mean - control_mean).abs(),
                    significant,
                    adopted,
                    confidence_interval: (control_mean.max(0.0), test_mean.min(1.0)),
                };
                let mut adjuster = self.runtime_adjuster.lock().await;
                if adjuster.apply_if_significant(
                    "critique_always",
                    if adopted { 1.0 } else { 0.0 },
                    &report,
                    "",
                ) {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "[A/B ADOPTED] Critique protocol adopted (test={:.2} vs control={:.2})\n",
                            test_mean, control_mean
                        ),
                    })
                    .await?;
                }
            }
        }

        // Final: commit the turn under the sigma lock.
        let result = self
            .commit_turn(
                &sigma_lock,
                changes,
                turn_outcome,
                &winner_id,
                &final_response,
                &prompt_arc,
                latency_ms,
                nash_weight_updates,
                stall_risk,
            )
            .await;

        // Clear planning hints only after a successful commit so a failed turn
        // re-injects the same hints on the next attempt.
        if result.is_ok() {
            self.pending_planning_hints.lock().await.clear();
        }

        result
    }

    async fn process_proposed_artifacts(
        &self,
        proposed: HashMap<String, (String, String)>,
        snapshot: &BTreeMap<String, Arc<Artifact>>,
    ) -> Result<ArtifactProcessOutcome> {
        let current_sigma_snap = ConversationState {
            artifacts: snapshot.clone(),
            ..Default::default()
        };
        let all_names: Vec<String> = snapshot.keys().cloned().collect();
        let mut changes = Vec::new();
        let mut turn_outcome = TurnOutcome::Unknown;

        for (name, (lang, new_content)) in proposed {
            if let Err(e) = AstValidator::validate(&new_content, &lang) {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[diff] artifact \"{name}\" rejected: AST validation failed: {e}"
                    ),
                })
                .await?;
                return Ok(ArtifactProcessOutcome::Invalid);
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

            let delta = DiffEngine::generate_delta(&current.content, &new_content, current.version);

            let (p_fail, mc_confidence) = self
                .mc_runner
                .predict(current, &delta, 10)
                .await
                .unwrap_or((0.5, 0.0));
            if p_fail > 0.5 {
                return Ok(ArtifactProcessOutcome::Invalid);
            }
            let mc_variance = if mc_confidence > 0.0 {
                1.0 - mc_confidence
            } else {
                0.5
            };

            let new_metrics = QualityEngine::analyze_artifact(
                &Artifact {
                    content: new_content.clone(),
                    ..(**current).clone()
                },
                &all_names,
            );

            // --- Suggestion 8: Historical Regression Testing ---
            if let Some(gold) = self.gold_state.lock().await.as_ref() {
                let mut temp_sigma = ConversationState::default();
                temp_sigma.artifacts.insert(
                    name.clone(),
                    Arc::new(Artifact {
                        content: new_content.clone(),
                        metrics: new_metrics.clone(),
                        ..(**current).clone()
                    }),
                );
                let report = crate::engines::quality::RegressionDetector::detect(gold, &temp_sigma);
                if report.drift_score > 0.5 {
                    crate::log_warn!(self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[regression] detected significant quality drop in \"{name}\" (drift={:.2})\n", report.drift_score),
                    }).await, "Failed to emit regression warning");
                    return Ok(ArtifactProcessOutcome::Regressive);
                }
            }

            // --- Suggestion 7: RAG-Powered Evidence Anchoring ---
            if let Some(bridge) = self.memory_bridge.lock().await.store.as_ref() {
                let anchored = crate::engines::reasoning::ReasoningEngine::anchor_evidence_rag(
                    &new_content,
                    &current_sigma_snap,
                    bridge,
                )
                .await;
                let unanchored_claims: Vec<_> =
                    anchored.iter().filter(|c| c.confidence < 0.4).collect();
                if !unanchored_claims.is_empty() {
                    crate::log_warn!(
                        self.emit(StreamEvent::TokenReceived {
                            agent_id: "System".to_string(),
                            token: format!(
                                "[warning] {} claims in \"{name}\" lack evidence anchoring\n",
                                unanchored_claims.len()
                            ),
                        })
                        .await,
                        "Failed to emit anchoring warning"
                    );
                }
            }

            if RegressionDetector::is_regressive(&current.metrics, &new_metrics) {
                return Ok(ArtifactProcessOutcome::Regressive);
            }

            let mut final_metrics = new_metrics.clone();
            // Visual-fidelity scoring required GPU frame capture, which was never
            // wired up (no window/event loop), so it always evaluated to 0.0.
            final_metrics.visual_fidelity = 0.0;
            final_metrics.health_score *= 1.0 - (mc_variance * 0.3).min(0.15);

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
                    metrics: final_metrics.clone(),
                    skeleton: String::new(),
                },
                vec![
                    "ast_valid".to_string(),
                    "mc_safe".to_string(),
                    "quality_checked".to_string(),
                ],
            );

            if lang.to_lowercase() == "rust" || lang.to_lowercase() == "rs" {
                // --- Sovereign-Tier: Recursive Formal Verification ---
                let temp_art = Artifact {
                    name: name.clone(),
                    content: new_content.clone(),
                    language: lang.clone(),
                    version: current.version + 1,
                    history: vec![],
                    ast_versions: BTreeMap::new(),
                    proof_attachments: vec![],
                    metrics: new_metrics.clone(),
                    skeleton: String::new(),
                };
                match crate::engines::verification::InvariantChecker::verify_artifact(&temp_art)
                    .await
                {
                    Ok(Err(err)) => {
                        crate::log_warn!(
                            self.emit(StreamEvent::TokenReceived {
                                agent_id: "System".to_string(),
                                token: format!(
                                    "[verus] formal verification failed for \"{name}\":\n{err}\n"
                                ),
                            })
                            .await,
                            "Failed to emit verus error"
                        );
                        return Ok(ArtifactProcessOutcome::Ready(
                            vec![],
                            TurnOutcome::VerificationFailed,
                        ));
                    }
                    Err(e) => {
                        tracing::warn!("verus execution error: {:?}", e);
                    }
                    _ => {}
                }

                let sandbox_result = SandboxResult {
                    exit_code: 0,
                    stdout: new_content.clone(),
                    stderr: String::new(),
                };
                let tmp = std::env::temp_dir();
                match LinterGuard::check(
                    &sandbox_result,
                    tmp.to_str().unwrap_or("/tmp"),
                    self.nix_env.as_ref(),
                )
                .await
                {
                    Ok(report) if !report.passed => return Ok(ArtifactProcessOutcome::Invalid),
                    Err(e) => {
                        crate::log_warn!(
                            self.emit(StreamEvent::TokenReceived {
                                agent_id: "System".to_string(),
                                token: format!("[lint] check failed for \"{name}\": {e}\n"),
                            })
                            .await,
                            "Failed to emit lint error"
                        );
                        return Ok(ArtifactProcessOutcome::Invalid);
                    }
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

        Ok(ArtifactProcessOutcome::Ready(changes, turn_outcome))
    }

    /// Sub-phase of `commit_turn`: apply artifact changes to sigma, build the Turn, run quality
    /// scoring and memory feedback, then advance the turn counter.
    /// Returns `(turn, quality_score, certainty, surprise, current_i, artifact_snapshot)`.
    #[allow(clippy::too_many_arguments)]
    async fn apply_turn_to_state(
        &self,
        sigma_lock: &Mutex<ConversationState>,
        changes: Vec<PreparedArtifactChange>,
        turn_outcome: TurnOutcome,
        agent_id: &str,
        response: &str,
        prompt: &str,
        latency_ms: u64,
        nash_weight_updates: &BTreeMap<String, f64>,
        stall_risk: f64,
    ) -> Result<Option<(Turn, f64, f64, f64, u32, BTreeMap<String, Arc<Artifact>>)>> {
        if turn_outcome == TurnOutcome::VerificationFailed {
            let mut failures = self.verification_failures.lock().await;
            let count = failures.entry(agent_id.to_string()).or_insert(0);
            *count += 1;
            if *count > 2 {
                let mut sigma = sigma_lock.lock().await;
                let w = sigma
                    .agent_weights
                    .entry(agent_id.to_string())
                    .or_insert(1.0);
                *w = (*w * 0.5).max(0.0);
            }
        }

        let mut sigma = sigma_lock.lock().await;
        let current_i = sigma.iteration_index;
        let artifact_snapshot = sigma.artifacts.clone();

        let mut turn_diffs = Vec::new();
        for change in changes {
            let artifact_arc = sigma
                .artifacts
                .entry(change.name.clone())
                .or_insert_with(|| {
                    Arc::new(Artifact {
                        name: change.name.clone(),
                        language: change.lang.clone(),
                        content: String::new(),
                        version: 0,
                        history: vec![],
                        ast_versions: BTreeMap::new(),
                        proof_attachments: vec![],
                        metrics: ArtifactMetrics::default(),
                        skeleton: String::new(),
                    })
                });
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
                *node_p = kalman.update_adaptive(0.8, 1.0);
            }
            turn_diffs.push((change.name, change.delta));
        }

        let certainty = CertaintyAnalyzer::compute(response, 0.1);

        let surprise = {
            let mut se = self.surprise_engine.lock().await;
            se.record_prediction(agent_id, certainty);
            let s = se.compute_surprise(agent_id, turn_outcome);
            let current_w = sigma.agent_weights.get(agent_id).copied().unwrap_or(1.0);
            sigma.agent_weights.insert(
                agent_id.to_string(),
                se.calibrate_weight(agent_id, current_w),
            );
            s
        };
        if surprise > 0.5 && sigma.turns.len() >= 2 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[sandbox] High Surprise detected: {:.2}", surprise),
            })
            .await?;
        }

        let combined_diff_text: String = turn_diffs
            .iter()
            .map(|(_, d)| d.diff_text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let consistency_score = Some(ConsistencyScorer::score(response, &combined_diff_text));
        let dq_score = {
            let intell = self.intelligence.lock().await;
            intell.diff_quality_score(agent_id)
        };

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
            consistency_score,
            diff_quality_score: Some(dq_score),
            persona_disclosure: None,
        };
        let serialized = serde_json::to_vec(&turn)?;
        turn.signature = self.signer.sign(&serialized);

        // Transparency duty: attach a signed PersonaDisclosure to every named-agent turn.
        {
            let principal = self.principal.lock().await;
            if principal.constraints.require_persona_disclosure && agent_id != "System" {
                use sha2::Digest;
                let prompt_hash: [u8; 32] = sha2::Sha256::digest(prompt.as_bytes()).into();
                let mut disclosure = PersonaDisclosure {
                    turn_index: turn.index,
                    agent_id: agent_id.to_string(),
                    persona_name: agent_id.to_string(),
                    system_prompt_hash: prompt_hash,
                    signature: vec![],
                };
                self.signer.sign_persona_disclosure(&mut disclosure);
                let pid = principal.id.to_string();
                let sid = sigma.session_id.clone();
                turn.persona_disclosure = Some(disclosure.clone());
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: pid,
                        event: FiduciaryDutyEvent::PersonaDisclosed(disclosure),
                        session_id: sid,
                        timestamp: ConversationState::now(),
                    })
                    .await,
                    "fiduciary persona signal emit failed"
                );
            }
        }

        let quality_score = {
            let base = {
                let obs = self.observer.lock().await;
                QualityScorer::score_with_context(&turn, &sigma.session_id, Some((&obs, agent_id)))
            };
            let surprise_penalty = (surprise - 0.5).max(0.0) * 0.6;
            let artifact_health = if !sigma.artifacts.is_empty() {
                sigma
                    .artifacts
                    .values()
                    .map(|a| a.metrics.health_score)
                    .sum::<f64>()
                    / sigma.artifacts.len() as f64
            } else {
                1.0
            };
            ((base - surprise_penalty) * artifact_health).max(0.0)
        };

        {
            let avg_quality = if !sigma.turns.is_empty() {
                let sum: f64 = sigma.turns.iter().filter_map(|t| t.certainty).sum();
                let count = sigma
                    .turns
                    .iter()
                    .filter(|t| t.certainty.is_some())
                    .count()
                    .max(1);
                sum / count as f64
            } else {
                0.5
            };
            let mut bridge = self.memory_bridge.lock().await;
            bridge.record_recall_feedback(current_i, quality_score - avg_quality);
            bridge.update_ranker(turn_outcome);
        }

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
                    skips.insert(alert.agent_id.clone(), sigma.iteration_index + 2);
                }
            }
        }

        {
            let mut coll = self.collective.lock().await;
            coll.update_specialization(&turn);
        }

        ComputeManager::manage_budget(
            &mut sigma,
            CostEntry {
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
            },
        );

        if sigma.turns.len() >= MAX_SESSION_TURNS {
            return Err(anyhow::anyhow!(
                "Session turn limit ({}) exceeded",
                MAX_SESSION_TURNS
            ));
        }

        // Fiduciary gate: Care/Autonomy duty — block or signal based on certainty and autonomy level.
        {
            let principal = self.principal.lock().await;
            if let Some(certainty) = turn.certainty {
                const CARE_CERTAINTY_FLOOR: f64 = 0.55;
                // Critically low certainty: block commits for SemiAutonomous (Care duty).
                const CARE_CERTAINTY_CRITICAL: f64 = 0.30;
                if certainty < CARE_CERTAINTY_FLOOR {
                    let event = FiduciaryDutyEvent::CertaintyGateFired {
                        turn_index: turn.index,
                        certainty,
                        threshold: CARE_CERTAINTY_FLOOR,
                    };
                    let pid = principal.id.to_string();
                    let sid = sigma.session_id.clone();
                    crate::log_warn!(
                        self.emit(StreamEvent::FiduciarySignal {
                            principal_id: pid,
                            event,
                            session_id: sid,
                            timestamp: ConversationState::now(),
                        })
                        .await,
                        "fiduciary signal emit failed"
                    );
                    if certainty < CARE_CERTAINTY_CRITICAL
                        && principal.constraints.max_autonomy_level == AutonomyLevel::SemiAutonomous
                    {
                        return Ok(None);
                    }
                }
            }
            // Sync principal session_id to sigma on first turn.
            if sigma.principal_id.is_none() {
                sigma.principal_id = Some(principal.id.to_string());
            }
        }

        sigma.push_turn(turn.clone());
        if let Err(e) = self.turn_tx.send(turn.clone()) {
            tracing::debug!(err = %e, "turn broadcast: no active swarm subscribers");
        }
        sigma.iteration_index += 1;

        // Account duty: persist decision to ledger.
        if let Some(ref principal_id) = sigma.principal_id {
            let event = DecisionLedger::commit_turn(&turn, &sigma.state_hash, agent_id);
            if let Err(e) = DecisionLedger::persist(
                self.state_manager.db(),
                &sigma.session_id,
                principal_id,
                turn.index,
                &event,
            ) {
                tracing::warn!(err = %e, "decision ledger persist failed");
            } else {
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: principal_id.clone(),
                        event,
                        session_id: sigma.session_id.clone(),
                        timestamp: ConversationState::now(),
                    })
                    .await,
                    "fiduciary signal emit failed"
                );
            }
        }

        {
            let task_cat = turn.task_category.unwrap_or(TaskCategory::Research);
            let quality = RewardVector::from_turn(&turn).weighted_score(task_cat);

            // Update EMA of recent quality (Task 8: adaptive evolution rate)
            let new_ema = loop {
                let old_bits = self
                    .recent_quality_ema
                    .load(std::sync::atomic::Ordering::Acquire);
                let old_val = f64::from_bits(old_bits);
                let candidate = 0.8 * old_val + 0.2 * quality;
                match self.recent_quality_ema.compare_exchange_weak(
                    old_bits,
                    candidate.to_bits(),
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Relaxed,
                ) {
                    Ok(_) => break candidate,
                    Err(_) => continue,
                }
            };

            // Adaptive evolution interval: faster when stalling, slower when improving
            let evolve_interval = if new_ema > 0.7 {
                10u32
            } else if new_ema > 0.4 {
                5
            } else {
                2
            };

            let mut evolver = self.prompt_evolver.lock().await;
            // Feed back to the specific template that was rendered, not the agent name
            if let Some(tmpl_id) = self.last_rendered_template_id.lock().await.as_deref() {
                evolver.record_outcome(tmpl_id, quality);
            }
            if sigma.iteration_index % evolve_interval == 0 {
                evolver.evolve();
                let mut cache = self.template_cache.write().await;
                for tmpl in evolver.population.iter().take(3) {
                    cache.insert(format!("{:?}", tmpl.task_category), tmpl.clone());
                }
            }
        }

        if matches!(
            turn_outcome,
            TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
        ) && let Some(certainty) = turn.certainty
            && certainty > 0.7
        {
            let seed_cat = turn.task_category.unwrap_or(TaskCategory::Research);
            let mut evolver = self.prompt_evolver.lock().await;
            evolver.seed_from_successful_turn(prompt, seed_cat);
        }

        if matches!(
            turn_outcome,
            TurnOutcome::TestsPassed | TurnOutcome::Compiled
        ) {
            for (name, artifact) in &sigma.artifacts {
                if let Ok(improved) =
                    crate::engines::self_improvement::SelfCodeModifier::propose_improvement(
                        name,
                        &artifact.content,
                    )
                    && improved != artifact.content
                    && AstValidator::validate(&improved, &artifact.language).is_ok()
                    && let Err(e) = self.file_writer.write_artifact(name, &improved).await
                {
                    warn!(artifact = %name, error = %e, "self-improvement write failed");
                }
            }
        }

        let prev_hash = sigma.state_hash;
        sigma.state_hash = HashChain::compute(&sigma, &prev_hash)?;

        sigma.agent_weights = match turn.task_category {
            Some(cat) => {
                crate::engines::consensus::InfluenceWeightManager::calculate_weights_for_category(
                    &sigma, cat, 0.9,
                )
            }
            None => crate::engines::consensus::InfluenceWeightManager::calculate_weights(&sigma),
        }
        .into_iter()
        .collect();

        for (id, nash_score) in nash_weight_updates {
            let w = sigma.agent_weights.entry(id.clone()).or_insert(0.5);
            *w = *w * 0.9 + nash_score * 0.1;
        }
        {
            let current_turn = sigma.iteration_index;
            let mut skips = self.skip_until.lock().await;
            for (id, &w) in &sigma.agent_weights {
                if w < 0.1 {
                    let until = skips.entry(id.clone()).or_insert(0);
                    *until = (*until).max(current_turn + 2);
                }
            }
        }

        if stall_risk > 0.6 {
            let mut coll = self.collective.lock().await;
            if coll.meta_optimizer.select_best(TaskCategory::Research)
                == MetaStrategy::DirectImplementation
            {
                coll.meta_optimizer
                    .record(MetaStrategy::DebateAndCritique, 0.6);
                tracing::info!(
                    stall_risk,
                    "stall detected: switching meta-strategy to DebateAndCritique"
                );
            }
        }

        let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
        let measurement = if response.contains("OPTIMAL") || response.contains("CONVERGED") {
            1.0
        } else {
            certainty * 0.8
        };
        let next_p = KalmanConvergence::new(current_p).update_adaptive(measurement, certainty);
        self.completion_probability
            .store(next_p.to_bits(), Ordering::Release);
        sigma.completion_probability = next_p;
        tracing::info!(
            turn = turn.index,
            agent = %agent_id,
            outcome = ?turn.outcome,
            response_len = turn.content.len(),
            convergence = next_p,
            "turn committed"
        );
        self.emit(StreamEvent::ConvergenceUpdated {
            p: next_p,
            certainty,
            agent_weights: sigma
                .agent_weights
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        })
        .await?;

        let mode_transition: Option<(String, String, String)> = {
            let mode_name = sigma.mode_library.current_name().to_string();

            // 1. Confidence-drop: rapid certainty decline triggers adversarial challenge
            if let Some(prev_turn) = sigma.turns.iter().rev().nth(1) {
                let prev_cert = prev_turn.certainty.unwrap_or(0.5);
                let curr_cert = turn.certainty.unwrap_or(0.5);
                if prev_cert - curr_cert > 0.3 && mode_name != "StressTest" {
                    sigma.mode_library.switch_to_name("StressTest");
                    tracing::info!(prev_cert, curr_cert, "confidence drop detected, switching to StressTest");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, format!("confidence drop {:.2} -> {:.2}, switching to StressTest", prev_cert, curr_cert)))
                } else {
                    None
                }
            } else {
                None
            }
            // 2. High-convergence: finalize when convergence probability is high
            .or_else(|| {
                if sigma.completion_probability > 0.85 && mode_name != "Convergence" {
                    sigma.mode_library.switch_to_name("Convergence");
                    tracing::info!(convergence = %sigma.completion_probability, "high convergence, switching to Convergence to finalize");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, format!("convergence {:.2} > 0.85, switching to Convergence", sigma.completion_probability)))
                } else {
                    None
                }
            })
            // 3. Oscillation: repeated content hashes indicate stuck loop
            .or_else(|| {
                if stall_risk > 0.6 && mode_name != "Socratic" && mode_name != "Generative" {
                    sigma.mode_library.switch_to_name("Socratic");
                    tracing::info!(stall_risk, "oscillation detected, switching to Socratic");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, format!("oscillation (stall_risk {:.2}), switching to Socratic", stall_risk)))
                } else {
                    None
                }
            })
            // 4. Mode return: after 2+ turns in a non-default mode, return if certainty recovers
            .or_else(|| {
                let turns_in_mode = sigma.turns.iter().rev()
                    .take_while(|t| t.model_id != "User")
                    .count();
                if turns_in_mode >= 2 && turn.certainty.unwrap_or(0.0) > 0.6
                    && mode_name != "Convergence"
                {
                    sigma.mode_library.switch_to_name("Convergence");
                    tracing::info!("certainty recovered, returning to Convergence");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, "certainty recovered above 0.6, returning to Convergence".to_string()))
                } else {
                    None
                }
            })
            // 5. Original stall detection: 3 consecutive low-certainty turns in Convergence
            .or_else(|| {
                let recent_stall = sigma.turns.iter().rev().take(3)
                    .filter(|t| t.certainty.unwrap_or(1.0) < 0.3 && t.model_id != "User")
                    .count() >= 3;
                if recent_stall && mode_name == "Convergence" {
                    sigma.mode_library.switch_to_name("Generative");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, "3 consecutive low-certainty turns, switching to Generative".to_string()))
                } else {
                    None
                }
            })
        };
        if let Some((old_name, new_name, reason)) = mode_transition {
            drop(sigma);
            self.emit(StreamEvent::ModeTransition {
                from_name: old_name,
                to_name: new_name.clone(),
                reason,
                synthesized: false,
            })
            .await?;
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[MODE → {}]\n", new_name),
            })
            .await?;
            sigma = sigma_lock.lock().await;
            sigma.mode_active_turns = 0;
        }

        // Clear novel_signal now that it has been consumed by this turn's prompt.
        sigma.novel_signal = None;

        // Surprise amplification: compute novel signal for next turn.
        if sigma.mode_library.current().surprise_handling
            == crate::types::mode::SurpriseHandling::Amplify
        {
            let mut scorer = crate::engines::novelty::NoveltyScorer::new();
            for t in sigma.turns.iter().rev().skip(1).take(5) {
                scorer.absorb(&t.content);
            }
            let novel = scorer.top_novel_sentences(response, 1);
            if let Some((sentence, score)) = novel.into_iter().next()
                && score > 0.3
            {
                sigma.novel_signal = Some(sentence);
            }
        }

        // Rejection loop: when OPTIMAL fires in RejectionLoop mode, inject rejection prompt.
        if response.contains("OPTIMAL")
            && sigma.mode_library.current().loop_structure
                == crate::types::mode::LoopStructure::RejectionLoop
            && !sigma.rejection_loop_active
        {
            sigma.rejection_loop_active = true;
            sigma.novel_signal = Some(
                "REJECTION TURN: The swarm has reached OPTIMAL. Your task now: reject the entire frame. \
                 What fundamental assumption is wrong? What would a structurally different approach look like? \
                 If you find a genuine new frame, begin with REJECT_FRAME: [description]. \
                 If you cannot find one, respond with FRAME_EXHAUSTED.".to_string()
            );
        } else if sigma.rejection_loop_active {
            if response.contains("REJECT_FRAME:") {
                let new_seed = response
                    .find("REJECT_FRAME:")
                    .map(|i| {
                        response[i + 13..]
                            .lines()
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string()
                    })
                    .unwrap_or_default();
                sigma.rejection_loop_active = false;
                sigma.novel_signal = if new_seed.is_empty() {
                    None
                } else {
                    Some(new_seed)
                };
                let old_name = sigma.mode_library.current_name().to_string();
                sigma.mode_library.switch_to_name("Generative");
                let new_name = sigma.mode_library.current_name().to_string();
                sigma.mode_active_turns = 0;
                drop(sigma);
                self.emit(StreamEvent::ModeTransition {
                    from_name: old_name,
                    to_name: new_name.clone(),
                    reason: "Rejection frame found — switching to Generative".to_string(),
                    synthesized: false,
                })
                .await?;
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[MODE → {}]\n", new_name),
                })
                .await?;
                sigma = sigma_lock.lock().await;
            } else if response.contains("FRAME_EXHAUSTED") {
                sigma.rejection_loop_active = false;
            }
        }

        // Mode synthesis: track active turns and inject synthesis prompt when stalled.
        {
            let avg_certainty = if !sigma.turns.is_empty() {
                let sum: f64 = sigma
                    .turns
                    .iter()
                    .rev()
                    .take(6)
                    .filter_map(|t| t.certainty)
                    .sum();
                let count = sigma
                    .turns
                    .iter()
                    .rev()
                    .take(6)
                    .filter(|t| t.certainty.is_some())
                    .count()
                    .max(1);
                sum / count as f64
            } else {
                1.0
            };
            sigma.mode_active_turns = sigma.mode_active_turns.saturating_add(1);
            if sigma.mode_active_turns >= 6 && avg_certainty < 0.4 && sigma.novel_signal.is_none() {
                let current_mode_name = sigma.mode_library.current_name().to_string();
                let active_turns = sigma.mode_active_turns;
                sigma.novel_signal = Some(format!(
                    "[META-MODE-SYNTHESIS] The current mode \"{}\" has not produced progress in {} turns. \
                     Analyze the conversation and propose a new mode definition as JSON on a single code block:\n\
                     ```json\n\
                     {{\"name\": \"...\", \"description\": \"...\", \
                     \"context_distribution\": \"Shared\", \
                     \"convergence_direction\": \"TowardAgreement\", \
                     \"surprise_handling\": \"Neutral\", \
                     \"termination\": {{\"OptimalSignal\": null}}, \
                     \"role_assignment\": \"Homogeneous\", \
                     \"loop_structure\": \"Linear\", \
                     \"prompt_prefix\": \"...\"}}\n\
                     ```\n\
                     Valid values: context_distribution: Shared|Divergent|RoleFiltered; \
                     convergence_direction: TowardAgreement|TowardDivergence|TowardTradeoffMap|TowardNovelty; \
                     surprise_handling: Amplify|Suppress|Neutral; \
                     termination: {{\"OptimalSignal\":null}} or {{\"Exhaustion\":{{\"max_turns\":N}}}} or {{\"RejectionCycles\":{{\"n\":N}}}}; \
                     role_assignment: Homogeneous|AdversarialPairs|Specialized; \
                     loop_structure: Linear|RejectionLoop|TreeSearch",
                    current_mode_name, active_turns
                ));
            }
        }

        // Try to parse a mode synthesized by the agent from this response.
        if let Some(new_mode) = crate::types::mode::ModeLibrary::try_parse_synthesized(
            response,
            format!(
                "Synthesized after {} turns in {}",
                sigma.mode_active_turns,
                sigma.mode_library.current_name()
            ),
        ) {
            let old_name = sigma.mode_library.current_name().to_string();
            let new_name = new_mode.name.clone();
            let idx = sigma.mode_library.upsert(new_mode);
            sigma.mode_library.switch_to_index(idx);
            sigma.mode_active_turns = 0;
            drop(sigma);
            self.emit(StreamEvent::ModeTransition {
                from_name: old_name,
                to_name: new_name.clone(),
                reason: "Mode synthesized by agent".to_string(),
                synthesized: true,
            })
            .await?;
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[MODE SYNTHESIZED → {}]\n", new_name),
            })
            .await?;
            sigma = sigma_lock.lock().await;
        }

        if let Err(e) = InvariantChecker::check_all(&sigma) {
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
            return Ok(None);
        }
        {
            let mut counters = self.rollback_counters.lock().await;
            counters.insert(agent_id.to_string(), 0);
        }
        self.state_manager.checkpoint_async(&sigma).await?;
        self.emit(StreamEvent::CheckpointWritten(current_i)).await?;

        Ok(Some((
            turn,
            quality_score,
            certainty,
            surprise,
            current_i,
            artifact_snapshot,
        )))
    }

    /// Sub-phase of `commit_turn`: file I/O, verification, post-commit state updates, and
    /// convergence reporting. `sigma` must NOT be held by the caller on entry.
    #[allow(clippy::too_many_arguments)]
    async fn finalize_committed_turn(
        &self,
        sigma_lock: &Mutex<ConversationState>,
        mut turn: Turn,
        quality_score: f64,
        next_p: f64,
        agent_id: &str,
        response: &str,
        latency_ms: u64,
    ) -> Result<bool> {
        // Snapshot what we need for I/O from sigma, then drop lock.
        let (io_artifacts, all_artifacts, turn_diffs) = {
            let sigma = sigma_lock.lock().await;
            let io: Vec<(String, Arc<Artifact>)> = turn
                .diffs
                .iter()
                .filter_map(|(name, _)| {
                    sigma
                        .artifacts
                        .get(name)
                        .map(|a| (name.clone(), Arc::clone(a)))
                })
                .collect();
            let all = sigma.artifacts.clone();
            let diffs = turn.diffs.clone();
            (io, all, diffs)
        };

        for (name, artifact) in &io_artifacts {
            match self.file_writer.write_artifact_with_proof(artifact).await {
                Ok(WriteOutcome::Written(path)) => {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[write] {}\n", path.display()),
                    })
                    .await?;
                }
                Ok(WriteOutcome::Skipped(_)) => {}
                Ok(WriteOutcome::VerificationFailed(stderr)) => {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "[write] {name}: verification failed, original restored\n{stderr}"
                        ),
                    })
                    .await?;
                }
                Err(e) => {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[write] error writing {name}: {e}\n"),
                    })
                    .await?;
                }
            }
        }

        let verification_results = if !io_artifacts.is_empty() {
            self.run_verification(&all_artifacts, &turn_diffs).await
        } else {
            vec![]
        };
        for (tool_name, output, passed) in &verification_results {
            let status = if *passed { "PASS" } else { "FAIL" };
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[verify] {} [{}]\n{}\n",
                    tool_name,
                    status,
                    Self::truncate_str(output, 500)
                ),
            })
            .await?;
        }
        if !verification_results.is_empty() && verification_results.iter().all(|(_, _, p)| *p) {
            turn.outcome = TurnOutcome::TestsPassed;
            let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
            self.completion_probability
                .store((current_p + 0.15).min(1.0).to_bits(), Ordering::Release);
        }

        // Re-acquire lock for remaining state updates.
        let mut sigma = sigma_lock.lock().await;
        sigma.last_verification = verification_results
            .iter()
            .map(|(name, output, passed)| (name.clone(), output.clone(), *passed))
            .collect();
        {
            let intell = self.intelligence.lock().await;
            intell.update_profile_with_latency(&turn, quality_score, latency_ms);
        }
        {
            let mut ctx = self.session_ctx.lock().await;
            ctx.record_turn(turn.outcome);
        }
        self.swarm.broadcast_turn(turn.clone())?;

        {
            let prev = sigma.state_hash;
            sigma.state_hash = HashChain::compute(&sigma, &prev)?;
        }

        if let Some(ref auditor_tx) = self.auditor_tx
            && !auditor_tx.is_closed()
            && let Err(e) = auditor_tx.send(sigma.clone()).await
        {
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
            let metrics = viz.compute_metrics(&sigma);
            drop(viz);
            self.emit(StreamEvent::GodViewUpdated {
                frame: metrics.frame,
                avg_certainty: metrics.avg_certainty,
                avg_surprise: metrics.avg_surprise,
                agent_count: metrics.agent_count,
            })
            .await?;
        }

        self.emit(StreamEvent::ArtifactsUpdated(
            sigma
                .artifacts
                .iter()
                .map(|(name, a)| crate::types::events::ArtifactSnapshot {
                    name: name.clone(),
                    skeleton: a.skeleton.clone(),
                    version: a.version,
                    diff_count: a.history.len(),
                })
                .collect(),
        ))
        .await?;

        self.emit(StreamEvent::TokenReceived {
            agent_id: "System".to_string(),
            token: format!(
                "\n[Turn Complete | P(C): {:.2} | Hash: {:02x?}]\n",
                next_p,
                &sigma.state_hash[..4]
            ),
        })
        .await?;
        self.emit(StreamEvent::TurnComplete(turn.clone())).await?;

        {
            let mut audit_rx: MutexGuard<'_, mpsc::UnboundedReceiver<AuditAlert>> =
                self.audit_rx.lock().await;
            while let Ok(alert) = audit_rx.try_recv() {
                // Only act on alerts for the current iteration; stale alerts are not indicative of tampering.
                if alert.iteration_index == sigma.iteration_index {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "[audit] Hash mismatch at iteration {}: expected {:02x?}, got {:02x?}\n",
                            alert.iteration_index, &alert.expected_hash[..4], &alert.actual_hash[..4]
                        ),
                    })
                    .await?;
                }
            }
        }

        {
            let session_id = sigma.session_id.clone();
            let compiled = matches!(
                turn.outcome,
                TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
            );
            let content_key = Self::truncate_str(response, 500).to_string();
            let preview = Self::truncate_str(response, 200).to_string();
            let metadata = serde_json::json!({"content": preview, "outcome": format!("{:?}", turn.outcome), "agent": agent_id}).to_string();
            let record = MemoryRecord {
                turn_id: turn.index,
                session_id: session_id.clone(),
                embedding: vec![],
                content_hash: content_key,
                timestamp: turn.timestamp,
                metadata_json: metadata,
                outcome: Some(OutcomeRecord {
                    compiled,
                    tests_passed: turn.outcome == TurnOutcome::TestsPassed,
                    quality_delta: 0.0,
                    was_rolled_back: false,
                    convergence_contribution: next_p,
                }),
                is_negative: false,
            };
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(session_id.clone());
            bridge.push_record(&session_id, record);
        }

        if next_p > 0.70 && next_p <= 0.85 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[convergence] p={next_p:.2}, moderate confidence, continuing refinement"
                ),
            })
            .await?;
        } else if next_p > 0.85 && next_p <= 0.95 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[convergence] p={next_p:.2}, high confidence, final polish"),
            })
            .await?;
        }

        let is_converged = next_p > 0.95;
        if is_converged {
            let eval = SelfImprovementEngine::evaluate_session(&sigma);
            let report = AnalyticsEngine::generate_report(&sigma);
            let exec_summary = ConvergenceReport::generate(&sigma);
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[self-improve] {:?} | [analytics] {:?} | [release] {}",
                    eval, report, exec_summary
                ),
            })
            .await?;
            if let Some(mortem) = PostMortemGenerator::generate(&sigma) {
                let bridge = self.memory_bridge.lock().await;
                if let Err(e) = bridge
                    .store_failure_lesson_async(&sigma.session_id, &mortem)
                    .await
                {
                    warn!(session = %sigma.session_id, error = %e, "failed to store post-mortem");
                }
            }
            let mut session_store =
                MemoryStore::new(&format!("/tmp/crosstalk-{}", sigma.session_id));
            session_store.init().await?;
            self.session_memory_map
                .lock()
                .await
                .insert(sigma.session_id.clone(), Arc::new(session_store));
        }

        Ok(is_converged)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, fields(agent = %agent_id))]
    async fn commit_turn(
        &self,
        sigma_lock: &Mutex<ConversationState>,
        changes: Vec<PreparedArtifactChange>,
        turn_outcome: TurnOutcome,
        agent_id: &str,
        response: &str,
        prompt: &str,
        latency_ms: u64,
        nash_weight_updates: BTreeMap<String, f64>,
        stall_risk: f64,
    ) -> Result<bool> {
        let Some((turn, quality_score, _certainty, _surprise, _current_i, _artifact_snapshot)) =
            self.apply_turn_to_state(
                sigma_lock,
                changes,
                turn_outcome,
                agent_id,
                response,
                prompt,
                latency_ms,
                &nash_weight_updates,
                stall_risk,
            )
            .await?
        else {
            return Ok(false);
        };

        // next_p was stored atomically in apply_turn_to_state; read it back.
        let next_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));

        self.finalize_committed_turn(
            sigma_lock,
            turn,
            quality_score,
            next_p,
            agent_id,
            response,
            latency_ms,
        )
        .await
    }

    async fn run_verification(
        &self,
        artifacts: &BTreeMap<String, Arc<Artifact>>,
        diffs: &[(String, crate::types::artifact::ArtifactDiff)],
    ) -> Vec<(String, String, bool)> {
        let workspace = &self.file_writer.root;
        let modified_names: std::collections::HashSet<&str> =
            diffs.iter().map(|(name, _)| name.as_str()).collect();
        let modified_artifacts: Vec<&Arc<Artifact>> = artifacts
            .values()
            .filter(|a| modified_names.contains(a.name.as_str()))
            .filter(|a| !a.name.contains(':'))
            .collect();
        let mut results = Vec::new();
        let tool_sets: Vec<(&str, serde_json::Value)> = {
            let mut tools = Vec::new();
            let has_rust = modified_artifacts
                .iter()
                .any(|a| matches!(a.language.to_lowercase().as_str(), "rust" | "rs"));
            if has_rust && workspace.join("Cargo.toml").exists() {
                tools.push(("cargo", serde_json::json!({"args": ["check"]})));
                tools.push((
                    "cargo",
                    serde_json::json!({"args": ["test", "--no-fail-fast"]}),
                ));
            }
            for a in modified_artifacts
                .iter()
                .filter(|a| a.language.to_lowercase() == "python" && a.name.ends_with(".py"))
            {
                tools.push((
                    "python3",
                    serde_json::json!({"args": ["-m", "py_compile", &a.name]}),
                ));
            }
            tools
        };
        for (cmd, args) in tool_sets {
            let label = format!(
                "{} {}",
                cmd,
                args["args"]
                    .as_array()
                    .map(|a| a
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" "))
                    .unwrap_or_default()
            );
            match self.tool_call("orchestrator", cmd, args).await {
                Ok(result) => {
                    let is_error = result
                        .get("isError")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let output = result.get("content")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                        .and_then(|e| e.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or_else(|| {
                            tracing::warn!(tool = %label, "tool result JSON missing content[0].text field");
                            ""
                        })
                        .to_string();
                    results.push((label, output, !is_error));
                }
                Err(e) => {
                    results.push((label, format!("{e}"), false));
                }
            }
        }
        results
    }

    async fn build_differential_prompt(&self, sigma: &ConversationState) -> String {
        let mut p = String::with_capacity(32_000);

        // Use evolved template as base if available, preferring select_for_agent over cache lookup
        let task_category = sigma
            .turns
            .last()
            .and_then(|t| t.task_category)
            .unwrap_or(TaskCategory::Research);
        let category_key = format!("{task_category:?}");

        // Try select_for_agent first; fall back to template_cache if population is empty
        let selected_tmpl: Option<crate::types::intelligence::PromptTemplate> = {
            let evolver = self.prompt_evolver.lock().await;
            let obs = self.observer.lock().await;
            if !evolver.population.is_empty() {
                let first_agent = self.agents.first().map(|a| a.name()).unwrap_or("unknown");
                evolver
                    .select_for_agent(first_agent, &obs.elo_ratings)
                    .cloned()
            } else {
                drop(evolver);
                drop(obs);
                let cache = self.template_cache.read().await;
                cache.get(&category_key).cloned()
            }
        };

        if let Some(tmpl) = selected_tmpl {
            let vars = std::collections::BTreeMap::from([
                ("session_id".to_string(), sigma.session_id.clone()),
                ("turn_index".to_string(), sigma.iteration_index.to_string()),
            ]);
            if let Ok(rendered) = tmpl.render(&vars) {
                // Record which template was used so the post-turn block can feed back quality
                *self.last_rendered_template_id.lock().await = Some(tmpl.id.clone());
                // Keep cache up to date for fallback on subsequent turns
                self.template_cache.write().await.insert(category_key, tmpl);
                p.push_str(&rendered);
                p.push('\n');
            }
        } else {
            *self.last_rendered_template_id.lock().await = None;
        }

        // Inject prior session lessons on the first turn of a new session (Task 7)
        if sigma.iteration_index <= 1 {
            let lessons = self.prior_lessons.lock().await;
            if !lessons.is_empty() {
                p.push_str("\n[PRIOR SESSION CONTEXT]:\n");
                for lesson in lessons.iter().take(2) {
                    crate::log_warn!(
                        writeln!(
                            p,
                            "- Session summary: \"{}\"\n  Outcome: {} | Winner: {} | Turns: {} | Topologies: {}",
                            lesson.task_summary,
                            lesson.final_outcome,
                            lesson.winning_model,
                            lesson.turn_count,
                            lesson.topology_sequence.join(" → ")
                        ),
                        "Failed to write prior lesson to prompt"
                    );
                }
                p.push('\n');
            }
        }

        if p.is_empty() {
            p.push_str(&format!("Project Context: {}\n\n", sigma.session_id));
        }

        if let Some(last_turn) = sigma.turns.last().filter(|t| t.model_id != "User") {
            crate::log_warn!(
                writeln!(
                    p,
                    "[ITERATION {}/prior] Prior consensus (turn {}, convergence {:.0}%):\n{}\n\nBuild on this analysis. Add depth, fix gaps, and refine. Do NOT repeat the same points verbatim.\n",
                    sigma.iteration_index,
                    last_turn.index,
                    sigma.completion_probability * 100.0,
                    Self::truncate_str(&last_turn.content, 2000),
                ),
                "Failed to write prior turn summary"
            );
        }

        p.push_str("Artifacts (Semantic Skeleton + Active Nodes):\n");

        let artifact_count = sigma.artifacts.len();
        let total_budget: usize = if artifact_count > 10 {
            12_000
        } else if artifact_count > 5 {
            18_000
        } else {
            30_000
        };
        let overhead = 2_000;
        let artifact_budget = if artifact_count == 0 {
            total_budget
        } else {
            (total_budget - overhead) / artifact_count
        };

        for artifact in sigma.artifacts.values() {
            p.push_str(&format!(
                "--- Artifact: {} [v{}] ({}) ---\n",
                artifact.name, artifact.version, artifact.language
            ));
            if artifact.version == 0 {
                let content = &artifact.content;
                if !artifact.skeleton.is_empty() && content.len() > artifact_budget {
                    p.push_str("Skeleton:\n");
                    p.push_str(&artifact.skeleton);
                    let remaining = artifact_budget.saturating_sub(artifact.skeleton.len());
                    if remaining > 200 {
                        crate::log_warn!(
                            writeln!(
                                p,
                                "\nKey excerpt ({} of {} chars):",
                                remaining,
                                content.len()
                            ),
                            "Failed to write key excerpt header"
                        );
                        p.push_str(&content[..remaining.min(content.len())]);
                    }
                } else if content.len() <= artifact_budget {
                    p.push_str("Full Content:\n");
                    p.push_str(content);
                } else {
                    p.push_str("Content (truncated):\n");
                    p.push_str(&content[..artifact_budget.min(content.len())]);
                    crate::log_warn!(
                        writeln!(
                            p,
                            "\n... ({} more chars)",
                            content.len().saturating_sub(artifact_budget)
                        ),
                        "Failed to write truncation notice"
                    );
                }
            } else {
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
            let questions = if signals.questions.is_empty() {
                String::new()
            } else {
                format!(
                    " open_questions=[{}]",
                    signals
                        .questions
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            };
            let code_count = signals.code_blocks.len();
            let code_tag = if code_count > 0 {
                format!(" code_blocks={code_count}")
            } else {
                String::new()
            };
            let certainty_tag = t
                .certainty
                .map(|c| format!(" certainty={c:.2}"))
                .unwrap_or_default();
            crate::log_warn!(
                writeln!(
                    p,
                    "Turn {} by {} ({}){}{}{}{}{}: {}",
                    t.index,
                    t.model_id,
                    outcome_tag,
                    certainty_tag,
                    decisions,
                    problems,
                    questions,
                    code_tag,
                    Self::truncate_str(&t.content, 150),
                ),
                "Failed to write history summary"
            );
        }
        if !sigma.last_verification.is_empty() {
            p.push_str("\nVerification Results from Last Turn:\n");
            for (tool, output, passed) in &sigma.last_verification {
                let status = if *passed { "PASS" } else { "FAIL" };
                let snippet = Self::truncate_str(output, 300);
                crate::log_warn!(
                    writeln!(p, "  {} [{}]: {}", tool, status, snippet),
                    "Failed to write verification result"
                );
            }
        }

        if !sigma.last_tool_outputs.is_empty() {
            p.push_str("\nTool Results from Previous Turn:\n");
            for (name, output) in &sigma.last_tool_outputs {
                crate::log_warn!(
                    writeln!(p, "  [{}]: {}", name, Self::truncate_str(output, 500)),
                    "Failed to write tool result"
                );
            }
        }

        if sigma.mode_library.current().convergence_direction
            == crate::types::mode::ConvergenceDirection::TowardAgreement
        {
            p.push_str(
                "\n[INSTRUCTIONS]\n\
                 1. Address any open questions or problems from prior turns before proposing new changes.\n\
                 2. Ground every claim in evidence: cite specific artifact lines, prior turn numbers, or test results.\n\
                 3. Use ```lang:filename to propose code changes. Include the COMPLETE file content.\n\
                 4. When your analysis is complete and all issues are resolved, tag with 'OPTIMAL'.\n\
                 5. Prefer precise, verifiable statements over vague assertions.\n"
            );
        } else {
            p.push_str(
                "\n[INSTRUCTIONS]\n\
                 1. Explore divergent approaches. Challenge assumptions from prior turns.\n\
                 2. Use ```lang:filename to propose code changes.\n\
                 3. Surface disagreements explicitly rather than silently accepting prior consensus.\n"
            );
        }
        let mode_prefix = sigma.mode_library.current().prompt_prefix.clone();
        let base = format!("{}\n\n{}", mode_prefix, p);
        if let Some(ref signal) = sigma.novel_signal {
            format!(
                "[NOVEL SIGNAL — build on this, do not ignore it]\n{}\n\n{}",
                signal, base
            )
        } else {
            base
        }
    }

    fn divergent_context_for_role(
        role: &str,
        artifacts: &std::collections::HashMap<
            String,
            std::sync::Arc<crate::types::artifact::Artifact>,
        >,
    ) -> String {
        if artifacts.is_empty() {
            return String::new();
        }
        let focus = match role {
            "Skeptic" | "StressTest" => "the weakest claims and most uncertain statements",
            "Architect" | "Generative" => {
                "the overall structure and what is missing or could be extended"
            }
            "Verifier" => "formal definitions, theorems, and proof sketches",
            "Historian" => "references to prior work and historical context",
            r if r.contains("Devil") => "the assumptions that the authors take for granted",
            _ => "the most important sections for your analysis",
        };
        format!(
            "\n\n[FOCUS] Your divergent context assignment: concentrate on {} in the provided documents. Other agents are focusing on different aspects.",
            focus
        )
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

        // Colon separator: lang:name or lang:name:linenum
        if let Some(pos) = hint.find(':') {
            let l = hint[..pos].trim().trim_matches('"').to_string();
            let mut n = hint[pos + 1..].trim().to_string();
            // Strip trailing :suffix after filename (e.g. "file.py:71" or "file.py:ClassName.method")
            while let Some(colon) = n.rfind(':') {
                let suffix = &n[colon + 1..];
                if suffix.chars().all(|c| c.is_ascii_digit())
                    || !suffix.contains('.')
                    || suffix.starts_with(|c: char| c.is_uppercase())
                {
                    n.truncate(colon);
                } else {
                    break;
                }
            }
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
        if candidate.contains('.')
            && !candidate.contains(' ')
            && !candidate.contains("..")
            && !candidate.is_empty()
            && candidate.len() < 200
        {
            Some(
                candidate
                    .trim_start_matches("./")
                    .trim_start_matches('/')
                    .to_string(),
            )
        } else {
            None
        }
    }

    /// Parse `[TOOL: name(args)]` directives from an agent response.
    /// Returns a vec of `(tool_name, raw_args)` pairs.
    fn parse_tool_directives(response: &str) -> Vec<(String, String)> {
        let mut directives = Vec::new();
        for line in response.lines() {
            let line = line.trim();
            if let Some(rest) = line
                .strip_prefix("[TOOL:")
                .and_then(|s| s.strip_suffix(']'))
            {
                let rest = rest.trim();
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim().to_string();
                    let args = rest[paren + 1..].trim_end_matches(')').trim().to_string();
                    if !name.is_empty() {
                        directives.push((name, args));
                    }
                }
            }
        }
        directives
    }

    /// Execute a parsed tool directive, returning the tool output as a string.
    /// Only `memory_query` and a whitelist of safe `shell_exec` commands are allowed.
    async fn execute_tool_directive(
        &self,
        tool_name: &str,
        args: &str,
        sigma_lock: &Arc<Mutex<ConversationState>>,
    ) -> String {
        match tool_name {
            "memory_query" => {
                let (sid, turn_idx) = {
                    let s = sigma_lock.lock().await;
                    (s.session_id.clone(), s.iteration_index)
                };
                let summary = {
                    let mut bridge = self.memory_bridge.lock().await;
                    bridge
                        .recall_relevant_summary(&sid, args, 3, turn_idx)
                        .await
                        .unwrap_or_default()
                };
                if summary.is_empty() {
                    "[memory_query] No results found.".to_string()
                } else {
                    format!("[memory_query results]:\n{summary}")
                }
            }
            "shell_exec" => {
                const ALLOWED_PREFIXES: &[&str] = &[
                    "git log",
                    "git status",
                    "git diff",
                    "git show",
                    "cargo check",
                    "cargo test",
                    "cargo clippy",
                    "ls ",
                    "cat ",
                    "head ",
                    "tail ",
                    "wc ",
                    "grep ",
                    "find ",
                    "diff ",
                    "file ",
                ];
                let trimmed = args.trim();
                if !ALLOWED_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
                    return format!("[shell_exec] Command not in whitelist: {trimmed}");
                }
                const INJECTION_CHARS: &[char] = &[';', '|', '&', '$', '`', '>', '<', '(', ')'];
                if trimmed.contains(INJECTION_CHARS) {
                    return format!(
                        "[shell_exec] Command contains disallowed characters: {trimmed}"
                    );
                }
                let cwd = self.file_writer.root.as_path();
                match tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    tokio::process::Command::new("sh")
                        .arg("-c")
                        .arg(trimmed)
                        .current_dir(cwd)
                        .output(),
                )
                .await
                {
                    Ok(Ok(out)) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let truncated = stdout.chars().take(2000).collect::<String>();
                        if !stderr.is_empty() {
                            format!(
                                "[shell_exec output]:\n{truncated}\n[stderr]: {}",
                                stderr.chars().take(500).collect::<String>()
                            )
                        } else {
                            format!("[shell_exec output]:\n{truncated}")
                        }
                    }
                    Ok(Err(e)) => format!("[shell_exec] Error: {e}"),
                    Err(_) => "[shell_exec] Timed out after 30s".to_string(),
                }
            }
            "write_file" => {
                let Some((path_str, content)) = args.split_once('\n') else {
                    return "[write_file] Expected: path\\ncontent".to_string();
                };
                let path_str = path_str.trim();
                if path_str.contains("..")
                    || path_str.starts_with('/')
                    || path_str.starts_with('\\')
                {
                    return "[write_file] Path traversal not allowed".to_string();
                }
                let target = self.file_writer.root.join(path_str);
                if let Some(parent) = target.parent()
                    && let Err(e) = tokio::fs::create_dir_all(parent).await
                {
                    return format!("[write_file] Cannot create directory: {e}");
                }
                let canonical_root = match self.file_writer.root.canonicalize() {
                    Ok(r) => r,
                    Err(e) => return format!("[write_file] Cannot resolve workspace: {e}"),
                };
                match tokio::fs::write(&target, content).await {
                    Ok(()) => {
                        if let Ok(canonical_target) = target.canonicalize()
                            && !canonical_target.starts_with(&canonical_root)
                        {
                            if let Err(e) = tokio::fs::remove_file(&target).await {
                                tracing::error!(
                                    path = %target.display(),
                                    err = %e,
                                    "failed to remove workspace-escaping file; it may persist on disk"
                                );
                            }
                            return "[write_file] Path escapes workspace after resolution"
                                .to_string();
                        }
                        tracing::info!(path = %target.display(), bytes = content.len(), "file written via tool directive");
                        let git_msg = Self::git_stage_file(&self.file_writer.root, &target).await;
                        format!(
                            "[write_file] Wrote {} ({} bytes){}",
                            target.display(),
                            content.len(),
                            git_msg
                        )
                    }
                    Err(e) => format!("[write_file] Error: {e}"),
                }
            }
            "read_file" => {
                let path_str = args.trim();
                if path_str.contains("..")
                    || path_str.starts_with('/')
                    || path_str.starts_with('\\')
                {
                    return "[read_file] Path traversal not allowed".to_string();
                }
                let target = self.file_writer.root.join(path_str);
                if let Ok(canonical) = target.canonicalize()
                    && let Ok(canonical_root) = self.file_writer.root.canonicalize()
                    && !canonical.starts_with(&canonical_root)
                {
                    return "[read_file] Path escapes workspace".to_string();
                }
                match tokio::fs::read_to_string(&target).await {
                    Ok(content) => {
                        let truncated: String = content.chars().take(4000).collect();
                        format!("[read_file {}]:\n{truncated}", target.display())
                    }
                    Err(e) => format!("[read_file] Error: {e}"),
                }
            }
            other => format!("[TOOL] Unknown tool: {other}"),
        }
    }

    async fn git_stage_file(repo_root: &std::path::Path, file: &std::path::Path) -> String {
        let rel = file.strip_prefix(repo_root).unwrap_or(file);
        match tokio::process::Command::new("git")
            .args(["add", "--", &rel.display().to_string()])
            .current_dir(repo_root)
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                tracing::debug!(file = %rel.display(), "git staged");
                String::new()
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                format!(" [git add failed: {}]", stderr.trim())
            }
            Err(_) => String::new(),
        }
    }

    pub async fn git_commit_session(repo_root: &std::path::Path, session_id: &str, turn: u32) {
        let msg = format!("crosstalk: session {} turn {}", session_id, turn);
        let status = tokio::process::Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .current_dir(repo_root)
            .status()
            .await;
        let has_staged = matches!(status, Ok(s) if !s.success());
        if !has_staged {
            return;
        }
        match tokio::process::Command::new("git")
            .args(["commit", "-m", &msg])
            .current_dir(repo_root)
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                tracing::info!(session = session_id, turn, "git commit created");
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::warn!(session = session_id, err = %stderr.trim(), "git commit failed");
            }
            Err(e) => {
                tracing::warn!(session = session_id, err = %e, "git command failed");
            }
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
                if parts.len() < 2 {
                    i += 1;
                    continue;
                }
                let l = parts[1].trim().to_string();
                let n = if parts.len() >= 3 {
                    parts[2].trim().to_string()
                } else {
                    String::new()
                };
                (l, n)
            } else {
                let rest = trimmed.trim_start_matches('`').trim();
                if rest.is_empty() {
                    i += 1;
                    continue;
                }
                Self::parse_fence_hint(rest)
            };

            // Pre-fence hint: check the nearest non-empty line above for a filename
            if name.is_empty() {
                let hint = all_lines[..i]
                    .iter()
                    .rev()
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
            if i < all_lines.len() {
                i += 1;
            } // consume closing fence

            if content_lines.is_empty() {
                continue;
            }

            // First-line comment may carry the filename
            if name.is_empty()
                && let Some(fname) = Self::extract_comment_filename(content_lines[0])
            {
                name = fname;
                content_lines.remove(0);
                // drop optional blank separator line
                if content_lines
                    .first()
                    .map(|l| l.trim().is_empty())
                    .unwrap_or(false)
                {
                    content_lines.remove(0);
                }
            }

            // Infer lang from filename extension if still unknown
            if (lang.is_empty() || lang == "text" || lang == "plaintext")
                && !name.is_empty()
                && let Some(ext) = name.rsplit('.').next()
            {
                let inferred = Self::ext_to_lang(ext);
                if !inferred.is_empty() {
                    lang = inferred.to_string();
                }
            }

            if lang.is_empty() {
                continue;
            }

            // Synthesize a name if none found
            if name.is_empty() {
                unnamed_count += 1;
                name = format!("artifact_{}.{}", unnamed_count, Self::lang_to_ext(&lang));
            }

            // Normalize path separators
            let name = name
                .trim_start_matches("./")
                .trim_start_matches('/')
                .to_string();

            if name.contains("..") {
                i += 1;
                continue;
            }

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

    pub async fn finalize_session(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<()> {
        let sigma = sigma_lock.lock().await;
        let eval = SelfImprovementEngine::evaluate_session(&sigma);
        self.emit(StreamEvent::TokenReceived {
            agent_id: "System".to_string(),
            token: format!(
                "
[Self-Improvement] Session Evaluation: convergence_p={:.2}, failure_rate={:.2}
",
                eval.metrics.get("convergence_p").unwrap_or(&0.0),
                eval.metrics.get("failure_rate").unwrap_or(&0.0)
            ),
        })
        .await?;

        if let Some(pm) = PostMortemGenerator::generate(&sigma) {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "
[Self-Improvement] Post-Mortem: detected root cause {:?}
",
                    pm.root_cause
                ),
            })
            .await?;
        }

        {
            let intell = self.intelligence.lock().await;
            let templates_arc = intell.templates();
            let calibration_arc = intell.calibration();
            let mut templates = templates_arc.write().await;
            let mut calibration = calibration_arc.write().await;
            let mut learner = crate::engines::self_improvement::ContinuousLearner {
                prompt_library: &mut templates,
                calibration: &mut calibration,
            };
            learner.run(
                &sigma,
                PostMortemGenerator::generate(&sigma),
                0.5,
                sigma.completion_probability,
            );
        }

        // Persist Elo ratings for cross-session continuity.
        {
            let obs = self.observer.lock().await;
            let elo_json = obs.export_elo_ratings();
            let record = MemoryRecord {
                turn_id: 0,
                session_id: "elo_ratings".to_string(),
                embedding: vec![0.0; 64],
                content_hash: "elo_snapshot".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                metadata_json: elo_json,
                outcome: None,
                is_negative: false,
            };
            self.memory_store
                .sessions
                .entry("elo_ratings".to_string())
                .or_default()
                .push(record);
        }

        // Persist prompt evolver population for cross-session evolution.
        {
            let evolver = self.prompt_evolver.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("prompt_population".to_string())
                .or_default()
                .push(MemoryRecord {
                    turn_id: 0,
                    session_id: "prompt_population".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "prompt_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: evolver.export_state_json(),
                    outcome: None,
                    is_negative: false,
                });
        }

        // Persist topology scores for cross-session learning.
        {
            let topo = self.topology.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("topology_scores".to_string())
                .or_default()
                .push(MemoryRecord {
                    turn_id: 0,
                    session_id: "topology_scores".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "topology_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: topo.export_scores_json(),
                    outcome: None,
                    is_negative: false,
                });
        }

        // Persist collective agent profiles and meta-strategy outcomes.
        {
            let coll = self.collective.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("collective_profiles".to_string())
                .or_default()
                .push(MemoryRecord {
                    turn_id: 0,
                    session_id: "collective_profiles".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "collective_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: coll.export_state_json(),
                    outcome: None,
                    is_negative: false,
                });
        }

        // Persist memory ranker weights for cross-session recall tuning.
        {
            let bridge = self.memory_bridge.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("ranker_weights".to_string())
                .or_default()
                .push(MemoryRecord {
                    turn_id: 0,
                    session_id: "ranker_weights".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "ranker_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: bridge.export_ranker_weights_json(),
                    outcome: None,
                    is_negative: false,
                });
        }

        // Distill and persist a SessionLesson for future sessions.
        {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let task_summary = sigma
                .turns
                .first()
                .map(|t| t.content.chars().take(200).collect::<String>())
                .unwrap_or_default();
            let topo = self.topology.lock().await;
            let topology_sequence: Vec<String> = topo
                .history
                .iter()
                .map(|(_, t, _)| format!("{t:?}"))
                .collect();
            drop(topo);
            let final_outcome = if sigma.completion_probability > 0.7 {
                "succeeded"
            } else if sigma.completion_probability > 0.3 {
                "stalled"
            } else {
                "failed"
            }
            .to_string();
            let obs = self.observer.lock().await;
            let winning_model = obs
                .ranked_agents()
                .into_iter()
                .next()
                .map(|(id, _)| id)
                .unwrap_or_default();
            drop(obs);
            let quality_trajectory: Vec<f64> = sigma
                .turns
                .iter()
                .rev()
                .take(10)
                .map(|t| {
                    RewardVector::from_turn(t)
                        .weighted_score(t.task_category.unwrap_or(TaskCategory::Research))
                })
                .collect();
            let lesson = SessionLesson {
                task_summary,
                topology_sequence,
                final_outcome,
                winning_model,
                quality_trajectory,
                turn_count: sigma.iteration_index,
                timestamp: ts,
            };
            if let Ok(lesson_json) = serde_json::to_string(&lesson) {
                self.memory_store
                    .sessions
                    .entry("session_lessons".to_string())
                    .or_default()
                    .push(MemoryRecord {
                        turn_id: 0,
                        session_id: "session_lessons".to_string(),
                        embedding: vec![0.0; 64],
                        content_hash: "lesson_snapshot".to_string(),
                        timestamp: ts,
                        metadata_json: lesson_json,
                        outcome: None,
                        is_negative: false,
                    });
            }
        }

        // Enforce data retention policy (fiduciary duty).
        {
            let principal = self.principal.lock().await;
            if let Ok(Some(event)) = crate::engines::data_minimizer::DataMinimizer::enforce(
                self.state_manager.db(),
                &sigma.session_id,
                &principal.constraints,
            ) {
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: principal.id.to_string(),
                        event,
                        session_id: sigma.session_id.clone(),
                        timestamp: ConversationState::now(),
                    })
                    .await,
                    "data minimizer fiduciary signal failed"
                );
            }
        }

        Ok(())
    }

    pub fn get_completion_probability(&self) -> f64 {
        f64::from_bits(
            self.completion_probability
                .load(std::sync::atomic::Ordering::Acquire),
        )
    }
}
