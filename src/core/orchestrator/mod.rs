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
            nix_env: nix_env
                .map(|env| std::collections::HashMap::from([("NIX_ENV".to_string(), env)])),
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

}

mod agents;
mod synthesis;
mod artifacts;
mod verification;
mod parsing;
mod lifecycle;
