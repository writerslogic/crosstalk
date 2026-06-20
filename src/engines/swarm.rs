use crate::types::consensus::MergeVote;
use crate::types::conversation::Turn;
use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub id: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub estimated_turns: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Idle,
    Processing,
    WaitingMerge,
    Merging,
    Error,
    Running,
    Complete,
    Failed,
}

pub struct SwarmController {
    pub nodes: Arc<DashMap<String, NodeStatus>>,
    pub node_tx: DashMap<String, mpsc::UnboundedSender<String>>,
    pub spawn_count: AtomicU32,
}

impl SwarmController {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(DashMap::new()),
            node_tx: DashMap::new(),
            spawn_count: AtomicU32::new(0),
        }
    }

    pub fn spawn_node(&self, id: &str, mut turn_rx: tokio::sync::broadcast::Receiver<Turn>) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.node_tx.insert(id.to_string(), tx);
        self.nodes.insert(id.to_string(), NodeStatus::Idle);

        let id_clone = id.to_string();
        let nodes_ref = Arc::clone(&self.nodes);

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(_task) = rx.recv() => {
                        nodes_ref.insert(id_clone.clone(), NodeStatus::Processing);
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        nodes_ref.insert(id_clone.clone(), NodeStatus::Idle);
                    }
                    result = turn_rx.recv() => {
                        match result {
                            Ok(_) => {
                                nodes_ref.insert(id_clone.clone(), NodeStatus::WaitingMerge);
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
        });
        self.spawn_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    pub async fn shutdown(&self) {
        for node in self.node_tx.iter() {
            if node.value().send("SHUTDOWN".to_string()).is_err() {
                tracing::warn!(node = %node.key(), "shutdown signal failed: receiver dropped");
            }
        }
    }

    pub fn broadcast_turn(&self, turn: Turn) -> Result<()> {
        for node in self.node_tx.iter() {
            if node
                .value()
                .send(format!("SYNC_TURN:{}", turn.index))
                .is_err()
            {
                tracing::warn!(node = %node.key(), turn = turn.index, "sync broadcast failed");
            }
        }
        Ok(())
    }
}

impl Default for SwarmController {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TaskDecomposer;
impl TaskDecomposer {
    pub fn decompose(description: &str, n_tracks: usize) -> Vec<SubTask> {
        if n_tracks == 0 {
            return vec![];
        }
        let lines: Vec<&str> = description
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        if lines.is_empty() {
            return vec![];
        }
        let chunk_size = (lines.len() as f64 / n_tracks as f64).ceil() as usize;
        lines
            .chunks(chunk_size.max(1))
            .enumerate()
            .map(|(i, chunk)| SubTask {
                id: format!("task-{}", i),
                description: chunk.join("\n"),
                dependencies: if i > 0 {
                    vec![format!("task-{}", i - 1)]
                } else {
                    vec![]
                },
                estimated_turns: chunk.len() as u32,
            })
            .collect()
    }
}

pub struct LeaderElection {
    pub node_id: Arc<str>,
    pub current_term: u64,
}

impl LeaderElection {
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: Arc::from(node_id),
            current_term: 0,
        }
    }

    pub fn elect_leader(nodes: &[String]) -> Option<String> {
        nodes.iter().min().cloned()
    }

    pub async fn run_election_cycle<N: RaftNetwork + Send + Sync>(
        &mut self,
        peers: &[Arc<N>],
        mut heartbeat_rx: mpsc::Receiver<()>,
    ) -> bool {
        self.current_term += 1;
        let term = self.current_term;
        let candidate_id = Arc::clone(&self.node_id);

        // Check for preempting heartbeat first
        if heartbeat_rx.try_recv().is_ok() {
            return false;
        }

        if peers.is_empty() {
            // No peers; only win if no heartbeat preempted
            return true;
        }

        let mut grants = 1u32; // vote for self
        let total = peers.len() as u32 + 1;

        for peer in peers {
            if let Ok(true) = peer.request_vote(term, Arc::clone(&candidate_id)).await {
                grants += 1;
            }
        }

        grants > total / 2
    }
}

#[allow(async_fn_in_trait)]
pub trait RaftNetwork {
    async fn request_vote(&self, term: u64, candidate_id: Arc<str>) -> Result<bool>;
}

// ── AgentAssigner ────────────────────────────────────────────────────────────

pub struct AgentAssigner;

impl AgentAssigner {
    pub fn assign(
        tasks: &[SubTask],
        capabilities: &HashMap<String, f64>,
    ) -> HashMap<String, String> {
        if capabilities.is_empty() {
            return HashMap::new();
        }
        let best_agent = capabilities
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        tasks
            .iter()
            .map(|t| (t.id.clone(), best_agent.clone()))
            .collect()
    }
}

// ── ConflictDetector ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictSeverity {
    Minor,
    Major,
}

#[derive(Debug, Clone)]
pub struct Conflict {
    pub artifact: String,
    pub severity: ConflictSeverity,
    pub proposed_by: String,
    pub committed_by: String,
}

pub struct ConflictDetector;

impl ConflictDetector {
    pub fn check(proposed: &[(&str, &str)], committed: &[(&str, &str)]) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        for &(p_art, p_agent) in proposed {
            for &(c_art, c_agent) in committed {
                if p_art == c_art {
                    conflicts.push(Conflict {
                        artifact: p_art.to_string(),
                        severity: ConflictSeverity::Major,
                        proposed_by: p_agent.to_string(),
                        committed_by: c_agent.to_string(),
                    });
                }
            }
        }
        conflicts
    }
}

// ── ProgressMonitor ──────────────────────────────────────────────────────────

pub struct ProgressReport {
    pub completion_ratio: f64,
    pub complete: usize,
    pub running: usize,
    pub failed: usize,
    pub waiting_merge: usize,
}

pub struct ProgressMonitor;

impl ProgressMonitor {
    pub fn check<K: std::hash::Hash + Eq>(nodes: &DashMap<K, NodeStatus>) -> ProgressReport {
        let total = nodes.len();
        if total == 0 {
            return ProgressReport {
                completion_ratio: 0.0,
                complete: 0,
                running: 0,
                failed: 0,
                waiting_merge: 0,
            };
        }
        let mut complete = 0;
        let mut running = 0;
        let mut failed = 0;
        let mut waiting_merge = 0;
        for entry in nodes.iter() {
            match *entry.value() {
                NodeStatus::Complete => complete += 1,
                NodeStatus::Running | NodeStatus::Processing => running += 1,
                NodeStatus::Failed | NodeStatus::Error => failed += 1,
                NodeStatus::WaitingMerge | NodeStatus::Merging => waiting_merge += 1,
                NodeStatus::Idle => {}
            }
        }
        ProgressReport {
            completion_ratio: complete as f64 / total as f64,
            complete,
            running,
            failed,
            waiting_merge,
        }
    }
}

// ── SwarmTelemetry ───────────────────────────────────────────────────────────

pub struct SwarmTelemetry {
    spawns: AtomicU32,
    merges: AtomicU32,
    conflicts: AtomicU32,
}

impl SwarmTelemetry {
    pub fn new() -> Self {
        Self {
            spawns: AtomicU32::new(0),
            merges: AtomicU32::new(0),
            conflicts: AtomicU32::new(0),
        }
    }
    pub fn record_spawn(&self) {
        self.spawns.fetch_add(1, AtomicOrdering::Relaxed);
    }
    pub fn record_merge(&self) {
        self.merges.fetch_add(1, AtomicOrdering::Relaxed);
    }
    pub fn record_conflict(&self) {
        self.conflicts.fetch_add(1, AtomicOrdering::Relaxed);
    }
    pub fn snapshot(&self) -> (u32, u32, u32) {
        (
            self.spawns.load(AtomicOrdering::Relaxed),
            self.merges.load(AtomicOrdering::Relaxed),
            self.conflicts.load(AtomicOrdering::Relaxed),
        )
    }
}

impl Default for SwarmTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

// ── GlobalMergeGate ──────────────────────────────────────────────────────────

pub struct GlobalMergeGate;

impl GlobalMergeGate {
    pub fn has_quorum(votes: &[MergeVote], total_nodes: usize) -> bool {
        if votes.is_empty() || total_nodes == 0 {
            return false;
        }
        let approvals = votes.iter().filter(|v| v.approve).count();
        approvals > total_nodes / 2
    }
}
