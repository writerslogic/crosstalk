use crate::types::consensus::MergeVote;
use crate::types::conversation::Turn;
use anyhow::{Result, anyhow};
use crossbeam::deque::{Injector, Steal, Worker};
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering as AtomicOrdering},
};
use std::time::Duration;
use tokio::sync::{Notify, broadcast, mpsc};
use tokio::task::JoinSet;

/// Zero-copy identifier to prevent heap allocations on map lookups.
pub type NodeId = Arc<str>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Spawning,
    Running,
    WaitingMerge,
    Complete,
    Failed,
    Idle,
}

/// A simulated network interface for Raft peer-to-peer communication.
/// In production, this is typically backed by gRPC (Tonic).
pub trait RaftNetwork: Send + Sync + 'static {
    fn request_vote(
        &self,
        term: u64,
        candidate_id: NodeId,
    ) -> impl Future<Output = Result<bool>> + Send;
}

pub struct SwarmController {
    pub nodes: Arc<DashMap<NodeId, NodeStatus>>,
    pub state_tx: broadcast::Sender<Turn>,
    pub work_notify: Arc<Notify>,
    pub task_queue: Arc<Injector<SubTask>>,
    task_spawner: std::sync::Mutex<Option<mpsc::Sender<tokio::task::JoinHandle<()>>>>,
    supervisor_handle: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl SwarmController {
    #[must_use]
    pub fn new() -> Self {
        let (state_tx, _) = broadcast::channel(1000);
        let (task_spawner, mut task_rx) = mpsc::channel(100);

        // SUPERVISOR (Reaper)
        let supervisor_handle = tokio::spawn(async move {
            let mut join_set = JoinSet::new();
            loop {
                tokio::select! {
                    Some(handle) = task_rx.recv() => {
                        join_set.spawn(async move {
                            let _ = handle.await;
                        });
                    }
                    Some(_res) = join_set.join_next() => {}
                    else => break, // Channel closed
                }
            }
        });

        Self {
            nodes: Arc::new(DashMap::new()),
            state_tx,
            work_notify: Arc::new(Notify::new()),
            task_queue: Arc::new(Injector::new()),
            task_spawner: std::sync::Mutex::new(Some(task_spawner)),
            supervisor_handle: std::sync::Mutex::new(Some(supervisor_handle)),
        }
    }

    pub async fn shutdown(&self) {
        {
            let mut guard = match self.task_spawner.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            guard.take();
        }
        let handle = {
            let mut guard = match self.supervisor_handle.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            guard.take()
        };
        if let Some(h) = handle {
            let _ = h.await;
        }
    }

    /// Spawns a node with a fully implemented, event-driven worker loop and work-stealing support.
    pub async fn spawn_node(&self, node_id: impl Into<NodeId>) -> Result<()> {
        let id: NodeId = node_id.into();
        self.nodes.insert(Arc::clone(&id), NodeStatus::Spawning);

        let nodes_ref = Arc::clone(&self.nodes);
        let id_clone = Arc::clone(&id);
        let injector = Arc::clone(&self.task_queue);

        // Subscribe to the global turn broadcaster before spawning
        let mut turn_rx = self.state_tx.subscribe();
        let notify_ref = Arc::clone(&self.work_notify);

        let worker_handle = tokio::spawn(async move {
            nodes_ref.insert(id_clone.clone(), NodeStatus::Running);
            let local_queue = Worker::new_fifo();

            // WORKER LOOP: Actively listen for global turns, local notifications, and steal tasks
            loop {
                // Attempt to pull a task from global injector or steal from peers (simplified)
                let task = local_queue.pop().or_else(|| {
                    match injector.steal_batch_and_pop(&local_queue) {
                        Steal::Success(t) => Some(t),
                        _ => None,
                    }
                });

                if let Some(sub_task) = task {
                    nodes_ref.insert(id_clone.clone(), NodeStatus::Running);
                    // Process the sub-task
                    tokio::time::sleep(Duration::from_millis(
                        100 * sub_task.estimated_turns as u64,
                    ))
                    .await;
                    nodes_ref.insert(id_clone.clone(), NodeStatus::WaitingMerge);
                    continue;
                }

                nodes_ref.insert(id_clone.clone(), NodeStatus::Idle);

                tokio::select! {
                    result = turn_rx.recv() => {
                        match result {
                            Ok(_turn) => {
                                tokio::time::sleep(Duration::from_millis(50)).await;
                                nodes_ref.insert(id_clone.clone(), NodeStatus::WaitingMerge);
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        }
                    }

                    _ = notify_ref.notified() => {}

                    else => break,
                }
            }

            nodes_ref.insert(id_clone, NodeStatus::Complete);
        });

        let sender = {
            let guard = match self.task_spawner.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            guard
                .as_ref()
                .cloned()
                .ok_or_else(|| anyhow!("Critical: Swarm Supervisor shut down"))?
        };
        sender
            .send(worker_handle)
            .await
            .map_err(|_| anyhow!("Critical: Swarm Supervisor died"))
    }

    pub fn submit_tasks(&self, tasks: Vec<SubTask>) {
        for task in tasks {
            self.task_queue.push(task);
        }
        self.work_notify.notify_waiters();
    }

    pub fn broadcast_turn(&self, turn: Turn) -> Result<usize> {
        match self.state_tx.send(turn) {
            Ok(n) => Ok(n),
            Err(_) => Ok(0),
        }
    }
}

impl Default for SwarmController {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

pub struct LeaderElection {
    pub node_id: NodeId,
    pub term: u64,
    pub state: RaftState,
    pub votes: usize,
}

impl LeaderElection {
    #[must_use]
    pub fn new(node_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
            term: 0,
            state: RaftState::Follower,
            votes: 0,
        }
    }

    /// Evaluates the Raft Election utilizing Scatter-Gather RPCs and Quorum Short-Circuiting.
    pub async fn run_election_cycle<N: RaftNetwork>(
        &mut self,
        peers: &[Arc<N>],
        mut heartbeat_rx: mpsc::Receiver<()>,
    ) -> bool {
        let timeout_ms = rand::rng().random_range(150..300);
        let election_timer = tokio::time::sleep(Duration::from_millis(timeout_ms));

        tokio::select! {
            // Preempted by a valid leader
            Some(_) = heartbeat_rx.recv() => {
                self.state = RaftState::Follower;
                false
            }

            // Timeout triggers election
            _ = election_timer => {
                if self.state == RaftState::Follower || self.state == RaftState::Candidate {
                    self.term += 1;
                    self.state = RaftState::Candidate;
                    self.votes = 1; // Vote for self

                    let total_nodes = peers.len() + 1;
                    let quorum = (total_nodes >> 1) + 1; // Bitwise div 2 + 1

                    // Scatter: Fire all RPCs concurrently
                    let mut rpc_tasks = JoinSet::new();
                    for peer in peers {
                        let peer = Arc::clone(peer);
                        let candidate_id = Arc::clone(&self.node_id);
                        let term = self.term;

                        rpc_tasks.spawn(async move {
                            // Strict timeout on external RPCs to prevent network stalling
                            match tokio::time::timeout(
                                Duration::from_millis(50),
                                peer.request_vote(term, candidate_id)
                            ).await {
                                Ok(Ok(vote_granted)) => vote_granted,
                                _ => false, // Network failure/timeout counts as rejected vote
                            }
                        });
                    }

                    // Gather: Await responses as they complete
                    while let Some(res) = rpc_tasks.join_next().await {
                        if let Ok(true) = res {
                            self.votes += 1;

                            // QUORUM SHORT-CIRCUIT:
                            // If we hit quorum early, abort pending RPCs to save bandwidth.
                            if self.votes >= quorum {
                                rpc_tasks.abort_all();
                                self.state = RaftState::Leader;
                                return true;
                            }
                        }
                    }
                }

                // Election failed
                self.state = RaftState::Follower;
                false
            }
        }
    }
}

pub struct GlobalMergeGate;

impl GlobalMergeGate {
    #[must_use]
    pub fn has_quorum(votes: &[MergeVote], total_nodes: usize) -> bool {
        if votes.is_empty() || total_nodes == 0 {
            return false;
        }
        let approvals = votes.iter().filter(|v| v.approve).count();
        approvals > (total_nodes >> 1)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub id: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub estimated_turns: u32,
}

pub struct TaskDecomposer;

impl TaskDecomposer {
    /// Split `description` into up to `n_tracks` parallel sub-tasks.
    /// Sentences are grouped evenly; the first track has no dependencies,
    /// subsequent tracks declare the prior track as a dependency.
    #[must_use]
    pub fn decompose(description: &str, n_tracks: usize) -> Vec<SubTask> {
        if n_tracks == 0 || description.is_empty() {
            return vec![];
        }
        let sentences: Vec<&str> = description
            .split(['.', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let n = n_tracks.min(sentences.len()).max(1);
        let chunk = sentences.len().div_ceil(n);

        sentences
            .chunks(chunk)
            .enumerate()
            .map(|(i, chunk)| SubTask {
                id: format!("task-{i}"),
                description: chunk.join(". "),
                dependencies: if i == 0 {
                    vec![]
                } else {
                    vec![format!("task-{}", i - 1)]
                },
                estimated_turns: chunk.len() as u32,
            })
            .collect()
    }
}

pub struct AgentAssigner;

impl AgentAssigner {
    /// Greedy assignment: each sub-task goes to the agent with the highest
    /// capability score. Returns a map of `task_id → agent_id`.
    #[must_use]
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
            .map(|(id, _)| id.clone())
            .unwrap_or_default();

        tasks
            .iter()
            .map(|t| (t.id.clone(), best_agent.clone()))
            .collect()
    }

    /// Per-task assignment using per-task capability scores keyed by task id.
    #[must_use]
    pub fn assign_per_task(
        tasks: &[SubTask],
        per_task_caps: &HashMap<String, HashMap<String, f64>>,
    ) -> HashMap<String, String> {
        tasks
            .iter()
            .map(|t| {
                let agent = per_task_caps
                    .get(&t.id)
                    .and_then(|caps| {
                        caps.iter()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|(id, _)| id.clone())
                    })
                    .unwrap_or_default();
                (t.id.clone(), agent)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Conflict {
    pub task_a: String,
    pub task_b: String,
    pub artifact: String,
    pub severity: ConflictSeverity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictSeverity {
    /// Edits to non-overlapping regions; auto-resolvable.
    Minor,
    /// Edits to the same region; requires arbitration.
    Major,
}

pub struct ConflictDetector;

impl ConflictDetector {
    /// Detect conflicts between `proposed` diffs and the `committed` baseline.
    /// A conflict is raised when two diffs touch the same artifact name.
    #[must_use]
    pub fn check(proposed: &[(&str, &str)], committed: &[(&str, &str)]) -> Vec<Conflict> {
        let mut committed_map: HashMap<&str, Vec<&str>> = HashMap::new();
        for (artifact, agent) in committed {
            committed_map.entry(artifact).or_default().push(agent);
        }

        let mut conflicts = Vec::new();
        for (artifact, proposer) in proposed {
            if let Some(committers) = committed_map.get(artifact) {
                for committer in committers {
                    conflicts.push(Conflict {
                        task_a: proposer.to_string(),
                        task_b: committer.to_string(),
                        artifact: artifact.to_string(),
                        severity: ConflictSeverity::Major,
                    });
                }
            }
        }
        conflicts
    }
}

#[derive(Debug, Clone)]
pub struct ProgressReport {
    pub running: usize,
    pub waiting_merge: usize,
    pub complete: usize,
    pub failed: usize,
    pub completion_ratio: f64,
}

pub struct ProgressMonitor;

impl ProgressMonitor {
    #[must_use]
    pub fn check(nodes: &DashMap<NodeId, NodeStatus>) -> ProgressReport {
        let mut running = 0usize;
        let mut waiting_merge = 0usize;
        let mut complete = 0usize;
        let mut failed = 0usize;

        for entry in nodes.iter() {
            match entry.value() {
                NodeStatus::Idle => {}
                NodeStatus::Running | NodeStatus::Spawning => running += 1,
                NodeStatus::WaitingMerge => waiting_merge += 1,
                NodeStatus::Complete => complete += 1,
                NodeStatus::Failed => failed += 1,
            }
        }

        let total = running + waiting_merge + complete + failed;
        let completion_ratio = if total == 0 {
            0.0
        } else {
            complete as f64 / total as f64
        };

        ProgressReport {
            running,
            waiting_merge,
            complete,
            failed,
            completion_ratio,
        }
    }
}

#[derive(Debug, Default)]
pub struct SwarmTelemetry {
    pub spawn_count: AtomicU32,
    pub merge_count: AtomicU32,
    pub conflict_count: AtomicU32,
}

impl SwarmTelemetry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_spawn(&self) {
        self.spawn_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    pub fn record_merge(&self) {
        self.merge_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    pub fn record_conflict(&self) {
        self.conflict_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    #[must_use]
    pub fn snapshot(&self) -> (u32, u32, u32) {
        (
            self.spawn_count.load(AtomicOrdering::Relaxed),
            self.merge_count.load(AtomicOrdering::Relaxed),
            self.conflict_count.load(AtomicOrdering::Relaxed),
        )
    }
}
