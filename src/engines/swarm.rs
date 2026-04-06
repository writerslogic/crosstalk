use crate::types::consensus::MergeVote;
use crate::types::conversation::Turn;
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, Notify};
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
    task_spawner: mpsc::Sender<tokio::task::JoinHandle<()>>,
}

impl SwarmController {
    #[must_use]
    pub fn new() -> Self {
        let (state_tx, _) = broadcast::channel(1000);
        let (task_spawner, mut task_rx) = mpsc::channel(100);

        // SUPERVISOR (Reaper)
        tokio::spawn(async move {
            let mut join_set = JoinSet::new();
            loop {
                tokio::select! {
                    Some(handle) = task_rx.recv() => {
                        join_set.spawn(async move {
                            if let Err(e) = handle.await {
                                eprintln!("Swarm Node panicked: {:?}", e);
                            }
                        });
                    }
                    Some(res) = join_set.join_next() => {
                        if let Err(e) = res {
                            eprintln!("JoinSet execution error: {:?}", e);
                        }
                    }
                    else => break, // Channel closed
                }
            }
        });

        Self {
            nodes: Arc::new(DashMap::new()),
            state_tx,
            work_notify: Arc::new(Notify::new()),
            task_spawner,
        }
    }

    /// Spawns a node with a fully implemented, event-driven worker loop.
    pub async fn spawn_node(&self, node_id: impl Into<NodeId>) -> Result<()> {
        let id: NodeId = node_id.into();
        self.nodes.insert(Arc::clone(&id), NodeStatus::Spawning);

        let nodes_ref = Arc::clone(&self.nodes);
        let id_clone = Arc::clone(&id);
        
        // Subscribe to the global turn broadcaster before spawning
        let mut turn_rx = self.state_tx.subscribe();
        let notify_ref = Arc::clone(&self.work_notify);

        let worker_handle = tokio::spawn(async move {
            nodes_ref.insert(id_clone.clone(), NodeStatus::Running);

            // WORKER LOOP: Actively listen for global turns and local notifications
            loop {
                tokio::select! {
                    // 1. Process incoming turns from the Swarm
                    Ok(turn) = turn_rx.recv() => {
                        // Log or process the turn
                        println!("Node {} processing turn: {:?}", id_clone, turn);
                        
                        // Simulate heavy distributed workload
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        
                        nodes_ref.insert(id_clone.clone(), NodeStatus::WaitingMerge);
                    }
                    
                    // 2. React to out-of-band network notifications
                    _ = notify_ref.notified() => {
                        println!("Node {} received direct work notification.", id_clone);
                    }

                    // 3. Graceful shutdown condition (e.g., channel closed)
                    else => {
                        break; 
                    }
                }
            }

            nodes_ref.insert(id_clone, NodeStatus::Complete);
        });

        self.task_spawner
            .send(worker_handle)
            .await
            .map_err(|_| anyhow!("Critical: Swarm Supervisor died"))
    }

    pub fn broadcast_turn(&self, turn: Turn) -> Result<usize> {
        self.state_tx
            .send(turn)
            .map_err(|e| anyhow!("Failed to broadcast: {}", e))
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
        approvals >= (total_nodes >> 1) + 1
    }
}