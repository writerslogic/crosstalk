use crate::types::conversation::Turn;
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, broadcast};
use tokio::task::JoinSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Spawning,
    Running,
    WaitingMerge,
    Complete,
    Failed,
}

pub struct SwarmController {
    pub nodes: Arc<DashMap<String, NodeStatus>>,
    pub state_tx: broadcast::Sender<Turn>,
    pub work_notify: Arc<Notify>,
    pub join_set: JoinSet<()>,
}

impl SwarmController {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);
        Self {
            nodes: Arc::new(DashMap::new()),
            state_tx: tx,
            work_notify: Arc::new(Notify::new()),
            join_set: JoinSet::new(),
        }
    }

    pub async fn spawn_node(&self, node_id: String) {
        self.nodes.insert(node_id.clone(), NodeStatus::Spawning);
        self.nodes.insert(node_id, NodeStatus::Running);
    }

    pub fn broadcast_turn(&self, turn: Turn) {
        let _ = self.state_tx.send(turn);
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
    pub term: u64,
    pub state: RaftState,
    pub votes: usize,
}

impl LeaderElection {
    #[must_use]
    pub fn new() -> Self {
        Self {
            term: 0,
            state: RaftState::Follower,
            votes: 0,
        }
    }

    pub async fn run_election_cycle(&mut self, node_count: usize) -> bool {
        let mut rng = rand::rng();
        let timeout_ms = rng.random_range(150..300);
        tokio::time::sleep(Duration::from_millis(timeout_ms)).await;

        if self.state == RaftState::Follower {
            self.term += 1;
            self.state = RaftState::Candidate;
            self.votes = 1;

            if self.votes > node_count / 2 {
                self.state = RaftState::Leader;
                return true;
            }
        }
        false
    }
}

impl Default for LeaderElection {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GlobalMergeGate;

impl GlobalMergeGate {
    #[must_use]
    pub fn collect_votes(votes: Vec<crate::types::consensus::MergeVote>) -> bool {
        if votes.is_empty() {
            return false;
        }
        votes.iter().all(|v| v.approve)
    }
}
