use dashmap::DashMap;
use tokio::task::JoinSet;
use tokio::sync::{broadcast, mpsc, Notify};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::types::{ConversationState, Turn};

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
}

impl SwarmController {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);
        Self {
            nodes: Arc::new(DashMap::new()),
            state_tx: tx,
            work_notify: Arc::new(Notify::new()),
        }
    }

    pub async fn spawn_node(&self, node_id: String) {
        self.nodes.insert(node_id.clone(), NodeStatus::Spawning);
        // In a real impl, this would tokio::spawn an Orchestrator instance
        self.nodes.insert(node_id, NodeStatus::Running);
    }

    pub fn broadcast_turn(&self, turn: Turn) {
        let _ = self.state_tx.send(turn);
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
    pub fn new() -> Self {
        Self { term: 0, state: RaftState::Follower, votes: 0 }
    }

    pub fn start_election(&mut self) {
        self.term += 1;
        self.state = RaftState::Candidate;
        self.votes = 1; // Vote for self
    }
}

pub struct MergeVote {
    pub node_id: String,
    pub approve: bool,
    pub reason: String,
}

pub struct GlobalMergeGate;

impl GlobalMergeGate {
    pub fn collect_votes(votes: Vec<MergeVote>) -> bool {
        votes.iter().all(|v| v.approve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leader_election_init() {
        let mut le = LeaderElection::new();
        le.start_election();
        assert_eq!(le.state, RaftState::Candidate);
        assert_eq!(le.term, 1);
    }

    #[test]
    fn test_merge_gate() {
        let votes = vec![
            MergeVote { node_id: "1".into(), approve: true, reason: "".into() },
            MergeVote { node_id: "2".into(), approve: true, reason: "".into() },
        ];
        assert!(GlobalMergeGate::collect_votes(votes));
    }
}
