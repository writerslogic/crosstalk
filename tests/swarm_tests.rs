use crosstalk::engines::swarm::{
    AgentAssigner, ConflictDetector, ConflictSeverity, GlobalMergeGate, LeaderElection, NodeStatus,
    ProgressMonitor, SwarmController, SwarmTelemetry, TaskDecomposer,
};
use crosstalk::types::consensus::MergeVote;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

// ── TaskDecomposer ────────────────────────────────────────────────────────────

#[test]
fn decompose_empty_returns_empty() {
    assert!(TaskDecomposer::decompose("", 4).is_empty());
}

#[test]
fn decompose_zero_tracks_returns_empty() {
    assert!(TaskDecomposer::decompose("Do something.", 0).is_empty());
}

#[test]
fn decompose_produces_at_most_n_tracks() {
    let desc = "Step one. Step two. Step three. Step four. Step five.";
    let tasks = TaskDecomposer::decompose(desc, 3);
    assert!(tasks.len() <= 3, "got {} tracks, expected ≤3", tasks.len());
    assert!(!tasks.is_empty());
}

#[test]
fn decompose_first_task_has_no_dependencies() {
    let tasks = TaskDecomposer::decompose("A. B. C.", 2);
    assert!(tasks[0].dependencies.is_empty());
}

#[test]
fn decompose_subsequent_tasks_depend_on_prior() {
    let tasks = TaskDecomposer::decompose("A\nB\nC\nD", 2);
    assert_eq!(tasks[1].dependencies, vec!["task-0"]);
}

#[test]
fn decompose_ids_are_sequential() {
    let tasks = TaskDecomposer::decompose("A. B. C.", 3);
    for (i, t) in tasks.iter().enumerate() {
        assert_eq!(t.id, format!("task-{i}"));
    }
}

// ── AgentAssigner ─────────────────────────────────────────────────────────────

#[test]
fn assign_empty_capabilities_returns_empty() {
    let tasks = TaskDecomposer::decompose("A. B.", 2);
    let result = AgentAssigner::assign(&tasks, &HashMap::new());
    assert!(result.is_empty());
}

#[test]
fn assign_picks_highest_capability_agent() {
    let tasks = TaskDecomposer::decompose("A.", 1);
    let caps: HashMap<String, f64> = [("weak".into(), 0.3), ("strong".into(), 0.9)]
        .into_iter()
        .collect();
    let result = AgentAssigner::assign(&tasks, &caps);
    assert_eq!(result["task-0"], "strong");
}

#[test]
fn assign_all_tasks_get_an_agent() {
    let tasks = TaskDecomposer::decompose("A. B. C. D.", 4);
    let caps: HashMap<String, f64> = [("agent-x".into(), 0.8)].into_iter().collect();
    let result = AgentAssigner::assign(&tasks, &caps);
    assert_eq!(result.len(), tasks.len());
}

// ── ConflictDetector ──────────────────────────────────────────────────────────

#[test]
fn no_conflicts_when_different_artifacts() {
    let proposed = vec![("file_a.rs", "agent-1")];
    let committed = vec![("file_b.rs", "agent-2")];
    assert!(ConflictDetector::check(&proposed, &committed).is_empty());
}

#[test]
fn conflict_detected_on_same_artifact() {
    let proposed = vec![("lib.rs", "agent-1")];
    let committed = vec![("lib.rs", "agent-2")];
    let conflicts = ConflictDetector::check(&proposed, &committed);
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].severity, ConflictSeverity::Major);
    assert_eq!(conflicts[0].artifact, "lib.rs");
}

#[test]
fn multiple_conflicts_detected() {
    let proposed = vec![("a.rs", "p1"), ("b.rs", "p1")];
    let committed = vec![("a.rs", "c1"), ("b.rs", "c2")];
    let conflicts = ConflictDetector::check(&proposed, &committed);
    assert_eq!(conflicts.len(), 2);
}

// ── ProgressMonitor ───────────────────────────────────────────────────────────

#[test]
fn progress_empty_nodes_returns_zero_ratio() {
    let nodes: DashMap<String, NodeStatus> = DashMap::new();
    let report = ProgressMonitor::check(&nodes);
    assert_eq!(report.completion_ratio, 0.0);
    assert_eq!(report.complete, 0);
}

#[test]
fn progress_all_complete_returns_ratio_one() {
    let nodes: DashMap<Arc<str>, NodeStatus> = DashMap::new();
    nodes.insert(Arc::from("n1"), NodeStatus::Complete);
    nodes.insert(Arc::from("n2"), NodeStatus::Complete);
    let report = ProgressMonitor::check(&nodes);
    assert!((report.completion_ratio - 1.0).abs() < 1e-9);
    assert_eq!(report.complete, 2);
}

#[test]
fn progress_mixed_statuses_counted_correctly() {
    let nodes: DashMap<Arc<str>, NodeStatus> = DashMap::new();
    nodes.insert(Arc::from("n1"), NodeStatus::Running);
    nodes.insert(Arc::from("n2"), NodeStatus::Complete);
    nodes.insert(Arc::from("n3"), NodeStatus::Failed);
    nodes.insert(Arc::from("n4"), NodeStatus::WaitingMerge);
    let report = ProgressMonitor::check(&nodes);
    assert_eq!(report.running, 1);
    assert_eq!(report.complete, 1);
    assert_eq!(report.failed, 1);
    assert_eq!(report.waiting_merge, 1);
    assert!((report.completion_ratio - 0.25).abs() < 1e-9);
}

// ── SwarmTelemetry ────────────────────────────────────────────────────────────

#[test]
fn telemetry_initial_snapshot_is_zero() {
    let t = SwarmTelemetry::new();
    assert_eq!(t.snapshot(), (0, 0, 0));
}

#[test]
fn telemetry_records_events_correctly() {
    let t = SwarmTelemetry::new();
    t.record_spawn();
    t.record_spawn();
    t.record_merge();
    t.record_conflict();
    assert_eq!(t.snapshot(), (2, 1, 1));
}

// ── GlobalMergeGate ───────────────────────────────────────────────────────────

#[test]
fn merge_gate_empty_votes_no_quorum() {
    assert!(!GlobalMergeGate::has_quorum(&[], 5));
}

#[test]
fn merge_gate_majority_approves_quorum() {
    let votes = vec![
        MergeVote {
            node_id: "a".into(),
            approve: true,
            reason: String::new(),
        },
        MergeVote {
            node_id: "b".into(),
            approve: true,
            reason: String::new(),
        },
        MergeVote {
            node_id: "c".into(),
            approve: false,
            reason: String::new(),
        },
    ];
    assert!(GlobalMergeGate::has_quorum(&votes, 3));
}

#[test]
fn merge_gate_minority_approves_no_quorum() {
    let votes = vec![
        MergeVote {
            node_id: "a".into(),
            approve: true,
            reason: String::new(),
        },
        MergeVote {
            node_id: "b".into(),
            approve: false,
            reason: String::new(),
        },
        MergeVote {
            node_id: "c".into(),
            approve: false,
            reason: String::new(),
        },
    ];
    assert!(!GlobalMergeGate::has_quorum(&votes, 3));
}

// ── LeaderElection ────────────────────────────────────────────────────────────

struct AlwaysGrantNetwork;

impl crosstalk::engines::swarm::RaftNetwork for AlwaysGrantNetwork {
    async fn request_vote(
        &self,
        _term: u64,
        _candidate_id: Arc<str>,
    ) -> anyhow::Result<bool> {
        Ok(true)
    }
}

struct AlwaysDenyNetwork;

impl crosstalk::engines::swarm::RaftNetwork for AlwaysDenyNetwork {
    async fn request_vote(
        &self,
        _term: u64,
        _candidate_id: Arc<str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }
}

#[tokio::test]
async fn election_wins_when_all_peers_grant() {
    let mut election = LeaderElection::new("node-0");
    let peers: Vec<Arc<AlwaysGrantNetwork>> =
        vec![Arc::new(AlwaysGrantNetwork), Arc::new(AlwaysGrantNetwork)];
    let (_tx, rx) = mpsc::channel(1);
    let won = election.run_election_cycle(&peers, rx).await;
    assert!(won, "should win election when all peers grant vote");
}

#[tokio::test]
async fn election_loses_when_all_peers_deny() {
    let mut election = LeaderElection::new("node-0");
    let peers: Vec<Arc<AlwaysDenyNetwork>> = vec![
        Arc::new(AlwaysDenyNetwork),
        Arc::new(AlwaysDenyNetwork),
        Arc::new(AlwaysDenyNetwork),
        Arc::new(AlwaysDenyNetwork),
    ];
    let (_tx, rx) = mpsc::channel(1);
    let won = election.run_election_cycle(&peers, rx).await;
    assert!(!won, "should lose when no peers grant vote");
}

#[tokio::test]
async fn election_preempted_by_heartbeat() {
    let mut election = LeaderElection::new("node-0");
    let peers: Vec<Arc<AlwaysGrantNetwork>> = vec![];
    let (tx, rx) = mpsc::channel(1);
    // Send heartbeat immediately so the election is preempted
    tx.send(()).await.unwrap();
    let won = election.run_election_cycle(&peers, rx).await;
    assert!(!won, "election preempted by heartbeat should not win");
}

// ── SwarmController ───────────────────────────────────────────────────────────

#[tokio::test]
async fn spawn_node_registers_in_map() {
    let ctrl = SwarmController::new();
    let (tx, _) = tokio::sync::broadcast::channel::<crosstalk::types::conversation::Turn>(16);
    ctrl.spawn_node("node-a", tx.subscribe());
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert!(ctrl.nodes.contains_key("node-a"));
}

#[tokio::test]
async fn broadcast_turn_succeeds() {
    use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
    let ctrl = SwarmController::new();
    let (tx, _) = tokio::sync::broadcast::channel::<Turn>(16);
    ctrl.spawn_node("n1", tx.subscribe());
    ctrl.spawn_node("n2", tx.subscribe());
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let turn = Turn {
        index: 0,
        model_id: "test".into(),
        content: "hello".into(),
        timestamp: ConversationState::now(),
        diffs: vec![],
        certainty: None,
        outcome: TurnOutcome::Unknown,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
    };
    let result = ctrl.broadcast_turn(turn);
    assert!(result.is_ok(), "broadcast_turn should succeed");
}
