use crosstalk::engines::planning::{
    BranchManager, BranchRegistry, ContextPruner, CriticalPathAnalyzer, DifficultyEstimator,
    GoalScheduler, MilestoneDetector, PlanningEngine, SessionManager,
};
use crosstalk::types::conversation::{ConversationState, Turn, TurnOutcome};
use crosstalk::types::planning::{GoalNode, GoalStatus, GoalTree, SessionManifest};
use tempfile::tempdir;

fn make_node(id: &str, title: &str, status: GoalStatus) -> GoalNode {
    GoalNode {
        id: id.to_string(),
        title: title.to_string(),
        children: vec![],
        status,
        assigned_turn: None,
        deadline: None,
        dependencies: vec![],
    }
}

fn make_turn(index: u32, outcome: TurnOutcome) -> Turn {
    Turn {
        index,
        model_id: "m".into(),
        content: "content".into(),
        timestamp: 0,
        diffs: vec![],
        certainty: None,
        outcome,
        task_category: None,
        structure: None,
        signature: vec![],
        surprise_signal: None,
    }
}

// ── GoalTree methods ──────────────────────────────────────────────────────────

#[test]
fn goal_tree_empty_has_zero_leaves() {
    let tree = GoalTree::default();
    assert!(tree.get_leaves().is_empty());
}

#[test]
fn goal_tree_add_child_inserts_under_parent() {
    let mut tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    let child = make_node("c1", "Child 1", GoalStatus::Pending);
    assert!(tree.add_child("root", child));
    assert_eq!(tree.root.as_ref().unwrap().children.len(), 1);
}

#[test]
fn goal_tree_add_child_returns_false_for_unknown_parent() {
    let mut tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    assert!(!tree.add_child("nonexistent", make_node("x", "X", GoalStatus::Pending)));
}

#[test]
fn goal_tree_get_leaves_returns_childless_nodes() {
    let mut tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    tree.add_child("root", make_node("c1", "C1", GoalStatus::Complete));
    tree.add_child("root", make_node("c2", "C2", GoalStatus::Pending));
    let leaves = tree.get_leaves();
    let ids: Vec<&str> = leaves.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"c1"));
    assert!(ids.contains(&"c2"));
    assert!(!ids.contains(&"root"));
}

#[test]
fn goal_tree_get_subtree_finds_nested_node() {
    let mut tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    tree.add_child("root", make_node("child", "Child", GoalStatus::InProgress));
    let sub = tree.get_subtree("child");
    assert!(sub.is_some());
    assert_eq!(sub.unwrap().id, "child");
}

#[test]
fn goal_tree_get_subtree_returns_none_for_missing() {
    let tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    assert!(tree.get_subtree("missing").is_none());
}

#[test]
fn goal_tree_analyze_depth_five_levels() {
    let mut tree = GoalTree {
        root: Some(make_node("l1", "L1", GoalStatus::Pending)),
    };
    tree.add_child("l1", make_node("l2", "L2", GoalStatus::Pending));
    tree.add_child("l2", make_node("l3", "L3", GoalStatus::Pending));
    tree.add_child("l3", make_node("l4", "L4", GoalStatus::Pending));
    tree.add_child("l4", make_node("l5", "L5", GoalStatus::Complete));
    let a = tree.analyze();
    assert_eq!(a.depth, 5);
    assert_eq!(a.leaf_count, 1);
    assert_eq!(a.total_count, 5);
    assert!((a.completion_ratio - 0.2).abs() < 1e-9);
}

#[test]
fn goal_tree_completion_ratio_all_complete() {
    let mut tree = GoalTree {
        root: Some(make_node("r", "R", GoalStatus::Complete)),
    };
    tree.add_child("r", make_node("c1", "C1", GoalStatus::Complete));
    tree.add_child("r", make_node("c2", "C2", GoalStatus::Complete));
    assert!((tree.completion_ratio() - 1.0).abs() < 1e-9);
}

// ── PlanningEngine::update_goal_status ────────────────────────────────────────

#[test]
fn update_goal_status_all_children_complete_marks_parent() {
    let mut root = make_node("r", "Root", GoalStatus::Pending);
    root.children
        .push(make_node("c1", "C1", GoalStatus::Complete));
    root.children
        .push(make_node("c2", "C2", GoalStatus::Complete));
    PlanningEngine::update_goal_status(&mut root);
    assert_eq!(root.status, GoalStatus::Complete);
}

#[test]
fn update_goal_status_blocked_child_blocks_parent() {
    let mut root = make_node("r", "Root", GoalStatus::Pending);
    root.children
        .push(make_node("c1", "C1", GoalStatus::Complete));
    root.children
        .push(make_node("c2", "C2", GoalStatus::Blocked));
    PlanningEngine::update_goal_status(&mut root);
    assert_eq!(root.status, GoalStatus::Blocked);
}

#[test]
fn update_goal_status_inprogress_child_sets_parent_inprogress() {
    let mut root = make_node("r", "Root", GoalStatus::Pending);
    root.children
        .push(make_node("c1", "C1", GoalStatus::InProgress));
    root.children
        .push(make_node("c2", "C2", GoalStatus::Pending));
    PlanningEngine::update_goal_status(&mut root);
    assert_eq!(root.status, GoalStatus::InProgress);
}

// ── DifficultyEstimator ───────────────────────────────────────────────────────

#[test]
fn difficulty_empty_turns_returns_half() {
    assert!((DifficultyEstimator::estimate(&[]) - 0.5).abs() < 1e-9);
}

#[test]
fn difficulty_all_rejected_is_high() {
    let turns = vec![
        make_turn(0, TurnOutcome::Rejected),
        make_turn(1, TurnOutcome::Rejected),
    ];
    assert!(DifficultyEstimator::estimate(&turns) > 0.7);
}

#[test]
fn difficulty_all_tests_passed_is_low() {
    let turns = vec![
        make_turn(0, TurnOutcome::TestsPassed),
        make_turn(1, TurnOutcome::TestsPassed),
    ];
    assert!(DifficultyEstimator::estimate(&turns) < 0.3);
}

// ── GoalScheduler ─────────────────────────────────────────────────────────────

#[test]
fn scheduler_empty_tree_returns_empty() {
    let tree = GoalTree::default();
    let batches = GoalScheduler::schedule(&tree).unwrap();
    assert!(batches.is_empty());
}

#[test]
fn scheduler_flat_tree_single_root_has_batches() {
    let tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    let batches = GoalScheduler::schedule(&tree).unwrap();
    assert!(!batches.is_empty());
    let all_ids: Vec<&str> = batches
        .iter()
        .flat_map(|b| b.iter().map(|s| s.as_str()))
        .collect();
    assert!(all_ids.contains(&"root"));
}

#[test]
fn scheduler_parent_in_different_batch_from_child() {
    let mut tree = GoalTree {
        root: Some(make_node("parent", "Parent", GoalStatus::Pending)),
    };
    tree.add_child("parent", make_node("child", "Child", GoalStatus::Pending));
    let batches = GoalScheduler::schedule(&tree).unwrap();
    // There should be at least 2 batches (parent depends on child)
    assert!(batches.len() >= 2);
}

// ── CriticalPathAnalyzer ──────────────────────────────────────────────────────

#[test]
fn critical_path_empty_tree_is_empty() {
    let tree = GoalTree::default();
    assert!(CriticalPathAnalyzer::compute(&tree).is_empty());
}

#[test]
fn critical_path_single_node() {
    let tree = GoalTree {
        root: Some(make_node("only", "Only", GoalStatus::Pending)),
    };
    assert_eq!(CriticalPathAnalyzer::compute(&tree), vec!["only"]);
}

#[test]
fn critical_path_follows_longest_branch() {
    let mut tree = GoalTree {
        root: Some(make_node("root", "Root", GoalStatus::Pending)),
    };
    tree.add_child("root", make_node("short", "Short", GoalStatus::Pending));
    tree.add_child("root", make_node("long", "Long", GoalStatus::Pending));
    tree.add_child("long", make_node("deeper", "Deeper", GoalStatus::Pending));
    let path = CriticalPathAnalyzer::compute(&tree);
    assert_eq!(path.len(), 3);
    assert_eq!(path[0], "root");
    assert_eq!(path[1], "long");
    assert_eq!(path[2], "deeper");
}

// ── MilestoneDetector ─────────────────────────────────────────────────────────

#[test]
fn milestone_none_when_no_progress() {
    let sigma = ConversationState::new("ms-test");
    assert!(MilestoneDetector::check(&sigma, 0.0).is_none());
}

#[test]
fn milestone_triggered_at_fifty_percent() {
    let mut sigma = ConversationState::new("ms-half");
    // 1 complete out of 2 total = 0.5 ratio
    let mut root = make_node("r", "Root", GoalStatus::Pending);
    root.children
        .push(make_node("c1", "C1", GoalStatus::Complete));
    sigma.goal_tree = GoalTree { root: Some(root) };
    let m = MilestoneDetector::check(&sigma, 0.0);
    assert!(m.is_some());
    assert!(m.unwrap().title.contains("50%"));
}

#[test]
fn milestone_no_duplicate_when_already_past_threshold() {
    let mut sigma = ConversationState::new("ms-dup");
    let mut root = make_node("r", "Root", GoalStatus::Pending);
    root.children
        .push(make_node("c1", "C1", GoalStatus::Complete));
    sigma.goal_tree = GoalTree { root: Some(root) };
    // prev_ratio already at 0.5 → no duplicate trigger
    assert!(MilestoneDetector::check(&sigma, 0.5).is_none());
}

// ── BranchManager ─────────────────────────────────────────────────────────────

#[test]
fn fork_creates_different_session_id() {
    let sigma = ConversationState::new("original");
    let fork = BranchManager::fork(&sigma);
    assert_ne!(fork.session_id, sigma.session_id);
    assert!(fork.session_id.starts_with("original-fork-"));
}

#[test]
fn fork_preserves_turn_history() {
    let mut sigma = ConversationState::new("s");
    sigma.turns.push(make_turn(0, TurnOutcome::Compiled));
    let fork = BranchManager::fork(&sigma);
    assert_eq!(fork.turns.len(), 1);
}

// ── SessionManager ────────────────────────────────────────────────────────────

#[test]
fn session_manager_save_and_load_roundtrip() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    let mut state = ConversationState::new("my-session");
    state.turns.push(make_turn(0, TurnOutcome::TestsPassed));
    mgr.save("my-session", &state).unwrap();
    let loaded = mgr.load("my-session").unwrap();
    assert_eq!(loaded.session_id, "my-session");
    assert_eq!(loaded.turns.len(), 1);
}

#[test]
fn session_manager_load_missing_returns_error() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    assert!(mgr.load("does-not-exist").is_err());
}

#[test]
fn session_manager_list_returns_saved_ids() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    mgr.save("alpha", &ConversationState::new("alpha")).unwrap();
    mgr.save("beta", &ConversationState::new("beta")).unwrap();
    let mut ids = mgr.list().unwrap();
    ids.sort();
    assert_eq!(ids, vec!["alpha", "beta"]);
}

#[test]
fn session_manager_delete_removes_entry() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    mgr.save("to-delete", &ConversationState::new("to-delete"))
        .unwrap();
    mgr.delete("to-delete").unwrap();
    assert!(mgr.load("to-delete").is_err());
}

// ── ContextPruner ─────────────────────────────────────────────────────────────

#[test]
fn context_pruner_returns_recent_turns() {
    let mut sigma = ConversationState::new("prune-test");
    for i in 0u32..20 {
        sigma.turns.push(make_turn(i, TurnOutcome::Unknown));
    }
    let pruned = ContextPruner::prune(&sigma, "root", 5);
    assert!(
        pruned.len() <= 5 + 5,
        "should return at most max_turns + 5 critical"
    );
}

#[test]
fn context_pruner_empty_state_returns_empty() {
    let sigma = ConversationState::new("empty");
    let pruned = ContextPruner::prune(&sigma, "root", 10);
    assert!(pruned.is_empty());
}

// ── Acceptance criteria: >50% token reduction ─────────────────────────────────

#[test]
fn context_pruner_reduces_by_more_than_50_percent() {
    let mut sigma = ConversationState::new("prune-ratio");
    for i in 0u32..30 {
        sigma.turns.push(make_turn(i, TurnOutcome::Unknown));
    }
    let total = sigma.turns.len();
    let pruned = ContextPruner::prune(&sigma, "nonexistent-goal", 5);
    assert!(
        pruned.len() * 2 <= total,
        "pruned {} of {} turns — must reduce by >50%",
        pruned.len(),
        total
    );
}

// ── Checksum roundtrip ─────────────────────────────────────────────────────────

#[test]
fn checksum_roundtrip_succeeds() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    let sigma = ConversationState::new("ck-ok");
    let ck = mgr.save_with_checksum("ck-ok", &sigma).unwrap();
    let loaded = mgr.load_with_checksum("ck-ok", &ck).unwrap();
    assert_eq!(loaded.session_id, "ck-ok");
}

#[test]
fn checksum_mismatch_returns_error() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    let sigma = ConversationState::new("ck-fail");
    mgr.save_with_checksum("ck-fail", &sigma).unwrap();
    let bad_ck = [0u8; 32];
    let result = mgr.load_with_checksum("ck-fail", &bad_ck);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Checksum mismatch")
    );
}

// ── SessionManifest ────────────────────────────────────────────────────────────

#[test]
fn manifest_save_and_load_roundtrip() {
    let dir = tempdir().unwrap();
    let mgr = SessionManager::new(dir.path().to_str().unwrap()).unwrap();
    let manifest = SessionManifest {
        author: "alice".to_string(),
        created_at: 1_000_000,
        total_turns: 42,
        total_tokens_used: 100_000,
        cost_estimate_usd: 1.23,
    };
    mgr.save_manifest(&manifest).unwrap();
    let loaded = mgr.load_manifest("alice").unwrap();
    assert_eq!(loaded.total_turns, 42);
    assert!((loaded.cost_estimate_usd - 1.23).abs() < f64::EPSILON);
}

// ── BranchRegistry ────────────────────────────────────────────────────────────

#[test]
fn branch_registry_records_parent_lineage() {
    let dir = tempdir().unwrap();
    let reg = BranchRegistry::new(dir.path().to_str().unwrap()).unwrap();
    reg.register("child-1", "parent-root").unwrap();
    reg.register("child-2", "parent-root").unwrap();
    assert_eq!(
        reg.parent_of("child-1").unwrap().as_deref(),
        Some("parent-root")
    );
    let children = reg.list_children("parent-root").unwrap();
    assert_eq!(children.len(), 2);
}

#[test]
fn branch_manager_fork_creates_unique_session_id() {
    let sigma = ConversationState::new("main");
    let fork = BranchManager::fork(&sigma);
    assert_ne!(fork.session_id, sigma.session_id);
    assert!(fork.session_id.starts_with("main-fork-"));
}
