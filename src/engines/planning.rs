use crate::types::conversation::{ConversationState, Turn};
use crate::types::planning::{GoalNode, GoalStatus, GoalTree, Milestone, SessionManifest};
use anyhow::{anyhow, Result};
use petgraph::algo::toposort;
use petgraph::graph::DiGraph;
use sha2::{Digest, Sha256};
use sled::Db;
use std::cmp::Ordering;
use std::collections::HashMap;

pub struct PlanningEngine;

impl PlanningEngine {
    /// Single-pass, cache-friendly state aggregation.
    /// Eliminates the triple-iteration over children.
    pub fn update_goal_status(node: &mut GoalNode) {
        if node.children.is_empty() {
            return;
        }

        let mut all_complete = true;
        let mut any_blocked = false;
        let mut any_work_seen = false;

        for child in &mut node.children {
            // Depth-first recursion to update leaves first
            Self::update_goal_status(child);

            // Evaluate state in a single CPU cycle
            match child.status {
                GoalStatus::Blocked => {
                    any_blocked = true;
                    all_complete = false;
                }
                GoalStatus::InProgress => {
                    any_work_seen = true;
                    all_complete = false;
                }
                GoalStatus::Complete => {
                    any_work_seen = true; // Completed children imply work was done
                }
                _ => {
                    all_complete = false;
                }
            }
        }

        // Apply state transition hierarchically
        if any_blocked {
            node.status = GoalStatus::Blocked;
        } else if all_complete {
            node.status = GoalStatus::Complete;
        } else if any_work_seen {
            node.status = GoalStatus::InProgress;
        }
    }
}

pub struct BranchManager;

impl BranchManager {
    #[must_use]
    pub fn fork(sigma: &ConversationState) -> ConversationState {
        let mut fork = sigma.clone();
        fork.session_id = format!("{}-fork-{}", sigma.session_id, ConversationState::now());
        fork
    }
}

/// We embed `assigned_turn` DIRECTLY into the struct.
/// This completely eliminates the need for the O(N^2) `find_node` tree search.
#[derive(Eq, PartialEq)]
struct PrunableGoal {
    criticality: u32,
    assigned_turn: Option<u32>,
}

impl Ord for PrunableGoal {
    fn cmp(&self, other: &Self) -> Ordering {
        // Standard Max-Heap ordering. The highest criticality is popped first.
        self.criticality.cmp(&other.criticality)
    }
}

impl PartialOrd for PrunableGoal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ContextPruner;

impl ContextPruner {
    #[must_use]
    pub fn prune(sigma: &ConversationState, _active_goal_id: &str, max_turns: usize) -> Vec<Turn> {
        let now = ConversationState::now();
        let mut goals = Vec::new();

        // 1. Flatten the tree into a vector in O(N) time
        if let Some(ref root) = sigma.goal_tree.root {
            Self::flatten_goals(root, 0, now, &mut goals);
        }

        // 2. O(N log N) Fast Sorting (Replacing the clunky BinaryHeap)
        // Sorts descending so the most critical are at the front.
        goals.sort_unstable_by(|a, b| b.criticality.cmp(&a.criticality));

        // 3. Extract the critical turn indices (Maximum of 5)
        // Because n=5, a simple Vec `.contains()` fits entirely in the L1 CPU Cache 
        // and is mathematically faster than hashing elements into a HashSet.
        let critical_turn_indices: Vec<u32> = goals
            .into_iter()
            .filter_map(|g| g.assigned_turn)
            .take(5)
            .collect();

        let total_turns = sigma.turns.len();

        // 4. Stream and filter in a single pass
        sigma
            .turns
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let is_recent = total_turns.saturating_sub(*idx) <= max_turns;
                let is_critical = critical_turn_indices.contains(&(*idx as u32));
                is_recent || is_critical
            })
            .map(|(_, turn)| turn.clone())
            .collect()
    }

    /// Recursively flattens the tree into a Vec while calculating criticality.
    fn flatten_goals(node: &GoalNode, depth: u32, now: u64, list: &mut Vec<PrunableGoal>) {
        list.push(PrunableGoal {
            criticality: Self::calculate_criticality(node, depth, now),
            assigned_turn: node.assigned_turn, // Copied out to prevent future tree-searching
        });

        for child in &node.children {
            Self::flatten_goals(child, depth + 1, now, list);
        }
    }

    /// Evaluates structural dependency depth against time-decay urgency.
    fn calculate_criticality(node: &GoalNode, depth: u32, now: u64) -> u32 {
        let depth_factor = depth * 100;

        let urgency_factor = match node.deadline {
            Some(d) => {
                if d <= now {
                    5000 // Overdue is highest priority
                } else {
                    let diff = d - now;
                    (86400 / diff.max(1)).min(4000) as u32
                }
            }
            None => 0,
        };

        depth_factor + urgency_factor
    }
}

pub struct DifficultyEstimator;

impl DifficultyEstimator {
    /// Estimate task difficulty in [0.0, 1.0] from recent turn history.
    /// Heuristics: failed/stalled outcomes raise difficulty; short content lowers it.
    #[must_use]
    pub fn estimate(turns: &[Turn]) -> f64 {
        if turns.is_empty() {
            return 0.5;
        }
        use crate::types::conversation::TurnOutcome;
        let mut score = 0.0f64;
        for t in turns {
            score += match t.outcome {
                TurnOutcome::Rejected | TurnOutcome::RolledBack => 0.8,
                TurnOutcome::Stalled => 0.6,
                TurnOutcome::Unknown => 0.5,
                TurnOutcome::Compiled => 0.3,
                TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence => 0.1,
            };
        }
        (score / turns.len() as f64).clamp(0.0, 1.0)
    }
}

pub struct GoalScheduler;

impl GoalScheduler {
    /// Topological schedule of `tree` using petgraph.
    /// Returns parallel batches: goals in the same batch have no mutual dependencies
    /// and can execute concurrently. Inner vec is a single parallel batch.
    pub fn schedule(tree: &GoalTree) -> Result<Vec<Vec<String>>> {
        let root = match &tree.root {
            Some(r) => r,
            None => return Ok(vec![]),
        };

        // Collect all nodes and build id→index map
        let mut nodes: Vec<&GoalNode> = Vec::new();
        Self::collect_nodes(root, &mut nodes);

        let mut graph = DiGraph::<String, ()>::new();
        let mut id_to_idx: HashMap<&str, petgraph::graph::NodeIndex> = HashMap::new();

        for node in &nodes {
            let idx = graph.add_node(node.id.clone());
            id_to_idx.insert(&node.id, idx);
        }

        // Parent→child edges (parent depends on children completing first)
        Self::add_edges(root, &id_to_idx, &mut graph);
        // Explicit dependency edges
        for node in &nodes {
            if let Some(&src) = id_to_idx.get(node.id.as_str()) {
                for dep in &node.dependencies {
                    if let Some(&dst) = id_to_idx.get(dep.as_str()) {
                        graph.add_edge(src, dst, ());
                    }
                }
            }
        }

        let sorted = toposort(&graph, None)
            .map_err(|_| anyhow!("Goal dependency cycle detected"))?;

        // Group into parallel batches using longest-path levels
        let mut level: HashMap<petgraph::graph::NodeIndex, usize> = HashMap::new();
        for &idx in &sorted {
            let l = graph
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .map(|p| level.get(&p).copied().unwrap_or(0) + 1)
                .max()
                .unwrap_or(0);
            level.insert(idx, l);
        }

        let max_level = level.values().copied().max().unwrap_or(0);
        let mut batches: Vec<Vec<String>> = vec![Vec::new(); max_level + 1];
        for (idx, l) in &level {
            batches[*l].push(graph[*idx].clone());
        }
        Ok(batches.into_iter().filter(|b| !b.is_empty()).collect())
    }

    fn collect_nodes<'a>(node: &'a GoalNode, out: &mut Vec<&'a GoalNode>) {
        out.push(node);
        for child in &node.children {
            Self::collect_nodes(child, out);
        }
    }

    fn add_edges(
        node: &GoalNode,
        map: &HashMap<&str, petgraph::graph::NodeIndex>,
        graph: &mut DiGraph<String, ()>,
    ) {
        if let Some(&parent_idx) = map.get(node.id.as_str()) {
            for child in &node.children {
                if let Some(&child_idx) = map.get(child.id.as_str()) {
                    graph.add_edge(parent_idx, child_idx, ());
                }
                Self::add_edges(child, map, graph);
            }
        }
    }
}

pub struct CriticalPathAnalyzer;

impl CriticalPathAnalyzer {
    /// Returns the sequence of goal IDs forming the longest root-to-leaf path.
    #[must_use]
    pub fn compute(tree: &GoalTree) -> Vec<String> {
        match &tree.root {
            None => vec![],
            Some(root) => Self::longest_path(root),
        }
    }

    fn longest_path(node: &GoalNode) -> Vec<String> {
        if node.children.is_empty() {
            return vec![node.id.clone()];
        }
        let best = node
            .children
            .iter()
            .map(Self::longest_path)
            .max_by_key(|p| p.len())
            .unwrap_or_default();
        let mut path = vec![node.id.clone()];
        path.extend(best);
        path
    }
}

pub struct MilestoneDetector;

impl MilestoneDetector {
    const THRESHOLD: f64 = 0.5;

    /// Check whether the current goal completion ratio crosses a milestone boundary.
    /// Returns a `Milestone` when a root-level goal becomes Complete, or when
    /// overall completion crosses 50% or 100%.
    #[must_use]
    pub fn check(sigma: &ConversationState, prev_ratio: f64) -> Option<Milestone> {
        let analysis = sigma.goal_tree.analyze();
        let ratio = analysis.completion_ratio;
        let now = ConversationState::now();

        // Root goal completed
        if let Some(root) = &sigma.goal_tree.root
            && root.status == GoalStatus::Complete
            && prev_ratio < 1.0
        {
            return Some(Milestone {
                id: format!("root-complete-{now}"),
                title: format!("Root goal '{}' completed", root.title),
                triggered_at: now,
                completion_ratio: ratio,
            });
        }

        // Crossed 50% threshold
        if prev_ratio < Self::THRESHOLD && ratio >= Self::THRESHOLD {
            return Some(Milestone {
                id: format!("half-complete-{now}"),
                title: "50% of goals completed".to_string(),
                triggered_at: now,
                completion_ratio: ratio,
            });
        }

        // All goals complete
        if prev_ratio < 1.0 && ratio >= 1.0 && analysis.total_count > 0 {
            return Some(Milestone {
                id: format!("all-complete-{now}"),
                title: "All goals completed".to_string(),
                triggered_at: now,
                completion_ratio: ratio,
            });
        }

        None
    }
}

pub struct SessionManager {
    db: Db,
}

impl SessionManager {
    const TREE_NAME: &'static str = "sessions";

    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn save(&self, session_id: &str, state: &ConversationState) -> Result<()> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        let bytes = serde_json::to_vec(state)?;
        tree.insert(session_id.as_bytes(), bytes)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn load(&self, session_id: &str) -> Result<ConversationState> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        let bytes = tree
            .get(session_id.as_bytes())?
            .ok_or_else(|| anyhow!("Session '{}' not found", session_id))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn list(&self) -> Result<Vec<String>> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        tree.iter()
            .keys()
            .map(|k| {
                k.map_err(|e| anyhow!(e))
                    .and_then(|b| String::from_utf8(b.to_vec()).map_err(|e| anyhow!(e)))
            })
            .collect()
    }

    pub fn delete(&self, session_id: &str) -> Result<()> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        tree.remove(session_id.as_bytes())?;
        Ok(())
    }

    pub fn save_with_checksum(
        &self,
        session_id: &str,
        state: &ConversationState,
    ) -> Result<[u8; 32]> {
        let bytes = serde_json::to_vec(state)?;
        let checksum: [u8; 32] = Sha256::digest(&bytes).into();
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        tree.insert(session_id.as_bytes(), bytes)?;
        let ck_key = format!("{session_id}:checksum");
        tree.insert(ck_key.as_bytes(), checksum.as_slice())?;
        self.db.flush()?;
        Ok(checksum)
    }

    pub fn load_with_checksum(
        &self,
        session_id: &str,
        expected: &[u8; 32],
    ) -> Result<ConversationState> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        let bytes = tree
            .get(session_id.as_bytes())?
            .ok_or_else(|| anyhow!("Session '{session_id}' not found"))?;
        let actual: [u8; 32] = Sha256::digest(&bytes).into();
        if &actual != expected {
            return Err(anyhow!("Checksum mismatch for session '{session_id}'"));
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn save_manifest(&self, manifest: &SessionManifest) -> Result<()> {
        let tree = self.db.open_tree("meta")?;
        let bytes = serde_json::to_vec(manifest)?;
        tree.insert(manifest.author.as_bytes(), bytes)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn load_manifest(&self, author: &str) -> Result<SessionManifest> {
        let tree = self.db.open_tree("meta")?;
        let bytes = tree
            .get(author.as_bytes())?
            .ok_or_else(|| anyhow!("Manifest for '{author}' not found"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

pub struct BranchRegistry {
    db: Db,
}

impl BranchRegistry {
    const TREE_NAME: &'static str = "branches";

    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn register(&self, branch_id: &str, parent_id: &str) -> Result<()> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        tree.insert(branch_id.as_bytes(), parent_id.as_bytes())?;
        self.db.flush()?;
        Ok(())
    }

    pub fn parent_of(&self, branch_id: &str) -> Result<Option<String>> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        match tree.get(branch_id.as_bytes())? {
            Some(v) => Ok(Some(String::from_utf8(v.to_vec())?)),
            None => Ok(None),
        }
    }

    pub fn list_children(&self, parent_id: &str) -> Result<Vec<String>> {
        let tree = self.db.open_tree(Self::TREE_NAME)?;
        let mut children = Vec::new();
        for item in tree.iter() {
            let (k, v) = item?;
            if v.as_ref() == parent_id.as_bytes() {
                children.push(String::from_utf8(k.to_vec())?);
            }
        }
        Ok(children)
    }
}

