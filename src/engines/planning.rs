use crate::types::conversation::{ConversationState, Turn};
use crate::types::planning::{GoalNode, GoalStatus};
use std::cmp::Ordering;

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
        let mut any_in_progress = false;

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
                    any_in_progress = true;
                    all_complete = false;
                }
                GoalStatus::Complete => {
                    any_in_progress = true; // Completed children imply work was done
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
        } else if any_in_progress {
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