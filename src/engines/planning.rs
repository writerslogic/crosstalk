use crate::types::conversation::{ConversationState, Turn};
use crate::types::planning::{GoalNode, GoalStatus};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub struct PlanningEngine;

impl PlanningEngine {
    pub fn update_goal_status(node: &mut GoalNode) {
        if node.children.is_empty() {
            return;
        }

        for child in &mut node.children {
            Self::update_goal_status(child);
        }

        let all_complete = node
            .children
            .iter()
            .all(|c| c.status == GoalStatus::Complete);
        let any_blocked = node
            .children
            .iter()
            .any(|c| c.status == GoalStatus::Blocked);
        let any_in_progress = node
            .children
            .iter()
            .any(|c| c.status == GoalStatus::InProgress || c.status == GoalStatus::Complete);

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
        let now = ConversationState::now();
        fork.session_id = format!("{}-fork-{}", sigma.session_id, now);
        // Lineage tracking could be added to metadata
        fork
    }
}

#[derive(Eq, PartialEq)]
struct PrunableGoal {
    id: String,
    criticality: u32,
}

impl Ord for PrunableGoal {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (lowest criticality first)
        other.criticality.cmp(&self.criticality)
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
        let mut heap = BinaryHeap::new();
        let now = ConversationState::now();

        // Populate heap with goals and their real criticality
        if let Some(ref root) = sigma.goal_tree.root {
            Self::populate_heap(root, 0, now, &mut heap);
        }

        // Pruning logic: prioritize keeping turns associated with high-criticality goals
        // and always keep the most recent N turns.
        let mut critical_turn_indices = std::collections::HashSet::new();

        // Take top 5 most critical goals and protect their assigned turns
        let mut count = 0;
        while let Some(goal) = heap.pop() {
            if count >= 5 {
                break;
            }
            if let Some(ref root) = sigma.goal_tree.root
                && let Some(node) = Self::find_node(root, &goal.id)
                && let Some(turn_idx) = node.assigned_turn
            {
                critical_turn_indices.insert(turn_idx);
            }
            count += 1;
        }

        sigma
            .turns
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let is_recent = sigma.turns.len().saturating_sub(*idx) <= max_turns;
                let is_critical = critical_turn_indices.contains(&(*idx as u32));
                is_recent || is_critical
            })
            .map(|(_, turn)| turn.clone())
            .collect()
    }

    fn populate_heap(node: &GoalNode, depth: u32, now: u64, heap: &mut BinaryHeap<PrunableGoal>) {
        let criticality = Self::calculate_criticality(node, depth, now);
        heap.push(PrunableGoal {
            id: node.id.clone(),
            criticality,
        });
        for child in &node.children {
            Self::populate_heap(child, depth + 1, now, heap);
        }
    }

    fn calculate_criticality(node: &GoalNode, depth: u32, now: u64) -> u32 {
        // Higher depth = more specific = higher base criticality (dependency depth)
        let depth_factor = depth * 100;

        // Time-to-deadline urgency
        let urgency_factor = match node.deadline {
            Some(d) => {
                if d <= now {
                    5000 // Overdue is highly critical
                } else {
                    let diff = d - now;
                    // Scale: 1 hour = 2400, 24 hours = 100
                    (86400 / diff.max(1)).min(4000) as u32
                }
            }
            None => 0,
        };

        depth_factor + urgency_factor
    }

    fn find_node<'a>(node: &'a GoalNode, id: &str) -> Option<&'a GoalNode> {
        if node.id == id {
            return Some(node);
        }
        for child in &node.children {
            if let Some(found) = Self::find_node(child, id) {
                return Some(found);
            }
        }
        None
    }
}
