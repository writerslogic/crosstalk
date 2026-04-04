use crate::types::{GoalNode, GoalStatus, ConversationState, Turn};
use std::cmp::Ordering;

pub struct PlanningEngine;

impl PlanningEngine {
    pub fn update_goal_status(node: &mut GoalNode) {
        if node.children.is_empty() { return; }

        for child in &mut node.children {
            Self::update_goal_status(child);
        }

        let all_complete = node.children.iter().all(|c| c.status == GoalStatus::Complete);
        let any_in_progress = node.children.iter().any(|c| c.status == GoalStatus::InProgress || c.status == GoalStatus::Complete);

        if all_complete {
            node.status = GoalStatus::Complete;
        } else if any_in_progress {
            node.status = GoalStatus::InProgress;
        }
    }
}

pub struct BranchManager;

impl BranchManager {
    pub fn fork(sigma: &ConversationState) -> ConversationState {
        let mut fork = sigma.clone();
        fork.session_id = format!("{}-fork-{}", sigma.session_id, ConversationState::now());
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
        other.criticality.cmp(&self.criticality) // Min-heap for lowest criticality first
    }
}

impl PartialOrd for PrunableGoal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ContextPruner;

impl ContextPruner {
    pub fn prune(sigma: &ConversationState, _active_goal_id: &str, max_turns: usize) -> Vec<Turn> {
        // Simple pruning: keep last N turns
        sigma.turns.iter().rev().take(max_turns).cloned().rev().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_propagation() {
        let child = GoalNode {
            id: "1.1".into(),
            title: "child".into(),
            children: vec![],
            status: GoalStatus::Complete,
            assigned_turn: None,
        };
        let mut root = GoalNode {
            id: "1".into(),
            title: "root".into(),
            children: vec![child],
            status: GoalStatus::Pending,
            assigned_turn: None,
        };
        PlanningEngine::update_goal_status(&mut root);
        assert_eq!(root.status, GoalStatus::Complete);
    }

    #[test]
    fn test_branch_fork() {
        let sigma = ConversationState::new("base");
        let fork = BranchManager::fork(&sigma);
        assert!(fork.session_id.starts_with("base-fork-"));
    }
}
