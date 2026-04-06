use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    Pending,
    InProgress,
    Complete,
    Blocked,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoalNode {
    pub id: String,
    pub title: String,
    pub children: Vec<GoalNode>,
    pub status: GoalStatus,
    pub assigned_turn: Option<u32>,
    pub deadline: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GoalTree {
    pub root: Option<GoalNode>,
}
