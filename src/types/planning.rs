use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SessionManifest {
    pub author: String,
    pub created_at: u64,
    pub total_turns: u32,
    pub total_tokens_used: u64,
    pub cost_estimate_usd: f64,
}

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
    /// Explicit prerequisite goal IDs (for DAG scheduling beyond the tree structure).
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub difficulty_score: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct GoalTreeAnalysis {
    pub depth: usize,
    pub leaf_count: usize,
    pub total_count: usize,
    pub completion_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Milestone {
    pub id: String,
    pub title: String,
    pub triggered_at: u64,
    pub completion_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GoalTree {
    pub root: Option<GoalNode>,
}

impl GoalTree {
    /// Add `node` as a direct child of the goal with `parent_id`.
    /// Returns `true` if the parent was found and the node was inserted.
    pub fn add_child(&mut self, parent_id: &str, node: GoalNode) -> bool {
        match &mut self.root {
            Some(root) => Self::insert_child(root, parent_id, &mut Some(node)),
            None => false,
        }
    }

    fn insert_child(current: &mut GoalNode, parent_id: &str, node: &mut Option<GoalNode>) -> bool {
        if current.id == parent_id {
            if let Some(n) = node.take() {
                current.children.push(n);
            }
            return true;
        }
        for child in &mut current.children {
            if Self::insert_child(child, parent_id, node) {
                return true;
            }
        }
        false
    }

    /// Collect all leaf nodes (nodes with no children).
    #[must_use]
    pub fn get_leaves(&self) -> Vec<&GoalNode> {
        let mut leaves = Vec::new();
        if let Some(root) = &self.root {
            Self::collect_leaves(root, &mut leaves);
        }
        leaves
    }

    fn collect_leaves<'a>(node: &'a GoalNode, out: &mut Vec<&'a GoalNode>) {
        if node.children.is_empty() {
            out.push(node);
        } else {
            for child in &node.children {
                Self::collect_leaves(child, out);
            }
        }
    }

    /// Find the subtree rooted at the node with `id`. Returns `None` if not found.
    #[must_use]
    pub fn get_subtree(&self, id: &str) -> Option<&GoalNode> {
        self.root.as_ref().and_then(|r| Self::find_node(r, id))
    }

    pub fn find_node<'a>(node: &'a GoalNode, id: &str) -> Option<&'a GoalNode> {
        if node.id == id {
            return Some(node);
        }
        node.children.iter().find_map(|c| Self::find_node(c, id))
    }

    /// Compute structural and completion statistics for the tree.
    #[must_use]
    pub fn analyze(&self) -> GoalTreeAnalysis {
        match &self.root {
            None => GoalTreeAnalysis {
                depth: 0,
                leaf_count: 0,
                total_count: 0,
                completion_ratio: 0.0,
            },
            Some(root) => {
                let depth = Self::compute_depth(root);
                let mut total = 0usize;
                let mut complete = 0usize;
                let mut leaves = 0usize;
                Self::collect_stats(root, &mut total, &mut complete, &mut leaves);
                let completion_ratio = if total == 0 {
                    0.0
                } else {
                    complete as f64 / total as f64
                };
                GoalTreeAnalysis {
                    depth,
                    leaf_count: leaves,
                    total_count: total,
                    completion_ratio,
                }
            }
        }
    }

    fn compute_depth(node: &GoalNode) -> usize {
        if node.children.is_empty() {
            return 1;
        }
        1 + node
            .children
            .iter()
            .map(Self::compute_depth)
            .max()
            .unwrap_or(0)
    }

    fn collect_stats(node: &GoalNode, total: &mut usize, complete: &mut usize, leaves: &mut usize) {
        *total += 1;
        if node.status == GoalStatus::Complete {
            *complete += 1;
        }
        if node.children.is_empty() {
            *leaves += 1;
        }
        for child in &node.children {
            Self::collect_stats(child, total, complete, leaves);
        }
    }

    /// Fraction of nodes with `Complete` status.
    #[must_use]
    pub fn completion_ratio(&self) -> f64 {
        self.analyze().completion_ratio
    }

    /// Walk the dependency ID graph and return `Err` if a cycle is detected.
    ///
    /// The tree structure (`children`) cannot form cycles because nodes are
    /// owned values.  However, `GoalNode::dependencies` holds String IDs that
    /// CAN reference ancestors, creating a logical cycle.  This method builds
    /// the dependency graph from all nodes in the tree and runs a DFS to detect
    /// back-edges.
    pub fn validate_acyclic(&self) -> Result<(), String> {
        // Collect all node IDs and their dependency edges.
        let mut dep_graph: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        if let Some(root) = &self.root {
            Self::collect_dep_graph(root, &mut dep_graph);
        }
        // DFS cycle detection using a visited set and a per-path stack set.
        let mut visited = std::collections::HashSet::new();
        let mut in_stack = std::collections::HashSet::new();
        for id in dep_graph.keys().cloned().collect::<Vec<_>>() {
            if Self::dfs_cycle(&id, &dep_graph, &mut visited, &mut in_stack) {
                return Err(format!("dependency cycle detected involving goal '{id}'"));
            }
        }
        Ok(())
    }

    fn collect_dep_graph(
        node: &GoalNode,
        graph: &mut std::collections::HashMap<String, Vec<String>>,
    ) {
        graph.insert(node.id.clone(), node.dependencies.clone());
        for child in &node.children {
            Self::collect_dep_graph(child, graph);
        }
    }

    fn dfs_cycle(
        id: &str,
        graph: &std::collections::HashMap<String, Vec<String>>,
        visited: &mut std::collections::HashSet<String>,
        in_stack: &mut std::collections::HashSet<String>,
    ) -> bool {
        if in_stack.contains(id) {
            return true;
        }
        if visited.contains(id) {
            return false;
        }
        visited.insert(id.to_string());
        in_stack.insert(id.to_string());
        if let Some(deps) = graph.get(id) {
            for dep in deps {
                if Self::dfs_cycle(dep, graph, visited, in_stack) {
                    return true;
                }
            }
        }
        in_stack.remove(id);
        false
    }
}
