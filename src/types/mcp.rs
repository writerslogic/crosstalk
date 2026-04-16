use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionTier {
    ReadOnly,
    ScopedWrite,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub agent_id: String,
    pub tier: PermissionTier,
    pub allowed_paths: Vec<PathBuf>,
}
