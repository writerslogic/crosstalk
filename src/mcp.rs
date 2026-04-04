use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::timeout;
use crate::environment::CliBridge;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionTier {
    ReadOnly,
    ScopedWrite(Vec<PathBuf>),
    Full,
}

pub struct PermissionManager {
    pub tiers: HashMap<String, PermissionTier>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self { tiers: HashMap::new() }
    }

    pub fn check(&self, agent_id: &str, tool_name: &str) -> bool {
        let tier = self.tiers.get(agent_id).unwrap_or(&PermissionTier::ReadOnly);
        match tier {
            PermissionTier::Full => true,
            PermissionTier::ReadOnly => {
                // Read-only tools
                matches!(tool_name, "git" | "cargo")
            }
            PermissionTier::ScopedWrite(_) => true,
        }
    }
}

pub struct McpGateway {
    pub tools: HashMap<String, McpTool>,
    pub permissions: PermissionManager,
}

impl McpGateway {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            permissions: PermissionManager::new(),
        }
    }

    pub fn register_tool(&mut self, tool: McpTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn handle_initialize(&self) -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "Crosstalk-MCP-Hub", "version": "0.1.0" }
        })
    }

    pub fn handle_tools_list(&self, agent_id: &str) -> serde_json::Value {
        let tools: Vec<&McpTool> = self.tools.values()
            .filter(|t| self.permissions.check(agent_id, &t.name))
            .collect();
        serde_json::json!({ "tools": tools })
    }

    pub async fn handle_tool_call(&self, agent_id: &str, name: &str, args: serde_json::Value) -> Result<ToolResult> {
        if !self.permissions.check(agent_id, name) {
            return Err(anyhow!("Permission denied for tool: {}", name));
        }

        if !self.tools.contains_key(name) {
            return Err(anyhow!("Tool not found: {}", name));
        }

        let cli_args: Vec<String> = args["args"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();

        let bin = name.to_string();
        
        let fut = async move {
            CliBridge::invoke(&bin, cli_args)
        };

        match timeout(Duration::from_secs(60), fut).await {
            Ok(res) => res,
            Err(_) => Ok(ToolResult {
                tool_name: name.to_string(),
                success: false,
                output: String::new(),
                error: Some("Timeout after 60s".to_string()),
                elapsed_ms: 60000,
            }),
        }
    }
}

impl Default for McpGateway {
    fn default() -> Self {
        Self::new()
    }
}
