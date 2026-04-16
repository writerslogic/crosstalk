use anyhow::Result;
use crate::types::mcp::{McpTool, ToolResult, Permission, PermissionTier};
use crate::mcp::bridge::CliBridge;
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::sync::Mutex;
use std::sync::Arc;

pub struct McpGateway {
    tools: Arc<Mutex<HashMap<String, McpTool>>>,
    permissions: Arc<Mutex<HashMap<String, Permission>>>,
    workspace_root: String,
    nix_env: Arc<Mutex<Option<std::collections::HashMap<String, String>>>>,
}

impl McpGateway {
    pub fn new(workspace_root: String) -> Self {
        Self {
            tools: Arc::new(Mutex::new(HashMap::new())),
            permissions: Arc::new(Mutex::new(HashMap::new())),
            workspace_root,
            nix_env: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn handle_request(&self, agent_id: &str, method: &str, params: Value) -> Result<Value> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "Crosstalk Gateway",
                    "version": "0.1.0"
                }
            })),
            "tools/list" => {
                let tools = self.tools.lock().await;
                let list: Vec<McpTool> = tools.values().cloned().collect();
                Ok(json!({ "tools": list }))
            },
            "tools/call" => {
                let name = params.get("name").and_then(|v| v.as_str()).ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                
                self.call_tool(agent_id, name, args).await
            },
            _ => Err(anyhow::anyhow!("Method not found: {}", method)),
        }
    }

    async fn call_tool(&self, agent_id: &str, name: &str, args: Value) -> Result<Value> {
        // --- Track 07-D: Access Scoping ---
        let perms = self.permissions.lock().await;
        if let Some(p) = perms.get(agent_id) {
            if p.tier == PermissionTier::ReadOnly && (name.contains("build") || name.contains("write")) {
                return Err(anyhow::anyhow!("Permission denied: Agent {} cannot execute write tool {}", agent_id, name));
            }
        }

        // --- Track 07-G: CLI Bridges ---
        let mut cli_args = Vec::new();
        if let Some(arr) = args.get("args").and_then(|v| v.as_array()) {
            for v in arr {
                if let Some(s) = v.as_str() {
                    cli_args.push(s.to_string());
                }
            }
        }

        let result = CliBridge::call(name, cli_args, &self.workspace_root).await?;
        
        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": if result.success { result.output } else { result.error.unwrap_or_else(|| "Unknown error".to_string()) }
                }
            ],
            "isError": !result.success
        }))
    }


    pub async fn set_nix_env(&self, env: Option<std::collections::HashMap<String, String>>) {
        *self.nix_env.lock().await = env;
    }

    pub async fn register_tool(&self, tool: McpTool) {
        self.tools.lock().await.insert(tool.name.clone(), tool);
    }
}
