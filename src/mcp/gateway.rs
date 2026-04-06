use crate::mcp::bridge::CliBridge;
use crate::mcp::transport::{JsonRpcRequest, JsonRpcTransport};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use tokio::time::timeout;

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
        Self {
            tiers: HashMap::new(),
        }
    }

    pub fn check(&self, agent_id: &str, tool_name: &str, args: &serde_json::Value) -> bool {
        let tier = self
            .tiers
            .get(agent_id)
            .unwrap_or(&PermissionTier::ReadOnly);
        match tier {
            PermissionTier::Full => true,
            PermissionTier::ReadOnly => {
                // Read-only tools only, no arguments that could be interpreted as write paths
                matches!(
                    tool_name,
                    "git" | "cargo" | "rustc" | "clippy" | "tree-sitter"
                ) && !self.contains_write_patterns(args)
            }
            PermissionTier::ScopedWrite(allowed_paths) => {
                // Check if any path in args is outside allowed paths
                self.validate_paths(args, allowed_paths)
            }
        }
    }

    fn contains_write_patterns(&self, args: &serde_json::Value) -> bool {
        let s = args.to_string();
        s.contains("rm ") || s.contains("mv ") || s.contains(">") || s.contains(">>")
    }

    fn validate_paths(&self, args: &serde_json::Value, allowed: &[PathBuf]) -> bool {
        let mut strings = vec![];
        Self::collect_strings(args, &mut strings);

        for s in strings {
            if s.is_empty() {
                continue;
            }
            let p = Path::new(&s);
            let normalized = Self::normalize_path(p);

            // Check absolute paths
            if p.is_absolute() {
                if !allowed.iter().any(|a| normalized.starts_with(a)) {
                    return false;
                }
            } else if s.contains("..") {
                // For relative paths with .., ensure they don't escape any allowed root
                let mut is_safe = false;
                for base in allowed {
                    let full = base.join(p);
                    let normalized_full = Self::normalize_path(&full);
                    if normalized_full.starts_with(base) {
                        is_safe = true;
                        break;
                    }
                }
                if !is_safe {
                    return false;
                }
            }
        }
        true
    }

    fn collect_strings(val: &serde_json::Value, out: &mut Vec<String>) {
        match val {
            serde_json::Value::String(s) => out.push(s.clone()),
            serde_json::Value::Array(arr) => {
                for v in arr {
                    Self::collect_strings(v, out);
                }
            }
            serde_json::Value::Object(obj) => {
                for v in obj.values() {
                    Self::collect_strings(v, out);
                }
            }
            _ => {}
        }
    }

    fn normalize_path(path: &Path) -> PathBuf {
        use std::path::Component;
        let mut components = vec![];
        for component in path.components() {
            match component {
                Component::Normal(c) => components.push(Component::Normal(c)),
                Component::ParentDir => {
                    components.pop();
                }
                Component::CurDir => {}
                Component::RootDir => {
                    components.clear();
                    components.push(Component::RootDir);
                }
                Component::Prefix(p) => {
                    components.clear();
                    components.push(Component::Prefix(p));
                }
            }
        }
        if components.is_empty() {
            PathBuf::from(".")
        } else {
            components.into_iter().collect()
        }
    }
}

pub struct McpGateway {
    pub tools: HashMap<String, McpTool>,
    pub permissions: PermissionManager,
    pub tool_timeouts: HashMap<String, u32>,
    pub disabled_tools: Vec<String>,
}

impl McpGateway {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            permissions: PermissionManager::new(),
            tool_timeouts: HashMap::new(),
            disabled_tools: vec![],
        }
    }

    pub fn register_tool(&mut self, tool: McpTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub async fn run_server(self_arc: Arc<Mutex<Self>>, socket_path: &str) -> Result<()> {
        let _ = std::fs::remove_file(socket_path);
        let listener = UnixListener::bind(socket_path)?;
        println!("[mcp] Gateway listening on {}", socket_path);

        loop {
            let (stream, _) = listener.accept().await?;
            let gateway = self_arc.clone();
            tokio::spawn(async move {
                let mut transport = JsonRpcTransport::new(stream);
                while let Ok(Some(msg)) = transport.next_message().await {
                    if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(&msg) {
                        let mut g = gateway.lock().await;
                        let result = g.handle_request("default_agent", req).await;
                        match result {
                            Ok(res) => {
                                let _ = transport
                                    .send_response(Some(res), None, serde_json::json!(0))
                                    .await;
                            }
                            Err(e) => {
                                let _ = transport
                                    .send_response(
                                        None,
                                        Some(serde_json::json!(e.to_string())),
                                        serde_json::json!(0),
                                    )
                                    .await;
                            }
                        }
                    }
                }
            });
        }
    }

    pub async fn handle_request(
        &mut self,
        agent_id: &str,
        req: JsonRpcRequest,
    ) -> Result<serde_json::Value> {
        match req.method.as_str() {
            "initialize" => Ok(self.handle_initialize()),
            "tools/list" => Ok(self.handle_tools_list(agent_id)),
            "tools/call" => {
                let name = req.params["name"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing tool name"))?;
                let args = req.params["arguments"].clone();
                let res = self.handle_tool_call(agent_id, name, args).await?;
                Ok(serde_json::to_value(res)?)
            }
            _ => Err(anyhow!("Method not found: {}", req.method)),
        }
    }

    pub fn handle_initialize(&self) -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "Crosstalk-MCP-Hub", "version": "0.1.0" }
        })
    }

    pub fn handle_tools_list(&self, agent_id: &str) -> serde_json::Value {
        let tools: Vec<&McpTool> = self
            .tools
            .values()
            .filter(|t| !self.disabled_tools.contains(&t.name))
            .filter(|t| {
                self.permissions
                    .check(agent_id, &t.name, &serde_json::json!({}))
            })
            .collect();
        serde_json::json!({ "tools": tools })
    }

    pub async fn handle_tool_call(
        &mut self,
        agent_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult> {
        if self.disabled_tools.contains(&name.to_string()) {
            return Err(anyhow!(
                "Tool {} is disabled due to repeated timeouts",
                name
            ));
        }

        if !self.permissions.check(agent_id, name, &args) {
            return Err(anyhow!(
                "Permission denied for tool: {} with args: {:?}",
                name,
                args
            ));
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
        let fut = async move { CliBridge::invoke(&bin, cli_args) };

        match timeout(Duration::from_secs(60), fut).await {
            Ok(res) => {
                // Success! Reset timeout count
                self.tool_timeouts.remove(name);
                res
            }
            Err(_) => {
                let count = self.tool_timeouts.entry(name.to_string()).or_insert(0);
                *count += 1;
                if *count >= 3 {
                    println!(
                        "[mcp] Disabling tool {} due to {} consecutive timeouts",
                        name, count
                    );
                    self.disabled_tools.push(name.to_string());
                }
                Ok(ToolResult {
                    tool_name: name.to_string(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("Timeout after 60s (Occurrence {})", count)),
                    elapsed_ms: 60000,
                })
            }
        }
    }
}

impl Default for McpGateway {
    fn default() -> Self {
        Self::new()
    }
}
