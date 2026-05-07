use anyhow::Result;
use crate::types::mcp::{
    McpResource, McpTool, JsonRpcRequest, PermissionManager, PermissionTier,
    TimeoutManager,
};
use crate::mcp::bridge::CliBridge;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Critical tools that require explicit confirmation before execution.
const CRITICAL_TOOLS: &[&str] = &["rm", "rmdir", "kill", "shutdown", "reboot", "format"];

pub struct McpGateway {
    tools: HashMap<String, McpTool>,
    pub permissions: PermissionManager,
    pub timeout_manager: TimeoutManager,
    workspace_root: String,
    nix_env: Option<HashMap<String, String>>,
    resources: Vec<McpResource>,
    prompt_templates: Vec<Value>,
    confirmation_override: Option<bool>,
}

impl McpGateway {
    /// Create a gateway with no workspace root (for tests).
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            permissions: PermissionManager::new(),
            timeout_manager: TimeoutManager::new(30, 3),
            workspace_root: ".".to_string(),
            nix_env: None,
            resources: Vec::new(),
            prompt_templates: Vec::new(),
            confirmation_override: None,
        }
    }

    /// Create a gateway rooted at a specific workspace directory.
    pub fn with_workspace(workspace_root: String) -> Self {
        Self {
            workspace_root,
            ..Self::new()
        }
    }

    pub fn register_tool(&mut self, tool: McpTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn add_resource(&mut self, resource: McpResource) {
        self.resources.push(resource);
    }

    pub fn add_prompt_template(&mut self, template: Value) {
        self.prompt_templates.push(template);
    }

    pub fn set_confirmation_override(&mut self, val: Option<bool>) {
        self.confirmation_override = val;
    }

    pub fn handle_initialize(&self) -> Value {
        json!({
            "protocolVersion": "1.0",
            "capabilities": {
                "tools": {},
                "sampling": {},
                "logging": {},
                "resources": { "list": true },
                "prompts": { "list": true }
            },
            "serverInfo": {
                "name": "Crosstalk-MCP-Hub",
                "version": "0.1.0"
            }
        })
    }

    pub fn handle_tools_list(&self, agent_id: &str) -> Value {
        let tier = self.permissions.tiers.get(agent_id);
        if tier.is_none() {
            return json!({ "tools": [] });
        }
        let list: Vec<&McpTool> = self.tools.values().collect();
        json!({ "tools": list })
    }

    /// Dispatch a sampling request through the local MCP Unix-socket transport.
    ///
    /// Connects to `$XDG_RUNTIME_DIR/crosstalk-mcp.sock` (falling back to
    /// `/tmp/crosstalk-mcp.sock`), sends a `sampling/createMessage` JSON-RPC request,
    /// and returns the first text content block from the response.
    /// To wire a real remote worker pool, replace the `UnixStream::connect` call with
    /// pool selection (round-robin or random) against registered worker endpoints.
    pub async fn remote_sampling(prompt: &str, model_id: &str) -> Result<String> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::UnixStream;

        // Prefer XDG_RUNTIME_DIR (user-private, mode 0700) over /tmp.
        let socket_path = std::env::var("XDG_RUNTIME_DIR")
            .map(|d| std::path::PathBuf::from(d).join("crosstalk-mcp.sock"))
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/crosstalk-mcp.sock"));

        // Validate that the socket is owned by the current user before connecting,
        // preventing an attacker from pre-creating the socket path.
        {
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::metadata(&socket_path).map_err(|e| {
                anyhow::anyhow!(
                    "MCP sampling unavailable: socket not found at {} ({}). \
                     Start the MCP server or wire a remote worker pool.",
                    socket_path.display(), e
                )
            })?;
            unsafe extern "C" { fn getuid() -> u32; }
            let current_uid = unsafe { getuid() };
            if meta.uid() != current_uid {
                return Err(anyhow::anyhow!(
                    "MCP socket at {} is not owned by the current user (owner uid={}, current uid={}); \
                     refusing to connect",
                    socket_path.display(), meta.uid(), current_uid
                ));
            }
        }

        tracing::info!(model = %model_id, socket = %socket_path.display(), "dispatching sampling/createMessage via local MCP transport");

        let mut stream = UnixStream::connect(&socket_path).await.map_err(|e| {
            anyhow::anyhow!(
                "MCP sampling unavailable: could not connect to {} ({}). \
                 Start the MCP server or wire a remote worker pool.",
                socket_path.display(), e
            )
        })?;

        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sampling/createMessage",
            "params": {
                "messages": [{"role": "user", "content": {"type": "text", "text": prompt}}],
                "modelPreferences": {"hints": [{"name": model_id}]},
                "maxTokens": 4096
            }
        });

        let mut req_bytes = serde_json::to_vec(&request)?;
        req_bytes.push(b'\n');
        stream.write_all(&req_bytes).await.map_err(|e| {
            anyhow::anyhow!("failed to write sampling request to MCP socket: {}", e)
        })?;

        // Read exactly one newline-delimited JSON response, capped at 1 MiB.
        const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;
        let (reader, _writer) = stream.into_split();
        let mut lines = BufReader::new(tokio::io::AsyncReadExt::take(reader, MAX_RESPONSE_BYTES)).lines();
        let line = lines
            .next_line()
            .await
            .map_err(|e| anyhow::anyhow!("failed to read MCP sampling response: {}", e))?
            .ok_or_else(|| anyhow::anyhow!("MCP socket closed without a response"))?;

        if line.len() > 512 * 1024 {
            tracing::warn!(bytes = line.len(), "large MCP sampling response");
        }

        let response: Value = serde_json::from_str(&line)
            .map_err(|e| anyhow::anyhow!("invalid JSON from MCP socket: {}", e))?;

        // MCP sampling response: result.content[0].text  (or result as a plain string)
        if let Some(err) = response.get("error") {
            return Err(anyhow::anyhow!("MCP sampling error: {}", err));
        }

        let result = response
            .get("result")
            .ok_or_else(|| anyhow::anyhow!("MCP response missing 'result' field: {:?}", response))?;

        // Try structured content array first, then fall back to plain string result
        let text = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| result.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!("MCP sampling result has no extractable text: {:?}", result)
            })?;

        tracing::info!(model = %model_id, chars = text.len(), "received MCP sampling response");
        Ok(text.to_string())
    }

    /// High-level dispatch by method name (used by tests and internal routing).
    pub async fn dispatch(
        &mut self,
        agent_id: &str,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: json!(1),
        };
        self.handle_request(agent_id, req).await
    }

    /// Handle a full JSON-RPC request envelope.
    pub async fn handle_request(
        &mut self,
        agent_id: &str,
        req: JsonRpcRequest,
    ) -> Result<Value> {
        match req.method.as_str() {
            "initialize" => Ok(self.handle_initialize()),
            "tools/list" => Ok(self.handle_tools_list(agent_id)),
            "resources/list" => {
                Ok(json!({ "resources": self.resources }))
            }
            "prompts/list" => {
                Ok(json!({ "prompts": self.prompt_templates }))
            }
            "tools/call" => {
                let name = req.params.get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?
                    .to_string();
                let args = req.params.get("arguments")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Missing tool arguments"))?;
                self.call_tool(agent_id, &name, args).await
            }
            other => Err(anyhow::anyhow!("Method not found: {}", other)),
        }
    }

    async fn call_tool(&mut self, agent_id: &str, name: &str, args: Value) -> Result<Value> {
        // Permission check
        self.permissions
            .check_with_reason(agent_id, name, &args)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Critical-tool confirmation gate
        let needs_confirmation = CRITICAL_TOOLS.contains(&name) || matches!(
            self.permissions.tiers.get(agent_id),
            Some(PermissionTier::CriticalConfirmation(_))
        );
        if needs_confirmation {
            match self.confirmation_override {
                Some(true) => { /* confirmed */ }
                Some(false) => {
                    return Err(anyhow::anyhow!(
                        "Tool {} not confirmed by operator",
                        name
                    ));
                }
                None => {
                    return Err(anyhow::anyhow!(
                        "Tool {} not confirmed by operator",
                        name
                    ));
                }
            }
        }

        // Timeout-disabled check
        if self.timeout_manager.is_disabled(name) {
            return Err(anyhow::anyhow!("Tool {} is disabled due to repeated timeouts", name));
        }

        // Tool lookup
        if !self.tools.contains_key(name) {
            return Err(anyhow::anyhow!("Tool not found: {}", name));
        }

        // Build CLI args
        let mut cli_args = Vec::new();
        if let Some(arr) = args.get("args").and_then(|v| v.as_array()) {
            for v in arr {
                if let Some(s) = v.as_str() {
                    cli_args.push(s.to_string());
                }
            }
        }

        let tool_result = CliBridge::call(name, cli_args, &self.workspace_root).await?;

        Ok(json!({
            "content": [{
                "type": "text",
                "text": if tool_result.success {
                    tool_result.output
                } else {
                    tool_result.error.unwrap_or_else(|| "Unknown error".to_string())
                }
            }],
            "isError": !tool_result.success
        }))
    }

    pub fn set_nix_env(&mut self, env: Option<HashMap<String, String>>) {
        self.nix_env = env;
    }
}

impl Default for McpGateway {
    fn default() -> Self {
        Self::new()
    }
}
