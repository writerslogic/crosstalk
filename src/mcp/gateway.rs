use crate::mcp::bridge::CliBridge;
use crate::mcp::transport::{JsonRpcRequest, JsonRpcTransport};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
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
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum PermissionError {
    #[error("Tool '{0}' is not allowed for this permission tier")]
    ToolNotAllowed(String),
    #[error("Write operation blocked: detected write pattern in args")]
    WriteBlocked(String),
    #[error("Attempted path traversal: {0} blocked")]
    PathTraversal(String),
    #[error("Path '{0}' is outside allowed directories")]
    PathOutOfScope(String),
    #[error("Agent '{0}' has been disabled after repeated permission violations")]
    AgentDisabled(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionTier {
    ReadOnly,
    ScopedWrite(Vec<PathBuf>),
    /// Tools in the Vec are the only ones allowed; each call requires user confirmation.
    CriticalConfirmation(Vec<String>),
    Full,
}

#[derive(Default)]
pub struct PermissionManager {
    pub tiers: HashMap<String, PermissionTier>,
    pub failed_checks: HashMap<String, u32>,
    pub disabled_agents: Vec<String>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stateless check used for tool listing. Does not record failures.
    pub fn check(&self, agent_id: &str, tool_name: &str, args: &serde_json::Value) -> bool {
        if self.disabled_agents.contains(&agent_id.to_string()) {
            return false;
        }
        self.check_inner(agent_id, tool_name, args).is_ok()
    }

    /// Stateful check used for actual tool calls. Records failures and disables agents after 5.
    pub fn check_with_reason(
        &mut self,
        agent_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<(), PermissionError> {
        if self.disabled_agents.contains(&agent_id.to_string()) {
            return Err(PermissionError::AgentDisabled(agent_id.to_string()));
        }

        let result = self.check_inner(agent_id, tool_name, args);

        if result.is_err() {
            let count = self.failed_checks.entry(agent_id.to_string()).or_insert(0);
            *count += 1;
            if *count >= 5 && !self.disabled_agents.contains(&agent_id.to_string()) {
                self.disabled_agents.push(agent_id.to_string());
            }
        }

        result
    }

    fn check_inner(
        &self,
        agent_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<(), PermissionError> {
        let tier = self.tiers.get(agent_id).unwrap_or(&PermissionTier::ReadOnly);
        match tier {
            PermissionTier::Full => Ok(()),
            PermissionTier::ReadOnly => {
                if !matches!(
                    tool_name,
                    "git" | "cargo" | "rustc" | "clippy" | "tree-sitter"
                ) {
                    return Err(PermissionError::ToolNotAllowed(tool_name.to_string()));
                }
                if self.contains_write_patterns(args) {
                    return Err(PermissionError::WriteBlocked(
                        "detected shell redirect or destructive operator".to_string(),
                    ));
                }
                Ok(())
            }
            PermissionTier::ScopedWrite(allowed_paths) => {
                self.validate_paths_with_error(args, allowed_paths)
            }
            PermissionTier::CriticalConfirmation(allowed_tools) => {
                if !allowed_tools.contains(&tool_name.to_string()) {
                    return Err(PermissionError::ToolNotAllowed(tool_name.to_string()));
                }
                Ok(())
            }
        }
    }

    /// Strict path validation returning a typed error with violation details.
    pub fn validate_path_strict(
        path: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<(), PermissionError> {
        let path_str = path.to_string_lossy().to_string();

        if path_str.contains("..") {
            if path.is_absolute() {
                let normalized = Self::normalize_path(path);
                if !allowed_dirs.iter().any(|a| normalized.starts_with(a)) {
                    return Err(PermissionError::PathTraversal(path_str));
                }
            } else {
                let mut is_safe = false;
                for base in allowed_dirs {
                    let full = base.join(path);
                    let normalized_full = Self::normalize_path(&full);
                    if normalized_full.starts_with(base) {
                        is_safe = true;
                        break;
                    }
                }
                if !is_safe {
                    return Err(PermissionError::PathTraversal(path_str));
                }
            }
            return Ok(());
        }

        if path.is_absolute() {
            let normalized = Self::normalize_path(path);
            if !allowed_dirs.iter().any(|a| normalized.starts_with(a)) {
                return Err(PermissionError::PathOutOfScope(path_str));
            }
        }

        Ok(())
    }

    fn validate_paths_with_error(
        &self,
        args: &serde_json::Value,
        allowed: &[PathBuf],
    ) -> Result<(), PermissionError> {
        let mut strings = vec![];
        Self::collect_strings(args, &mut strings);

        for s in strings {
            if s.is_empty() {
                continue;
            }
            let p = Path::new(&s);
            Self::validate_path_strict(p, allowed).map_err(|e| match e {
                PermissionError::PathTraversal(_) => {
                    PermissionError::PathTraversal(format!(
                        "Attempted path traversal: {} blocked",
                        s
                    ))
                }
                PermissionError::PathOutOfScope(_) => PermissionError::PathOutOfScope(format!(
                    "Path '{}' is outside allowed directories",
                    s
                )),
                other => other,
            })?;
        }
        Ok(())
    }

    fn contains_write_patterns(&self, args: &serde_json::Value) -> bool {
        let s = args.to_string();
        s.contains("rm ") || s.contains("mv ") || s.contains('>') || s.contains(">>")
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
        let mut components: Vec<Component> = vec![];
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

/// Tools that always require user confirmation before execution.
/// Format: (binary_name, optional_subcommand_that_triggers_confirmation)
static CRITICAL_TOOLS: &[(&str, Option<&str>)] = &[
    ("rm", None),
    ("git", Some("push")),
    ("cargo", Some("clean")),
];

pub struct McpGateway {
    pub tools: HashMap<String, McpTool>,
    pub permissions: PermissionManager,
    pub tool_timeouts: HashMap<String, u32>,
    pub disabled_tools: Vec<String>,
    pub resources: Vec<McpResource>,
    pub prompt_templates: Vec<serde_json::Value>,
    /// Controls critical-tool confirmation behaviour.
    /// None = prompt via stdin (production), Some(true) = auto-approve, Some(false) = auto-deny.
    pub confirmation_override: Option<bool>,
}

impl McpGateway {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            permissions: PermissionManager::new(),
            tool_timeouts: HashMap::new(),
            disabled_tools: vec![],
            resources: vec![],
            prompt_templates: vec![],
            confirmation_override: None,
        }
    }

    pub fn register_tool(&mut self, tool: McpTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn add_resource(&mut self, resource: McpResource) {
        self.resources.push(resource);
    }

    pub fn add_prompt_template(&mut self, template: serde_json::Value) {
        self.prompt_templates.push(template);
    }

    pub async fn run_server(self_arc: Arc<Mutex<Self>>, socket_path: &str) -> Result<()> {
        let _ = std::fs::remove_file(socket_path);
        let listener = UnixListener::bind(socket_path)
            .map_err(|e| anyhow!("Failed to bind Unix socket at {}: {}", socket_path, e))?;

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
            "resources/list" => Ok(self.handle_resources_list()),
            "prompts/list" => Ok(self.handle_prompts_list()),
            _ => Err(anyhow!("Method not found: {}", req.method)),
        }
    }

    pub fn handle_initialize(&self) -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": "1.0",
            "capabilities": {
                "sampling": {},
                "logging": {},
                "resources": { "list": true },
                "prompts": { "list": true },
                "tools": {}
            },
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

    pub fn handle_resources_list(&self) -> serde_json::Value {
        serde_json::json!({ "resources": self.resources })
    }

    pub fn handle_prompts_list(&self) -> serde_json::Value {
        serde_json::json!({ "prompts": self.prompt_templates })
    }

    fn is_critical_tool(&self, name: &str, args: &serde_json::Value) -> bool {
        for (tool, subcommand) in CRITICAL_TOOLS {
            if name == *tool {
                match subcommand {
                    None => return true,
                    Some(sub) => {
                        if args.to_string().contains(sub) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    async fn prompt_confirmation(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        match self.confirmation_override {
            Some(v) => v,
            None => {
                print!(
                    "[mcp] CRITICAL: '{}' with args {} requires confirmation. Proceed? (y/N): ",
                    tool_name, args
                );
                std::io::stdout().flush().ok();
                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).is_err() {
                    return false;
                }
                input.trim().eq_ignore_ascii_case("y")
            }
        }
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

        if let Err(e) = self.permissions.check_with_reason(agent_id, name, &args) {
            return Err(anyhow!("Permission denied: {}", e));
        }

        // Confirmation required for: universally critical tools (rm, git push, cargo clean)
        // and any tool called by a CriticalConfirmation-tier agent.
        let needs_confirmation = self.is_critical_tool(name, &args)
            || matches!(
                self.permissions.tiers.get(agent_id),
                Some(PermissionTier::CriticalConfirmation(_))
            );

        if needs_confirmation && !self.prompt_confirmation(name, &args).await {
            return Err(anyhow!(
                "Execution of critical tool '{}' was not confirmed by user",
                name
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
        let fut = async move { CliBridge::invoke(&bin, cli_args, None) };

        match timeout(Duration::from_secs(60), fut).await {
            Ok(res) => {
                self.tool_timeouts.remove(name);
                res
            }
            Err(_) => {
                let count = self.tool_timeouts.entry(name.to_string()).or_insert(0);
                *count += 1;
                if *count >= 3 {
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
