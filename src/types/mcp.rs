use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionTier {
    ReadOnly,
    ScopedWrite(Vec<PathBuf>),
    Full,
    CriticalConfirmation(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub agent_id: String,
    pub tier: PermissionTier,
    pub allowed_paths: Vec<PathBuf>,
}

// --- PermissionError ---

#[derive(Debug, thiserror::Error)]
pub enum PermissionError {
    #[error("Path traversal detected: {0}")]
    PathTraversal(String),

    #[error("Path out of scope: {0}")]
    PathOutOfScope(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Agent disabled: {0}")]
    AgentDisabled(String),

    #[error("Confirmation not confirmed: {0}")]
    NotConfirmed(String),
}

// --- PermissionManager ---

#[derive(Debug, Clone)]
pub struct PermissionManager {
    pub tiers: HashMap<String, PermissionTier>,
    pub disabled_agents: HashSet<String>,
    failure_counts: HashMap<String, u32>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self {
            tiers: HashMap::new(),
            disabled_agents: HashSet::new(),
            failure_counts: HashMap::new(),
        }
    }

    /// Check whether `agent_id` may call `tool_name` with `args`.
    /// Returns Ok(()) on success or Err(PermissionError) on denial.
    /// Tracks consecutive failures and disables agents after 5.
    pub fn check_with_reason(
        &mut self,
        agent_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<(), PermissionError> {
        if self.disabled_agents.contains(agent_id) {
            return Err(PermissionError::AgentDisabled(agent_id.to_string()));
        }

        let tier = match self.tiers.get(agent_id) {
            Some(t) => t.clone(),
            None => {
                self.record_failure(agent_id);
                return Err(PermissionError::PermissionDenied(format!(
                    "Agent {} has no permissions configured",
                    agent_id
                )));
            }
        };

        match &tier {
            PermissionTier::ReadOnly => {
                // ReadOnly agents cannot call write-like tools
                self.record_failure(agent_id);
                Err(PermissionError::PermissionDenied(format!(
                    "Agent {} (ReadOnly) cannot call tool {}",
                    agent_id, tool_name
                )))
            }
            PermissionTier::ScopedWrite(allowed) => {
                // Check that all path arguments are within allowed paths
                if let Some(arr) = args.get("args").and_then(|v| v.as_array()) {
                    if arr.len() > 100 {
                        return Err(PermissionError::PermissionDenied(format!(
                            "Too many arguments ({}) for agent {}",
                            arr.len(),
                            agent_id
                        )));
                    }
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            let p = Path::new(s);
                            Self::validate_path_strict(p, allowed).map_err(|e| {
                                self.record_failure(agent_id);
                                match e {
                                    PermissionError::PathTraversal(_) => {
                                        PermissionError::PermissionDenied(format!(
                                            "Permission denied for agent {}: {}",
                                            agent_id, s
                                        ))
                                    }
                                    PermissionError::PathOutOfScope(_) => {
                                        PermissionError::PermissionDenied(format!(
                                            "Permission denied for agent {}: {}",
                                            agent_id, s
                                        ))
                                    }
                                    other => other,
                                }
                            })?;
                        }
                    }
                }
                self.failure_counts.remove(agent_id);
                Ok(())
            }
            PermissionTier::Full => {
                self.failure_counts.remove(agent_id);
                Ok(())
            }
            PermissionTier::CriticalConfirmation(allowed_tools) => {
                if !allowed_tools.contains(&tool_name.to_string()) {
                    self.record_failure(agent_id);
                    return Err(PermissionError::PermissionDenied(format!(
                        "Permission denied: tool {} not in allowed list for agent {}",
                        tool_name, agent_id
                    )));
                }
                // Confirmation check is handled at the gateway level
                self.failure_counts.remove(agent_id);
                Ok(())
            }
        }
    }

    /// Validate that a path does not contain traversal and is within allowed directories.
    pub fn validate_path_strict(path: &Path, allowed: &[PathBuf]) -> Result<(), PermissionError> {
        let path_str = path.to_string_lossy();

        // Check for path traversal
        if path_str.contains("..") {
            return Err(PermissionError::PathTraversal(path_str.to_string()));
        }

        // Check that path is within at least one allowed directory
        for dir in allowed {
            if path.starts_with(dir) {
                return Ok(());
            }
        }

        Err(PermissionError::PathOutOfScope(path_str.to_string()))
    }

    fn record_failure(&mut self, agent_id: &str) {
        let count = self.failure_counts.entry(agent_id.to_string()).or_insert(0);
        *count += 1;
        if *count >= 5 {
            self.disabled_agents.insert(agent_id.to_string());
        }
    }
}

impl Default for PermissionManager {
    fn default() -> Self {
        Self::new()
    }
}

// --- TimeoutManager ---

#[derive(Debug, Clone)]
pub struct TimeoutManager {
    default_timeout_secs: u64,
    max_failures: u32,
    failure_counts: HashMap<String, u32>,
    disabled: HashSet<String>,
    per_tool_timeout: HashMap<String, u64>,
}

impl TimeoutManager {
    pub fn new(default_timeout_secs: u64, max_failures: u32) -> Self {
        Self {
            default_timeout_secs,
            max_failures,
            failure_counts: HashMap::new(),
            disabled: HashSet::new(),
            per_tool_timeout: HashMap::new(),
        }
    }

    /// Record a timeout for a tool. Returns true if the tool is now disabled.
    pub fn record_timeout(&mut self, tool: &str) -> bool {
        let count = self.failure_counts.entry(tool.to_string()).or_insert(0);
        *count += 1;
        if *count >= self.max_failures {
            self.disabled.insert(tool.to_string());
            true
        } else {
            false
        }
    }

    /// Record a successful call, resetting the failure count.
    pub fn record_success(&mut self, tool: &str) {
        self.failure_counts.remove(tool);
        self.disabled.remove(tool);
    }

    /// Check if a tool has been disabled due to repeated timeouts.
    pub fn is_disabled(&self, tool: &str) -> bool {
        self.disabled.contains(tool)
    }

    /// Get the current failure count for a tool.
    pub fn failure_count(&self, tool: &str) -> u32 {
        self.failure_counts.get(tool).copied().unwrap_or(0)
    }

    /// Set a per-tool timeout override.
    pub fn set_timeout(&mut self, tool: &str, secs: u64) {
        self.per_tool_timeout.insert(tool.to_string(), secs);
    }

    /// Get the effective timeout duration for a tool.
    pub fn duration_for(&self, tool: &str) -> Duration {
        let secs = self
            .per_tool_timeout
            .get(tool)
            .copied()
            .unwrap_or(self.default_timeout_secs);
        Duration::from_secs(secs)
    }
}

// --- JsonRpcRequest ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: serde_json::Value,
    pub id: serde_json::Value,
}
