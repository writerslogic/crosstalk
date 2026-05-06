use anyhow::{Context, Result};
use crate::types::mcp::ToolResult;
use serde_json::json;
use std::collections::HashMap;
use std::process::Command;
use std::process::Stdio;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct FlagDef {
    pub long: String,
    pub short: Option<String>,
    pub description: String,
    pub takes_value: bool,
}

#[derive(Debug, Clone)]
pub struct InputSchema {
    pub flags: Vec<FlagDef>,
    pub raw_usage: String,
}

impl InputSchema {
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut desc_parts = Vec::new();
        for flag in &self.flags {
            let flag_str = if flag.takes_value {
                format!("--{} <VALUE>", flag.long)
            } else {
                format!("--{}", flag.long)
            };
            desc_parts.push(format!("{}: {}", flag_str, flag.description));
        }
        let desc = desc_parts.join("\n");
        json!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": desc
                }
            }
        })
    }
}

pub struct CliBridge;

impl CliBridge {
    pub async fn call(binary: &str, args: Vec<String>, current_dir: &str) -> Result<ToolResult> {
        if binary.is_empty()
            || !binary.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(anyhow::anyhow!("invalid binary name (must be [a-zA-Z0-9_-]+): {binary}"));
        }

        // Reject traversal components then canonicalize to prevent path traversal.
        let dir_path = std::path::Path::new(current_dir);
        if dir_path
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(anyhow::anyhow!(
                "invalid current_dir (contains '..'): {current_dir}"
            ));
        }
        let canonical_dir = std::fs::canonicalize(dir_path)
            .with_context(|| format!("failed to canonicalize current_dir: {current_dir}"))?;

        const OUTPUT_LIMIT: usize = 10 * 1024 * 1024; // 10 MiB per stream
        const TIMEOUT_SECS: u64 = 60;
        let start = Instant::now();

        let child = tokio::process::Command::new(binary)
            .args(&args)
            .current_dir(&canonical_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn {}: {}", binary, e))?;

        let output = match tokio::time::timeout(
            tokio::time::Duration::from_secs(TIMEOUT_SECS),
            child.wait_with_output(),
        )
        .await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(anyhow::anyhow!("failed to wait on {}: {}", binary, e)),
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "TimeoutError: command exceeded {}s",
                    TIMEOUT_SECS
                ))
            }
        };

        let stdout = String::from_utf8_lossy(
            &output.stdout[..output.stdout.len().min(OUTPUT_LIMIT)],
        )
        .to_string();
        let stderr = String::from_utf8_lossy(
            &output.stderr[..output.stderr.len().min(OUTPUT_LIMIT)],
        )
        .to_string();

        Ok(ToolResult {
            tool_name: binary.to_string(),
            success: output.status.success(),
            output: stdout,
            error: if stderr.is_empty() { None } else { Some(stderr) },
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Synchronous invocation of a binary with optional environment override.
    pub fn invoke(
        binary: &str,
        args: Vec<String>,
        env_override: Option<&HashMap<String, String>>,
    ) -> Result<ToolResult> {
        let resolved = Self::resolve_binary(binary, env_override)?;
        let start = Instant::now();

        let mut cmd = Command::new(&resolved);
        cmd.args(&args);
        if let Some(env) = env_override {
            for (k, v) in env {
                cmd.env(k, v);
            }
        }

        let output = cmd.output().map_err(|e| {
            anyhow::anyhow!("binary not found: {} ({})", binary, e)
        })?;

        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(ToolResult {
            tool_name: binary.to_string(),
            success,
            output: stdout,
            error: if stderr.is_empty() { None } else { Some(stderr) },
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Async invocation with a timeout (in seconds).
    pub async fn invoke_with_timeout(
        binary: &str,
        args: Vec<String>,
        env_override: Option<&HashMap<String, String>>,
        timeout_secs: u64,
    ) -> Result<ToolResult> {
        let resolved = Self::resolve_binary(binary, env_override)?;
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new(&resolved);
        cmd.args(&args);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        if let Some(env) = env_override {
            for (k, v) in env {
                cmd.env(k, v);
            }
        }

        let child = cmd.spawn().map_err(|e| {
            anyhow::anyhow!("binary not found: {} ({})", binary, e)
        })?;

        let timeout = tokio::time::Duration::from_secs(timeout_secs);
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let success = output.status.success();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Ok(ToolResult {
                    tool_name: binary.to_string(),
                    success,
                    output: stdout,
                    error: if stderr.is_empty() { None } else { Some(stderr) },
                    elapsed_ms: start.elapsed().as_millis() as u64,
                })
            }
            Ok(Err(e)) => Err(anyhow::anyhow!("process error: {}", e)),
            Err(_) => {
                Err(anyhow::anyhow!("TimeoutError: command exceeded {}s", timeout_secs))
            }
        }
    }

    /// Parse --help output into an InputSchema.
    pub fn validate_schema(binary: &str) -> Result<InputSchema> {
        let resolved = Self::resolve_binary(binary, None)?;
        let output = Command::new(&resolved)
            .arg("--help")
            .output()
            .map_err(|e| anyhow::anyhow!("binary not found: {} ({})", binary, e))?;

        let text = String::from_utf8_lossy(&output.stdout).to_string()
            + &String::from_utf8_lossy(&output.stderr);

        let mut flags = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for line in text.lines() {
            // Extract all --flag occurrences from any line (usage lines, option descriptions, etc.)
            let mut rest = line;
            while let Some(idx) = rest.find("--") {
                rest = &rest[idx + 2..];
                let flag_end = rest.find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_').unwrap_or(rest.len());
                let flag = &rest[..flag_end];
                if !flag.is_empty() && seen.insert(flag.to_string()) {
                    let takes_value = rest[flag_end..].starts_with('=') || rest[flag_end..].starts_with("[=");
                    flags.push(FlagDef {
                        long: flag.to_string(),
                        short: None,
                        description: String::new(),
                        takes_value,
                    });
                }
                rest = &rest[flag_end..];
            }
        }

        Ok(InputSchema {
            flags,
            raw_usage: text,
        })
    }

    fn resolve_binary(binary: &str, env_override: Option<&HashMap<String, String>>) -> Result<String> {
        // If absolute path, check existence directly
        if binary.starts_with('/') {
            if std::path::Path::new(binary).exists() {
                return Ok(binary.to_string());
            }
            return Err(anyhow::anyhow!("binary not found: {}", binary));
        }
        // If env_override has PATH, search there
        if let Some(env) = env_override
            && let Some(path_val) = env.get("PATH")
        {
            for dir in path_val.split(':') {
                let candidate = std::path::Path::new(dir).join(binary);
                if candidate.exists() {
                    return Ok(candidate.to_str().unwrap_or(binary).to_string());
                }
            }
        }
        // Fallback to which
        match which::which(binary) {
            Ok(p) => Ok(p.to_str().unwrap_or(binary).to_string()),
            Err(_) => Err(anyhow::anyhow!("binary not found: {}", binary)),
        }
    }
}

pub struct CargoBridge;

impl CargoBridge {
    pub fn build(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["build".to_string()];
        if opts.get("release").map(|v| v == "true").unwrap_or(false) {
            args.push("--release".to_string());
        }
        args
    }

    pub fn test(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["test".to_string()];
        if let Some(name) = opts.get("name") {
            args.push(name.clone());
        }
        args
    }

    pub fn clippy(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["clippy".to_string()];
        if opts.get("deny_warnings").map(|v| v == "true").unwrap_or(false) {
            args.extend(["--".to_string(), "-D".to_string(), "warnings".to_string()]);
        }
        args
    }

    pub fn fmt(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["fmt".to_string()];
        if opts.get("check").map(|v| v == "true").unwrap_or(false) {
            args.push("--check".to_string());
        }
        args
    }
}

pub struct GitBridge;

impl GitBridge {
    pub fn status(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["status".to_string()];
        if opts.get("short").map(|v| v == "true").unwrap_or(false) {
            args.push("--short".to_string());
        }
        args
    }

    pub fn diff(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["diff".to_string()];
        if opts.get("staged").map(|v| v == "true").unwrap_or(false) {
            args.push("--staged".to_string());
        }
        args
    }

    pub fn log(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["log".to_string()];
        if opts.get("oneline").map(|v| v == "true").unwrap_or(false) {
            args.push("--oneline".to_string());
        }
        if let Some(n) = opts.get("n") {
            args.push(format!("-{}", n));
        }
        args
    }

    pub fn commit(opts: &HashMap<String, String>) -> Vec<String> {
        let mut args = vec!["commit".to_string()];
        if let Some(msg) = opts.get("message") {
            args.push("-m".to_string());
            args.push(msg.clone());
        }
        args
    }
}
