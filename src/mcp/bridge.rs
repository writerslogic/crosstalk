use crate::mcp::gateway::{McpTool, ToolResult};
use anyhow::{Result, anyhow};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub struct ToolDiscovery;

impl ToolDiscovery {
    pub fn scan() -> Vec<McpTool> {
        let mut tools = vec![];
        let known_binaries = [
            "cargo",
            "git",
            "rustfmt",
            "clippy",
            "rustc",
            "tree-sitter",
            "nix",
        ];

        for bin in known_binaries {
            if let Ok(path) = which::which(bin) {
                let version = Command::new(bin)
                    .arg("--version")
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|_| "Unknown version".to_string());

                let help = Command::new(bin)
                    .arg("--help")
                    .output()
                    .map(|o| {
                        let full_help = String::from_utf8_lossy(&o.stdout);
                        full_help.lines().take(5).collect::<Vec<_>>().join("\n")
                    })
                    .unwrap_or_else(|_| "No help available".to_string());

                let description = format!(
                    "System tool: {} ({})\nLocation: {:?}\nSummary:\n{}",
                    bin, version, path, help
                );

                tools.push(McpTool {
                    name: bin.to_string(),
                    description,
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Command line arguments to pass to the tool"
                            }
                        }
                    }),
                    version: None,
                });
            }
        }
        tools
    }

    pub fn scan_with_versions() -> Vec<McpTool> {
        let mut tools = vec![];
        let known_binaries = [
            "cargo",
            "git",
            "rustfmt",
            "clippy",
            "rustc",
            "tree-sitter",
            "nix",
        ];

        for bin in known_binaries {
            if let Ok(path) = which::which(bin) {
                let version = Command::new(bin)
                    .arg("--version")
                    .output()
                    .map(|o| {
                        let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                        if s.is_empty() { None } else { Some(s) }
                    })
                    .unwrap_or(None);

                let schema = CliBridge::validate_schema(bin)
                    .map(|s| s.to_json_schema())
                    .unwrap_or_else(|_| serde_json::json!({
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Command line arguments to pass to the tool"
                            }
                        }
                    }));

                let version_str = version.as_deref().unwrap_or("Unknown version");
                let description = format!(
                    "System tool: {} ({})\nLocation: {:?}",
                    bin, version_str, path
                );

                tools.push(McpTool {
                    name: bin.to_string(),
                    description,
                    input_schema: schema,
                    version,
                });
            }
        }
        tools
    }
}

/// A parsed flag definition extracted from --help output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FlagDef {
    pub long: String,
    pub short: Option<String>,
    pub description: String,
    pub takes_value: bool,
}

/// Schema describing a CLI tool's accepted inputs, used for MCP tool registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    pub flags: Vec<FlagDef>,
    pub raw_usage: String,
}

impl InputSchema {
    /// Convert to a JSON schema value suitable for MCP tool registration.
    pub fn to_json_schema(&self) -> serde_json::Value {
        let flag_descriptions: Vec<String> = self
            .flags
            .iter()
            .map(|f| {
                let short = f
                    .short
                    .as_deref()
                    .map(|s| format!("-{}, ", s))
                    .unwrap_or_default();
                let value = if f.takes_value { " <VALUE>" } else { "" };
                format!("{}--{}{}: {}", short, f.long, value, f.description)
            })
            .collect();

        serde_json::json!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": format!(
                        "Arguments for this tool.\nKnown flags:\n{}",
                        flag_descriptions.join("\n")
                    )
                }
            }
        })
    }
}

pub struct CargoBridge;

impl CargoBridge {
    pub fn build(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["build".to_string()];
        if args.get("release").map(|v| v == "true").unwrap_or(false) {
            cli.push("--release".to_string());
        }
        if let Some(pkg) = args.get("package") {
            cli.extend(["--package".to_string(), pkg.clone()]);
        }
        cli
    }

    pub fn test(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["test".to_string()];
        if let Some(name) = args.get("name") {
            cli.push(name.clone());
        }
        if args.get("no_run").map(|v| v == "true").unwrap_or(false) {
            cli.push("--no-run".to_string());
        }
        cli
    }

    pub fn check(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["check".to_string()];
        if let Some(pkg) = args.get("package") {
            cli.extend(["--package".to_string(), pkg.clone()]);
        }
        cli
    }

    pub fn clippy(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["clippy".to_string()];
        if args.get("deny_warnings").map(|v| v == "true").unwrap_or(false) {
            cli.extend(["--".to_string(), "-D".to_string(), "warnings".to_string()]);
        }
        cli
    }

    pub fn fmt(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["fmt".to_string()];
        if args.get("check").map(|v| v == "true").unwrap_or(false) {
            cli.push("--check".to_string());
        }
        cli
    }
}

pub struct GitBridge;

impl GitBridge {
    pub fn status(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["status".to_string()];
        if args.get("short").map(|v| v == "true").unwrap_or(false) {
            cli.push("--short".to_string());
        }
        cli
    }

    pub fn diff(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["diff".to_string()];
        if args.get("staged").map(|v| v == "true").unwrap_or(false) {
            cli.push("--staged".to_string());
        }
        if let Some(path) = args.get("path") {
            cli.push("--".to_string());
            cli.push(path.clone());
        }
        cli
    }

    pub fn log(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["log".to_string()];
        if let Some(n) = args.get("n") {
            cli.extend([format!("-{n}")]);
        }
        if args.get("oneline").map(|v| v == "true").unwrap_or(false) {
            cli.push("--oneline".to_string());
        }
        cli
    }

    pub fn add(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["add".to_string()];
        if args.get("all").map(|v| v == "true").unwrap_or(false) {
            cli.push("--all".to_string());
        } else if let Some(path) = args.get("path") {
            cli.push(path.clone());
        }
        cli
    }

    pub fn commit(args: &HashMap<String, String>) -> Vec<String> {
        let mut cli = vec!["commit".to_string()];
        if let Some(msg) = args.get("message") {
            cli.extend(["-m".to_string(), msg.clone()]);
        }
        cli
    }
}

pub struct CliBridge;

impl CliBridge {
    /// Resolve a binary name or path to an absolute PathBuf.
    ///
    /// - Absolute path or path with `/`: verified to exist on disk.
    /// - Plain name + env_override with PATH: searched within isolated PATH.
    /// - Plain name without override: falls back to system `which`.
    fn resolve_binary(
        binary_path: &str,
        env_override: Option<&HashMap<String, String>>,
    ) -> Result<PathBuf> {
        let p = Path::new(binary_path);

        if p.is_absolute() || binary_path.contains('/') {
            if p.is_file() {
                return Ok(p.to_path_buf());
            }
            return Err(anyhow!("Binary not found at path: {}", binary_path));
        }

        if let Some(env) = env_override
            && let Some(path_val) = env.get("PATH")
        {
            for dir in path_val.split(':') {
                let candidate = Path::new(dir).join(binary_path);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
            return Err(anyhow!(
                "Binary '{}' not found in isolated PATH: {}",
                binary_path,
                path_val
            ));
        }

        which::which(binary_path)
            .map_err(|_| anyhow!("Binary '{}' not found in system PATH", binary_path))
    }

    /// Execute a CLI binary, optionally within a Nix-synthesized isolated environment.
    ///
    /// When `env_override` is `Some`, the child process runs with a cleared environment
    /// containing only the supplied variables (PATH, LD_LIBRARY_PATH, etc. from NixManager).
    pub fn invoke(
        binary_path: &str,
        args: Vec<String>,
        env_override: Option<&HashMap<String, String>>,
    ) -> Result<ToolResult> {
        let start = Instant::now();

        let resolved = Self::resolve_binary(binary_path, env_override)?;

        let mut cmd = Command::new(&resolved);
        cmd.args(&args);

        if let Some(env) = env_override {
            // Clear parent environment for isolation; inject only Nix-provided vars.
            cmd.env_clear().envs(env);
        }

        let output = cmd.output()?;

        Ok(ToolResult {
            tool_name: binary_path.to_string(),
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string(),
            error: {
                let s = String::from_utf8_lossy(&output.stderr).to_string();
                if s.is_empty() { None } else { Some(s) }
            },
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Execute a CLI binary with a wall-clock timeout.
    ///
    /// Uses `tokio::process::Command` so cancellation actually terminates the child.
    /// Returns `Err` containing "TimeoutError" if the process does not finish within
    /// `timeout_secs`.
    pub async fn invoke_with_timeout(
        binary_path: &str,
        args: Vec<String>,
        env_override: Option<&HashMap<String, String>>,
        timeout_secs: u64,
    ) -> Result<ToolResult> {
        use tokio::process::Command as TokioCommand;
        use tokio::time::{Duration, timeout};

        let start = Instant::now();
        let resolved = Self::resolve_binary(binary_path, env_override)?;

        let mut cmd = TokioCommand::new(&resolved);
        cmd.args(&args);
        if let Some(env) = env_override {
            cmd.env_clear().envs(env);
        }

        let tool_name = binary_path.to_string();

        match timeout(Duration::from_secs(timeout_secs), cmd.output()).await {
            Ok(Ok(output)) => Ok(ToolResult {
                tool_name,
                success: output.status.success(),
                output: String::from_utf8_lossy(&output.stdout).to_string(),
                error: if output.status.success() {
                    None
                } else {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                },
                elapsed_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(Err(e)) => Err(anyhow!("Process spawn error: {}", e)),
            Err(_) => Err(anyhow!(
                "TimeoutError: '{}' did not complete within {}s",
                binary_path,
                timeout_secs
            )),
        }
    }

    /// Parse the `--help` output of a binary to produce an `InputSchema`.
    ///
    /// Recognises the flag patterns emitted by clap, docopt, and most Unix tools:
    /// ```text
    ///     -v, --verbose          Be verbose
    ///     -o, --output <FILE>    Output path
    ///         --no-cache         Skip cache
    /// ```
    pub fn validate_schema(binary_path: &str) -> Result<InputSchema> {
        let resolved = Self::resolve_binary(binary_path, None)?;

        // Some tools (e.g. git) write help to stderr.
        let output = Command::new(&resolved).arg("--help").output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let help_text = if stdout.trim().is_empty() {
            stderr.to_string()
        } else {
            stdout.to_string()
        };

        let flags = Self::parse_flags_from_help(&help_text);

        let raw_usage = help_text
            .lines()
            .find(|l| {
                let lower = l.to_lowercase();
                lower.contains("usage")
            })
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(InputSchema { flags, raw_usage })
    }

    fn parse_flags_from_help(help: &str) -> Vec<FlagDef> {
        // Matches lines like:
        //   -v, --verbose          Be verbose
        //   -o, --output <FILE>    Output path
        //       --no-cache         Skip cache
        let re = Regex::new(
            r"(?x)
            ^\s*
            (?:-(?P<short>[A-Za-z0-9]),\s*)?            # optional: -X,
            (?:--(?P<long>[a-zA-Z][a-zA-Z0-9_-]*))?    # optional: --long-flag
            (?:\s+(?P<val>[<\[]\S+[>\]]))?              # optional: <VALUE> or [VALUE]
            \s{2,}                                      # separator (>=2 spaces)
            (?P<desc>.+)                                # description
        ",
        )
        .expect("static regex is valid");

        let mut flags = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for line in help.lines() {
            let Some(caps) = re.captures(line) else {
                continue;
            };

            let long = caps.name("long").map(|m| m.as_str().to_string());
            let short = caps.name("short").map(|m| m.as_str().to_string());
            let takes_value = caps.name("val").is_some();
            let desc = caps
                .name("desc")
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();

            if long.is_none() && short.is_none() {
                continue;
            }

            let key = long.clone().unwrap_or_else(|| {
                format!("short:{}", short.as_deref().unwrap_or(""))
            });

            if !seen.insert(key.clone()) {
                continue;
            }

            flags.push(FlagDef {
                long: long.unwrap_or(key),
                short,
                description: desc,
                takes_value,
            });
        }

        // Fallback: for tools like `git` that embed flags inline on usage lines
        // (e.g. `git [-v | --version] [-h | --help] ...`), extract any
        // `--flagname` tokens we haven't already captured.
        if flags.is_empty() {
            let inline_re =
                Regex::new(r"--(?P<long>[a-zA-Z][a-zA-Z0-9_-]*)").expect("static regex is valid");
            for caps in inline_re.captures_iter(help) {
                let long = caps["long"].to_string();
                if seen.insert(long.clone()) {
                    flags.push(FlagDef {
                        long,
                        short: None,
                        description: String::new(),
                        takes_value: false,
                    });
                }
            }
        }

        flags
    }
}

