use crate::types::mcp::McpTool;
use anyhow::Result;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

fn is_valid_nix_dep_name(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
}

pub struct NixManager {
    pub dependencies: Vec<String>,
    pub cache_dir: PathBuf,
    cache_ttl_secs: i64,
}

impl NixManager {
    pub fn new(dependencies: Vec<String>) -> Result<Self> {
        let cache_dir = std::env::temp_dir().join("crosstalk-nix-cache");
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            dependencies,
            cache_dir,
            cache_ttl_secs: 3600,
        })
    }

    pub fn generate_flake_static(dependencies: &[String]) -> String {
        let deps = dependencies
            .iter()
            .filter(|d| is_valid_nix_dep_name(d))
            .map(|d| format!("pkgs.{d}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            r#"{{
  description = "Crosstalk Generated Environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = {{ self, nixpkgs }}: let
    system = builtins.currentSystem;
    pkgs = nixpkgs.legacyPackages.${{system}};
  in {{
    devShells.${{system}}.default = pkgs.mkShell {{
      buildInputs = [ {deps} ];
    }};
  }};
}}"#
        )
    }

    pub fn generate_flake(&self) -> Result<String> {
        Ok(Self::generate_flake_static(&self.dependencies))
    }

    pub async fn synthesize(&mut self) -> Result<HashMap<String, String>> {
        let cache_key = self.cache_key();
        let cache_path = self.cache_dir.join(format!("{cache_key}.json"));

        if let Ok(data) = std::fs::read(&cache_path)
            && let Ok(entry) = serde_json::from_slice::<serde_json::Value>(&data)
        {
            let created_at = entry["created_at"].as_i64().unwrap_or(0);
            let now = chrono::Utc::now().timestamp();
            if now - created_at < self.cache_ttl_secs
                && let Some(env) = entry["env"].as_object()
            {
                let mut result = HashMap::new();
                for (k, v) in env {
                    if let Some(s) = v.as_str() {
                        result.insert(k.clone(), s.to_string());
                    }
                }
                return Ok(result);
            }
        }

        let env = tokio::task::spawn_blocking(|| -> Result<HashMap<String, String>> {
            let output = Command::new("nix")
                .args(["develop", "--command", "env"])
                .output()
                .map_err(|e| anyhow::anyhow!("nix not available: {e}"))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow::anyhow!("nix develop failed: {stderr}"));
            }

            let mut env = HashMap::new();
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    env.insert(k.to_string(), v.to_string());
                }
            }
            Ok(env)
        })
        .await
        .map_err(|e| anyhow::anyhow!("nix synthesize task panicked: {e}"))??;

        // Write cache with created_at timestamp
        let cache_key = self.cache_key();
        let cache_path = self.cache_dir.join(format!("{cache_key}.json"));
        let cache_entry = serde_json::json!({
            "created_at": chrono::Utc::now().timestamp(),
            "env": &env,
        });
        if let Err(e) = std::fs::create_dir_all(&self.cache_dir) {
            tracing::warn!("Failed to create cache dir {:?}: {e}", self.cache_dir);
        }
        if let Err(e) = std::fs::write(
            &cache_path,
            serde_json::to_string(&cache_entry).unwrap_or_default(),
        ) {
            tracing::warn!("Failed to write cache file {:?}: {e}", cache_path);
        }

        Ok(env)
    }

    pub fn get_env(&self, key: &str) -> Option<String> {
        let cache_key = self.cache_key();
        let cache_path = self.cache_dir.join(format!("{cache_key}.json"));

        if let Ok(data) = std::fs::read(&cache_path)
            && let Ok(entry) = serde_json::from_slice::<serde_json::Value>(&data)
        {
            let created_at = entry["created_at"].as_i64().unwrap_or(0);
            let now = chrono::Utc::now().timestamp();
            if now - created_at < self.cache_ttl_secs {
                return entry["env"][key].as_str().map(|s| s.to_string());
            }
        }
        None
    }

    fn cache_key(&self) -> String {
        let mut sorted = self.dependencies.clone();
        sorted.sort();
        let input = sorted.join(",");
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn validate_environment(dependencies: &[String]) -> Result<Vec<String>> {
        let mut missing = Vec::new();
        for dep in dependencies {
            if !is_valid_nix_dep_name(dep) {
                missing.push(dep.clone());
                continue;
            }
            if which::which(dep).is_err() {
                missing.push(dep.clone());
            }
        }
        Ok(missing)
    }
}

pub struct ToolDiscovery;

impl ToolDiscovery {
    pub fn scan() -> Vec<McpTool> {
        let mut tools = Vec::new();
        let known_bins = [
            "cargo", "rustc", "git", "rustfmt", "python3", "python", "node", "go",
        ];

        for bin in known_bins {
            if which::which(bin).is_ok() {
                tools.push(McpTool {
                    name: bin.to_string(),
                    description: format!("System binary: {}", bin),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": { "type": "string" }
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
        let mut tools = Vec::new();
        let known_bins = ["cargo", "rustc", "git", "rustfmt"];

        for bin in known_bins {
            if let Ok(path) = which::which(bin) {
                let version = Command::new(&path)
                    .arg("--version")
                    .output()
                    .ok()
                    .and_then(|o| {
                        if o.status.success() {
                            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                        } else {
                            None
                        }
                    });
                tools.push(McpTool {
                    name: bin.to_string(),
                    description: format!("System binary: {}", bin),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        }
                    }),
                    version,
                });
            }
        }
        tools
    }
}
