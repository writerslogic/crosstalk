use crate::mcp::gateway::{McpTool, ToolResult};
use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;
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
                // Extract version info
                let version = Command::new(bin)
                    .arg("--version")
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|_| "Unknown version".to_string());

                // Extract help summary
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
                });
            }
        }
        tools
    }
}

pub struct CliBridge;

impl CliBridge {
    pub fn invoke(bin: &str, args: Vec<String>) -> Result<ToolResult> {
        let start = Instant::now();

        let final_bin = bin.to_string();
        let final_args = args;

        // Check for Nix environment if nix is available and a flake exists
        if Path::new("flake.nix").exists() && which::which("nix").is_ok() {
            let mut nix_args = vec!["develop", "--command", bin];
            nix_args.extend(final_args.iter().map(|s| s.as_str()));

            let output = Command::new("nix").args(&nix_args).output()?;

            return Ok(ToolResult {
                tool_name: bin.to_string(),
                success: output.status.success(),
                output: String::from_utf8_lossy(&output.stdout).to_string(),
                error: if output.status.success() {
                    None
                } else {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                },
                elapsed_ms: start.elapsed().as_millis() as u64,
            });
        }

        let output = Command::new(final_bin).args(final_args).output()?;

        Ok(ToolResult {
            tool_name: bin.to_string(),
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string(),
            error: if output.status.success() {
                None
            } else {
                Some(String::from_utf8_lossy(&output.stderr).to_string())
            },
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
}

pub struct NixSynthesizer;

impl NixSynthesizer {
    pub fn generate_flake(dependencies: &[String]) -> String {
        let deps_str = dependencies.join(" ");
        format!(
            r#"{{
  description = "Crosstalk dynamic environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = {{ self, nixpkgs }}: let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in {{
    devShell.x86_64-linux = pkgs.mkShell {{
      buildInputs = [ {} ];
    }};
  }};
}}"#,
            deps_str
        )
    }

    pub fn write_flake(content: &str) -> Result<()> {
        fs::write("flake.nix", content).map_err(|e| anyhow!("Failed to write flake.nix: {}", e))
    }
}
