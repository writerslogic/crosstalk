use crate::mcp::{McpTool, ToolResult};
use std::process::Command;
use anyhow::Result;
use std::time::Instant;

pub struct ToolDiscovery;

impl ToolDiscovery {
    pub fn scan() -> Vec<McpTool> {
        let mut tools = vec![];
        let known_binaries = ["cargo", "git", "rustfmt", "clippy", "rustc", "tree-sitter", "nix"];

        for bin in known_binaries {
            if let Ok(path) = which::which(bin) {
                let description = format!("System tool: {} at {:?}", bin, path);
                tools.push(McpTool {
                    name: bin.to_string(),
                    description,
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": { "type": "string" }
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
        let output = Command::new(bin)
            .args(args)
            .output()?;

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
}
