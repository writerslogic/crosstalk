use anyhow::Result;
use crate::types::mcp::McpTool;
use std::process::Command;
use serde_json::json;

pub struct NixManager;

impl NixManager {
    pub fn generate_flake(dependencies: &[String]) -> String {
        let deps = dependencies.join(" ");
        format!(r#"{
  description = "Crosstalk Generated Environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs\/nixos-unstable";
  outputs = { self, nixpkgs }: let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = with pkgs; [ {} ];
    };
  };
}"#, deps)
    }
}

pub struct ToolDiscovery;

impl ToolDiscovery {
    pub fn scan() -> Vec<McpTool> {
        let mut tools = Vec::new();
        let known_bins = ["cargo", "rustc", "git", "rustfmt"];

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
                });
            }
        }
        tools
    }
}
