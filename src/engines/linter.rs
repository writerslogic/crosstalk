use anyhow::Result;
use crate::engines::sandbox::SandboxResult;
use std::process::Command;
use std::path::Path;

pub struct LintReport {
    pub passed: bool,
    pub errors: Vec<String>,
}

pub struct LinterGuard;

impl LinterGuard {
    pub async fn check(result: &SandboxResult, workspace_root: &str, nix_env: Option<&String>) -> Result<LintReport> {
        // If we have a nix_env, we should run within that environment shell.
        // For the prototype, we assume the environment already has cargo/clippy.
        
        let mut errors = Vec::new();
        
        // 1. Clippy
        let clippy = Command::new("cargo")
            .args(["clippy", "--all-targets", "--", "-D", "warnings"])
            .current_dir(workspace_root)
            .output()?;
            
        if !clippy.status.success() {
            errors.push(String::from_utf8_lossy(&clippy.stderr).to_string());
        }
        
        // 2. Format check
        let fmt = Command::new("cargo")
            .args(["fmt", "--", "--check"])
            .current_dir(workspace_root)
            .output()?;
            
        if !fmt.status.success() {
            errors.push(String::from_utf8_lossy(&fmt.stderr).to_string());
        }
        
        Ok(LintReport {
            passed: errors.is_empty(),
            errors,
        })
    }
}
