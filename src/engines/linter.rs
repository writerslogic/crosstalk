// linter.rs (Upgraded)
use crate::engines::sandbox::SandboxResult;
use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

pub struct LinterGuard;

impl LinterGuard {
    pub async fn check(sandbox_result: &SandboxResult, workspace_dir: &str) -> Result<()> {
        if sandbox_result.exit_code != 0 {
            return Err(anyhow!("Sandbox failed: {}", sandbox_result.stderr));
        }

        // Run linting asynchronously with a strict timeout to prevent infinite macro loops
        Self::run_secure_clippy(workspace_dir).await?;
        Ok(())
    }

    async fn run_secure_clippy(workspace_dir: &str) -> Result<()> {
        // Skip clippy check if workspace_dir doesn't have Cargo.toml
        let cargo_path = Path::new(workspace_dir).join("Cargo.toml");
        if !cargo_path.exists() {
            return Ok(());
        }

        // We use tokio::process for non-blocking execution and apply a strict 30s timeout.
        // In production, run this INSIDE a Docker container/gVisor, NOT on bare metal.
        let clippy_task = Command::new("cargo")
            .current_dir(workspace_dir)
            .args([
                "clippy",
                "--all-targets",
                "--offline", // Prevent malicious crates from being downloaded
                "--",
                "-D", "warnings",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let output = timeout(Duration::from_secs(30), clippy_task.wait_with_output())
            .await
            .map_err(|_| anyhow!("Clippy execution timed out (Possible malicious build.rs loop)"))??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Clippy rejected the code:\n{}", stderr));
        }

        Ok(())
    }
}