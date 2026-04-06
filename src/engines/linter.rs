use crate::engines::sandbox::SandboxResult;
use anyhow::{anyhow, Result};
use std::process::Command;

pub struct LinterGuard;

impl LinterGuard {
    pub fn check(sandbox_result: &SandboxResult) -> Result<()> {
        if sandbox_result.exit_code != 0 {
            return Err(anyhow!(
                "Sandbox execution failed with exit code: {}. stderr: {}",
                sandbox_result.exit_code,
                sandbox_result.stderr
            ));
        }

        if !sandbox_result.stdout.is_empty() {
            Self::run_clippy()?;
            Self::run_fmt_check()?;
        }

        Ok(())
    }

    fn run_clippy() -> Result<()> {
        let output = Command::new("cargo")
            .args(["clippy", "--all-targets", "--", "-D", "warnings"])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Clippy linting failed:\n{}", stderr));
        }

        Ok(())
    }

    fn run_fmt_check() -> Result<()> {
        let output = Command::new("cargo")
            .args(["fmt", "--", "--check"])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Code formatting check failed:\n{}", stderr));
        }

        Ok(())
    }
}
