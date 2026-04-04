use std::process::Command;
use anyhow::{Result, anyhow};

pub struct LinterGuard;

impl LinterGuard {
    /// Runs clippy and fmt checks on the current workspace.
    pub fn check_workspace() -> Result<()> {
        // 1. Cargo Fmt check
        let fmt_status = Command::new("cargo")
            .args(["fmt", "--check"])
            .status()?;
        
        if !fmt_status.success() {
            return Err(anyhow!("Formatting check failed. Please run 'cargo fmt'."));
        }

        // 2. Cargo Clippy check
        let clippy_status = Command::new("cargo")
            .args(["clippy", "--all-targets", "--", "-D", "warnings"])
            .status()?;

        if !clippy_status.success() {
            return Err(anyhow!("Clippy check failed with warnings or errors."));
        }

        Ok(())
    }
}
