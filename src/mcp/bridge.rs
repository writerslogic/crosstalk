use crate::types::mcp::ToolResult;
use anyhow::{Context, Result};
use std::process::Stdio;
use std::time::Instant;

pub struct CliBridge;

impl CliBridge {
    pub async fn call(binary: &str, args: Vec<String>, current_dir: &str) -> Result<ToolResult> {
        if binary.is_empty()
            || !binary
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(anyhow::anyhow!(
                "invalid binary name (must be [a-zA-Z0-9_-]+): {binary}"
            ));
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
        let canonical_dir = tokio::fs::canonicalize(dir_path)
            .await
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
                ));
            }
        };

        let stdout =
            String::from_utf8_lossy(&output.stdout[..output.stdout.len().min(OUTPUT_LIMIT)])
                .to_string();
        let stderr =
            String::from_utf8_lossy(&output.stderr[..output.stderr.len().min(OUTPUT_LIMIT)])
                .to_string();

        Ok(ToolResult {
            tool_name: binary.to_string(),
            success: output.status.success(),
            output: stdout,
            error: if stderr.is_empty() {
                None
            } else {
                Some(stderr)
            },
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
}
