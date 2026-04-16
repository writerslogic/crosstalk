use anyhow::Result;
use crate::types::mcp::ToolResult;
use std::process::Command;
use std::time::Instant;

pub struct CliBridge;

impl CliBridge {
    pub async fn call(binary: &str, args: Vec<String>, current_dir: &str) -> Result<ToolResult> {
        let start = Instant::now();
        
        let output = Command::new(binary)
            .args(&args)
            .current_dir(current_dir)
            .output()?;

        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(ToolResult {
            tool_name: binary.to_string(),
            success,
            output: stdout,
            error: if stderr.is_empty() { None } else { Some(stderr) },
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
}
