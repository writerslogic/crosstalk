use anyhow::Result;
use tokio::net::{UnixListener, UnixStream};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

const MAX_CONSECUTIVE_PARSE_ERRORS: u32 = 10;

pub struct McpTransport {
    pub socket_path: String,
}

impl McpTransport {
    pub fn new(path: &str) -> Self {
        Self { socket_path: path.to_string() }
    }

    pub async fn listen(&self) -> Result<UnixListener> {
        let path = Path::new(&self.socket_path);
        if path.exists() {
            crate::log_warn!(fs::remove_file(path), "failed to remove socket file");
        }
        UnixListener::bind(path).map_err(|e| anyhow::anyhow!(e))
    }

    pub async fn handle_connection(stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();
        let mut consecutive_errors = 0u32;

        while let Some(line) = lines.next_line().await? {
            let request: Value = match serde_json::from_str(&line) {
                Ok(v) => { consecutive_errors = 0; v }
                Err(e) => {
                    consecutive_errors += 1;
                    let err_resp = json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": { "code": -32700, "message": format!("Parse error: {e}") }
                    });
                    let mut s = serde_json::to_string(&err_resp)?;
                    s.push('\n');
                    writer.write_all(s.as_bytes()).await?;
                    if consecutive_errors >= MAX_CONSECUTIVE_PARSE_ERRORS {
                        break;
                    }
                    continue;
                }
            };
            let id = request.get("id").cloned();

            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": "Connected to Crosstalk MCP Gateway"
            });

            let mut resp_str = serde_json::to_string(&response)?;
            resp_str.push('\n');
            writer.write_all(resp_str.as_bytes()).await?;
        }
        Ok(())
    }
}
