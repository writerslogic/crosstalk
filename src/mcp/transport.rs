use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

const MAX_CONSECUTIVE_PARSE_ERRORS: u32 = 10;

pub struct McpTransport {
    pub socket_path: String,
}

impl McpTransport {
    pub fn new(path: &str) -> Self {
        Self {
            socket_path: path.to_string(),
        }
    }

    pub async fn listen(&self) -> Result<UnixListener> {
        let path = Path::new(&self.socket_path);
        if match tokio::fs::try_exists(path).await {
            Ok(exists) => exists,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(anyhow::anyhow!("failed to check socket path: {e}")),
        } {
            crate::log_warn!(
                tokio::fs::remove_file(path).await,
                "failed to remove socket file"
            );
        }
        UnixListener::bind(path).map_err(|e| anyhow::anyhow!(e))
    }

    pub async fn handle_connection(stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();
        let mut consecutive_errors = 0u32;

        while let Some(line) = lines.next_line().await? {
            let request: Value = match serde_json::from_str(&line) {
                Ok(v) => {
                    consecutive_errors = 0;
                    v
                }
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
            let method = request.get("method").and_then(|v| v.as_str()).unwrap_or("");

            let response = match method {
                "initialize" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": { "name": "crosstalk", "version": "0.1.0" }
                    }
                }),
                "ping" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "pong": true }
                }),
                "notifications/initialized" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "ok": true }
                }),
                _ => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": "Method not found" }
                }),
            };

            let mut resp_str = serde_json::to_string(&response)?;
            resp_str.push('\n');
            writer.write_all(resp_str.as_bytes()).await?;
        }
        Ok(())
    }
}
