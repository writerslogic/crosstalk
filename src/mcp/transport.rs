use anyhow::Result;
use tokio::net::{UnixListener, UnixStream};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

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
            let _ = fs::remove_file(path);
        }
        UnixListener::bind(path).map_err(|e| anyhow::anyhow!(e))
    }

    pub async fn handle_connection(stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();

        while let Some(line) = lines.next_line().await? {
            let request: Value = serde_json::from_str(&line)?;
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
