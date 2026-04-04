use anyhow::{Result, anyhow};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::net::UnixStream;
use tokio_util::codec::{Framed, LinesCodec};

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: serde_json::Value,
    pub id: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<serde_json::Value>,
    pub id: serde_json::Value,
}

pub struct JsonRpcTransport {
    framed: Framed<UnixStream, LinesCodec>,
}

impl JsonRpcTransport {
    pub async fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let stream = UnixStream::connect(path).await?;
        let framed = Framed::new(stream, LinesCodec::new());
        Ok(Self { framed })
    }

    pub fn new(stream: UnixStream) -> Self {
        let framed = Framed::new(stream, LinesCodec::new());
        Self { framed }
    }

    pub async fn send_request(&mut self, method: &str, params: serde_json::Value, id: i64) -> Result<()> {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: serde_json::json!(id),
        };
        let msg = serde_json::to_string(&req)?;
        self.framed.send(msg).await?;
        Ok(())
    }

    pub async fn send_response(&mut self, result: Option<serde_json::Value>, error: Option<serde_json::Value>, id: serde_json::Value) -> Result<()> {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result,
            error,
            id,
        };
        let msg = serde_json::to_string(&resp)?;
        self.framed.send(msg).await?;
        Ok(())
    }

    pub async fn next_message(&mut self) -> Result<Option<String>> {
        match self.framed.next().await {
            Some(Ok(line)) => Ok(Some(line)),
            Some(Err(e)) => Err(anyhow!("Transport error: {:?}", e)),
            None => Ok(None),
        }
    }
}
