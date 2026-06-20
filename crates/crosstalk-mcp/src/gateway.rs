//! MCP gateway configuration.
//!
//! Magic literals for the gateway have been swept into
//! `crosstalk_core::consts`.

use std::time::Duration;

use crosstalk_core::consts::{
    DEFAULT_GATEWAY_PORT, DEFAULT_GATEWAY_TIMEOUT, MAX_PAYLOAD_BYTES,
};

/// Configuration for the MCP gateway server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayConfig {
    /// Listen port.
    pub port: u16,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum inbound payload size in bytes.
    pub max_payload_bytes: usize,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            port: DEFAULT_GATEWAY_PORT,
            timeout: DEFAULT_GATEWAY_TIMEOUT,
            max_payload_bytes: MAX_PAYLOAD_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_named_consts() {
        let cfg = GatewayConfig::default();
        assert_eq!(cfg.port, DEFAULT_GATEWAY_PORT);
        assert_eq!(cfg.timeout, DEFAULT_GATEWAY_TIMEOUT);
        assert_eq!(cfg.max_payload_bytes, MAX_PAYLOAD_BYTES);
    }
}