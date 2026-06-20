use anyhow::{Result, anyhow};
use serde_json::Value;

/// Resolve a W3C DID URI to its DID Document via the did:web method.
///
/// Only `did:web` is supported. Other method prefixes return an error.
/// The resolved document is returned as raw JSON for storage in `Principal::did_document`.
pub struct DidResolver;

impl DidResolver {
    /// Resolve `did` → DID Document JSON.
    pub async fn resolve(did: &str) -> Result<Value> {
        if !did.starts_with("did:web:") {
            return Err(anyhow!(
                "Unsupported DID method in '{}'; only did:web is supported",
                did
            ));
        }

        let authority = did.strip_prefix("did:web:").unwrap_or("");
        if authority.is_empty() {
            return Err(anyhow!("Invalid did:web URI: authority is empty"));
        }

        // did:web spec: colons after the authority encode path segments as slashes.
        let parts: Vec<&str> = authority.splitn(2, ':').collect();
        let host = parts[0];

        // Reject hosts that could resolve to internal/private networks (SSRF mitigation).
        let host_lower = host.to_lowercase();
        if host_lower == "localhost"
            || host_lower.starts_with("127.")
            || host_lower.starts_with("10.")
            || host_lower.starts_with("192.168.")
            || host_lower.starts_with("169.254.")
            || host_lower.starts_with("[::1]")
            || host_lower.starts_with("[fe80:")
            || host_lower.starts_with("0.")
            || (host_lower.starts_with("172.")
                && host_lower
                    .split('.')
                    .nth(1)
                    .and_then(|s| s.parse::<u8>().ok())
                    .is_some_and(|b| (16..=31).contains(&b)))
        {
            return Err(anyhow!(
                "DID resolver: host '{}' resolves to private/reserved range",
                host
            ));
        }

        let path = if parts.len() == 2 {
            let raw_path = parts[1].replace(':', "/");
            if raw_path.split('/').any(|seg| seg == ".." || seg == ".") {
                return Err(anyhow!("DID resolver: path traversal in DID authority"));
            }
            format!("{}/did.json", raw_path)
        } else {
            ".well-known/did.json".to_string()
        };

        let url = format!("https://{}/{}", host, path);
        let doc: Value = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| anyhow!("DID resolver: request to {} failed: {}", url, e))?
            .error_for_status()
            .map_err(|e| anyhow!("DID resolver: HTTP error from {}: {}", url, e))?
            .json()
            .await
            .map_err(|e| anyhow!("DID resolver: invalid JSON from {}: {}", url, e))?;

        Ok(doc)
    }
}
