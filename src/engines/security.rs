use crate::types::conversation::Turn;
use crate::types::fiduciary::PersonaDisclosure;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use zeroize::Zeroizing;

static AWS_REGEX: OnceLock<Regex> = OnceLock::new();
static GH_REGEX: OnceLock<Regex> = OnceLock::new();
static INJECTION_PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();

fn aws_regex() -> &'static Regex {
    AWS_REGEX.get_or_init(|| Regex::new(r"AKIA[0-9A-Z]{16}").expect("valid regex"))
}
fn gh_regex() -> &'static Regex {
    GH_REGEX.get_or_init(|| Regex::new(r"ghp_[a-zA-Z0-9]{36}").expect("valid regex"))
}
fn injection_patterns() -> &'static Vec<Regex> {
    INJECTION_PATTERNS.get_or_init(|| {
        [
            r"(?i)ignore all prior instructions",
            r"(?i)you are now an? (.*) agent",
            r"(?i)system prompt override",
        ]
        .iter()
        .map(|p| Regex::new(p).expect("valid regex"))
        .collect()
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretFinding {
    pub line: usize,
    pub pattern_name: String,
    pub redacted_match: String,
}

pub struct SecretScanner;

impl SecretScanner {
    /// Comprehensive artifact scan for hardcoded secrets and high-entropy tokens.
    pub fn scan(content: &str) -> Vec<SecretFinding> {
        let mut findings = Vec::new();
        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;

            if let Some(m) = aws_regex().find(line) {
                findings.push(SecretFinding {
                    line: line_num,
                    pattern_name: "AWS_ACCESS_KEY".to_string(),
                    redacted_match: format!("{}***", &m.as_str()[..4.min(m.len())]),
                });
            }
            if let Some(m) = gh_regex().find(line) {
                findings.push(SecretFinding {
                    line: line_num,
                    pattern_name: "GITHUB_TOKEN".to_string(),
                    redacted_match: format!("{}***", &m.as_str()[..4.min(m.len())]),
                });
            }

            for token in line.split(|c: char| !c.is_alphanumeric()) {
                if token.len() >= 20 && Self::calculate_entropy(token) > 4.5 {
                    findings.push(SecretFinding {
                        line: line_num,
                        pattern_name: "HighEntropyToken".to_string(),
                        redacted_match: format!("{}***", &token[..4.min(token.len())]),
                    });
                }
            }
        }
        findings
    }

    fn calculate_entropy(s: &str) -> f64 {
        if s.is_empty() {
            return 0.0;
        }
        let mut freq = HashMap::new();
        for c in s.chars() {
            *freq.entry(c).or_insert(0) += 1;
        }
        let len = s.len() as f64;
        freq.values()
            .map(|&count| {
                let p = count as f64 / len;
                -p * p.log2()
            })
            .sum()
    }
}

pub struct InjectionShield;

impl InjectionShield {
    #[must_use]
    pub fn sanitize(input: &str) -> String {
        let mut sanitized = input.to_string();
        for re in injection_patterns() {
            sanitized = re.replace_all(&sanitized, "[REDACTED]").to_string();
        }
        sanitized
    }
}

pub struct TurnSigner {
    signing_key: SigningKey,
}

impl TurnSigner {
    #[must_use]
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let bytes = Zeroizing::new(rng.random::<[u8; 32]>());
        let signing_key = SigningKey::from_bytes(&bytes);
        Self { signing_key }
    }

    /// Load the signing key seed from `db`, generating and persisting a fresh
    /// one on first use. A stable key is required for signatures to remain
    /// verifiable across process restarts (cross-session tamper evidence).
    pub fn with_persisted_key(db: &sled::Db) -> Result<Self, anyhow::Error> {
        const SEED_KEY: &str = "turn_signer_seed";
        let tree = db.open_tree("signing")?;
        let seed: Zeroizing<[u8; 32]> = match tree.get(SEED_KEY)? {
            Some(stored) => Zeroizing::new(stored.as_ref().try_into().map_err(|_| {
                anyhow::anyhow!("stored signing seed has invalid length: {}", stored.len())
            })?),
            None => {
                let mut rng = rand::rng();
                let bytes = Zeroizing::new(rng.random::<[u8; 32]>());
                tree.insert(SEED_KEY, bytes.as_slice())?;
                tree.flush()?;
                bytes
            }
        };
        Ok(Self {
            signing_key: SigningKey::from_bytes(&seed),
        })
    }

    #[must_use]
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        self.signing_key.sign(data).to_bytes().to_vec()
    }

    pub fn sign_persona_disclosure(&self, disclosure: &mut PersonaDisclosure) {
        let mut payload = disclosure.clone();
        payload.signature = vec![];
        if let Ok(data) = serde_json::to_vec(&payload) {
            disclosure.signature = self.sign(&data);
        }
    }

    pub fn verify_turn(&self, turn: &Turn) -> Result<bool, anyhow::Error> {
        let mut clean = turn.clone();
        clean.signature = vec![];
        let data = serde_json::to_vec(&clean)
            .map_err(|e| anyhow::anyhow!("failed to serialize turn for verification: {e}"))?;
        let sig_bytes: &[u8; 64] = turn.signature.as_slice().try_into().map_err(|_| {
            anyhow::anyhow!(
                "invalid signature length: expected 64 bytes, got {}",
                turn.signature.len()
            )
        })?;
        let sig = Signature::from_bytes(sig_bytes);
        let verifying_key: VerifyingKey = (&self.signing_key).into();
        Ok(verifying_key.verify(&data, &sig).is_ok())
    }
}

impl Default for TurnSigner {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

pub struct ZeroTrustPolicy {
    rules: Vec<(String, Option<String>, RiskLevel)>,
}

impl ZeroTrustPolicy {
    pub fn new() -> Self {
        let rules = vec![
            ("rm".to_string(), None, RiskLevel::Critical),
            ("curl".to_string(), None, RiskLevel::Critical),
            ("git".to_string(), Some("push".to_string()), RiskLevel::High),
            ("cargo".to_string(), None, RiskLevel::Low),
        ];
        Self { rules }
    }

    pub fn classify(&self, tool: &str, args: &str) -> RiskLevel {
        let tool_bin = Path::new(tool)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(tool);
        for (tool_pat, arg_pat, risk) in &self.rules {
            if tool_bin.contains(tool_pat) {
                if let Some(a) = arg_pat {
                    if args.contains(a) {
                        return risk.clone();
                    }
                } else {
                    return risk.clone();
                }
            }
        }
        RiskLevel::Medium
    }
}

impl Default for ZeroTrustPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub event: String,
    pub risk_level: RiskLevel,
    pub actor: String,
    pub signature: Vec<u8>,
}

#[derive(Clone)]
pub struct AuditLogger {
    pub db: Arc<sled::Db>,
    pub signer: Arc<TurnSigner>,
}

impl AuditLogger {
    pub fn new(db: Arc<sled::Db>, signer: Arc<TurnSigner>) -> Self {
        Self { db, signer }
    }

    pub fn log(&self, event: &str, risk_level: RiskLevel, actor: &str) -> anyhow::Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let mut entry = AuditEntry {
            timestamp,
            event: event.to_string(),
            risk_level,
            actor: actor.to_string(),
            signature: vec![],
        };
        let data = serde_json::to_vec(&entry)?;
        entry.signature = self.signer.sign(&data);
        let tree = self.db.open_tree("audit_log")?;
        tree.insert(timestamp.to_be_bytes(), serde_json::to_vec(&entry)?)?;
        tree.flush()?;
        Ok(())
    }

    pub fn log_file_access(&self, path: &Path, actor: &str) -> anyhow::Result<()> {
        let canonical: PathBuf = path.to_path_buf();
        let event_hash = {
            let mut h = sha2::Sha256::new();
            h.update(canonical.display().to_string().as_bytes());
            format!("{:x}", h.finalize())
        };
        self.log(
            &format!("file_access:{}:{}", canonical.display(), event_hash),
            RiskLevel::Low,
            actor,
        )
    }
}
