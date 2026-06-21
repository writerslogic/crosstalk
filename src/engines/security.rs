use crate::types::conversation::Turn;
use crate::types::fiduciary::PersonaDisclosure;
use argon2::Argon2;
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
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
    ///
    /// If `CROSSTALK_SIGNING_PASSPHRASE` is set the seed is encrypted at rest
    /// (ChaCha20-Poly1305 under an Argon2id-derived key); otherwise it is stored
    /// in the clear with a warning.
    pub fn with_persisted_key(db: &sled::Db) -> Result<Self, anyhow::Error> {
        let passphrase = std::env::var("CROSSTALK_SIGNING_PASSPHRASE").ok();
        let signer = Self::with_persisted_key_passphrase(db, passphrase.as_deref())?;
        // Cross-domain pin: if the operator recorded the identity out-of-band,
        // a seed that no longer matches it is a substitution and must abort.
        if let Ok(expected) = std::env::var("CROSSTALK_EXPECTED_PUBKEY")
            && !expected.trim().is_empty()
            && !expected
                .trim()
                .eq_ignore_ascii_case(&signer.verifying_key_hex())
        {
            return Err(anyhow::anyhow!(
                "signing identity {} does not match pinned CROSSTALK_EXPECTED_PUBKEY {}",
                signer.verifying_key_hex(),
                expected.trim()
            ));
        }
        tracing::info!(identity = %signer.verifying_key_hex(), "turn signing identity");
        Ok(signer)
    }

    /// As [`Self::with_persisted_key`] but with the passphrase supplied
    /// explicitly (the env var is read only by the wrapper). Exposed so callers
    /// and tests can drive the encrypted path deterministically without relying
    /// on process-global environment state.
    pub fn with_persisted_key_passphrase(
        db: &sled::Db,
        passphrase: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        const SEED_KEY: &str = "turn_signer_seed";
        let tree = db.open_tree("signing")?;
        let seed: Zeroizing<[u8; 32]> = match (tree.get(SEED_KEY)?, passphrase) {
            (None, pass) => {
                let seed = Zeroizing::new(rand::rng().random::<[u8; 32]>());
                store_seed(&tree, SEED_KEY, &seed, pass)?;
                seed
            }
            (Some(blob), Some(pass)) if is_encrypted(&blob) => decrypt_seed(&blob, pass)?,
            (Some(blob), Some(pass)) => {
                // Stored in the clear but a passphrase is now set: migrate by
                // re-storing it encrypted so it is no longer readable at rest.
                let seed = read_plaintext_seed(&blob)?;
                store_seed(&tree, SEED_KEY, &seed, Some(pass))?;
                seed
            }
            (Some(blob), None) if is_encrypted(&blob) => {
                return Err(anyhow::anyhow!(
                    "signing seed is encrypted but CROSSTALK_SIGNING_PASSPHRASE is not set; \
                     set it to the original passphrase to keep prior signatures verifiable"
                ));
            }
            (Some(blob), None) => read_plaintext_seed(&blob)?,
        };
        let signer = Self {
            signing_key: SigningKey::from_bytes(&seed),
        };
        // Pin the public identity on first run and check it on every load: a seed
        // swapped without a matching pin update (or vice versa) is tamper evidence.
        const PUBKEY_KEY: &str = "turn_signer_pubkey";
        let pubkey_hex = signer.verifying_key_hex();
        match tree.get(PUBKEY_KEY)? {
            Some(stored) if stored.as_ref() != pubkey_hex.as_bytes() => {
                return Err(anyhow::anyhow!(
                    "signing seed does not match the pinned public key in this database; \
                     the seed or the pin was tampered with"
                ));
            }
            Some(_) => {}
            None => {
                tree.insert(PUBKEY_KEY, pubkey_hex.as_bytes())?;
                tree.flush()?;
            }
        }
        Ok(signer)
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

    /// The public (verifying) half of the signing key. Safe to publish; this is
    /// the identity that should be pinned out-of-band so a swapped seed cannot
    /// silently re-sign a forged transcript.
    #[must_use]
    pub fn verifying_key(&self) -> VerifyingKey {
        (&self.signing_key).into()
    }

    /// Hex-encoded public key — the value to record out-of-band and to pass via
    /// `CROSSTALK_EXPECTED_PUBKEY` for cross-domain pinning.
    #[must_use]
    pub fn verifying_key_hex(&self) -> String {
        to_hex(&self.verifying_key().to_bytes())
    }

    /// Self-verification against this signer's own key. Prefer a [`TurnVerifier`]
    /// built from the *pinned* public key for trust decisions: verifying with the
    /// same secret you are protecting is circular.
    pub fn verify_turn(&self, turn: &Turn) -> Result<bool, anyhow::Error> {
        verify_turn_signature(turn, &self.verifying_key())
    }
}

/// Verifies turn signatures using only a public key — no secret required, so a
/// transcript can be checked by a party that never holds the signing seed.
pub struct TurnVerifier {
    verifying_key: VerifyingKey,
}

impl TurnVerifier {
    #[must_use]
    pub fn new(verifying_key: VerifyingKey) -> Self {
        Self { verifying_key }
    }

    pub fn from_hex(hex: &str) -> Result<Self, anyhow::Error> {
        let bytes = from_hex(hex.trim())?;
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("public key must be 32 bytes, got {}", bytes.len()))?;
        let verifying_key = VerifyingKey::from_bytes(&arr)
            .map_err(|e| anyhow::anyhow!("invalid ed25519 public key: {e}"))?;
        Ok(Self { verifying_key })
    }

    /// Resolve the pinned verifier for `db`: `CROSSTALK_EXPECTED_PUBKEY` is
    /// authoritative when set (cross-domain pin); otherwise the public key
    /// recorded in the `signing` tree on first run is used. Returns `None` when
    /// no key has been pinned yet (a brand-new database).
    pub fn pinned(db: &sled::Db) -> Result<Option<Self>, anyhow::Error> {
        if let Ok(hex) = std::env::var("CROSSTALK_EXPECTED_PUBKEY")
            && !hex.trim().is_empty()
        {
            return Ok(Some(Self::from_hex(hex.trim())?));
        }
        let tree = db.open_tree("signing")?;
        match tree.get("turn_signer_pubkey")? {
            Some(v) => {
                let hex = std::str::from_utf8(&v)
                    .map_err(|e| anyhow::anyhow!("pinned public key is not valid utf-8: {e}"))?;
                Ok(Some(Self::from_hex(hex)?))
            }
            None => Ok(None),
        }
    }

    #[must_use]
    pub fn key_hex(&self) -> String {
        to_hex(&self.verifying_key.to_bytes())
    }

    pub fn verify_turn(&self, turn: &Turn) -> Result<bool, anyhow::Error> {
        verify_turn_signature(turn, &self.verifying_key)
    }
}

fn verify_turn_signature(turn: &Turn, verifying_key: &VerifyingKey) -> Result<bool, anyhow::Error> {
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
    Ok(verifying_key.verify(&data, &sig).is_ok())
}

impl Default for TurnSigner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Signing-seed storage (plaintext or passphrase-encrypted at rest) ─────────
//
// Stored blob layout under the `signing` tree:
//   plaintext : [0x00] ++ seed(32)        (a bare 32-byte value is also accepted
//                                           for backward compatibility)
//   encrypted : [0x01] ++ salt(16) ++ nonce(12) ++ ciphertext(48)

const SEED_PLAINTEXT_TAG: u8 = 0x00;
const SEED_ENCRYPTED_TAG: u8 = 0x01;
const SEED_SALT_LEN: usize = 16;
const SEED_NONCE_LEN: usize = 12;

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn from_hex(s: &str) -> Result<Vec<u8>, anyhow::Error> {
    if !s.len().is_multiple_of(2) {
        return Err(anyhow::anyhow!("hex string has odd length: {}", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| anyhow::anyhow!("invalid hex digit: {e}"))
        })
        .collect()
}

fn is_encrypted(blob: &[u8]) -> bool {
    blob.first() == Some(&SEED_ENCRYPTED_TAG)
}

fn read_plaintext_seed(blob: &[u8]) -> Result<Zeroizing<[u8; 32]>, anyhow::Error> {
    let raw: &[u8] = match blob {
        [SEED_PLAINTEXT_TAG, rest @ ..] => rest,
        _ => blob, // legacy: bare 32-byte seed
    };
    let arr: [u8; 32] = raw
        .try_into()
        .map_err(|_| anyhow::anyhow!("stored signing seed has invalid length: {}", raw.len()))?;
    Ok(Zeroizing::new(arr))
}

fn derive_seed_key(passphrase: &str, salt: &[u8]) -> Result<Zeroizing<[u8; 32]>, anyhow::Error> {
    let mut key = Zeroizing::new([0u8; 32]);
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, key.as_mut_slice())
        .map_err(|e| anyhow::anyhow!("argon2 key derivation failed: {e}"))?;
    Ok(key)
}

fn store_seed(
    tree: &sled::Tree,
    key: &str,
    seed: &[u8; 32],
    passphrase: Option<&str>,
) -> Result<(), anyhow::Error> {
    let blob = match passphrase {
        Some(pass) => {
            let mut rng = rand::rng();
            let salt = rng.random::<[u8; SEED_SALT_LEN]>();
            let nonce = rng.random::<[u8; SEED_NONCE_LEN]>();
            let derived = derive_seed_key(pass, &salt)?;
            let cipher = ChaCha20Poly1305::new(Key::from_slice(derived.as_slice()));
            let ciphertext = cipher
                .encrypt(Nonce::from_slice(&nonce), seed.as_slice())
                .map_err(|e| anyhow::anyhow!("signing seed encryption failed: {e}"))?;
            let mut blob =
                Vec::with_capacity(1 + SEED_SALT_LEN + SEED_NONCE_LEN + ciphertext.len());
            blob.push(SEED_ENCRYPTED_TAG);
            blob.extend_from_slice(&salt);
            blob.extend_from_slice(&nonce);
            blob.extend_from_slice(&ciphertext);
            blob
        }
        None => {
            tracing::warn!(
                "signing seed stored unencrypted; set CROSSTALK_SIGNING_PASSPHRASE to encrypt it at rest"
            );
            let mut blob = Vec::with_capacity(1 + 32);
            blob.push(SEED_PLAINTEXT_TAG);
            blob.extend_from_slice(seed);
            blob
        }
    };
    tree.insert(key, blob)?;
    tree.flush()?;
    Ok(())
}

fn decrypt_seed(blob: &[u8], passphrase: &str) -> Result<Zeroizing<[u8; 32]>, anyhow::Error> {
    const HEADER: usize = 1 + SEED_SALT_LEN + SEED_NONCE_LEN;
    if blob.len() <= HEADER {
        return Err(anyhow::anyhow!("encrypted signing seed is truncated"));
    }
    let salt = &blob[1..1 + SEED_SALT_LEN];
    let nonce = &blob[1 + SEED_SALT_LEN..HEADER];
    let ciphertext = &blob[HEADER..];
    let derived = derive_seed_key(passphrase, salt)?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(derived.as_slice()));
    let plaintext = cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| {
            anyhow::anyhow!("signing seed decryption failed (wrong CROSSTALK_SIGNING_PASSPHRASE?)")
        })?;
    read_plaintext_seed(&plaintext)
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
