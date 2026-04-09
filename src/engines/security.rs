use crate::types::conversation::Turn;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use zeroize::Zeroizing;
use std::sync::{Arc, OnceLock};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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

pub struct SecretScanner;

impl SecretScanner {
    #[must_use]
    pub fn scan(content: &str) -> Vec<String> {
        let mut findings = vec![];
        if aws_regex().is_match(content) {
            findings.push("AWS_ACCESS_KEY".to_string());
        }
        if gh_regex().is_match(content) {
            findings.push("GITHUB_TOKEN".to_string());
        }

        // Entropy-based detection
        for token in content.split_whitespace() {
            if token.len() > 20 && Self::calculate_entropy(token) > 4.5 {
                findings.push("HighEntropyToken".to_string());
            }
        }

        findings
    }

    fn calculate_entropy(s: &str) -> f64 {
        let mut freq = std::collections::HashMap::new();
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

pub struct ShellSanity;

impl ShellSanity {
    #[must_use]
    pub fn is_dangerous(cmd: &str) -> bool {
        let tokens = Self::tokenize(cmd);
        if tokens.is_empty() {
            return false;
        }

        let bin = &tokens[0];
        let allowlist: HashSet<&str> = [
            "cargo",
            "rustc",
            "git",
            "ls",
            "cat",
            "grep",
            "rustfmt",
            "clippy",
            "tree-sitter",
        ]
        .into_iter()
        .collect();

        // Identify the binary name even if it's an absolute path
        let bin_path = std::path::Path::new(bin);
        let bin_name = bin_path.file_name().and_then(|s| s.to_str()).unwrap_or(bin);

        if !allowlist.contains(bin_name) {
            return true;
        }

        // Secondary check for dangerous patterns within any token or redirection
        let dangerous_bins = ["rm", "curl", "wget", "nc", "netcat", "dd", "mkfs"];
        for token in &tokens {
            if dangerous_bins.iter().any(|&db| token == db) {
                return true;
            }
            if token.contains(">")
                || token.contains("/dev/")
                || token.contains("|")
                || token.contains("&")
                || token.contains(";")
            {
                // Heuristic: reject redirection, piping, or command chaining for now
                return true;
            }
        }

        false
    }

    fn tokenize(cmd: &str) -> Vec<String> {
        let mut tokens = vec![];
        let mut current = String::new();
        let mut in_quote: Option<char> = None;
        let mut escaped = false;

        for c in cmd.chars() {
            if escaped {
                current.push(c);
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if let Some(q) = in_quote {
                if c == q {
                    in_quote = None;
                } else {
                    current.push(c);
                }
            } else if c == '\'' || c == '"' {
                in_quote = Some(c);
            } else if c.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else {
                current.push(c);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
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
        let bytes = Zeroizing::new(<[u8; 32]>::from(rng.random::<[u8; 32]>()));
        let signing_key = SigningKey::from_bytes(&bytes);
        Self { signing_key }
    }

    #[must_use]
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        self.signing_key.sign(data).to_bytes().to_vec()
    }

    #[must_use]
    pub fn verify(&self, data: &[u8], signature_bytes: &[u8]) -> bool {
        let Ok(sig_bytes) = signature_bytes.try_into() else {
            return false;
        };
        let sig = Signature::from_bytes(sig_bytes);
        let verifying_key: VerifyingKey = (&self.signing_key).into();
        verifying_key.verify(data, &sig).is_ok()
    }
}

impl Default for TurnSigner {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnSigner {
    pub fn verify_turn(&self, turn: &Turn) -> bool {
        let mut clean = turn.clone();
        clean.signature = vec![];
        let Ok(data) = serde_json::to_vec(&clean) else { return false };
        self.verify(&data, &turn.signature)
    }

    pub fn verify_chain(&self, turns: &[Turn]) -> bool {
        turns.iter().all(|t| self.verify_turn(t))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretFinding {
    pub line: usize,
    pub pattern_name: String,
    pub redacted_match: String,
}

impl SecretScanner {
    pub fn scan_text(content: &str) -> Vec<SecretFinding> {
        let mut findings = Vec::new();
        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;
            if let Some(m) = aws_regex().find(line) {
                findings.push(SecretFinding {
                    line: line_num,
                    pattern_name: "AWS_ACCESS_KEY".to_string(),
                    redacted_match: format!("{}***", &m.as_str()[..4]),
                });
            }
            if let Some(m) = gh_regex().find(line) {
                findings.push(SecretFinding {
                    line: line_num,
                    pattern_name: "GITHUB_TOKEN".to_string(),
                    redacted_match: format!("{}***", &m.as_str()[..4]),
                });
            }
            for token in line.split_whitespace() {
                if token.len() > 20 && Self::calculate_entropy(token) > 4.5 {
                    findings.push(SecretFinding {
                        line: line_num,
                        pattern_name: "HighEntropyToken".to_string(),
                        redacted_match: format!("{}***", &token[..4.min(token.len())]),
                    });
                    break;
                }
            }
        }
        findings
    }
}

const PROXY_VARS: &[&str] = &[
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
];

pub struct ExfilBlock;

impl ExfilBlock {
    #[must_use]
    pub fn sanitize_env(mut env: HashMap<String, String>) -> HashMap<String, String> {
        for &var in PROXY_VARS {
            env.remove(var);
        }
        env
    }

    #[must_use]
    pub fn detect_proxy_vars() -> Vec<String> {
        PROXY_VARS
            .iter()
            .filter(|&&v| std::env::var(v).is_ok())
            .map(|&v| v.to_string())
            .collect()
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
    #[must_use]
    pub fn new() -> Self {
        let rules = vec![
            ("rm".to_string(), None, RiskLevel::Critical),
            ("curl".to_string(), None, RiskLevel::Critical),
            ("wget".to_string(), None, RiskLevel::Critical),
            ("git".to_string(), Some("push".to_string()), RiskLevel::High),
            ("git".to_string(), Some("force".to_string()), RiskLevel::Critical),
            ("cargo".to_string(), Some("clean".to_string()), RiskLevel::Medium),
            ("cargo".to_string(), None, RiskLevel::Low),
            ("rustc".to_string(), None, RiskLevel::Low),
            ("git".to_string(), None, RiskLevel::Low),
            ("ls".to_string(), None, RiskLevel::Low),
            ("cat".to_string(), None, RiskLevel::Low),
            ("grep".to_string(), None, RiskLevel::Low),
        ];
        Self { rules }
    }

    #[must_use]
    pub fn classify(&self, tool: &str, args: &str) -> RiskLevel {
        // Specific rules (tool + arg pattern) take priority over generic (tool only).
        // Among specifics, take the highest risk. Among generics, take the first match.
        let mut specific: Option<RiskLevel> = None;
        let mut generic: Option<RiskLevel> = None;

        let tool_bin = std::path::Path::new(tool)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(tool);
        for (tool_pat, arg_pat, risk) in &self.rules {
            if !tool_bin.contains(tool_pat.as_str()) {
                continue;
            }
            match arg_pat {
                Some(a) if args.contains(a.as_str()) => {
                    if specific.as_ref().map(|r| risk > r).unwrap_or(true) {
                        specific = Some(risk.clone());
                    }
                }
                None if generic.is_none() => {
                    generic = Some(risk.clone());
                }
                _ => {}
            }
        }

        specific.or(generic).unwrap_or(RiskLevel::Medium)
    }

    pub fn add_rule(&mut self, tool: &str, arg_pattern: Option<&str>, level: RiskLevel) {
        self.rules
            .push((tool.to_string(), arg_pattern.map(|s| s.to_string()), level));
    }

    #[must_use]
    pub fn requires_confirmation(&self, level: &RiskLevel) -> bool {
        matches!(level, RiskLevel::Medium | RiskLevel::High | RiskLevel::Critical)
    }
}

impl Default for ZeroTrustPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzResult {
    pub target: String,
    pub iterations: u64,
    pub crashes: Vec<String>,
}

pub struct FuzzRunner;

impl FuzzRunner {
    #[must_use]
    pub fn parse_output(target: &str, stdout: &str) -> FuzzResult {
        let iterations = stdout
            .lines()
            .find(|l| l.contains("run") || l.contains("exec"))
            .and_then(|l| l.split_whitespace().find_map(|t| t.parse::<u64>().ok()))
            .unwrap_or(0);
        let crashes: Vec<String> = stdout
            .lines()
            .filter(|l| {
                l.contains("ERROR:") || l.contains("CRASH") ||
                l.contains("assertion failed") || l.contains("panicked at")
            })
            .map(|l| l.trim().to_string())
            .collect();
        FuzzResult { target: target.to_string(), iterations, crashes }
    }

    pub async fn run(target: &str, max_iterations: u64) -> anyhow::Result<FuzzResult> {
        let target = target.to_string();
        let target_for_cmd = target.clone();
        let out = tokio::task::spawn_blocking(move || {
            std::process::Command::new("cargo")
                .args(["fuzz", "run", &target_for_cmd, "--", &format!("-runs={max_iterations}")])
                .output()
        })
        .await??;
        let combined = [
            std::str::from_utf8(&out.stdout).unwrap_or(""),
            std::str::from_utf8(&out.stderr).unwrap_or(""),
        ]
        .join("\n");
        Ok(Self::parse_output(&target, &combined))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub vulnerabilities: Vec<String>,
    pub clean: bool,
}

pub struct AuditRunner;

impl AuditRunner {
    #[must_use]
    pub fn parse_output(stdout: &str) -> AuditResult {
        let vulnerabilities: Vec<String> = stdout
            .lines()
            .filter(|l| l.contains("RUSTSEC-") || l.contains("Vulnerable crate"))
            .map(|l| l.trim().to_string())
            .collect();
        let clean = vulnerabilities.is_empty()
            || stdout.contains("No vulnerabilities found");
        AuditResult { vulnerabilities, clean }
    }

    pub async fn run() -> anyhow::Result<AuditResult> {
        let out = tokio::task::spawn_blocking(|| {
            std::process::Command::new("cargo").args(["audit"]).output()
        })
        .await??;
        let stdout = std::str::from_utf8(&out.stdout).unwrap_or("");
        Ok(Self::parse_output(stdout))
    }
}

// ── Drift Monitoring ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshot {
    pub path: PathBuf,
    pub mtime: u64,
    pub size: u64,
    pub hash: String,
}

pub struct DriftMonitor {
    pub root: PathBuf,
    pub snapshots: HashMap<PathBuf, FileSnapshot>,
}

impl DriftMonitor {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), snapshots: HashMap::new() }
    }

    pub fn snapshot_all(&mut self) -> anyhow::Result<()> {
        let mut new_snapshots = HashMap::new();
        self.scan_dir(&self.root.clone(), &mut new_snapshots)?;
        self.snapshots = new_snapshots;
        Ok(())
    }

    fn scan_dir(&self, dir: &Path, acc: &mut HashMap<PathBuf, FileSnapshot>) -> anyhow::Result<()> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    if path.file_name().and_then(|s| s.to_str()) == Some(".git") || 
                       path.file_name().and_then(|s| s.to_str()) == Some("target") {
                        continue;
                    }
                    self.scan_dir(&path, acc)?;
                } else {
                    let metadata = entry.metadata()?;
                    let mtime = metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs();
                    let size = metadata.len();
                    // Basic hash for drift detection
                    let content = std::fs::read(&path).unwrap_or_default();
                    let hash = format!("{:x}", sha2::Sha256::digest(&content));
                    acc.insert(path.clone(), FileSnapshot { path, mtime, size, hash });
                }
            }
        }
        Ok(())
    }

    pub fn detect_drift(&self) -> Vec<String> {
        let mut drift = vec![];
        let mut current = HashMap::new();
        let _ = self.scan_dir(&self.root, &mut current);

        for (path, snap) in &self.snapshots {
            match current.get(path) {
                Some(curr) if curr.hash != snap.hash => {
                    drift.push(format!("Modified: {}", path.display()));
                }
                None => {
                    drift.push(format!("Deleted: {}", path.display()));
                }
                _ => {}
            }
        }
        for path in current.keys() {
            if !self.snapshots.contains_key(path) {
                drift.push(format!("Added: {}", path.display()));
            }
        }
        drift
    }
}

use sha2::Digest;

// ── Signed Audit Logger ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub event: String,
    pub risk_level: RiskLevel,
    pub actor: String,
    pub signature: Vec<u8>,
}

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
        let key = timestamp.to_be_bytes();
        let val = serde_json::to_vec(&entry)?;
        tree.insert(key, val)?;
        Ok(())
    }

    pub fn get_recent(&self, limit: usize) -> anyhow::Result<Vec<AuditEntry>> {
        let tree = self.db.open_tree("audit_log")?;
        let mut entries = vec![];
        for res in tree.iter().rev().take(limit) {
            let (_, val) = res?;
            let entry: AuditEntry = serde_json::from_slice(&val)?;
            entries.push(entry);
        }
        Ok(entries)
    }
}

