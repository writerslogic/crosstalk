use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::Rng;
use regex::Regex;
use std::collections::HashSet;

pub struct SecretScanner;

impl SecretScanner {
    #[must_use]
    pub fn scan(content: &str) -> Vec<String> {
        let mut findings = vec![];
        let aws_regex = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
        let gh_regex = Regex::new(r"ghp_[a-zA-Z0-9]{36}").unwrap();

        if aws_regex.is_match(content) {
            findings.push("AWS_ACCESS_KEY".to_string());
        }
        if gh_regex.is_match(content) {
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
        let forbidden = [
            r"(?i)ignore all prior instructions",
            r"(?i)you are now an? (.*) agent",
            r"(?i)system prompt override",
        ];
        let mut sanitized = input.to_string();
        for f in forbidden {
            let re = Regex::new(f).unwrap();
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
        let bytes: [u8; 32] = rng.random();
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
