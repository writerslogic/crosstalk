use regex::Regex;
use ed25519_dalek::{SigningKey, VerifyingKey, Signer, Verifier, Signature};
use rand::Rng;

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

        findings
    }
}

pub struct ShellSanity;

impl ShellSanity {
    #[must_use]
    pub fn is_dangerous(cmd: &str) -> bool {
        let dangerous_patterns = [
            "rm -rf /", "rm -rf ~", "curl", "wget", "nc", "netcat", "dd", "mkfs",
        ];
        for p in dangerous_patterns {
            if cmd.contains(p) {
                return true;
            }
        }
        false
    }
}

pub struct InjectionShield;

impl InjectionShield {
    #[must_use]
    pub fn sanitize(input: &str) -> String {
        let forbidden = ["Ignore previous instructions", "System override", "Switch role"];
        let mut sanitized = input.to_string();
        for f in forbidden {
            sanitized = sanitized.replace(f, "[REDACTED]");
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
        let Ok(sig_bytes) = signature_bytes.try_into() else { return false; };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_scanner() {
        let content = "My key is AKIA1234567890ABCDEF";
        assert_eq!(SecretScanner::scan(content).len(), 1);
    }

    #[test]
    fn test_shell_sanity() {
        assert!(ShellSanity::is_dangerous("rm -rf /"));
        assert!(!ShellSanity::is_dangerous("cargo test"));
    }

    #[test]
    fn test_turn_signer() {
        let signer = TurnSigner::new();
        let data = b"turn data";
        let sig = signer.sign(data);
        assert!(signer.verify(data, &sig));
    }
}
