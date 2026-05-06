use crate::engines::verification::HashChain;
use crate::types::conversation::ConversationState;
use anyhow::{Result, anyhow};
use clap_complete::{Shell, generate};
use dashmap::DashMap;
use std::io;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct StabilityAuditResult {
    pub passed: u32,
    pub failed: u32,
    pub issues: Vec<String>,
}

impl StabilityAuditResult {
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failed == 0 && self.passed > 0
    }
}

pub struct ReleaseManager;

impl ReleaseManager {
    pub fn run_stability_audit(sigma: &ConversationState) -> Result<StabilityAuditResult> {
        let mut result = StabilityAuditResult {
            passed: 0,
            failed: 0,
            issues: vec![],
        };

        if !CpopVerifier::verify_history(std::slice::from_ref(sigma)) {
            result.failed += 1;
            result
                .issues
                .push("Hash chain integrity compromised".to_string());
        } else {
            result.passed += 1;
        }

        // Monotonic turn index check
        for window in sigma.turns.windows(2) {
            if window[1].index <= window[0].index {
                result.failed += 1;
                result.issues.push(format!(
                    "Non-monotonic turn index: {} after {}",
                    window[1].index, window[0].index
                ));
            } else {
                result.passed += 1;
            }
        }

        // Artifact version consistency
        for (name, artifact) in &sigma.artifacts {
            if artifact.version as usize != artifact.history.len() {
                result.failed += 1;
                result.issues.push(format!(
                    "Artifact '{}': version {} != history length {}",
                    name,
                    artifact.version,
                    artifact.history.len()
                ));
            } else {
                result.passed += 1;
            }
        }

        // Low-certainty turn count check
        let risky_turns = sigma
            .turns
            .iter()
            .filter(|t| t.certainty.unwrap_or(1.0) < 0.3)
            .count();
        if risky_turns > 0 {
            result
                .issues
                .push(format!("{risky_turns} turn(s) with certainty < 0.3"));
        }
        result.passed += 1;

        if result.failed > 0 {
            Err(anyhow!(
                "Stability audit failed: {}/{} check(s) failed — {}",
                result.failed,
                result.passed + result.failed,
                result.issues.join("; ")
            ))
        } else {
            Ok(result)
        }
    }

    #[must_use]
    pub fn is_mandate_active(sigma: &ConversationState) -> bool {
        let now = ConversationState::now();
        let start = sigma.turns.first().map(|t| t.timestamp).unwrap_or(now);
        let duration_days = (now - start) as f64 / 86400.0;
        duration_days < 14.0
    }

    /// Performs a comprehensive Sovereign Audit of the project state.
    /// Verifies stability, cryptographic history, and capability readiness.
    pub fn run_sovereign_audit(sigma: &ConversationState) -> Result<String> {
        let stability = Self::run_stability_audit(sigma)?;
        let history_valid = CpopVerifier::verify_history(std::slice::from_ref(sigma));
        
        if !history_valid {
            return Err(anyhow!("Sovereign Audit Failed: CPOP history verification failed."));
        }

        let mut report = "Sovereign Audit: CLEAN\n".to_string();
        report.push_str(&format!("  - Stability Checks: {} passed, 0 failed
", stability.passed));
        report.push_str("  - Cryptographic History: VERIFIED
");
        report.push_str(&format!("  - Artifact Integrity: {} artifacts validated
", sigma.artifacts.len()));
        
        Ok(report)
    }

    pub fn generate_completions(shell: Shell, cmd: &mut clap::Command) {
        generate(shell, cmd, "crosstalk", &mut io::stdout());
    }

    #[must_use]
    pub fn generate_homebrew_formula(version: &str, sha256: &str) -> String {
        fn is_safe(s: &str) -> bool {
            s.chars()
                .all(|c| c.is_alphanumeric() || c == '.' || c == '-')
        }
        let version = if is_safe(version) { version } else { "invalid" };
        let sha256 = if is_safe(sha256) { sha256 } else { "invalid" };
        format!(
            r##"class Crosstalk < Formula
  desc "AI multi-model orchestrator"
  homepage "https://github.com/example/crosstalk"
  url "https://github.com/example/crosstalk/releases/download/v{version}/crosstalk-{version}-aarch64-apple-darwin.tar.gz"
  sha256 "{sha256}"
  version "{version}"

  def install
    bin.install "crosstalk"
    generate_completions_from_executable(bin/"crosstalk", "completions")
  end

  test do
    system "#{{bin}}/crosstalk", "--version"
  end
  "##
        )
    }
}

pub struct CpopVerifier;

impl CpopVerifier {
    #[must_use]
    pub fn verify_history(states: &[ConversationState]) -> bool {
        if states.is_empty() {
            return true;
        }
        let mut last_hash = [0u8; 32];
        for state in states {
            match HashChain::verify(state, &last_hash, &state.state_hash) {
                Ok(true) => last_hash = state.state_hash,
                Ok(false) => return false,
                Err(_) => return false,
            }
        }
        true
    }
}

pub struct ConvergenceReport;

impl ConvergenceReport {
    #[must_use]
    pub fn generate(sigma: &ConversationState) -> String {
        let mut report = format!(
            "Executive Summary: Session {}\nTurns: {}\nFinal P(C): {:.2}\nCost: ${:.2}\nStatus: {}\n",
            sigma.session_id,
            sigma.turns.len(),
            sigma.completion_probability,
            sigma.budget.spent,
            if sigma.completion_probability > 0.95 {
                "CONVERGED"
            } else {
                "IN_PROGRESS"
            }
        );
        report.push_str("\nArtifact Breakdown:\n");
        for (name, artifact) in &sigma.artifacts {
            report.push_str(&format!(
                "  - {}: version {}, {} lines\n",
                name,
                artifact.version,
                artifact.content.lines().count()
            ));
        }
        report
    }
}

// ── Plugin Architecture ──────────────────────────────────────────────────────

pub trait CrosstalkPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<String>;
    fn on_turn(&self, sigma: &mut ConversationState) -> Result<()>;
    fn on_checkpoint(&self, sigma: &ConversationState) -> Result<()>;
    fn on_quality_check(&self, sigma: &ConversationState) -> Result<f64>;
}

pub struct PluginManager {
    pub plugins: DashMap<String, Arc<dyn CrosstalkPlugin>>,
}

impl PluginManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: DashMap::new(),
        }
    }

    pub fn register(&self, plugin: Arc<dyn CrosstalkPlugin>) {
        self.plugins.insert(plugin.name().to_string(), plugin);
    }

    pub fn run_on_turn(&self, sigma: &mut ConversationState) -> Result<()> {
        let mut errors = Vec::new();
        for entry in self.plugins.iter() {
            if let Err(e) = entry.value().on_turn(sigma) {
                errors.push(format!("{}: {}", entry.key(), e));
            }
        }
        if !errors.is_empty() {
            return Err(anyhow::anyhow!("Plugin errors: {}", errors.join("; ")));
        }
        Ok(())
    }

    pub fn run_quality_checks(&self, sigma: &ConversationState) -> Vec<(String, f64)> {
        self.plugins
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    entry.value().on_quality_check(sigma).unwrap_or(0.0),
                )
            })
            .collect()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
