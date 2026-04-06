use crate::engines::verification::HashChain;
use crate::types::conversation::ConversationState;
use anyhow::{Result, anyhow};
use clap_complete::{Shell, generate};
use std::io;

#[derive(Debug, Clone)]
pub struct StabilityAuditResult {
    pub passed: u32,
    pub failed: u32,
    pub issues: Vec<String>,
}

impl StabilityAuditResult {
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failed == 0
    }
}

pub struct ReleaseManager;

impl ReleaseManager {
    pub fn run_stability_audit(sigma: &ConversationState) -> Result<StabilityAuditResult> {
        let mut result = StabilityAuditResult { passed: 0, failed: 0, issues: vec![] };

        if !CpopVerifier::verify_history(std::slice::from_ref(sigma)) {
            result.failed += 1;
            result.issues.push("Hash chain integrity compromised".to_string());
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
                    name, artifact.version, artifact.history.len()
                ));
            } else {
                result.passed += 1;
            }
        }

        // Low-certainty turn count check
        let risky_turns = sigma.turns.iter()
            .filter(|t| t.certainty.unwrap_or(1.0) < 0.3)
            .count();
        if risky_turns > 0 {
            result.issues.push(format!(
                "{risky_turns} turn(s) with certainty < 0.3"
            ));
        }
        result.passed += 1;

        if result.failed > 0 {
            Err(anyhow!(
                "Stability audit failed: {} issue(s) — {}",
                result.failed,
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

    pub fn generate_completions(shell: Shell, cmd: &mut clap::Command) {
        generate(shell, cmd, "crosstalk", &mut io::stdout());
    }

    #[must_use]
    pub fn generate_homebrew_formula(version: &str, sha256: &str) -> String {
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
            if !HashChain::verify(state, &last_hash, &state.state_hash).unwrap_or(false) {
                return false;
            }
            last_hash = state.state_hash;
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
            if sigma.completion_probability > 0.95 { "CONVERGED" } else { "IN_PROGRESS" }
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
