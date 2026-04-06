use crate::engines::verification::HashChain;
use crate::types::conversation::ConversationState;
use anyhow::{Result, anyhow};
use clap_complete::{Shell, generate};
use std::io;

pub struct ReleaseManager;

impl ReleaseManager {
    #[must_use]
    pub fn run_stability_audit(sigma: &ConversationState) -> Result<()> {
        println!(
            "Starting stability audit for session {}...",
            sigma.session_id
        );

        // Verify CPOP hash chain integrity as a stability check
        if !CpopVerifier::verify_history(&[sigma.clone()]) {
            return Err(anyhow!(
                "Stability audit failed: Hash chain integrity compromised"
            ));
        }

        // Check for any turns with high uncertainty
        let risky_turns = sigma
            .turns
            .iter()
            .filter(|t| t.certainty.unwrap_or(1.0) < 0.3)
            .count();

        if risky_turns > 0 {
            println!(
                "Warning: Stability audit found {} turns with low certainty scores.",
                risky_turns
            );
        }

        println!("Stability audit passed.");
        Ok(())
    }

    pub fn is_mandate_active(sigma: &ConversationState) -> bool {
        let now = ConversationState::now();
        let start = sigma.turns.first().map(|t| t.timestamp).unwrap_or(now);
        let duration_days = (now - start) as f64 / 86400.0;
        duration_days < 14.0 // 14-day mandate window
    }

    pub fn generate_completions(shell: Shell, cmd: &mut clap::Command) {
        generate(shell, cmd, "crosstalk", &mut io::stdout());
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
            if !HashChain::verify(state, &last_hash, &state.state_hash) {
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
