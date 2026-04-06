use crate::types::conversation::ConversationState;
use anyhow::{Result, anyhow};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub struct HashChain;

impl HashChain {
    /// Computes the hash for the current state σ, linked to the previous hash.
    pub fn compute(sigma: &ConversationState, previous_hash: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        let serialized = serde_json::to_vec(sigma).expect("Failed to serialize state for hashing");
        hasher.update(&serialized);
        hasher.update(previous_hash);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verifies if the current state hash matches the expected hash.
    #[must_use]
    pub fn verify(
        sigma: &ConversationState,
        previous_hash: &[u8; 32],
        current_hash: &[u8; 32],
    ) -> bool {
        let computed = Self::compute(sigma, previous_hash);
        &computed == current_hash
    }
}

pub struct InvariantChecker;

impl InvariantChecker {
    pub fn check_all(sigma: &ConversationState) -> Result<()> {
        // 1. Monotonic indices: check if turns are ordered
        for i in 1..sigma.turns.len() {
            if sigma.turns[i].index <= sigma.turns[i - 1].index {
                return Err(anyhow!("Invariant violation: Non-monotonic turn indices"));
            }
        }

        // 2. No orphan turns: Every turn index <= iteration_index
        for turn in &sigma.turns {
            if turn.index >= sigma.iteration_index {
                return Err(anyhow!("Invariant violation: Orphan turn detected"));
            }
        }

        // 3. Artifact consistency: version == history.len()
        for artifact in sigma.artifacts.values() {
            if artifact.version as usize != artifact.history.len() {
                return Err(anyhow!(
                    "Invariant violation: Artifact {} version mismatch",
                    artifact.name
                ));
            }
        }

        Ok(())
    }
}

pub struct ContinuousAuditor;

impl ContinuousAuditor {
    pub fn audit_step(sigma: &ConversationState, last_hash: &[u8; 32]) -> Result<[u8; 32]> {
        let new_hash = HashChain::compute(sigma, last_hash);
        Ok(new_hash)
    }
}

pub struct TautologyFilter;

impl TautologyFilter {
    #[must_use]
    pub fn is_tautological(content: &str, history: &[String]) -> bool {
        for prev in history {
            if content.trim() == prev.trim() {
                return true;
            }
            if Self::cosine_similarity(content, prev) > 0.95 {
                return true;
            }
        }
        false
    }

    fn cosine_similarity(a: &str, b: &str) -> f64 {
        let vec_a = Self::get_3gram_freq(a);
        let vec_b = Self::get_3gram_freq(b);

        let mut dot_product = 0.0;
        for (gram, freq) in &vec_a {
            if let Some(f_b) = vec_b.get(gram) {
                dot_product += freq * f_b;
            }
        }

        let mag_a = vec_a.values().map(|f| f * f).sum::<f64>().sqrt();
        let mag_b = vec_b.values().map(|f| f * f).sum::<f64>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot_product / (mag_a * mag_b)
    }

    fn get_3gram_freq(text: &str) -> HashMap<String, f64> {
        let mut freqs = HashMap::new();
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.len() < 3 {
            return freqs;
        }

        for i in 0..chars.len() - 2 {
            let gram: String = chars[i..i + 3].iter().collect();
            *freqs.entry(gram).or_insert(0.0) += 1.0;
        }
        freqs
    }
}
