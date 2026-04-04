use crate::types::ConversationState;
use sha2::{Sha256, Digest};
use anyhow::Result;

pub struct HashChain;

impl HashChain {
    /// Computes the hash for the current state σ, linked to the previous hash.
    /// H(σ_i) = SHA256(serialize(σ_i) || previous_hash)
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
    pub fn verify(sigma: &ConversationState, previous_hash: &[u8; 32], current_hash: &[u8; 32]) -> bool {
        let computed = Self::compute(sigma, previous_hash);
        &computed == current_hash
    }
}

pub struct ContinuousAuditor;

impl ContinuousAuditor {
    pub fn audit_step(sigma: &ConversationState, last_hash: &[u8; 32]) -> Result<[u8; 32]> {
        let new_hash = HashChain::compute(sigma, last_hash);
        // In a real auditor, we'd compare this against the stored hash in Sled.
        Ok(new_hash)
    }
}

pub struct TautologyFilter;

impl TautologyFilter {
    pub fn is_tautological(content: &str, history: &[String]) -> bool {
        for prev in history {
            if content.trim() == prev.trim() {
                return true;
            }
        }
        false
    }
}
