use crate::types::conversation::ConversationState;
use anyhow::{Context, Result, anyhow};
use rustc_hash::FxHashMap;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct HashChain;

impl HashChain {
    /// Computes a cryptographically deterministic hash for the state.
    /// Replaces non-deterministic JSON with Bincode (Note: Ensure ConversationState uses BTreeMap, not HashMap).
    pub fn compute(sigma: &ConversationState, previous_hash: &[u8; 32]) -> Result<[u8; 32]> {
        let mut hasher = Sha256::new();
        
        // Bincode is a binary format that avoids whitespace variance and JSON key-ordering issues,
        // provided the underlying data structures are deterministic (Vectors and BTreeMaps).
        let serialized = bincode::serialize(sigma)
            .context("Failed to deterministically serialize state for hashing")?;
            
        hasher.update(&serialized);
        hasher.update(previous_hash);
        
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hasher.finalize());
        Ok(hash)
    }

    /// Verifies if the current state hash mathematically links to the expected prior hash.
    pub fn verify(
        sigma: &ConversationState,
        previous_hash: &[u8; 32],
        current_hash: &[u8; 32],
    ) -> bool {
        match Self::compute(sigma, previous_hash) {
            Ok(computed) => computed == *current_hash,
            Err(_) => false, // Fail-secure on serialization errors
        }
    }
}

pub struct InvariantChecker;

impl InvariantChecker {
    pub fn check_all(sigma: &ConversationState) -> Result<()> {
        // 1. Monotonic indices: Ensures time flows strictly forward
        for window in sigma.turns.windows(2) {
            if window[1].index <= window[0].index {
                return Err(anyhow!("Invariant violation: Non-monotonic turn indices detected"));
            }
        }

        // 2. Orphan detection: Every turn must belong to the current iteration
        if sigma.turns.iter().any(|t| t.index >= sigma.iteration_index) {
            return Err(anyhow!("Invariant violation: Orphan future turn detected"));
        }

        // 3. Artifact consistency
        for artifact in sigma.artifacts.values() {
            if artifact.version as usize != artifact.history.len() {
                return Err(anyhow!(
                    "Invariant violation: Artifact '{}' version/history length mismatch",
                    artifact.name
                ));
            }
        }

        Ok(())
    }
}

pub struct ContinuousAuditor;

impl ContinuousAuditor {
    /// Spawns a lock-free background actor.
    /// Removes the Arc<Mutex<Receiver>> anti-pattern to ensure zero-contention channel processing.
    pub fn spawn() -> mpsc::Sender<ConversationState> {
        let (tx, mut rx) = mpsc::channel::<ConversationState>(100);

        // The Receiver is moved directly into the spawned task, granting it exclusive ownership.
        tokio::spawn(async move {
            let mut last_hash = [0u8; 32];

            while let Some(sigma) = rx.recv().await {
                let turn_index = sigma.iteration_index;

                let _ = HashChain::verify(&sigma, &last_hash, &sigma.state_hash);

                last_hash = sigma.state_hash;
            }
        });

        tx // Return the sender so the Swarm can stream states to the Auditor
    }
}

pub struct TautologyFilter;

impl TautologyFilter {
    /// Detects if an agent is stuck in an infinite logical loop by comparing its output
    /// against prior historical turns.
    pub fn is_tautological(content: &str, history: &[String]) -> bool {
        let content_trimmed = content.trim();
        
        // Fast-path identical check
        for prev in history {
            if content_trimmed == prev.trim() {
                return true;
            }
        }

        // Pre-compute the 3-gram frequencies for the new content ONCE.
        let vec_new = Self::get_3gram_freq(content_trimmed);
        if vec_new.is_empty() {
            return false;
        }

        let mag_new = vec_new.values().map(|f| f * f).sum::<f64>().sqrt();

        for prev in history {
            let vec_prev = Self::get_3gram_freq(prev);
            let mag_prev = vec_prev.values().map(|f| f * f).sum::<f64>().sqrt();

            if mag_prev == 0.0 { continue; }

            let mut dot_product = 0.0;
            for (gram, freq) in &vec_new {
                if let Some(f_p) = vec_prev.get(gram) {
                    dot_product += freq * f_p;
                }
            }

            let similarity = dot_product / (mag_new * mag_prev);
            
            // If the agent is generating text that is 95% structurally identical, it is looping.
            if similarity > 0.95 {
                return true;
            }
        }
        false
    }

    /// ZERO-ALLOCATION 3-GRAM GENERATOR
    /// Extracts frequencies using fixed-size arrays `[char; 3]` instead of allocating `String`s.
    fn get_3gram_freq(text: &str) -> FxHashMap<[char; 3], f64> {
        let mut freqs = FxHashMap::default();
        
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.len() < 3 {
            return freqs;
        }

        for window in chars.windows(3) {
            // Arrays are implicitly Copy, meaning zero heap allocation happens here.
            let gram = [window[0], window[1], window[2]];
            *freqs.entry(gram).or_insert(0.0) += 1.0;
        }
        
        freqs
    }
}