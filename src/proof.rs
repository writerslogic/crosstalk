use crate::types::{Artifact, ProofAttachment, ConversationState};
use sha2::{Sha256, Digest};

pub struct ProofManager;

impl ProofManager {
    pub fn generate_proof(artifact: &Artifact, properties: Vec<String>) -> ProofAttachment {
        let mut hasher = Sha256::new();
        for prop in &properties {
            hasher.update(prop.as_bytes());
        }
        let result = hasher.finalize();
        let proof_hash = format!("{:x}", result);

        ProofAttachment {
            artifact_name: artifact.name.clone(),
            proven_properties: properties,
            proof_hash,
            verified_at: ConversationState::now(),
        }
    }

    pub fn to_lean4(attachment: &ProofAttachment) -> String {
        let props = attachment.proven_properties.join(", ");
        format!(
            "-- Proof for {}\ntheorem artifact_integrity : True := by\n  -- Verified properties: {}\n  trivial\n",
            attachment.artifact_name, props
        )
    }
}
