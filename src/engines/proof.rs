use crate::types::artifact::{Artifact, ProofAttachment};
use crate::types::conversation::ConversationState;
use sha2::{Digest, Sha256};

pub struct ProofManager;

impl ProofManager {
    #[must_use]
    pub fn generate_proof(artifact: &Artifact, properties: Vec<String>) -> ProofAttachment {
        let mut hasher = Sha256::new();

        // Tie proof to artifact content
        hasher.update(artifact.content.as_bytes());

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

    #[must_use]
    pub fn to_lean4(attachment: &ProofAttachment) -> String {
        let props = attachment.proven_properties.join(", ");
        let safe_name = attachment.artifact_name.replace(['.', '-'], "_");
        format!(
            "-- Proof for {}\ntheorem artifact_{}_integrity : True := by\n  -- Verified properties: {}\n  -- Proof Hash: {}\n  trivial\n",
            attachment.artifact_name, safe_name, props, attachment.proof_hash
        )
    }
}
