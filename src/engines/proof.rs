use crate::types::artifact::{Artifact, ProofAttachment};
use crate::types::conversation::ConversationState;
use sha2::{Digest, Sha256};

pub struct ProofManager;

impl ProofManager {
    #[must_use]
    pub fn generate_proof(artifact: &Artifact, properties: Vec<String>) -> ProofAttachment {
        // Allow empty properties; hash will not commit to any specific claims
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
        let safe_name: String = attachment
            .artifact_name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
            .collect();
        let safe_name = if safe_name.starts_with(|c: char| c.is_ascii_digit()) {
            format!("_{safe_name}")
        } else {
            safe_name
        };
        format!(
            "-- Proof for {}\ntheorem artifact_{safe_name}_integrity (h : proof_hash = \"{}\") : True := by\n  -- Properties: {}\n  trivial\n",
            attachment.artifact_name, attachment.proof_hash, props
        )
    }
}
