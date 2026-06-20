use serde::{Deserialize, Serialize};

/// A disclosure attached to a turn when an agent operated under a declared persona.
/// Signed by `TurnSigner` so the principal can verify authenticity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaDisclosure {
    pub turn_index: u32,
    pub agent_id: String,
    pub persona_name: String,
    /// SHA-256 of the system prompt that established the persona.
    pub system_prompt_hash: [u8; 32],
    /// ed25519 signature over the canonical JSON of this struct with
    /// `signature` set to `[]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signature: Vec<u8>,
}

/// Events emitted when system behavior touches a fiduciary duty boundary.
/// Carried by `StreamEvent::FiduciarySignal` to the principal's channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FiduciaryDutyEvent {
    /// An agent was operating under a persona; disclosure attached to turn.
    PersonaDisclosed(PersonaDisclosure),
    /// A turn was committed; entry added to the decision ledger (Account duty).
    DecisionCommitted {
        turn_index: u32,
        decision_excerpt: String,
        /// SHA-256 of the rationale context used to produce this turn.
        rationale_hash: [u8; 32],
        /// `state_hash` of the prior `ConversationState` — the chain link.
        chain_link: [u8; 32],
    },
    /// Convergence certainty was below the Care threshold; action deferred.
    CertaintyGateFired {
        turn_index: u32,
        certainty: f64,
        threshold: f64,
    },
    /// A tool call was blocked because it exceeded the principal's autonomy level.
    ToolCallBlocked { tool_name: String, reason: String },
    /// A memory record was purged per the principal's retention policy.
    RetentionPurge {
        session_id: String,
        records_deleted: usize,
    },
    /// Principal consent was checked for a specific scope.
    ConsentChecked { scope: String, granted: bool },
}
