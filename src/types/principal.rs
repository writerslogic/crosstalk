use serde::{Deserialize, Serialize};

/// Unique identifier for a principal. May be a W3C DID URI (e.g. `did:web:example.com`)
/// or a locally-generated UUID. DID-format IDs enable ToIP credential verification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrincipalId(pub String);

impl PrincipalId {
    pub fn new_uuid() -> Self {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rng().fill_bytes(&mut bytes);
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn is_did(&self) -> bool {
        self.0.starts_with("did:")
    }
}

impl std::fmt::Display for PrincipalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The ceiling on how much the system can decide unilaterally on the principal's behalf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutonomyLevel {
    /// System presents analysis and options; principal makes all decisions.
    AdvisoryOnly,
    /// System acts but principal receives a disclosure event and can veto
    /// before the result is committed to artifacts.
    #[default]
    SemiAutonomous,
    /// System acts and logs decisions for post-hoc review.
    FullAutonomous,
}

/// Regulatory frameworks the principal operates under. Affects data handling
/// and retention enforcement in the `DataMinimizer`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RegulatoryRegime {
    #[default]
    None,
    GdprEu,
    CcpaCa,
    HipaaUs,
    Custom(String),
}

/// Scopes of data use the principal has explicitly consented to. Only scopes
/// present here are exercised by the system; all others are refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentScope {
    /// Retain conversation context within this session only.
    InSessionMemory,
    /// Persist memory records across sessions for this principal.
    CrossSessionMemory,
    /// Allow agents to invoke external tools via the MCP gateway.
    ExternalToolUse,
    /// Write structured telemetry logs for observability.
    TelemetryLogging,
}

/// A consent grant from the principal, optionally signed with their key.
/// The signature (if present) covers the canonical JSON of this struct with
/// `signature` set to `[]`, encoded as UTF-8.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub session_id: String,
    pub granted_at: u64,
    pub scope: Vec<ConsentScope>,
    pub revocable: bool,
    /// ed25519 signature over the consent payload. `None` when the principal
    /// has no key material (consent is asserted, not cryptographically bound).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signature: Vec<u8>,
}

impl ConsentRecord {
    pub fn has(&self, scope: &ConsentScope) -> bool {
        self.scope.contains(scope)
    }
}

/// Hard constraints the principal places on system behavior. These are
/// enforced at the `FiduciaryGate` in the orchestrator and the MCP gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalConstraints {
    pub max_autonomy_level: AutonomyLevel,
    /// Maximum number of days memory records may be retained. `None` means
    /// session-scoped only (records are purged at session end).
    pub data_retention_days: Option<u32>,
    /// Tool categories the principal permits agents to invoke.
    /// Empty means all categories are permitted (no restriction). Populate to
    /// restrict agents to a specific allowlist of category strings.
    pub allowed_tool_categories: Vec<String>,
    /// When true, any turn where an agent was operating under a declared
    /// persona must carry a signed `PersonaDisclosure`.
    pub require_persona_disclosure: bool,
}

impl Default for PrincipalConstraints {
    fn default() -> Self {
        Self {
            max_autonomy_level: AutonomyLevel::SemiAutonomous,
            data_retention_days: None,
            allowed_tool_categories: Vec::new(),
            require_persona_disclosure: true,
        }
    }
}

/// ed25519 public key material used to verify the principal's signed assertions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalKeyMaterial {
    /// Raw 32-byte ed25519 verifying key.
    pub verifying_key: [u8; 32],
}

/// The entity on whose behalf a Crosstalk session operates. All fiduciary
/// duties are owed to this principal. Flows through `ConversationState` and
/// the MCP gateway session context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principal {
    pub id: PrincipalId,
    pub display_name: String,
    /// W3C DID Document, present when `id` is a DID URI and resolution has
    /// been performed by `did_resolver`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub did_document: Option<serde_json::Value>,
    pub regulatory_regime: RegulatoryRegime,
    /// Plain-text goal statements declared by the principal at session start.
    pub interests: Vec<String>,
    pub constraints: PrincipalConstraints,
    pub consent: ConsentRecord,
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_material: Option<PrincipalKeyMaterial>,
}

impl Principal {
    /// Construct a minimal anonymous principal with sensible defaults.
    /// Used when no principal context is provided at session start.
    pub fn anonymous(session_id: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: PrincipalId::new_uuid(),
            display_name: "anonymous".to_string(),
            did_document: None,
            regulatory_regime: RegulatoryRegime::None,
            interests: Vec::new(),
            constraints: PrincipalConstraints::default(),
            consent: ConsentRecord {
                session_id: session_id.to_string(),
                granted_at: now,
                scope: vec![ConsentScope::InSessionMemory],
                revocable: true,
                signature: Vec::new(),
            },
            created_at: now,
            key_material: None,
        }
    }
}
