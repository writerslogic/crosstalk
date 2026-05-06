use crate::types::conversation::Turn;

#[derive(Debug, Clone)]
pub struct ArtifactSnapshot {
    pub name: String,
    pub skeleton: String,
    pub version: u32,
    pub diff_count: usize,
}

#[derive(Debug, Clone)]
pub struct EntropyEntry {
    pub artifact_name: String,
    pub scores: Vec<(String, f64)>, // (agent_id, score)
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TokenReceived {
        agent_id: String,
        token: String,
    },
    TurnComplete(Turn),
    ConvergenceUpdated {
        p: f64,
        certainty: f64,
        agent_weights: Vec<(String, f64)>,
    },
    ArtifactsUpdated(Vec<ArtifactSnapshot>),
    EntropyUpdated(Vec<EntropyEntry>),
    GodViewUpdated {
        frame: u64,
        avg_certainty: f64,
        avg_surprise: f64,
        agent_count: usize,
    },
    CheckpointWritten(u32),
    Error(String),
}

#[derive(Debug, Clone)]
pub enum ControlSignal {
    Pause,
    Resume,
    Rewind(u32),
    Shutdown,
    Inject(String),
    LockCode(String), // artifact name
    MuteAgent(String), // agent id
    DampenSwarm(f64), // dampening factor
}
