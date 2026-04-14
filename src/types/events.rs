use crate::types::conversation::Turn;

#[derive(Debug, Clone)]
pub struct ArtifactSnapshot {
    pub name: String,
    pub skeleton: String,
    pub version: u32,
    pub diff_count: usize,
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
}
