use crate::types::conversation::Turn;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TokenReceived { agent_id: String, token: String },
    TurnComplete(Turn),
    ConvergenceUpdated { p: f64, certainty: f64 },
    ArtifactsUpdated(Vec<(String, String)>),
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
