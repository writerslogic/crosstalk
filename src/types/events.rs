use crate::types::conversation::Turn;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TokenReceived(String),
    TurnComplete(Turn),
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
