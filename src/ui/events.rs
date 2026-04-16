use crate::types::events::{ControlSignal, StreamEvent};
use crate::ui::app::{App, AppMode};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::time::Duration;
use tokio::sync::mpsc;

pub enum Action {
    None,
    Send(ControlSignal),
    SendTwo(ControlSignal, ControlSignal),
    Shutdown,
}

/// Drain all pending stream events into the App.
pub fn drain_stream_events(app: &mut App, stream_rx: &mut mpsc::Receiver<StreamEvent>) {
    while let Ok(ev) = stream_rx.try_recv() {
        match ev {
            StreamEvent::TokenReceived { agent_id, token } => app.push_token(&agent_id, &token),
            StreamEvent::TurnComplete(turn) => app.commit_turn(&turn),
            StreamEvent::ConvergenceUpdated {
                p,
                certainty,
                agent_weights,
            } => {
                app.set_convergence(p, certainty);
                app.agent_weights = agent_weights.into_iter().collect();
            }
            StreamEvent::ArtifactsUpdated(list) => app.artifacts = list,
            StreamEvent::EntropyUpdated(entries) => {
                app.entropy_scores = entries.into_iter().map(|e| crate::ui::app::EntropyRow {
                    artifact: e.artifact_name,
                    agents: e.scores,
                }).collect();
            }
            StreamEvent::CheckpointWritten(idx) => {
                app.push_event(format!("Checkpoint i_{idx}"));
            }
            StreamEvent::Error(msg) => app.push_event(format!("Error: {msg}")),
        }
    }
}

/// Poll for a keyboard event (non-blocking, up to `timeout`).
pub fn poll_key(timeout: Duration) -> Option<event::KeyEvent> {
    if event::poll(timeout).ok()? {
        if let Event::Key(key) = event::read().ok()? {
            if key.kind == KeyEventKind::Press {
                return Some(key);
            }
        }
    }
    None
}

/// Translate a key press into an App mutation + Action.
pub fn handle_key(app: &mut App, key: event::KeyEvent) -> Action {
    if app.showing_inject {
        match key.code {
            KeyCode::Esc => {
                app.inject_buffer.clear();
                app.showing_inject = false;
                app.mode = AppMode::Streaming;
                Action::Send(ControlSignal::Resume)
            }
            KeyCode::Enter => {
                let text = std::mem::take(&mut app.inject_buffer);
                app.showing_inject = false;
                app.mode = AppMode::Streaming;
                Action::SendTwo(ControlSignal::Inject(text), ControlSignal::Resume)
            }
            KeyCode::Char(c) => {
                app.inject_buffer.push(c);
                Action::None
            }
            KeyCode::Backspace => {
                app.inject_buffer.pop();
                Action::None
            }
            _ => Action::None,
        }
    } else {
        match key.code {
            KeyCode::Char('q') => {
                app.shutdown = true;
                Action::Shutdown
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.shutdown = true;
                Action::Shutdown
            }
            KeyCode::Char('i') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.mode = AppMode::Paused;
                app.showing_inject = true;
                Action::Send(ControlSignal::Pause)
            }
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let idx = app.turn_index.saturating_sub(1);
                app.mode = AppMode::Rewinding;
                Action::Send(ControlSignal::Rewind(idx))
            }
            KeyCode::Char('p') => {
                if app.mode == AppMode::Paused {
                    app.mode = AppMode::Streaming;
                    Action::Send(ControlSignal::Resume)
                } else {
                    app.mode = AppMode::Paused;
                    Action::Send(ControlSignal::Pause)
                }
            }
            KeyCode::Tab => {
                app.cycle_focus();
                Action::None
            }
            KeyCode::Char('j') => {
                app.scroll_down();
                Action::None
            }
            KeyCode::Char('k') => {
                app.scroll_up();
                Action::None
            }
            KeyCode::Char('g') => {
                app.scroll_top();
                Action::None
            }
            KeyCode::Char('G') => {
                app.scroll_bottom();
                Action::None
            }
            _ => Action::None,
        }
    }
}
