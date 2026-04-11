use crate::types::events::{ControlSignal, StreamEvent};
use crate::ui::app::{App, AppMode};
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};

enum Action {
    None,
    Send(ControlSignal),
    SendTwo(ControlSignal, ControlSignal),
    Shutdown,
}

pub async fn run_event_loop(
    app: Arc<Mutex<App>>,
    ctrl_tx: mpsc::Sender<ControlSignal>,
    mut stream_rx: mpsc::Receiver<StreamEvent>,
) -> Result<()> {
    loop {
        // Single lock acquisition: drain all pending stream events + check shutdown
        {
            let mut a = app.lock().await;
            while let Ok(ev) = stream_rx.try_recv() {
                match ev {
                    StreamEvent::TokenReceived { agent_id, token } => a.push_token(&agent_id, &token),
                    StreamEvent::TurnComplete(turn) => a.commit_turn(&turn),
                    StreamEvent::ConvergenceUpdated { p, certainty } => a.set_convergence(p, certainty),
                    StreamEvent::ArtifactsUpdated(list) => a.artifacts = list,
                    StreamEvent::CheckpointWritten(idx) => {
                        a.push_event(format!("Checkpoint i_{idx}"));
                    }
                    StreamEvent::Error(msg) => a.push_event(format!("Error: {msg}")),
                }
            }
            if a.shutdown {
                return Ok(());
            }
        }

        let kb = tokio::task::spawn_blocking(|| -> Result<Option<Event>> {
            if event::poll(Duration::from_millis(16))? {
                Ok(Some(event::read()?))
            } else {
                Ok(None)
            }
        })
        .await??;

        let Some(ev) = kb else { continue };

        let Event::Key(key) = ev else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        // Single lock acquisition: process key + derive action
        let action = {
            let mut a = app.lock().await;

            if a.showing_inject {
                match key.code {
                    KeyCode::Esc => {
                        a.inject_buffer.clear();
                        a.showing_inject = false;
                        a.mode = AppMode::Streaming;
                        Action::Send(ControlSignal::Resume)
                    }
                    KeyCode::Enter => {
                        let text = std::mem::take(&mut a.inject_buffer);
                        a.showing_inject = false;
                        a.mode = AppMode::Streaming;
                        // Send the injected text, then immediately resume
                        Action::SendTwo(ControlSignal::Inject(text), ControlSignal::Resume)
                    }
                    KeyCode::Char(c) => {
                        a.inject_buffer.push(c);
                        Action::None
                    }
                    KeyCode::Backspace => {
                        a.inject_buffer.pop();
                        Action::None
                    }
                    _ => Action::None,
                }
            } else {
                match key.code {
                    KeyCode::Char('q') => {
                        a.shutdown = true;
                        Action::Shutdown
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        a.shutdown = true;
                        Action::Shutdown
                    }
                    KeyCode::Char('i') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        a.mode = AppMode::Paused;
                        a.showing_inject = true;
                        Action::Send(ControlSignal::Pause)
                    }
                    KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        let idx = a.turn_index.saturating_sub(1);
                        a.mode = AppMode::Rewinding;
                        Action::Send(ControlSignal::Rewind(idx))
                    }
                    KeyCode::Char('p') => {
                        if a.mode == AppMode::Paused {
                            a.mode = AppMode::Streaming;
                            Action::Send(ControlSignal::Resume)
                        } else {
                            a.mode = AppMode::Paused;
                            Action::Send(ControlSignal::Pause)
                        }
                    }
                    KeyCode::Tab => {
                        a.cycle_focus();
                        Action::None
                    }
                    KeyCode::Char('j') => {
                        a.scroll_down();
                        Action::None
                    }
                    KeyCode::Char('k') => {
                        a.scroll_up();
                        Action::None
                    }
                    KeyCode::Char('g') => {
                        a.scroll_top();
                        Action::None
                    }
                    KeyCode::Char('G') => {
                        a.scroll_bottom();
                        Action::None
                    }
                    _ => Action::None,
                }
            }
        };

        match action {
            Action::Shutdown => {
                ctrl_tx
                    .send(ControlSignal::Shutdown)
                    .await
                    .context("failed to send Shutdown control signal")?;
                return Ok(());
            }
            Action::Send(signal) => {
                ctrl_tx
                    .send(signal)
                    .await
                    .context("failed to send control signal")?;
            }
            Action::SendTwo(first, second) => {
                ctrl_tx
                    .send(first)
                    .await
                    .context("failed to send first control signal")?;
                ctrl_tx
                    .send(second)
                    .await
                    .context("failed to send second control signal")?;
            }
            Action::None => {}
        }
    }
}
