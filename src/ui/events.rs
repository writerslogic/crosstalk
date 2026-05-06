use crate::types::events::{ControlSignal, StreamEvent};
use crate::ui::app::{App, AppMode};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;
use tokio::sync::mpsc;

pub enum Action {
    None,
    Send(ControlSignal),
    SendTwo(ControlSignal, ControlSignal),
    Shutdown,
}

fn log_event(ev: &StreamEvent) {
    let ts = chrono::Local::now().format("%H:%M:%S%.3f");

    if let StreamEvent::TokenReceived { agent_id, token } = ev {
        thread_local! {
            static BUF: std::cell::RefCell<HashMap<String, String>> =
                std::cell::RefCell::new(HashMap::new());
        }
        let lines: Vec<String> = BUF.with(|b| {
            let mut map = b.borrow_mut();
            let entry = map.entry(agent_id.clone()).or_default();
            entry.push_str(token);
            let mut out = Vec::new();
            while let Some(pos) = entry.find('\n') {
                let line = entry[..pos].trim().to_string();
                if !line.is_empty() {
                    out.push(format!("{ts} [{agent_id}] {line}"));
                }
                *entry = entry[pos + 1..].to_string();
            }
            out
        });
        if !lines.is_empty()
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/crosstalk.log") {
                for line in lines {
                    crate::log_warn!(writeln!(f, "{line}"), "failed to write log event");
                }
            }
        return;
    }

    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/crosstalk.log") {
        match ev {
            StreamEvent::TokenReceived { .. } => unreachable!(),
            StreamEvent::TurnComplete(turn) => {
                crate::log_warn!(writeln!(
                    f, "{ts} [TURN] i_{} by {} outcome={:?} cert={:.2} diffs={}",
                    turn.index, turn.model_id, turn.outcome,
                    turn.certainty.unwrap_or(0.0), turn.diffs.len()
                ), "failed to write log event");
            }
            StreamEvent::ConvergenceUpdated { p, certainty, agent_weights } => {
                let weights_str: String = agent_weights.iter()
                    .map(|(a, w)| format!("{a}={w:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                crate::log_warn!(writeln!(f, "{ts} [CONV] p={p:.3} cert={certainty:.3} weights=[{weights_str}]"), "failed to write log event");
            }
            StreamEvent::ArtifactsUpdated(list) => {
                crate::log_warn!(writeln!(f, "{ts} [ARTIFACTS] {} artifact(s) updated", list.len()), "failed to write log event");
            }
            StreamEvent::EntropyUpdated(entries) => {
                crate::log_warn!(writeln!(f, "{ts} [ENTROPY] {} entries", entries.len()), "failed to write log event");
            }
            StreamEvent::GodViewUpdated { frame, avg_certainty, avg_surprise, agent_count } => {
                crate::log_warn!(writeln!(
                    f, "{ts} [GODVIEW] frame={frame} cert={avg_certainty:.2} surprise={avg_surprise:.2} agents={agent_count}"
                ), "failed to write log event");
            }
            StreamEvent::CheckpointWritten(idx) => {
                crate::log_warn!(writeln!(f, "{ts} [CKPT] i_{idx}"), "failed to write log event");
            }
            StreamEvent::Error(msg) => {
                crate::log_warn!(writeln!(f, "{ts} [ERROR] {msg}"), "failed to write log event");
            }
        }
    }
}

/// Drain all pending stream events into the App.
pub fn drain_stream_events(app: &mut App, stream_rx: &mut mpsc::Receiver<StreamEvent>) {
    while let Ok(ev) = stream_rx.try_recv() {
        log_event(&ev);
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
                let trend = if p > 0.8 {
                    "nearly converged"
                } else if p > 0.5 {
                    "making progress"
                } else if p > 0.2 {
                    "exploring"
                } else {
                    "early stage"
                };
                app.push_event(format!(
                    "Convergence {:.0}%, certainty {:.0}% — {}",
                    p * 100.0, certainty * 100.0, trend
                ));
            }
            StreamEvent::ArtifactsUpdated(list) => {
                let changed: Vec<&str> = list.iter()
                    .filter(|a| a.diff_count > 0)
                    .map(|a| a.name.as_str())
                    .collect();
                if !changed.is_empty() {
                    app.push_event(format!(
                        "[artifacts] {} modified: {}",
                        changed.len(),
                        changed.join(", ")
                    ));
                }
                app.artifacts = list;
            }
            StreamEvent::EntropyUpdated(entries) => {
                app.entropy_scores = entries.into_iter().map(|e| crate::ui::app::EntropyRow {
                    artifact: e.artifact_name,
                    agents: e.scores,
                }).collect();
            }
            StreamEvent::GodViewUpdated { frame, avg_certainty, avg_surprise, .. } => {
                app.godview_frame = frame;
                app.godview_certainty = avg_certainty;
                app.godview_surprise = avg_surprise;
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
    if event::poll(timeout).ok()?
        && let Event::Key(key) = event::read().ok()?
        && key.kind == KeyEventKind::Press
    {
        return Some(key);
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
