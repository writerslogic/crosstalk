use crate::types::conversation::ConversationState;
use crate::types::events::{ControlSignal, StreamEvent};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table, Wrap},
};
use std::io;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedPane {
    GhostStream,
    Artifacts,
    History,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UIMode {
    Normal,
    Insert,
    Rewind,
    Playback,
    Intercept,
}

/// CrosstalkUI is the **rendering layer** for the Crosstalk terminal interface.
///
/// Responsibilities of CrosstalkUI:
/// - Owns the `ratatui` terminal and drives the render/event loop.
/// - Holds transient, TUI-only input state (`ghost_stream_buffer`, `input_buffer`,
///   `mode`, `active_pane`, etc.) that has no meaning outside the render loop.
/// - Receives [`StreamEvent`]s from the orchestrator and forwards [`ControlSignal`]s back.
///
/// What CrosstalkUI is NOT:
/// - It is not a duplicate of [`crate::ui::app::App`].  `App` owns the structured
///   conversation model (`streaming_buffer`, `artifacts`, `entropy_scores`, agent
///   weights, FPS tracking, etc.) and is designed to be shared across multiple
///   rendering back-ends (TUI, headless, GodView).  `CrosstalkUI` reads from
///   [`ConversationState`] directly for its simpler, legacy rendering path.
///
/// If you need richer state (entropy heatmap, scroll offsets, FPS), wire in an
/// `App` instance instead of duplicating its fields here.
pub struct CrosstalkUI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    event_rx: mpsc::Receiver<StreamEvent>,
    control_tx: mpsc::Sender<ControlSignal>,
    /// Accumulates streamed tokens between turns; cleared on `TurnComplete`.
    /// TUI-only: not persisted in `ConversationState` or `App`.
    /// Stored as `Arc<String>` so render can clone the Arc (pointer copy) instead
    /// of copying the full buffer each frame.
    ghost_stream_buffer: Arc<String>,
    /// Current interaction mode (Normal / Insert / Playback / etc.).
    /// TUI-only: drives keybinding dispatch and status bar display.
    mode: UIMode,
    /// Characters typed by the user during `UIMode::Insert`.
    /// TUI-only: flushed as a `ControlSignal::Inject` on Enter.
    input_buffer: String,
    /// Which pane currently has visual focus for Tab cycling.
    /// TUI-only: affects border highlight color only.
    active_pane: FocusedPane,
    /// Iteration index shown during `UIMode::Playback`.
    /// TUI-only: mirrors `ConversationState::iteration_index` during scrubbing.
    playback_index: u32,
    /// Row selection cursor used in `UIMode::Intercept`.
    /// TUI-only: indexes into artifact/agent lists for Lock/Mute actions.
    selection_index: usize,
}

impl CrosstalkUI {
    pub fn new(
        event_rx: mpsc::Receiver<StreamEvent>,
        control_tx: mpsc::Sender<ControlSignal>,
    ) -> Result<Self, io::Error> {
        let stdout = io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            event_rx,
            control_tx,
            ghost_stream_buffer: Arc::new(String::new()),
            mode: UIMode::Normal,
            input_buffer: String::new(),
            active_pane: FocusedPane::GhostStream,
            playback_index: 0,
            selection_index: 0,
        })
    }

    /// The main UI loop
    pub async fn run(&mut self, mut sigma: ConversationState) -> Result<(), io::Error> {
        use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
        use std::time::Duration;

        loop {
            // 1. Process Events
            if self.mode == UIMode::Normal || self.mode == UIMode::Insert {
                while let Ok(event) = self.event_rx.try_recv() {
                    match event {
                        StreamEvent::TokenReceived { agent_id: _, token } => {
                            let safe: String = token
                                .chars()
                                .filter(|&c| c >= ' ' || c == '\n' || c == '\t')
                                .collect();
                            Arc::make_mut(&mut self.ghost_stream_buffer).push_str(&safe);
                        }
                        StreamEvent::TurnComplete(turn) => {
                            sigma.push_turn(turn);
                            sigma.iteration_index += 1;
                            Arc::make_mut(&mut self.ghost_stream_buffer).clear();
                        }
                        StreamEvent::Error(err) => {
                            let safe_err: String = err
                                .chars()
                                .filter(|&c| c >= ' ' || c == '\n' || c == '\t')
                                .collect();
                            Arc::make_mut(&mut self.ghost_stream_buffer)
                                .push_str(&format!("\n[ERROR] {}\n", safe_err));
                        }
                        _ => {}
                    }
                }
            }

            // 2. Render
            self.render(&sigma)?;

            // 3. Handle Input
            if event::poll(Duration::from_millis(16))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match self.mode {
                    UIMode::Normal => match key.code {
                        KeyCode::Char('q') => {
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Shutdown).await,
                                "failed to send shutdown signal"
                            );
                            break;
                        }
                        KeyCode::Char('i') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.mode = UIMode::Insert;
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Pause).await,
                                "failed to send pause signal"
                            );
                        }
                        KeyCode::Char('p') => {
                            self.mode = UIMode::Playback;
                            self.playback_index = sigma.iteration_index;
                        }
                        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.mode = UIMode::Intercept;
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Pause).await,
                                "failed to send pause signal"
                            );
                        }
                        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            crate::log_warn!(self.capture_to_svg(&sigma), "failed to capture SVG");
                        }
                        KeyCode::Tab => {
                            self.active_pane = match self.active_pane {
                                FocusedPane::GhostStream => FocusedPane::Artifacts,
                                FocusedPane::Artifacts => FocusedPane::History,
                                FocusedPane::History => FocusedPane::GhostStream,
                            };
                        }
                        _ => {}
                    },
                    UIMode::Playback => match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => {
                            self.mode = UIMode::Normal;
                        }
                        KeyCode::Left | KeyCode::Char(',') => {
                            if self.playback_index > 0 {
                                self.playback_index -= 1;
                                crate::log_warn!(
                                    self.control_tx
                                        .send(ControlSignal::Rewind(self.playback_index))
                                        .await,
                                    "failed to send rewind signal"
                                );
                            }
                        }
                        KeyCode::Right | KeyCode::Char('.') => {
                            if self.playback_index < sigma.iteration_index {
                                self.playback_index += 1;
                                crate::log_warn!(
                                    self.control_tx
                                        .send(ControlSignal::Rewind(self.playback_index))
                                        .await,
                                    "failed to send rewind signal"
                                );
                            }
                        }
                        _ => {}
                    },
                    UIMode::Insert => match key.code {
                        KeyCode::Enter => {
                            let raw = std::mem::take(&mut self.input_buffer);
                            // Strip ASCII control characters before the text enters
                            // the AI pipeline; keeps printable chars, newline, and tab.
                            let content: String = raw
                                .chars()
                                .filter(|&c| c >= ' ' || c == '\n' || c == '\t')
                                .collect();
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Inject(content)).await,
                                "failed to send inject signal"
                            );
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Resume).await,
                                "failed to send resume signal"
                            );
                            self.mode = UIMode::Normal;
                        }
                        KeyCode::Char(c) => {
                            self.input_buffer.push(c);
                        }
                        KeyCode::Backspace => {
                            self.input_buffer.pop();
                        }
                        KeyCode::Esc => {
                            self.mode = UIMode::Normal;
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Resume).await,
                                "failed to send resume signal"
                            );
                        }
                        _ => {}
                    },
                    UIMode::Intercept => match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            self.selection_index = self.selection_index.saturating_sub(1);
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            self.selection_index = self.selection_index.saturating_add(1);
                        }
                        KeyCode::Char('l') => {
                            let names: Vec<_> = sigma.artifacts.keys().cloned().collect();
                            if !names.is_empty() {
                                let target = &names[self.selection_index % names.len()];
                                crate::log_warn!(
                                    self.control_tx
                                        .send(ControlSignal::LockCode(target.clone()))
                                        .await,
                                    "failed to send lock signal"
                                );
                            }
                        }
                        KeyCode::Char('m') => {
                            if !sigma.agent_weights.is_empty() {
                                let idx = self.selection_index % sigma.agent_weights.len();
                                if let Some((target, _)) = sigma.agent_weights.iter().nth(idx) {
                                    crate::log_warn!(
                                        self.control_tx
                                            .send(ControlSignal::MuteAgent(target.clone()))
                                            .await,
                                        "failed to send mute signal"
                                    );
                                }
                            }
                        }
                        KeyCode::Char('d') => {
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::DampenSwarm(0.5)).await,
                                "failed to send dampen signal"
                            );
                        }
                        KeyCode::Esc => {
                            self.mode = UIMode::Normal;
                            crate::log_warn!(
                                self.control_tx.send(ControlSignal::Resume).await,
                                "failed to send resume signal"
                            );
                        }
                        _ => {}
                    },
                    UIMode::Rewind => {}
                }
            }
        }
        Ok(())
    }

    /// Renders the current state σ to the terminal.
    pub fn render(&mut self, sigma: &ConversationState) -> Result<(), io::Error> {
        let ghost_stream_content = Arc::clone(&self.ghost_stream_buffer);
        let mode = self.mode;
        let playback_index = self.playback_index;
        let active_pane = self.active_pane;
        let input_buffer = self.input_buffer.clone();

        self.terminal.draw(|f| {
            let area = f.area();

            // Root Layout
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Top Bar
                    Constraint::Min(10),   // Center Dashboard
                    Constraint::Length(3), // Neural Intercept Area
                    Constraint::Length(1), // Status Bar
                ])
                .split(area);

            // 1. Top Bar
            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50),
                    Constraint::Percentage(50),
                ])
                .split(chunks[0]);

            let sigma_state = Paragraph::new(format!(
                "Ω CROSSTALK | Session: {} | i_{} {}",
                sigma.session_id,
                if mode == UIMode::Playback { playback_index } else { sigma.iteration_index },
                if mode == UIMode::Playback { "[PLAYBACK]" } else { "" }
            ))
            .block(Block::default().borders(Borders::ALL).title(" σ State "));
            f.render_widget(sigma_state, top_chunks[0]);

            let mu_indicators = Paragraph::new(" μ Agents: [Gemini-1.5] [GPT-4o] ")
                .block(Block::default().borders(Borders::ALL).title(" μ Agents "));
            f.render_widget(mu_indicators, top_chunks[1]);

            // 2. Center Dashboard
            let dashboard_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(70),
                    Constraint::Percentage(30),
                ])
                .split(chunks[1]);

            let ghost_block = Block::default()
                .borders(Borders::ALL)
                .title(" Ghost Stream / Code View ")
                .border_style(if active_pane == FocusedPane::GhostStream {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default()
                });

            let display_text = if mode == UIMode::Playback {
                Text::from(format!("[Playback of iteration i_{}]", playback_index))
            } else {
                render_heatmap_content(ghost_stream_content.as_str())
            };

            let ghost = Paragraph::new(display_text)
                .block(ghost_block)
                .wrap(Wrap { trim: false });
            f.render_widget(ghost, dashboard_chunks[0]);

            let right_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(40), // Interactive AST
                    Constraint::Percentage(30), // Δα Diffs
                    Constraint::Percentage(30), // Entropy Map
                ])
                .split(dashboard_chunks[1]);

            // AST side panel
            let mut ast_items = vec![];
            for artifact in sigma.artifacts.values() {
                ast_items.push(ListItem::new(Span::styled(format!("📁 {}", artifact.name), Style::default().add_modifier(Modifier::BOLD))));
                for line in artifact.skeleton.lines() {
                    ast_items.push(ListItem::new(format!("  {}", line)));
                }
            }
            let ast_list = List::new(ast_items)
                .block(Block::default().borders(Borders::ALL).title(" AST Navigator ")
                .border_style(if active_pane == FocusedPane::Artifacts { Style::default().fg(Color::Yellow) } else { Style::default() }));
            f.render_widget(ast_list, right_chunks[0]);

            // Δα Panel
            let diffs: Vec<ListItem> = sigma.turns.iter().rev().take(5).map(|t| {
                ListItem::new(format!("i_{}: {} (Δx{})", t.index, t.model_id, t.diffs.len()))
            }).collect();
            let diffs_list = List::new(diffs)
                .block(Block::default().borders(Borders::ALL).title(" Δα Diffs "));
            f.render_widget(diffs_list, right_chunks[1]);

            // Entropy Map Panel
            let mut friction_rows = vec![];
            for (name, artifact) in &sigma.artifacts {
                let friction = if artifact.history.len() > 1 { 0.5 } else { 0.0 };
                let color = if friction > 0.7 { Color::Red } else if friction > 0.3 { Color::Yellow } else { Color::Green };
                friction_rows.push(Row::new(vec![
                    name.clone(),
                    format!("v{}", artifact.version),
                    format!("{:.2}", friction),
                ]).style(Style::default().fg(color)));
            }
            let entropy_table = Table::new(
                friction_rows,
                [Constraint::Percentage(50), Constraint::Percentage(25), Constraint::Percentage(25)],
            )
            .header(Row::new(vec!["Artifact", "Ver", "Fric"]).style(Style::default().add_modifier(Modifier::BOLD)))
            .block(Block::default().title(" Entropy Map ").borders(Borders::ALL));
            f.render_widget(entropy_table, right_chunks[2]);

            // 3. Neural Intercept Area
            let input_text = match mode {
                UIMode::Normal => " [NORMAL] Wait for turn or Ctrl+I to intercept... ".to_string(),
                UIMode::Insert => format!(" [INSERT] > {}", input_buffer),
                UIMode::Rewind => " [REWIND] Select checkpoint index... ".to_string(),
                UIMode::Playback => format!(" [PLAYBACK] i_{} | Use < / > to seek, Esc to exit", playback_index),
                UIMode::Intercept => " [STEER] [L] Lock Code | [M] Mute Agent | [D] Dampen Swarm ".to_string(),
            };
            let input_para = Paragraph::new(input_text)
                .block(Block::default().title(" Neural Intercept ").borders(Borders::ALL).border_style(
                    if matches!(mode, UIMode::Insert | UIMode::Intercept) { Style::default().fg(Color::Cyan) } else { Style::default() }
                ));
            f.render_widget(input_para, chunks[2]);

            // 4. Status Bar
            let status = Paragraph::new(" [Tab] Cycle Panes | [Ctrl+I] Inject | [p] Playback | [Ctrl+S] Save SVG | [q] Quit ");
            f.render_widget(status, chunks[3]);
        })?;
        Ok(())
    }

    pub fn capture_to_svg(&self, sigma: &ConversationState) -> io::Result<()> {
        use std::fs;
        use std::fs::File;
        use std::io::Write;

        let safe_id: String = sigma
            .session_id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .collect();
        let capture_dir = std::env::temp_dir().join("crosstalk-captures");
        fs::create_dir_all(&capture_dir)?;
        let filename = format!("capture_{}_{}.svg", safe_id, sigma.iteration_index);
        let mut file = File::create(capture_dir.join(filename))?;

        writeln!(
            file,
            r##"<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">"##
        )?;
        writeln!(
            file,
            r##"  <rect width="100%" height="100%" fill="#0a0a0f" />"##
        )?;
        writeln!(
            file,
            r##"  <text x="20" y="40" font-family="monospace" font-size="16" fill="#00ff88">Ω CROSSTALK SNAPSHOT</text>"##
        )?;
        writeln!(
            file,
            r##"  <text x="20" y="70" font-family="monospace" font-size="12" fill="#ffffff">Session: {}</text>"##,
            sigma.session_id
        )?;
        writeln!(
            file,
            r##"  <text x="20" y="90" font-family="monospace" font-size="12" fill="#ffffff">Iteration: i_{}</text>"##,
            sigma.iteration_index
        )?;

        writeln!(
            file,
            r##"  <rect x="20" y="120" width="540" height="300" stroke="#ffffff" fill="none" />"##
        )?;
        writeln!(
            file,
            r##"  <text x="30" y="140" font-family="monospace" font-size="10" fill="#ffffff">Ghost Stream Content...</text>"##
        )?;

        writeln!(
            file,
            r##"  <rect x="580" y="120" width="200" height="140" stroke="#ffffff" fill="none" />"##
        )?;
        writeln!(
            file,
            r##"  <text x="590" y="140" font-family="monospace" font-size="10" fill="#ffffff">Δα Diffs</text>"##
        )?;

        writeln!(file, "</svg>")?;
        Ok(())
    }
}

fn render_heatmap_content<'a>(content: &'a str) -> Text<'a> {
    let mut lines = vec![];
    for line in content.lines() {
        let mut spans = vec![];
        for word in line.split_whitespace() {
            let friction = (word.len() as f32 % 10.0) / 10.0;
            let color = if friction > 0.7 {
                Color::Rgb(100, 0, 0)
            } else if friction > 0.4 {
                Color::Rgb(60, 60, 0)
            } else {
                Color::Reset
            };
            spans.push(Span::styled(
                word.to_string() + " ",
                Style::default().bg(color),
            ));
        }
        lines.push(Line::from(spans));
    }
    Text::from(lines)
}
