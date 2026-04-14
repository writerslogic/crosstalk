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
}

pub struct CrosstalkUI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    event_rx: mpsc::Receiver<StreamEvent>,
    control_tx: mpsc::Sender<ControlSignal>,
    ghost_stream_buffer: String,
    mode: UIMode,
    input_buffer: String,
    active_pane: FocusedPane,
    playback_index: u32,
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
            ghost_stream_buffer: String::new(),
            mode: UIMode::Normal,
            input_buffer: String::new(),
            active_pane: FocusedPane::GhostStream,
            playback_index: 0,
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
                            self.ghost_stream_buffer.push_str(&token);
                        }
                        StreamEvent::TurnComplete(turn) => {
                            sigma.turns.push(turn);
                            sigma.iteration_index += 1;
                            self.ghost_stream_buffer.clear();
                        }
                        StreamEvent::Error(err) => {
                            self.ghost_stream_buffer
                                .push_str(&format!("\n[ERROR] {}\n", err));
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
                            let _ = self.control_tx.send(ControlSignal::Shutdown).await;
                            break;
                        }
                        KeyCode::Char('i') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.mode = UIMode::Insert;
                            let _ = self.control_tx.send(ControlSignal::Pause).await;
                        }
                        KeyCode::Char('p') => {
                            self.mode = UIMode::Playback;
                            self.playback_index = sigma.iteration_index;
                        }
                        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            let _ = self.capture_to_svg(&sigma);
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
                                let _ = self
                                    .control_tx
                                    .send(ControlSignal::Rewind(self.playback_index))
                                    .await;
                            }
                        }
                        KeyCode::Right | KeyCode::Char('.') => {
                            if self.playback_index < sigma.iteration_index {
                                self.playback_index += 1;
                                let _ = self
                                    .control_tx
                                    .send(ControlSignal::Rewind(self.playback_index))
                                    .await;
                            }
                        }
                        _ => {}
                    },
                    UIMode::Insert => match key.code {
                        KeyCode::Enter => {
                            let content = std::mem::take(&mut self.input_buffer);
                            let _ = self.control_tx.send(ControlSignal::Inject(content)).await;
                            let _ = self.control_tx.send(ControlSignal::Resume).await;
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
                            let _ = self.control_tx.send(ControlSignal::Resume).await;
                        }
                        _ => {}
                    },
                    UIMode::Rewind => {
                        if key.code == KeyCode::Esc {
                            self.mode = UIMode::Normal;
                            let _ = self.control_tx.send(ControlSignal::Resume).await;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Renders the current state σ to the terminal.
    pub fn render(&mut self, sigma: &ConversationState) -> Result<(), io::Error> {
        let ghost_stream_content = self.ghost_stream_buffer.clone();
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
                render_heatmap_content(&ghost_stream_content)
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
            };
            let input_para = Paragraph::new(input_text)
                .block(Block::default().title(" Neural Intercept ").borders(Borders::ALL).border_style(
                    if matches!(mode, UIMode::Insert) { Style::default().fg(Color::Cyan) } else { Style::default() }
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
