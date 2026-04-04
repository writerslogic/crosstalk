use crate::types::{ControlSignal, ConversationState, StreamEvent};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table, Wrap},
    Terminal,
};
use std::io;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedPane {
    GhostStream,
    Artifacts,
    History,
}

pub enum UIMode {
    Normal,
    Insert,
    Rewind,
}

pub struct CrosstalkUI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    event_rx: mpsc::Receiver<StreamEvent>,
    control_tx: mpsc::Sender<ControlSignal>,
    ghost_stream_buffer: String,
    mode: UIMode,
    input_buffer: String,
    active_pane: FocusedPane,
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
        })
    }

    /// The main UI loop
    pub async fn run(&mut self, mut sigma: ConversationState) -> Result<(), io::Error> {
        use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
        use std::time::Duration;

        loop {
            // 1. Process Events
            while let Ok(event) = self.event_rx.try_recv() {
                match event {
                    StreamEvent::TokenReceived(token) => {
                        self.ghost_stream_buffer.push_str(&token);
                    }
                    StreamEvent::TurnComplete(turn) => {
                        sigma.turns.push(turn);
                        sigma.iteration_index += 1;
                        self.ghost_stream_buffer.clear();
                    }
                    StreamEvent::Error(err) => {
                        self.ghost_stream_buffer.push_str(&format!("\n[ERROR] {}\n", err));
                    }
                    _ => {}
                }
            }

            // 2. Render
            self.render(&sigma)?;

            // 3. Handle Input
            if event::poll(Duration::from_millis(16))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
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
                                KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    self.mode = UIMode::Rewind;
                                    let _ = self.control_tx.send(ControlSignal::Pause).await;
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
                            UIMode::Rewind => match key.code {
                                KeyCode::Esc => {
                                    self.mode = UIMode::Normal;
                                    let _ = self.control_tx.send(ControlSignal::Resume).await;
                                }
                                _ => {
                                    // Rewind selection logic placeholder
                                }
                            },
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Renders the current state σ to the terminal.
    pub fn render(&mut self, sigma: &ConversationState) -> Result<(), io::Error> {
        self.terminal.draw(|f| {
            let area = f.area();

            // Root Layout
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Top Bar (σ State)
                    Constraint::Min(10),   // Center Dashboard
                    Constraint::Length(3), // Neural Intercept Area
                    Constraint::Length(1), // Status Bar
                ])
                .split(area);

            // 1. Top Bar (σ State)
            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50), // Session info
                    Constraint::Percentage(50), // μ Indicators
                ])
                .split(chunks[0]);

            let sigma_state = Paragraph::new(format!(
                "Ω CROSSTALK | Session: {} | i_{}",
                sigma.session_id, sigma.iteration_index
            ))
            .block(Block::default().borders(Borders::ALL).title(" σ State "));
            f.render_widget(sigma_state, top_chunks[0]);

            let agents_list = " μ Agents: [Gemini-1.5] [GPT-4o] "; // Dynamic later
            let mu_indicators = Paragraph::new(agents_list)
                .block(Block::default().borders(Borders::ALL).title(" μ Agents "));
            f.render_widget(mu_indicators, top_chunks[1]);

            // 2. Center Dashboard
            let dashboard_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(70), // Left: Ghost Stream
                    Constraint::Percentage(30), // Right: Metadata/Diffs
                ])
                .split(chunks[1]);

            let ghost_block = Block::default()
                .borders(Borders::ALL)
                .title(" Ghost Stream ")
                .border_style(if self.active_pane == FocusedPane::GhostStream {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default()
                });
            let ghost = Paragraph::new(self.ghost_stream_buffer.as_str())
                .block(ghost_block)
                .wrap(Wrap { trim: false });
            f.render_widget(ghost, dashboard_chunks[0]);

            let right_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(50), // Δα Diffs
                    Constraint::Percentage(50), // Entropy Map
                ])
                .split(dashboard_chunks[1]);

            // Δα Panel
            let diffs: Vec<ListItem> = sigma.turns.iter().rev().take(5).map(|t| {
                ListItem::new(format!("i_{}: {} (Δx{})", t.index, t.model_id, t.diffs.len()))
            }).collect();
            let diffs_list = List::new(diffs)
                .block(Block::default().borders(Borders::ALL).title(" Δα Diffs "));
            f.render_widget(diffs_list, right_chunks[0]);

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
            f.render_widget(entropy_table, right_chunks[1]);

            // 3. Neural Intercept Area
            let input_text = match self.mode {
                UIMode::Normal => " [NORMAL] Wait for turn or Ctrl+I to intercept... ".to_string(),
                UIMode::Insert => format!(" [INSERT] > {}", self.input_buffer),
                UIMode::Rewind => " [REWIND] Select checkpoint index... ".to_string(),
            };
            let input_para = Paragraph::new(input_text)
                .block(Block::default().title(" Neural Intercept ").borders(Borders::ALL).border_style(
                    if matches!(self.mode, UIMode::Insert) { Style::default().fg(Color::Cyan) } else { Style::default() }
                ));
            f.render_widget(input_para, chunks[2]);

            // 4. Status Bar
            let status = Paragraph::new(" [Tab] Cycle Panes | [Ctrl+I] Inject | [Ctrl+R] Rewind | [q] Quit ");
            f.render_widget(status, chunks[3]);
        })?;
        Ok(())
    }
}
