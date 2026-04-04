use crate::types::ConversationState;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::io;

pub struct CrosstalkUI {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl CrosstalkUI {
    pub fn new() -> Result<Self, io::Error> {
        let stdout = io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    /// Renders the current state σ to the terminal.
    pub fn render(&mut self, sigma: &ConversationState) -> Result<(), io::Error> {
        self.terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Header (Session ID)
                    Constraint::Min(10),   // Turn Log (σ.turns)
                    Constraint::Length(5), // Artifact Status (α)
                    Constraint::Length(3), // Status Bar (Iteration i)
                ])
                .split(f.area());

            // 1. Header
            let header = Paragraph::new(format!("Ω CROSSTALK | Session: {}", sigma.session_id))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(header, chunks[0]);

            // 2. Turn Log
            let turns: Vec<ListItem> = sigma.turns.iter()
                .map(|t| ListItem::new(format!("[{}] {}: {}", t.index, t.model_id, t.content)))
                .collect();
            let log = List::new(turns)
                .block(Block::default().title("Dialogue Stream (σ)").borders(Borders::ALL));
            f.render_widget(log, chunks[1]);

            // 3. Artifact Status
            let arts = sigma.artifacts.keys().cloned().collect::<Vec<_>>().join(", ");
            let artifact_view = Paragraph::new(format!("Active Artifacts (α): {}", arts))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(artifact_view, chunks[2]);

            // 4. Status Bar
            let status = Paragraph::new(format!("Iteration: i_{} | Controls: [Space] Pause | [R] Rewind", sigma.iteration_index))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(status, chunks[3]);
        })?;
        Ok(())
    }
}