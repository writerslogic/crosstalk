use anyhow::Result;
use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use std::{
    env,
    io,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;

// ── Provider definitions ──────────────────────────────────────────────────────

struct ProviderDef {
    name: &'static str,
    env_key: &'static str,
    list_url: &'static str,
    auth_style: AuthStyle,
    /// JSON pointer into response: "data" → array of objects with "id", "models" → objects with "name"
    array_key: &'static str,
    id_field: &'static str,
    /// Only include IDs matching this prefix (empty = include all)
    prefix_filter: &'static [&'static str],
}

enum AuthStyle {
    Bearer,
    XApiKey,
    QueryKey,  // appended as ?key=...
}

const PROVIDERS: &[ProviderDef] = &[
    ProviderDef {
        name: "OpenRouter",
        env_key: "OPENROUTER_API_KEY",
        list_url: "https://openrouter.ai/api/v1/models",
        auth_style: AuthStyle::Bearer,
        array_key: "data",
        id_field: "id",
        prefix_filter: &[],
    },
    ProviderDef {
        name: "OpenAI",
        env_key: "OPENAI_API_KEY",
        list_url: "https://api.openai.com/v1/models",
        auth_style: AuthStyle::Bearer,
        array_key: "data",
        id_field: "id",
        prefix_filter: &["gpt-", "o1", "o3", "o4", "chatgpt"],
    },
    ProviderDef {
        name: "Anthropic",
        env_key: "ANTHROPIC_API_KEY",
        list_url: "https://api.anthropic.com/v1/models",
        auth_style: AuthStyle::XApiKey,
        array_key: "data",
        id_field: "id",
        prefix_filter: &["claude"],
    },
    ProviderDef {
        name: "Gemini",
        env_key: "GEMINI_API_KEY",
        list_url: "https://generativelanguage.googleapis.com/v1beta/models",
        auth_style: AuthStyle::QueryKey,
        array_key: "models",
        id_field: "name",
        prefix_filter: &["models/gemini"],
    },
    ProviderDef {
        name: "DeepSeek",
        env_key: "DEEPSEEK_API_KEY",
        list_url: "https://api.deepseek.com/v1/models",
        auth_style: AuthStyle::Bearer,
        array_key: "data",
        id_field: "id",
        prefix_filter: &["deepseek"],
    },
    ProviderDef {
        name: "Mistral",
        env_key: "MISTRAL_API_KEY",
        list_url: "https://api.mistral.ai/v1/models",
        auth_style: AuthStyle::Bearer,
        array_key: "data",
        id_field: "id",
        prefix_filter: &[],
    },
    ProviderDef {
        name: "Groq",
        env_key: "GROQ_API_KEY",
        list_url: "https://api.groq.com/openai/v1/models",
        auth_style: AuthStyle::Bearer,
        array_key: "data",
        id_field: "id",
        prefix_filter: &[],
    },
];

// ── UI state ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Item {
    Header { provider: String, loading: bool },
    Model { id: String, selected: bool },
}

struct SelectState {
    items: Vec<Item>,
    cursor: usize,
    scroll: usize,
    done: bool,
    confirm: bool,
}

impl SelectState {
    fn new() -> Self {
        let mut items = Vec::new();
        for def in PROVIDERS {
            if env::var(def.env_key).is_ok() {
                items.push(Item::Header {
                    provider: def.name.to_string(),
                    loading: true,
                });
            }
        }
        // Move cursor to first model entry (skip headers)
        let cursor = items.iter().position(|i| matches!(i, Item::Model { .. })).unwrap_or(0);
        Self { items, cursor, scroll: 0, done: false, confirm: false }
    }

    fn add_models(&mut self, provider: &str, ids: Vec<String>) {
        // Find the header index for this provider
        let header_pos = self.items.iter().position(|i| {
            matches!(i, Item::Header { provider: p, .. } if p == provider)
        });
        let Some(pos) = header_pos else { return };

        // Mark header as no longer loading
        if let Item::Header { loading, .. } = &mut self.items[pos] {
            *loading = false;
        }

        if ids.is_empty() {
            return;
        }

        // Find where to insert: after this header, before the next header
        let insert_at = pos + 1;
        let new_items: Vec<Item> = ids
            .into_iter()
            .map(|id| Item::Model { id, selected: false })
            .collect();

        // Insert new items; bump cursor if it's past the insert point
        let count = new_items.len();
        for (i, item) in new_items.into_iter().enumerate() {
            self.items.insert(insert_at + i, item);
        }
        if self.cursor > pos {
            self.cursor += count;
        }

        // Move cursor to first model if still on a header
        if matches!(self.items.get(self.cursor), Some(Item::Header { .. })) {
            self.advance_to_model(1);
        }
    }

    fn advance_to_model(&mut self, direction: isize) {
        let len = self.items.len();
        if len == 0 { return; }
        let mut i = self.cursor as isize;
        loop {
            i += direction;
            if i < 0 { i = len as isize - 1; }
            if i >= len as isize { i = 0; }
            if i == self.cursor as isize { break; }
            if matches!(self.items[i as usize], Item::Model { .. }) {
                self.cursor = i as usize;
                break;
            }
        }
    }

    fn move_up(&mut self) {
        if self.cursor == 0 { return; }
        self.cursor -= 1;
        if matches!(self.items.get(self.cursor), Some(Item::Header { .. })) {
            if self.cursor > 0 { self.cursor -= 1; }
        }
        self.clamp_scroll();
    }

    fn move_down(&mut self) {
        if self.cursor + 1 >= self.items.len() { return; }
        self.cursor += 1;
        if matches!(self.items.get(self.cursor), Some(Item::Header { .. })) {
            if self.cursor + 1 < self.items.len() { self.cursor += 1; }
        }
        self.clamp_scroll();
    }

    fn toggle(&mut self) {
        if let Some(Item::Model { selected, .. }) = self.items.get_mut(self.cursor) {
            *selected = !*selected;
        }
    }

    fn selected_ids(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|i| {
                if let Item::Model { id, selected: true, .. } = i {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn clamp_scroll(&mut self) {
        // Keep cursor in a visible window (assume visible_height ~= 20)
        let visible = 20usize;
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if self.cursor >= self.scroll + visible {
            self.scroll = self.cursor.saturating_sub(visible - 1);
        }
    }

    fn any_loading(&self) -> bool {
        self.items.iter().any(|i| matches!(i, Item::Header { loading: true, .. }))
    }
}

// ── Render ────────────────────────────────────────────────────────────────────

fn draw_selector(frame: &mut Frame, state: &SelectState) {
    let area = frame.area();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(area);

    let status = if state.any_loading() {
        " Querying providers...  [↑↓] Navigate  [Space] Toggle  [Enter] Start  [q] Quit"
    } else {
        " [↑↓] Navigate  [Space] Toggle  [Enter] Start  [q] Quit"
    };

    let header = Paragraph::new(status)
        .block(Block::default().borders(Borders::ALL).title(" Crosstalk — Select Models "));
    frame.render_widget(header, layout[0]);

    let visible_items: Vec<ListItem> = state
        .items
        .iter()
        .enumerate()
        .skip(state.scroll)
        .take(area.height.saturating_sub(10) as usize)
        .map(|(i, item)| {
            let is_cursor = i == state.cursor;
            match item {
                Item::Header { provider, loading } => {
                    let label = if *loading {
                        format!(" {provider}  (loading...)")
                    } else {
                        format!(" {provider}")
                    };
                    ListItem::new(label)
                        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                }
                Item::Model { id, selected, .. } => {
                    let check = if *selected { "[✓]" } else { "[ ]" };
                    let label = format!("   {check} {id}");
                    let style = if is_cursor {
                        Style::default().fg(Color::Black).bg(Color::White)
                    } else if *selected {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default()
                    };
                    ListItem::new(label).style(style)
                }
            }
        })
        .collect();

    let mut list_state = ListState::default();
    let visible_cursor = state.cursor.saturating_sub(state.scroll);
    list_state.select(Some(visible_cursor));

    let list = List::new(visible_items)
        .block(Block::default().borders(Borders::ALL).title(" Models "));
    frame.render_stateful_widget(list, layout[1], &mut list_state);

    let selected = state.selected_ids();
    let footer_text = if selected.is_empty() {
        " No models selected".to_string()
    } else {
        format!(" {} selected: {}", selected.len(), selected.join(", "))
    };
    let footer = Paragraph::new(footer_text)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title(" Selection "));
    frame.render_widget(footer, layout[2]);
}

// ── Main entry point ──────────────────────────────────────────────────────────

pub async fn run_model_selector() -> Result<Vec<String>> {
    let state = Arc::new(Mutex::new(SelectState::new()));

    // Spawn fetch tasks for each configured provider
    let (tx, mut rx) = mpsc::channel::<(String, Vec<String>)>(32);
    for def in PROVIDERS {
        if let Ok(api_key) = env::var(def.env_key) {
            let tx = tx.clone();
            let provider_name = def.name.to_string();
            let list_url = def.list_url;
            let id_field = def.id_field;
            let array_key = def.array_key;
            let prefix_filter: Vec<&'static str> = def.prefix_filter.to_vec();
            let auth_style_val = match def.auth_style {
                AuthStyle::Bearer => 0u8,
                AuthStyle::XApiKey => 1u8,
                AuthStyle::QueryKey => 2u8,
            };
            tokio::spawn(async move {
                // inline fetch to avoid borrowing def across await
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new());

                let url = if auth_style_val == 2 {
                    format!("{}?key={}", list_url, api_key)
                } else {
                    list_url.to_string()
                };

                let mut req = client.get(&url);
                if auth_style_val == 0 {
                    req = req.header("Authorization", format!("Bearer {}", api_key));
                } else if auth_style_val == 1 {
                    req = req
                        .header("x-api-key", &api_key)
                        .header("anthropic-version", "2023-06-01");
                }

                let ids = async {
                    let resp = req.send().await.ok()?.error_for_status().ok()?;
                    let body: serde_json::Value = resp.json().await.ok()?;
                    let array = body.get(array_key)?.as_array()?;
                    let mut ids: Vec<String> = array
                        .iter()
                        .filter_map(|item| item.get(id_field)?.as_str().map(|s| s.to_string()))
                        .filter(|id| {
                            if prefix_filter.is_empty() {
                                true
                            } else {
                                prefix_filter.iter().any(|p| id.starts_with(p))
                            }
                        })
                        .map(|id| id.strip_prefix("models/").unwrap_or(&id).to_string())
                        .collect();
                    ids.sort();
                    ids.dedup();
                    Some(ids)
                }
                .await
                .unwrap_or_default();

                let _ = tx.send((provider_name, ids)).await;
            });
        }
    }
    drop(tx);

    // Set up TUI
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let result = loop {
        // Drain any completed provider fetches
        while let Ok((provider, ids)) = rx.try_recv() {
            state.lock().unwrap().add_models(&provider, ids);
        }

        {
            let s = state.lock().unwrap();
            terminal.draw(|f| draw_selector(f, &s))?;
            if s.done {
                break if s.confirm { s.selected_ids() } else { vec![] };
            }
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                let mut s = state.lock().unwrap();
                match key.code {
                    KeyCode::Up | KeyCode::Char('k') => s.move_up(),
                    KeyCode::Down | KeyCode::Char('j') => s.move_down(),
                    KeyCode::Char(' ') => s.toggle(),
                    KeyCode::Enter => {
                        s.confirm = true;
                        s.done = true;
                    }
                    KeyCode::Char('q') | KeyCode::Esc => {
                        s.done = true;
                    }
                    _ => {}
                }
            }
        }
    };

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(result)
}
