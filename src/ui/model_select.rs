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
    env, io,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;

// ── Provider definitions ──────────────────────────────────────────────────────

// ── Auto model selection ──────────────────────────────────────────────────────

struct ProviderFlagship {
    env_key: &'static str,
    code_model: &'static str,
    reasoning_model: &'static str,
    general_model: &'static str,
}

/// Static fallback model IDs used when dynamic discovery is unavailable.
/// These are periodically updated but may become stale between releases.
/// The dynamic discovery path (fetch_openrouter_top_models) is preferred.
const FLAGSHIP_MODELS: &[ProviderFlagship] = &[
    ProviderFlagship {
        env_key: "ANTHROPIC_API_KEY",
        code_model: "claude-sonnet-4-6",
        reasoning_model: "claude-opus-4-6",
        general_model: "claude-sonnet-4-6",
    },
    ProviderFlagship {
        env_key: "OPENAI_API_KEY",
        code_model: "gpt-4o",
        reasoning_model: "o3-mini",
        general_model: "gpt-4o",
    },
    ProviderFlagship {
        env_key: "DEEPSEEK_API_KEY",
        code_model: "deepseek-chat",
        reasoning_model: "deepseek-reasoner",
        general_model: "deepseek-chat",
    },
    ProviderFlagship {
        env_key: "MISTRAL_API_KEY",
        code_model: "codestral-latest",
        reasoning_model: "codestral-latest",
        general_model: "codestral-latest",
    },
    ProviderFlagship {
        env_key: "GROQ_API_KEY",
        code_model: "llama-3.3-70b-versatile",
        reasoning_model: "llama-3.3-70b-versatile",
        general_model: "llama-3.3-70b-versatile",
    },
];

/// Fetch top models from OpenRouter's /api/v1/models endpoint, sorted by
/// context length descending (proxy for recency/capability). Returns up to
/// `limit` model IDs from distinct providers, excluding known-weak models.
pub async fn fetch_openrouter_top_models(limit: usize) -> Vec<String> {
    let api_key = match env::var("OPENROUTER_API_KEY") {
        Ok(k) => k,
        Err(_) => return Vec::new(),
    };
    let client = reqwest::Client::new();
    let resp = match tokio::time::timeout(
        std::time::Duration::from_secs(8),
        client
            .get("https://openrouter.ai/api/v1/models")
            .header("Authorization", format!("Bearer {}", api_key))
            .send(),
    )
    .await
    {
        Ok(Ok(r)) => r,
        _ => return Vec::new(),
    };
    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(data) = body.get("data").and_then(|d| d.as_array()) else {
        return Vec::new();
    };

    let blocked_substrings: [&str; 0] = [];
    let preferred_providers = [
        "anthropic",
        "openai",
        "google",
        "x-ai",
        "deepseek",
        "qwen",
        "mistralai",
        "meta-llama",
        "minimax",
        "xiaomi",
    ];

    let mut candidates: Vec<(String, i64)> = data
        .iter()
        .filter_map(|m| {
            let id = m.get("id")?.as_str()?;
            let ctx = m
                .get("context_length")
                .and_then(|c| c.as_i64())
                .unwrap_or(0);
            if blocked_substrings.iter().any(|b| id.contains(b)) {
                return None;
            }
            if !preferred_providers.iter().any(|p| id.starts_with(p)) {
                return None;
            }
            Some((id.to_string(), ctx))
        })
        .collect();

    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    let mut seen_providers: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result = Vec::new();
    for (id, _) in &candidates {
        let provider = id.split('/').next().unwrap_or(id);
        if seen_providers.contains(provider) {
            continue;
        }
        seen_providers.insert(provider.to_string());
        result.push(format!("openrouter:{}", id));
        if result.len() >= limit {
            break;
        }
    }
    result
}

enum TaskCategory {
    Code,
    Reasoning,
    Creative,
    General,
}

fn classify_task(task: &str) -> TaskCategory {
    let t = task.to_lowercase();
    if t.contains("code")
        || t.contains("implement")
        || t.contains("function")
        || t.contains("bug")
        || t.contains("debug")
        || t.contains("rust")
        || t.contains("python")
        || t.contains("fix")
        || t.contains("refactor")
        || t.contains("compile")
        || t.contains("test")
        || t.contains("lint")
    {
        TaskCategory::Code
    } else if t.contains("math")
        || t.contains("proof")
        || t.contains("reason")
        || t.contains("logic")
        || t.contains("analyze")
        || t.contains("calculate")
        || t.contains("derive")
        || t.contains("solve")
    {
        TaskCategory::Reasoning
    } else if t.contains("write")
        || t.contains("essay")
        || t.contains("story")
        || t.contains("creative")
        || t.contains("poem")
        || t.contains("draft")
        || t.contains("summarize")
        || t.contains("explain")
    {
        TaskCategory::Creative
    } else {
        TaskCategory::General
    }
}

const DEBATE_ROLES: &[&str] = &[
    "Skeptic",
    "Architect",
    "Verifier",
    "Historian",
    "Devil's Advocate",
];

/// Models known to produce low-quality output, hallucinate excessively,
/// or fail to follow instructions reliably. Filtered from all selection paths.
/// Models excluded from auto-selection due to known quality issues:
/// obsolete generations, small parameter counts with poor instruction
/// following, or models that consistently fail multi-turn synthesis tasks.
const BLOCKED_MODELS: &[&str] = &[
    // OpenAI legacy
    "gpt-3.5",
    "gpt-4-turbo",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4.1",
    "gpt-4o-mini",
    "o1-mini",
    // Anthropic legacy
    "claude-instant",
    "claude-2",
    "claude-3-haiku",
    "claude-3-sonnet",
    // Google legacy
    "gemini-1.0",
    "gemini-1.5-flash-8b",
    "palm-2",
    // Meta small models (instruction following too weak for synthesis)
    "llama-2",
    "llama-3-8b",
    "llama-3.1-8b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    // Mistral small models
    "mistral-tiny",
    "mistral-small",
    "mistral-7b",
    "mixtral-8x7b",
    // Microsoft small models
    "phi-2",
    "phi-3",
    "phi-3-mini",
    "phi-3.5-mini",
    // Cohere legacy
    "command-r",
    "command-light",
    "command-r-plus-04-2024",
    // Google small models
    "gemma-2b",
    "gemma-7b",
    "gemma-2-2b",
    // Qwen small models
    "qwen-2.5-7b",
    "qwen-2.5-14b",
    "qwen-2-7b",
    // Other weak models
    "yi-6b",
    "yi-34b",
    "nous-hermes-2",
    "toppy-m-7b",
    "mythomist",
    "cinematika",
    "bagel",
    "psyfighter",
    "noromaid",
];

fn is_blocked_model(id: &str) -> bool {
    let lower = id.to_lowercase();
    BLOCKED_MODELS.iter().any(|b| lower.contains(b))
}

/// Detect which providers have API keys and return the best model IDs for the
/// given task, capped at 7 to keep the swarm manageable. Tries dynamic
/// discovery via OpenRouter first, then falls back to static FLAGSHIP_MODELS.
pub fn auto_select_models(task: &str) -> Vec<String> {
    let category = classify_task(task);

    let mut selected: Vec<String> = FLAGSHIP_MODELS
        .iter()
        .filter(|p| env::var(p.env_key).is_ok())
        .map(|p| match category {
            TaskCategory::Code => p.code_model,
            TaskCategory::Reasoning => p.reasoning_model,
            TaskCategory::Creative | TaskCategory::General => p.general_model,
        })
        .map(|s| s.to_string())
        .filter(|s| !is_blocked_model(s))
        .collect();

    if env::var("ANTHROPIC_API_KEY").is_ok() {
        if let Some(base_model) = selected.iter().find(|s| s.contains("claude")).cloned() {
            for role in DEBATE_ROLES {
                if selected.len() >= 7 {
                    break;
                }
                selected.push(format!("{}#{}", base_model, role));
            }
        }
    }

    selected.truncate(7);
    selected
}

/// Async version: fetches live models from OpenRouter and merges with
/// static providers. Call this instead of auto_select_models when a
/// tokio runtime is available.
pub async fn auto_select_models_dynamic(task: &str) -> Vec<String> {
    let category = classify_task(task);

    let mut selected: Vec<String> = FLAGSHIP_MODELS
        .iter()
        .filter(|p| env::var(p.env_key).is_ok())
        .filter(|p| p.env_key != "OPENROUTER_API_KEY")
        .map(|p| match category {
            TaskCategory::Code => p.code_model,
            TaskCategory::Reasoning => p.reasoning_model,
            TaskCategory::Creative | TaskCategory::General => p.general_model,
        })
        .map(|s| s.to_string())
        .filter(|s| !is_blocked_model(s))
        .collect();

    let dynamic = fetch_openrouter_top_models(4).await;
    for m in dynamic {
        if !is_blocked_model(&m) && !selected.iter().any(|s| s == &m) {
            selected.push(m);
        }
    }

    if env::var("ANTHROPIC_API_KEY").is_ok() {
        if let Some(base_model) = selected.iter().find(|s| s.contains("claude")).cloned() {
            for role in DEBATE_ROLES {
                if selected.len() >= 7 {
                    break;
                }
                selected.push(format!("{}#{}", base_model, role));
            }
        }
    }

    selected.truncate(7);
    selected
}

// ── Task wizard ───────────────────────────────────────────────────────────────

struct WizardState {
    fields: [String; 3], // task, workspace, iterations
    focused: usize,
    done: bool,
    cancelled: bool,
}

impl WizardState {
    fn new() -> Self {
        Self {
            fields: [String::new(), String::new(), String::new()],
            focused: 0,
            done: false,
            cancelled: false,
        }
    }

    fn task(&self) -> &str {
        &self.fields[0]
    }

    fn workspace(&self) -> Option<String> {
        let s = self.fields[1].trim();
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    }

    fn iterations(&self) -> u32 {
        self.fields[2].trim().parse().unwrap_or(0)
    }
}

fn draw_wizard(frame: &mut Frame, state: &WizardState) {
    let area = frame.area();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    let labels = [
        "Task (required)",
        "Workspace directory (optional, Enter to skip)",
        "Max iterations (0 = auto-converge, Enter for default)",
    ];
    let hints = [
        " Describe what you want Crosstalk to work on.",
        " Path to a directory whose files should be loaded as context.",
        " Maximum number of agent turns before stopping.",
    ];

    let title_text = " Crosstalk — Interactive Setup  [Tab/↓] Next field  [Shift+Tab/↑] Prev  [Enter] Start  [Esc] Quit";
    let title = Paragraph::new(title_text)
        .block(Block::default().borders(Borders::ALL).title(" Crosstalk "));
    frame.render_widget(title, layout[0]);

    for (i, (label, hint)) in labels.iter().zip(hints.iter()).enumerate() {
        let cursor = if state.focused == i { "█" } else { " " };
        let active = state.focused == i;
        let value = &state.fields[i];
        let display = if value.is_empty() && !active {
            format!("  {hint}")
        } else {
            format!("  {}{cursor}", value)
        };
        let style = if active {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" {label} "))
            .border_style(if active {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });
        let p = Paragraph::new(display).style(style).block(block);
        if i < 3 {
            frame.render_widget(p, layout[i + 1]);
        }
    }
}

/// Run an interactive setup wizard that collects the task, workspace, and
/// iteration count from the user. Returns `(task, workspace, iterations)`.
pub async fn run_task_wizard() -> Result<(String, Option<String>, u32)> {
    let mut state = WizardState::new();

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let result = loop {
        terminal.draw(|f| draw_wizard(f, &state))?;

        if state.done || state.cancelled {
            break if state.cancelled || state.task().trim().is_empty() {
                None
            } else {
                Some((
                    state.task().trim().to_string(),
                    state.workspace(),
                    state.iterations(),
                ))
            };
        }

        if !event::poll(std::time::Duration::from_millis(50))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Esc => state.cancelled = true,
            KeyCode::Enter => {
                if state.focused == 2 || (state.focused == 1 && state.task().trim().is_empty()) {
                    // stay on task if empty
                    if state.task().trim().is_empty() {
                        state.focused = 0;
                    } else {
                        state.done = true;
                    }
                } else if state.focused == 0 && state.task().trim().is_empty() {
                    // can't advance without a task
                } else {
                    state.focused = (state.focused + 1).min(2);
                }
            }
            KeyCode::Tab | KeyCode::Down => {
                if state.focused < 2 {
                    state.focused += 1;
                }
            }
            KeyCode::BackTab | KeyCode::Up => {
                if state.focused > 0 {
                    state.focused -= 1;
                }
            }
            KeyCode::Backspace => {
                state.fields[state.focused].pop();
            }
            KeyCode::Char(c) => {
                state.fields[state.focused].push(c);
            }
            _ => {}
        }
    };

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    match result {
        Some((task, workspace, iterations)) => Ok((task, workspace, iterations)),
        None => anyhow::bail!("Wizard cancelled."),
    }
}

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
    QueryKey, // appended as ?key=...
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
        let cursor = items
            .iter()
            .position(|i| matches!(i, Item::Model { .. }))
            .unwrap_or(0);
        Self {
            items,
            cursor,
            scroll: 0,
            done: false,
            confirm: false,
        }
    }

    fn add_models(&mut self, provider: &str, ids: Vec<String>) {
        // Find the header index for this provider
        let header_pos = self
            .items
            .iter()
            .position(|i| matches!(i, Item::Header { provider: p, .. } if p == provider));
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
            .map(|id| Item::Model {
                id,
                selected: false,
            })
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
        if len == 0 {
            return;
        }
        let mut i = self.cursor as isize;
        let mut steps = 0;
        loop {
            i += direction;
            steps += 1;
            if i < 0 {
                i = len as isize - 1;
            }
            if i >= len as isize {
                i = 0;
            }
            if i == self.cursor as isize || steps >= len {
                break;
            }
            if matches!(self.items[i as usize], Item::Model { .. }) {
                self.cursor = i as usize;
                break;
            }
        }
    }

    fn move_up(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor -= 1;
        if matches!(self.items.get(self.cursor), Some(Item::Header { .. })) && self.cursor > 0 {
            self.cursor -= 1;
        }
        self.clamp_scroll();
    }

    fn move_down(&mut self) {
        if self.cursor + 1 >= self.items.len() {
            return;
        }
        self.cursor += 1;
        if matches!(self.items.get(self.cursor), Some(Item::Header { .. }))
            && self.cursor + 1 < self.items.len()
        {
            self.cursor += 1;
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
                if let Item::Model {
                    id, selected: true, ..
                } = i
                {
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
        self.items
            .iter()
            .any(|i| matches!(i, Item::Header { loading: true, .. }))
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
        " Querying providers...  [↑↓] Navigate  [Space] Toggle  [a] Auto-select  [Enter] Start  [q] Quit"
    } else {
        " [↑↓] Navigate  [Space] Toggle  [a] Auto-select for task  [Enter] Start  [q] Quit"
    };

    let header = Paragraph::new(status).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Crosstalk — Select Models "),
    );
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
                    ListItem::new(label).style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )
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

    let list =
        List::new(visible_items).block(Block::default().borders(Borders::ALL).title(" Models "));
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

pub async fn run_model_selector(task_hint: &str) -> Result<Vec<String>> {
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
                    .unwrap_or_else(|e| {
                        tracing::warn!("reqwest::Client::builder().build() failed: {e}; falling back to default client (timeout may not apply)");
                        reqwest::Client::new()
                    });

                let url = list_url.to_string();

                let mut req = client.get(&url);
                if auth_style_val == 0 {
                    req = req.header("Authorization", format!("Bearer {}", api_key));
                } else if auth_style_val == 1 {
                    req = req
                        .header("x-api-key", &api_key)
                        .header("anthropic-version", "2023-06-01");
                } else if auth_style_val == 2 {
                    req = req.header("x-goog-api-key", &api_key);
                }

                let ids = async {
                    let resp = req.send().await.map_err(|e| {
                        tracing::warn!(provider = %provider_name, err = %e, "model list request failed");
                        e
                    }).ok()?.error_for_status().map_err(|e| {
                        tracing::warn!(provider = %provider_name, status = %e, "model list HTTP error");
                        e
                    }).ok()?;
                    let body: serde_json::Value = resp.json().await.map_err(|e| {
                        tracing::warn!(provider = %provider_name, err = %e, "model list JSON parse failed");
                        e
                    }).ok()?;
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

                crate::log_warn!(
                    tx.send((provider_name, ids)).await,
                    "failed to send model list"
                );
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
            state
                .lock()
                .map_err(|_| anyhow::anyhow!("model_select state mutex poisoned"))?
                .add_models(&provider, ids);
        }

        {
            let s = state
                .lock()
                .map_err(|_| anyhow::anyhow!("model_select state mutex poisoned"))?;
            terminal.draw(|f| draw_selector(f, &s))?;
            if s.done {
                break if s.confirm { s.selected_ids() } else { vec![] };
            }
        }

        if event::poll(std::time::Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            let mut s = state
                .lock()
                .map_err(|_| anyhow::anyhow!("model_select state mutex poisoned"))?;
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => s.move_up(),
                KeyCode::Down | KeyCode::Char('j') => s.move_down(),
                KeyCode::Char(' ') => s.toggle(),
                KeyCode::Char('a') => {
                    drop(s);
                    disable_raw_mode()?;
                    io::stdout().execute(LeaveAlternateScreen)?;
                    return Ok(auto_select_models(task_hint));
                }
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
    };

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(result)
}

// ── API Key Setup Wizard ─────────────────────────────────────────────────────

const PROVIDER_KEYS: &[(&str, &str, &str)] = &[
    (
        "Anthropic (Claude)",
        "ANTHROPIC_API_KEY",
        "https://console.anthropic.com/settings/keys",
    ),
    (
        "OpenAI (GPT)",
        "OPENAI_API_KEY",
        "https://platform.openai.com/api-keys",
    ),
    (
        "OpenRouter",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/keys",
    ),
    (
        "DeepSeek",
        "DEEPSEEK_API_KEY",
        "https://platform.deepseek.com/api_keys",
    ),
    (
        "Mistral",
        "MISTRAL_API_KEY",
        "https://console.mistral.ai/api-keys",
    ),
    ("Groq", "GROQ_API_KEY", "https://console.groq.com/keys"),
];

pub fn has_any_api_key() -> bool {
    PROVIDER_KEYS
        .iter()
        .any(|(_, key, _)| env::var(key).is_ok())
        || env::var("LLM_API_KEY").is_ok()
}

pub async fn run_api_key_setup() -> Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    let mut focused = 0usize;
    let mut inputs: Vec<String> = PROVIDER_KEYS.iter().map(|_| String::new()).collect();
    let mut done = false;
    let mut saved = false;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(5),
                    Constraint::Min(10),
                    Constraint::Length(3),
                ])
                .split(area);

            let welcome = Paragraph::new(
                "Welcome to Crosstalk!\n\nNo API keys found. Enter at least one key below.\nKeys will be saved to ~/.env for future sessions."
            )
            .block(Block::default().borders(Borders::ALL).title(" Setup "))
            .wrap(Wrap { trim: false });
            f.render_widget(welcome, chunks[0]);

            let items: Vec<ListItem> = PROVIDER_KEYS
                .iter()
                .enumerate()
                .map(|(i, (name, env_key, url))| {
                    let is_focused = i == focused;
                    let value = &inputs[i];
                    let display = if value.is_empty() {
                        format!("  {name}\n    {env_key} = (paste key here)\n    {url}")
                    } else {
                        let masked = format!("{}...{}", &value[..4.min(value.len())], &value[value.len().saturating_sub(4)..]);
                        format!("  {name}\n    {env_key} = {masked}")
                    };
                    let style = if is_focused {
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                    } else if !value.is_empty() {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    ListItem::new(display).style(style)
                })
                .collect();
            let list = List::new(items).block(
                Block::default().borders(Borders::ALL).title(" API Keys (Tab to switch, paste key, Enter to save) "),
            );
            f.render_widget(list, chunks[1]);

            let has_key = inputs.iter().any(|v| !v.is_empty());
            let hint = if has_key {
                " Press Enter to save and continue | Esc to quit "
            } else {
                " Paste an API key for any provider above | Esc to quit "
            };
            let status = Paragraph::new(hint);
            f.render_widget(status, chunks[2]);
        })?;

        if done {
            break;
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Tab | KeyCode::Down => {
                        focused = (focused + 1) % PROVIDER_KEYS.len();
                    }
                    KeyCode::BackTab | KeyCode::Up => {
                        focused = focused.checked_sub(1).unwrap_or(PROVIDER_KEYS.len() - 1);
                    }
                    KeyCode::Enter => {
                        if inputs.iter().any(|v| !v.is_empty()) {
                            saved = true;
                            done = true;
                        }
                    }
                    KeyCode::Esc => {
                        done = true;
                    }
                    KeyCode::Char(c) => {
                        inputs[focused].push(c);
                    }
                    KeyCode::Backspace => {
                        inputs[focused].pop();
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    if saved {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let env_path = format!("{home}/.env");
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&env_path)?;
        use std::io::Write;
        for (i, (_, env_key, _)) in PROVIDER_KEYS.iter().enumerate() {
            let value = inputs[i].trim();
            if !value.is_empty() {
                writeln!(file, "{env_key}={value}")?;
                // SAFETY: single-threaded at this point (before tokio runtime spawns tasks)
                unsafe {
                    env::set_var(env_key, value);
                }
            }
        }
        tracing::info!(path = %env_path, "API keys saved");
    } else {
        anyhow::bail!("Setup cancelled. Set at least one API key to use Crosstalk.");
    }

    Ok(())
}
