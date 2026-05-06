use crosstalk::log_warn;
use crosstalk::core::agent_trait::PromptAgent;
use crosstalk::core::factory::ModelFactory;
use crosstalk::core::orchestrator::Orchestrator;
use crosstalk::core::state::StateManager;
use crosstalk::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crosstalk::types::events::{ControlSignal, StreamEvent};
use crosstalk::ui::app::App;
use crosstalk::ui::events::{self as ui_events, Action};
use crosstalk::ui::model_select;
use crosstalk::ui::render;
use crossterm::ExecutableCommand;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::sync::mpsc;

use clap::Parser;

#[derive(Parser)]
#[command(name = "Crosstalk", version = "1.0", about = "AI Multi-Model Mediator")]
struct Args {
    #[arg(short, long)]
    task: String,

    #[arg(short, long, num_args = 0..)]
    models: Vec<String>,

    #[arg(short, long, default_value_t = 0)]
    iterations: u32,

    #[arg(short, long)]
    workspace: Option<String>,

    #[arg(short = 'f', long, num_args = 0..)]
    files: Vec<String>,
}

fn lang_from_ext(path: &str) -> String {
    match path.rsplit('.').next().unwrap_or("") {
        "rs" => "rust",
        "py" => "python",
        "js" => "javascript",
        "ts" => "typescript",
        "md" => "markdown",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "sh" | "bash" => "shell",
        "html" => "html",
        "css" => "css",
        "sql" => "sql",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" => "cpp",
        ext => ext,
    }.to_string()
}

const SKIP_DIRS: &[&str] = &[
    ".git", "__pycache__", "node_modules", "target", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info", ".DS_Store",
];

fn should_skip(path: &std::path::Path) -> bool {
    for component in path.components() {
        let name = component.as_os_str().to_string_lossy();
        if SKIP_DIRS.iter().any(|s| name == *s || name.ends_with(".egg-info")) {
            return true;
        }
    }
    false
}

fn is_likely_text(path: &std::path::Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(ext,
        "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "md" | "txt" | "toml" |
        "yaml" | "yml" | "json" | "sh" | "bash" | "zsh" | "html" | "css" |
        "sql" | "go" | "java" | "c" | "h" | "cpp" | "hpp" | "cc" | "rb" |
        "ex" | "exs" | "hs" | "ml" | "mli" | "r" | "R" | "jl" | "lua" |
        "pl" | "pm" | "swift" | "kt" | "scala" | "clj" | "erl" | "cfg" |
        "ini" | "conf" | "env" | "xml" | "csv" | "tsv" | "makefile" |
        "dockerfile" | "gitignore" | "lock"
    ) || path.file_name().is_some_and(|n| {
        let n = n.to_string_lossy().to_lowercase();
        n == "makefile" || n == "dockerfile" || n == ".gitignore" || n == "cargo.lock"
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 0. Initialize structured logging -- one file per run, old logs wiped
    let log_dir = std::env::var("XDG_STATE_HOME")
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{h}/.local/state"))
                .unwrap_or_else(|_| "/tmp".to_string())
        });
    let log_path = format!("{log_dir}/crosstalk");
    log_warn!(std::fs::create_dir_all(&log_path), "Failed to create log directory");
    // Wipe previous run logs
    if let Ok(entries) = std::fs::read_dir(&log_path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("log") {
                log_warn!(std::fs::remove_file(&p), "Failed to remove log file");
            }
        }
    }
    let run_ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let run_log = std::path::PathBuf::from(&log_path).join(format!("crosstalk-{run_ts}.log"));
    let log_file = std::fs::File::create(&run_log)?;
    let (non_blocking, _guard) = tracing_appender::non_blocking(log_file);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("crosstalk=info")),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    // 1. Load environment variables
    dotenv::dotenv().ok();
    let args = Args::parse();

    let model_ids: Vec<String> = if args.models.is_empty() {
        let selected = model_select::run_model_selector().await?;
        if selected.is_empty() {
            anyhow::bail!("No models selected.");
        }
        selected
    } else {
        args.models
    };

    // 2. Pre-flight credential check
    if let Err(e) = ModelFactory::check_env(&model_ids) {
        eprintln!("PRE-FLIGHT ERROR: {}", e);
        anyhow::bail!("Initialization aborted due to missing credentials.");
    }

    // 3. Initialize Agents
    let mut agents: Vec<Box<dyn PromptAgent>> = vec![];
    for m in &model_ids {
        agents.push(ModelFactory::create_agent(m)?);
    }
    if agents.is_empty() {
        anyhow::bail!("No valid models provided. Use --models <model_id>");
    }

    // 4. Initialize Core State
    let session_id = "main-session";
    let manager = StateManager::new(&format!("/tmp/crosstalk_{}", std::process::id()))?;
    let sigma = Arc::new(Mutex::new(ConversationState::new(session_id)));

    if let Some(ref ws) = args.workspace {
        let patterns = if args.files.is_empty() {
            vec!["**/*".to_string()]
        } else {
            args.files.clone()
        };
        let mut s = sigma.lock().await;
        for pattern in &patterns {
            let full_pattern = format!("{}/{}", ws, pattern);
            for path in glob::glob(&full_pattern).unwrap_or_else(|_| glob::glob("").unwrap()).flatten() {
                if !path.is_file() || should_skip(&path) || !is_likely_text(&path) {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let name = path.strip_prefix(ws).unwrap_or(&path).display().to_string();
                    let lang = lang_from_ext(&name);
                    s.ingest_file(name, lang, content);
                }
            }
        }
        drop(s);
    }

    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(1000);
    let (control_tx, control_rx) = mpsc::channel::<ControlSignal>(100);

    // 5. Initialize Orchestrator (may fail if engines fail to init)
    let workspace_root = args.workspace.as_deref().map(std::path::PathBuf::from);
    let omicron = match Orchestrator::new(manager, agents, event_tx, control_rx, workspace_root).await {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ORCHESTRATOR INIT ERROR: {}", e);
            anyhow::bail!("Failed to start orchestration engine.");
        }
    };

    let task_content = {
        let s = sigma.lock().await;
        if s.artifacts.is_empty() {
            args.task.clone()
        } else {
            format!("{}\n\n[Workspace: {} files loaded as artifacts]", args.task, s.artifacts.len())
        }
    };

    {
        let mut s = sigma.lock().await;
        s.turns.push(Turn {
            index: 0,
            model_id: "User".to_string(),
            content: task_content,
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: Some(1.0),
            outcome: TurnOutcome::Unknown,
            task_category: Some(TaskCategory::Research),
            structure: Some(TurnStructure::FreeForm),
            signature: vec![],
            surprise_signal: None,
            consistency_score: None,
            diff_quality_score: None,
        });
        s.iteration_index = 1;
    }

    let app = Arc::new(Mutex::new(App::new(session_id)));
    {
        let mut a = app.lock().await;
        a.push_event(format!("Session started with {} agent(s)", model_ids.len()));
        for m in &model_ids {
            a.agent_list.push(m.clone());
            a.push_event(format!("  Agent: {}", m));
        }
        let artifact_count = sigma.lock().await.artifacts.len();
        if artifact_count > 0 {
            a.push_event(format!("Workspace: {} files loaded", artifact_count));
        }
    }

    // 6. Set panic hook before spawning any tasks
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log_warn!(disable_raw_mode(), "Failed to disable raw mode");
        log_warn!(io::stdout().execute(LeaveAlternateScreen), "Failed to leave alternate screen");
        prev_hook(info);
    }));

    // 7. Spawn background tasks
    let sigma_orch = Arc::clone(&sigma);
    let app_orch = Arc::clone(&app);
    let iterations = args.iterations;
    let omicron_orch = Arc::new(omicron);
    let omicron_spawn = Arc::clone(&omicron_orch);
    tokio::spawn(async move {
        let mut i = 0u32;
        loop {
            let sigma_in = Arc::clone(&sigma_orch);
            let omicron_in = Arc::clone(&omicron_spawn);
            let res = tokio::task::spawn(async move { omicron_in.run_turn(sigma_in).await }).await;

            match res {
                Ok(Ok(optimal)) => {
                    i += 1;
                    if optimal {
                        let mut a = app_orch.lock().await;
                        a.push_event(format!("Converged after {} turn(s)", i));
                        break;
                    }
                    if iterations > 0 && i >= iterations {
                        let mut a = app_orch.lock().await;
                        a.push_event(format!("Completed {} iteration(s)", i));
                        break;
                    }
                }
                Ok(Err(e)) => {
                    let mut app_err = app_orch.lock().await;
                    app_err.push_event(format!("Turn {} error: {}", i + 1, e));
                    drop(app_err);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    if iterations > 0 && i >= iterations {
                        break;
                    }
                }
                Err(e) => {
                    let mut app_err = app_orch.lock().await;
                    app_err.push_event(format!("Turn {} panic: {}", i + 1, e));
                    drop(app_err);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    break;
                }
            }
        }
        let mut a = app_orch.lock().await;
        a.push_event("Session ending...".to_string());
        a.shutdown = true;
    });

    let mut event_rx = event_rx;
    let ctrl_tx = control_tx;

    // 8. Initialize TUI
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    // 9. Main loop: drain events, handle keys, render
    loop {
        let action = {
            let mut a = app.lock().await;
            ui_events::drain_stream_events(&mut a, &mut event_rx);
            if a.shutdown {
                break;
            }

            let action = match ui_events::poll_key(Duration::from_millis(16)) {
                Some(key) => ui_events::handle_key(&mut a, key),
                None => Action::None,
            };

            a.tick_fps();
            terminal.draw(|f| render::draw(f, &a))?;
            action
        };

        match action {
            Action::Shutdown => {
                log_warn!(ctrl_tx.send(ControlSignal::Shutdown).await, "failed to send shutdown signal");
                break;
            }
            Action::Send(sig) => {
                log_warn!(ctrl_tx.send(sig).await, "failed to send control signal");
            }
            Action::SendTwo(s1, s2) => {
                log_warn!(ctrl_tx.send(s1).await, "failed to send control signal 1");
                log_warn!(ctrl_tx.send(s2).await, "failed to send control signal 2");
            }
            Action::None => {}
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    // Graceful shutdown: finalize session, persist memory, shut down orchestrator.
    log_warn!(omicron_orch.finalize_session(Arc::clone(&sigma)).await, "session finalization failed");
    {
        let bridge = omicron_orch.memory_bridge.lock().await;
        let session_id = sigma.lock().await.session_id.clone();
        let records = bridge.take_snapshot(&session_id);
        for record in records {
            log_warn!(omicron_orch.memory_store.store(record).await, "failed to persist memory record");
        }
    }
    omicron_orch.shutdown().await;

    // Print session summary
    {
        let s = sigma.lock().await;
        let a = app.lock().await;
        let turns = s.turns.len().saturating_sub(1); // exclude initial user turn
        let artifacts = s.artifacts.len();
        let conv = s.completion_probability;
        let errors: Vec<&String> = a.recent_events.iter().filter(|e| e.contains("ERROR") || e.contains("error:") || e.contains("PANIC")).collect();
        eprintln!("\n--- Crosstalk Session Summary ---");
        eprintln!("  Turns completed: {}", turns);
        eprintln!("  Artifacts:       {}", artifacts);
        eprintln!("  Convergence:     {:.1}%", conv * 100.0);
        if !errors.is_empty() {
            eprintln!("  Errors ({}):", errors.len());
            for e in errors.iter().take(5) {
                eprintln!("    {}", e);
            }
            if errors.len() > 5 {
                eprintln!("    ... and {} more", errors.len() - 5);
            }
        }
        eprintln!("  Log: /tmp/crosstalk.log");
        eprintln!("---");
    }

    Ok(())
}
