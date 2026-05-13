use crosstalk::log_warn;
use crosstalk::core::agent_trait::PromptAgent;
use crosstalk::core::factory::ModelFactory;
use futures::future::join_all;
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
use crossterm::cursor::MoveTo;
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
#[command(name = "Crosstalk", version = "1.0.5", about = "AI Multi-Model Mediator")]
struct Args {
    #[arg(short, long)]
    task: Option<String>,

    #[arg(short, long, num_args = 0..)]
    models: Vec<String>,

    #[arg(short, long, default_value_t = 0)]
    iterations: u32,

    #[arg(short, long)]
    workspace: Option<String>,

    #[arg(short = 'f', long, num_args = 0..)]
    files: Vec<String>,

    #[arg(long, default_value_t = 300)]
    agent_timeout_secs: u64,

    #[arg(long, num_args = 1.., value_name = "SHELL")]
    generate_completions: Vec<String>,

    /// Automatically select the best available models for the task.
    #[arg(short = 'A', long, default_value_t = false)]
    auto: bool,

    /// After consensus, write agent-proposed changes back to the source files.
    #[arg(short = 'e', long, default_value_t = false)]
    edit: bool,

    /// Resume a prior session by its session ID (restores the latest checkpoint).
    #[arg(long)]
    resume: Option<String>,
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
    // Load ~/.env then .env (project-local), silently ignoring missing files.
    if let Ok(home) = std::env::var("HOME") {
        let _ = dotenv::from_path(std::path::Path::new(&home).join(".env"));
    }
    let _ = dotenv::dotenv();

    // 0. Initialize structured logging -- rotate, keeping last 5 logs
    let log_dir = std::env::var("XDG_STATE_HOME")
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{h}/.local/state"))
                .unwrap_or_else(|_| "/tmp".to_string())
        });
    let log_path = format!("{log_dir}/crosstalk");
    log_warn!(std::fs::create_dir_all(&log_path), "Failed to create log directory");
    // Rotate: keep the 5 most recent logs (ISO timestamps sort lexicographically)
    if let Ok(entries) = std::fs::read_dir(&log_path) {
        let mut logs: Vec<_> = entries
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("log"))
            .map(|e| e.path())
            .collect();
        logs.sort();
        for old in logs.iter().rev().skip(5) {
            log_warn!(std::fs::remove_file(old), "Failed to remove old log file");
        }
    }
    let run_ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let run_log = std::path::PathBuf::from(&log_path).join(format!("crosstalk-{run_ts}.log"));
    let log_file = std::fs::File::create(&run_log)?;
    let (non_blocking, _guard) = tracing_appender::non_blocking(log_file);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("crosstalk=debug")),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    tracing::info!("crosstalk session starting");

    // 1. Parse CLI args (env already loaded above)
    let args = Args::parse();

    if !args.generate_completions.is_empty() {
        use clap::CommandFactory;
        use clap_complete::{Shell, generate};
        let mut cmd = Args::command();
        for shell_str in &args.generate_completions {
            match shell_str.parse::<Shell>() {
                Ok(shell) => generate(shell, &mut cmd, "crosstalk", &mut std::io::stdout()),
                Err(_) => anyhow::bail!("Unknown shell '{shell_str}'. Supported: bash, zsh, fish, powershell, elvish"),
            }
        }
        return Ok(());
    }

    // If no task provided, run the interactive wizard to collect task + optional workspace/iterations.
    let (task_str, wizard_workspace, wizard_iterations) = if args.task.is_none() {
        let (t, ws, iters) = model_select::run_task_wizard().await?;
        (t, ws, iters)
    } else {
        (args.task.unwrap_or_default(), args.workspace.clone(), args.iterations)
    };

    let model_ids: Vec<String> = if !args.models.is_empty() {
        args.models
    } else if args.auto {
        let ids = model_select::auto_select_models_dynamic(&task_str).await;
        if ids.is_empty() {
            anyhow::bail!("No API keys found. Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY.");
        }
        ids
    } else {
        let selected = model_select::run_model_selector(&task_str).await?;
        if selected.is_empty() {
            anyhow::bail!("No models selected.");
        }
        selected
    };

    tracing::info!("crosstalk session starting, models: {:?}", model_ids);

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

    // 3b. Validate agent endpoints in parallel; fallback to OpenRouter on failure
    {
        let validation_futures: Vec<_> = agents
            .iter()
            .map(|a| crosstalk::core::factory::validate_agent(a.as_ref()))
            .collect();
        let results = join_all(validation_futures).await;
        let has_openrouter = std::env::var("OPENROUTER_API_KEY").is_ok();
        let mut valid_agents: Vec<Box<dyn PromptAgent>> = Vec::new();
        for (agent, ok) in agents.into_iter().zip(results.into_iter()) {
            if ok {
                valid_agents.push(agent);
            } else {
                let name = agent.name().to_string();
                if has_openrouter && !name.starts_with("openrouter:") {
                    if let Ok(fallback) = ModelFactory::create_openrouter_fallback(&name) {
                        if crosstalk::core::factory::validate_agent(fallback.as_ref()).await {
                            tracing::info!(agent = %name, "native provider failed, using OpenRouter fallback");
                            valid_agents.push(fallback);
                            continue;
                        }
                    }
                }
                tracing::warn!(agent = %name, "agent validation failed, removing");
            }
        }
        agents = valid_agents;
        if agents.is_empty() {
            anyhow::bail!("All agents failed endpoint validation. Check model IDs and API keys.");
        }
    }

    // 4. Initialize Core State
    let session_id = args.resume.clone().unwrap_or_else(|| run_ts.clone());
    let data_dir = std::env::var("XDG_DATA_HOME")
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{h}/.local/share"))
                .unwrap_or_else(|_| "/tmp".to_string())
        });
    let state_path = format!("{data_dir}/crosstalk/{session_id}");
    log_warn!(std::fs::create_dir_all(&state_path), "Failed to create state directory");
    let manager = StateManager::new(&state_path)?;
    let sigma = Arc::new(Mutex::new(if args.resume.is_some() {
        // Restore the latest checkpoint if resuming a prior session
        manager
            .list_checkpoints()
            .ok()
            .and_then(|mut idxs| { idxs.sort_unstable(); idxs.last().copied() })
            .and_then(|idx| manager.restore(idx).ok().flatten())
            .unwrap_or_else(|| ConversationState::new(&session_id))
    } else {
        ConversationState::new(&session_id)
    }));

    let effective_workspace = wizard_workspace.as_deref()
        .or(args.workspace.as_deref());

    if let Some(ws) = effective_workspace {
        let patterns = if args.files.is_empty() {
            vec!["**/*".to_string()]
        } else {
            args.files.clone()
        };
        let mut s = sigma.lock().await;
        for pattern in &patterns {
            let full_pattern = format!("{}/{}", ws, pattern);
            const MAX_GLOB_FILES: usize = 10_000;
            let paths: Vec<_> = match glob::glob(&full_pattern) {
                Ok(iter) => iter.flatten().take(MAX_GLOB_FILES + 1).collect(),
                Err(e) => {
                    tracing::warn!("invalid glob pattern '{}': {}", full_pattern, e);
                    continue;
                }
            };
            if paths.len() > MAX_GLOB_FILES {
                tracing::warn!(
                    pattern = %full_pattern,
                    "glob matched more than {} files, truncating",
                    MAX_GLOB_FILES
                );
            }
            let paths = &paths[..paths.len().min(MAX_GLOB_FILES)];
            for path in paths {
                if !path.is_file() || should_skip(path) || !is_likely_text(path) {
                    continue;
                }
                match tokio::fs::read_to_string(&path).await {
                    Ok(content) => {
                        let name = path.strip_prefix(ws).unwrap_or(&path).display().to_string();
                        let lang = lang_from_ext(&name);
                        s.ingest_file(name, lang, content);
                    }
                    Err(e) => tracing::debug!("skipping {}: {}", path.display(), e),
                }
            }
        }
        drop(s);
    } else if !args.files.is_empty() {
        let mut s = sigma.lock().await;
        for file_path in &args.files {
            let path = std::path::Path::new(file_path);
            if !path.is_file() || should_skip(path) || !is_likely_text(path) {
                tracing::debug!("skipping {}: not a text file or filtered", file_path);
                continue;
            }
            match tokio::fs::read_to_string(path).await {
                Ok(content) => {
                    let name = path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| file_path.clone());
                    let lang = lang_from_ext(&name);
                    s.ingest_file(name, lang, content);
                }
                Err(e) => tracing::warn!("failed to read {}: {}", file_path, e),
            }
        }
        drop(s);
    }

    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(1000);
    let (control_tx, control_rx) = mpsc::channel::<ControlSignal>(100);

    // 5. Initialize Orchestrator (may fail if engines fail to init)
    let workspace_root = effective_workspace.map(std::path::PathBuf::from);
    let omicron = match Orchestrator::new(manager, agents, event_tx, control_rx, workspace_root).await {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ORCHESTRATOR INIT ERROR: {}", e);
            anyhow::bail!("Failed to start orchestration engine.");
        }
    };

    let task = task_str;
    let task_content = {
        let s = sigma.lock().await;
        if s.artifacts.is_empty() {
            task.clone()
        } else {
            let file_names: Vec<&str> = s.artifacts.keys().map(|k| k.as_str()).collect();
            let names_str = file_names.join(", ");
            let edit_instruction = if args.edit {
                format!("\n\n[EDIT MODE] After reaching consensus, produce a final artifact named exactly `{}` containing the complete revised document with all agreed changes applied.", names_str)
            } else {
                String::new()
            };
            format!(
                "{}\n\n[GROUNDING CONSTRAINT] All claims must be grounded in the attached document(s): {}. Quote the specific section you are referencing. Do not assert implementation details, algorithms, or assumptions not explicitly stated in the text.{}\n\n[Workspace: {} file(s) loaded]",
                task, names_str, edit_instruction, s.artifacts.len()
            )
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
            persona_disclosure: None,
        });
        s.iteration_index = 1;
    }

    let initial_mode_idx = {
        let mut s = sigma.lock().await;
        let idx = crosstalk::types::mode::ModeDefinition::detect_preset_index(&task);
        s.mode_library.switch_to_index(idx);
        idx
    };

    let app = Arc::new(Mutex::new(App::new(&session_id)));
    {
        let mut a = app.lock().await;
        {
            let s = sigma.lock().await;
            a.current_mode_name = s.mode_library.current_name().to_string();
        }
        if initial_mode_idx != 0 {
            let mode_name = a.current_mode_name.clone();
            a.push_event(format!("Initial mode: {}", mode_name));
        }
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
    let iterations = wizard_iterations;
    let turn_timeout = Duration::from_secs(args.agent_timeout_secs);
    let omicron_orch = Arc::new(omicron);
    let omicron_spawn = Arc::clone(&omicron_orch);
    tokio::spawn(async move {
        let mut i = 0u32;
        loop {
            let sigma_in = Arc::clone(&sigma_orch);
            let omicron_in = Arc::clone(&omicron_spawn);
            let join = tokio::task::spawn(async move { omicron_in.run_turn(sigma_in).await });
            let res = match tokio::time::timeout(turn_timeout, join).await {
                Err(_elapsed) => {
                    let mut a = app_orch.lock().await;
                    a.push_event(format!("Turn {} timed out after {}s", i + 1, turn_timeout.as_secs()));
                    drop(a);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    if iterations > 0 && i >= iterations { break; }
                    continue;
                }
                Ok(r) => r,
            };

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
                    let session_id = sigma_orch.lock().await.session_id.clone();
                    Orchestrator::git_commit_session(&omicron_spawn.file_writer.root, &session_id, i).await;
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

    // SIGTERM handler: mirrors Ctrl+C shutdown path
    #[cfg(unix)]
    {
        let ctrl_tx_sigterm = ctrl_tx.clone();
        let app_sigterm = Arc::clone(&app);
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            if let Ok(mut sig) = signal(SignalKind::terminate()) {
                sig.recv().await;
                tracing::info!("SIGTERM received, initiating graceful shutdown");
                if let Err(e) = ctrl_tx_sigterm.send(ControlSignal::Shutdown).await {
                    tracing::warn!("Failed to send shutdown signal: {e}");
                }
                app_sigterm.lock().await.shutdown = true;
            }
        });
    }

    // 8. Initialize TUI
    enable_raw_mode()?;
    io::stdout().execute(MoveTo(0, 0))?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

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
            action
        };

        {
            let a = app.lock().await;
            if a.shutdown {
                break;
            }
            terminal.draw(|f| render::draw(f, &a))?;
        }

        match action {
            Action::Shutdown => {
                if ctrl_tx.send(ControlSignal::Shutdown).await.is_err() {
                    tracing::warn!("failed to send shutdown signal; forcing shutdown flag");
                    app.lock().await.shutdown = true;
                }
                break;
            }
            Action::Send(sig) => {
                if ctrl_tx.send(sig).await.is_err() {
                    tracing::warn!("failed to send control signal; shutting down");
                    app.lock().await.shutdown = true;
                    break;
                }
            }
            Action::SendTwo(s1, s2) => {
                if ctrl_tx.send(s1).await.is_err() || ctrl_tx.send(s2).await.is_err() {
                    tracing::warn!("failed to send control signal; shutting down");
                    app.lock().await.shutdown = true;
                    break;
                }
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

    if args.edit {
        let ws_root = effective_workspace.map(std::path::Path::new);
        let canonical_root = ws_root.and_then(|p| p.canonicalize().ok());
        let s = sigma.lock().await;
        for (name, artifact) in &s.artifacts {
            if name.contains("..") || name.starts_with('/') || name.starts_with('\\') {
                tracing::warn!(artifact = %name, "skipping artifact with unsafe path");
                continue;
            }
            let target = ws_root
                .map(|ws| ws.join(name))
                .unwrap_or_else(|| std::path::PathBuf::from(name));
            if let Some(ref root) = canonical_root {
                if let Ok(ct) = target.canonicalize() {
                    if !ct.starts_with(root) {
                        tracing::warn!(artifact = %name, "artifact path escapes workspace");
                        continue;
                    }
                }
            }
            if target.exists() && artifact.version > 1 {
                match tokio::fs::write(&target, &artifact.content).await {
                    Ok(()) => tracing::info!(path = %target.display(), bytes = artifact.content.len(), "edit-mode wrote artifact"),
                    Err(e) => tracing::warn!(path = %target.display(), err = %e, "edit-mode write failed"),
                }
            }
        }
        drop(s);
    }

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
        eprintln!("  Log: {}", run_log.display());
        eprintln!("---");
    }

    tracing::info!("session complete");
    drop(_guard);

    Ok(())
}

