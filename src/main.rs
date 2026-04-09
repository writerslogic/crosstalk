use crosstalk::core::agent_trait::PromptAgent;
use crosstalk::core::factory::ModelFactory;
use crosstalk::core::orchestrator::Orchestrator;
use crosstalk::core::state::StateManager;
use crosstalk::types::events::{ControlSignal, StreamEvent};
use crosstalk::types::conversation::{ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure};
use crosstalk::ui::app::App;
use crosstalk::ui::events::run_event_loop;
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(1000);
    let (control_tx, control_rx) = mpsc::channel::<ControlSignal>(100);

    // 5. Initialize Orchestrator (may fail if engines fail to init)
    let omicron = match Orchestrator::new(manager, agents, event_tx, control_rx).await {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ORCHESTRATOR INIT ERROR: {}", e);
            anyhow::bail!("Failed to start orchestration engine.");
        }
    };

    {
        let mut s = sigma.lock().await;
        s.turns.push(Turn {
            index: 0,
            model_id: "User".to_string(),
            content: args.task,
            timestamp: ConversationState::now(),
            diffs: vec![],
            certainty: Some(1.0),
            outcome: TurnOutcome::Unknown,
            task_category: Some(TaskCategory::Research),
            structure: Some(TurnStructure::FreeForm),
            signature: vec![],
            surprise_signal: None,
        });
        s.iteration_index = 1;
    }

    let app = Arc::new(Mutex::new(App::new(session_id)));

    // 6. Spawn background tasks
    let sigma_orch = Arc::clone(&sigma);
    let app_orch = Arc::clone(&app);
    let iterations = args.iterations;
    let omicron_orch = Arc::new(omicron);
    tokio::spawn(async move {
        let mut i = 0u32;
        loop {
            let sigma_in = Arc::clone(&sigma_orch);
            let omicron_in = Arc::clone(&omicron_orch);
            let res = tokio::task::spawn(async move {
                omicron_in.run_turn(sigma_in).await
            }).await;

            match res {
                Ok(Ok(optimal)) => {
                    i += 1;
                    if optimal || (iterations > 0 && i >= iterations) {
                        break;
                    }
                }
                Ok(Err(e)) => {
                    let mut app_err = app_orch.lock().await;
                    app_err.push_event(format!("ORCHESTRATOR ERROR: {}", e));
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    if iterations > 0 && i >= iterations { break; }
                }
                Err(e) => {
                    let mut app_err = app_orch.lock().await;
                    app_err.push_event(format!("ORCHESTRATOR PANIC: {}", e));
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    break;
                }
            }
        }
        app_orch.lock().await.shutdown = true;
    });

    let event_app = Arc::clone(&app);
    let ctrl_tx = control_tx;
    tokio::spawn(async move {
        let _ = run_event_loop(event_app, ctrl_tx, event_rx).await;
    });

    // 7. Initialize TUI
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode().ok();
        let _ = io::stdout().execute(LeaveAlternateScreen);
        prev_hook(info);
    }));

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    // 8. Main TUI render loop
    loop {
        let mut app_guard = app.lock().await;
        if app_guard.shutdown {
            break;
        }
        app_guard.tick_fps();
        terminal.draw(|f| render::draw(f, &app_guard))?;
        drop(app_guard);
        tokio::time::sleep(Duration::from_millis(16)).await;
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
