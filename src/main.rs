use crosstalk::core::agent_trait::PromptAgent;
use crosstalk::core::factory::ModelFactory;
use crosstalk::core::orchestrator::Orchestrator;
use crosstalk::core::state::StateManager;
use crosstalk::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crosstalk::types::events::{ControlSignal, StreamEvent};
use crosstalk::ui::app::App;
use crosstalk::ui::events::run_event_loop;
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

    #[arg(short, long, num_args = 1..)]
    models: Vec<String>,

    #[arg(short, long, default_value_t = 0)]
    iterations: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    let args = Args::parse();

    let session_id = "main-session";
    let manager = StateManager::new("/tmp/crosstalk")?;
    let sigma = Arc::new(Mutex::new(ConversationState::new(session_id)));

    let mut agents: Vec<Box<dyn PromptAgent>> = vec![];
    for m in &args.models {
        agents.push(ModelFactory::create_agent(m)?);
    }
    if agents.is_empty() {
        anyhow::bail!("No valid models provided. Use --models <model_id>");
    }

    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(1000);
    let (control_tx, control_rx) = mpsc::channel::<ControlSignal>(100);

    let omicron = Orchestrator::new(manager, agents, event_tx, control_rx);

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
    }

    let app = Arc::new(Mutex::new(App::new(session_id)));

    // Orchestrator task
    let sigma_orch = Arc::clone(&sigma);
    let app_orch = Arc::clone(&app);
    let iterations = args.iterations;
    tokio::spawn(async move {
        let mut i = 0u32;
        while let Ok(optimal) = omicron.run_turn(Arc::clone(&sigma_orch)).await {
            i += 1;
            if optimal || (iterations > 0 && i >= iterations) {
                break;
            }
        }
        app_orch.lock().await.shutdown = true;
    });

    // Event loop task (keyboard + stream events)
    let event_app = Arc::clone(&app);
    let ctrl_tx = control_tx;
    tokio::spawn(async move {
        let _ = run_event_loop(event_app, ctrl_tx, event_rx).await;
    });

    // Install panic hook to restore terminal before unwinding
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = io::stdout().execute(LeaveAlternateScreen);
        prev_hook(info);
    }));

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    // Render loop at ~60 fps
    let mut tick = tokio::time::interval(Duration::from_millis(16));
    loop {
        tick.tick().await;
        let mut app_guard = app.lock().await;
        if app_guard.shutdown {
            break;
        }
        app_guard.tick_fps();
        terminal.draw(|f| render::draw(f, &app_guard))?;
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
