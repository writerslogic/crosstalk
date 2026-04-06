use crosstalk::core::agent_trait::PromptAgent;
use crosstalk::core::factory::ModelFactory;
use crosstalk::core::orchestrator::Orchestrator;
use crosstalk::core::state::StateManager;
use crosstalk::types::conversation::{
    ConversationState, TaskCategory, Turn, TurnOutcome, TurnStructure,
};
use crosstalk::types::events::{ControlSignal, StreamEvent};
use crosstalk::ui::tui::CrosstalkUI;
use std::sync::Arc;
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

    let dir = "/tmp/crosstalk";
    let manager = StateManager::new(dir)?;

    let sigma = Arc::new(Mutex::new(ConversationState::new("main-session")));

    let mut agents: Vec<Box<dyn PromptAgent>> = vec![];

    for m in &args.models {
        agents.push(ModelFactory::create_agent(m)?);
    }

    if agents.is_empty() {
        anyhow::bail!("No valid models provided. Use --models <model_id>");
    }

    // Event System Setup
    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>(1000);
    let (control_tx, control_rx) = mpsc::channel::<ControlSignal>(100);

    let omicron = Orchestrator::new(manager, agents, event_tx, control_rx);
    let mut ui = CrosstalkUI::new(event_rx, control_tx)?;

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
        });
    }

    let sigma_orchestrator = Arc::clone(&sigma);

    // Run UI and Orchestrator concurrently
    tokio::spawn(async move {
        let mut i = 0;
        loop {
            match omicron.run_turn(Arc::clone(&sigma_orchestrator)).await {
                Ok(optimal) => {
                    i += 1;
                    if optimal || (args.iterations > 0 && i >= args.iterations) {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Orchestrator error: {:?}", e);
                    break;
                }
            }
        }
    });

    // Main thread handles UI rendering
    let initial_state = {
        let s = sigma.lock().await;
        s.clone()
    };
    ui.run(initial_state).await?;

    Ok(())
}
