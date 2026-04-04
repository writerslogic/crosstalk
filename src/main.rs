use crosstalk::agent_trait::PromptAgent;
use crosstalk::orchestrator::Orchestrator;
use crosstalk::state::StateManager;
use crosstalk::types::{self, ConversationState};
use rig::prelude::*;
use rig::providers::{gemini, openai};

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

    let mut sigma = ConversationState::new("main-session");

    let mut agents: Vec<Box<dyn PromptAgent>> = vec![];

    for m in args.models {
        if m.contains("gemini") {
            let api_key = std::env::var("GEMINI_API_KEY")?;
            let client = gemini::Client::new(&api_key)
                .map_err(|e| anyhow::anyhow!("Gemini client error: {e:?}"))?;
            let agent = client.agent(&m).build();
            agents.push(Box::new((m.clone(), agent)));
        } else if m.contains("gpt") {
            let api_key = std::env::var("OPENAI_API_KEY")?;
            let client = openai::Client::new(&api_key)
                .map_err(|e| anyhow::anyhow!("OpenAI client error: {e:?}"))?;
            let agent = client.agent(&m).build();
            agents.push(Box::new((m.clone(), agent)));
        }
    }

    if agents.is_empty() {
        anyhow::bail!("No valid models provided. Use --models <model_id>");
    }

    let omicron = Orchestrator::new(manager, agents);

    sigma.turns.push(types::Turn {
        index: 0,
        model_id: "User".to_string(),
        content: args.task,
        timestamp: ConversationState::now(),
        diffs: vec![],
    });

    println!("Starting debate loop...");

    let mut i = 0;
    loop {
        let optimal = omicron.run_turn(&mut sigma).await?;
        i += 1;

        if optimal || (args.iterations > 0 && i >= args.iterations) {
            println!("Process finished at i_{i}");
            break;
        }
    }

    Ok(())
}
