use crate::agent_trait::PromptAgent;
use rig::providers::{gemini, openai};
use rig::prelude::*;
use anyhow::{Result, anyhow};
use std::env;

pub struct ModelFactory;

impl ModelFactory {
    pub fn create_agent(model_id: &str) -> Result<Box<dyn PromptAgent>> {
        if model_id.starts_with("gemini") {
            let api_key = env::var("GEMINI_API_KEY")
                .map_err(|_| anyhow!("Missing GEMINI_API_KEY"))?;
            let client = gemini::Client::new(&api_key).map_err(|e| anyhow!("Gemini client error: {:?}", e))?;
            let agent = client.agent(model_id).build();
            Ok(Box::new((model_id.to_string(), agent)))
        } else if model_id.starts_with("gpt") {
            let api_key = env::var("OPENAI_API_KEY")
                .map_err(|_| anyhow!("Missing OPENAI_API_KEY"))?;
            let client = openai::Client::new(&api_key).map_err(|e| anyhow!("OpenAI client error: {:?}", e))?;
            let agent = client.agent(model_id).build();
            Ok(Box::new((model_id.to_string(), agent)))
        } else {
            Err(anyhow!("Unsupported model provider for ID: {}", model_id))
        }
    }

    pub fn check_env(models: &[String]) -> Result<()> {
        for m in models {
            if m.starts_with("gemini") && env::var("GEMINI_API_KEY").is_err() {
                return Err(anyhow!("Environment Variable GEMINI_API_KEY is not set for {}", m));
            }
            if m.starts_with("gpt") && env::var("OPENAI_API_KEY").is_err() {
                return Err(anyhow!("Environment Variable OPENAI_API_KEY is not set for {}", m));
            }
        }
        Ok(())
    }
}
