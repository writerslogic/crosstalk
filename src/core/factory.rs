use crate::core::agent_trait::{AgentWrapper, PromptAgent};
use anyhow::Result;
use rig::client::CompletionClient;

pub struct ModelFactory;

fn require_env(key: &str) -> Result<String> {
    let val = std::env::var(key).map_err(|_| anyhow::anyhow!("Missing {key}"))?;
    if val.trim().is_empty() {
        return Err(anyhow::anyhow!("{key} is set but empty"));
    }
    Ok(val)
}

impl ModelFactory {
    pub fn create_agent(model_id: &str) -> Result<Box<dyn PromptAgent>> {
        let actual_model_id = if model_id.contains(':') {
            model_id.rsplit(':').next().unwrap()
        } else {
            model_id
        };

        if actual_model_id.is_empty() {
            return Err(anyhow::anyhow!("Empty model ID"));
        }

        let model_id_lower = model_id.to_lowercase();

        if model_id.contains('/') || model_id_lower.starts_with("openrouter:") {
            return Self::create_openrouter_agent(model_id, actual_model_id);
        }

        if model_id_lower.contains("sonnet") || model_id_lower.contains("claude") || model_id_lower.contains("opus") || model_id_lower.contains("haiku") {
            let api_key = require_env("ANTHROPIC_API_KEY")?;
            let client = rig::providers::anthropic::Client::new(&api_key)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<
                rig::providers::anthropic::completion::CompletionModel,
                rig::providers::anthropic::streaming::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else if model_id_lower.starts_with("gpt") || model_id_lower.starts_with("o1") || model_id_lower.starts_with("o3") || model_id_lower.starts_with("o4") || model_id_lower.starts_with("chat") {
            let api_key = require_env("OPENAI_API_KEY")?;
            let client = rig::providers::openai::Client::new(&api_key)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<
                rig::providers::openai::responses_api::ResponsesCompletionModel,
                rig::providers::openai::responses_api::streaming::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else {
            Self::create_openrouter_agent(model_id, actual_model_id)
        }
    }

    fn create_openrouter_agent(model_id: &str, actual_model_id: &str) -> Result<Box<dyn PromptAgent>> {
        let api_key = require_env("OPENROUTER_API_KEY")
            .or_else(|_| require_env("LLM_API_KEY"))
            .map_err(|_| anyhow::anyhow!("Missing OPENROUTER_API_KEY or LLM_API_KEY for model {}", model_id))?;
        let base_url = std::env::var("OPENROUTER_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
        let model_name = actual_model_id.trim_start_matches("openrouter:");
        let client = rig::providers::openai::CompletionsClient::builder()
            .api_key(&api_key)
            .base_url(&base_url)
            .build()?;
        let agent = client.agent(model_name).build();
        Ok(Box::new(AgentWrapper::<
            rig::providers::openai::completion::CompletionModel,
            rig::providers::openai::completion::streaming::StreamingCompletionResponse,
            _,
        >::new(model_id.to_string(), agent)))
    }

    pub fn check_env(model_ids: &[String]) -> Result<()> {
        for model_id in model_ids {
            let model_lower = model_id.to_lowercase();
            let key = if model_lower.contains('/') || model_lower.starts_with("openrouter:") {
                "OPENROUTER_API_KEY"
            } else if model_lower.contains("claude") || model_lower.contains("sonnet") || model_lower.contains("opus") || model_lower.contains("haiku") {
                "ANTHROPIC_API_KEY"
            } else if model_lower.starts_with("gpt") || model_lower.starts_with("o1") || model_lower.starts_with("o3") || model_lower.starts_with("o4") || model_lower.starts_with("chat") {
                "OPENAI_API_KEY"
            } else {
                "OPENROUTER_API_KEY"
            };
            std::env::var(key)
                .or_else(|_| std::env::var("LLM_API_KEY"))
                .map_err(|_| anyhow::anyhow!("Missing {} (or LLM_API_KEY) for model {}", key, model_id))?;
        }
        Ok(())
    }
}


