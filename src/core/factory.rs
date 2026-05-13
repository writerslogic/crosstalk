use crate::core::agent_trait::{AgentWrapper, PromptAgent};
use anyhow::Result;
use rig::client::CompletionClient;
use std::time::Duration;

pub async fn validate_agent(agent: &dyn PromptAgent) -> bool {
    match tokio::time::timeout(Duration::from_secs(30), agent.prompt("ping")).await {
        Ok(Ok(_)) => true,
        Ok(Err(e)) => {
            tracing::warn!(agent = %agent.name(), err = %e, "agent validation prompt failed");
            false
        }
        Err(_) => {
            tracing::warn!(agent = %agent.name(), "agent validation timed out");
            false
        }
    }
}

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
        let base_id = if let Some(i) = model_id.find('#') { &model_id[..i] } else { model_id };
        let actual_model_id = if base_id.contains(':') {
            base_id.rsplit(':').next().unwrap_or(base_id)
        } else {
            base_id
        };

        if actual_model_id.is_empty() {
            return Err(anyhow::anyhow!("Empty model ID"));
        }

        let model_id_lower = base_id.to_lowercase();

        if base_id.contains('/') || model_id_lower.starts_with("openrouter:") {
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
        } else if model_id_lower.starts_with("deepseek") {
            let api_key = require_env("DEEPSEEK_API_KEY")?;
            let client = rig::providers::deepseek::Client::new(&api_key)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<
                rig::providers::deepseek::CompletionModel,
                rig::providers::deepseek::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else if model_id_lower.starts_with("codestral") || model_id_lower.starts_with("mistral") || model_id_lower.starts_with("pixtral") || model_id_lower.starts_with("ministral") {
            let api_key = require_env("MISTRAL_API_KEY")?;
            let client = rig::providers::mistral::Client::new(&api_key)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<
                rig::providers::mistral::CompletionModel,
                rig::providers::mistral::completion::CompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else if model_id_lower.starts_with("llama") || model_id_lower.starts_with("gemma") || model_id_lower.starts_with("mixtral") {
            let api_key = require_env("GROQ_API_KEY")?;
            let client = rig::providers::groq::Client::new(&api_key)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<
                rig::providers::groq::CompletionModel,
                rig::providers::groq::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else {
            Self::create_openrouter_agent(model_id, actual_model_id)
        }
    }

    fn create_openrouter_agent(model_id: &str, actual_model_id: &str) -> Result<Box<dyn PromptAgent>> {
        let api_key = require_env("OPENROUTER_API_KEY")
            .or_else(|_| {
                tracing::warn!(model = model_id, "OPENROUTER_API_KEY not set; falling back to LLM_API_KEY");
                require_env("LLM_API_KEY")
            })
            .map_err(|_| anyhow::anyhow!("Missing OPENROUTER_API_KEY (and LLM_API_KEY fallback) for model {}", model_id))?;
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

    /// Map a bare model ID (e.g. "claude-opus-4-6") to its OpenRouter provider
    /// path (e.g. "anthropic/claude-opus-4-6"). Returns None if no mapping is known.
    pub fn openrouter_path(model_id: &str) -> Option<String> {
        let base = model_id.split('#').next().unwrap_or(model_id);
        let bare = if base.contains(':') { base.rsplit(':').next().unwrap_or(base) } else { base };
        let lower = bare.to_lowercase();
        let provider = if lower.contains("claude") || lower.contains("sonnet") || lower.contains("opus") || lower.contains("haiku") {
            "anthropic"
        } else if lower.starts_with("gpt") || lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4") {
            "openai"
        } else if lower.starts_with("deepseek") {
            let bare_mapped = match bare {
                "deepseek-reasoner" => "deepseek-r1",
                "deepseek-chat" => "deepseek-chat-v3-0324",
                other => other,
            };
            let role_suffix = model_id.find('#').map(|i| &model_id[i..]).unwrap_or("");
            return Some(format!("openrouter:deepseek/{bare_mapped}{role_suffix}"));
        } else if lower.starts_with("codestral") || lower.starts_with("mistral") || lower.starts_with("pixtral") || lower.starts_with("ministral") {
            "mistralai"
        } else if lower.starts_with("llama") {
            "meta-llama"
        } else if lower.starts_with("gemma") {
            "google"
        } else {
            return None;
        };
        let role_suffix = model_id.find('#').map(|i| &model_id[i..]).unwrap_or("");
        Some(format!("openrouter:{provider}/{bare}{role_suffix}"))
    }

    /// Try to create an agent via OpenRouter as a fallback for a model whose
    /// native provider is unavailable (e.g. out of credits).
    pub fn create_openrouter_fallback(model_id: &str) -> Result<Box<dyn PromptAgent>> {
        let or_id = Self::openrouter_path(model_id)
            .ok_or_else(|| anyhow::anyhow!("No OpenRouter mapping for {}", model_id))?;
        let bare = or_id.trim_start_matches("openrouter:");
        tracing::info!(model = model_id, fallback = %or_id, "rerouting agent through OpenRouter");
        Self::create_openrouter_agent(&or_id, bare)
    }

    pub fn check_env(model_ids: &[String]) -> Result<()> {
        for model_id in model_ids {
            let base = model_id.split('#').next().unwrap_or(model_id);
            let model_lower = base.to_lowercase();
            let key = if model_lower.contains('/') || model_lower.starts_with("openrouter:") {
                "OPENROUTER_API_KEY"
            } else if model_lower.contains("claude") || model_lower.contains("sonnet") || model_lower.contains("opus") || model_lower.contains("haiku") {
                "ANTHROPIC_API_KEY"
            } else if model_lower.starts_with("gpt") || model_lower.starts_with("o1") || model_lower.starts_with("o3") || model_lower.starts_with("o4") || model_lower.starts_with("chat") {
                "OPENAI_API_KEY"
            } else if model_lower.starts_with("deepseek") {
                "DEEPSEEK_API_KEY"
            } else if model_lower.starts_with("codestral") || model_lower.starts_with("mistral") || model_lower.starts_with("pixtral") || model_lower.starts_with("ministral") {
                "MISTRAL_API_KEY"
            } else if model_lower.starts_with("llama") || model_lower.starts_with("gemma") || model_lower.starts_with("mixtral") {
                "GROQ_API_KEY"
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


