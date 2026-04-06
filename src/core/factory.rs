use crate::core::agent_trait::{AgentWrapper, PromptAgent};
use anyhow::{Result, anyhow};
use rig::client::CompletionClient;
use rig::providers::{anthropic, gemini, openai};
use std::env;

pub struct ModelFactory;

impl ModelFactory {
    pub fn create_agent(model_id: &str) -> Result<Box<dyn PromptAgent>> {
        let model_id_lower = model_id.to_lowercase();

        if model_id_lower.starts_with("gemini") {
            let api_key =
                env::var("GEMINI_API_KEY").map_err(|_| anyhow!("Missing GEMINI_API_KEY"))?;
            let client = gemini::Client::new(&api_key)?;
            let agent = client.agent(model_id).build();
            Ok(Box::new(AgentWrapper::<
                gemini::CompletionModel,
                gemini::streaming::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else if model_id_lower.starts_with("claude") {
            let api_key =
                env::var("ANTHROPIC_API_KEY").map_err(|_| anyhow!("Missing ANTHROPIC_API_KEY"))?;
            let client = anthropic::Client::new(&api_key)?;
            let agent = client.agent(model_id).build();
            Ok(Box::new(AgentWrapper::<
                anthropic::completion::CompletionModel,
                anthropic::streaming::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        } else {
            // Use OpenAI-compatible client for all other providers
            let (api_key, base_url) = if model_id_lower.starts_with("gpt")
                || model_id_lower.starts_with("o1")
                || model_id_lower.starts_with("o3")
            {
                (
                    env::var("OPENAI_API_KEY").map_err(|_| anyhow!("Missing OPENAI_API_KEY"))?,
                    None,
                )
            } else if model_id_lower.starts_with("deepseek") {
                (
                    env::var("DEEPSEEK_API_KEY")
                        .map_err(|_| anyhow!("Missing DEEPSEEK_API_KEY"))?,
                    Some("https://api.deepseek.com/v1"),
                )
            } else if model_id_lower.starts_with("mistral") {
                (
                    env::var("MISTRAL_API_KEY").map_err(|_| anyhow!("Missing MISTRAL_API_KEY"))?,
                    Some("https://api.mistral.ai/v1"),
                )
            } else if model_id_lower.contains("groq") {
                (
                    env::var("GROQ_API_KEY").map_err(|_| anyhow!("Missing GROQ_API_KEY"))?,
                    Some("https://api.groq.com/openai/v1"),
                )
            } else if model_id_lower.contains("perplexity") {
                (
                    env::var("PERPLEXITY_API_KEY")
                        .map_err(|_| anyhow!("Missing PERPLEXITY_API_KEY"))?,
                    Some("https://api.perplexity.ai"),
                )
            } else if model_id_lower.starts_with("sambanova") {
                (
                    env::var("SAMBANOVA_API_KEY")
                        .map_err(|_| anyhow!("Missing SAMBANOVA_API_KEY"))?,
                    Some("https://api.sambanova.ai/v1"),
                )
            } else if model_id_lower.starts_with("hyperbolic") {
                (
                    env::var("HYPERBOLIC_API_KEY")
                        .map_err(|_| anyhow!("Missing HYPERBOLIC_API_KEY"))?,
                    Some("https://api.hyperbolic.xyz/v1"),
                )
            } else if model_id_lower.starts_with("moonshot") {
                (
                    env::var("MOONSHOT_API_KEY")
                        .map_err(|_| anyhow!("Missing MOONSHOT_API_KEY"))?,
                    Some("https://api.moonshot.cn/v1"),
                )
            } else if model_id_lower.starts_with("ai21") {
                (
                    env::var("AI21_API_KEY").map_err(|_| anyhow!("Missing AI21_API_KEY"))?,
                    Some("https://api.ai21.com/studio/v1"),
                )
            } else if model_id_lower.starts_with("cohere") || model_id_lower.starts_with("command")
            {
                (
                    env::var("COHERE_API_KEY").map_err(|_| anyhow!("Missing COHERE_API_KEY"))?,
                    Some("https://api.cohere.ai/v1"),
                )
            } else {
                // Fallback to OpenRouter for everything else (or if it contains /)
                (
                    env::var("OPENROUTER_API_KEY")
                        .map_err(|_| anyhow!("Missing OPENROUTER_API_KEY"))?,
                    Some("https://openrouter.ai/api/v1"),
                )
            };

            let client = if let Some(url) = base_url {
                openai::Client::builder()
                    .api_key(&api_key)
                    .base_url(url)
                    .build()
            } else {
                openai::Client::new(&api_key)
            }?;

            let agent = client.agent(model_id).build();
            Ok(Box::new(AgentWrapper::<
                openai::responses_api::ResponsesCompletionModel,
                openai::responses_api::streaming::StreamingCompletionResponse,
                _,
            >::new(model_id.to_string(), agent)))
        }
    }

    pub fn check_env(models: &[String]) -> Result<()> {
        let keys = [
            ("gemini", "GEMINI_API_KEY"),
            ("gpt", "OPENAI_API_KEY"),
            ("o1", "OPENAI_API_KEY"),
            ("o3", "OPENAI_API_KEY"),
            ("claude", "ANTHROPIC_API_KEY"),
            ("deepseek", "DEEPSEEK_API_KEY"),
            ("mistral", "MISTRAL_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("perplexity", "PERPLEXITY_API_KEY"),
            ("cohere", "COHERE_API_KEY"),
            ("command", "COHERE_API_KEY"),
            ("moonshot", "MOONSHOT_API_KEY"),
            ("hyperbolic", "HYPERBOLIC_API_KEY"),
            ("sambanova", "SAMBANOVA_API_KEY"),
            ("ai21", "AI21_API_KEY"),
        ];

        for m in models {
            let m_lower = m.to_lowercase();
            let mut found = false;
            for (prefix, key) in keys {
                if m_lower.contains(prefix) {
                    if env::var(key).is_err() {
                        return Err(anyhow!("Environment Variable {} is not set for {}", key, m));
                    }
                    found = true;
                    break;
                }
            }
            if !found && env::var("OPENROUTER_API_KEY").is_err() {
                return Err(anyhow!(
                    "Environment Variable OPENROUTER_API_KEY is not set for unknown model {}",
                    m
                ));
            }
        }

        // Also check keys that might be used by tools
        let tool_keys = [
            "HF_API_KEY",
            "DEEPGRAM_API_KEY",
            "ORIGINALITY_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY",
        ];
        for key in tool_keys {
            if env::var(key).is_ok() {
                println!("[factory] Tool key recognized: {}", key);
            }
        }

        Ok(())
    }
}
