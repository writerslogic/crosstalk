use crate::core::agent_trait::{AgentWrapper, PromptAgent};
use anyhow::{Result, anyhow};
use rig::client::CompletionClient;
use rig::providers::{anthropic, gemini, openai};
use std::env;
use zeroize::Zeroizing;

fn require_env(key: &str) -> Result<Zeroizing<String>> {
    env::var(key).map(Zeroizing::new).map_err(|_| anyhow!("Missing required environment variable: {}", key))
}

pub struct ModelFactory;

impl ModelFactory {
    pub fn create_agent(model_id: &str) -> Result<Box<dyn PromptAgent>> {
        let model_id_lower = model_id.to_lowercase();
        
        // Robust Model Mapping
        let actual_model_id = if !model_id.contains('/') && model_id_lower.contains("opus-4.6") {
            "claude-3-opus-20240229"
        } else if !model_id.contains('/') && model_id_lower.contains("sonnet-latest") {
            "claude-3-5-sonnet-20241022"
        } else {
            model_id
        };

        if actual_model_id.starts_with("gemini") {
            let client = gemini::Client::new(&*require_env("GEMINI_API_KEY")?)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<gemini::CompletionModel, gemini::streaming::StreamingCompletionResponse, _>::new(model_id.to_string(), agent)))
        } else if actual_model_id.starts_with("claude") {
            let client = anthropic::Client::new(&*require_env("ANTHROPIC_API_KEY")?)?;
            let agent = client.agent(actual_model_id).build();
            Ok(Box::new(AgentWrapper::<anthropic::completion::CompletionModel, anthropic::streaming::StreamingCompletionResponse, _>::new(model_id.to_string(), agent)))
        } else {
            let (key, base_url): (&str, Option<&str>) = if actual_model_id.starts_with("gpt") || actual_model_id.starts_with("o1") || actual_model_id.starts_with("o3") { ("OPENAI_API_KEY", None) } 
            else if actual_model_id.starts_with("deepseek") { ("DEEPSEEK_API_KEY", Some("https://api.deepseek.com/v1")) } 
            else if actual_model_id.starts_with("mistral") { ("MISTRAL_API_KEY", Some("https://api.mistral.ai/v1")) } 
            else if actual_model_id.contains("groq") { ("GROQ_API_KEY", Some("https://api.groq.com/openai/v1")) } 
            else if actual_model_id.contains("perplexity") { ("PERPLEXITY_API_KEY", Some("https://api.perplexity.ai")) } 
            else if actual_model_id.starts_with("sambanova") { ("SAMBANOVA_API_KEY", Some("https://api.sambanova.ai/v1")) } 
            else if actual_model_id.starts_with("hyperbolic") { ("HYPERBOLIC_API_KEY", Some("https://api.hyperbolic.xyz/v1")) } 
            else if actual_model_id.starts_with("moonshot") { ("MOONSHOT_API_KEY", Some("https://api.moonshot.cn/v1")) } 
            else if actual_model_id.starts_with("ai21") { ("AI21_API_KEY", Some("https://api.ai21.com/studio/v1")) } 
            else if actual_model_id.starts_with("cohere") || actual_model_id.starts_with("command") { ("COHERE_API_KEY", Some("https://api.cohere.ai/v1")) } 
            else { ("OPENROUTER_API_KEY", Some("https://openrouter.ai/api/v1")) };
            
            let api_key = require_env(key)?;
            
            // Explicitly use OpenAI Completions extension to avoid 404s on /responses
            use rig::providers::openai::OpenAICompletionsExt;
            let builder = rig::client::Client::<OpenAICompletionsExt>::builder().api_key(&*api_key);
            let client = if let Some(url) = base_url { builder.base_url(url).build() } else { builder.build() }?;
            
            let agent = client.agent(actual_model_id).build();
            
            Ok(Box::new(AgentWrapper::<openai::completion::CompletionModel, openai::streaming::StreamingCompletionResponse, _>::new(model_id.to_string(), agent)))
        }
    }
    pub fn check_env(requested_models: &[String]) -> Result<()> {
        let all_keys = [("OpenAI", "OPENAI_API_KEY"), ("Anthropic", "ANTHROPIC_API_KEY"), ("Gemini", "GEMINI_API_KEY"), ("DeepSeek", "DEEPSEEK_API_KEY"), ("Mistral", "MISTRAL_API_KEY"), ("Groq", "GROQ_API_KEY"), ("Perplexity", "PERPLEXITY_API_KEY"), ("Cohere", "COHERE_API_KEY"), ("Moonshot", "MOONSHOT_API_KEY"), ("Hyperbolic", "HYPERBOLIC_API_KEY"), ("SambaNova", "SAMBANOVA_API_KEY"), ("AI21", "AI21_API_KEY"), ("OpenRouter", "OPENROUTER_API_KEY")];
        println!("\n[System] --- Swarm Capacity Report ---");
        for (name, key) in all_keys { if env::var(key).is_ok() { println!("  [✓] {} Provider: ACTIVE", name); } }
        println!("[System] ------------------------------\n");
        for m in requested_models {
            let m_lower = m.to_lowercase();
            let mut found = false;
            if m_lower.starts_with("gpt") || m_lower.starts_with("o1") || m_lower.starts_with("o3") { if env::var("OPENAI_API_KEY").is_ok() { found = true; } }
            else { for (name, key) in all_keys { if m_lower.contains(&name.to_lowercase()) { if env::var(key).is_ok() { found = true; break; } } } }
            if !found && env::var("OPENROUTER_API_KEY").is_err() { return Err(anyhow!("Model {} is unrecognized and no OPENROUTER_API_KEY was found.", m)); }
        }
        Ok(())
    }
}