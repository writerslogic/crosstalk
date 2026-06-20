//! LLM agent client for SWE-bench agent turns.
//!
//! Supports two API shapes:
//! - **Anthropic** (`https://api.anthropic.com/v1/messages`) — Claude models.
//! - **OpenAI-compatible** — OpenRouter, Groq, Together AI, Ollama, etc.
//!
//! When multiple models are provided, the agent rotates to the next model on
//! every 429/529 response instead of waiting, maximising throughput across
//! free-tier rate limits.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default free OpenRouter models, tried in order on rate-limit.
pub const DEFAULT_MODELS: &[&str] = &[
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
];

/// Default model for the Fast tier (used as CLI default).
pub const DEFAULT_MODEL: &str = SONNET_MODEL;

/// Sonnet — Fast tier for exploration turns and simple topologies.
pub const SONNET_MODEL: &str = "claude-sonnet-4-6";

/// Opus — Reasoning tier for complex topologies and escalation.
pub const OPUS_MODEL: &str = "claude-opus-4-6";

pub const OPENROUTER_BASE: &str = "https://openrouter.ai/api/v1";
pub const ANTHROPIC_BASE: &str = "https://api.anthropic.com/v1";

/// Model tier for the heterogeneous swarm router.
///
/// `Fast` maps to Sonnet, `Reasoning` maps to Opus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Fast,
    Reasoning,
}

fn cost_per_token(model: &str) -> (f64, f64) {
    if model.ends_with(":free") {
        return (0.0, 0.0);
    }
    if model.contains("opus") {
        return (15.0 / 1_000_000.0, 75.0 / 1_000_000.0);
    }
    if model.contains("sonnet") {
        return (3.0 / 1_000_000.0, 15.0 / 1_000_000.0);
    }
    if model.contains("haiku") {
        return (0.80 / 1_000_000.0, 4.0 / 1_000_000.0);
    }
    if model.contains("gpt-4o-mini") {
        return (0.15 / 1_000_000.0, 0.60 / 1_000_000.0);
    }
    (0.50 / 1_000_000.0, 1.50 / 1_000_000.0)
}

const SYSTEM_PROMPT: &str = "\
You are an expert software engineer in a SWE-bench evaluation environment. \
The repository is at /testbed. Fix the bug in the problem statement.

Available tools (one per line, exactly as shown):
  [TOOL: shell_exec(bash_command)]      — run a shell command
  [TOOL: file_read(path)]               — read a file
  [TOOL: file_write(path, content)]     — overwrite a file
  [TOOL: apply_patch(unified_diff)]     — apply a unified diff

NAVIGATION RULES — follow exactly:
  GOOD: [TOOL: shell_exec(grep -n 'function_name' /testbed/pkg/module.py)]
  GOOD: [TOOL: shell_exec(sed -n '40,70p' /testbed/pkg/module.py)]
  GOOD: [TOOL: shell_exec(grep -rn 'ErrorClass' /testbed/pkg/ --include='*.py' -l)]
  BAD:  [TOOL: shell_exec(cat /testbed/pkg/big_file.py)]   ← NEVER do this
  BAD:  [TOOL: shell_exec(find /testbed -name '*.py')]     ← too much output
  BAD:  [TOOL: shell_exec(grep -r 'keyword' /testbed)]     ← must use -l or -n with path

STRATEGY — follow turn budget strictly:
  Turn 0: grep for the error message or symbol from the problem statement.
  Turn 1: read only the 20–40 relevant lines around the suspect location.
  Turn 2: emit [PATCH] with your best fix. Do not wait for certainty.
  Turn 3+: if tests fail, read the failure output and emit a revised [PATCH].

[PATCH] format (unified diff, absolute paths):
  [PATCH]
  --- a/testbed/pkg/module.py
  +++ b/testbed/pkg/module.py
  @@ -N,M +N,M @@
   context
  -old line
  +new line
  [/PATCH]

HARD RULES:
  - Emit [PATCH] no later than turn 2. Exploring without patching wastes budget.
  - Never cat, head, or tail a file larger than ~50 lines without sed range.
  - Make the minimal change. Do not reformat unrelated code.
  - Tool output is capped at 3000 chars. Design commands to return targeted output.";

// ── Shared message type ──────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct Msg {
    role: String,
    content: String,
}

// ── Anthropic response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct AnthropicResp {
    content: Vec<AnthropicBlock>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── OpenAI-compatible response types ────────────────────────────────────────

#[derive(Deserialize)]
struct OpenAiResp {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ── Agent ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiType {
    Anthropic,
    OpenAiCompat,
}

/// Model ID for the Haiku fallback used when all free OpenRouter models are
/// rate-limited. This is the cheapest paid Claude model.
pub const HAIKU_FALLBACK_MODEL: &str = "claude-haiku-4-5-20251001";

/// Stateful LLM conversation for one SWE-bench instance.
///
/// When `models` contains more than one entry and the provider returns a
/// rate-limit error (429/529), the agent immediately rotates to the next
/// model and retries — no waiting. A full rotation without a single success
/// falls back to a short sleep before the next cycle.
///
/// If `fallback_key` is set (an Anthropic API key), a single Claude Haiku
/// call is made after all OpenRouter free-model rotations are exhausted
/// before giving up entirely.
pub struct ClaudeAgent {
    client: reqwest::Client,
    api_key: String,
    models: Vec<String>,
    model_idx: usize,
    api_type: ApiType,
    base_url: String,
    messages: Vec<Msg>,
    /// Optional Anthropic API key used as a last-resort Haiku fallback.
    fallback_key: Option<String>,
    /// The Fast-tier model (Haiku). Stored so `set_tier` can restore it.
    fast_model: String,
    /// The Reasoning-tier model (Sonnet). Empty string = tier disabled.
    reasoning_model: String,
}

impl ClaudeAgent {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("reqwest client"),
            api_key,
            models: vec![model.clone()],
            model_idx: 0,
            api_type: ApiType::Anthropic,
            base_url: ANTHROPIC_BASE.to_string(),
            messages: Vec::new(),
            fallback_key: None,
            fast_model: model,
            reasoning_model: String::new(),
        }
    }

    pub fn new_openai_compat(api_key: String, models: Vec<String>, base_url: String) -> Self {
        assert!(!models.is_empty(), "model list must not be empty");
        let fast = models[0].clone();
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("reqwest client"),
            api_key,
            models,
            model_idx: 0,
            api_type: ApiType::OpenAiCompat,
            base_url,
            messages: Vec::new(),
            fallback_key: None,
            fast_model: fast,
            reasoning_model: String::new(),
        }
    }

    /// Set an Anthropic API key used as a last-resort Haiku fallback when all
    /// OpenRouter free models are rate-limited.
    pub fn with_haiku_fallback(mut self, key: String) -> Self {
        self.fallback_key = Some(key);
        self
    }

    /// Set the Reasoning-tier model (Sonnet) for swarm escalation.
    pub fn with_reasoning_model(mut self, model: String) -> Self {
        self.reasoning_model = model;
        self
    }

    /// Switch the active model to the given tier.
    ///
    /// For the Anthropic path (`models.len() == 1`) this replaces
    /// `models[0]`; the message history is preserved.  If `Reasoning`
    /// is requested but no reasoning model has been configured, `Fast`
    /// is used as a safe fallback.
    pub fn set_tier(&mut self, tier: ModelTier) {
        let target = match tier {
            ModelTier::Fast => self.fast_model.clone(),
            ModelTier::Reasoning if !self.reasoning_model.is_empty() => {
                self.reasoning_model.clone()
            }
            ModelTier::Reasoning => self.fast_model.clone(),
        };
        let idx = self.model_idx % self.models.len();
        self.models[idx] = target;
    }

    pub fn reset(&mut self) {
        self.messages.clear();
    }

    fn current_model(&self) -> &str {
        &self.models[self.model_idx % self.models.len()]
    }

    /// Send `user_content` and return `(assistant_text, cost_usd)`.
    pub async fn send(&mut self, user_content: String) -> Result<(String, f64)> {
        self.messages.push(Msg {
            role: "user".into(),
            content: user_content,
        });

        let (text, in_tok, out_tok) = match self.api_type {
            ApiType::Anthropic => self.send_anthropic().await?,
            ApiType::OpenAiCompat => self.send_openai_compat().await?,
        };

        let (in_rate, out_rate) = cost_per_token(self.current_model());
        let cost = in_tok as f64 * in_rate + out_tok as f64 * out_rate;

        self.messages.push(Msg {
            role: "assistant".into(),
            content: text.clone(),
        });
        Ok((text, cost))
    }

    async fn send_anthropic(&self) -> Result<(String, u32, u32)> {
        let body = serde_json::json!({
            "model":      self.current_model(),
            "max_tokens": 4096,
            "system":     SYSTEM_PROMPT,
            "messages":   self.messages,
        });

        let resp = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Anthropic API request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API {status}: {err}");
        }

        let api: AnthropicResp = resp.json().await.context("parse Anthropic response")?;
        let text = api
            .content
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n");

        Ok((text, api.usage.input_tokens, api.usage.output_tokens))
    }

    async fn send_openai_compat(&mut self) -> Result<(String, u32, u32)> {
        let mut msgs = vec![Msg {
            role: "system".into(),
            content: SYSTEM_PROMPT.to_string(),
        }];
        msgs.extend(self.messages.iter().cloned());

        let n = self.models.len();
        // One full rotation = n attempts; after a full rotation with no success, sleep once.
        const MAX_ROTATIONS: usize = 3;

        for rotation in 0..MAX_ROTATIONS {
            for _ in 0..n {
                let model = self.current_model().to_owned();
                let body = serde_json::json!({
                    "model":      model,
                    "max_tokens": 4096,
                    "messages":   msgs,
                });

                let resp = self
                    .client
                    .post(format!("{}/chat/completions", self.base_url))
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("content-type", "application/json")
                    .header(
                        "HTTP-Referer",
                        "https://github.com/crosstalk-ai/crosstalk-eval",
                    )
                    .json(&body)
                    .send()
                    .await
                    .context("OpenAI-compat API request failed")?;

                let status = resp.status();

                if status == 429 || status == 529 {
                    let body_text = resp.text().await.unwrap_or_default();
                    self.model_idx = (self.model_idx + 1) % n;
                    tracing::info!(
                        next_model = %self.current_model(),
                        "Rate limited on {model} — rotating to next model"
                    );
                    // If we just completed a full rotation, sleep before trying again.
                    if self.model_idx == 0 {
                        let wait: u64 = serde_json::from_str::<serde_json::Value>(&body_text)
                            .ok()
                            .and_then(|v| v["error"]["metadata"]["retry_after_seconds"].as_u64())
                            .unwrap_or(30)
                            .min(60);
                        tracing::info!(
                            wait_secs = wait,
                            rotation,
                            "All models rate-limited — sleeping"
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    }
                    continue;
                }

                if !status.is_success() {
                    let err = resp.text().await.unwrap_or_default();
                    anyhow::bail!("OpenAI-compat API {status}: {err}");
                }

                let api: OpenAiResp = resp.json().await.context("parse OpenAI-compat response")?;
                let text = api
                    .choices
                    .into_iter()
                    .next()
                    .and_then(|c| c.message.content)
                    .unwrap_or_default();

                let usage = api.usage.unwrap_or_default();
                return Ok((text, usage.prompt_tokens, usage.completion_tokens));
            }
        }

        // All free models exhausted — try Anthropic Haiku as a paid fallback.
        if let Some(ref fallback_api_key) = self.fallback_key {
            tracing::info!("All free models rate-limited — falling back to Haiku");
            let body = serde_json::json!({
                "model":      HAIKU_FALLBACK_MODEL,
                "max_tokens": 4096,
                "system":     SYSTEM_PROMPT,
                "messages":   self.messages,
            });
            let resp = self
                .client
                .post(format!("{}/messages", ANTHROPIC_BASE))
                .header("x-api-key", fallback_api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .context("Haiku fallback API request failed")?;

            let status = resp.status();
            if !status.is_success() {
                let err = resp.text().await.unwrap_or_default();
                anyhow::bail!("Haiku fallback API {status}: {err}");
            }

            let api: AnthropicResp = resp.json().await.context("parse Haiku fallback response")?;
            let text = api
                .content
                .iter()
                .filter(|b| b.block_type == "text")
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");

            return Ok((text, api.usage.input_tokens, api.usage.output_tokens));
        }

        anyhow::bail!(
            "All {} models rate-limited after {} rotations",
            n,
            MAX_ROTATIONS
        )
    }
}
