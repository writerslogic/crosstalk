use futures::{Stream, StreamExt};
use rig::completion::PromptError;
use std::future::Future;
use std::pin::Pin;

type StreamResult<'a> = Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'a>>;
type StreamFuture<'a> = Pin<Box<dyn Future<Output = Result<StreamResult<'a>, anyhow::Error>> + Send + 'a>>;

pub trait PromptAgent: Send + Sync {
    fn name(&self) -> &str;
    fn prompt<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'a>>;

    fn stream_prompt<'a>(&'a self, prompt: &'a str) -> StreamFuture<'a>;
}

pub struct AgentWrapper<M, R, T> {
    name: String,
    agent: T,
    _phantom: std::marker::PhantomData<(M, R)>,
}

impl<M, R, T> AgentWrapper<M, R, T> {
    pub fn new(name: String, agent: T) -> Self {
        Self {
            name,
            agent,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<M, R, T> PromptAgent for AgentWrapper<M, R, T>
where
    M: rig::completion::CompletionModel + Send + Sync + 'static,
    <M as rig::completion::CompletionModel>::StreamingResponse:
        rig::wasm_compat::WasmCompatSend + rig::completion::GetTokenUsage,
    R: Clone + Unpin + rig::completion::GetTokenUsage + Send + Sync + 'static,
    T: rig::completion::Prompt + rig::streaming::StreamingPrompt<M, R, Hook = ()> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn prompt<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'a>> {
        use std::future::IntoFuture;
        Box::pin(self.agent.prompt(prompt).into_future())
    }

    fn stream_prompt<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'a>>,
                        anyhow::Error,
                    >,
                > + Send
                + 'a,
        >,
    > {
        let agent = &self.agent;
        Box::pin(async move {
            let stream = agent.stream_prompt(prompt).await;

            let mapped_stream = stream.filter_map(|res| match res {
                Ok(rig::agent::MultiTurnStreamItem::StreamAssistantItem(
                    rig::streaming::StreamedAssistantContent::Text(text),
                )) => futures::future::ready(Some(Ok(text.text))),
                Ok(_) => futures::future::ready(None),
                Err(e) => {
                    futures::future::ready(Some(Err(anyhow::anyhow!("Stream error: {:?}", e))))
                }
            });

            Ok(Box::pin(mapped_stream)
                as Pin<
                    Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'a>,
                >)
        })
    }
}
