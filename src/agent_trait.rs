use rig::completion::{Prompt, PromptError};
use std::pin::Pin;
use std::future::{Future, IntoFuture};

pub trait PromptAgent: Send + Sync {
    fn name(&self) -> &str;
    fn prompt<'a>(&'a self, prompt: &'a str) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'a>>;
}

impl<T: Prompt + Send + Sync> PromptAgent for (String, T) {
    fn name(&self) -> &str {
        &self.0
    }
    fn prompt<'a>(&'a self, prompt: &'a str) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'a>> {
        Box::pin(self.1.prompt(prompt).into_future())
    }
}
