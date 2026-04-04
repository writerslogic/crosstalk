use crosstalk::agent_trait::PromptAgent;
use crosstalk::orchestrator::Orchestrator;
use crosstalk::state::StateManager;
use crosstalk::types::ConversationState;
use rig::completion::PromptError;
use std::future::Future;
use std::pin::Pin;
use tempfile::tempdir;

struct MockAgent {
    name: String,
    response: String,
}

impl PromptAgent for MockAgent {
    fn name(&self) -> &str {
        &self.name
    }
    fn prompt<'a>(
        &'a self,
        _prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'a>> {
        let r = self.response.clone();
        Box::pin(async move { Ok(r) })
    }
}

#[tokio::test]
async fn test_orchestrator_turn_logic() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");
    let manager = StateManager::new(path).expect("Failed to create StateManager");

    let agent = Box::new(MockAgent {
        name: "MockModel".to_string(),
        response: "Hello, I am a model.".to_string(),
    });

    let omicron = Orchestrator::new(manager, vec![agent]);
    let mut sigma = ConversationState::new("test-session");

    let is_optimal = omicron.run_turn(&mut sigma).await.expect("Turn failed");

    assert!(!is_optimal);
    assert_eq!(sigma.iteration_index, 1);
}

#[tokio::test]
async fn test_orchestrator_convergence() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");
    let manager = StateManager::new(path).expect("Failed to create StateManager");

    let agent = Box::new(MockAgent {
        name: "MockModel".to_string(),
        response: "The solution is OPTIMAL".to_string(),
    });

    let omicron = Orchestrator::new(manager, vec![agent]);
    let mut sigma = ConversationState::new("test-session");

    let is_optimal = omicron.run_turn(&mut sigma).await.expect("Turn failed");

    assert!(is_optimal);
}

#[tokio::test]
async fn test_orchestrator_rewind() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");
    let manager = StateManager::new(path).expect("Failed to create StateManager");

    let agent = Box::new(MockAgent {
        name: "MockModel".to_string(),
        response: "Step 1".to_string(),
    });

    let omicron = Orchestrator::new(manager, vec![agent]);
    let mut sigma = ConversationState::new("test-session");

    omicron.run_turn(&mut sigma).await.expect("Turn 1 failed");
    assert_eq!(sigma.iteration_index, 1);

    let rewound = omicron.rewind(0).expect("Rewind failed");
    assert_eq!(rewound.iteration_index, 0);
    assert_eq!(rewound.turns.len(), 0);
}

#[tokio::test]
async fn test_orchestrator_artifact_capture() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");
    let manager = StateManager::new(path).expect("Failed to create StateManager");

    let agent = Box::new(MockAgent {
        name: "MockModel".to_string(),
        response: "Here is a file:\n```rust:test.rs\nprintln!(\"Hello\");\n```".to_string(),
    });

    let omicron = Orchestrator::new(manager, vec![agent]);
    let mut sigma = ConversationState::new("test-session");

    omicron.run_turn(&mut sigma).await.expect("Turn 1 failed");

    assert_eq!(sigma.artifacts.len(), 1);
    let art = sigma.artifacts.get("test.rs").expect("Artifact missing");
    assert_eq!(art.content, "println!(\"Hello\");");
    assert_eq!(art.version, 1);
    assert_eq!(sigma.turns[0].diffs.len(), 1);
}

#[tokio::test]
async fn test_orchestrator_ast_validation() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().to_str().expect("Failed to get path");
    let manager = StateManager::new(path).expect("Failed to create StateManager");

    let agent = Box::new(MockAgent {
        name: "MockModel".to_string(),
        response: "Invalid rust code:\n```rust:broken.rs\nfn main() { println!(\"oops\") \n```"
            .to_string(),
    });

    let omicron = Orchestrator::new(manager, vec![agent]);
    let mut sigma = ConversationState::new("test-session");

    omicron.run_turn(&mut sigma).await.expect("Turn failed");

    // Artifact should be rejected
    assert!(sigma.artifacts.is_empty());
}
