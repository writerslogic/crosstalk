use crosstalk::mcp::bridge::{CargoBridge, GitBridge};
use crosstalk::mcp::gateway::{
    McpGateway, McpResource, McpTool, PermissionError, PermissionManager, PermissionTier,
};
use crosstalk::mcp::transport::JsonRpcRequest;
use std::collections::HashMap;
use std::path::PathBuf;

fn make_tool(name: &str) -> McpTool {
    McpTool {
        name: name.to_string(),
        description: "desc".to_string(),
        input_schema: serde_json::json!({}),
        version: None,
    }
}

fn make_req(method: &str, params: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: method.to_string(),
        params,
        id: serde_json::json!(1),
    }
}

// --- Handshake & Capabilities ---

#[tokio::test]
async fn test_mcp_initialize() {
    let gateway = McpGateway::new();
    let res = gateway.handle_initialize();
    assert_eq!(res["serverInfo"]["name"], "Crosstalk-MCP-Hub");
    assert_eq!(res["serverInfo"]["version"], "0.1.0");
}

#[tokio::test]
async fn test_mcp_protocol_version() {
    let gateway = McpGateway::new();
    assert_eq!(gateway.handle_initialize()["protocolVersion"], "1.0");
}

#[tokio::test]
async fn test_mcp_capabilities_sampling() {
    let gateway = McpGateway::new();
    assert!(gateway.handle_initialize()["capabilities"]["sampling"].is_object());
}

#[tokio::test]
async fn test_mcp_capabilities_logging() {
    let gateway = McpGateway::new();
    assert!(gateway.handle_initialize()["capabilities"]["logging"].is_object());
}

#[tokio::test]
async fn test_mcp_capabilities_resources_list() {
    let gateway = McpGateway::new();
    assert_eq!(
        gateway.handle_initialize()["capabilities"]["resources"]["list"],
        true
    );
}

#[tokio::test]
async fn test_mcp_capabilities_prompts_list() {
    let gateway = McpGateway::new();
    assert_eq!(
        gateway.handle_initialize()["capabilities"]["prompts"]["list"],
        true
    );
}

// --- Resource & Prompt listing ---

#[tokio::test]
async fn test_resources_list_method() {
    let mut gateway = McpGateway::new();
    gateway.add_resource(McpResource {
        uri: "file:///tmp/artifact.bin".to_string(),
        name: "artifact".to_string(),
        description: Some("build artifact".to_string()),
        mime_type: Some("application/octet-stream".to_string()),
    });

    let req = make_req("resources/list", serde_json::json!({}));
    let res = gateway.handle_request("agent", req).await.unwrap();
    let resources = res["resources"].as_array().unwrap();
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0]["uri"], "file:///tmp/artifact.bin");
}

#[tokio::test]
async fn test_prompts_list_method() {
    let mut gateway = McpGateway::new();
    gateway.add_prompt_template(serde_json::json!({
        "id": "code_generation_v1",
        "description": "Generate Rust code from a spec"
    }));

    let req = make_req("prompts/list", serde_json::json!({}));
    let res = gateway.handle_request("agent", req).await.unwrap();
    let prompts = res["prompts"].as_array().unwrap();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0]["id"], "code_generation_v1");
}

// --- Tool listing ---

#[tokio::test]
async fn test_mcp_list_tools() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("test_tool"));
    gateway
        .permissions
        .tiers
        .insert("agent_1".to_string(), PermissionTier::Full);

    let res = gateway.handle_tools_list("agent_1");
    let tools = res["tools"].as_array().unwrap();
    assert!(tools.iter().any(|t| t["name"] == "test_tool"));
}

// --- Path validation ---

#[tokio::test]
async fn test_validate_path_strict_traversal_blocked() {
    let allowed = vec![PathBuf::from("/tmp/allowed")];
    let result =
        PermissionManager::validate_path_strict(std::path::Path::new("../../secret.txt"), &allowed);
    assert!(matches!(result, Err(PermissionError::PathTraversal(_))));
    assert!(result.unwrap_err().to_string().contains("../../secret.txt"));
}

#[tokio::test]
async fn test_validate_path_strict_out_of_scope_blocked() {
    let allowed = vec![PathBuf::from("/tmp/allowed")];
    let result =
        PermissionManager::validate_path_strict(std::path::Path::new("/etc/passwd"), &allowed);
    assert!(matches!(result, Err(PermissionError::PathOutOfScope(_))));
    assert!(result.unwrap_err().to_string().contains("/etc/passwd"));
}

#[tokio::test]
async fn test_validate_path_strict_valid_path_allowed() {
    let allowed = vec![PathBuf::from("/tmp/allowed")];
    let result = PermissionManager::validate_path_strict(
        std::path::Path::new("/tmp/allowed/output.txt"),
        &allowed,
    );
    assert!(result.is_ok());
}

// --- Permission enforcement ---

#[tokio::test]
async fn test_permission_read_only_blocks_write() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("cargo"));

    let req = make_req(
        "tools/call",
        serde_json::json!({
            "name": "cargo",
            "arguments": { "args": ["build", ">", "output.txt"] }
        }),
    );

    let res = gateway.handle_request("agent_ro", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Permission denied"));
}

#[tokio::test]
async fn test_permission_scoped_write() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("cargo"));

    let allowed_path = PathBuf::from("/tmp/allowed");
    gateway.permissions.tiers.insert(
        "agent_scoped".to_string(),
        PermissionTier::ScopedWrite(vec![allowed_path]),
    );

    let req_fail = make_req(
        "tools/call",
        serde_json::json!({ "name": "cargo", "arguments": { "args": ["/etc/passwd"] } }),
    );
    let res_fail = gateway.handle_request("agent_scoped", req_fail).await;
    assert!(res_fail.is_err());
    let err = res_fail.unwrap_err().to_string();
    assert!(err.contains("Permission denied"));
    assert!(err.contains("/etc/passwd"));

    let req_escape = make_req(
        "tools/call",
        serde_json::json!({ "name": "cargo", "arguments": { "args": ["../../secret"] } }),
    );
    let res_escape = gateway.handle_request("agent_scoped", req_escape).await;
    assert!(res_escape.is_err());
    assert!(res_escape.unwrap_err().to_string().contains("../../secret"));
}

// --- Agent disabling after repeated failures ---

#[tokio::test]
async fn test_agent_disabled_after_5_failures() {
    let mut mgr = PermissionManager::new();
    for _ in 0..5 {
        let _ = mgr.check_with_reason("bad_agent", "forbidden_tool", &serde_json::json!({}));
    }
    assert!(mgr.disabled_agents.contains(&"bad_agent".to_string()));

    let result = mgr.check_with_reason("bad_agent", "git", &serde_json::json!({}));
    assert!(matches!(result, Err(PermissionError::AgentDisabled(_))));
}

// --- Critical tool confirmation ---

#[tokio::test]
async fn test_critical_tool_denied_without_confirmation() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("rm"));
    gateway
        .permissions
        .tiers
        .insert("agent_full".to_string(), PermissionTier::Full);
    gateway.confirmation_override = Some(false);

    let req = make_req(
        "tools/call",
        serde_json::json!({ "name": "rm", "arguments": { "args": ["/tmp/file"] } }),
    );
    let res = gateway.handle_request("agent_full", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("not confirmed"));
}

#[tokio::test]
async fn test_critical_tool_confirmed_passes_gate() {
    let mut gateway = McpGateway::new();
    // rm is not registered; after confirmation succeeds we expect "Tool not found", not "not confirmed".
    gateway
        .permissions
        .tiers
        .insert("agent_full".to_string(), PermissionTier::Full);
    gateway.confirmation_override = Some(true);

    let req = make_req(
        "tools/call",
        serde_json::json!({ "name": "rm", "arguments": { "args": [] } }),
    );
    let res = gateway.handle_request("agent_full", req).await;
    assert!(res.is_err());
    let msg = res.unwrap_err().to_string();
    assert!(!msg.contains("not confirmed"), "should have passed confirmation gate");
    assert!(msg.contains("Tool not found"));
}

#[tokio::test]
async fn test_critical_confirmation_tier_requires_confirmation() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("deploy"));
    gateway.permissions.tiers.insert(
        "agent_cc".to_string(),
        PermissionTier::CriticalConfirmation(vec!["deploy".to_string()]),
    );
    gateway.confirmation_override = Some(false);

    let req = make_req(
        "tools/call",
        serde_json::json!({ "name": "deploy", "arguments": {} }),
    );
    let res = gateway.handle_request("agent_cc", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("not confirmed"));
}

#[tokio::test]
async fn test_critical_confirmation_tier_blocks_unlisted_tools() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("forbidden"));
    gateway.permissions.tiers.insert(
        "agent_cc".to_string(),
        PermissionTier::CriticalConfirmation(vec!["allowed_tool".to_string()]),
    );

    let req = make_req(
        "tools/call",
        serde_json::json!({ "name": "forbidden", "arguments": {} }),
    );
    let res = gateway.handle_request("agent_cc", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Permission denied"));
}

// --- Integration: denied tool recovery via permission upgrade ---

#[tokio::test]
async fn test_denied_tool_recovery_via_permission_upgrade() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("special_tool"));

    let req = make_req(
        "tools/call",
        serde_json::json!({ "name": "special_tool", "arguments": {} }),
    );
    let denied = gateway.handle_request("agent_recover", req).await;
    assert!(denied.is_err());
    assert!(denied.unwrap_err().to_string().contains("Permission denied"));

    // Upgrade to Full; permission check now passes.
    gateway
        .permissions
        .tiers
        .insert("agent_recover".to_string(), PermissionTier::Full);

    let req2 = make_req(
        "tools/call",
        serde_json::json!({ "name": "special_tool", "arguments": {} }),
    );
    // Will fail at CliBridge (binary not on PATH), but not at the permission gate.
    let res2 = gateway.handle_request("agent_recover", req2).await;
    if let Err(e) = &res2 {
        assert!(
            !e.to_string().contains("Permission denied"),
            "should be past the permission gate after upgrade"
        );
    }
}

// ── CargoBridge argument mapping ──────────────────────────────────────────────

#[test]
fn cargo_build_basic() {
    let args = CargoBridge::build(&HashMap::new());
    assert_eq!(args, vec!["build"]);
}

#[test]
fn cargo_build_release_flag() {
    let mut m = HashMap::new();
    m.insert("release".to_string(), "true".to_string());
    let args = CargoBridge::build(&m);
    assert!(args.contains(&"--release".to_string()));
}

#[test]
fn cargo_test_with_name() {
    let mut m = HashMap::new();
    m.insert("name".to_string(), "my_test".to_string());
    let args = CargoBridge::test(&m);
    assert!(args.contains(&"my_test".to_string()));
}

#[test]
fn cargo_clippy_deny_warnings() {
    let mut m = HashMap::new();
    m.insert("deny_warnings".to_string(), "true".to_string());
    let args = CargoBridge::clippy(&m);
    assert!(args.contains(&"-D".to_string()));
    assert!(args.contains(&"warnings".to_string()));
}

#[test]
fn cargo_fmt_check_flag() {
    let mut m = HashMap::new();
    m.insert("check".to_string(), "true".to_string());
    let args = CargoBridge::fmt(&m);
    assert!(args.contains(&"--check".to_string()));
}

// ── GitBridge argument mapping ────────────────────────────────────────────────

#[test]
fn git_status_short_flag() {
    let mut m = HashMap::new();
    m.insert("short".to_string(), "true".to_string());
    let args = GitBridge::status(&m);
    assert!(args.contains(&"--short".to_string()));
}

#[test]
fn git_diff_staged_flag() {
    let mut m = HashMap::new();
    m.insert("staged".to_string(), "true".to_string());
    let args = GitBridge::diff(&m);
    assert!(args.contains(&"--staged".to_string()));
}

#[test]
fn git_log_oneline_and_n() {
    let mut m = HashMap::new();
    m.insert("oneline".to_string(), "true".to_string());
    m.insert("n".to_string(), "5".to_string());
    let args = GitBridge::log(&m);
    assert!(args.contains(&"--oneline".to_string()));
    assert!(args.iter().any(|a| a.contains('5')));
}

#[test]
fn git_commit_with_message() {
    let mut m = HashMap::new();
    m.insert("message".to_string(), "fix: typo".to_string());
    let args = GitBridge::commit(&m);
    assert!(args.contains(&"-m".to_string()));
    assert!(args.contains(&"fix: typo".to_string()));
}

// ── Tool timeout disabling ────────────────────────────────────────────────────

#[tokio::test]
async fn disabled_tool_returns_error_immediately() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(make_tool("slow_tool"));
    gateway.disabled_tools.push("slow_tool".to_string());
    gateway
        .permissions
        .tiers
        .insert("agent".to_string(), PermissionTier::Full);

    let req = make_req("tools/call", serde_json::json!({ "name": "slow_tool", "arguments": {} }));
    let res = gateway.handle_request("agent", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("disabled"));
}

// ── RPC framing ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn handle_unknown_method_returns_error() {
    let mut gateway = McpGateway::new();
    let req = make_req("unknown/method", serde_json::json!({}));
    let res = gateway.handle_request("agent", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Method not found"));
}
