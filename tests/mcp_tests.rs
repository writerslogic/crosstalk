use crosstalk::mcp::gateway::{McpGateway, McpTool, PermissionTier};
use crosstalk::mcp::transport::JsonRpcRequest;
use std::path::PathBuf;

#[tokio::test]
async fn test_mcp_initialize() {
    let gateway = McpGateway::new();
    let res = gateway.handle_initialize();
    assert_eq!(res["serverInfo"]["name"], "Crosstalk-MCP-Hub");
}

#[tokio::test]
async fn test_mcp_list_tools() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(McpTool {
        name: "test_tool".to_string(),
        description: "desc".to_string(),
        input_schema: serde_json::json!({}),
    });

    gateway
        .permissions
        .tiers
        .insert("agent_1".to_string(), PermissionTier::Full);

    let res = gateway.handle_tools_list("agent_1");
    let tools = res["tools"].as_array().unwrap();
    assert!(tools.iter().any(|t| t["name"] == "test_tool"));
}

#[tokio::test]
async fn test_permission_read_only_blocks_write() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(McpTool {
        name: "cargo".to_string(),
        description: "desc".to_string(),
        input_schema: serde_json::json!({}),
    });

    // Read-only is default
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "cargo",
            "arguments": { "args": ["build", ">", "output.txt"] }
        }),
        id: serde_json::json!(1),
    };

    let res = gateway.handle_request("agent_ro", req).await;
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("Permission denied"));
}

#[tokio::test]
async fn test_permission_scoped_write() {
    let mut gateway = McpGateway::new();
    gateway.register_tool(McpTool {
        name: "cargo".to_string(),
        description: "desc".to_string(),
        input_schema: serde_json::json!({}),
    });

    let allowed_path = PathBuf::from("/tmp/allowed");
    gateway.permissions.tiers.insert(
        "agent_scoped".to_string(),
        PermissionTier::ScopedWrite(vec![allowed_path]),
    );

    // Try to access unauthorized path
    let req_fail = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "cargo",
            "arguments": { "args": ["/etc/passwd"] }
        }),
        id: serde_json::json!(1),
    };
    let res_fail = gateway.handle_request("agent_scoped", req_fail).await;
    assert!(res_fail.is_err());

    // Escape via ..
    let req_escape = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: serde_json::json!({
            "name": "cargo",
            "arguments": { "args": ["../../secret"] }
        }),
        id: serde_json::json!(1),
    };
    let res_escape = gateway.handle_request("agent_scoped", req_escape).await;
    assert!(res_escape.is_err());
}
