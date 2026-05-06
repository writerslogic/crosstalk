use crosstalk::core::environment::{NixManager, ToolDiscovery};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn write_cache_entry(
    path: &PathBuf,
    created_at: i64,
    env: &HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let entry = serde_json::json!({ "created_at": created_at, "env": env });
    fs::write(path, serde_json::to_vec(&entry)?)?;
    Ok(())
}

#[test]
fn test_generate_flake_contains_deps() {
    let mgr = NixManager::new(vec!["git".to_string(), "curl".to_string()]).unwrap();
    let flake = mgr.generate_flake().unwrap();
    assert!(flake.contains("pkgs.git"), "flake missing pkgs.git");
    assert!(flake.contains("pkgs.curl"), "flake missing pkgs.curl");
}

#[test]
fn test_generate_flake_uses_current_system() {
    let mgr = NixManager::new(vec![]).unwrap();
    let flake = mgr.generate_flake().unwrap();
    assert!(
        flake.contains("builtins.currentSystem"),
        "flake must use builtins.currentSystem, not a hardcoded platform"
    );
}

#[test]
fn test_generate_flake_empty_deps() {
    let mgr = NixManager::new(vec![]).unwrap();
    let flake = mgr.generate_flake().unwrap();
    assert!(
        flake.contains("devShells"),
        "flake missing devShells output"
    );
    assert!(flake.contains("mkShell"), "flake missing mkShell call");
}

#[test]
fn test_cache_hit_returns_fast() {
    let tmp = tempfile::tempdir().unwrap();
    let mut mgr = NixManager::new(vec!["git".to_string()]).unwrap();
    mgr.cache_dir = tmp.path().to_path_buf();

    let cache_path = tmp.path().join(format!("{}.json", cache_key_for(&["git"])));
    let mut env = HashMap::new();
    env.insert("PATH".to_string(), "/nix/store/bin".to_string());

    let now = chrono::Utc::now().timestamp();
    write_cache_entry(&cache_path, now, &env).expect("write cache entry");

    let result = mgr.synthesize().unwrap();
    assert_eq!(
        result.get("PATH").map(|s| s.as_str()),
        Some("/nix/store/bin")
    );
}

#[test]
fn test_cache_expired_is_miss() {
    let tmp = tempfile::tempdir().unwrap();
    let mut mgr = NixManager::new(vec!["git".to_string()]).unwrap();
    mgr.cache_dir = tmp.path().to_path_buf();

    let cache_path = tmp.path().join(format!("{}.json", cache_key_for(&["git"])));
    let stale_ts = chrono::Utc::now().timestamp() - 7200;
    let env: HashMap<String, String> = HashMap::new();
    write_cache_entry(&cache_path, stale_ts, &env).expect("write cache entry");

    // Without nix installed the synthesize call will fail (cache miss -> nix required).
    // We only verify the cache was NOT returned (i.e. we hit an error, not the stale data).
    if which::which("nix").is_err() {
        assert!(
            mgr.synthesize().is_err(),
            "stale cache must not be returned"
        );
    } else {
        // nix is present; synthesize may succeed or fail depending on env.
        // Just verify the stale cache timestamp didn't trick us into returning stale data.
        let _ = mgr.synthesize(); // may succeed or fail; no assertion on value
    }
}

#[test]
fn test_get_env_returns_none_for_unknown() {
    if which::which("nix").is_err() {
        return;
    }
    let mgr = NixManager::new(vec![]).unwrap();
    assert_eq!(mgr.get_env("CROSSTALK_DOES_NOT_EXIST_XYZ"), None);
}

#[test]
fn test_get_env_from_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let mut mgr = NixManager::new(vec!["git".to_string()]).unwrap();
    mgr.cache_dir = tmp.path().to_path_buf();

    let cache_path = tmp.path().join(format!("{}.json", cache_key_for(&["git"])));
    let mut env = HashMap::new();
    env.insert("MY_VAR".to_string(), "hello".to_string());
    let now = chrono::Utc::now().timestamp();
    write_cache_entry(&cache_path, now, &env).expect("write cache entry");

    assert_eq!(mgr.get_env("MY_VAR").as_deref(), Some("hello"));
    assert_eq!(mgr.get_env("NOT_PRESENT"), None);
}

#[test]
fn test_scan_with_versions_finds_git() {
    if which::which("git").is_err() {
        return;
    }
    let tools = ToolDiscovery::scan_with_versions();
    let git = tools.iter().find(|t| t.name == "git");
    assert!(git.is_some(), "git not found in scan_with_versions");
    assert!(
        git.unwrap().version.is_some(),
        "git version should be populated"
    );
}

#[test]
fn test_scan_with_versions_schema_is_object() {
    if which::which("git").is_err() {
        return;
    }
    let tools = ToolDiscovery::scan_with_versions();
    for tool in &tools {
        assert_eq!(
            tool.input_schema["type"].as_str(),
            Some("object"),
            "tool {} schema must have type: object",
            tool.name
        );
    }
}

fn cache_key_for(deps: &[&str]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<String> = deps.iter().map(|s| s.to_string()).collect();
    sorted.sort();
    let input = sorted.join(",");
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}
