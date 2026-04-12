use crosstalk::mcp::bridge::{CliBridge, InputSchema};
use std::collections::HashMap;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Return the absolute path of a well-known binary, or skip the test if absent.
macro_rules! require_bin {
    ($name:expr) => {
        match which::which($name) {
            Ok(p) => p,
            Err(_) => return, // binary not available in this environment; skip
        }
    };
}

// ── invoke: basic execution ───────────────────────────────────────────────────

#[test]
fn test_invoke_basic_git_version() {
    let _ = require_bin!("git");
    let result = CliBridge::invoke("git", vec!["--version".into()], None).unwrap();
    assert!(result.success);
    assert!(result.output.contains("git"));
    assert!(result.elapsed_ms < 5000);
}

#[test]
fn test_invoke_basic_cargo_version() {
    let _ = require_bin!("cargo");
    let result = CliBridge::invoke("cargo", vec!["--version".into()], None).unwrap();
    assert!(result.success);
    assert!(result.output.contains("cargo"));
}

#[test]
fn test_invoke_basic_rustfmt_version() {
    let _ = require_bin!("rustfmt");
    let result = CliBridge::invoke("rustfmt", vec!["--version".into()], None).unwrap();
    assert!(result.success);
}

// ── invoke: failure / stderr capture ─────────────────────────────────────────

#[test]
fn test_invoke_exit_code_failure_sets_success_false() {
    let false_bin = match which::which("false") {
        Ok(p) => p,
        Err(_) => std::path::PathBuf::from("/bin/false"),
    };
    let result =
        CliBridge::invoke(false_bin.to_str().unwrap(), vec![], None).unwrap();
    assert!(!result.success);
}

#[test]
fn test_invoke_captures_stderr_on_failure() {
    let _ = require_bin!("git");
    // `git log` inside a non-git temp dir emits a clear error to stderr.
    let tmp = tempfile::tempdir().unwrap();
    let mut env: HashMap<String, String> = std::env::vars().collect();
    // Override HOME to prevent git from picking up a parent repo.
    env.insert("HOME".into(), tmp.path().to_str().unwrap().into());
    env.insert(
        "PATH".into(),
        std::env::var("PATH").unwrap_or_default(),
    );
    env.insert("GIT_DIR".into(), "/nonexistent_git_dir".into());

    let result = CliBridge::invoke("git", vec!["log".into()], Some(&env)).unwrap();
    assert!(!result.success);
    assert!(result.error.is_some());
}

// ── invoke: binary not found ──────────────────────────────────────────────────

#[test]
fn test_invoke_binary_not_found_returns_err() {
    let result = CliBridge::invoke("__nonexistent_binary_xyz__", vec![], None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_invoke_absolute_path_not_found_returns_err() {
    let result = CliBridge::invoke("/nonexistent/path/to/tool", vec![], None);
    assert!(result.is_err());
}

// ── invoke: absolute path ─────────────────────────────────────────────────────

#[test]
fn test_invoke_absolute_binary_path() {
    let git_path = require_bin!("git");
    let result =
        CliBridge::invoke(git_path.to_str().unwrap(), vec!["--version".into()], None)
            .unwrap();
    assert!(result.success);
    assert!(result.output.contains("git"));
}

// ── invoke: env injection / isolation ────────────────────────────────────────

#[test]
fn test_invoke_env_override_injects_vars() {
    // Use `printenv` to verify that CROSSTALK_TEST_VAR reaches the child process.
    let printenv = match which::which("printenv") {
        Ok(p) => p,
        Err(_) => std::path::PathBuf::from("/usr/bin/printenv"),
    };
    if !printenv.exists() {
        return;
    }

    let mut env = HashMap::new();
    env.insert("CROSSTALK_TEST_VAR".into(), "hello_from_nix".into());
    // printenv needs PATH to do nothing in particular, but we supply the binary
    // by absolute path so PATH is irrelevant here.

    let result = CliBridge::invoke(
        printenv.to_str().unwrap(),
        vec!["CROSSTALK_TEST_VAR".into()],
        Some(&env),
    )
    .unwrap();
    assert!(result.success, "stderr: {:?}", result.error);
    assert!(result.output.trim() == "hello_from_nix");
}

#[test]
fn test_invoke_env_override_additive() {
    // env_override is additive: supplied vars are merged into the parent env.
    let printenv = match which::which("printenv") {
        Ok(p) => p,
        Err(_) => std::path::PathBuf::from("/usr/bin/printenv"),
    };
    if !printenv.exists() {
        return;
    }

    let mut env = HashMap::new();
    env.insert("CROSSTALK_ADDED_VAR".into(), "present".into());

    let result = CliBridge::invoke(
        printenv.to_str().unwrap(),
        vec!["CROSSTALK_ADDED_VAR".into()],
        Some(&env),
    )
    .unwrap();
    assert_eq!(result.output.trim(), "present");
}

#[test]
fn test_invoke_env_override_custom_path_resolution() {
    // When env_override provides PATH, CliBridge must find the binary in that PATH.
    let tmp = tempfile::tempdir().unwrap();
    let bin_dir = tmp.path().join("bin");
    std::fs::create_dir_all(&bin_dir).unwrap();

    // Write a tiny shell script that acts as a stub binary.
    let stub = bin_dir.join("crosstalk_stub");
    std::fs::write(&stub, "#!/bin/sh\necho stub_ok\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    let mut env = HashMap::new();
    env.insert("PATH".into(), bin_dir.to_str().unwrap().into());

    let result =
        CliBridge::invoke("crosstalk_stub", vec![], Some(&env)).unwrap();
    assert!(result.success, "stderr: {:?}", result.error);
    assert!(result.output.trim() == "stub_ok");
}

// ── invoke_with_timeout: success ──────────────────────────────────────────────

#[tokio::test]
async fn test_invoke_with_timeout_completes_successfully() {
    let _ = require_bin!("git");
    let result =
        CliBridge::invoke_with_timeout("git", vec!["--version".into()], None, 10)
            .await
            .unwrap();
    assert!(result.success);
    assert!(result.output.contains("git"));
}

// ── invoke_with_timeout: timeout fires ───────────────────────────────────────

#[tokio::test]
async fn test_invoke_with_timeout_exceeded_returns_err() {
    let sleep_bin = match which::which("sleep") {
        Ok(p) => p,
        Err(_) => std::path::PathBuf::from("/bin/sleep"),
    };
    if !sleep_bin.exists() {
        return;
    }

    let err = CliBridge::invoke_with_timeout(
        sleep_bin.to_str().unwrap(),
        vec!["10".into()],
        None,
        1, // 1-second timeout against a 10-second sleep
    )
    .await
    .unwrap_err();

    assert!(
        err.to_string().contains("TimeoutError"),
        "unexpected error: {}",
        err
    );
}

// ── invoke_with_timeout: binary not found ─────────────────────────────────────

#[tokio::test]
async fn test_invoke_with_timeout_binary_not_found() {
    let err =
        CliBridge::invoke_with_timeout("__no_such_binary__", vec![], None, 5)
            .await
            .unwrap_err();
    assert!(err.to_string().contains("not found"));
}

// ── validate_schema ───────────────────────────────────────────────────────────

#[test]
fn test_validate_schema_cargo() {
    let _ = require_bin!("cargo");
    let schema = CliBridge::validate_schema("cargo").unwrap();
    assert!(
        !schema.flags.is_empty(),
        "cargo --help should yield at least one flag"
    );
    // cargo universally exposes --version and --help
    let has_version = schema.flags.iter().any(|f| f.long == "version");
    let has_help = schema.flags.iter().any(|f| f.long == "help");
    assert!(has_version || has_help, "expected --version or --help in cargo schema");
}

#[test]
fn test_validate_schema_git() {
    let _ = require_bin!("git");
    let schema = CliBridge::validate_schema("git").unwrap();
    assert!(!schema.flags.is_empty());
    // git --help always lists --version
    let has_version = schema.flags.iter().any(|f| f.long == "version");
    assert!(has_version, "expected --version flag in git schema; got: {:?}", schema.flags);
}

#[test]
fn test_validate_schema_rustfmt() {
    let _ = require_bin!("rustfmt");
    let schema = CliBridge::validate_schema("rustfmt").unwrap();
    assert!(!schema.flags.is_empty());
}

#[test]
fn test_validate_schema_to_json_schema() {
    let schema = InputSchema {
        flags: vec![
            crosstalk::mcp::bridge::FlagDef {
                long: "verbose".into(),
                short: Some("v".into()),
                description: "Enable verbose output".into(),
                takes_value: false,
            },
            crosstalk::mcp::bridge::FlagDef {
                long: "output".into(),
                short: Some("o".into()),
                description: "Output file".into(),
                takes_value: true,
            },
        ],
        raw_usage: "USAGE: tool [OPTIONS]".into(),
    };
    let json = schema.to_json_schema();
    let desc = json["properties"]["args"]["description"].as_str().unwrap();
    assert!(desc.contains("--verbose"));
    assert!(desc.contains("--output <VALUE>"));
}

#[test]
fn test_validate_schema_binary_not_found() {
    let err = CliBridge::validate_schema("__no_such_binary__").unwrap_err();
    assert!(err.to_string().contains("not found"));
}
