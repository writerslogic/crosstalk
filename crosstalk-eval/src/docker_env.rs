//! Live SWE-bench environment backed by Docker.
//!
//! [`SwebenchHarnessEnv`] implements [`SweBenchEnvironment`] by dispatching
//! every tool call to a running Docker container via `docker exec`. This
//! replaces the probabilistic [`MockSweBenchEnvironment`] with real code
//! execution against the actual repository state.
//!
//! ## Container lifecycle
//!
//! SWE-bench pre-builds per-instance Docker images. Before using this adapter
//! you must start a container from the appropriate image. The image naming
//! convention (as of SWE-bench v2) is:
//!
//! ```text
//! sweb.eval.x86_64.<instance_id_sanitized>:latest
//! ```
//!
//! where `<instance_id_sanitized>` replaces `-` and `.` with `__`.
//!
//! **Quick start (single instance):**
//!
//! ```bash
//! # Build image (run once per instance)
//! python -m swebench.harness.run_evaluation \
//!     --predictions_path gold \
//!     --instance_ids "django__django-15790" \
//!     --run_id build_only
//!
//! # Start container (keeps it alive for the Rust harness)
//! docker run -d --name ct_django__django-15790 \
//!     sweb.eval.x86_64.django__django__15790:latest \
//!     tail -f /dev/null
//!
//! # Pass container name to SwebenchHarnessEnv::from_container_id(…)
//! ```
//!
//! Or use [`SwebenchHarnessEnv::spawn`] which runs the `docker run` step for you.
//!
//! ## MCP / Wasm boundary
//!
//! In production Crosstalk, tool calls pass through `McpGateway::dispatch`
//! and may execute inside a Wasmtime sandbox (`SandboxManager::execute`).
//! For SWE-bench we bypass the sandbox because the containers themselves
//! already provide isolation. The trait boundary in `SweBenchEnvironment`
//! keeps the rest of the harness decoupled from this choice.

use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use regex::Regex;
use tokio::io::AsyncWriteExt as _;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::swe_bench_runner::{EnvResponse, SweBenchEnvironment, TestRunResult};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Working directory inside every SWE-bench container.
#[allow(dead_code)]
const WORKSPACE: &str = "/testbed";

/// Timeout for general shell commands (git diff, grep, cat, …).
#[allow(dead_code)]
const EXEC_TIMEOUT_SECS: u64 = 120;

/// Timeout for patch application (git apply).
#[allow(dead_code)]
const PATCH_TIMEOUT_SECS: u64 = 30;

/// Timeout for test suite execution — pytest can be slow.
#[allow(dead_code)]
const TEST_TIMEOUT_SECS: u64 = 300;

/// Maximum bytes kept in stdout/stderr before smart truncation.
#[allow(dead_code)]
const OUTPUT_TRUNCATE_BYTES: usize = 8_000;

// ─── Internal output type ────────────────────────────────────────────────────

#[allow(dead_code)]
struct ExecOutput {
    success: bool,
    stdout: String,
    stderr: String,
    elapsed_ms: u64,
}

// ─── Public adapter ───────────────────────────────────────────────────────────

/// Live SWE-bench environment that dispatches tool calls to a Docker container.
///
/// Each instance should have its own container. The container must be
/// running before any method is called. See module-level docs for lifecycle
/// instructions.
#[allow(dead_code)]
pub struct SwebenchHarnessEnv {
    /// Docker container name or ID (e.g. `"ct_django__django-15790"`).
    container_id: String,
    /// SWE-bench instance ID (used for logging and temp-file naming).
    instance_id: String,
}

#[allow(dead_code)]
impl SwebenchHarnessEnv {
    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    /// Wrap an already-running container.
    pub fn from_container_id(
        container_id: impl Into<String>,
        instance_id: impl Into<String>,
    ) -> Self {
        Self {
            container_id: container_id.into(),
            instance_id: instance_id.into(),
        }
    }

    /// Start a new detached container from `image` and return an env bound to it.
    ///
    /// The container is named `crosstalk_<instance_id>` and started with
    /// `tail -f /dev/null` so it stays alive until [`SwebenchHarnessEnv::stop`]
    /// is called.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let env = SwebenchHarnessEnv::spawn(
    ///     "sweb.eval.x86_64.django__django__15790:latest",
    ///     "django__django-15790",
    /// ).await?;
    /// ```
    pub async fn spawn(image: &str, instance_id: &str) -> Result<Self> {
        let name = format!("crosstalk_{}", instance_id.replace(['-', '.'], "_"));

        // Remove any stale container with the same name (idempotent setup).
        let _ = Command::new("docker")
            .args(["rm", "-f", &name])
            .output()
            .await;

        let out = Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                &name,
                "-e",
                "DEBIAN_FRONTEND=noninteractive",
                image,
                "tail",
                "-f",
                "/dev/null",
            ])
            .output()
            .await
            .context("docker run failed")?;

        if !out.status.success() {
            bail!(
                "docker run exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }

        tracing::info!(image, container = %name, "Container started");
        Ok(Self::from_container_id(name, instance_id))
    }

    /// Stop and remove the container.
    pub async fn stop(self) -> Result<()> {
        let out = Command::new("docker")
            .args(["rm", "-f", &self.container_id])
            .output()
            .await
            .context("docker rm failed")?;

        if !out.status.success() {
            bail!(
                "docker rm exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        tracing::info!(container = %self.container_id, "Container stopped");
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Run `bash -c <cmd>` inside the container with a wall-clock timeout.
    ///
    /// stdout and stderr are both captured, smart-truncated, and returned in
    /// [`ExecOutput`]. The caller decides whether a non-zero exit code is an
    /// error or just an unsuccessful result.
    async fn exec(&self, cmd: &str, timeout_secs: u64) -> Result<ExecOutput> {
        let start = Instant::now();

        let run = async {
            Command::new("docker")
                .args(["exec", &self.container_id, "bash", "-c", cmd])
                .output()
                .await
                .context("docker exec failed")
        };

        let output = timeout(Duration::from_secs(timeout_secs), run)
            .await
            .map_err(|_| anyhow::anyhow!("docker exec timed out after {timeout_secs}s: {cmd}"))??;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let stdout = truncate_output(
            &String::from_utf8_lossy(&output.stdout),
            OUTPUT_TRUNCATE_BYTES,
        );
        let stderr = truncate_output(
            &String::from_utf8_lossy(&output.stderr),
            OUTPUT_TRUNCATE_BYTES,
        );

        Ok(ExecOutput {
            success: output.status.success(),
            stdout,
            stderr,
            elapsed_ms,
        })
    }

    /// Run `bash -c <cmd>` inside the container, piping `input` to stdin.
    ///
    /// Used by `apply_patch` and `file_write` to avoid embedding large
    /// content in a shell command string (and the injection risk that entails).
    async fn exec_with_stdin(
        &self,
        cmd: &str,
        input: &[u8],
        timeout_secs: u64,
    ) -> Result<ExecOutput> {
        let start = Instant::now();

        let run = async {
            let mut child = Command::new("docker")
                .args(["exec", "-i", &self.container_id, "bash", "-c", cmd])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .context("docker exec spawn failed")?;

            // Write input then close stdin so the child sees EOF.
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(input).await.context("stdin write failed")?;
                // Dropping `stdin` closes the pipe.
            }

            child
                .wait_with_output()
                .await
                .context("wait_with_output failed")
        };

        let output = timeout(Duration::from_secs(timeout_secs), run)
            .await
            .map_err(|_| {
                anyhow::anyhow!("docker exec (stdin) timed out after {timeout_secs}s")
            })??;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        Ok(ExecOutput {
            success: output.status.success(),
            stdout: truncate_output(
                &String::from_utf8_lossy(&output.stdout),
                OUTPUT_TRUNCATE_BYTES,
            ),
            stderr: truncate_output(
                &String::from_utf8_lossy(&output.stderr),
                OUTPUT_TRUNCATE_BYTES,
            ),
            elapsed_ms,
        })
    }
}

// ─── Trait implementation ─────────────────────────────────────────────────────

impl SweBenchEnvironment for SwebenchHarnessEnv {
    /// Execute an arbitrary shell command inside the container.
    ///
    /// Output is capped at [`OUTPUT_TRUNCATE_BYTES`] with head + tail kept so
    /// agents see both the start of long outputs (e.g. pip install logs) and
    /// the tail (tracebacks, assertion errors).
    async fn shell_exec(&mut self, cmd: &str) -> Result<EnvResponse> {
        // Prepend workspace cd so relative paths in the command resolve correctly.
        let full_cmd = format!("cd {WORKSPACE} && {cmd}");
        tracing::debug!(container = %self.container_id, cmd = %full_cmd, "shell_exec");

        let out = self.exec(&full_cmd, EXEC_TIMEOUT_SECS).await?;
        Ok(EnvResponse {
            success: out.success,
            stdout: out.stdout,
            stderr: out.stderr,
            elapsed_ms: out.elapsed_ms,
        })
    }

    /// Read a file from the container workspace.
    ///
    /// Paths are resolved relative to `/testbed`. Absolute paths outside
    /// `/testbed` and paths containing `..` are rejected.
    async fn file_read(&mut self, path: &str) -> Result<EnvResponse> {
        let abs = workspace_path(path)?;
        tracing::debug!(container = %self.container_id, path = %abs, "file_read");

        // Use docker exec directly (no bash -c) to avoid any shell interpretation.
        let start = Instant::now();
        let run = async {
            Command::new("docker")
                .args(["exec", &self.container_id, "cat", &abs])
                .output()
                .await
                .context("docker exec cat failed")
        };

        let output = timeout(Duration::from_secs(EXEC_TIMEOUT_SECS), run)
            .await
            .map_err(|_| anyhow::anyhow!("file_read timed out: {abs}"))??;

        Ok(EnvResponse {
            success: output.status.success(),
            stdout: truncate_output(
                &String::from_utf8_lossy(&output.stdout),
                OUTPUT_TRUNCATE_BYTES,
            ),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Write `content` to `path` inside the container workspace.
    ///
    /// Content is piped via stdin so it never touches the shell command string,
    /// eliminating injection risk from agent-generated file content.
    async fn file_write(&mut self, path: &str, content: &str) -> Result<EnvResponse> {
        let abs = workspace_path(path)?;
        shell_safe_path(&abs)?; // extra guard for single-quote safety in the cmd
        tracing::debug!(container = %self.container_id, path = %abs, bytes = content.len(), "file_write");

        // mkdir -p the parent directory, then write content via stdin.
        let cmd = format!("mkdir -p '$(dirname {abs})' && cat > '{abs}'");
        let out = self
            .exec_with_stdin(&cmd, content.as_bytes(), EXEC_TIMEOUT_SECS)
            .await?;

        Ok(EnvResponse {
            success: out.success,
            stdout: out.stdout,
            stderr: out.stderr,
            elapsed_ms: out.elapsed_ms,
        })
    }

    /// Apply a unified diff patch to the repository.
    ///
    /// The patch is piped to a temp file inside the container via stdin, then
    /// applied with `git apply`. If `git apply` rejects it (exit != 0), the
    /// stderr is returned so the Metacognition engine can diagnose the failure.
    ///
    /// Two-step approach (write then apply) lets `git apply --check` run first
    /// without consuming the pipe, and keeps the diff available for diagnostics.
    async fn apply_patch(&mut self, patch: &str) -> Result<EnvResponse> {
        tracing::debug!(
            container = %self.container_id,
            bytes     = patch.len(),
            "apply_patch"
        );

        // Use a per-instance temp path to avoid collisions if containers share a host.
        let tmp = format!(
            "/tmp/ct_patch_{}.diff",
            sanitize_for_filename(&self.instance_id)
        );
        let cmd = format!(
            "cat > '{tmp}' && \
             cd {WORKSPACE} && \
             git apply --check '{tmp}' 2>&1 && \
             git apply '{tmp}' 2>&1"
        );

        let out = self
            .exec_with_stdin(&cmd, patch.as_bytes(), PATCH_TIMEOUT_SECS)
            .await?;
        Ok(EnvResponse {
            success: out.success,
            stdout: out.stdout,
            stderr: out.stderr,
            elapsed_ms: out.elapsed_ms,
        })
    }

    /// Run the test IDs listed in `fail_to_pass` with pytest inside the container.
    ///
    /// Each test in `fail_to_pass` is checked individually: `resolved` is `true`
    /// iff every listed test appears as `PASSED` in the pytest output.
    ///
    /// The run is capped at [`TEST_TIMEOUT_SECS`] (5 minutes) to prevent the
    /// orchestrator from blocking on a hung container.
    async fn run_tests(&mut self, fail_to_pass: &[String]) -> Result<TestRunResult> {
        if fail_to_pass.is_empty() {
            return Ok(TestRunResult {
                resolved: true,
                passed: 0,
                failed: 0,
                elapsed_ms: 0,
            });
        }

        // Build the pytest invocation. Each test ID is single-quoted to handle
        // colons and brackets that are special to bash.
        let test_args: Vec<String> = fail_to_pass.iter().map(|t| format!("'{t}'")).collect();

        let cmd = format!(
            "cd {WORKSPACE} && \
             python -m pytest {args} \
             -v --tb=short --no-header -rN 2>&1",
            args = test_args.join(" ")
        );

        tracing::debug!(
            container = %self.container_id,
            n_tests   = fail_to_pass.len(),
            "run_tests"
        );

        let out = self
            .exec(&cmd, TEST_TIMEOUT_SECS)
            .await
            .context("pytest execution failed")?;

        let (passed, failed, resolved) = parse_pytest_results(&out.stdout, fail_to_pass);

        tracing::info!(
            container = %self.container_id,
            passed, failed, resolved,
            "run_tests complete"
        );

        Ok(TestRunResult {
            resolved,
            passed,
            failed,
            elapsed_ms: out.elapsed_ms,
        })
    }

    /// Reset the workspace to a clean state.
    ///
    /// Runs `git reset --hard HEAD` and `git clean -fd` to undo all agent
    /// modifications. Called by the runner at the start of each instance so
    /// a single container can be reused across the dataset.
    async fn reset(&mut self) -> Result<()> {
        let cmd = format!("cd {WORKSPACE} && git reset --hard HEAD && git clean -fd");
        let out = self.exec(&cmd, EXEC_TIMEOUT_SECS).await?;
        if !out.success {
            bail!("workspace reset failed: {}", out.stderr);
        }
        tracing::debug!(container = %self.container_id, "workspace reset");
        Ok(())
    }
}

// ─── Output processing ────────────────────────────────────────────────────────

/// Truncate `s` to approximately `max_bytes` while preserving the head (for
/// context) and the tail (for tracebacks/errors).
///
/// The format is:
/// ```text
/// <first N/2 bytes>
/// …[TRUNCATED M bytes]…
/// <last N/2 bytes>
/// ```
#[allow(dead_code)]
fn truncate_output(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let half = max_bytes / 2;
    let head_end = s.floor_char_boundary(half);
    let tail_start = s.floor_char_boundary(s.len() - half);

    let head = &s[..head_end];
    let tail = &s[tail_start..];
    let cut = tail_start - head_end;

    format!("{head}\n…[TRUNCATED {cut} bytes]…\n{tail}")
}

/// Parse pytest `-v` output and determine which `fail_to_pass` tests passed.
///
/// Handles both full verbose format (`PASSED tests/test_x.py::test_y`) and
/// the short summary format (`N passed, M failed in Xs`).
///
/// Returns `(passed_count, failed_count, resolved)` where `resolved` is `true`
/// iff every test in `fail_to_pass` appears as `PASSED` in the output.
#[allow(dead_code)]
fn parse_pytest_results(output: &str, fail_to_pass: &[String]) -> (usize, usize, bool) {
    // Individual test status lines: "PASSED path::name" / "FAILED path::name"
    static PASS_RE: OnceLock<Regex> = OnceLock::new();
    static FAIL_RE: OnceLock<Regex> = OnceLock::new();
    static SUMMARY_RE: OnceLock<Regex> = OnceLock::new();

    let pass_re = PASS_RE.get_or_init(|| Regex::new(r"^PASSED\s+(.+)$").expect("static regex"));
    let fail_re = FAIL_RE.get_or_init(|| {
        Regex::new(r"^(?:FAILED|ERROR)\s+(.+?)(?:\s+-.*)?$").expect("static regex")
    });
    let summary_re = SUMMARY_RE.get_or_init(|| Regex::new(r"(\d+) passed").expect("static regex"));

    let mut passed_ids: Vec<&str> = Vec::new();
    let mut failed_ids: Vec<&str> = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if let Some(cap) = pass_re.captures(line) {
            passed_ids.push(cap.get(1).map(|m| m.as_str()).unwrap_or(""));
        } else if let Some(cap) = fail_re.captures(line) {
            failed_ids.push(cap.get(1).map(|m| m.as_str()).unwrap_or(""));
        }
    }

    // If pytest ran in quiet mode (no per-test lines), fall back to summary.
    let (passed, failed) = if passed_ids.is_empty() && failed_ids.is_empty() {
        let p = summary_re
            .captures(output)
            .and_then(|c| c[1].parse::<usize>().ok())
            .unwrap_or(0);
        let f = fail_to_pass.len().saturating_sub(p);
        (p, f)
    } else {
        (passed_ids.len(), failed_ids.len())
    };

    // Resolved iff every fail_to_pass test appears in the PASSED set.
    let resolved = fail_to_pass.iter().all(|expected| {
        passed_ids.iter().any(|seen| {
            // Normalize: strip leading "./" and match by suffix to handle
            // slight path discrepancies between dataset and pytest output.
            seen.trim_start_matches("./")
                .ends_with(expected.trim_start_matches("./"))
                || expected
                    .trim_start_matches("./")
                    .ends_with(seen.trim_start_matches("./"))
                || *seen == expected.as_str()
        })
    });

    (passed, failed, resolved)
}

// ─── Path helpers ─────────────────────────────────────────────────────────────

/// Resolve `path` to an absolute path under `/testbed`.
///
/// Rejects `..` traversal and absolute paths outside `/testbed`.
#[allow(dead_code)]
fn workspace_path(path: &str) -> Result<String> {
    if path.contains("..") {
        bail!("path traversal rejected: {path}");
    }
    if path.starts_with("/testbed/") || path == "/testbed" {
        return Ok(path.to_owned());
    }
    if path.starts_with('/') {
        bail!("absolute path outside /testbed rejected: {path}");
    }
    Ok(format!(
        "{WORKSPACE}/{}",
        path.trim_start_matches("./").trim_start_matches('/')
    ))
}

/// Reject paths containing shell metacharacters that would break single-quote
/// escaping in `bash -c '…'` commands.
#[allow(dead_code)]
fn shell_safe_path(path: &str) -> Result<()> {
    const FORBIDDEN: &[char] = &['\'', '\\', '$', '`', '\0', '\n'];
    if path.chars().any(|c| FORBIDDEN.contains(&c)) {
        bail!("path contains shell-unsafe characters: {path}");
    }
    Ok(())
}

/// Replace characters not safe in filenames with underscores.
#[allow(dead_code)]
fn sanitize_for_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_string_unchanged() {
        let s = "hello world";
        assert_eq!(truncate_output(s, 8_000), s);
    }

    #[test]
    fn truncate_long_string_keeps_head_and_tail() {
        let s: String = "A".repeat(20_000);
        let result = truncate_output(&s, 8_000);
        assert!(result.len() < s.len());
        assert!(result.starts_with('A'));
        assert!(result.ends_with('A'));
        assert!(result.contains("TRUNCATED"));
    }

    #[test]
    fn workspace_path_relative() {
        assert_eq!(
            workspace_path("src/core.py").unwrap(),
            "/testbed/src/core.py"
        );
    }

    #[test]
    fn workspace_path_already_absolute() {
        assert_eq!(
            workspace_path("/testbed/src/core.py").unwrap(),
            "/testbed/src/core.py"
        );
    }

    #[test]
    fn workspace_path_rejects_traversal() {
        assert!(workspace_path("../etc/passwd").is_err());
    }

    #[test]
    fn workspace_path_rejects_outside_absolute() {
        assert!(workspace_path("/etc/passwd").is_err());
    }

    #[test]
    fn shell_safe_path_rejects_single_quote() {
        assert!(shell_safe_path("/testbed/it's_bad.py").is_err());
    }

    #[test]
    fn shell_safe_path_accepts_normal_path() {
        assert!(shell_safe_path("/testbed/src/core.py").is_ok());
    }

    #[test]
    fn parse_pytest_all_pass() {
        let output = "\
PASSED tests/test_core.py::test_empty_input_0
PASSED tests/test_core.py::test_basic
= 2 passed in 0.12s =";
        let tests = vec!["tests/test_core.py::test_empty_input_0".to_string()];
        let (passed, failed, resolved) = parse_pytest_results(output, &tests);
        assert_eq!(passed, 2);
        assert_eq!(failed, 0);
        assert!(resolved);
    }

    #[test]
    fn parse_pytest_one_fail() {
        let output = "\
PASSED tests/test_core.py::test_basic
FAILED tests/test_core.py::test_empty_input_0 - AssertionError
= 1 passed, 1 failed in 0.14s =";
        let tests = vec!["tests/test_core.py::test_empty_input_0".to_string()];
        let (passed, failed, resolved) = parse_pytest_results(output, &tests);
        assert_eq!(passed, 1);
        assert_eq!(failed, 1);
        assert!(!resolved);
    }

    #[test]
    fn parse_pytest_empty_output_unresolved() {
        let (_, _, resolved) = parse_pytest_results("", &["t::x".to_string()]);
        assert!(!resolved);
    }

    #[test]
    fn sanitize_for_filename_replaces_specials() {
        assert_eq!(
            sanitize_for_filename("django__django-15790"),
            "django__django-15790"
        );
        assert_eq!(sanitize_for_filename("a.b/c:d"), "a_b_c_d");
    }
}
