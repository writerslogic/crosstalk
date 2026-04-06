use crate::engines::sandbox::SandboxResult;
use crate::types::artifact::Artifact;
use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

// ─── Severity ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

// ─── Diagnostic ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

// ─── LintReport ──────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct LintReport {
    pub warnings: Vec<Diagnostic>,
    pub errors: Vec<Diagnostic>,
    pub passed: bool,
}

impl LintReport {
    pub fn filter_by_severity(&self, min: Severity) -> Vec<&Diagnostic> {
        self.warnings
            .iter()
            .chain(self.errors.iter())
            .filter(|d| d.severity >= min)
            .collect()
    }
}

// ─── ArtifactLintReport ──────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct ArtifactLintReport {
    pub diagnostics: Vec<Diagnostic>,
    pub passed: bool,
    pub skipped: bool,
}

// ─── SuggestedFix ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SuggestedFix {
    pub description: String,
    pub original: String,
    pub replacement: String,
    pub safe: bool,
}

// ─── LinterGuard ─────────────────────────────────────────────────────────────

pub struct LinterGuard;

impl LinterGuard {
    /// Full workspace lint. Returns a `LintReport`; errors block, warnings log.
    pub async fn check(sandbox_result: &SandboxResult, workspace_dir: &str) -> Result<LintReport> {
        if sandbox_result.exit_code != 0 {
            return Err(anyhow!("Sandbox failed: {}", sandbox_result.stderr));
        }
        Self::run_clippy(workspace_dir).await
    }

    /// Lint a single artifact without a full workspace build.
    /// Fast-path: if the diff is comment-only, skip and return skipped=true.
    pub async fn check_artifact(artifact: &Artifact) -> Result<ArtifactLintReport> {
        if Self::only_comments_changed(&artifact.content) {
            return Ok(ArtifactLintReport {
                diagnostics: vec![],
                passed: true,
                skipped: true,
            });
        }

        let lang = artifact.language.to_lowercase();
        if lang != "rust" && lang != "rs" {
            // Non-Rust artifacts: run a syntax-only heuristic scan
            let diags = Self::heuristic_scan(&artifact.content);
            let passed = !diags.iter().any(|d| d.severity == Severity::Error);
            return Ok(ArtifactLintReport { diagnostics: diags, passed, skipped: false });
        }

        const MAX_ARTIFACT_BYTES: usize = 512 * 1024;
        if artifact.content.len() > MAX_ARTIFACT_BYTES {
            return Err(anyhow!("Artifact too large to lint ({} bytes)", artifact.content.len()));
        }

        let uid = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join("crosstalk-linter");
        tokio::fs::create_dir_all(&dir).await?;
        let src = dir.join(format!("artifact_{uid}.rs"));
        tokio::fs::write(&src, &artifact.content).await?;

        let rustc = Command::new("rustc")
            .args([
                "--edition",
                "2021",
                "--emit=metadata",
                "--error-format=json",
                "-A",
                "unused",
                src.to_str().unwrap_or("artifact_check.rs"),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let output = timeout(Duration::from_millis(500), rustc.wait_with_output())
            .await
            .map_err(|_| anyhow!("Artifact lint timed out (>500ms)"))??;

        let _ = tokio::fs::remove_file(&src).await;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let diags = Self::parse_json_diagnostics(&stderr);
        let passed = !diags.iter().any(|d| d.severity == Severity::Error);

        Ok(ArtifactLintReport { diagnostics: diags, passed, skipped: false })
    }

    /// Generate safe auto-fixes for common warnings in a `LintReport`.
    pub fn suggest_fixes(report: &LintReport) -> Vec<SuggestedFix> {
        let mut fixes = Vec::new();
        for diag in report.warnings.iter().chain(report.errors.iter()) {
            if let Some(fix) = Self::fix_for(diag) {
                fixes.push(fix);
            }
        }
        fixes
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    async fn run_clippy(workspace_dir: &str) -> Result<LintReport> {
        let cargo_path = Path::new(workspace_dir).join("Cargo.toml");
        if !cargo_path.exists() {
            return Ok(LintReport { warnings: vec![], errors: vec![], passed: true });
        }

        let child = Command::new("cargo")
            .current_dir(workspace_dir)
            .args([
                "clippy",
                "--all-targets",
                "--offline",
                "--message-format=json",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let output = timeout(Duration::from_secs(30), child.wait_with_output())
            .await
            .map_err(|_| anyhow!("Clippy timed out (possible malicious build.rs loop)"))??;

        let combined = String::from_utf8_lossy(&output.stdout);
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        for line in combined.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if v["reason"] != "compiler-message" {
                    continue;
                }
                let msg = &v["message"];
                let level = msg["level"].as_str().unwrap_or("note");
                let text = msg["message"].as_str().unwrap_or("").to_string();
                let code = msg["code"]["code"].as_str().map(|s| s.to_string());

                let (line_n, col_n) = msg["spans"]
                    .as_array()
                    .and_then(|spans| spans.first())
                    .map(|sp| {
                        (
                            sp["line_start"].as_u64().map(|n| n as u32),
                            sp["column_start"].as_u64().map(|n| n as u32),
                        )
                    })
                    .unwrap_or((None, None));

                let diag = Diagnostic {
                    severity: match level {
                        "error" => Severity::Error,
                        "warning" => Severity::Warning,
                        _ => Severity::Info,
                    },
                    code,
                    message: text,
                    line: line_n,
                    column: col_n,
                };

                match diag.severity {
                    Severity::Error => errors.push(diag),
                    _ => warnings.push(diag),
                }
            }
        }

        let passed = errors.is_empty();
        Ok(LintReport { warnings, errors, passed })
    }

    fn parse_json_diagnostics(stderr: &str) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for line in stderr.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let level = v["level"].as_str().unwrap_or("note");
                let text = v["message"].as_str().unwrap_or("").to_string();
                if text.is_empty() {
                    continue;
                }
                let code = v["code"]["code"].as_str().map(|s| s.to_string());
                let (line_n, col_n) = v["spans"]
                    .as_array()
                    .and_then(|s| s.first())
                    .map(|sp| {
                        (
                            sp["line_start"].as_u64().map(|n| n as u32),
                            sp["column_start"].as_u64().map(|n| n as u32),
                        )
                    })
                    .unwrap_or((None, None));

                diags.push(Diagnostic {
                    severity: match level {
                        "error" => Severity::Error,
                        "warning" => Severity::Warning,
                        _ => Severity::Info,
                    },
                    code,
                    message: text,
                    line: line_n,
                    column: col_n,
                });
            }
        }
        diags
    }

    /// Heuristic scan for non-Rust artifacts (JS, Python, etc.).
    fn heuristic_scan(content: &str) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for (i, line) in content.lines().enumerate() {
            let line_n = (i + 1) as u32;
            // Detect TODO/FIXME as Info
            if line.contains("TODO") || line.contains("FIXME") {
                diags.push(Diagnostic {
                    severity: Severity::Info,
                    code: Some("advisory".to_string()),
                    message: format!("TODO/FIXME marker: {}", line.trim()),
                    line: Some(line_n),
                    column: None,
                });
            }
            // Detect unreachable patterns
            if line.trim_start().starts_with("unreachable!") {
                diags.push(Diagnostic {
                    severity: Severity::Warning,
                    code: Some("unreachable".to_string()),
                    message: "Unreachable code macro found".to_string(),
                    line: Some(line_n),
                    column: None,
                });
            }
        }
        diags
    }

    /// True if the content contains only comment lines and blank lines (no code).
    fn only_comments_changed(content: &str) -> bool {
        content.lines().all(|l| {
            let t = l.trim();
            t.is_empty() || t.starts_with("//") || t.starts_with('#') || t.starts_with("/*") || t.starts_with('*')
        })
    }

    fn fix_for(diag: &Diagnostic) -> Option<SuggestedFix> {
        let code = diag.code.as_deref().unwrap_or("");
        match code {
            "unused_imports" => Some(SuggestedFix {
                description: "Remove unused import".to_string(),
                original: diag.message.clone(),
                replacement: String::new(),
                safe: true,
            }),
            "dead_code" => Some(SuggestedFix {
                description: "Prefix with underscore to suppress dead_code warning".to_string(),
                original: diag.message.clone(),
                replacement: format!("_{}", diag.message),
                safe: true,
            }),
            "clippy::needless_return" => Some(SuggestedFix {
                description: "Remove explicit return from last expression".to_string(),
                original: diag.message.clone(),
                replacement: diag.message.trim_start_matches("return ").trim_end_matches(';').to_string(),
                safe: true,
            }),
            _ => None,
        }
    }
}
