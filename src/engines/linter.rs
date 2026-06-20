use crate::engines::sandbox::SandboxResult;
use crate::types::artifact::Artifact;
use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

#[derive(Debug, Default)]
pub struct LintReport {
    pub passed: bool,
    pub warnings: Vec<Diagnostic>,
    pub errors: Vec<Diagnostic>,
}

#[derive(Debug)]
pub struct ArtifactLintReport {
    pub skipped: bool,
    pub passed: bool,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
pub struct SuggestedFix {
    pub safe: bool,
    pub description: String,
    pub replacement: String,
}

// ── LintReport helpers ───────────────────────────────────────────────────────

impl LintReport {
    /// Return all diagnostics (from both `warnings` and `errors`) whose
    /// severity is >= the requested level.  Ordering: Info < Warning < Error,
    /// so asking for `Info` returns everything while `Error` returns only
    /// errors.
    pub fn filter_by_severity(&self, severity: Severity) -> Vec<&Diagnostic> {
        self.warnings
            .iter()
            .chain(self.errors.iter())
            .filter(|d| d.severity >= severity)
            .collect()
    }
}

// ── LinterGuard ──────────────────────────────────────────────────────────────

pub struct LinterGuard;

impl LinterGuard {
    pub async fn check(
        result: &SandboxResult,
        workspace_root: &str,
        nix_env: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<LintReport> {
        if result.exit_code != 0 {
            bail!(
                "Sandbox failed (exit code {}): {}",
                result.exit_code,
                result.stderr
            );
        }

        let mut errors: Vec<Diagnostic> = Vec::new();

        if !Path::new(workspace_root).is_dir() {
            errors.push(Diagnostic {
                severity: Severity::Error,
                code: None,
                message: format!("Workspace root does not exist: {}", workspace_root),
                line: None,
                column: None,
            });
            return Ok(LintReport {
                passed: false,
                errors,
                ..Default::default()
            });
        }

        let canonical_root = tokio::fs::canonicalize(workspace_root)
            .await
            .with_context(|| {
                format!("failed to canonicalize workspace root: {}", workspace_root)
            })?;

        // Only run clippy if workspace has a Cargo.toml
        if tokio::fs::try_exists(canonical_root.join("Cargo.toml"))
            .await
            .unwrap_or(false)
        {
            let mut cmd = if let Some(env) = nix_env {
                let mut c = Command::new("nix");
                c.args([
                    "develop",
                    "-c",
                    "cargo",
                    "clippy",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ]);
                for (k, v) in env {
                    c.env(k, v);
                }
                c
            } else {
                let mut c = Command::new("cargo");
                c.args(["clippy", "--all-targets", "--", "-D", "warnings"]);
                c
            };
            cmd.current_dir(&canonical_root);

            let clippy = tokio::task::spawn_blocking(move || cmd.output())
                .await
                .context("clippy spawn_blocking task failed")?
                .context("failed to run clippy")?;
            if !clippy.status.success() {
                errors.push(Diagnostic {
                    severity: Severity::Error,
                    code: None,
                    message: {
                        let raw = String::from_utf8_lossy(&clippy.stderr);
                        if raw.len() > 8192 {
                            format!("{}... (truncated)", &raw[..8192])
                        } else {
                            raw.to_string()
                        }
                    },
                    line: None,
                    column: None,
                });
            }
        }

        Ok(LintReport {
            passed: errors.is_empty(),
            errors,
            ..Default::default()
        })
    }

    /// Lint a single artifact using heuristic checks (no external tooling).
    pub async fn check_artifact(artifact: &Artifact) -> Result<ArtifactLintReport> {
        let content = &artifact.content;

        // Skip blank-only content
        if content.trim().is_empty() {
            return Ok(ArtifactLintReport {
                skipped: true,
                passed: true,
                diagnostics: vec![],
            });
        }

        // Skip comment-only content
        let non_comment_lines: Vec<&str> = content
            .lines()
            .filter(|l| {
                let trimmed = l.trim();
                !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with('#')
            })
            .collect();

        if non_comment_lines.is_empty() {
            return Ok(ArtifactLintReport {
                skipped: true,
                passed: true,
                diagnostics: vec![],
            });
        }

        let mut diagnostics = Vec::new();

        // Heuristic: detect TODO/FIXME comments
        for (i, line) in content.lines().enumerate() {
            let upper = line.to_uppercase();
            if upper.contains("TODO") || upper.contains("FIXME") {
                diagnostics.push(Diagnostic {
                    severity: Severity::Info,
                    code: Some("todo_comment".to_string()),
                    message: format!("TODO/FIXME comment: {}", line.trim()),
                    line: Some((i + 1) as u32),
                    column: None,
                });
            }
        }

        // Heuristic: detect unreachable!() / unimplemented!() / todo!()
        for (i, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains("unreachable!(")
                || trimmed.contains("unimplemented!(")
                || trimmed.contains("todo!(")
            {
                diagnostics.push(Diagnostic {
                    severity: Severity::Warning,
                    code: Some("unreachable_macro".to_string()),
                    message: format!("Potentially unreachable macro: {}", trimmed),
                    line: Some((i + 1) as u32),
                    column: None,
                });
            }
        }

        Ok(ArtifactLintReport {
            skipped: false,
            passed: diagnostics.is_empty(),
            diagnostics,
        })
    }

    /// Suggest auto-fixable repairs for diagnostics in a `LintReport`.
    pub fn suggest_fixes(report: &LintReport) -> Vec<SuggestedFix> {
        let mut fixes = Vec::new();

        for diag in report.warnings.iter().chain(report.errors.iter()) {
            let code = diag.code.as_deref().unwrap_or("");

            if code == "unused_imports" || diag.message.contains("unused import") {
                fixes.push(SuggestedFix {
                    safe: true,
                    description: format!("Remove unused import: {}", diag.message),
                    replacement: String::new(),
                });
            } else if code == "clippy::needless_return" || diag.message.contains("needless_return")
            {
                // Strip leading "return " and trailing ";"
                let expr = diag
                    .message
                    .trim()
                    .strip_prefix("return ")
                    .unwrap_or(&diag.message)
                    .trim_end_matches(';')
                    .trim()
                    .to_string();
                fixes.push(SuggestedFix {
                    safe: true,
                    description: "Remove needless return".to_string(),
                    replacement: expr,
                });
            }
        }

        fixes
    }
}
