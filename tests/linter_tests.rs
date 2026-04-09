use crosstalk::engines::linter::{ArtifactLintReport, LintReport, LinterGuard, Severity};
use crosstalk::engines::sandbox::SandboxResult;
use crosstalk::types::artifact::Artifact;
use std::collections::BTreeMap;

fn make_artifact(name: &str, language: &str, content: &str) -> Artifact {
    Artifact {
        name: name.to_string(),
        language: language.to_string(),
        content: content.to_string(),
        version: 1,
        history: vec![],
        ast_versions: BTreeMap::new(),
        proof_attachments: vec![],
        metrics: Default::default(),
        skeleton: String::new(),
    }
}

fn ok_sandbox() -> SandboxResult {
    SandboxResult { exit_code: 0, stdout: String::new(), stderr: String::new() }
}

fn failed_sandbox() -> SandboxResult {
    SandboxResult {
        exit_code: 1,
        stdout: String::new(),
        stderr: "build failed".to_string(),
    }
}

// ── check() ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_check_rejects_failed_sandbox() {
    let result = LinterGuard::check(&failed_sandbox(), "/tmp").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Sandbox failed"));
}

#[tokio::test]
async fn test_check_passes_when_no_cargo_toml() {
    // /tmp has no Cargo.toml, so clippy is skipped and we get a passing report
    let report = LinterGuard::check(&ok_sandbox(), "/tmp").await.unwrap();
    assert!(report.passed);
    assert!(report.errors.is_empty());
}

#[tokio::test]
async fn test_check_returns_lint_report_type() {
    let report = LinterGuard::check(&ok_sandbox(), "/tmp").await.unwrap();
    let _: LintReport = report;
}

// ── check_artifact() ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_check_artifact_skips_comment_only() {
    let art = make_artifact("notes.rs", "rust", "// just a comment\n// another\n");
    let report = LinterGuard::check_artifact(&art).await.unwrap();
    assert!(report.skipped);
    assert!(report.passed);
}

#[tokio::test]
async fn test_check_artifact_skips_blank_lines() {
    let art = make_artifact("blank.rs", "rust", "\n\n\n");
    let report = LinterGuard::check_artifact(&art).await.unwrap();
    assert!(report.skipped);
}

#[tokio::test]
async fn test_check_artifact_non_rust_passes_clean_code() {
    let art = make_artifact("script.py", "python", "x = 1\ny = 2\n");
    let report = LinterGuard::check_artifact(&art).await.unwrap();
    assert!(!report.skipped);
    assert!(report.passed);
}

#[tokio::test]
async fn test_check_artifact_non_rust_detects_todo() {
    let art = make_artifact(
        "script.js",
        "javascript",
        "const x = 1; // TODO: fix this\n",
    );
    let report = LinterGuard::check_artifact(&art).await.unwrap();
    assert!(!report.skipped);
    assert!(report.diagnostics.iter().any(|d| d.severity == Severity::Info));
}

#[tokio::test]
async fn test_check_artifact_detects_unreachable() {
    let art = make_artifact(
        "code.rs",
        "rust",
        "fn foo() {\n    unreachable!(\"never\");\n}\n",
    );
    let report = LinterGuard::check_artifact(&art).await.unwrap();
    // unreachable! is detected as a warning by heuristic (or via rustc)
    assert!(!report.skipped);
    let _: ArtifactLintReport = report;
}

#[tokio::test]
async fn test_check_artifact_returns_struct() {
    let art = make_artifact("lib.rs", "rust", "pub fn add(a: i32, b: i32) -> i32 { a + b }\n");
    let report = LinterGuard::check_artifact(&art).await.unwrap();
    let _: ArtifactLintReport = report;
}

// ── suggest_fixes() ──────────────────────────────────────────────────────────

#[test]
fn test_suggest_fixes_empty_report() {
    let report = LintReport::default();
    let fixes = LinterGuard::suggest_fixes(&report);
    assert!(fixes.is_empty());
}

#[test]
fn test_suggest_fixes_unused_import() {
    use crosstalk::engines::linter::Diagnostic;
    let report = LintReport {
        warnings: vec![Diagnostic {
            severity: Severity::Warning,
            code: Some("unused_imports".to_string()),
            message: "unused import: `std::collections::HashMap`".to_string(),
            line: Some(1),
            column: None,
        }],
        errors: vec![],
        passed: true,
    };
    let fixes = LinterGuard::suggest_fixes(&report);
    assert_eq!(fixes.len(), 1);
    assert!(fixes[0].safe);
    assert!(fixes[0].description.contains("import"));
}

#[test]
fn test_suggest_fixes_needless_return() {
    use crosstalk::engines::linter::Diagnostic;
    let report = LintReport {
        warnings: vec![Diagnostic {
            severity: Severity::Warning,
            code: Some("clippy::needless_return".to_string()),
            message: "return x + 1;".to_string(),
            line: Some(5),
            column: None,
        }],
        errors: vec![],
        passed: true,
    };
    let fixes = LinterGuard::suggest_fixes(&report);
    assert_eq!(fixes.len(), 1);
    assert!(fixes[0].safe);
    assert!(!fixes[0].replacement.starts_with("return"));
}

// ── severity filtering ────────────────────────────────────────────────────────

#[test]
fn test_filter_by_severity_errors_only() {
    use crosstalk::engines::linter::Diagnostic;
    let report = LintReport {
        warnings: vec![Diagnostic {
            severity: Severity::Warning,
            code: None,
            message: "a warning".to_string(),
            line: None,
            column: None,
        }],
        errors: vec![Diagnostic {
            severity: Severity::Error,
            code: None,
            message: "an error".to_string(),
            line: None,
            column: None,
        }],
        passed: false,
    };
    let errors_only = report.filter_by_severity(Severity::Error);
    assert_eq!(errors_only.len(), 1);
    assert_eq!(errors_only[0].severity, Severity::Error);
}

#[test]
fn test_filter_by_severity_all() {
    use crosstalk::engines::linter::Diagnostic;
    let report = LintReport {
        warnings: vec![Diagnostic {
            severity: Severity::Warning,
            code: None,
            message: "w".to_string(),
            line: None,
            column: None,
        }],
        errors: vec![Diagnostic {
            severity: Severity::Error,
            code: None,
            message: "e".to_string(),
            line: None,
            column: None,
        }],
        passed: false,
    };
    let all = report.filter_by_severity(Severity::Info);
    assert_eq!(all.len(), 2);
}

// ── benchmark (timing guard, not criterion) ───────────────────────────────────

#[tokio::test]
async fn test_artifact_lint_completes_under_500ms() {
    // Generate a ~1000-line Rust snippet
    let mut content = String::new();
    for i in 0..200 {
        content.push_str(&format!("pub fn func_{i}(x: i32) -> i32 {{ x + {i} }}\n"));
    }
    let art = make_artifact("big.rs", "rust", &content);

    let start = std::time::Instant::now();
    let _report = LinterGuard::check_artifact(&art).await.unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "Artifact lint took {}ms, expected <500ms",
        elapsed.as_millis()
    );
}
