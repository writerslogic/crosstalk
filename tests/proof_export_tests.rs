use crosstalk::engines::verification::ProofExporter;
use std::fs;

// ----------------------------------------------------------------
// 1. to_lean4 emits the theorem keyword and the invariant name.
// ----------------------------------------------------------------
#[test]
fn to_lean4_contains_theorem_keyword() {
    let out = ProofExporter::to_lean4("my_invariant", "x is always positive").unwrap();
    assert!(out.contains("theorem"), "output must contain 'theorem'");
}

#[test]
fn to_lean4_includes_invariant_name() {
    let out = ProofExporter::to_lean4("turns_ordered", "consecutive turns increase").unwrap();
    assert!(out.contains("turns_ordered"), "output must contain the invariant name");
}

// ----------------------------------------------------------------
// 2. to_lean4_theorem produces well-structured Lean 4 syntax.
// ----------------------------------------------------------------
#[test]
fn to_lean4_theorem_valid_structure() {
    let out = ProofExporter::to_lean4_theorem(
        "trivial_lemma",
        "(h : 1 = 1) : 1 = 1",
        "h",
    )
    .unwrap();
    assert!(out.starts_with("theorem trivial_lemma"), "must start with 'theorem <name>'");
    assert!(out.contains(":="), "must include ':='");
    assert!(out.contains("h"), "proof term must appear in output");
}

// ----------------------------------------------------------------
// 3. Reject empty theorem/invariant names.
// ----------------------------------------------------------------
#[test]
fn to_lean4_rejects_empty_name() {
    assert!(ProofExporter::to_lean4("", "some sketch").is_err());
    assert!(ProofExporter::to_lean4("   ", "some sketch").is_err());
}

#[test]
fn to_lean4_theorem_rejects_empty_name() {
    assert!(ProofExporter::to_lean4_theorem("", "True", "trivial").is_err());
}

#[test]
fn to_lean4_theorem_rejects_empty_statement() {
    assert!(ProofExporter::to_lean4_theorem("foo", "", "trivial").is_err());
}

// ----------------------------------------------------------------
// 4. export_all_proofs creates Invariants.lean in the output dir.
// ----------------------------------------------------------------
#[tokio::test]
async fn export_all_proofs_creates_lean_file() {
    let dir = tempfile::tempdir().unwrap();
    ProofExporter::export_all_proofs(dir.path().to_str().unwrap()).await.unwrap();
    let path = dir.path().join("Invariants.lean");
    assert!(path.exists(), "Invariants.lean must be created");
    assert!(path.extension().and_then(|e| e.to_str()) == Some("lean"));
}

// ----------------------------------------------------------------
// 5. Exported file contains all three core invariant theorems.
// ----------------------------------------------------------------
#[tokio::test]
async fn exported_file_contains_all_three_invariants() {
    let dir = tempfile::tempdir().unwrap();
    ProofExporter::export_all_proofs(dir.path().to_str().unwrap()).await.unwrap();
    let content = fs::read_to_string(dir.path().join("Invariants.lean")).unwrap();
    assert!(content.contains("monotonic_indices"), "must export monotonic_indices");
    assert!(content.contains("hash_chain_integrity"), "must export hash_chain_integrity");
    assert!(content.contains("artifact_version_consistency"), "must export artifact_version_consistency");
}

// ----------------------------------------------------------------
// 6. Exported file has the Lean 4 namespace wrapper and imports.
// ----------------------------------------------------------------
#[tokio::test]
async fn exported_file_has_namespace_and_imports() {
    let dir = tempfile::tempdir().unwrap();
    ProofExporter::export_all_proofs(dir.path().to_str().unwrap()).await.unwrap();
    let content = fs::read_to_string(dir.path().join("Invariants.lean")).unwrap();
    assert!(content.contains("import Mathlib"), "must import Mathlib");
    assert!(content.contains("namespace Crosstalk"), "must open Crosstalk namespace");
    assert!(content.contains("end Crosstalk"), "must close Crosstalk namespace");
}

// ----------------------------------------------------------------
// 7. Lean 4 type-check via `lean` binary (skipped when not installed).
// ----------------------------------------------------------------
#[tokio::test]
async fn lean_type_check_when_available() {
    let lean_available = std::process::Command::new("lean")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !lean_available {
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    ProofExporter::export_all_proofs(dir.path().to_str().unwrap()).await.unwrap();
    let lean_file = dir.path().join("Invariants.lean");

    let status = std::process::Command::new("lean")
        .arg(lean_file.to_str().unwrap())
        .status()
        .expect("failed to invoke lean");

    assert!(status.success(), "Invariants.lean must type-check without errors");
}
