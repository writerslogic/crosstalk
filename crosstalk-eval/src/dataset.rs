//! GSM8K dataset loader.
//!
//! GSM8K (Grade School Math 8K) is a dataset of 8,500 high-quality math word
//! problems from OpenAI. Each entry is a JSONL record with two fields:
//!   - `question`: the problem statement
//!   - `answer`: a chain-of-thought solution ending with `#### <number>`
//!
//! Download: <https://huggingface.co/datasets/openai/gsm8k>
//! Expected file: `data/gsm8k_test.jsonl` (1,319 problems)

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    sync::OnceLock,
};

/// A single GSM8K problem with its parsed ground-truth numeric answer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MathProblem {
    pub question: String,
    /// Final numeric answer extracted from the `#### <number>` suffix.
    pub answer: f64,
}

/// Raw JSONL record shape for a GSM8K entry.
#[derive(Deserialize)]
struct GsmRecord {
    question: String,
    answer: String,
}

/// Load and parse a GSM8K JSONL file.
///
/// Lines that fail JSON parsing or lack an `#### <number>` suffix are skipped
/// with a `WARN`-level log message. Returns an error only if the file cannot
/// be opened or yields zero valid problems.
pub fn load_gsm8k(path: &Path) -> Result<Vec<MathProblem>> {
    let file =
        File::open(path).with_context(|| format!("Cannot open dataset: {}", path.display()))?;
    let reader = BufReader::new(file);
    let re = answer_regex();

    let mut problems = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("IO error reading line {line_no}"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_record(line, re) {
            Ok(p) => problems.push(p),
            Err(e) => tracing::warn!("Skipping line {line_no}: {e}"),
        }
    }

    anyhow::ensure!(
        !problems.is_empty(),
        "No valid problems found in {}",
        path.display()
    );
    Ok(problems)
}

/// Validate whether a free-text model response contains the correct answer.
///
/// Scans the response for any number token and compares it against `expected`
/// with a tolerance of `max(0.01, |expected| × 1e-6)` to handle rounding.
#[allow(dead_code)]
pub fn validate_answer(response: &str, expected: f64) -> bool {
    static NUM_RE: OnceLock<Regex> = OnceLock::new();
    let re =
        NUM_RE.get_or_init(|| Regex::new(r"-?\d[\d,]*(?:\.\d+)?").expect("static regex is valid"));
    let tol = 0.01_f64.max(expected.abs() * 1e-6);
    re.find_iter(response)
        .filter_map(|m| m.as_str().replace(',', "").parse::<f64>().ok())
        .any(|n| (n - expected).abs() <= tol)
}

/// Generate synthetic arithmetic problems for testing without the dataset file.
///
/// Produces deterministic problems of the form:
///   "A store sells {a} apples per box. {b} boxes were sold Monday.
///    {a} more were sold Tuesday. How many total?"
/// with answer = a × b + a.
pub fn synthetic_math_questions(n: usize) -> Vec<MathProblem> {
    (0..n)
        .map(|i| {
            let a = (i + 3) as f64;
            let b = (i + 7) as f64;
            MathProblem {
                question: format!(
                    "A store sells {a} apples per box and {b} boxes were sold on Monday. \
                     On Tuesday they sold {a} more apples. How many apples were sold in total?"
                ),
                answer: a * b + a,
            }
        })
        .collect()
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

fn parse_record(line: &str, re: &Regex) -> Result<MathProblem> {
    let record: GsmRecord = serde_json::from_str(line).context("Invalid JSON")?;
    let answer = extract_answer(&record.answer, re)
        .with_context(|| format!("No '#### <number>' in: {:?}", record.answer))?;
    Ok(MathProblem {
        question: record.question,
        answer,
    })
}

/// Extract the numeric answer from a GSM8K answer string.
///
/// GSM8K answers end with a line like `#### 72` or `#### 1,234`.
fn extract_answer(text: &str, re: &Regex) -> Option<f64> {
    re.captures(text)
        .and_then(|c| c[1].replace(',', "").parse::<f64>().ok())
}

fn answer_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"####\s*(-?\d[\d,]*)").expect("static regex is valid"))
}
