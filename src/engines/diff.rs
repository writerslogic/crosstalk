use crate::types::artifact::ArtifactDiff;
use similar::{ChangeTag, TextDiff};
use std::fmt::Write;

pub struct DiffEngine;

impl DiffEngine {
    pub fn calculate_similarity(a: &str, b: &str) -> f64 {
        if a == b { return 1.0; }
        let emb_a = crate::engines::memory::embed_text(a);
        let emb_b = crate::engines::memory::embed_text(b);
        crate::engines::memory::cosine_sim(&emb_a, &emb_b) as f64
    }

    pub fn generate_delta(old: &str, new: &str, version: u32) -> ArtifactDiff {
        let diff = TextDiff::from_lines(old, new);
        let mut diff_text = String::new();
        for change in diff.iter_all_changes() {
            let sign = match change.tag() {
                ChangeTag::Delete => "-",
                ChangeTag::Insert => "+",
                ChangeTag::Equal => " ",
            };
            // SAFETY: write! to a String is infallible; String's fmt::Write never returns Err
            write!(diff_text, "{}{}", sign, change).unwrap();
        }
        ArtifactDiff {
            original_version: version,
            diff_text,
            new_version: version,
        }
    }

    pub fn apply_patch(base: &str, patch: &ArtifactDiff) -> String {
        let mut result = String::new();
        let base_lines: Vec<&str> = base.lines().collect();
        let mut base_idx = 0;
        for line in patch.diff_text.lines() {
            if let Some(rest) = line.strip_prefix('+') {
                result.push_str(rest);
                result.push('\n');
            } else if let Some(rest) = line.strip_prefix(' ') {
                result.push_str(rest);
                result.push('\n');
                if base_idx < base_lines.len() {
                    base_idx += 1;
                }
            } else if line.starts_with('-')
                && base_idx < base_lines.len() {
                    base_idx += 1;
                }
        }
        for line in base_lines.iter().skip(base_idx) {
            result.push_str(line);
            result.push('\n');
        }
        result
    }
}
