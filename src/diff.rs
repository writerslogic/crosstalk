use crate::types::ArtifactDiff;
use similar::{ChangeTag, TextDiff};
use std::fmt::Write;

pub struct DiffEngine;

impl DiffEngine {
    /// Generates a character-level unified diff between the current artifact and the model's new version.
    /// Uses Myers diff algorithm (via `similar`) at the character level.
    #[must_use]
    pub fn generate_delta(old_content: &str, new_content: &str, version: u32) -> ArtifactDiff {
        let diff = TextDiff::from_chars(old_content, new_content);
        let mut diff_text = String::new();

        // Generate a standard unified diff format (with 3 characters of context)
        for group in diff.grouped_ops(3) {
            let mut group_text = String::new();
            let mut old_range = 0..0;
            let mut new_range = 0..0;
            let mut first = true;

            for op in group {
                for change in diff.iter_changes(&op) {
                    if first {
                        old_range.start = change.old_index().unwrap_or(0);
                        new_range.start = change.new_index().unwrap_or(0);
                        first = false;
                    }
                    old_range.end = change.old_index().map_or(old_range.end, |i| i + 1);
                    new_range.end = change.new_index().map_or(new_range.end, |i| i + 1);

                    let sign = match change.tag() {
                        ChangeTag::Delete => "-",
                        ChangeTag::Insert => "+",
                        ChangeTag::Equal => " ",
                    };
                    let _ = write!(group_text, "{sign}{change}");
                }
            }

            // Add the hunk header: @@ -start,len +start,len @@
            let header = format!(
                "@@ -{},{} +{},{} @@\n",
                old_range.start,
                old_range.end - old_range.start,
                new_range.start,
                new_range.end - new_range.start
            );
            diff_text.push_str(&header);
            diff_text.push_str(&group_text);
        }

        ArtifactDiff {
            original_version: version,
            new_version: version + 1,
            diff_text,
        }
    }

    /// Reconstructs the new content by applying the character-level patch to the base.
    #[must_use]
    pub fn apply_patch(base_content: &str, delta: &ArtifactDiff) -> String {
        if delta.diff_text.is_empty() {
            return base_content.to_string();
        }

        let base_chars: Vec<char> = base_content.chars().collect();
        let mut result_chars: Vec<char> = base_chars;
        let mut offset: i32 = 0;

        let diff_lines: Vec<&str> = delta.diff_text.lines().collect();
        let mut i = 0;
        while i < diff_lines.len() {
            let line = diff_lines[i];
            if line.starts_with("@@") {
                // Parse @@ -start,len +start,len @@
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    i += 1;
                    continue;
                }

                let old_part = parts[1].trim_start_matches('-');
                let old_info: Vec<&str> = old_part.split(',').collect();
                let start_idx = old_info[0].parse::<usize>().unwrap_or(0);
                let old_len = if old_info.len() > 1 {
                    old_info[1].parse::<usize>().unwrap_or(0)
                } else {
                    1
                };

                let mut hunk_content = vec![];
                i += 1;
                while i < diff_lines.len() && !diff_lines[i].starts_with("@@") {
                    let h_line = diff_lines[i];
                    if h_line.starts_with(' ') || h_line.starts_with('+') {
                        hunk_content.extend(h_line[1..].chars());
                    }
                    i += 1;
                }

                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap
                )]
                let apply_pos = (start_idx as i32 + offset) as usize;
                let actual_old_len = old_len.min(result_chars.len().saturating_sub(apply_pos));

                result_chars.splice(apply_pos..apply_pos + actual_old_len, hunk_content.clone());
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let new_offset_change = hunk_content.len() as i32 - actual_old_len as i32;
                offset += new_offset_change;

                continue; // i is already advanced
            }
            i += 1;
        }

        result_chars.into_iter().collect()
    }

    /// Replays all deltas sequentially from the initial artifact state.
    #[must_use]
    pub fn reconstruct_from_history(initial_content: &str, history: &[ArtifactDiff]) -> String {
        let mut current = initial_content.to_string();
        for delta in history {
            current = Self::apply_patch(&current, delta);
        }
        current
    }

    /// Generates a delta then applies it; returns true if the result matches `new`.
    #[must_use]
    pub fn validate_round_trip(old: &str, new: &str) -> bool {
        let delta = Self::generate_delta(old, new, 0);
        let reconstructed = Self::apply_patch(old, &delta);
        reconstructed == new
    }

    /// Calculates the "friction" (disagreement) between two strings.
    /// Normalized to [0.0, 1.0] where 1.0 is total disagreement.
    #[must_use]
    pub fn calculate_friction(a: &str, b: &str) -> f64 {
        if a == b {
            return 0.0;
        }
        let diff = TextDiff::from_chars(a, b);
        let total_chars = a.chars().count().max(b.chars().count());
        if total_chars == 0 {
            return 0.0;
        }
        let distance = diff.ratio(); // ratio() returns similarity 0.0 to 1.0
        f64::from(1.0 - distance)
    }
}
