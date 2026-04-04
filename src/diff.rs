use crate::types::ArtifactDiff;
use similar::{ChangeTag, TextDiff};

pub struct DiffEngine;

impl DiffEngine {
    /// Generates a character-level unified diff between the current artifact and the model's new version.
    /// Uses Myers diff algorithm (via `similar`) at the character level.
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
                    group_text.push_str(&format!("{}{}", sign, change));
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
    pub fn apply_patch(base_content: &str, delta: &ArtifactDiff) -> String {
        if delta.diff_text.is_empty() {
            return base_content.to_string();
        }

        let base_chars: Vec<char> = base_content.chars().collect();
        let mut result_chars: Vec<char> = base_chars.clone();
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
                        // For character-level, we might have multiple characters per diff "line" if TextDiff grouped them,
                        // but similar's from_chars usually gives one char per change.
                        hunk_content.extend(h_line[1..].chars());
                    }
                    i += 1;
                }

                let apply_pos = (start_idx as i32 + offset) as usize;
                let actual_old_len = old_len.min(result_chars.len().saturating_sub(apply_pos));

                result_chars.splice(apply_pos..apply_pos + actual_old_len, hunk_content.clone());
                offset += hunk_content.len() as i32 - actual_old_len as i32;

                continue; // i is already advanced
            }
            i += 1;
        }

        result_chars.into_iter().collect()
    }

    /// Replays all deltas sequentially from the initial artifact state.
    pub fn reconstruct_from_history(initial_content: &str, history: &[ArtifactDiff]) -> String {
        let mut current = initial_content.to_string();
        for delta in history {
            current = Self::apply_patch(&current, delta);
        }
        current
    }

    /// Generates a delta then applies it; returns true if the result matches `new`.
    pub fn validate_round_trip(old: &str, new: &str) -> bool {
        let delta = Self::generate_delta(old, new, 0);
        let reconstructed = Self::apply_patch(old, &delta);
        reconstructed == new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_apply_patch() {
        let old = "abc";
        let new = "axc";
        let delta = DiffEngine::generate_delta(old, new, 1);
        let patched = DiffEngine::apply_patch(old, &delta);
        assert_eq!(patched, new);
    }

    #[test]
    fn test_identical_strings() {
        let text = "same";
        let delta = DiffEngine::generate_delta(text, text, 1);
        assert!(delta.diff_text.is_empty());
        let patched = DiffEngine::apply_patch(text, &delta);
        assert_eq!(patched, text);
    }

    #[test]
    fn test_empty_old() {
        let old = "";
        let new = "new";
        let delta = DiffEngine::generate_delta(old, new, 1);
        let patched = DiffEngine::apply_patch(old, &delta);
        assert_eq!(patched, new);
    }

    #[test]
    fn test_multi_char_changes() {
        let old = "The quick brown fox";
        let new = "The fast orange fox";
        assert!(DiffEngine::validate_round_trip(old, new));
    }

    #[test]
    fn test_reconstruct_from_history() {
        let v0 = "v0";
        let v1 = "v1";
        let v2 = "v2";
        let d1 = DiffEngine::generate_delta(v0, v1, 0);
        let d2 = DiffEngine::generate_delta(v1, v2, 1);
        let history = vec![d1, d2];
        let reconstructed = DiffEngine::reconstruct_from_history(v0, &history);
        assert_eq!(reconstructed, v2);
    }
}
