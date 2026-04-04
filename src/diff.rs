use crate::types::ArtifactDiff;
use similar::TextDiff;

pub struct DiffEngine;

impl DiffEngine {
    /// Generates a unified diff between old and new content.
    /// Uses the `similar` crate's built-in unified diff formatter for correctness.
    /// Stores whether the new content had a trailing newline as a prefix byte in diff_text:
    /// 'T' = trailing newline, 'N' = no trailing newline.
    #[must_use]
    pub fn generate_delta(old_content: &str, new_content: &str, version: u32) -> ArtifactDiff {
        if old_content == new_content {
            return ArtifactDiff {
                original_version: version,
                new_version: version + 1,
                diff_text: String::new(),
            };
        }

        let trail_marker = if new_content.ends_with('\n') { "T\n" } else { "N\n" };

        let old_norm = if old_content.is_empty() { "\n".to_string() } else if !old_content.ends_with('\n') { format!("{old_content}\n") } else { old_content.to_string() };
        let new_norm = if new_content.is_empty() { "\n".to_string() } else if !new_content.ends_with('\n') { format!("{new_content}\n") } else { new_content.to_string() };

        let diff = TextDiff::from_lines(&old_norm, &new_norm);
        let unified = diff
            .unified_diff()
            .context_radius(3)
            .header("old", "new")
            .to_string();

        ArtifactDiff {
            original_version: version,
            new_version: version + 1,
            diff_text: format!("{trail_marker}{unified}"),
        }
    }

    /// Reconstructs the new content by applying the unified diff patch to the base.
    #[must_use]
    pub fn apply_patch(base_content: &str, delta: &ArtifactDiff) -> String {
        if delta.diff_text.is_empty() {
            return base_content.to_string();
        }

        let (new_has_trailing_nl, diff_body) = if delta.diff_text.starts_with("T\n") {
            (true, &delta.diff_text[2..])
        } else if delta.diff_text.starts_with("N\n") {
            (false, &delta.diff_text[2..])
        } else {
            (base_content.ends_with('\n'), delta.diff_text.as_str())
        };

        let base_norm = if base_content.is_empty() { "\n".to_string() } else if !base_content.ends_with('\n') { format!("{base_content}\n") } else { base_content.to_string() };
        let base_lines: Vec<&str> = base_norm.lines().collect();
        let mut result: Vec<String> = Vec::new();
        let mut base_idx: usize = 0;

        let diff_lines: Vec<&str> = diff_body.lines().collect();
        let mut i = 0;

        while i < diff_lines.len() && !diff_lines[i].starts_with("@@") {
            i += 1;
        }

        while i < diff_lines.len() {
            let line = diff_lines[i];
            if line.starts_with("@@") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let old_part = parts[1].trim_start_matches('-');
                    let old_info: Vec<&str> = old_part.split(',').collect();
                    let hunk_start = old_info[0]
                        .parse::<usize>()
                        .unwrap_or(1)
                        .saturating_sub(1);

                    while base_idx < hunk_start && base_idx < base_lines.len() {
                        result.push(base_lines[base_idx].to_string());
                        base_idx += 1;
                    }
                }
                i += 1;
                continue;
            }

            if let Some(rest) = line.strip_prefix(' ') {
                result.push(rest.to_string());
                base_idx += 1;
            } else if let Some(rest) = line.strip_prefix('+') {
                result.push(rest.to_string());
            } else if line.starts_with('-') {
                base_idx += 1;
            } else if line == "\\ No newline at end of file" {
                // skip
            } else {
                result.push(line.to_string());
                base_idx += 1;
            }
            i += 1;
        }

        while base_idx < base_lines.len() {
            result.push(base_lines[base_idx].to_string());
            base_idx += 1;
        }

        // Handle the empty-to-content and content-to-empty cases
        if base_content.is_empty() && result.len() == 1 && result[0].is_empty() {
            result.clear();
        }
        if result.is_empty() {
            return String::new();
        }

        let mut output = result.join("\n");
        if new_has_trailing_nl && !output.ends_with('\n') {
            output.push('\n');
        } else if !new_has_trailing_nl && output.ends_with('\n') {
            output.pop();
        }
        output
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_delta_single_char_change() {
        let delta = DiffEngine::generate_delta("hello", "hallo", 0);
        assert_eq!(delta.original_version, 0);
        assert_eq!(delta.new_version, 1);
        assert!(!delta.diff_text.is_empty());
    }

    #[test]
    fn test_generate_delta_identical_strings() {
        let delta = DiffEngine::generate_delta("same", "same", 5);
        assert_eq!(delta.original_version, 5);
        assert_eq!(delta.new_version, 6);
        assert!(delta.diff_text.is_empty(), "identical strings should produce empty diff");
    }

    #[test]
    fn test_round_trip_simple() {
        assert!(DiffEngine::validate_round_trip("hello world", "hello rust"));
    }

    #[test]
    fn test_round_trip_empty_to_content() {
        assert!(DiffEngine::validate_round_trip("", "new content"));
    }

    #[test]
    fn test_round_trip_content_to_empty() {
        assert!(DiffEngine::validate_round_trip("old content", ""));
    }

    #[test]
    fn test_round_trip_both_empty() {
        assert!(DiffEngine::validate_round_trip("", ""));
    }

    #[test]
    fn test_round_trip_multiline() {
        let old = "fn main() {\n    println!(\"hello\");\n}\n";
        let new = "fn main() {\n    println!(\"goodbye\");\n    return;\n}\n";
        assert!(DiffEngine::validate_round_trip(old, new));
    }

    #[test]
    fn test_round_trip_whitespace_only_change() {
        assert!(DiffEngine::validate_round_trip("a b c", "a  b  c"));
    }

    #[test]
    fn test_reconstruct_from_history() {
        let v0 = "version zero";
        let v1 = "version one";
        let v2 = "version two";

        let d1 = DiffEngine::generate_delta(v0, v1, 0);
        let d2 = DiffEngine::generate_delta(v1, v2, 1);

        let reconstructed = DiffEngine::reconstruct_from_history(v0, &[d1, d2]);
        assert_eq!(reconstructed, v2);
    }

    #[test]
    fn test_reconstruct_five_step_history() {
        let versions = ["a", "ab", "abc", "abcd", "abcde", "abcdef"];
        let mut history = vec![];
        for i in 0..versions.len() - 1 {
            history.push(DiffEngine::generate_delta(versions[i], versions[i + 1], i as u32));
        }
        let result = DiffEngine::reconstruct_from_history(versions[0], &history);
        assert_eq!(result, "abcdef");
    }

    #[test]
    fn test_apply_patch_empty_diff() {
        let delta = ArtifactDiff {
            original_version: 0,
            new_version: 1,
            diff_text: String::new(),
        };
        assert_eq!(DiffEngine::apply_patch("unchanged", &delta), "unchanged");
    }

    #[test]
    fn test_friction_identical() {
        assert_eq!(DiffEngine::calculate_friction("same", "same"), 0.0);
    }

    #[test]
    fn test_friction_completely_different() {
        let f = DiffEngine::calculate_friction("aaa", "zzz");
        assert!(f > 0.5, "completely different strings should have high friction: {f}");
    }

    #[test]
    fn test_friction_empty_strings() {
        assert_eq!(DiffEngine::calculate_friction("", ""), 0.0);
    }
}
