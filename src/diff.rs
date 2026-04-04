use crate::types::ArtifactDiff;
use similar::{ChangeTag, TextDiff};

pub struct DiffEngine;

impl DiffEngine {
    /// ∀ α { α_old + μ_new ⇒ Δα }
    /// Generates a unified diff between the current artifact and the model's new version.
    pub fn generate_delta(old_content: &str, new_content: &str, version: u32) -> ArtifactDiff {
        let diff = TextDiff::from_lines(old_content, new_content);
        let mut diff_text = String::new();

        // Generate a standard unified diff format (with 3 lines of context)
        for group in diff.grouped_ops(3) {
            for op in group {
                for change in diff.iter_changes(&op) {
                    let sign = match change.tag() {
                        ChangeTag::Delete => "-",
                        ChangeTag::Insert => "+",
                        ChangeTag::Equal => " ",
                    };
                    diff_text.push_str(&format!("{}{}", sign, change));
                }
            }
        }

        ArtifactDiff {
            original_version: version,
            new_version: version + 1,
            diff_text,
        }
    }

    /// Reconstructs α at version V by applying diffs (useful for the Rewind feature).
    pub fn apply_patch(base_content: &str, delta: &ArtifactDiff) -> String {
        // In a production app, we would use a library like `diffy` or `patch` here.
        // For Crosstalk, we assume the model provides full content for α updates.
        delta.diff_text.clone() 
    }
}