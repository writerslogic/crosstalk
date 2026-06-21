use super::*;

impl Orchestrator {
    pub(super) fn lang_to_ext(lang: &str) -> &'static str {
        match lang.to_lowercase().as_str() {
            "rust" | "rs" => "rs",
            "python" | "py" => "py",
            "javascript" | "js" => "js",
            "typescript" | "ts" => "ts",
            "go" => "go",
            "java" => "java",
            "c" => "c",
            "cpp" | "c++" => "cpp",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            "toml" => "toml",
            "markdown" | "md" => "md",
            "html" => "html",
            "css" => "css",
            "sql" => "sql",
            "bash" | "sh" => "sh",
            _ => "txt",
        }
    }

    pub(super) fn ext_to_lang(ext: &str) -> &'static str {
        match ext.to_lowercase().as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" => "javascript",
            "ts" => "typescript",
            "go" => "go",
            "java" => "java",
            "c" => "c",
            "cpp" | "cc" | "cxx" => "cpp",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            "toml" => "toml",
            "md" => "markdown",
            "html" | "htm" => "html",
            "css" => "css",
            "sql" => "sql",
            "sh" | "bash" => "bash",
            _ => "",
        }
    }

    /// Resolve (lang, name) from a fence annotation string (everything after the backticks).
    pub(super) fn parse_fence_hint(hint: &str) -> (String, String) {
        // Strip outer debug-format quotes: "rust" → rust
        let hint = hint.trim().trim_matches('"');

        // Colon separator: lang:name or lang:name:linenum
        if let Some(pos) = hint.find(':') {
            let l = hint[..pos].trim().trim_matches('"').to_string();
            let mut n = hint[pos + 1..].trim().to_string();
            // Strip trailing :suffix after filename (e.g. "file.py:71" or "file.py:ClassName.method")
            while let Some(colon) = n.rfind(':') {
                let suffix = &n[colon + 1..];
                if suffix.chars().all(|c| c.is_ascii_digit())
                    || !suffix.contains('.')
                    || suffix.starts_with(|c: char| c.is_uppercase())
                {
                    n.truncate(colon);
                } else {
                    break;
                }
            }
            if !l.is_empty() {
                return (l, n);
            }
        }

        // Looks like a bare filename with extension (e.g. `main.rs`, `src/lib.rs`)
        if hint.contains('.') && !hint.contains(' ') {
            let ext = hint.rsplit('.').next().unwrap_or("");
            let lang = Self::ext_to_lang(ext).to_string();
            return (lang, hint.to_string());
        }

        // Space separator: lang name
        if let Some(pos) = hint.find(' ') {
            let l = hint[..pos].trim().to_string();
            let n = hint[pos + 1..].trim().to_string();
            if !l.is_empty() && !n.is_empty() {
                return (l, n);
            }
        }

        // Bare lang token — name resolved later
        (hint.to_string(), String::new())
    }

    /// Check whether a line immediately preceding a fence looks like a filename hint.
    /// Matches: `### foo.rs`, `**foo.rs**`, `File: foo.rs`, `> foo.rs`
    pub(super) fn extract_pre_fence_name(line: &str) -> Option<&str> {
        let s = line
            .trim()
            .trim_start_matches('#')
            .trim_start_matches('>')
            .trim_start_matches('*')
            .trim_end_matches('*')
            .trim_end_matches(':')
            .trim();
        let s = s
            .strip_prefix("File:")
            .or_else(|| s.strip_prefix("file:"))
            .or_else(|| s.strip_prefix("Filename:"))
            .or_else(|| s.strip_prefix("filename:"))
            .map(str::trim)
            .unwrap_or(s);
        // Accept only if it looks like a filename: has extension, no whitespace
        if s.contains('.') && !s.contains(' ') && s.len() < 128 {
            Some(s)
        } else {
            None
        }
    }

    /// Try to pull a filename out of a first-line comment inside a code block.
    /// Handles: `// filename: foo.rs`, `# foo.py`, `-- name: query.sql`, `/* foo.c */`
    pub(super) fn extract_comment_filename(line: &str) -> Option<String> {
        let t = line.trim();
        let rest = if let Some(r) = t.strip_prefix("//") {
            r
        } else if let Some(r) = t.strip_prefix('#') {
            r
        } else if let Some(r) = t.strip_prefix("--") {
            r
        } else if let Some(r) = t.strip_prefix("/*").and_then(|r| r.strip_suffix("*/")) {
            r
        } else {
            return None;
        };
        let rest = rest.trim();
        let candidate = rest
            .strip_prefix("filename:")
            .or_else(|| rest.strip_prefix("file:"))
            .or_else(|| rest.strip_prefix("path:"))
            .or_else(|| rest.strip_prefix("name:"))
            .map(str::trim)
            .unwrap_or(rest);
        if candidate.contains('.')
            && !candidate.contains(' ')
            && !candidate.contains("..")
            && !candidate.is_empty()
            && candidate.len() < 200
        {
            Some(
                candidate
                    .trim_start_matches("./")
                    .trim_start_matches('/')
                    .to_string(),
            )
        } else {
            None
        }
    }

    /// Parse `[TOOL: name(args)]` directives from an agent response.
    /// Returns a vec of `(tool_name, raw_args)` pairs.
    pub(super) fn parse_tool_directives(response: &str) -> Vec<(String, String)> {
        let mut directives = Vec::new();
        for line in response.lines() {
            let line = line.trim();
            if let Some(rest) = line
                .strip_prefix("[TOOL:")
                .and_then(|s| s.strip_suffix(']'))
            {
                let rest = rest.trim();
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim().to_string();
                    let args = rest[paren + 1..].trim_end_matches(')').trim().to_string();
                    if !name.is_empty() {
                        directives.push((name, args));
                    }
                }
            }
        }
        directives
    }

    /// Execute a parsed tool directive, returning the tool output as a string.
    /// Only `memory_query` and a whitelist of safe `shell_exec` commands are allowed.
    pub(super) async fn execute_tool_directive(
        &self,
        tool_name: &str,
        args: &str,
        sigma_lock: &Arc<Mutex<ConversationState>>,
    ) -> String {
        match tool_name {
            "memory_query" => {
                let (sid, turn_idx) = {
                    let s = sigma_lock.lock().await;
                    (s.session_id.clone(), s.iteration_index)
                };
                let summary = {
                    let mut bridge = self.memory_bridge.lock().await;
                    bridge
                        .recall_relevant_summary(&sid, args, 3, turn_idx)
                        .await
                        .unwrap_or_default()
                };
                if summary.is_empty() {
                    "[memory_query] No results found.".to_string()
                } else {
                    format!("[memory_query results]:\n{summary}")
                }
            }
            "shell_exec" => {
                let trimmed = args.trim();
                if let Err(msg) = validate_shell_exec(trimmed) {
                    return format!("[shell_exec] {msg}");
                }
                let cwd = self.file_writer.root.as_path();
                match tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    tokio::process::Command::new("sh")
                        .arg("-c")
                        .arg(trimmed)
                        .current_dir(cwd)
                        .output(),
                )
                .await
                {
                    Ok(Ok(out)) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let truncated = stdout.chars().take(2000).collect::<String>();
                        if !stderr.is_empty() {
                            format!(
                                "[shell_exec output]:\n{truncated}\n[stderr]: {}",
                                stderr.chars().take(500).collect::<String>()
                            )
                        } else {
                            format!("[shell_exec output]:\n{truncated}")
                        }
                    }
                    Ok(Err(e)) => format!("[shell_exec] Error: {e}"),
                    Err(_) => "[shell_exec] Timed out after 30s".to_string(),
                }
            }
            "write_file" => {
                let Some((path_str, content)) = args.split_once('\n') else {
                    return "[write_file] Expected: path\\ncontent".to_string();
                };
                let path_str = path_str.trim();
                if path_str.contains("..")
                    || path_str.starts_with('/')
                    || path_str.starts_with('\\')
                {
                    return "[write_file] Path traversal not allowed".to_string();
                }
                let target = self.file_writer.root.join(path_str);
                if let Some(parent) = target.parent()
                    && let Err(e) = tokio::fs::create_dir_all(parent).await
                {
                    return format!("[write_file] Cannot create directory: {e}");
                }
                let canonical_root = match self.file_writer.root.canonicalize() {
                    Ok(r) => r,
                    Err(e) => return format!("[write_file] Cannot resolve workspace: {e}"),
                };
                match tokio::fs::write(&target, content).await {
                    Ok(()) => {
                        if let Ok(canonical_target) = target.canonicalize()
                            && !canonical_target.starts_with(&canonical_root)
                        {
                            if let Err(e) = tokio::fs::remove_file(&target).await {
                                tracing::error!(
                                    path = %target.display(),
                                    err = %e,
                                    "failed to remove workspace-escaping file; it may persist on disk"
                                );
                            }
                            return "[write_file] Path escapes workspace after resolution"
                                .to_string();
                        }
                        tracing::info!(path = %target.display(), bytes = content.len(), "file written via tool directive");
                        let git_msg = Self::git_stage_file(&self.file_writer.root, &target).await;
                        format!(
                            "[write_file] Wrote {} ({} bytes){}",
                            target.display(),
                            content.len(),
                            git_msg
                        )
                    }
                    Err(e) => format!("[write_file] Error: {e}"),
                }
            }
            "read_file" => {
                let path_str = args.trim();
                if path_str.contains("..")
                    || path_str.starts_with('/')
                    || path_str.starts_with('\\')
                {
                    return "[read_file] Path traversal not allowed".to_string();
                }
                let target = self.file_writer.root.join(path_str);
                if let Ok(canonical) = target.canonicalize()
                    && let Ok(canonical_root) = self.file_writer.root.canonicalize()
                    && !canonical.starts_with(&canonical_root)
                {
                    return "[read_file] Path escapes workspace".to_string();
                }
                match tokio::fs::read_to_string(&target).await {
                    Ok(content) => {
                        let truncated: String = content.chars().take(4000).collect();
                        format!("[read_file {}]:\n{truncated}", target.display())
                    }
                    Err(e) => format!("[read_file] Error: {e}"),
                }
            }
            other => format!("[TOOL] Unknown tool: {other}"),
        }
    }

    pub(super) async fn git_stage_file(repo_root: &std::path::Path, file: &std::path::Path) -> String {
        let rel = file.strip_prefix(repo_root).unwrap_or(file);
        match tokio::process::Command::new("git")
            .args(["add", "--", &rel.display().to_string()])
            .current_dir(repo_root)
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                tracing::debug!(file = %rel.display(), "git staged");
                String::new()
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                format!(" [git add failed: {}]", stderr.trim())
            }
            Err(_) => String::new(),
        }
    }

    pub async fn git_commit_session(repo_root: &std::path::Path, session_id: &str, turn: u32) {
        let msg = format!("crosstalk: session {} turn {}", session_id, turn);
        let status = tokio::process::Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .current_dir(repo_root)
            .status()
            .await;
        let has_staged = matches!(status, Ok(s) if !s.success());
        if !has_staged {
            return;
        }
        match tokio::process::Command::new("git")
            .args(["commit", "-m", &msg])
            .current_dir(repo_root)
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                tracing::info!(session = session_id, turn, "git commit created");
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::warn!(session = session_id, err = %stderr.trim(), "git commit failed");
            }
            Err(e) => {
                tracing::warn!(session = session_id, err = %e, "git command failed");
            }
        }
    }

    pub(super) fn parse_artifacts(response: &str) -> HashMap<String, (String, String)> {
        let mut artifacts = HashMap::new();
        let all_lines: Vec<&str> = response.lines().collect();
        let mut i = 0usize;
        let mut unnamed_count = 0usize;

        while i < all_lines.len() {
            let trimmed = all_lines[i].trim();

            if !trimmed.starts_with("```") && !trimmed.starts_with("Δα:") {
                i += 1;
                continue;
            }

            // Resolve (lang, name) from the fence line itself
            let (mut lang, mut name) = if trimmed.starts_with("Δα:") {
                let parts: Vec<&str> = trimmed.splitn(3, ':').collect();
                if parts.len() < 2 {
                    i += 1;
                    continue;
                }
                let l = parts[1].trim().to_string();
                let n = if parts.len() >= 3 {
                    parts[2].trim().to_string()
                } else {
                    String::new()
                };
                (l, n)
            } else {
                let rest = trimmed.trim_start_matches('`').trim();
                if rest.is_empty() {
                    i += 1;
                    continue;
                }
                Self::parse_fence_hint(rest)
            };

            // Pre-fence hint: check the nearest non-empty line above for a filename
            if name.is_empty() {
                let hint = all_lines[..i]
                    .iter()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .and_then(|l| Self::extract_pre_fence_name(l));
                if let Some(h) = hint {
                    name = h.to_string();
                }
            }

            // Collect content lines until closing fence
            i += 1;
            let content_start = i;
            while i < all_lines.len() && !all_lines[i].trim().starts_with("```") {
                i += 1;
            }
            let mut content_lines: Vec<&str> = all_lines[content_start..i].to_vec();
            if i < all_lines.len() {
                i += 1;
            } // consume closing fence

            if content_lines.is_empty() {
                continue;
            }

            // First-line comment may carry the filename
            if name.is_empty()
                && let Some(fname) = Self::extract_comment_filename(content_lines[0])
            {
                name = fname;
                content_lines.remove(0);
                // drop optional blank separator line
                if content_lines
                    .first()
                    .map(|l| l.trim().is_empty())
                    .unwrap_or(false)
                {
                    content_lines.remove(0);
                }
            }

            // Infer lang from filename extension if still unknown
            if (lang.is_empty() || lang == "text" || lang == "plaintext")
                && !name.is_empty()
                && let Some(ext) = name.rsplit('.').next()
            {
                let inferred = Self::ext_to_lang(ext);
                if !inferred.is_empty() {
                    lang = inferred.to_string();
                }
            }

            if lang.is_empty() {
                continue;
            }

            // Synthesize a name if none found
            if name.is_empty() {
                unnamed_count += 1;
                name = format!("artifact_{}.{}", unnamed_count, Self::lang_to_ext(&lang));
            }

            // Normalize path separators
            let name = name
                .trim_start_matches("./")
                .trim_start_matches('/')
                .to_string();

            if name.contains("..") {
                i += 1;
                continue;
            }

            let content = content_lines.join("\n").trim_end().to_string();
            if !content.is_empty() {
                // Last write wins — later refinements override earlier ones
                artifacts.insert(name, (lang, content));
            }
        }
        artifacts
    }

}

/// Validate a `shell_exec` directive: it must be an allowlisted read-only
/// command, free of shell metacharacters and control characters, with every
/// path argument confined to the workspace (no absolute paths, `~` home
/// expansion, or `..` traversal). `find` is intentionally excluded because its
/// `-delete`/`-exec` flags make it destructive even when path-confined.
///
/// Returns `Err(reason)` describing the first violation, or `Ok(())` if safe.
fn validate_shell_exec(trimmed: &str) -> Result<(), String> {
    const ALLOWED_PREFIXES: &[&str] = &[
        "git log",
        "git status",
        "git diff",
        "git show",
        "cargo check",
        "cargo test",
        "cargo clippy",
        "ls ",
        "cat ",
        "head ",
        "tail ",
        "wc ",
        "grep ",
        "diff ",
        "file ",
    ];
    if !ALLOWED_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
        return Err(format!("Command not in whitelist: {trimmed}"));
    }
    const INJECTION_CHARS: &[char] = &[';', '|', '&', '$', '`', '>', '<', '(', ')'];
    if trimmed.contains(INJECTION_CHARS) {
        return Err(format!("Command contains disallowed characters: {trimmed}"));
    }
    // Newlines/tabs/etc. would let `sh -c` run additional commands; reject them
    // regardless of how the caller assembled the args.
    if trimmed.contains(|c: char| c.is_control()) {
        return Err(format!("Command contains control characters: {trimmed}"));
    }
    // Confine every path argument to the workspace: no absolute paths, no home
    // expansion, no parent-dir traversal. Flags (`-r`, `--name`) and relative
    // paths/globs are allowed.
    for tok in trimmed.split_whitespace() {
        if tok.starts_with('/') || tok.starts_with('~') || tok.split('/').any(|c| c == "..") {
            return Err(format!("Path argument escapes the workspace: {tok}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod shell_exec_validation_tests {
    use super::validate_shell_exec;

    #[test]
    fn allows_workspace_relative_read_commands() {
        assert!(validate_shell_exec("cat src/main.rs").is_ok());
        assert!(validate_shell_exec("grep -r needle src/").is_ok());
        assert!(validate_shell_exec("git log").is_ok());
        assert!(validate_shell_exec("cargo test --lib").is_ok());
        assert!(validate_shell_exec("ls ./crates").is_ok());
    }

    #[test]
    fn rejects_non_whitelisted_commands() {
        assert!(validate_shell_exec("rm -rf src").is_err());
        assert!(validate_shell_exec("curl http://x").is_err());
    }

    #[test]
    fn rejects_shell_metacharacter_chaining() {
        assert!(validate_shell_exec("git log; rm -rf /").is_err());
        assert!(validate_shell_exec("cat x && cat y").is_err());
        assert!(validate_shell_exec("grep x file | sh").is_err());
        assert!(validate_shell_exec("cat $(secret)").is_err());
    }

    #[test]
    fn rejects_control_character_injection() {
        assert!(validate_shell_exec("git log\nrm -rf /").is_err());
        assert!(validate_shell_exec("cat x\ty").is_err());
    }

    #[test]
    fn rejects_paths_outside_workspace() {
        assert!(validate_shell_exec("cat /etc/passwd").is_err());
        assert!(validate_shell_exec("cat ~/.ssh/id_rsa").is_err());
        assert!(validate_shell_exec("cat ../../etc/passwd").is_err());
        assert!(validate_shell_exec("grep secret /var/log/auth.log").is_err());
        assert!(validate_shell_exec("cat src/../../../etc/passwd").is_err());
    }

    #[test]
    fn rejects_destructive_find() {
        // `find` is no longer whitelisted (was destructive via -delete/-exec).
        assert!(validate_shell_exec("find . -delete").is_err());
    }
}
