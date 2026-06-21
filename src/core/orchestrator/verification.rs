use super::*;

impl Orchestrator {
    pub(super) async fn run_verification(
        &self,
        artifacts: &BTreeMap<String, Arc<Artifact>>,
        diffs: &[(String, crate::types::artifact::ArtifactDiff)],
    ) -> Vec<(String, String, bool)> {
        let workspace = &self.file_writer.root;
        let modified_names: std::collections::HashSet<&str> =
            diffs.iter().map(|(name, _)| name.as_str()).collect();
        let modified_artifacts: Vec<&Arc<Artifact>> = artifacts
            .values()
            .filter(|a| modified_names.contains(a.name.as_str()))
            .filter(|a| !a.name.contains(':'))
            .collect();
        let mut results = Vec::new();
        let tool_sets: Vec<(&str, serde_json::Value)> = {
            let mut tools = Vec::new();
            let has_rust = modified_artifacts
                .iter()
                .any(|a| matches!(a.language.to_lowercase().as_str(), "rust" | "rs"));
            if has_rust && workspace.join("Cargo.toml").exists() {
                tools.push(("cargo", serde_json::json!({"args": ["check"]})));
                tools.push((
                    "cargo",
                    serde_json::json!({"args": ["test", "--no-fail-fast"]}),
                ));
            }
            for a in modified_artifacts
                .iter()
                .filter(|a| a.language.to_lowercase() == "python" && a.name.ends_with(".py"))
            {
                tools.push((
                    "python3",
                    serde_json::json!({"args": ["-m", "py_compile", &a.name]}),
                ));
            }
            tools
        };
        for (cmd, args) in tool_sets {
            let label = format!(
                "{} {}",
                cmd,
                args["args"]
                    .as_array()
                    .map(|a| a
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" "))
                    .unwrap_or_default()
            );
            match self.tool_call("orchestrator", cmd, args).await {
                Ok(result) => {
                    let is_error = result
                        .get("isError")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let output = result.get("content")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                        .and_then(|e| e.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or_else(|| {
                            tracing::warn!(tool = %label, "tool result JSON missing content[0].text field");
                            ""
                        })
                        .to_string();
                    results.push((label, output, !is_error));
                }
                Err(e) => {
                    results.push((label, format!("{e}"), false));
                }
            }
        }
        results
    }

    pub(super) async fn build_differential_prompt(&self, sigma: &ConversationState) -> String {
        let mut p = String::with_capacity(32_000);

        // Use evolved template as base if available, preferring select_for_agent over cache lookup
        let task_category = sigma
            .turns
            .last()
            .and_then(|t| t.task_category)
            .unwrap_or(TaskCategory::Research);
        let category_key = format!("{task_category:?}");

        // Try select_for_agent first; fall back to template_cache if population is empty
        let selected_tmpl: Option<crate::types::intelligence::PromptTemplate> = {
            let evolver = self.prompt_evolver.lock().await;
            let obs = self.observer.lock().await;
            if !evolver.population.is_empty() {
                let first_agent = self.agents.first().map(|a| a.name()).unwrap_or("unknown");
                evolver
                    .select_for_agent(first_agent, &obs.elo_ratings)
                    .cloned()
            } else {
                drop(evolver);
                drop(obs);
                let cache = self.template_cache.read().await;
                cache.get(&category_key).cloned()
            }
        };

        if let Some(tmpl) = selected_tmpl {
            let vars = std::collections::BTreeMap::from([
                ("session_id".to_string(), sigma.session_id.clone()),
                ("turn_index".to_string(), sigma.iteration_index.to_string()),
            ]);
            if let Ok(rendered) = tmpl.render(&vars) {
                // Record which template was used so the post-turn block can feed back quality
                *self.last_rendered_template_id.lock().await = Some(tmpl.id.clone());
                // Keep cache up to date for fallback on subsequent turns
                self.template_cache.write().await.insert(category_key, tmpl);
                p.push_str(&rendered);
                p.push('\n');
            }
        } else {
            *self.last_rendered_template_id.lock().await = None;
        }

        // Inject prior session lessons on the first turn of a new session (Task 7)
        if sigma.iteration_index <= 1 {
            let lessons = self.prior_lessons.lock().await;
            if !lessons.is_empty() {
                p.push_str("\n[PRIOR SESSION CONTEXT]:\n");
                for lesson in lessons.iter().take(2) {
                    crate::log_warn!(
                        writeln!(
                            p,
                            "- Session summary: \"{}\"\n  Outcome: {} | Winner: {} | Turns: {} | Topologies: {}",
                            lesson.task_summary,
                            lesson.final_outcome,
                            lesson.winning_model,
                            lesson.turn_count,
                            lesson.topology_sequence.join(" → ")
                        ),
                        "Failed to write prior lesson to prompt"
                    );
                }
                p.push('\n');
            }
        }

        if p.is_empty() {
            p.push_str(&format!("Project Context: {}\n\n", sigma.session_id));
        }

        if let Some(last_turn) = sigma.turns.last().filter(|t| t.model_id != "User") {
            crate::log_warn!(
                writeln!(
                    p,
                    "[ITERATION {}/prior] Prior consensus (turn {}, convergence {:.0}%):\n{}\n\nBuild on this analysis. Add depth, fix gaps, and refine. Do NOT repeat the same points verbatim.\n",
                    sigma.iteration_index,
                    last_turn.index,
                    sigma.completion_probability * 100.0,
                    Self::truncate_str(&last_turn.content, 2000),
                ),
                "Failed to write prior turn summary"
            );
        }

        p.push_str("Artifacts (Semantic Skeleton + Active Nodes):\n");

        let artifact_count = sigma.artifacts.len();
        let total_budget: usize = if artifact_count > 10 {
            12_000
        } else if artifact_count > 5 {
            18_000
        } else {
            30_000
        };
        let overhead = 2_000;
        let artifact_budget = if artifact_count == 0 {
            total_budget
        } else {
            (total_budget - overhead) / artifact_count
        };

        for artifact in sigma.artifacts.values() {
            p.push_str(&format!(
                "--- Artifact: {} [v{}] ({}) ---\n",
                artifact.name, artifact.version, artifact.language
            ));
            if artifact.version == 0 {
                let content = &artifact.content;
                if !artifact.skeleton.is_empty() && content.len() > artifact_budget {
                    p.push_str("Skeleton:\n");
                    p.push_str(&artifact.skeleton);
                    let remaining = artifact_budget.saturating_sub(artifact.skeleton.len());
                    if remaining > 200 {
                        crate::log_warn!(
                            writeln!(
                                p,
                                "\nKey excerpt ({} of {} chars):",
                                remaining,
                                content.len()
                            ),
                            "Failed to write key excerpt header"
                        );
                        p.push_str(&content[..remaining.min(content.len())]);
                    }
                } else if content.len() <= artifact_budget {
                    p.push_str("Full Content:\n");
                    p.push_str(content);
                } else {
                    p.push_str("Content (truncated):\n");
                    p.push_str(&content[..artifact_budget.min(content.len())]);
                    crate::log_warn!(
                        writeln!(
                            p,
                            "\n... ({} more chars)",
                            content.len().saturating_sub(artifact_budget)
                        ),
                        "Failed to write truncation notice"
                    );
                }
            } else {
                p.push_str("Skeleton:\n");
                p.push_str(&artifact.skeleton);
                p.push_str("\nActive Nodes (Full Content):\n");
                let mut active_node_ids = std::collections::HashSet::new();
                for turn in sigma.turns.iter().rev().take(2) {
                    for (name, _diff) in &turn.diffs {
                        if name == &artifact.name {
                            let changed_nodes = AstValidator::identify_changed_nodes(
                                "",
                                &artifact.content,
                                &artifact.language,
                            );
                            for id in changed_nodes {
                                active_node_ids.insert(id);
                            }
                        }
                    }
                }
                let nodes = AstValidator::extract_nodes(&artifact.content, &artifact.language);
                for id in active_node_ids {
                    if let Some(content) = nodes.get(&id) {
                        p.push_str(&format!("Node {}:\n{}\n", id, content));
                    }
                }
            }
            if let Some(last_diff) = artifact.history.last() {
                p.push_str("\nMost Recent Δα:\n");
                p.push_str(&last_diff.diff_text);
            }
            p.push('\n');
        }
        p.push_str("\nRecent History (compressed):\n");
        for t in sigma.turns.iter().rev().take(5).rev() {
            let signals = ReasoningEngine::extract_signals(&t.content);
            let outcome_tag = format!("{:?}", t.outcome);
            let decisions = if signals.decisions.is_empty() {
                String::new()
            } else {
                format!(" decisions=[{}]", signals.decisions.join("; "))
            };
            let problems = if signals.problems.is_empty() {
                String::new()
            } else {
                format!(" problems=[{}]", signals.problems.join("; "))
            };
            let questions = if signals.questions.is_empty() {
                String::new()
            } else {
                format!(
                    " open_questions=[{}]",
                    signals
                        .questions
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            };
            let code_count = signals.code_blocks.len();
            let code_tag = if code_count > 0 {
                format!(" code_blocks={code_count}")
            } else {
                String::new()
            };
            let certainty_tag = t
                .certainty
                .map(|c| format!(" certainty={c:.2}"))
                .unwrap_or_default();
            crate::log_warn!(
                writeln!(
                    p,
                    "Turn {} by {} ({}){}{}{}{}{}: {}",
                    t.index,
                    t.model_id,
                    outcome_tag,
                    certainty_tag,
                    decisions,
                    problems,
                    questions,
                    code_tag,
                    Self::truncate_str(&t.content, 150),
                ),
                "Failed to write history summary"
            );
        }
        if !sigma.last_verification.is_empty() {
            p.push_str("\nVerification Results from Last Turn:\n");
            for (tool, output, passed) in &sigma.last_verification {
                let status = if *passed { "PASS" } else { "FAIL" };
                let snippet = Self::truncate_str(output, 300);
                crate::log_warn!(
                    writeln!(p, "  {} [{}]: {}", tool, status, snippet),
                    "Failed to write verification result"
                );
            }
        }

        if !sigma.last_tool_outputs.is_empty() {
            p.push_str("\nTool Results from Previous Turn:\n");
            for (name, output) in &sigma.last_tool_outputs {
                crate::log_warn!(
                    writeln!(p, "  [{}]: {}", name, Self::truncate_str(output, 500)),
                    "Failed to write tool result"
                );
            }
        }

        if sigma.mode_library.current().convergence_direction
            == crate::types::mode::ConvergenceDirection::TowardAgreement
        {
            p.push_str(
                "\n[INSTRUCTIONS]\n\
                 1. Address any open questions or problems from prior turns before proposing new changes.\n\
                 2. Ground every claim in evidence: cite specific artifact lines, prior turn numbers, or test results.\n\
                 3. Use ```lang:filename to propose code changes. Include the COMPLETE file content.\n\
                 4. When your analysis is complete and all issues are resolved, tag with 'OPTIMAL'.\n\
                 5. Prefer precise, verifiable statements over vague assertions.\n"
            );
        } else {
            p.push_str(
                "\n[INSTRUCTIONS]\n\
                 1. Explore divergent approaches. Challenge assumptions from prior turns.\n\
                 2. Use ```lang:filename to propose code changes.\n\
                 3. Surface disagreements explicitly rather than silently accepting prior consensus.\n"
            );
        }
        let mode_prefix = sigma.mode_library.current().prompt_prefix.clone();
        let base = format!("{}\n\n{}", mode_prefix, p);
        if let Some(ref signal) = sigma.novel_signal {
            format!(
                "[NOVEL SIGNAL — build on this, do not ignore it]\n{}\n\n{}",
                signal, base
            )
        } else {
            base
        }
    }

    pub(super) fn divergent_context_for_role(
        role: &str,
        artifacts: &std::collections::HashMap<
            String,
            std::sync::Arc<crate::types::artifact::Artifact>,
        >,
    ) -> String {
        if artifacts.is_empty() {
            return String::new();
        }
        let focus = match role {
            "Skeptic" | "StressTest" => "the weakest claims and most uncertain statements",
            "Architect" | "Generative" => {
                "the overall structure and what is missing or could be extended"
            }
            "Verifier" => "formal definitions, theorems, and proof sketches",
            "Historian" => "references to prior work and historical context",
            r if r.contains("Devil") => "the assumptions that the authors take for granted",
            _ => "the most important sections for your analysis",
        };
        format!(
            "\n\n[FOCUS] Your divergent context assignment: concentrate on {} in the provided documents. Other agents are focusing on different aspects.",
            focus
        )
    }

}
