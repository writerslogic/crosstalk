use super::*;

impl Orchestrator {
    /// Phase 5: Synthesise raw agent responses into a single text + artifact block.
    /// Returns `(response, nash_weight_updates, stall_risk)`.
    pub(super) async fn synthesize_responses(
        &self,
        final_results: &[(String, String)],
        artifacts_snapshot: &BTreeMap<String, Arc<Artifact>>,
        weights: &BTreeMap<String, f64>,
    ) -> Result<(String, BTreeMap<String, f64>, f64)> {
        // Outlier detection: compute pairwise word-overlap similarity
        let mut outlier_penalty: HashMap<&str, f64> = HashMap::new();
        if final_results.len() >= 2 {
            let word_sets: Vec<(&str, std::collections::HashSet<&str>)> = final_results
                .iter()
                .map(|(id, text)| (id.as_str(), text.split_whitespace().collect()))
                .collect();
            for (i, (id, set_i)) in word_sets.iter().enumerate() {
                let mut sim_sum = 0.0;
                let mut count = 0;
                for (j, (_, set_j)) in word_sets.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let intersection = set_i.intersection(set_j).count() as f64;
                    let union = set_i.union(set_j).count().max(1) as f64;
                    sim_sum += intersection / union;
                    count += 1;
                }
                let mean_sim = if count > 0 {
                    sim_sum / count as f64
                } else {
                    1.0
                };
                if mean_sim < 0.3 {
                    outlier_penalty.insert(id, 0.1);
                }
            }

            // Entropy mapping (disagreement heatmap)
            let mut entropy_entries = Vec::new();
            let mut agent_artifact_proposals: std::collections::HashMap<
                String,
                std::collections::HashMap<String, String>,
            > = std::collections::HashMap::new();
            for (id, text) in final_results {
                for (art_name, (_, content)) in Self::parse_artifacts(text) {
                    agent_artifact_proposals
                        .entry(art_name)
                        .or_default()
                        .insert(id.clone(), content);
                }
            }
            for (art_name, proposals) in agent_artifact_proposals {
                let mut scores = Vec::new();
                let agents: Vec<(&String, &String)> = proposals.iter().collect();
                for (id_i, content_i) in &agents {
                    let mut dist_sum = 0.0;
                    let mut count = 0;
                    for (id_j, content_j) in &agents {
                        if id_i == id_j {
                            continue;
                        }
                        let diff =
                            similar::TextDiff::from_lines(content_i.as_str(), content_j.as_str());
                        dist_sum += 1.0 - diff.ratio();
                        count += 1;
                    }
                    let score: f64 = if count > 0 {
                        dist_sum as f64 / (count as f64).max(1.0)
                    } else {
                        0.0
                    };
                    scores.push(((*id_i).clone(), score));
                }
                entropy_entries.push(crate::types::events::EntropyEntry {
                    artifact_name: art_name,
                    scores,
                });
            }
            self.emit(crate::types::events::StreamEvent::EntropyUpdated(
                entropy_entries,
            ))
            .await?;
        }

        if final_results.len() > 1 {
            let outlier_names: Vec<&str> = outlier_penalty.keys().copied().collect();
            if outlier_names.is_empty() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "All {} agents broadly agree. Merging into consensus...\n",
                        final_results.len()
                    ),
                })
                .await?;
            } else {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "{} diverged from the group — downweighting outlier during synthesis\n",
                        outlier_names.join(", ")
                    ),
                })
                .await?;
            }
        }

        // Collective text synthesis (surprise + certainty + outlier calibrated)
        let synthesized_text = {
            let se = self.surprise_engine.lock().await;
            let text_proposals: Vec<(String, String, f64)> = final_results
                .iter()
                .map(|(id, text)| {
                    let base_w = weights.get(id).copied().unwrap_or(1.0);
                    let surprise_w = se.calibrate_weight(id, base_w);
                    let certainty = CertaintyAnalyzer::compute(text, 0.1);
                    let outlier_w = outlier_penalty.get(id.as_str()).copied().unwrap_or(1.0);
                    (id.clone(), text.clone(), surprise_w * certainty * outlier_w)
                })
                .collect();
            EnsembleEngine::merge_proposals(text_proposals, TaskCategory::Research, "")
        };

        // Collective artifact synthesis (Nash equilibrium resolution)
        let mut synthesized_artifacts = String::new();
        let mut nash_score_acc: std::collections::BTreeMap<String, (f64, u32)> =
            std::collections::BTreeMap::new();
        let artifact_proposals = Self::parse_artifacts(&synthesized_text);
        for (name, (lang, _content)) in artifact_proposals {
            let default_art = Arc::new(Artifact::default());
            let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
            let mut proposals_for_nash = Vec::new();
            for (agent_id, text) in final_results {
                let parsed = Self::parse_artifacts(text);
                if let Some((_, proposal_content)) = parsed.get(&name) {
                    let mut temp_art = (**current).clone();
                    temp_art.content = proposal_content.clone();
                    proposals_for_nash.push((agent_id.as_str(), temp_art, TurnOutcome::Compiled));
                }
            }
            if proposals_for_nash.is_empty() {
                continue;
            }
            let nash_refs: Vec<(&str, &Artifact, TurnOutcome)> = proposals_for_nash
                .iter()
                .map(|(id, art, out)| (*id, art, *out))
                .collect();
            for (agent_id, score) in
                crate::engines::consensus::NashSolver::compute_nash_scores(&nash_refs)
            {
                let e = nash_score_acc.entry(agent_id).or_insert((0.0_f64, 0_u32));
                e.0 += score;
                e.1 += 1;
            }
            let winning_content =
                crate::engines::consensus::NashSolver::resolve_with_synthesis(&nash_refs, current);
            crate::log_warn!(
                writeln!(
                    synthesized_artifacts,
                    "\n```{}:{}\n{}\n```",
                    lang, name, winning_content
                ),
                "Failed to write synthesized artifact"
            );
        }

        let nash_weight_updates: std::collections::BTreeMap<String, f64> = nash_score_acc
            .into_iter()
            .map(|(id, (sum, count))| (id, if count > 0 { sum / count as f64 } else { 0.5 }))
            .collect();

        // Stall detection
        let stall_risk = {
            let proposals_map: HashMap<String, String> = final_results
                .iter()
                .map(|(id, text)| {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    text.hash(&mut h);
                    (id.clone(), format!("{:x}", h.finish()))
                })
                .collect();
            let turn_entropy = if final_results.len() >= 2 {
                let word_sets: Vec<std::collections::HashSet<&str>> = final_results
                    .iter()
                    .map(|(_, t)| t.split_whitespace().collect())
                    .collect();
                let union_size = word_sets
                    .iter()
                    .flat_map(|s| s.iter())
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                let avg_size: f64 =
                    word_sets.iter().map(|s| s.len() as f64).sum::<f64>() / word_sets.len() as f64;
                if union_size > 0 {
                    1.0 - avg_size / union_size as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let mut sd = self.stall_detector.lock().await;
            sd.push_turn(proposals_map, turn_entropy)
        };
        if stall_risk > 0.6 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[stall] High stall risk detected: {:.2}\n", stall_risk),
            })
            .await?;
        }

        let response = format!("{}\n{}", synthesized_text, synthesized_artifacts);
        Ok((response, nash_weight_updates, stall_risk))
    }

    /// Phase 6: Run security, tautology, and fallacy filters on the synthesised response.
    /// Returns `Ok(false)` when the response must be dropped, `Ok(true)` when it passes.
    pub(super) async fn filter_response(
        &self,
        response: &str,
        history_contents: &[String],
    ) -> Result<bool> {
        let secrets = SecretScanner::scan(response);
        if !secrets.is_empty() {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Blocked: Security Violation]\n".to_string(),
            })
            .await?;
            return Ok(false);
        }

        let current_p = f64::from_bits(
            self.completion_probability
                .load(std::sync::atomic::Ordering::Acquire),
        );
        if current_p < 0.5 && TautologyFilter::is_tautological(response, history_contents) {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Pruned: Tautology]\n".to_string(),
            })
            .await?;
            return Ok(false);
        }

        let fallacies = FallacyDetector::scan(response, &[]);
        if !fallacies.is_empty() {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("\n[Warning: {} fallacies detected]\n", fallacies.len()),
            })
            .await?;
        }

        Ok(true)
    }

    /// Phase 7: Validate artifacts, self-heal up to 3 times, then reward/penalise agents.
    /// Returns `Some((changes, turn_outcome, final_response))` on success, `None` on total failure.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn validate_and_heal(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
        response: String,
        artifacts_snapshot: &BTreeMap<String, Arc<Artifact>>,
        active_agents: &[(usize, String)],
        weights: &BTreeMap<String, f64>,
        final_results: &[(String, String)],
        prompt: &Arc<str>,
        pre_turn_idx: u32,
        paused_rx: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Option<(Vec<PreparedArtifactChange>, TurnOutcome, String)>> {
        let mut retry_count = 0u32;
        let mut current_response = response;
        let mut final_prepared = None;
        let mut last_failure_regressive = false;

        while retry_count < 3 {
            let proposed_artifacts = Self::parse_artifacts(&current_response);
            match self
                .process_proposed_artifacts(proposed_artifacts, artifacts_snapshot)
                .await?
            {
                ArtifactProcessOutcome::Ready(changes, turn_outcome) => {
                    final_prepared = Some((changes, turn_outcome, current_response));
                    break;
                }
                outcome => {
                    last_failure_regressive = matches!(outcome, ArtifactProcessOutcome::Regressive);
                    retry_count += 1;
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "\n[Self-Healing] Synthesis failed validation. Attempting hot-patch cycle {}/3...\n",
                            retry_count
                        ),
                    })
                    .await?;

                    if retry_count == 2 {
                        let sigma = sigma_lock.lock().await;
                        let mut topo = self.topology.lock().await;
                        let retry_cat = sigma
                            .turns
                            .last()
                            .and_then(|t| t.task_category)
                            .unwrap_or(TaskCategory::Research);
                        let directive = topo.maybe_shift(&sigma, sigma.iteration_index, retry_cat);
                        drop(sigma);
                        if directive.is_none() {
                            topo.shift_to(
                                crate::engines::topology::DebateTopology::Mediated,
                                pre_turn_idx,
                                crate::engines::topology::TopologyReason::Deadlock,
                            );
                        }
                    }

                    let corrective_prompt = format!(
                        "{}\n\n[CRITICAL: Validation Failed]\nThe previous collective synthesis failed quality/safety gates. Re-implement the code blocks ensuring strict adherence to Rust safety and project invariants.",
                        prompt
                    );

                    let mut tasks = Vec::new();
                    for (idx, name) in active_agents {
                        let agent = &self.agents[*idx];
                        let agent_id = name.clone();
                        let p = corrective_prompt.clone();
                        let event_tx = self.event_tx.clone();
                        let mut p_rx = paused_rx.clone();
                        let rate_limiter = Arc::clone(&self.rate_limiter);

                        tasks.push(async move {
                            rate_limiter.wait_for_permit(&agent_id).await;
                            let mut stream = agent
                                .stream_prompt(&p)
                                .await
                                .map_err(|e| anyhow::anyhow!("Agent {agent_id} failure: {e:?}"))?;
                            let mut resp = String::new();
                            loop {
                                if *p_rx.borrow() {
                                    crate::log_warn!(
                                        p_rx.changed().await,
                                        "Failed to wait for pause state change during retry"
                                    );
                                    continue;
                                }
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(120),
                                    stream.next(),
                                )
                                .await
                                {
                                    Ok(Some(Ok(chunk))) => {
                                        resp.push_str(&chunk);
                                        event_tx
                                            .send(StreamEvent::TokenReceived {
                                                agent_id: agent_id.clone(),
                                                token: chunk,
                                            })
                                            .await?;
                                    }
                                    Err(_) => {
                                        return Err(anyhow::anyhow!(
                                            "Agent {agent_id} timed out waiting for response"
                                        ));
                                    }
                                    _ => break,
                                }
                            }
                            Ok::<(String, String), anyhow::Error>((agent_id, resp))
                        });
                    }

                    let new_proposals: Vec<(String, String)> = futures::future::join_all(tasks)
                        .await
                        .into_iter()
                        .flatten()
                        .collect();

                    if new_proposals.is_empty() {
                        break;
                    }

                    let text_proposals: Vec<(String, String, f64)> = new_proposals
                        .iter()
                        .map(|(id, text)| {
                            (
                                id.clone(),
                                text.clone(),
                                weights.get(id).copied().unwrap_or(1.0),
                            )
                        })
                        .collect();
                    let syn_text =
                        EnsembleEngine::merge_proposals(text_proposals, TaskCategory::Research, "");

                    let mut art_proposals: BTreeMap<String, (String, Vec<ArtifactDiff>)> =
                        BTreeMap::new();
                    for (_id, text) in &new_proposals {
                        for (name, (lang, content)) in Self::parse_artifacts(text) {
                            let default_art = Arc::new(Artifact::default());
                            let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                            let entry = art_proposals.entry(name).or_insert((lang, vec![]));
                            entry.1.push(DiffEngine::generate_delta(
                                &current.content,
                                &content,
                                current.version,
                            ));
                        }
                    }
                    let mut syn_arts = String::new();
                    for (name, (lang, diffs)) in art_proposals {
                        let default_art = Arc::new(Artifact::default());
                        let current = artifacts_snapshot.get(&name).unwrap_or(&default_art);
                        if let Some(merged) = SynthesisEngine::merge(
                            &current.content,
                            diffs
                                .into_iter()
                                .map(|d| ("Anonymous".to_string(), d))
                                .collect(),
                            &lang,
                        ) {
                            crate::log_warn!(
                                writeln!(syn_arts, "\n```{}:{}\n{}\n```", lang, name, merged),
                                "Failed to write merged artifact"
                            );
                        }
                    }
                    current_response = format!("{}\n{}", syn_text, syn_arts);
                }
            }
        }

        if final_prepared.is_none() {
            {
                let intell = self.intelligence.lock().await;
                for (id, _) in final_results {
                    intell.update_diff_quality(id, false, last_failure_regressive);
                }
            }
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "\n[Self-Healing Failed] Could not converge on a valid synthesis after 3 attempts. Aborting turn.\n".to_string(),
            })
            .await?;
        }

        Ok(final_prepared)
    }

    /// Phase 8: Update agent intelligence profiles and metacognitive state after a successful turn.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn update_agent_profiles(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
        final_results: &[(String, String)],
        winner_id: &str,
        turn_outcome: TurnOutcome,
        final_response: &str,
        pre_turn_idx: u32,
        pre_recent_turns: &[Turn],
        latency_ms: u64,
    ) -> Result<()> {
        // Reward agents that produced good diffs
        {
            let intell = self.intelligence.lock().await;
            for (id, _) in final_results {
                intell.update_diff_quality(id, true, false);
            }
        }

        if !final_results.is_empty() {
            let names: Vec<&str> = final_results.iter().map(|(id, _)| id.as_str()).collect();
            // Notify which artifacts will be written — done upstream, so just profile updates here.
            let sigma = sigma_lock.lock().await;
            let intell = self.intelligence.lock().await;
            for (id, text) in final_results {
                let p_turn = Turn {
                    index: sigma.iteration_index,
                    model_id: id.clone(),
                    content: text.clone(),
                    timestamp: ConversationState::now(),
                    diffs: vec![],
                    certainty: Some(CertaintyAnalyzer::compute(text, 0.1)),
                    outcome: if id == winner_id {
                        turn_outcome
                    } else {
                        TurnOutcome::Unknown
                    },
                    task_category: Some(TaskCategory::Research),
                    structure: Some(TurnStructure::FreeForm),
                    signature: vec![],
                    surprise_signal: None,
                    consistency_score: None,
                    diff_quality_score: None,
                    persona_disclosure: None,
                };
                intell.update_profile_with_latency(&p_turn, 0.7, latency_ms);
            }
            drop(intell);
            drop(sigma);
            let _ = names; // consumed above
        }

        // Metacognitive observation
        {
            let mut obs = self.observer.lock().await;
            let mut surprise = self.surprise_engine.lock().await;
            let mut interventions = Vec::new();
            for (agent_id_obs, response_text) in final_results {
                let agent_turn = Turn {
                    index: pre_turn_idx,
                    model_id: agent_id_obs.clone(),
                    content: response_text.clone(),
                    timestamp: ConversationState::now(),
                    diffs: vec![],
                    certainty: Some(CertaintyAnalyzer::compute(response_text, 0.1)),
                    outcome: if agent_id_obs == winner_id {
                        turn_outcome
                    } else {
                        TurnOutcome::Unknown
                    },
                    task_category: Some(TaskCategory::Research),
                    structure: None,
                    signature: vec![],
                    surprise_signal: None,
                    consistency_score: None,
                    diff_quality_score: None,
                    persona_disclosure: None,
                };
                let agent_interventions =
                    obs.observe_turn(&agent_turn, pre_recent_turns, &mut surprise);
                interventions.extend(agent_interventions);
            }
            drop(surprise);
            for intervention in &interventions {
                if let Some(block) = MetacognitiveObserver::format_interventions(
                    std::slice::from_ref(intervention),
                    &intervention.target_agent,
                ) {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "Observer".to_string(),
                        token: block,
                    })
                    .await?;
                }
            }

            let winner_turn = Turn {
                index: pre_turn_idx,
                model_id: winner_id.to_string(),
                content: final_response.to_string(),
                timestamp: ConversationState::now(),
                diffs: vec![],
                certainty: Some(CertaintyAnalyzer::compute(final_response, 0.1)),
                outcome: turn_outcome,
                task_category: Some(TaskCategory::Research),
                structure: None,
                signature: vec![],
                surprise_signal: None,
                consistency_score: None,
                diff_quality_score: None,
                persona_disclosure: None,
            };
            let current_quality = QualityScorer::score(&winner_turn);
            let prior_quality = pre_recent_turns
                .first()
                .map(QualityScorer::score)
                .unwrap_or(0.5);
            let improved = current_quality > prior_quality;
            {
                let pending = self.pending_interventions.lock().await;
                for intervention in pending.iter().filter(|i| i.target_agent == winner_id) {
                    obs.record_intervention_outcome(winner_id, intervention.source, improved);
                }
            }
            if !interventions.is_empty() {
                let mut pending = self.pending_interventions.lock().await;
                *pending = interventions;
            }

            if obs.should_eliminate(winner_id, 10) {
                let mut skips = self.skip_until.lock().await;
                skips.insert(winner_id.to_string(), pre_turn_idx + 100);
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "Observer".to_string(),
                    token: format!(
                        "[ELIMINATION] Agent {} removed from pool (Elo < 1200)\n",
                        winner_id
                    ),
                })
                .await?;
            }
        }

        // Topology: record outcome and check for shifts
        {
            let sigma = sigma_lock.lock().await;
            let topo_turn = Turn {
                index: pre_turn_idx,
                model_id: winner_id.to_string(),
                content: final_response.to_string(),
                timestamp: ConversationState::now(),
                diffs: vec![],
                certainty: None,
                outcome: turn_outcome,
                task_category: Some(TaskCategory::Research),
                structure: None,
                signature: vec![],
                surprise_signal: None,
                consistency_score: None,
                diff_quality_score: None,
                persona_disclosure: None,
            };
            let topo_cat = topo_turn.task_category.unwrap_or(TaskCategory::Research);
            let quality = RewardVector::from_turn(&topo_turn).weighted_score(topo_cat);
            // Use last ledger entry as a cost proxy; apply_turn_to_state hasn't run yet.
            let topo_cost = sigma
                .budget
                .entries
                .last()
                .map(|e| e.cost_usd)
                .unwrap_or(0.01);
            let mut topo = self.topology.lock().await;
            topo.record_turn_outcome(turn_outcome, quality, topo_cat, topo_cost, latency_ms);
            if let Some(directive) = topo.maybe_shift(&sigma, sigma.iteration_index, topo_cat) {
                if let Some(modifier) = &directive.prompt_modifier {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "Topology".to_string(),
                        token: format!("[TOPOLOGY SHIFT → {:?}] {modifier}\n", directive.topology),
                    })
                    .await?;
                }
                let mut stored = self.active_topology_directive.lock().await;
                *stored = Some(directive);
            }
        }

        Ok(())
    }

    #[instrument(skip_all, fields(session, turn))]
    pub async fn run_turn(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<bool> {
        // Phase 1: load memory context and detect prior regressions.
        let (
            pre_session_id,
            pre_turn_idx,
            pre_recent_turns,
            memory_examples,
            antipatterns,
            regression_prefix,
        ) = self.prepare_context_from_memory(&sigma_lock).await?;
        tracing::Span::current().record("session", pre_session_id.as_str());
        tracing::Span::current().record("turn", pre_turn_idx);
        tracing::info!(turn = pre_turn_idx, "turn starting");

        // Early convergence: skip agent calls when P(C) is high and last turn had no changes.
        {
            let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
            let last_turn_had_changes = pre_recent_turns
                .first()
                .map(|t| !t.diffs.is_empty())
                .unwrap_or(true);
            if current_p > 0.85 && !last_turn_had_changes && pre_turn_idx > 2 {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[convergence] P(C)={current_p:.2}, no artifact changes, skipping turn\n"
                    ),
                })
                .await?;
                return Ok(true);
            }
        }

        // Phase 2: analytics strategy + adaptive agent selection.
        let (strategy_critique, strategy_reduce_agents, adaptive_selection, s) =
            self.analyze_strategy_and_select_agents(&sigma_lock).await?;

        // Phase 3: build final prompt.
        let (raw_prompt, history_contents, active_agents, artifacts_snapshot) = self
            .build_prompt(
                &s,
                strategy_critique,
                &adaptive_selection,
                &memory_examples,
                &antipatterns,
                &regression_prefix,
            )
            .await?;

        const PROMPT_CHAR_CAP: usize = 24_000;
        let prompt_str = if raw_prompt.len() > PROMPT_CHAR_CAP {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[prompt] Truncated to {}K chars (was {} chars)\n",
                    PROMPT_CHAR_CAP / 1000,
                    raw_prompt.len()
                ),
            })
            .await?;
            let mut truncated = raw_prompt[..PROMPT_CHAR_CAP].to_string();
            truncated.push_str("\n... [truncated]");
            truncated
        } else {
            raw_prompt
        };

        tracing::info!(
            turn = pre_turn_idx,
            agents = active_agents.len(),
            prompt_len = prompt_str.len(),
            "calling agents"
        );
        let start_time = Instant::now();
        let prompt_arc: Arc<str> = Arc::from(prompt_str);

        // Phase 4: call agents (streaming, caching, control signal handling).
        let final_results = self
            .call_agents(
                &sigma_lock,
                Arc::clone(&prompt_arc),
                active_agents.clone(),
                strategy_reduce_agents,
            )
            .await?;

        // An empty result vec signals a control-signal early exit (Shutdown/Inject/Rewind).
        if final_results.is_empty() {
            // Determine the correct return value from the current state.
            let s = sigma_lock.lock().await;
            return Ok(s.iteration_index > pre_turn_idx);
        }

        let weights =
            crate::engines::consensus::InfluenceWeightManager::calculate_weights_with_recency(
                &*sigma_lock.lock().await,
                0.9,
            );

        // Phase 5: synthesise responses into a single text + artifact block.
        let (response, nash_weight_updates, stall_risk) = self
            .synthesize_responses(&final_results, &artifacts_snapshot, &weights)
            .await?;

        let latency_ms = start_time.elapsed().as_millis() as u64;
        tracing::info!(
            turn = pre_turn_idx,
            latency_ms,
            responses = final_results.len(),
            "agents responded"
        );
        let winner_id = "Collective Swarm".to_string();

        // Phase 6: security / tautology / fallacy filters.
        if !self.filter_response(&response, &history_contents).await? {
            return Ok(false);
        }

        // Phase 7: artifact validation with self-healing retry loop.
        let (_paused_tx, paused_rx) = tokio::sync::watch::channel(false);
        let Some((changes, turn_outcome, final_response)) = self
            .validate_and_heal(
                &sigma_lock,
                response,
                &artifacts_snapshot,
                &active_agents,
                &weights,
                &final_results,
                &prompt_arc,
                pre_turn_idx,
                paused_rx,
            )
            .await?
        else {
            return Ok(false);
        };

        if !changes.is_empty() {
            let names: Vec<&str> = changes.iter().map(|c| c.name.as_str()).collect();
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("Writing changes to {}\n", names.join(", ")),
            })
            .await?;
        }

        // Execute any [TOOL: ...] directives embedded in the final response (Task 9)
        {
            let directives = Self::parse_tool_directives(&final_response);
            let mut tool_outputs = Vec::new();
            for (tool_name, args) in &directives {
                let output = self
                    .execute_tool_directive(tool_name, args, &sigma_lock)
                    .await;
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "Tool".to_string(),
                    token: format!("{output}\n"),
                })
                .await?;
                tool_outputs.push((tool_name.clone(), output));
            }
            if !tool_outputs.is_empty() {
                sigma_lock.lock().await.last_tool_outputs = tool_outputs;
            }
        }

        // Phase 8: update agent profiles and metacognitive state.
        self.update_agent_profiles(
            &sigma_lock,
            &final_results,
            &winner_id,
            turn_outcome,
            &final_response,
            pre_turn_idx,
            &pre_recent_turns,
            latency_ms,
        )
        .await?;

        // A/B test: accumulate quality by critique arm; auto-adopt when significant.
        {
            let quality = match turn_outcome {
                TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence => 1.0,
                TurnOutcome::RolledBack
                | TurnOutcome::Rejected
                | TurnOutcome::VerificationFailed => 0.0,
                _ => 0.5,
            };
            if strategy_critique {
                self.ab_test_quality.lock().await.push(quality);
            } else {
                self.ab_control_quality.lock().await.push(quality);
            }
            let (control, test) = {
                let mut c = self.ab_control_quality.lock().await;
                let mut t = self.ab_test_quality.lock().await;
                if c.len() >= 10 && t.len() >= 10 {
                    (std::mem::take(&mut *c), std::mem::take(&mut *t))
                } else {
                    (vec![], vec![])
                }
            };
            if !control.is_empty() && !test.is_empty() {
                let significant =
                    crate::engines::self_improvement::AbTestManager::check_significance(
                        &control, &test,
                    );
                let control_mean = control.iter().sum::<f64>() / control.len() as f64;
                let test_mean = test.iter().sum::<f64>() / test.len() as f64;
                let adopted = significant && test_mean > control_mean;
                let report = crate::engines::self_improvement::AbTestReport {
                    hypothesis_id: "critique_protocol".to_string(),
                    control_mean,
                    test_mean,
                    effect_size: (test_mean - control_mean).abs(),
                    significant,
                    adopted,
                    confidence_interval: (control_mean.max(0.0), test_mean.min(1.0)),
                };
                let mut adjuster = self.runtime_adjuster.lock().await;
                if adjuster.apply_if_significant(
                    "critique_always",
                    if adopted { 1.0 } else { 0.0 },
                    &report,
                    "",
                ) {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "[A/B ADOPTED] Critique protocol adopted (test={:.2} vs control={:.2})\n",
                            test_mean, control_mean
                        ),
                    })
                    .await?;
                }
            }
        }

        // Final: commit the turn under the sigma lock.
        let result = self
            .commit_turn(
                &sigma_lock,
                changes,
                turn_outcome,
                &winner_id,
                &final_response,
                &prompt_arc,
                latency_ms,
                nash_weight_updates,
                stall_risk,
            )
            .await;

        // Clear planning hints only after a successful commit so a failed turn
        // re-injects the same hints on the next attempt.
        if result.is_ok() {
            self.pending_planning_hints.lock().await.clear();
        }

        result
    }
}
