use super::*;

impl Orchestrator {
    /// Phase 1: Load session metadata, recall memory context, and detect regressions.
    /// Returns `(session_id, turn_idx, recent_turns, memory_examples, antipatterns, regression_prefix)`.
    pub(super) async fn prepare_context_from_memory(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
    ) -> Result<(
        String,
        u32,
        Vec<Turn>,
        Vec<String>,
        Vec<MemoryRecord>,
        String,
    )> {
        let (session_id, turn_idx, recent_turns) = {
            let s = sigma_lock.lock().await;
            let recent: Vec<Turn> = s.turns.iter().rev().take(5).cloned().collect();
            (s.session_id.clone(), s.iteration_index, recent)
        };

        let recall_query = recent_turns
            .first()
            .map(|t| t.content.chars().take(200).collect::<String>())
            .unwrap_or_else(|| "latest turn context".to_string());

        let (memory_examples, antipatterns) = {
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(session_id.clone());
            let examples = vec![
                bridge
                    .recall_relevant_summary(&session_id, &recall_query, 3, turn_idx)
                    .await
                    .unwrap_or_default(),
            ];
            let anti = bridge.recall_antipatterns(&recall_query, 2).await;
            (examples, anti)
        };

        {
            let mut rx = self.resource_rx.lock().await;
            while let Ok(event) = rx.try_recv() {
                if let Some(alert) = &event.alert {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[compute] Resource alert: {}\n", alert),
                    })
                    .await?;
                }
            }
        }

        let regression_prefix = {
            let intell = self.intelligence.lock().await;
            if let Some(alert) = intell.detect_regression("swarm", &recent_turns) {
                RegressionFeedbackHandler::compose_corrective_prompt(
                    &alert,
                    "",
                    &RegressionFeedbackHandler::counter_examples(
                        &recent_turns,
                        TaskCategory::Research,
                    ),
                )
            } else {
                String::new()
            }
        };

        Ok((
            session_id,
            turn_idx,
            recent_turns,
            memory_examples,
            antipatterns,
            regression_prefix,
        ))
    }

    /// Phase 2: Apply analytics strategy recommendations and select active agents.
    /// Returns `(strategy_critique, strategy_reduce_agents, adaptive_selection, state_clone)`.
    pub(super) async fn analyze_strategy_and_select_agents(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
    ) -> Result<(bool, bool, Option<AdaptiveSelection>, ConversationState)> {
        let mut strategy_critique = false;
        let mut strategy_reduce_agents = false;
        let adaptive_selection;

        {
            let s = sigma_lock.lock().await;
            let sub_swarms = crate::engines::planning::SubSwarmGenerator::identify_sub_swarms(&s);
            for task in sub_swarms {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[Swarm] Spawning sub-orchestrator for complex task: {}\n",
                        task.description
                    ),
                })
                .await?;
                self.swarm.spawn_node(&task.id, self.turn_tx.subscribe());
            }

            let recs = crate::engines::analytics::StrategyRecommender::recommend(&s);
            for rec in &recs {
                if rec.confidence > 0.5 {
                    match rec.action.as_str() {
                        "switch_to_critique_protocol" => {
                            strategy_critique = true;
                            self.emit(StreamEvent::TokenReceived {
                                agent_id: "System".to_string(),
                                token:
                                    "Low success rate detected — switching to critique protocol\n"
                                        .to_string(),
                            })
                            .await?;
                        }
                        "reduce_parallel_inference" => {
                            strategy_reduce_agents = true;
                        }
                        _ => {}
                    }
                }
            }
            // Route high-impact recommendations to the planning layer.
            {
                let high_impact: Vec<String> = recs
                    .iter()
                    .filter(|r| r.expected_impact > 0.7 && r.confidence > 0.7)
                    .map(|r| r.action.clone())
                    .collect();
                if !high_impact.is_empty() {
                    let mut hints = self.pending_planning_hints.lock().await;
                    hints.extend(high_impact);
                    let excess = hints.len().saturating_sub(10);
                    if excess > 0 {
                        hints.drain(..excess);
                    }
                }
            }

            adaptive_selection = {
                let obs = self.observer.lock().await;
                let coll = self.collective.lock().await;
                Some(coll.select_strategy_adaptive(&s, &obs))
            };
        }

        let state_clone = {
            let guard = sigma_lock.lock().await;
            if guard.iteration_index == 0 && guard.turns.is_empty() {
                self.state_manager.checkpoint(&guard)?;
            }
            guard.clone()
        };

        Ok((
            strategy_critique,
            strategy_reduce_agents,
            adaptive_selection,
            state_clone,
        ))
    }

    /// Phase 3: Build the final prompt string including memory, metacognition, and topology.
    /// Returns `(prompt, history_contents, active_agents, artifacts_snapshot)`.
    pub(super) async fn build_prompt(
        &self,
        s: &ConversationState,
        strategy_critique: bool,
        adaptive_selection: &Option<AdaptiveSelection>,
        memory_examples: &[String],
        antipatterns: &[MemoryRecord],
        regression_prefix: &str,
    ) -> Result<(
        String,
        Vec<String>,
        Vec<(usize, String)>,
        BTreeMap<String, Arc<Artifact>>,
    )> {
        let history_contents: Vec<String> = s
            .turns
            .iter()
            .rev()
            .take(10)
            .map(|t| t.content.clone())
            .collect();

        let mut distilled_prompt = self.build_differential_prompt(s).await;

        if strategy_critique {
            distilled_prompt.push_str(
                "\n\nCRITICAL: Previous attempts had high failure rate. \
                 Before proposing changes, critique your own approach. \
                 Identify what could go wrong. Only proceed with the safest path.\n",
            );
        }

        if let Some(sel) = adaptive_selection {
            match sel.strategy {
                MetaStrategy::DebateAndCritique => {
                    distilled_prompt.push_str(
                        "\n\n[META-STRATEGY: DebateAndCritique] \
                         High variance in recent certainty scores detected. \
                         Critically examine all proposals and surface disagreements before converging.\n",
                    );
                }
                MetaStrategy::DirectImplementation => {
                    if let Some(ref agent) = sel.preferred_agent {
                        distilled_prompt.push_str(&format!(
                            "\n\n[META-STRATEGY: DirectImplementation] \
                             Dominant specialist {} detected (Elo > 1600). \
                             Prefer this agent's proposal for final synthesis.\n",
                            agent
                        ));
                    }
                }
                MetaStrategy::MemoryInjection => {
                    distilled_prompt.push_str(
                        "\n\n[META-STRATEGY: MemoryInjection] \
                         Convergence probability is low after multiple turns. \
                         Retrieve and apply relevant prior session lessons before proceeding.\n",
                    );
                }
                _ => {}
            }
        }

        let mut active = self.select_active_agents(s, adaptive_selection).await;

        // Apply topology-driven agent grouping
        {
            use crate::engines::topology::AgentGrouping;
            let directive = {
                let stored = self.active_topology_directive.lock().await;
                match stored.as_ref() {
                    Some(d) => d.clone(),
                    None => {
                        let topo = self.topology.lock().await;
                        topo.current_directive()
                    }
                }
            };
            match directive.agent_grouping {
                AgentGrouping::Single(idx) => {
                    if idx < active.len() {
                        active = vec![active[idx].clone()];
                    }
                }
                AgentGrouping::Pairs(ref pairs) => {
                    if let Some(&(a, b)) = pairs.first() {
                        let mut subset = Vec::new();
                        if a < active.len() {
                            subset.push(active[a].clone());
                        }
                        if b < active.len() {
                            subset.push(active[b].clone());
                        }
                        if !subset.is_empty() {
                            active = subset;
                        }
                    }
                }
                AgentGrouping::Branches(ref branches) => {
                    let branch_idx = (s.iteration_index as usize) % branches.len().max(1);
                    if let Some(branch) = branches.get(branch_idx) {
                        let subset: Vec<_> = branch
                            .iter()
                            .filter_map(|&i| active.get(i).cloned())
                            .collect();
                        if !subset.is_empty() {
                            active = subset;
                        }
                    }
                }
                AgentGrouping::All => {}
            }
            if let Some(modifier) = &directive.prompt_modifier {
                distilled_prompt.push('\n');
                distilled_prompt.push_str(modifier);
                distilled_prompt.push('\n');
            }
        }

        let structure = ReasoningEngine::select_structure(TaskCategory::Research, &active[0].1);
        match structure {
            TurnStructure::StepByStep => distilled_prompt
                .push_str("\nStructure your response with numbered reasoning steps."),
            TurnStructure::ProsCons => {
                distilled_prompt.push_str("\nExplicitly analyze tradeoffs (Pros vs Cons).")
            }
            TurnStructure::CodeFirst => {
                distilled_prompt.push_str("\nProvide the code delta (Δα) before any explanation.")
            }
            TurnStructure::Symbolic => distilled_prompt.push_str(
                "\nUse SSM (Symbolic Swarm Mode): use ∀, ∃, ⊢, ⊥, Δα, σ, μ. Minimize prose.",
            ),
            _ => {}
        }

        if !regression_prefix.is_empty() {
            distilled_prompt = format!("{regression_prefix}\n{distilled_prompt}");
        }
        if !memory_examples.is_empty() {
            distilled_prompt.push_str("\n\nSuccessful examples from similar tasks:\n");
            for (i, _ex) in memory_examples.iter().take(5).enumerate() {
                crate::log_warn!(
                    writeln!(
                        distilled_prompt,
                        "- [Example {}] (recalled from memory)",
                        i + 1
                    ),
                    "Failed to write example to prompt"
                );
            }
        }
        if !antipatterns.is_empty() {
            distilled_prompt.push_str("\n\nAntipatterns to AVOID (failed in similar tasks):\n");
            for ap in antipatterns.iter().take(3) {
                crate::log_warn!(
                    writeln!(
                        distilled_prompt,
                        "- [Session {}] {}",
                        ap.session_id, ap.metadata_json
                    ),
                    "Failed to write antipattern to prompt"
                );
            }
        }

        // Inject known failure patterns from intelligence store (Task 6)
        {
            let intell = self.intelligence.lock().await;
            let task_cat = s
                .turns
                .last()
                .and_then(|t| t.task_category)
                .unwrap_or(TaskCategory::Research);
            let patterns = intell.top_failure_patterns(task_cat, 3);
            if !patterns.is_empty() {
                distilled_prompt.push_str("\n\n[KNOWN FAILURE MODES — avoid these]:\n");
                for p in &patterns {
                    crate::log_warn!(
                        writeln!(distilled_prompt, "- {p}"),
                        "Failed to write failure pattern to prompt"
                    );
                }
            }
        }

        // Inject metacognitive observer feedback from prior turn
        {
            let obs = self.observer.lock().await;
            for (_, name) in &active {
                if let Some(state) = obs.epistemic_state(name)
                    && state.confidence < 0.5
                    && !state.defeated.is_empty()
                {
                    crate::log_warn!(
                        writeln!(
                            distilled_prompt,
                            "\n[EPISTEMIC UPDATE for {name}] Confidence: {:.0}%. \
                             Defeated assumptions: {:?}. Adjust your reasoning accordingly.",
                            state.confidence * 100.0,
                            state.defeated
                        ),
                        "Failed to write epistemic update to prompt"
                    );
                }
            }
        }

        // Inject pending interventions from prior turn's observer
        {
            let mut pending = self.pending_interventions.lock().await;
            if !pending.is_empty() {
                for (_, name) in &active {
                    if let Some(block) = MetacognitiveObserver::format_interventions(&pending, name)
                    {
                        distilled_prompt.push_str(&block);
                    }
                }
                pending.clear();
            }
        }

        // Inject high-impact analytics recommendations as planning hints.
        {
            let hints = self.pending_planning_hints.lock().await;
            if !hints.is_empty() {
                distilled_prompt.push_str("\n\n[PLANNING HINTS]:\n");
                for hint in hints.iter() {
                    crate::log_warn!(
                        writeln!(distilled_prompt, "- {hint}"),
                        "Failed to write planning hint to prompt"
                    );
                }
                // Hints are cleared in run_turn() only after successful commit.
            }
        }

        let artifacts_snapshot = s.artifacts.clone();
        Ok((
            distilled_prompt,
            history_contents,
            active,
            artifacts_snapshot,
        ))
    }

    /// Selects and orders the active agent list for a turn.
    pub(super) async fn select_active_agents(
        &self,
        s: &ConversationState,
        adaptive_selection: &Option<AdaptiveSelection>,
    ) -> Vec<(usize, String)> {
        let mut active = Vec::new();
        {
            let skips = self.skip_until.lock().await;
            for (idx, agent) in self.agents.iter().enumerate() {
                let until = skips.get(agent.name()).copied().unwrap_or(0);
                if s.iteration_index >= until {
                    active.push((idx, agent.name().to_string()));
                }
            }
        }
        if active.is_empty() {
            active.push((0, self.agents[0].name().to_string()));
        }

        {
            let vf = self.verification_failures.lock().await;
            active.retain(|(_, n)| vf.get(n).copied().unwrap_or(0) <= 3);
            if active.is_empty() {
                active.push((0, self.agents[0].name().to_string()));
            }
        }

        {
            let intell = self.intelligence.lock().await;
            let names: Vec<String> = active.iter().map(|(_, n)| n.clone()).collect();
            if let Ok(best) = intell.route_task_constrained(
                TaskCategory::Research,
                &names,
                u32::MAX,
                u64::MAX,
                &[],
            ) && let Some(pos) = active.iter().position(|(_, n)| *n == best)
            {
                active.swap(0, pos);
            }
        }

        {
            let obs = self.observer.lock().await;
            let task_cat = s
                .turns
                .last()
                .and_then(|t| t.task_category)
                .unwrap_or(TaskCategory::Research);
            if !obs.elo_ratings.is_empty() {
                active.sort_by(|a, b| {
                    let elo_a = obs.elo_for_category(&a.1, task_cat);
                    let elo_b = obs.elo_for_category(&b.1, task_cat);
                    elo_b
                        .partial_cmp(&elo_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        if let Some(AdaptiveSelection {
            preferred_agent: Some(preferred),
            ..
        }) = adaptive_selection
            && let Some(pos) = active.iter().position(|(_, n)| n == preferred)
        {
            active.swap(0, pos);
        }

        active
    }

    /// Phase 4: Call agents (with caching, rate-limiting, streaming, and control signal handling).
    /// Returns the collected `(agent_id, response_text)` pairs.
    pub(super) async fn call_agents(
        &self,
        sigma_lock: &Arc<Mutex<ConversationState>>,
        prompt: Arc<str>,
        mut active_agents: Vec<(usize, String)>,
        strategy_reduce_agents: bool,
    ) -> Result<Vec<(String, String)>> {
        let (paused_tx, paused_rx) = tokio::sync::watch::channel(false);

        if strategy_reduce_agents && active_agents.len() > 1 {
            active_agents.truncate(1);
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: "Budget burn rate high — reducing to single agent\n".to_string(),
            })
            .await?;
        }
        {
            let s = sigma_lock.lock().await;
            if s.budget.mode() == BudgetMode::Emergency && active_agents.len() > 1 {
                drop(s);
                active_agents.truncate(1);
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: "[compute] Emergency budget mode: single agent only\n".to_string(),
                })
                .await?;
            }
        }

        let mut cached_results = Vec::new();
        let mut uncached_agents = Vec::new();
        {
            let mut compute = self.compute.lock().await;
            for entry in &active_agents {
                if let Some(cached) = compute.cache.get(&prompt, &entry.1) {
                    cached_results.push((entry.1.clone(), cached));
                } else {
                    uncached_agents.push(entry.clone());
                }
            }
        }

        {
            let agent_names: Vec<&str> = uncached_agents.iter().map(|(_, n)| n.as_str()).collect();
            if !agent_names.is_empty() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("\nAsking {} for their take...\n", agent_names.join(" and ")),
                })
                .await?;
            }
            if !cached_results.is_empty() {
                let cached_names: Vec<&str> =
                    cached_results.iter().map(|(n, _)| n.as_str()).collect();
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("Reusing cached response from {}\n", cached_names.join(", ")),
                })
                .await?;
            }
        }

        let (is_divergent, artifacts_for_divergent) = {
            let s = sigma_lock.lock().await;
            let divergent = s.mode_library.current().context_distribution
                == crate::types::mode::ContextDistribution::Divergent;
            let arts: std::collections::HashMap<
                String,
                std::sync::Arc<crate::types::artifact::Artifact>,
            > = s
                .artifacts
                .iter()
                .map(|(k, v)| (k.clone(), std::sync::Arc::clone(v)))
                .collect();
            (divergent, arts)
        };

        let mut tasks = Vec::new();
        for (idx, name) in &uncached_agents {
            let agent = &self.agents[*idx];
            let agent_id = name.clone();
            let prompt = Arc::clone(&prompt);
            let event_tx = self.event_tx.clone();
            let mut p_rx = paused_rx.clone();
            let rate_limiter = Arc::clone(&self.rate_limiter);

            let divergent_supplement = if is_divergent {
                if let Some(hash_pos) = agent_id.find('#') {
                    let role = &agent_id[hash_pos + 1..];
                    Self::divergent_context_for_role(role, &artifacts_for_divergent)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let feedback = {
                let collective = self.collective.lock().await;
                collective
                    .profiles
                    .get(&agent_id)
                    .and_then(crate::engines::prompt_evolution::ClosedLoopFeedback::generate_corrective_directive)
            };

            tasks.push(async move {
                let mut agent_prompt = (*prompt).to_string();
                if let Some(f) = feedback {
                    agent_prompt = format!("{}\n\n{}", f, agent_prompt);
                }
                if !divergent_supplement.is_empty() {
                    agent_prompt.push_str(&divergent_supplement);
                }

                let mut delay_ms = 1_000u64;
                for attempt in 0u32..4 {
                    if attempt > 0 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                        delay_ms = (delay_ms * 2).min(30_000);
                    }

                    rate_limiter.wait_for_permit(&agent_id).await;
                    let mut stream = match agent.stream_prompt(&agent_prompt).await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::info!(agent = %agent_id, "local inference failed, attempting remote MCP fallback");
                            if let Ok(remote_res) = McpGateway::remote_sampling(&agent_prompt, &agent_id).await {
                                return Ok((agent_id, remote_res));
                            }
                            let e = anyhow::anyhow!("Agent {agent_id} failure: {e:?}");
                            if is_fatal_auth_error(&e) {
                                return Err(e);
                            }
                            if is_rate_limited(&e) && attempt < 3 {
                                event_tx
                                    .send(StreamEvent::TokenReceived {
                                        agent_id: agent_id.clone(),
                                        token: format!("\n[{agent_id}] rate-limited, retrying in {}s...\n", delay_ms / 1000),
                                    })
                                    .await?;
                                continue;
                            }
                            return Err(e);
                        }
                    };

                    let mut response = String::new();
                    let mut hit_rate_limit = false;
                    loop {
                        if *p_rx.borrow() {
                            crate::log_warn!(p_rx.changed().await, "Failed to wait for pause state change");
                            continue;
                        }
                        match tokio::time::timeout(std::time::Duration::from_secs(120), stream.next()).await {
                            Err(_) => return Err(anyhow::anyhow!("Agent {agent_id} timed out waiting for response")),
                            Ok(Some(Ok(chunk))) => {
                                response.push_str(&chunk);
                                event_tx
                                    .send(StreamEvent::TokenReceived { agent_id: agent_id.clone(), token: chunk })
                                    .await?;
                            }
                            Ok(Some(Err(e))) => {
                                let e = anyhow::anyhow!("Agent {agent_id} stream error: {e:?}");
                                if is_fatal_auth_error(&e) {
                                    return Err(e);
                                }
                                if is_rate_limited(&e) && attempt < 3 {
                                    hit_rate_limit = true;
                                    event_tx
                                        .send(StreamEvent::TokenReceived {
                                            agent_id: agent_id.clone(),
                                            token: format!("\n[{agent_id}] rate-limited mid-stream, retrying in {}s...\n", delay_ms / 1000),
                                        })
                                        .await?;
                                } else {
                                    return Err(e);
                                }
                                break;
                            }
                            Ok(None) => break,
                        }
                    }
                    if !hit_rate_limit {
                        tracing::info!(agent = %agent_id, response_len = response.len(), "agent responded");
                        return Ok((agent_id, response));
                    }
                }
                Err(anyhow::anyhow!("Agent {agent_id} exhausted rate-limit retries"))
            });
        }

        let mut results_fut = futures::future::join_all(tasks);
        let mut final_results = Vec::new();
        let mut ctrl_guard = self.control_rx.lock().await;
        let mut ctrl_open = true;

        loop {
            tokio::select! {
                res = &mut results_fut => {
                    for r in res {
                        match r {
                            Ok(val) => final_results.push(val),
                            Err(e) => {
                                let msg = e.to_string();
                                if msg.contains("timed out")
                                    && let Some(name) = msg.strip_prefix("Agent ").and_then(|s| s.split_whitespace().next()) {
                                        let turn = sigma_lock.lock().await.iteration_index;
                                        self.skip_until.lock().await.insert(name.to_string(), turn + 3);
                                        self.emit(StreamEvent::Error(format!("Agent {} timed out, skipping for 3 turns", name))).await?;
                                        continue;
                                    }
                                self.emit(StreamEvent::Error(format!("Agent dropped: {}", e))).await?;
                            }
                        }
                    }
                    final_results.append(&mut cached_results);
                    if final_results.is_empty() {
                        return Err(anyhow::anyhow!("All agents in swarm failed to respond."));
                    }
                    {
                        let mut compute = self.compute.lock().await;
                        for (id, text) in &final_results {
                            compute.cache.insert(&prompt, id, text.clone(), 1.0);
                        }
                    }
                    break;
                }
                signal = ctrl_guard.recv(), if ctrl_open => {
                    match signal {
                        Some(ControlSignal::Pause) => { crate::log_warn!(paused_tx.send(true), "Failed to send pause signal"); }
                        Some(ControlSignal::Resume) => { crate::log_warn!(paused_tx.send(false), "Failed to send resume signal"); }
                        Some(ControlSignal::Shutdown) => return Ok(vec![]),
                        Some(ControlSignal::LockCode(name)) => {
                            let mut sigma = sigma_lock.lock().await;
                            if let Some(r) = sigma.goal_tree.root.as_mut() {
                                r.title = format!("{} [LOCKED: {}]", r.title, name);
                                r.status = crate::types::planning::GoalStatus::Complete;
                            }
                            self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("[Steer] Locked artifact: {}\n", name) }).await?;
                        }
                        Some(ControlSignal::MuteAgent(id)) => {
                            let mut sigma = sigma_lock.lock().await;
                            sigma.agent_weights.retain(|k, _| k != &id);
                            self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("[Steer] Muted agent: {}\n", id) }).await?;
                        }
                        Some(ControlSignal::DampenSwarm(factor)) => {
                            let mut sigma = sigma_lock.lock().await;
                            for w in sigma.agent_weights.values_mut() {
                                *w *= factor;
                            }
                            self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("[Steer] Dampened swarm by factor: {:.2}\n", factor) }).await?;
                        }
                        Some(ControlSignal::Inject(text)) => {
                            self.emit(StreamEvent::TokenReceived { agent_id: "User".to_string(), token: format!("\n[Neural Intercept] Injecting: {}\n", text) }).await?;
                            let mut sigma = sigma_lock.lock().await;
                            let user_turn = Turn {
                                index: sigma.iteration_index,
                                model_id: "User".to_string(),
                                content: text.clone(),
                                timestamp: ConversationState::now(),
                                diffs: vec![],
                                certainty: Some(1.0),
                                outcome: TurnOutcome::Unknown,
                                task_category: Some(TaskCategory::Research),
                                structure: Some(TurnStructure::FreeForm),
                                signature: vec![],
                                surprise_signal: None,
                                consistency_score: None,
                                diff_quality_score: None,
                                persona_disclosure: None,
                            };
                            sigma.push_turn(user_turn);
                            sigma.iteration_index += 1;
                            return Ok(vec![]);
                        }
                        Some(ControlSignal::Rewind(index)) => {
                            if let Ok(Some(restored)) = self.state_manager.restore_async(index).await {
                                let mut s = sigma_lock.lock().await;
                                *s = restored;
                                self.emit(StreamEvent::TokenReceived { agent_id: "System".to_string(), token: format!("\n[Rewound to iteration {}]\n", index) }).await?;
                                return Ok(vec![]);
                            }
                        }
                        Some(ControlSignal::CycleMode) => {
                            let mut s = sigma_lock.lock().await;
                            let old = s.mode_library.current_name().to_string();
                            s.mode_library.cycle_next();
                            let new_name = s.mode_library.current_name().to_string();
                            drop(s);
                            self.emit(StreamEvent::ModeTransition {
                                from_name: old,
                                to_name: new_name,
                                reason: "User cycle".to_string(),
                                synthesized: false,
                            }).await?;
                        }
                        Some(ControlSignal::SetModeByName(name)) => {
                            let mut s = sigma_lock.lock().await;
                            let old = s.mode_library.current_name().to_string();
                            if s.mode_library.switch_to_name(&name) {
                                let new_name = name.clone();
                                drop(s);
                                self.emit(StreamEvent::ModeTransition {
                                    from_name: old,
                                    to_name: new_name,
                                    reason: "User override".to_string(),
                                    synthesized: false,
                                }).await?;
                            }
                        }
                        None => { ctrl_open = false; }
                    }
                }
            }
        }

        Ok(final_results)
    }
}
