use super::*;

impl Orchestrator {
    pub(super) async fn process_proposed_artifacts(
        &self,
        proposed: HashMap<String, (String, String)>,
        snapshot: &BTreeMap<String, Arc<Artifact>>,
    ) -> Result<ArtifactProcessOutcome> {
        let current_sigma_snap = ConversationState {
            artifacts: snapshot.clone(),
            ..Default::default()
        };
        let all_names: Vec<String> = snapshot.keys().cloned().collect();
        let mut changes = Vec::new();
        let mut turn_outcome = TurnOutcome::Unknown;

        for (name, (lang, new_content)) in proposed {
            if let Err(e) = AstValidator::validate(&new_content, &lang) {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[diff] artifact \"{name}\" rejected: AST validation failed: {e}"
                    ),
                })
                .await?;
                return Ok(ArtifactProcessOutcome::Invalid);
            }

            let dups = QualityEngine::detect_duplication(&new_content, snapshot);
            if !dups.is_empty() {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[quality] duplication detected for \"{name}\": {:?}", dups),
                })
                .await?;
            }

            let default_artifact = Arc::new(Artifact {
                name: name.clone(),
                language: lang.clone(),
                content: String::new(),
                version: 0,
                history: vec![],
                ast_versions: BTreeMap::new(),
                proof_attachments: vec![],
                metrics: ArtifactMetrics::default(),
                skeleton: String::new(),
            });
            let current = snapshot.get(&name).unwrap_or(&default_artifact);

            if current.content == new_content {
                continue;
            }

            let delta = DiffEngine::generate_delta(&current.content, &new_content, current.version);

            let (p_fail, mc_confidence) = self
                .mc_runner
                .predict(current, &delta, 10)
                .await
                .unwrap_or((0.5, 0.0));
            if p_fail > 0.5 {
                return Ok(ArtifactProcessOutcome::Invalid);
            }
            let mc_variance = if mc_confidence > 0.0 {
                1.0 - mc_confidence
            } else {
                0.5
            };

            let new_metrics = QualityEngine::analyze_artifact(
                &Artifact {
                    content: new_content.clone(),
                    ..(**current).clone()
                },
                &all_names,
            );

            // --- Suggestion 8: Historical Regression Testing ---
            if let Some(gold) = self.gold_state.lock().await.as_ref() {
                let mut temp_sigma = ConversationState::default();
                temp_sigma.artifacts.insert(
                    name.clone(),
                    Arc::new(Artifact {
                        content: new_content.clone(),
                        metrics: new_metrics.clone(),
                        ..(**current).clone()
                    }),
                );
                let report = crate::engines::quality::RegressionDetector::detect(gold, &temp_sigma);
                if report.drift_score > 0.5 {
                    crate::log_warn!(self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[regression] detected significant quality drop in \"{name}\" (drift={:.2})\n", report.drift_score),
                    }).await, "Failed to emit regression warning");
                    return Ok(ArtifactProcessOutcome::Regressive);
                }
            }

            // --- Suggestion 7: RAG-Powered Evidence Anchoring ---
            if let Some(bridge) = self.memory_bridge.lock().await.store.as_ref() {
                let anchored = crate::engines::reasoning::ReasoningEngine::anchor_evidence_rag(
                    &new_content,
                    &current_sigma_snap,
                    bridge,
                )
                .await;
                let unanchored_claims: Vec<_> =
                    anchored.iter().filter(|c| c.confidence < 0.4).collect();
                if !unanchored_claims.is_empty() {
                    crate::log_warn!(
                        self.emit(StreamEvent::TokenReceived {
                            agent_id: "System".to_string(),
                            token: format!(
                                "[warning] {} claims in \"{name}\" lack evidence anchoring\n",
                                unanchored_claims.len()
                            ),
                        })
                        .await,
                        "Failed to emit anchoring warning"
                    );
                }
            }

            if RegressionDetector::is_regressive(&current.metrics, &new_metrics) {
                return Ok(ArtifactProcessOutcome::Regressive);
            }

            let mut final_metrics = new_metrics.clone();
            // Visual-fidelity scoring required GPU frame capture, which was never
            // wired up (no window/event loop), so it always evaluated to 0.0.
            final_metrics.visual_fidelity = 0.0;
            final_metrics.health_score *= 1.0 - (mc_variance * 0.3).min(0.15);

            let node_updates: Vec<(String, String)> =
                AstValidator::extract_nodes(&new_content, &lang)
                    .into_iter()
                    .collect();

            let proof = ProofManager::generate_proof(
                &Artifact {
                    name: name.clone(),
                    content: new_content.clone(),
                    language: lang.clone(),
                    version: current.version + 1,
                    history: vec![],
                    ast_versions: BTreeMap::new(),
                    proof_attachments: vec![],
                    metrics: final_metrics.clone(),
                    skeleton: String::new(),
                },
                vec![
                    "ast_valid".to_string(),
                    "mc_safe".to_string(),
                    "quality_checked".to_string(),
                ],
            );

            if lang.to_lowercase() == "rust" || lang.to_lowercase() == "rs" {
                // --- Sovereign-Tier: Recursive Formal Verification ---
                let temp_art = Artifact {
                    name: name.clone(),
                    content: new_content.clone(),
                    language: lang.clone(),
                    version: current.version + 1,
                    history: vec![],
                    ast_versions: BTreeMap::new(),
                    proof_attachments: vec![],
                    metrics: new_metrics.clone(),
                    skeleton: String::new(),
                };
                match crate::engines::verification::InvariantChecker::verify_artifact(&temp_art)
                    .await
                {
                    Ok(Err(err)) => {
                        crate::log_warn!(
                            self.emit(StreamEvent::TokenReceived {
                                agent_id: "System".to_string(),
                                token: format!(
                                    "[verus] formal verification failed for \"{name}\":\n{err}\n"
                                ),
                            })
                            .await,
                            "Failed to emit verus error"
                        );
                        return Ok(ArtifactProcessOutcome::Ready(
                            vec![],
                            TurnOutcome::VerificationFailed,
                        ));
                    }
                    Err(e) => {
                        tracing::warn!("verus execution error: {:?}", e);
                    }
                    _ => {}
                }

                let sandbox_result = SandboxResult {
                    exit_code: 0,
                    stdout: new_content.clone(),
                    stderr: String::new(),
                    fuel_consumed: None,
                    elapsed_ms: 0,
                    resource_limit_hit: false,
                };
                let tmp = std::env::temp_dir();
                match LinterGuard::check(
                    &sandbox_result,
                    tmp.to_str().unwrap_or("/tmp"),
                    self.nix_env.as_ref(),
                )
                .await
                {
                    Ok(report) if !report.passed => return Ok(ArtifactProcessOutcome::Invalid),
                    Err(e) => {
                        crate::log_warn!(
                            self.emit(StreamEvent::TokenReceived {
                                agent_id: "System".to_string(),
                                token: format!("[lint] check failed for \"{name}\": {e}\n"),
                            })
                            .await,
                            "Failed to emit lint error"
                        );
                        return Ok(ArtifactProcessOutcome::Invalid);
                    }
                    _ => {}
                }
            }

            let skeleton = AstValidator::generate_skeleton(&new_content, &lang);

            changes.push(PreparedArtifactChange {
                name,
                lang,
                new_content,
                delta,
                new_metrics,
                node_updates,
                proof,
                skeleton,
            });
            turn_outcome = TurnOutcome::Compiled;
        }

        Ok(ArtifactProcessOutcome::Ready(changes, turn_outcome))
    }

    /// Sub-phase of `commit_turn`: apply artifact changes to sigma, build the Turn, run quality
    /// scoring and memory feedback, then advance the turn counter.
    /// Returns `(turn, quality_score, certainty, surprise, current_i, artifact_snapshot)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn apply_turn_to_state(
        &self,
        sigma_lock: &Mutex<ConversationState>,
        changes: Vec<PreparedArtifactChange>,
        turn_outcome: TurnOutcome,
        agent_id: &str,
        response: &str,
        prompt: &str,
        latency_ms: u64,
        nash_weight_updates: &BTreeMap<String, f64>,
        stall_risk: f64,
    ) -> Result<Option<(Turn, f64, f64, f64, u32, BTreeMap<String, Arc<Artifact>>)>> {
        if turn_outcome == TurnOutcome::VerificationFailed {
            let mut failures = self.verification_failures.lock().await;
            let count = failures.entry(agent_id.to_string()).or_insert(0);
            *count += 1;
            if *count > 2 {
                let mut sigma = sigma_lock.lock().await;
                let w = sigma
                    .agent_weights
                    .entry(agent_id.to_string())
                    .or_insert(1.0);
                *w = (*w * 0.5).max(0.0);
            }
        }

        let mut sigma = sigma_lock.lock().await;
        let current_i = sigma.iteration_index;
        let artifact_snapshot = sigma.artifacts.clone();

        let mut turn_diffs = Vec::new();
        for change in changes {
            let artifact_arc = sigma
                .artifacts
                .entry(change.name.clone())
                .or_insert_with(|| {
                    Arc::new(Artifact {
                        name: change.name.clone(),
                        language: change.lang.clone(),
                        content: String::new(),
                        version: 0,
                        history: vec![],
                        ast_versions: BTreeMap::new(),
                        proof_attachments: vec![],
                        metrics: ArtifactMetrics::default(),
                        skeleton: String::new(),
                    })
                });
            let mut artifact = (**artifact_arc).clone();
            artifact.history.push(change.delta.clone());
            artifact.content = change.new_content;
            artifact.version += 1;
            artifact.language = change.lang;
            artifact.metrics = change.new_metrics;
            artifact.skeleton = change.skeleton;
            artifact.proof_attachments.push(change.proof);
            for (node_id, content) in &change.node_updates {
                artifact
                    .ast_versions
                    .entry(node_id.clone())
                    .or_default()
                    .push((current_i, content.clone()));
            }
            *artifact_arc = Arc::new(artifact);
            for (node_id, _) in &change.node_updates {
                let node_p = sigma.node_consensus.entry(node_id.clone()).or_insert(0.1);
                let mut kalman = KalmanConvergence::new(*node_p);
                *node_p = kalman.update_adaptive(0.8, 1.0);
            }
            turn_diffs.push((change.name, change.delta));
        }

        let certainty = CertaintyAnalyzer::compute(response, 0.1);

        let surprise = {
            let mut se = self.surprise_engine.lock().await;
            se.record_prediction(agent_id, certainty);
            let s = se.compute_surprise(agent_id, turn_outcome);
            let current_w = sigma.agent_weights.get(agent_id).copied().unwrap_or(1.0);
            sigma.agent_weights.insert(
                agent_id.to_string(),
                se.calibrate_weight(agent_id, current_w),
            );
            s
        };
        if surprise > 0.5 && sigma.turns.len() >= 2 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[sandbox] High Surprise detected: {:.2}", surprise),
            })
            .await?;
        }

        let combined_diff_text: String = turn_diffs
            .iter()
            .map(|(_, d)| d.diff_text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let consistency_score = Some(ConsistencyScorer::score(response, &combined_diff_text));
        let dq_score = {
            let intell = self.intelligence.lock().await;
            intell.diff_quality_score(agent_id)
        };

        let mut turn = Turn {
            index: sigma.iteration_index,
            model_id: agent_id.to_string(),
            content: response.to_string(),
            timestamp: ConversationState::now(),
            diffs: turn_diffs,
            certainty: Some(certainty),
            outcome: turn_outcome,
            task_category: Some(TaskCategory::Research),
            structure: Some(ReasoningEngine::select_structure(
                TaskCategory::Research,
                agent_id,
            )),
            signature: vec![],
            surprise_signal: Some(surprise),
            consistency_score,
            diff_quality_score: Some(dq_score),
            persona_disclosure: None,
        };

        // Transparency duty: attach a signed PersonaDisclosure to every named-agent
        // turn *before* signing the turn, so the turn signature also covers it.
        {
            let principal = self.principal.lock().await;
            if principal.constraints.require_persona_disclosure && agent_id != "System" {
                use sha2::Digest;
                let prompt_hash: [u8; 32] = sha2::Sha256::digest(prompt.as_bytes()).into();
                let mut disclosure = PersonaDisclosure {
                    turn_index: turn.index,
                    agent_id: agent_id.to_string(),
                    persona_name: agent_id.to_string(),
                    system_prompt_hash: prompt_hash,
                    signature: vec![],
                };
                self.signer.sign_persona_disclosure(&mut disclosure);
                let pid = principal.id.to_string();
                let sid = sigma.session_id.clone();
                turn.persona_disclosure = Some(disclosure.clone());
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: pid,
                        event: FiduciaryDutyEvent::PersonaDisclosed(disclosure),
                        session_id: sid,
                        timestamp: ConversationState::now(),
                    })
                    .await,
                    "fiduciary persona signal emit failed"
                );
            }
        }

        let serialized = serde_json::to_vec(&turn)?;
        turn.signature = self.signer.sign(&serialized);

        let quality_score = {
            let base = {
                let obs = self.observer.lock().await;
                QualityScorer::score_with_context(&turn, &sigma.session_id, Some((&obs, agent_id)))
            };
            let surprise_penalty = (surprise - 0.5).max(0.0) * 0.6;
            let artifact_health = if !sigma.artifacts.is_empty() {
                sigma
                    .artifacts
                    .values()
                    .map(|a| a.metrics.health_score)
                    .sum::<f64>()
                    / sigma.artifacts.len() as f64
            } else {
                1.0
            };
            ((base - surprise_penalty) * artifact_health).max(0.0)
        };

        {
            let avg_quality = if !sigma.turns.is_empty() {
                let sum: f64 = sigma.turns.iter().filter_map(|t| t.certainty).sum();
                let count = sigma
                    .turns
                    .iter()
                    .filter(|t| t.certainty.is_some())
                    .count()
                    .max(1);
                sum / count as f64
            } else {
                0.5
            };
            let mut bridge = self.memory_bridge.lock().await;
            bridge.record_recall_feedback(current_i, quality_score - avg_quality);
            bridge.update_ranker(turn_outcome);
        }

        {
            let alert_info = {
                let intell = self.intelligence.lock().await;
                let recent_turns: Vec<Turn> = sigma
                    .turns
                    .iter()
                    .rev()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                intell.detect_regression(agent_id, &recent_turns)
            };
            if let Some(alert) = alert_info {
                let severity = if alert.baseline_mean > 0.0 {
                    (alert.baseline_mean - alert.recent_mean) / alert.baseline_mean
                } else {
                    0.0
                };
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!(
                        "[intelligence] Regression detected for {}: {:.2} -> {:.2} (severity {:.0}%)",
                        alert.agent_id, alert.baseline_mean, alert.recent_mean, severity * 100.0
                    ),
                })
                .await?;
                if severity > 0.3 {
                    let mut skips = self.skip_until.lock().await;
                    skips.insert(alert.agent_id.clone(), sigma.iteration_index + 2);
                }
            }
        }

        {
            let mut coll = self.collective.lock().await;
            coll.update_specialization(&turn);
        }

        ComputeManager::manage_budget(
            &mut sigma,
            CostEntry {
                turn_id: turn.index,
                model_id: agent_id.to_string(),
                usage: TokenUsage {
                    input_tokens: prompt.len() as u32 / 4,
                    output_tokens: response.len() as u32 / 4,
                    total_tokens: (prompt.len() + response.len()) as u32 / 4,
                },
                cost_usd: 0.01,
                latency_ms,
                timestamp: turn.timestamp,
            },
        );

        if sigma.turns.len() >= MAX_SESSION_TURNS {
            return Err(anyhow::anyhow!(
                "Session turn limit ({}) exceeded",
                MAX_SESSION_TURNS
            ));
        }

        // Fiduciary gate: Care/Autonomy duty — block or signal based on certainty and autonomy level.
        {
            let principal = self.principal.lock().await;
            if let Some(certainty) = turn.certainty {
                const CARE_CERTAINTY_FLOOR: f64 = 0.55;
                // Critically low certainty: block commits for SemiAutonomous (Care duty).
                const CARE_CERTAINTY_CRITICAL: f64 = 0.30;
                if certainty < CARE_CERTAINTY_FLOOR {
                    let event = FiduciaryDutyEvent::CertaintyGateFired {
                        turn_index: turn.index,
                        certainty,
                        threshold: CARE_CERTAINTY_FLOOR,
                    };
                    let pid = principal.id.to_string();
                    let sid = sigma.session_id.clone();
                    crate::log_warn!(
                        self.emit(StreamEvent::FiduciarySignal {
                            principal_id: pid,
                            event,
                            session_id: sid,
                            timestamp: ConversationState::now(),
                        })
                        .await,
                        "fiduciary signal emit failed"
                    );
                    if certainty < CARE_CERTAINTY_CRITICAL
                        && principal.constraints.max_autonomy_level == AutonomyLevel::SemiAutonomous
                    {
                        return Ok(None);
                    }
                }
            }
            // Sync principal session_id to sigma on first turn.
            if sigma.principal_id.is_none() {
                sigma.principal_id = Some(principal.id.to_string());
            }
        }

        sigma.push_turn(turn.clone());
        if let Err(e) = self.turn_tx.send(turn.clone()) {
            tracing::debug!(err = %e, "turn broadcast: no active swarm subscribers");
        }
        sigma.iteration_index += 1;

        // Account duty: persist decision to ledger.
        if let Some(ref principal_id) = sigma.principal_id {
            let event = DecisionLedger::commit_turn(&turn, &sigma.state_hash, agent_id);
            if let Err(e) = DecisionLedger::persist(
                self.state_manager.db(),
                &sigma.session_id,
                principal_id,
                turn.index,
                &event,
            ) {
                tracing::warn!(err = %e, "decision ledger persist failed");
            } else {
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: principal_id.clone(),
                        event,
                        session_id: sigma.session_id.clone(),
                        timestamp: ConversationState::now(),
                    })
                    .await,
                    "fiduciary signal emit failed"
                );
            }
        }

        {
            let task_cat = turn.task_category.unwrap_or(TaskCategory::Research);
            let quality = RewardVector::from_turn(&turn).weighted_score(task_cat);

            // Update EMA of recent quality (Task 8: adaptive evolution rate)
            let new_ema = loop {
                let old_bits = self
                    .recent_quality_ema
                    .load(std::sync::atomic::Ordering::Acquire);
                let old_val = f64::from_bits(old_bits);
                let candidate = 0.8 * old_val + 0.2 * quality;
                match self.recent_quality_ema.compare_exchange_weak(
                    old_bits,
                    candidate.to_bits(),
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Relaxed,
                ) {
                    Ok(_) => break candidate,
                    Err(_) => continue,
                }
            };

            // Adaptive evolution interval: faster when stalling, slower when improving
            let evolve_interval = if new_ema > 0.7 {
                10u32
            } else if new_ema > 0.4 {
                5
            } else {
                2
            };

            let mut evolver = self.prompt_evolver.lock().await;
            // Feed back to the specific template that was rendered, not the agent name
            if let Some(tmpl_id) = self.last_rendered_template_id.lock().await.as_deref() {
                evolver.record_outcome(tmpl_id, quality);
            }
            if sigma.iteration_index % evolve_interval == 0 {
                evolver.evolve();
                let mut cache = self.template_cache.write().await;
                for tmpl in evolver.population.iter().take(3) {
                    cache.insert(format!("{:?}", tmpl.task_category), tmpl.clone());
                }
            }
        }

        if matches!(
            turn_outcome,
            TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
        ) && let Some(certainty) = turn.certainty
            && certainty > 0.7
        {
            let seed_cat = turn.task_category.unwrap_or(TaskCategory::Research);
            let mut evolver = self.prompt_evolver.lock().await;
            evolver.seed_from_successful_turn(prompt, seed_cat);
        }

        if matches!(
            turn_outcome,
            TurnOutcome::TestsPassed | TurnOutcome::Compiled
        ) {
            for (name, artifact) in &sigma.artifacts {
                if let Ok(improved) =
                    crate::engines::self_improvement::SelfCodeModifier::propose_improvement(
                        name,
                        &artifact.content,
                    )
                    && improved != artifact.content
                    && AstValidator::validate(&improved, &artifact.language).is_ok()
                    && let Err(e) = self.file_writer.write_artifact(name, &improved).await
                {
                    warn!(artifact = %name, error = %e, "self-improvement write failed");
                }
            }
        }

        let prev_hash = sigma.state_hash;
        sigma.state_hash = HashChain::compute(&sigma, &prev_hash)?;

        sigma.agent_weights = match turn.task_category {
            Some(cat) => {
                crate::engines::consensus::InfluenceWeightManager::calculate_weights_for_category(
                    &sigma, cat, 0.9,
                )
            }
            None => crate::engines::consensus::InfluenceWeightManager::calculate_weights(&sigma),
        }
        .into_iter()
        .collect();

        for (id, nash_score) in nash_weight_updates {
            let w = sigma.agent_weights.entry(id.clone()).or_insert(0.5);
            *w = *w * 0.9 + nash_score * 0.1;
        }
        {
            let current_turn = sigma.iteration_index;
            let mut skips = self.skip_until.lock().await;
            for (id, &w) in &sigma.agent_weights {
                if w < 0.1 {
                    let until = skips.entry(id.clone()).or_insert(0);
                    *until = (*until).max(current_turn + 2);
                }
            }
        }

        if stall_risk > 0.6 {
            let mut coll = self.collective.lock().await;
            if coll.meta_optimizer.select_best(TaskCategory::Research)
                == MetaStrategy::DirectImplementation
            {
                coll.meta_optimizer
                    .record(MetaStrategy::DebateAndCritique, 0.6);
                tracing::info!(
                    stall_risk,
                    "stall detected: switching meta-strategy to DebateAndCritique"
                );
            }
        }

        let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
        let measurement = if response.contains("OPTIMAL") || response.contains("CONVERGED") {
            1.0
        } else {
            certainty * 0.8
        };
        let next_p = KalmanConvergence::new(current_p).update_adaptive(measurement, certainty);
        self.completion_probability
            .store(next_p.to_bits(), Ordering::Release);
        sigma.completion_probability = next_p;
        tracing::info!(
            turn = turn.index,
            agent = %agent_id,
            outcome = ?turn.outcome,
            response_len = turn.content.len(),
            convergence = next_p,
            "turn committed"
        );
        self.emit(StreamEvent::ConvergenceUpdated {
            p: next_p,
            certainty,
            agent_weights: sigma
                .agent_weights
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        })
        .await?;

        let mode_transition: Option<(String, String, String)> = {
            let mode_name = sigma.mode_library.current_name().to_string();

            // 1. Confidence-drop: rapid certainty decline triggers adversarial challenge
            if let Some(prev_turn) = sigma.turns.iter().rev().nth(1) {
                let prev_cert = prev_turn.certainty.unwrap_or(0.5);
                let curr_cert = turn.certainty.unwrap_or(0.5);
                if prev_cert - curr_cert > 0.3 && mode_name != "StressTest" {
                    sigma.mode_library.switch_to_name("StressTest");
                    tracing::info!(prev_cert, curr_cert, "confidence drop detected, switching to StressTest");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, format!("confidence drop {:.2} -> {:.2}, switching to StressTest", prev_cert, curr_cert)))
                } else {
                    None
                }
            } else {
                None
            }
            // 2. High-convergence: finalize when convergence probability is high
            .or_else(|| {
                if sigma.completion_probability > 0.85 && mode_name != "Convergence" {
                    sigma.mode_library.switch_to_name("Convergence");
                    tracing::info!(convergence = %sigma.completion_probability, "high convergence, switching to Convergence to finalize");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, format!("convergence {:.2} > 0.85, switching to Convergence", sigma.completion_probability)))
                } else {
                    None
                }
            })
            // 3. Oscillation: repeated content hashes indicate stuck loop
            .or_else(|| {
                if stall_risk > 0.6 && mode_name != "Socratic" && mode_name != "Generative" {
                    sigma.mode_library.switch_to_name("Socratic");
                    tracing::info!(stall_risk, "oscillation detected, switching to Socratic");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, format!("oscillation (stall_risk {:.2}), switching to Socratic", stall_risk)))
                } else {
                    None
                }
            })
            // 4. Mode return: after 2+ turns in a non-default mode, return if certainty recovers
            .or_else(|| {
                let turns_in_mode = sigma.turns.iter().rev()
                    .take_while(|t| t.model_id != "User")
                    .count();
                if turns_in_mode >= 2 && turn.certainty.unwrap_or(0.0) > 0.6
                    && mode_name != "Convergence"
                {
                    sigma.mode_library.switch_to_name("Convergence");
                    tracing::info!("certainty recovered, returning to Convergence");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, "certainty recovered above 0.6, returning to Convergence".to_string()))
                } else {
                    None
                }
            })
            // 5. Original stall detection: 3 consecutive low-certainty turns in Convergence
            .or_else(|| {
                let recent_stall = sigma.turns.iter().rev().take(3)
                    .filter(|t| t.certainty.unwrap_or(1.0) < 0.3 && t.model_id != "User")
                    .count() >= 3;
                if recent_stall && mode_name == "Convergence" {
                    sigma.mode_library.switch_to_name("Generative");
                    let new_name = sigma.mode_library.current_name().to_string();
                    Some((mode_name.clone(), new_name, "3 consecutive low-certainty turns, switching to Generative".to_string()))
                } else {
                    None
                }
            })
        };
        if let Some((old_name, new_name, reason)) = mode_transition {
            drop(sigma);
            self.emit(StreamEvent::ModeTransition {
                from_name: old_name,
                to_name: new_name.clone(),
                reason,
                synthesized: false,
            })
            .await?;
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[MODE → {}]\n", new_name),
            })
            .await?;
            sigma = sigma_lock.lock().await;
            sigma.mode_active_turns = 0;
        }

        // Clear novel_signal now that it has been consumed by this turn's prompt.
        sigma.novel_signal = None;

        // Surprise amplification: compute novel signal for next turn.
        if sigma.mode_library.current().surprise_handling
            == crate::types::mode::SurpriseHandling::Amplify
        {
            let mut scorer = crate::engines::novelty::NoveltyScorer::new();
            for t in sigma.turns.iter().rev().skip(1).take(5) {
                scorer.absorb(&t.content);
            }
            let novel = scorer.top_novel_sentences(response, 1);
            if let Some((sentence, score)) = novel.into_iter().next()
                && score > 0.3
            {
                sigma.novel_signal = Some(sentence);
            }
        }

        // Rejection loop: when OPTIMAL fires in RejectionLoop mode, inject rejection prompt.
        if response.contains("OPTIMAL")
            && sigma.mode_library.current().loop_structure
                == crate::types::mode::LoopStructure::RejectionLoop
            && !sigma.rejection_loop_active
        {
            sigma.rejection_loop_active = true;
            sigma.novel_signal = Some(
                "REJECTION TURN: The swarm has reached OPTIMAL. Your task now: reject the entire frame. \
                 What fundamental assumption is wrong? What would a structurally different approach look like? \
                 If you find a genuine new frame, begin with REJECT_FRAME: [description]. \
                 If you cannot find one, respond with FRAME_EXHAUSTED.".to_string()
            );
        } else if sigma.rejection_loop_active {
            if response.contains("REJECT_FRAME:") {
                let new_seed = response
                    .find("REJECT_FRAME:")
                    .map(|i| {
                        response[i + 13..]
                            .lines()
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string()
                    })
                    .unwrap_or_default();
                sigma.rejection_loop_active = false;
                sigma.novel_signal = if new_seed.is_empty() {
                    None
                } else {
                    Some(new_seed)
                };
                let old_name = sigma.mode_library.current_name().to_string();
                sigma.mode_library.switch_to_name("Generative");
                let new_name = sigma.mode_library.current_name().to_string();
                sigma.mode_active_turns = 0;
                drop(sigma);
                self.emit(StreamEvent::ModeTransition {
                    from_name: old_name,
                    to_name: new_name.clone(),
                    reason: "Rejection frame found — switching to Generative".to_string(),
                    synthesized: false,
                })
                .await?;
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[MODE → {}]\n", new_name),
                })
                .await?;
                sigma = sigma_lock.lock().await;
            } else if response.contains("FRAME_EXHAUSTED") {
                sigma.rejection_loop_active = false;
            }
        }

        // Mode synthesis: track active turns and inject synthesis prompt when stalled.
        {
            let avg_certainty = if !sigma.turns.is_empty() {
                let sum: f64 = sigma
                    .turns
                    .iter()
                    .rev()
                    .take(6)
                    .filter_map(|t| t.certainty)
                    .sum();
                let count = sigma
                    .turns
                    .iter()
                    .rev()
                    .take(6)
                    .filter(|t| t.certainty.is_some())
                    .count()
                    .max(1);
                sum / count as f64
            } else {
                1.0
            };
            sigma.mode_active_turns = sigma.mode_active_turns.saturating_add(1);
            if sigma.mode_active_turns >= 6 && avg_certainty < 0.4 && sigma.novel_signal.is_none() {
                let current_mode_name = sigma.mode_library.current_name().to_string();
                let active_turns = sigma.mode_active_turns;
                sigma.novel_signal = Some(format!(
                    "[META-MODE-SYNTHESIS] The current mode \"{}\" has not produced progress in {} turns. \
                     Analyze the conversation and propose a new mode definition as JSON on a single code block:\n\
                     ```json\n\
                     {{\"name\": \"...\", \"description\": \"...\", \
                     \"context_distribution\": \"Shared\", \
                     \"convergence_direction\": \"TowardAgreement\", \
                     \"surprise_handling\": \"Neutral\", \
                     \"termination\": {{\"OptimalSignal\": null}}, \
                     \"role_assignment\": \"Homogeneous\", \
                     \"loop_structure\": \"Linear\", \
                     \"prompt_prefix\": \"...\"}}\n\
                     ```\n\
                     Valid values: context_distribution: Shared|Divergent|RoleFiltered; \
                     convergence_direction: TowardAgreement|TowardDivergence|TowardTradeoffMap|TowardNovelty; \
                     surprise_handling: Amplify|Suppress|Neutral; \
                     termination: {{\"OptimalSignal\":null}} or {{\"Exhaustion\":{{\"max_turns\":N}}}} or {{\"RejectionCycles\":{{\"n\":N}}}}; \
                     role_assignment: Homogeneous|AdversarialPairs|Specialized; \
                     loop_structure: Linear|RejectionLoop|TreeSearch",
                    current_mode_name, active_turns
                ));
            }
        }

        // Try to parse a mode synthesized by the agent from this response.
        if let Some(new_mode) = crate::types::mode::ModeLibrary::try_parse_synthesized(
            response,
            format!(
                "Synthesized after {} turns in {}",
                sigma.mode_active_turns,
                sigma.mode_library.current_name()
            ),
        ) {
            let old_name = sigma.mode_library.current_name().to_string();
            let new_name = new_mode.name.clone();
            let idx = sigma.mode_library.upsert(new_mode);
            sigma.mode_library.switch_to_index(idx);
            sigma.mode_active_turns = 0;
            drop(sigma);
            self.emit(StreamEvent::ModeTransition {
                from_name: old_name,
                to_name: new_name.clone(),
                reason: "Mode synthesized by agent".to_string(),
                synthesized: true,
            })
            .await?;
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[MODE SYNTHESIZED → {}]\n", new_name),
            })
            .await?;
            sigma = sigma_lock.lock().await;
        }

        if let Err(e) = InvariantChecker::check_all(&sigma) {
            sigma.artifacts = artifact_snapshot;
            if let Some(t) = sigma.turns.last_mut() {
                t.outcome = TurnOutcome::RolledBack;
            }
            let should_skip = {
                let mut counters = self.rollback_counters.lock().await;
                let count = counters.entry(agent_id.to_string()).or_insert(0);
                *count += 1;
                let exceeded = *count >= 3;
                if exceeded {
                    *count = 0;
                }
                exceeded
            };
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[rollback] Invariant violation: {e}"),
            })
            .await?;
            if should_skip {
                self.emit(StreamEvent::TokenReceived {
                    agent_id: "System".to_string(),
                    token: format!("[rollback] Agent {agent_id} exceeded consecutive rollbacks"),
                })
                .await?;
                let mut skips = self.skip_until.lock().await;
                skips.insert(agent_id.to_string(), sigma.iteration_index + 1);
            }
            turn.outcome = TurnOutcome::RolledBack;
            {
                let intell = self.intelligence.lock().await;
                intell.update_profile_with_latency(&turn, 0.0, latency_ms);
            }
            {
                let mut ctx = self.session_ctx.lock().await;
                ctx.record_turn(TurnOutcome::RolledBack);
            }
            return Ok(None);
        }
        {
            let mut counters = self.rollback_counters.lock().await;
            counters.insert(agent_id.to_string(), 0);
        }
        self.state_manager.checkpoint_async(&sigma).await?;
        self.emit(StreamEvent::CheckpointWritten(current_i)).await?;

        Ok(Some((
            turn,
            quality_score,
            certainty,
            surprise,
            current_i,
            artifact_snapshot,
        )))
    }

    /// Sub-phase of `commit_turn`: file I/O, verification, post-commit state updates, and
    /// convergence reporting. `sigma` must NOT be held by the caller on entry.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn finalize_committed_turn(
        &self,
        sigma_lock: &Mutex<ConversationState>,
        mut turn: Turn,
        quality_score: f64,
        next_p: f64,
        agent_id: &str,
        response: &str,
        latency_ms: u64,
    ) -> Result<bool> {
        // Snapshot what we need for I/O from sigma, then drop lock.
        let (io_artifacts, all_artifacts, turn_diffs) = {
            let sigma = sigma_lock.lock().await;
            let io: Vec<(String, Arc<Artifact>)> = turn
                .diffs
                .iter()
                .filter_map(|(name, _)| {
                    sigma
                        .artifacts
                        .get(name)
                        .map(|a| (name.clone(), Arc::clone(a)))
                })
                .collect();
            let all = sigma.artifacts.clone();
            let diffs = turn.diffs.clone();
            (io, all, diffs)
        };

        for (name, artifact) in &io_artifacts {
            match self.file_writer.write_artifact_with_proof(artifact).await {
                Ok(WriteOutcome::Written(path)) => {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[write] {}\n", path.display()),
                    })
                    .await?;
                }
                Ok(WriteOutcome::Skipped(_)) => {}
                Ok(WriteOutcome::VerificationFailed(stderr)) => {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "[write] {name}: verification failed, original restored\n{stderr}"
                        ),
                    })
                    .await?;
                }
                Err(e) => {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!("[write] error writing {name}: {e}\n"),
                    })
                    .await?;
                }
            }
        }

        let verification_results = if !io_artifacts.is_empty() {
            self.run_verification(&all_artifacts, &turn_diffs).await
        } else {
            vec![]
        };
        for (tool_name, output, passed) in &verification_results {
            let status = if *passed { "PASS" } else { "FAIL" };
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[verify] {} [{}]\n{}\n",
                    tool_name,
                    status,
                    Self::truncate_str(output, 500)
                ),
            })
            .await?;
        }
        if !verification_results.is_empty() && verification_results.iter().all(|(_, _, p)| *p) {
            turn.outcome = TurnOutcome::TestsPassed;
            let current_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));
            self.completion_probability
                .store((current_p + 0.15).min(1.0).to_bits(), Ordering::Release);
        }

        // Re-acquire lock for remaining state updates.
        let mut sigma = sigma_lock.lock().await;
        sigma.last_verification = verification_results
            .iter()
            .map(|(name, output, passed)| (name.clone(), output.clone(), *passed))
            .collect();
        {
            let intell = self.intelligence.lock().await;
            intell.update_profile_with_latency(&turn, quality_score, latency_ms);
        }
        {
            let mut ctx = self.session_ctx.lock().await;
            ctx.record_turn(turn.outcome);
        }
        self.swarm.broadcast_turn(turn.clone())?;

        {
            let prev = sigma.state_hash;
            sigma.state_hash = HashChain::compute(&sigma, &prev)?;
        }

        if let Some(ref auditor_tx) = self.auditor_tx
            && !auditor_tx.is_closed()
            && let Err(e) = auditor_tx.send(sigma.clone()).await
        {
            self.emit(StreamEvent::Error(format!(
                "auditor send failed, auditor task dead: {e}"
            )))
            .await?;
        }

        if let Some(ref mut root) = sigma.goal_tree.root {
            PlanningEngine::update_goal_status(root);
        }

        {
            let mut viz = self.viz.lock().await;
            let metrics = viz.compute_metrics(&sigma);
            drop(viz);
            self.emit(StreamEvent::GodViewUpdated {
                frame: metrics.frame,
                avg_certainty: metrics.avg_certainty,
                avg_surprise: metrics.avg_surprise,
                agent_count: metrics.agent_count,
            })
            .await?;
        }

        self.emit(StreamEvent::ArtifactsUpdated(
            sigma
                .artifacts
                .iter()
                .map(|(name, a)| crate::types::events::ArtifactSnapshot {
                    name: name.clone(),
                    skeleton: a.skeleton.clone(),
                    version: a.version,
                    diff_count: a.history.len(),
                })
                .collect(),
        ))
        .await?;

        self.emit(StreamEvent::TokenReceived {
            agent_id: "System".to_string(),
            token: format!(
                "\n[Turn Complete | P(C): {:.2} | Hash: {:02x?}]\n",
                next_p,
                &sigma.state_hash[..4]
            ),
        })
        .await?;
        self.emit(StreamEvent::TurnComplete(turn.clone())).await?;

        {
            let mut audit_rx: MutexGuard<'_, mpsc::UnboundedReceiver<AuditAlert>> =
                self.audit_rx.lock().await;
            while let Ok(alert) = audit_rx.try_recv() {
                // Only act on alerts for the current iteration; stale alerts are not indicative of tampering.
                if alert.iteration_index == sigma.iteration_index {
                    self.emit(StreamEvent::TokenReceived {
                        agent_id: "System".to_string(),
                        token: format!(
                            "[audit] Hash mismatch at iteration {}: expected {:02x?}, got {:02x?}\n",
                            alert.iteration_index, &alert.expected_hash[..4], &alert.actual_hash[..4]
                        ),
                    })
                    .await?;
                }
            }
        }

        {
            let session_id = sigma.session_id.clone();
            let compiled = matches!(
                turn.outcome,
                TurnOutcome::Compiled | TurnOutcome::TestsPassed | TurnOutcome::AdvancedConvergence
            );
            let content_key = Self::truncate_str(response, 500).to_string();
            let preview = Self::truncate_str(response, 200).to_string();
            let metadata = serde_json::json!({"content": preview, "outcome": format!("{:?}", turn.outcome), "agent": agent_id}).to_string();
            let record = MemoryRecord {
                turn_id: turn.index,
                session_id: session_id.clone(),
                embedding: vec![],
                content_hash: content_key,
                timestamp: turn.timestamp,
                metadata_json: metadata,
                outcome: Some(OutcomeRecord {
                    compiled,
                    tests_passed: turn.outcome == TurnOutcome::TestsPassed,
                    quality_delta: 0.0,
                    was_rolled_back: false,
                    convergence_contribution: next_p,
                }),
                is_negative: false,
            };
            let mut bridge = self.memory_bridge.lock().await;
            bridge.open_session(session_id.clone());
            bridge.push_record(&session_id, record);
        }

        if next_p > 0.70 && next_p <= 0.85 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[convergence] p={next_p:.2}, moderate confidence, continuing refinement"
                ),
            })
            .await?;
        } else if next_p > 0.85 && next_p <= 0.95 {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!("[convergence] p={next_p:.2}, high confidence, final polish"),
            })
            .await?;
        }

        let is_converged = next_p > 0.95;
        if is_converged {
            let eval = SelfImprovementEngine::evaluate_session(&sigma);
            let report = AnalyticsEngine::generate_report(&sigma);
            let exec_summary = ConvergenceReport::generate(&sigma);
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "[self-improve] {:?} | [analytics] {:?} | [release] {}",
                    eval, report, exec_summary
                ),
            })
            .await?;
            if let Some(mortem) = PostMortemGenerator::generate(&sigma) {
                let bridge = self.memory_bridge.lock().await;
                if let Err(e) = bridge
                    .store_failure_lesson_async(&sigma.session_id, &mortem)
                    .await
                {
                    warn!(session = %sigma.session_id, error = %e, "failed to store post-mortem");
                }
            }
            let mut session_store =
                MemoryStore::new(&format!("/tmp/crosstalk-{}", sigma.session_id));
            session_store.init().await?;
            self.session_memory_map
                .lock()
                .await
                .insert(sigma.session_id.clone(), Arc::new(session_store));
        }

        Ok(is_converged)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, fields(agent = %agent_id))]
    pub(super) async fn commit_turn(
        &self,
        sigma_lock: &Mutex<ConversationState>,
        changes: Vec<PreparedArtifactChange>,
        turn_outcome: TurnOutcome,
        agent_id: &str,
        response: &str,
        prompt: &str,
        latency_ms: u64,
        nash_weight_updates: BTreeMap<String, f64>,
        stall_risk: f64,
    ) -> Result<bool> {
        let Some((turn, quality_score, _certainty, _surprise, _current_i, _artifact_snapshot)) =
            self.apply_turn_to_state(
                sigma_lock,
                changes,
                turn_outcome,
                agent_id,
                response,
                prompt,
                latency_ms,
                &nash_weight_updates,
                stall_risk,
            )
            .await?
        else {
            return Ok(false);
        };

        // next_p was stored atomically in apply_turn_to_state; read it back.
        let next_p = f64::from_bits(self.completion_probability.load(Ordering::Acquire));

        self.finalize_committed_turn(
            sigma_lock,
            turn,
            quality_score,
            next_p,
            agent_id,
            response,
            latency_ms,
        )
        .await
    }
}
