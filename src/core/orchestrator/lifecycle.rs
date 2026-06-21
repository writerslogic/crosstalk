use super::*;

impl Orchestrator {
    pub fn rewind(&self, index: u32) -> Result<ConversationState> {
        self.state_manager
            .restore(index)?
            .context(format!("Failed to rewind to index {index}"))
    }

    pub fn resume(&self, index: u32) -> Result<ConversationState> {
        self.rewind(index)
    }

    pub async fn finalize_session(&self, sigma_lock: Arc<Mutex<ConversationState>>) -> Result<()> {
        let sigma = sigma_lock.lock().await;
        let eval = SelfImprovementEngine::evaluate_session(&sigma);
        self.emit(StreamEvent::TokenReceived {
            agent_id: "System".to_string(),
            token: format!(
                "
[Self-Improvement] Session Evaluation: convergence_p={:.2}, failure_rate={:.2}
",
                eval.metrics.get("convergence_p").unwrap_or(&0.0),
                eval.metrics.get("failure_rate").unwrap_or(&0.0)
            ),
        })
        .await?;

        if let Some(pm) = PostMortemGenerator::generate(&sigma) {
            self.emit(StreamEvent::TokenReceived {
                agent_id: "System".to_string(),
                token: format!(
                    "
[Self-Improvement] Post-Mortem: detected root cause {:?}
",
                    pm.root_cause
                ),
            })
            .await?;
        }

        {
            let intell = self.intelligence.lock().await;
            let templates_arc = intell.templates();
            let calibration_arc = intell.calibration();
            let mut templates = templates_arc.write().await;
            let mut calibration = calibration_arc.write().await;
            let mut learner = crate::engines::self_improvement::ContinuousLearner {
                prompt_library: &mut templates,
                calibration: &mut calibration,
            };
            learner.run(
                &sigma,
                PostMortemGenerator::generate(&sigma),
                0.5,
                sigma.completion_probability,
            );
        }

        // Persist Elo ratings for cross-session continuity.
        {
            let obs = self.observer.lock().await;
            let elo_json = obs.export_elo_ratings();
            let record = MemoryRecord {
                turn_id: 0,
                session_id: "elo_ratings".to_string(),
                embedding: vec![0.0; 64],
                content_hash: "elo_snapshot".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                metadata_json: elo_json,
                outcome: None,
                is_negative: false,
            };
            self.memory_store
                .sessions
                .entry("elo_ratings".to_string())
                .or_default()
                .push(Arc::new(record));
        }

        // Persist prompt evolver population for cross-session evolution.
        {
            let evolver = self.prompt_evolver.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("prompt_population".to_string())
                .or_default()
                .push(Arc::new(MemoryRecord {
                    turn_id: 0,
                    session_id: "prompt_population".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "prompt_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: evolver.export_state_json(),
                    outcome: None,
                    is_negative: false,
                }));
        }

        // Persist topology scores for cross-session learning.
        {
            let topo = self.topology.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("topology_scores".to_string())
                .or_default()
                .push(Arc::new(MemoryRecord {
                    turn_id: 0,
                    session_id: "topology_scores".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "topology_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: topo.export_scores_json(),
                    outcome: None,
                    is_negative: false,
                }));
        }

        // Persist collective agent profiles and meta-strategy outcomes.
        {
            let coll = self.collective.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("collective_profiles".to_string())
                .or_default()
                .push(Arc::new(MemoryRecord {
                    turn_id: 0,
                    session_id: "collective_profiles".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "collective_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: coll.export_state_json(),
                    outcome: None,
                    is_negative: false,
                }));
        }

        // Persist memory ranker weights for cross-session recall tuning.
        {
            let bridge = self.memory_bridge.lock().await;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.memory_store
                .sessions
                .entry("ranker_weights".to_string())
                .or_default()
                .push(Arc::new(MemoryRecord {
                    turn_id: 0,
                    session_id: "ranker_weights".to_string(),
                    embedding: vec![0.0; 64],
                    content_hash: "ranker_snapshot".to_string(),
                    timestamp: ts,
                    metadata_json: bridge.export_ranker_weights_json(),
                    outcome: None,
                    is_negative: false,
                }));
        }

        // Distill and persist a SessionLesson for future sessions.
        {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let task_summary = sigma
                .turns
                .first()
                .map(|t| t.content.chars().take(200).collect::<String>())
                .unwrap_or_default();
            let topo = self.topology.lock().await;
            let topology_sequence: Vec<String> = topo
                .history
                .iter()
                .map(|(_, t, _)| format!("{t:?}"))
                .collect();
            drop(topo);
            let final_outcome = if sigma.completion_probability > 0.7 {
                "succeeded"
            } else if sigma.completion_probability > 0.3 {
                "stalled"
            } else {
                "failed"
            }
            .to_string();
            let obs = self.observer.lock().await;
            let winning_model = obs
                .ranked_agents()
                .into_iter()
                .next()
                .map(|(id, _)| id)
                .unwrap_or_default();
            drop(obs);
            let quality_trajectory: Vec<f64> = sigma
                .turns
                .iter()
                .rev()
                .take(10)
                .map(|t| {
                    RewardVector::from_turn(t)
                        .weighted_score(t.task_category.unwrap_or(TaskCategory::Research))
                })
                .collect();
            let lesson = SessionLesson {
                task_summary,
                topology_sequence,
                final_outcome,
                winning_model,
                quality_trajectory,
                turn_count: sigma.iteration_index,
                timestamp: ts,
            };
            if let Ok(lesson_json) = serde_json::to_string(&lesson) {
                self.memory_store
                    .sessions
                    .entry("session_lessons".to_string())
                    .or_default()
                    .push(Arc::new(MemoryRecord {
                        turn_id: 0,
                        session_id: "session_lessons".to_string(),
                        embedding: vec![0.0; 64],
                        content_hash: "lesson_snapshot".to_string(),
                        timestamp: ts,
                        metadata_json: lesson_json,
                        outcome: None,
                        is_negative: false,
                    }));
            }
        }

        // Enforce data retention policy (fiduciary duty).
        {
            let principal = self.principal.lock().await;
            if let Ok(Some(event)) = crate::engines::data_minimizer::DataMinimizer::enforce(
                self.state_manager.db(),
                &sigma.session_id,
                &principal.constraints,
            ) {
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: principal.id.to_string(),
                        event,
                        session_id: sigma.session_id.clone(),
                        timestamp: ConversationState::now(),
                    })
                    .await,
                    "data minimizer fiduciary signal failed"
                );
            }
        }

        Ok(())
    }

    pub fn get_completion_probability(&self) -> f64 {
        f64::from_bits(
            self.completion_probability
                .load(std::sync::atomic::Ordering::Acquire),
        )
    }
}
