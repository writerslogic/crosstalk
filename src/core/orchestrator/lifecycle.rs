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

    /// Append a cross-session snapshot record (Elo ratings, evolver population,
    /// topology scores, etc.) under a well-known pseudo-session key.
    fn persist_snapshot(&self, key: &str, content_hash: &str, metadata_json: String) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.memory_store
            .sessions
            .entry(key.to_string())
            .or_default()
            .push(Arc::new(MemoryRecord {
                turn_id: 0,
                session_id: key.to_string(),
                embedding: vec![0.0; 64],
                content_hash: content_hash.to_string(),
                timestamp,
                metadata_json,
                outcome: None,
                is_negative: false,
            }));
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

        // Snapshot the values the rest of finalization needs, then release the
        // sigma lock: the persistence below touches only engines/memory_store,
        // so holding the conversation-state lock across it is unnecessary.
        let session_id = sigma.session_id.clone();
        let lesson_task_summary = sigma
            .turns
            .first()
            .map(|t| t.content.chars().take(200).collect::<String>())
            .unwrap_or_default();
        let lesson_completion_p = sigma.completion_probability;
        let lesson_turn_count = sigma.iteration_index;
        let lesson_quality_trajectory: Vec<f64> = sigma
            .turns
            .iter()
            .rev()
            .take(10)
            .map(|t| {
                RewardVector::from_turn(t)
                    .weighted_score(t.task_category.unwrap_or(TaskCategory::Research))
            })
            .collect();
        drop(sigma);

        // Persist Elo ratings for cross-session continuity.
        let elo_json = self.observer.lock().await.export_elo_ratings();
        self.persist_snapshot("elo_ratings", "elo_snapshot", elo_json);

        // Persist prompt evolver population for cross-session evolution.
        let prompt_json = self.prompt_evolver.lock().await.export_state_json();
        self.persist_snapshot("prompt_population", "prompt_snapshot", prompt_json);

        // Persist topology scores for cross-session learning.
        let topology_json = self.topology.lock().await.export_scores_json();
        self.persist_snapshot("topology_scores", "topology_snapshot", topology_json);

        // Persist collective agent profiles and meta-strategy outcomes.
        let collective_json = self.collective.lock().await.export_state_json();
        self.persist_snapshot(
            "collective_profiles",
            "collective_snapshot",
            collective_json,
        );

        // Persist memory ranker weights for cross-session recall tuning.
        let ranker_json = self.memory_bridge.lock().await.export_ranker_weights_json();
        self.persist_snapshot("ranker_weights", "ranker_snapshot", ranker_json);

        // Distill and persist a SessionLesson for future sessions.
        {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let task_summary = lesson_task_summary;
            let topo = self.topology.lock().await;
            let topology_sequence: Vec<String> = topo
                .history
                .iter()
                .map(|(_, t, _)| format!("{t:?}"))
                .collect();
            drop(topo);
            let final_outcome = if lesson_completion_p > 0.7 {
                "succeeded"
            } else if lesson_completion_p > 0.3 {
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
            let quality_trajectory = lesson_quality_trajectory;
            let lesson = SessionLesson {
                task_summary,
                topology_sequence,
                final_outcome,
                winning_model,
                quality_trajectory,
                turn_count: lesson_turn_count,
                timestamp: ts,
            };
            if let Ok(lesson_json) = serde_json::to_string(&lesson) {
                self.persist_snapshot("session_lessons", "lesson_snapshot", lesson_json);
            }
        }

        // Enforce data retention policy (fiduciary duty).
        {
            let principal = self.principal.lock().await;
            if let Ok(Some(event)) = crate::engines::data_minimizer::DataMinimizer::enforce(
                self.state_manager.db(),
                &session_id,
                &principal.constraints,
            ) {
                crate::log_warn!(
                    self.emit(StreamEvent::FiduciarySignal {
                        principal_id: principal.id.to_string(),
                        event,
                        session_id: session_id.clone(),
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
