use crate::types::conversation::Turn;
use crate::types::events::ArtifactSnapshot;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Streaming,
    Paused,
    Rewinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedPane {
    GhostStream,
    Artifacts,
    EntropyMap,
    Events,
}

impl FocusedPane {
    pub fn cycle(self) -> Self {
        match self {
            Self::GhostStream => Self::Artifacts,
            Self::Artifacts => Self::EntropyMap,
            Self::EntropyMap => Self::Events,
            Self::Events => Self::GhostStream,
        }
    }
}

/// Per-artifact entropy entry: artifact name → [(agent_id, disagreement_score)]
#[derive(Debug, Clone)]
pub struct EntropyRow {
    pub artifact: String,
    pub agents: Vec<(String, f64)>,
}

pub struct App {
    pub session_id: String,
    pub turn_index: u32,
    pub streaming_buffer: String,
    pub artifacts: Vec<ArtifactSnapshot>,
    pub convergence: f64,
    pub certainty: f64,
    pub agent_weights: HashMap<String, f64>,
    /// Capped at 50 items; oldest evicted when full
    pub recent_events: VecDeque<String>,
    pub mode: AppMode,
    /// Row offset for events log scroll
    pub scroll_offset: usize,
    pub events_auto_scroll: bool,
    pub ghost_scroll: usize,
    pub ghost_auto_scroll: bool,
    pub artifact_scroll: usize,
    pub entropy_scroll: usize,
    /// Set to true to signal the render loop to exit
    pub shutdown: bool,
    /// Text being typed in the inject dialog
    pub inject_buffer: String,
    /// Whether the inject overlay is visible
    pub showing_inject: bool,
    /// Which pane has keyboard focus
    pub focused_pane: FocusedPane,
    /// 2D entropy heatmap data: rows=artifacts, cols=agents
    pub entropy_scores: Vec<EntropyRow>,
    /// Ordered list of active agent IDs (columns for heatmap)
    pub agent_list: Vec<String>,
    pub godview_certainty: f64,
    pub godview_surprise: f64,
    pub godview_frame: u64,
    /// Measured frames per second
    pub fps: f32,
    /// Tracks last render time for FPS computation
    last_render: Instant,
    /// Recent diff sizes per artifact per agent: artifact → agent → [diff_len]
    artifact_change_history: HashMap<String, HashMap<String, Vec<usize>>>,
    pub current_mode_name: String,
    pub mode_library_size: usize,
    pub showing_help: bool,
}

impl App {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            turn_index: 0,
            streaming_buffer: String::new(),
            artifacts: Vec::new(),
            convergence: 0.0,
            certainty: 0.0,
            agent_weights: HashMap::new(),
            recent_events: VecDeque::new(),
            mode: AppMode::Streaming,
            scroll_offset: 0,
            events_auto_scroll: true,
            ghost_scroll: 0,
            ghost_auto_scroll: true,
            artifact_scroll: 0,
            entropy_scroll: 0,
            shutdown: false,
            inject_buffer: String::new(),
            showing_inject: false,
            focused_pane: FocusedPane::GhostStream,
            entropy_scores: Vec::new(),
            agent_list: Vec::new(),
            godview_certainty: 0.0,
            godview_surprise: 0.0,
            godview_frame: 0,
            fps: 0.0,
            last_render: Instant::now(),
            artifact_change_history: HashMap::new(),
            current_mode_name: "Convergence".to_string(),
            mode_library_size: 6,
            showing_help: false,
        }
    }

    pub fn push_token(&mut self, agent_id: &str, token: &str) {
        if !self.agent_list.contains(&agent_id.to_string()) {
            self.agent_list.push(agent_id.to_string());
        }
        if !self.streaming_buffer.is_empty()
            && !self.streaming_buffer.ends_with(' ')
            && !token.starts_with(' ')
            && !token.starts_with('\n')
        {
            // Potential word break across tokens
        }

        let prefix = if self.streaming_buffer.is_empty() || self.streaming_buffer.ends_with('\n') {
            format!("[{agent_id}] ")
        } else {
            String::new()
        };

        self.streaming_buffer.push_str(&prefix);
        let sanitized_token: String = token
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect();
        self.streaming_buffer.push_str(&sanitized_token);
        const BUF_CAP: usize = 50 * 1024;
        if self.streaming_buffer.len() > BUF_CAP {
            let trim = self.streaming_buffer.len() - BUF_CAP;
            if trim >= self.streaming_buffer.len() {
                self.streaming_buffer.clear();
            } else {
                let safe = self
                    .streaming_buffer
                    .char_indices()
                    .find(|(i, _)| *i >= trim)
                    .map(|(i, _)| i)
                    .unwrap_or(self.streaming_buffer.len());
                self.streaming_buffer.drain(..safe);
            }
        }

        if self.ghost_auto_scroll {
            let line_count = self.streaming_buffer.chars().filter(|&c| c == '\n').count();
            self.ghost_scroll = line_count;
        }
    }

    pub fn commit_turn(&mut self, turn: &Turn) {
        self.turn_index = turn.index + 1;

        // Append turn separator instead of clearing — keep rolling conversation history
        let outcome_label = match turn.outcome {
            crate::types::conversation::TurnOutcome::Compiled => "COMPILED",
            crate::types::conversation::TurnOutcome::TestsPassed => "PASS",
            crate::types::conversation::TurnOutcome::AdvancedConvergence => "CONVERGING",
            crate::types::conversation::TurnOutcome::RolledBack => "ROLLBACK",
            crate::types::conversation::TurnOutcome::Rejected => "REJECTED",
            crate::types::conversation::TurnOutcome::Stalled => "STALLED",
            crate::types::conversation::TurnOutcome::Unknown => "OK",
            crate::types::conversation::TurnOutcome::VerificationFailed => "VERIFY_FAIL",
        };
        let certainty_str = turn
            .certainty
            .map(|c| format!("{:.0}%", c * 100.0))
            .unwrap_or_else(|| "?".to_string());
        self.streaming_buffer.push_str(&format!(
            "\n--- Turn {} | {} | {} | cert {} ---\n",
            turn.index, turn.model_id, outcome_label, certainty_str
        ));

        self.push_event(format!(
            "Turn {} by {} ({:?})",
            turn.index, turn.model_id, turn.outcome
        ));

        // Track agent in agent_list
        if !self.agent_list.contains(&turn.model_id) {
            self.agent_list.push(turn.model_id.clone());
        }

        // Update per-artifact change history for entropy computation
        for (artifact_name, diff) in &turn.diffs {
            let added = diff
                .diff_text
                .lines()
                .filter(|l| l.starts_with('+'))
                .count();
            let removed = diff
                .diff_text
                .lines()
                .filter(|l| l.starts_with('-'))
                .count();
            self.push_event(format!(
                "  [diff] {} (+{}, -{})",
                artifact_name, added, removed
            ));

            let entry = self
                .artifact_change_history
                .entry(artifact_name.clone())
                .or_default();
            entry
                .entry(turn.model_id.clone())
                .or_default()
                .push(diff.diff_text.len());
        }

        self.recompute_entropy();
    }

    fn recompute_entropy(&mut self) {
        self.entropy_scores.clear();
        for (artifact_name, agent_map) in &self.artifact_change_history {
            if agent_map.len() < 2 {
                // Only one agent touched this artifact — no disagreement possible
                let single_score: Vec<(String, f64)> =
                    agent_map.keys().map(|a| (a.clone(), 0.0)).collect();
                self.entropy_scores.push(EntropyRow {
                    artifact: artifact_name.clone(),
                    agents: single_score,
                });
                continue;
            }

            // Compute per-agent mean diff size
            let agent_means: Vec<(String, f64)> = agent_map
                .iter()
                .map(|(agent, sizes)| {
                    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
                    (agent.clone(), mean)
                })
                .collect();

            let max_mean = agent_means.iter().map(|(_, m)| *m).fold(0.0_f64, f64::max);

            let agents: Vec<(String, f64)> = if max_mean == 0.0 {
                agent_means.into_iter().map(|(a, _)| (a, 0.0)).collect()
            } else {
                let global_mean =
                    agent_means.iter().map(|(_, m)| m).sum::<f64>() / agent_means.len() as f64;
                agent_means
                    .into_iter()
                    .map(|(a, m)| {
                        let deviation = (m - global_mean).abs() / max_mean;
                        (a, deviation.clamp(0.0, 1.0))
                    })
                    .collect()
            };

            self.entropy_scores.push(EntropyRow {
                artifact: artifact_name.clone(),
                agents,
            });
        }

        self.entropy_scores
            .sort_by(|a, b| a.artifact.cmp(&b.artifact));
    }

    pub fn set_convergence(&mut self, p: f64, certainty: f64) {
        self.convergence = p;
        self.certainty = certainty;
    }

    pub fn push_event(&mut self, event: String) {
        self.recent_events.push_back(event);
        if self.recent_events.len() > 50 {
            self.recent_events.pop_front();
            self.scroll_offset = self.scroll_offset.saturating_sub(1);
        }
        if self.events_auto_scroll {
            self.scroll_offset = self.recent_events.len().saturating_sub(1);
        }
    }

    pub fn scroll_up(&mut self) {
        match self.focused_pane {
            FocusedPane::GhostStream => {
                self.ghost_auto_scroll = false;
                self.ghost_scroll = self.ghost_scroll.saturating_sub(1);
            }
            FocusedPane::Artifacts => self.artifact_scroll = self.artifact_scroll.saturating_sub(1),
            FocusedPane::EntropyMap => self.entropy_scroll = self.entropy_scroll.saturating_sub(1),
            FocusedPane::Events => {
                self.events_auto_scroll = false;
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
        }
    }

    pub fn scroll_down(&mut self) {
        match self.focused_pane {
            FocusedPane::GhostStream => {
                self.ghost_auto_scroll = false;
                self.ghost_scroll += 1;
            }
            FocusedPane::Artifacts => {
                let max = self.artifacts.len().saturating_sub(1);
                if self.artifact_scroll < max {
                    self.artifact_scroll += 1;
                }
            }
            FocusedPane::EntropyMap => {
                let max = self.entropy_scores.len().saturating_sub(1);
                if self.entropy_scroll < max {
                    self.entropy_scroll += 1;
                }
            }
            FocusedPane::Events => {
                self.events_auto_scroll = false;
                let max = self.recent_events.len().saturating_sub(1);
                if self.scroll_offset < max {
                    self.scroll_offset += 1;
                }
            }
        }
    }

    pub fn scroll_top(&mut self) {
        match self.focused_pane {
            FocusedPane::GhostStream => {
                self.ghost_auto_scroll = false;
                self.ghost_scroll = 0;
            }
            FocusedPane::Artifacts => self.artifact_scroll = 0,
            FocusedPane::EntropyMap => self.entropy_scroll = 0,
            FocusedPane::Events => {
                self.events_auto_scroll = false;
                self.scroll_offset = 0;
            }
        }
    }

    pub fn scroll_bottom(&mut self) {
        match self.focused_pane {
            FocusedPane::GhostStream => {
                self.ghost_auto_scroll = true;
                let line_count = self.streaming_buffer.chars().filter(|&c| c == '\n').count();
                self.ghost_scroll = line_count;
            }
            FocusedPane::Artifacts => self.artifact_scroll = self.artifacts.len().saturating_sub(1),
            FocusedPane::EntropyMap => {
                self.entropy_scroll = self.entropy_scores.len().saturating_sub(1)
            }
            FocusedPane::Events => {
                self.events_auto_scroll = true;
                self.scroll_offset = self.recent_events.len().saturating_sub(1);
            }
        }
    }

    pub fn cycle_focus(&mut self) {
        self.focused_pane = self.focused_pane.cycle();
    }

    /// Call once per rendered frame to update the FPS counter.
    pub fn tick_fps(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_render).as_secs_f32();
        if elapsed > 0.0 {
            self.fps = 1.0 / elapsed;
        }
        self.last_render = now;
    }
}
