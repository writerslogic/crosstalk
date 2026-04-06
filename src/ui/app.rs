use crate::types::conversation::Turn;
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
    /// (name, skeleton) pairs
    pub artifacts: Vec<(String, String)>,
    pub convergence: f64,
    pub certainty: f64,
    pub agent_weights: HashMap<String, f64>,
    /// Capped at 50 items; oldest evicted when full
    pub recent_events: VecDeque<String>,
    pub mode: AppMode,
    /// Row offset for events log scroll
    pub scroll_offset: usize,
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
    /// Measured frames per second
    pub fps: f32,
    /// Tracks last render time for FPS computation
    last_render: Instant,
    /// Recent diff sizes per artifact per agent: artifact → agent → [diff_len]
    artifact_change_history: HashMap<String, HashMap<String, Vec<usize>>>,
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
            shutdown: false,
            inject_buffer: String::new(),
            showing_inject: false,
            focused_pane: FocusedPane::GhostStream,
            entropy_scores: Vec::new(),
            agent_list: Vec::new(),
            fps: 0.0,
            last_render: Instant::now(),
            artifact_change_history: HashMap::new(),
        }
    }

    pub fn push_token(&mut self, token: &str) {
        self.streaming_buffer.push_str(token);
        const BUF_CAP: usize = 50 * 1024;
        if self.streaming_buffer.len() > BUF_CAP {
            let trim = self.streaming_buffer.len() - BUF_CAP;
            let safe = self.streaming_buffer
                .char_indices()
                .find(|(i, _)| *i >= trim)
                .map(|(i, _)| i)
                .unwrap_or(trim);
            self.streaming_buffer.drain(..safe);
        }
    }

    pub fn commit_turn(&mut self, turn: &Turn) {
        self.turn_index = turn.index + 1;
        self.streaming_buffer.clear();
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
                let single_score: Vec<(String, f64)> = agent_map
                    .keys()
                    .map(|a| (a.clone(), 0.0))
                    .collect();
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

            let max_mean = agent_means
                .iter()
                .map(|(_, m)| *m)
                .fold(0.0_f64, f64::max);

            let agents: Vec<(String, f64)> = if max_mean == 0.0 {
                agent_means.into_iter().map(|(a, _)| (a, 0.0)).collect()
            } else {
                let global_mean = agent_means.iter().map(|(_, m)| m).sum::<f64>()
                    / agent_means.len() as f64;
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

        self.entropy_scores.sort_by(|a, b| a.artifact.cmp(&b.artifact));
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
    }

    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        let max = self.recent_events.len().saturating_sub(1);
        if self.scroll_offset < max {
            self.scroll_offset += 1;
        }
    }

    pub fn scroll_top(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn scroll_bottom(&mut self) {
        self.scroll_offset = self.recent_events.len().saturating_sub(1);
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
