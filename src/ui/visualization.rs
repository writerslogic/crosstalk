use crate::types::conversation::ConversationState;
use std::collections::{HashMap, VecDeque};

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct GodViewMetrics {
    pub frame: u64,
    pub turn_count: usize,
    pub artifact_count: usize,
    pub avg_certainty: f64,
    pub avg_surprise: f64,
    pub completion_p: f64,
    pub agent_count: usize,
}

pub struct GodView {
    pub frame_count: u64,
}

impl Default for GodView {
    fn default() -> Self {
        Self::new()
    }
}

impl GodView {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }

    pub fn compute_metrics(&mut self, sigma: &ConversationState) -> GodViewMetrics {
        self.frame_count += 1;
        let turn_count = sigma.turns.len();
        let avg_certainty = if turn_count > 0 {
            sigma.turns.iter().filter_map(|t| t.certainty).sum::<f64>() / turn_count as f64
        } else {
            0.0
        };
        let avg_surprise = if turn_count > 0 {
            sigma
                .turns
                .iter()
                .filter_map(|t| t.surprise_signal)
                .sum::<f64>()
                / turn_count as f64
        } else {
            0.0
        };
        let agent_count = sigma.agent_weights.len();
        GodViewMetrics {
            frame: self.frame_count,
            turn_count,
            artifact_count: sigma.artifacts.len(),
            avg_certainty,
            avg_surprise,
            completion_p: sigma.completion_probability,
            agent_count,
        }
    }
}

pub struct LatentMapper;

impl LatentMapper {
    #[must_use]
    pub fn project_to_3d(content: &str) -> [f32; 3] {
        if content.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let emb = crate::engines::memory::embed_text(content);
        if emb.len() >= 3 {
            // Normalize the first 3 components of the embedding to [-1, 1]
            [
                emb[0].clamp(-1.0, 1.0),
                emb[1].clamp(-1.0, 1.0),
                emb[2].clamp(-1.0, 1.0),
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
}

pub struct ForceDirectedGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub k: f32, // Optimal distance
}

pub struct Node {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    pub weight: f32,
}

pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub strength: f32,
}

impl Default for ForceDirectedGraph {
    fn default() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            k: 10.0,
        }
    }
}

impl ForceDirectedGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute_layout_step(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let area = 10000.0;
        let k = (area / self.nodes.len() as f32).sqrt();

        // 1. Repulsive forces
        // Node count equals the number of unique agents (typically 2–10), so the
        // O(n²) all-pairs loop is acceptable in practice.
        debug_assert!(
            self.nodes.len() < 100,
            "force-directed repulsion is O(n²); {} nodes exceeds safe threshold",
            self.nodes.len()
        );
        for i in 0..self.nodes.len() {
            self.nodes[i].dx = 0.0;
            self.nodes[i].dy = 0.0;
            for j in 0..self.nodes.len() {
                if i != j {
                    let dx = self.nodes[i].x - self.nodes[j].x;
                    let dy = self.nodes[i].y - self.nodes[j].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.1);
                    let fr = (k * k) / dist;
                    self.nodes[i].dx += (dx / dist) * fr;
                    self.nodes[i].dy += (dy / dist) * fr;
                }
            }
        }

        // 2. Attractive forces
        for edge in &self.edges {
            if edge.source >= self.nodes.len() || edge.target >= self.nodes.len() {
                continue;
            }
            let dx = self.nodes[edge.source].x - self.nodes[edge.target].x;
            let dy = self.nodes[edge.source].y - self.nodes[edge.target].y;
            let dist = (dx * dx + dy * dy).sqrt().max(0.1);
            let fa = (dist * dist) / k;
            let d_x = (dx / dist) * fa;
            let d_y = (dy / dist) * fa;
            self.nodes[edge.source].dx -= d_x;
            self.nodes[edge.source].dy -= d_y;
            self.nodes[edge.target].dx += d_x;
            self.nodes[edge.target].dy += d_y;
        }

        // 3. Update positions
        let temp = 10.0; // Temperature
        for node in &mut self.nodes {
            let dist = (node.dx * node.dx + node.dy * node.dy).sqrt().max(0.1);
            node.x += (node.dx / dist) * dist.min(temp);
            node.y += (node.dy / dist) * dist.min(temp);
        }
    }

    pub fn update_from_sigma(&mut self, sigma: &ConversationState) {
        let mut agent_map = HashMap::new();
        for turn in &sigma.turns {
            let id = turn.model_id.clone();
            let weight = sigma.agent_weights.get(&id).copied().unwrap_or(1.0);
            agent_map.insert(id.clone(), weight);
        }

        // Rebuild graph nodes from agents
        self.nodes.clear();
        let mut id_to_idx = HashMap::new();
        for (id, weight) in agent_map {
            id_to_idx.insert(id.clone(), self.nodes.len());
            self.nodes.push(Node {
                id,
                x: rand::random::<f32>() * 100.0,
                y: rand::random::<f32>() * 100.0,
                dx: 0.0,
                dy: 0.0,
                weight: weight as f32,
            });
        }

        // Rebuild edges based on turn sequence (influence)
        self.edges.clear();
        for window in sigma.turns.windows(2) {
            let src_idx = id_to_idx.get(&window[0].model_id);
            let tgt_idx = id_to_idx.get(&window[1].model_id);
            if let (Some(&src), Some(&tgt)) = (src_idx, tgt_idx) {
                self.edges.push(Edge {
                    source: src,
                    target: tgt,
                    strength: 1.0,
                });
            }
        }
    }
}

impl LatentMapper {
    pub fn project_embedding_to_3d(embedding: &[f32]) -> [f32; 3] {
        if embedding.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let mut out = [0.0; 3];
        let sqrt3 = 3.0f32.sqrt();

        for (j, out_elem) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (i, &val) in embedding.iter().enumerate() {
                // Deterministic pseudo-random number based on (i, j)
                let h = Self::hash_two(i as u32, j as u32);
                let r = match h % 6 {
                    0 => sqrt3,
                    1 => -sqrt3,
                    _ => 0.0,
                };
                sum += val * r;
            }
            // Projection: x' = 1/sqrt(k) * x * R, k = 3
            *out_elem = sum / sqrt3;
        }
        out
    }

    fn hash_two(a: u32, b: u32) -> u32 {
        let mut h = a.wrapping_mul(0x45d9f3b) ^ b;
        h = ((h >> 16) ^ h).wrapping_mul(0x45d9f3b);
        h = ((h >> 16) ^ h).wrapping_mul(0x45d9f3b);
        h = (h >> 16) ^ h;
        h
    }
}

pub struct HeatmapGenerator;

impl HeatmapGenerator {
    pub fn generate_focus_map(artifact_content: &str, focus_points: Vec<usize>) -> Vec<f32> {
        let mut heatmap = vec![0.0; artifact_content.len()];
        for pos in focus_points {
            if pos < heatmap.len() {
                heatmap[pos] += 1.0;
            }
        }
        heatmap
    }
}

pub struct TimelineManager {
    pub checkpoints: VecDeque<ConversationState>,
    pub cursor: usize,
    index: HashMap<u32, usize>,
}

impl TimelineManager {
    pub fn new() -> Self {
        Self {
            checkpoints: VecDeque::new(),
            cursor: 0,
            index: HashMap::new(),
        }
    }

    pub fn push(&mut self, state: ConversationState) {
        let pos = self.checkpoints.len();
        self.index.insert(state.iteration_index, pos);
        self.checkpoints.push_back(state);
    }

    #[must_use]
    pub fn seek(&self, iteration: u32) -> Option<&ConversationState> {
        let pos = *self.index.get(&iteration)?;
        self.checkpoints.get(pos)
    }

    #[must_use]
    pub fn current(&self) -> Option<&ConversationState> {
        self.checkpoints.get(self.cursor)
    }

    pub fn step_forward(&mut self) {
        if self.cursor + 1 < self.checkpoints.len() {
            self.cursor += 1;
        }
    }

    pub fn step_back(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

impl Default for TimelineManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ReplayFrame {
    pub iteration: u32,
    pub session_id: String,
    pub completion_probability: f64,
    pub turn_count: usize,
}

pub struct ReplayEngine {
    pub frames: Vec<ReplayFrame>,
    pub playback_speed: f32,
    cursor: usize,
}

impl ReplayEngine {
    pub fn new(speed: f32) -> Self {
        Self {
            frames: Vec::new(),
            playback_speed: speed,
            cursor: 0,
        }
    }

    pub fn record_frame(&mut self, state: &ConversationState) {
        self.frames.push(ReplayFrame {
            iteration: state.iteration_index,
            session_id: state.session_id.clone(),
            completion_probability: state.completion_probability,
            turn_count: state.turns.len(),
        });
    }

    #[must_use]
    pub fn current_frame(&self) -> Option<&ReplayFrame> {
        self.frames.get(self.cursor)
    }

    /// Advance the cursor by `playback_speed` frames (rounded to nearest ≥1).
    /// Returns `true` if the cursor actually moved (i.e. was not already at the end).
    pub fn advance(&mut self) -> bool {
        let step = (self.playback_speed.round() as usize).max(1);
        let next = (self.cursor + step).min(self.frames.len().saturating_sub(1));
        let moved = next > self.cursor;
        self.cursor = next;
        moved
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

pub struct SvgExporter;

impl SvgExporter {
    #[must_use]
    pub fn export_graph(graph: &ForceDirectedGraph, width: f32, height: f32) -> String {
        let mut svg = format!(
            r##"<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#0a0a0f"/>"##
        );

        for edge in &graph.edges {
            if let (Some(src), Some(tgt)) =
                (graph.nodes.get(edge.source), graph.nodes.get(edge.target))
            {
                svg.push_str(&format!(
                    r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#444" stroke-width="1"/>"##,
                    src.x + width / 2.0,
                    src.y + height / 2.0,
                    tgt.x + width / 2.0,
                    tgt.y + height / 2.0,
                ));
            }
        }

        for node in &graph.nodes {
            let r = (node.weight * 5.0).clamp(3.0, 20.0);
            svg.push_str(&format!(
                r##"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="#00ff88"/><text x="{:.1}" y="{:.1}" font-size="10" fill="#fff">{}</text>"##,
                node.x + width / 2.0,
                node.y + height / 2.0,
                r,
                node.x + width / 2.0 + r + 2.0,
                node.y + height / 2.0 + 4.0,
                xml_escape(&node.id),
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    #[must_use]
    pub fn export_heatmap(artifact_name: &str, heatmap: &[f32], width: u32, height: u32) -> String {
        let max_val = heatmap.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
        let cell_w = (width as f32 / heatmap.len().max(1) as f32).max(1.0);

        let mut svg = format!(
            r##"<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#0a0a0f"/><text x="4" y="14" font-size="12" fill="#fff">{}</text>"##,
            xml_escape(artifact_name),
        );

        for (i, &val) in heatmap.iter().enumerate() {
            let intensity = (val / max_val * 255.0) as u8;
            let x = i as f32 * cell_w;
            svg.push_str(&format!(
                r##"<rect x="{x:.1}" y="20" width="{cell_w:.1}" height="{h}" fill="rgb({intensity},0,0)"/>"##,
                h = height - 20,
            ));
        }

        svg.push_str("</svg>");
        svg
    }
}

pub struct ThemeEngine {
    pub primary_color: [f32; 4], // RGBA
    pub accent_color: [f32; 4],
}

impl ThemeEngine {
    pub fn sovereign() -> Self {
        Self {
            primary_color: [0.04, 0.04, 0.06, 1.0], // #0a0a0f
            accent_color: [0.0, 1.0, 0.53, 1.0],    // #00ff88
        }
    }
}
