use crate::types::conversation::ConversationState;
use anyhow::Result;
use std::collections::VecDeque;

#[derive(Default)]
pub struct GodView {
    // wgpu handles would go here
    pub frame_count: u64,
}

impl GodView {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn render_frame(&mut self, _sigma: &ConversationState) -> Result<()> {
        self.frame_count += 1;
        Ok(())
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
        let area = 10000.0;
        let k = (area / self.nodes.len() as f32).sqrt();

        // 1. Repulsive forces
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
}

pub struct LatentMapper;

impl LatentMapper {
    pub fn project_to_3d(embedding: &[f32]) -> [f32; 3] {
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
}

impl TimelineManager {
    pub fn new() -> Self {
        Self { checkpoints: VecDeque::new(), cursor: 0 }
    }

    pub fn push(&mut self, state: ConversationState) {
        self.checkpoints.push_back(state);
    }

    #[must_use]
    pub fn seek(&self, iteration: u32) -> Option<&ConversationState> {
        self.checkpoints
            .iter()
            .find(|s| s.iteration_index == iteration)
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
        Self { frames: Vec::new(), playback_speed: speed, cursor: 0 }
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

    pub fn advance(&mut self) -> bool {
        if self.cursor + 1 < self.frames.len() {
            self.cursor += 1;
            true
        } else {
            false
        }
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
                node.id,
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    #[must_use]
    pub fn export_heatmap(
        artifact_name: &str,
        heatmap: &[f32],
        width: u32,
        height: u32,
    ) -> String {
        let max_val = heatmap.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
        let cell_w = (width as f32 / heatmap.len().max(1) as f32).max(1.0);

        let mut svg = format!(
            r##"<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#0a0a0f"/><text x="4" y="14" font-size="12" fill="#fff">{artifact_name}</text>"##
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
