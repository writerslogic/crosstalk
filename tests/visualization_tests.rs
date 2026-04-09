use crosstalk::types::conversation::ConversationState;
use crosstalk::ui::visualization::{
    Edge, ForceDirectedGraph, HeatmapGenerator, LatentMapper, Node, ReplayEngine, SvgExporter,
    ThemeEngine, TimelineManager,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn sigma(session_id: &str, iter: u32) -> ConversationState {
    let mut s = ConversationState::new(session_id);
    s.iteration_index = iter;
    s
}

fn simple_graph() -> ForceDirectedGraph {
    let mut g = ForceDirectedGraph::new();
    g.nodes.push(Node { id: "a".to_string(), x: -5.0, y: 0.0, dx: 0.0, dy: 0.0, weight: 1.0 });
    g.nodes.push(Node { id: "b".to_string(), x: 5.0, y: 0.0, dx: 0.0, dy: 0.0, weight: 1.0 });
    g.edges.push(Edge { source: 0, target: 1, strength: 1.0 });
    g
}

// ── TimelineManager ───────────────────────────────────────────────────────────

#[test]
fn timeline_empty_on_creation() {
    let tl = TimelineManager::new();
    assert!(tl.is_empty());
    assert_eq!(tl.len(), 0);
}

#[test]
fn timeline_push_and_len() {
    let mut tl = TimelineManager::new();
    tl.push(sigma("s", 0));
    tl.push(sigma("s", 1));
    assert_eq!(tl.len(), 2);
}

#[test]
fn timeline_seek_finds_correct_iteration() {
    let mut tl = TimelineManager::new();
    tl.push(sigma("s", 0));
    tl.push(sigma("s", 5));
    tl.push(sigma("s", 10));
    assert_eq!(tl.seek(5).unwrap().iteration_index, 5);
    assert!(tl.seek(99).is_none());
}

#[test]
fn timeline_step_forward_advances_cursor() {
    let mut tl = TimelineManager::new();
    tl.push(sigma("s", 0));
    tl.push(sigma("s", 1));
    tl.step_forward();
    assert_eq!(tl.current().unwrap().iteration_index, 1);
}

#[test]
fn timeline_step_back_does_not_underflow() {
    let mut tl = TimelineManager::new();
    tl.push(sigma("s", 0));
    tl.step_back(); // cursor already at 0
    assert_eq!(tl.current().unwrap().iteration_index, 0);
}

#[test]
fn timeline_step_forward_clamps_at_end() {
    let mut tl = TimelineManager::new();
    tl.push(sigma("s", 0));
    tl.step_forward(); // no-op, only one entry
    tl.step_forward();
    assert_eq!(tl.current().unwrap().iteration_index, 0);
}

// ── ReplayEngine ──────────────────────────────────────────────────────────────

#[test]
fn replay_engine_empty_on_creation() {
    let re = ReplayEngine::new(1.0);
    assert_eq!(re.frame_count(), 0);
    assert!(re.current_frame().is_none());
}

#[test]
fn replay_engine_records_frames() {
    let mut re = ReplayEngine::new(1.0);
    re.record_frame(&sigma("s", 0));
    re.record_frame(&sigma("s", 1));
    assert_eq!(re.frame_count(), 2);
}

#[test]
fn replay_engine_advance_moves_cursor() {
    let mut re = ReplayEngine::new(1.0);
    re.record_frame(&sigma("s", 0));
    re.record_frame(&sigma("s", 1));
    re.record_frame(&sigma("s", 2));
    // advance from 0→1 with one more frame remaining → returns true
    assert!(re.advance());
    assert_eq!(re.current_frame().unwrap().iteration, 1);
}

#[test]
fn replay_engine_advance_returns_false_at_end() {
    let mut re = ReplayEngine::new(1.0);
    re.record_frame(&sigma("s", 0));
    assert!(!re.advance());
}

#[test]
fn replay_engine_2x_speed_skips_frames() {
    let mut re = ReplayEngine::new(2.0);
    for i in 0..6u32 { re.record_frame(&sigma("s", i)); }
    re.advance(); // cursor: 0 → 2
    assert_eq!(re.current_frame().unwrap().iteration, 2);
}

#[test]
fn replay_engine_4x_speed_skips_four_frames() {
    let mut re = ReplayEngine::new(4.0);
    for i in 0..10u32 { re.record_frame(&sigma("s", i)); }
    re.advance(); // cursor: 0 → 4
    assert_eq!(re.current_frame().unwrap().iteration, 4);
}

#[test]
fn replay_engine_reset_returns_to_start() {
    let mut re = ReplayEngine::new(2.0);
    re.record_frame(&sigma("s", 0));
    re.record_frame(&sigma("s", 1));
    re.advance();
    re.reset();
    assert_eq!(re.current_frame().unwrap().iteration, 0);
}

#[test]
fn replay_frame_stores_completion_probability() {
    let mut re = ReplayEngine::new(1.0);
    let mut s = sigma("s", 0);
    s.completion_probability = 0.75;
    re.record_frame(&s);
    assert!((re.current_frame().unwrap().completion_probability - 0.75).abs() < 1e-9);
}

// ── SvgExporter ───────────────────────────────────────────────────────────────

#[test]
fn svg_export_graph_produces_svg_tag() {
    let g = simple_graph();
    let svg = SvgExporter::export_graph(&g, 400.0, 300.0);
    assert!(svg.starts_with("<svg"), "output must begin with <svg");
    assert!(svg.ends_with("</svg>"), "output must end with </svg>");
}

#[test]
fn svg_export_graph_contains_both_nodes() {
    let g = simple_graph();
    let svg = SvgExporter::export_graph(&g, 400.0, 300.0);
    assert!(svg.contains("a"), "SVG must contain node id 'a'");
    assert!(svg.contains("b"), "SVG must contain node id 'b'");
}

#[test]
fn svg_export_graph_contains_edge_line() {
    let g = simple_graph();
    let svg = SvgExporter::export_graph(&g, 400.0, 300.0);
    assert!(svg.contains("<line"), "SVG must contain edge line element");
}

#[test]
fn svg_export_heatmap_produces_valid_svg() {
    let heatmap = vec![0.0, 0.5, 1.0, 0.8, 0.2];
    let svg = SvgExporter::export_heatmap("artifact.rs", &heatmap, 200, 50);
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
    assert!(svg.contains("artifact.rs"));
}

#[test]
fn svg_export_empty_heatmap_is_valid() {
    let svg = SvgExporter::export_heatmap("empty.rs", &[], 100, 30);
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

// ── SvgExporter (structural validity) ────────────────────────────────────────

#[test]
fn svg_export_graph_is_well_formed_xml() {
    let g = simple_graph();
    let svg = SvgExporter::export_graph(&g, 400.0, 300.0);
    // Every opening tag must have a matching close or be self-closing.
    let open_svg = svg.matches("<svg").count();
    let close_svg = svg.matches("</svg>").count();
    assert_eq!(open_svg, 1);
    assert_eq!(close_svg, 1);
    // No unclosed angle brackets.
    assert_eq!(svg.matches('<').count(), svg.matches('>').count(),
        "every '<' must have a matching '>'");
}

#[test]
fn svg_export_heatmap_is_well_formed_xml() {
    let svg = SvgExporter::export_heatmap("x.rs", &[0.5, 1.0, 0.0], 200, 50);
    assert_eq!(svg.matches('<').count(), svg.matches('>').count());
}

// ── ForceDirectedGraph ────────────────────────────────────────────────────────

#[test]
fn force_graph_layout_step_moves_nodes() {
    let mut g = simple_graph();
    let x_before = g.nodes[0].x;
    g.compute_layout_step();
    let x_after = g.nodes[0].x;
    assert!(
        (x_before - x_after).abs() > 1e-6,
        "layout step must change node positions"
    );
}

#[test]
fn force_graph_stabilises_after_many_iterations() {
    let mut g = simple_graph();
    for _ in 0..100 {
        g.compute_layout_step();
    }
    let dist = {
        let dx = g.nodes[0].x - g.nodes[1].x;
        let dy = g.nodes[0].y - g.nodes[1].y;
        (dx * dx + dy * dy).sqrt()
    };
    assert!(dist > 0.0, "nodes should not collapse to the same point");
}

// ── ForceDirectedGraph (acceptance criteria) ──────────────────────────────────

#[test]
fn force_graph_ten_nodes_stabilises_distinct_positions() {
    let mut g = ForceDirectedGraph::new();
    for i in 0..10 {
        g.nodes.push(Node {
            id: format!("n{i}"),
            x: (i as f32) * 3.0,
            y: (i % 3) as f32 * 3.0,
            dx: 0.0,
            dy: 0.0,
            weight: 1.0,
        });
    }
    for i in 0..9 {
        g.edges.push(Edge { source: i, target: i + 1, strength: 1.0 });
    }
    for _ in 0..150 {
        g.compute_layout_step();
    }
    // Acceptance criterion: no two nodes collapsed to the same position.
    for i in 0..g.nodes.len() {
        for j in (i + 1)..g.nodes.len() {
            let dx = g.nodes[i].x - g.nodes[j].x;
            let dy = g.nodes[i].y - g.nodes[j].y;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!(dist > 0.1, "nodes {i} and {j} collapsed (dist={dist:.4})");
        }
    }
}

// ── TimelineManager (acceptance criteria) ─────────────────────────────────────

#[test]
fn timeline_scrub_over_100_iterations() {
    let mut tl = TimelineManager::new();
    for i in 0..=100u32 {
        tl.push(sigma("s", i));
    }
    assert_eq!(tl.len(), 101);
    assert_eq!(tl.seek(50).unwrap().iteration_index, 50);
    assert_eq!(tl.seek(100).unwrap().iteration_index, 100);
    assert!(tl.seek(101).is_none());
}

#[test]
fn timeline_step_through_all_100_iterations() {
    let mut tl = TimelineManager::new();
    for i in 0..100u32 {
        tl.push(sigma("s", i));
    }
    for _ in 0..99 {
        tl.step_forward();
    }
    assert_eq!(tl.current().unwrap().iteration_index, 99);
}

// ── LatentMapper ──────────────────────────────────────────────────────────────

#[test]
fn latent_mapper_empty_embedding_returns_origin() {
    let coords = LatentMapper::project_to_3d(&[]);
    assert_eq!(coords, [0.0, 0.0, 0.0]);
}

#[test]
fn latent_mapper_produces_3d_coords() {
    let embedding: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let coords = LatentMapper::project_to_3d(&embedding);
    assert_eq!(coords.len(), 3);
}

#[test]
fn latent_mapper_deterministic() {
    let embedding: Vec<f32> = vec![1.0, 0.5, 0.25, 0.1];
    let a = LatentMapper::project_to_3d(&embedding);
    let b = LatentMapper::project_to_3d(&embedding);
    assert_eq!(a, b, "projection must be deterministic");
}

// ── HeatmapGenerator ─────────────────────────────────────────────────────────

#[test]
fn heatmap_generator_all_zeros_for_no_focus() {
    let map = HeatmapGenerator::generate_focus_map("hello", vec![]);
    assert!(map.iter().all(|&v| v == 0.0));
}

#[test]
fn heatmap_generator_increments_at_focus_points() {
    let content = "abcde";
    let map = HeatmapGenerator::generate_focus_map(content, vec![0, 0, 2]);
    assert!((map[0] - 2.0).abs() < 1e-9);
    assert!((map[2] - 1.0).abs() < 1e-9);
    assert!((map[1]).abs() < 1e-9);
}

#[test]
fn heatmap_generator_out_of_bounds_ignored() {
    let content = "abc";
    let map = HeatmapGenerator::generate_focus_map(content, vec![100]);
    assert!(map.iter().all(|&v| v == 0.0));
}

// ── ThemeEngine ───────────────────────────────────────────────────────────────

#[test]
fn theme_sovereign_has_non_zero_accent() {
    let theme = ThemeEngine::sovereign();
    let [r, g, b, a] = theme.accent_color;
    assert!(r > 0.0 || g > 0.0 || b > 0.0 || a > 0.0);
}

#[test]
fn theme_alpha_is_one() {
    let theme = ThemeEngine::sovereign();
    assert!((theme.primary_color[3] - 1.0).abs() < 1e-6);
    assert!((theme.accent_color[3] - 1.0).abs() < 1e-6);
}
