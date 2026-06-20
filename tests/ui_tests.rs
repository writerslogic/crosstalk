use crosstalk::types::conversation::{TaskCategory, Turn, TurnOutcome, TurnStructure};
use crosstalk::types::events::ArtifactSnapshot;
use crosstalk::ui::app::{App, AppMode};
use crosstalk::ui::render;
use crosstalk::ui::visualization::{HeatmapGenerator, ThemeEngine};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

fn make_turn(index: u32, outcome: TurnOutcome) -> Turn {
    Turn {
        index,
        model_id: "test-agent".to_string(),
        content: "test content".to_string(),
        timestamp: 0,
        diffs: vec![],
        certainty: Some(0.8),
        outcome,
        task_category: Some(TaskCategory::CodeGeneration),
        structure: Some(TurnStructure::FreeForm),
        signature: vec![],
        surprise_signal: None,
        consistency_score: None,
        diff_quality_score: None,
        persona_disclosure: None,
    }
}

#[test]
fn push_token_accumulates() {
    let mut app = App::new("s1");
    app.push_token("test_agent", "Hello");
    app.push_token("test_agent", ", world");
    assert_eq!(app.streaming_buffer, "[test_agent] Hello, world");
}

#[test]
fn commit_turn_updates_turn_index() {
    let mut app = App::new("s1");
    assert_eq!(app.turn_index, 0);
    app.commit_turn(&make_turn(3, TurnOutcome::Compiled));
    assert_eq!(app.turn_index, 4);
}

#[test]
fn commit_turn_appends_separator() {
    let mut app = App::new("s1");
    app.push_token("test_agent", "some tokens");
    app.commit_turn(&make_turn(0, TurnOutcome::Compiled));
    assert!(app.streaming_buffer.contains("--- Turn 0"));
}

#[test]
fn draw_does_not_panic_on_empty_state() {
    let app = App::new("empty");
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render::draw(f, &app)).unwrap();
}

#[test]
fn draw_does_not_panic_on_full_state() {
    let mut app = App::new("full");
    for i in 0..60 {
        app.push_event(format!("event {i}"));
    }
    for i in 0..5 {
        app.artifacts.push(ArtifactSnapshot {
            name: format!("artifact_{i}"),
            skeleton: format!("v{i}"),
            version: i,
            diff_count: 0,
        });
    }
    app.agent_weights.insert("agent-a".to_string(), 0.84);
    app.agent_weights.insert("agent-b".to_string(), 1.2);
    app.set_convergence(0.62, 0.88);
    app.push_token("test_agent", "streaming content here");
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render::draw(f, &app)).unwrap();
}

#[test]
fn app_mode_transitions() {
    let mut app = App::new("s1");
    assert_eq!(app.mode, AppMode::Streaming);
    app.mode = AppMode::Paused;
    assert_eq!(app.mode, AppMode::Paused);
    app.mode = AppMode::Rewinding;
    assert_eq!(app.mode, AppMode::Rewinding);
    app.mode = AppMode::Streaming;
    assert_eq!(app.mode, AppMode::Streaming);
}

#[test]
fn events_log_caps_at_50() {
    let mut app = App::new("s1");
    for i in 0..60 {
        app.push_event(format!("event {i}"));
    }
    assert_eq!(app.recent_events.len(), 50);
    assert_eq!(app.recent_events.front().unwrap(), "event 10");
    assert_eq!(app.recent_events.back().unwrap(), "event 59");
}

#[test]
fn set_convergence_updates_fields() {
    let mut app = App::new("s1");
    app.set_convergence(0.75, 0.90);
    assert!((app.convergence - 0.75).abs() < f64::EPSILON);
    assert!((app.certainty - 0.90).abs() < f64::EPSILON);
}

#[test]
fn test_heatmap_generation() {
    let content = "fn main() {}";
    let focus = vec![0, 3, 3];
    let map = HeatmapGenerator::generate_focus_map(content, focus);
    assert_eq!(map[3], 2.0);
    assert_eq!(map[0], 1.0);
}

#[test]
fn test_theme_sovereign() {
    let theme = ThemeEngine::sovereign();
    assert_eq!(theme.accent_color[1], 1.0);
}
