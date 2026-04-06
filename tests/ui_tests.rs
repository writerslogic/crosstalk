use crosstalk::ui::visualization::{HeatmapGenerator, ThemeEngine};

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
