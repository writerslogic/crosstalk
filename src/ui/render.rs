use crate::ui::app::{App, AppMode, FocusedPane};
use crate::ui::inject::draw_inject_dialog;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, List, ListItem, Paragraph, Row, Table, Wrap},
};

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // σ State + μ Agents
            Constraint::Min(8),    // Ghost Stream + right panel
            Constraint::Length(3), // Convergence + Certainty gauges
            Constraint::Min(4),    // Δα Diffs / events
            Constraint::Length(1), // Status bar
        ])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[0]);

    let mode_label = match app.mode {
        AppMode::Streaming => "STREAMING",
        AppMode::Paused => "PAUSED",
        AppMode::Rewinding => "REWINDING",
    };
    let conv_pct = (app.convergence * 100.0) as u32;
    let sigma_text = format!(
        " i_{} | {} | Conv {}% | {} agents",
        app.turn_index,
        mode_label,
        conv_pct,
        app.agent_list.len()
    );
    let sigma_para = Paragraph::new(sigma_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" σ State — {} ", app.session_id)),
    );
    frame.render_widget(sigma_para, top[0]);

    let agents_line = if app.agent_list.is_empty() {
        Line::from(" (none connected)")
    } else {
        let active_idx = (app.turn_index as usize).saturating_sub(1) % app.agent_list.len().max(1);
        let mut spans = Vec::new();
        for (i, a) in app.agent_list.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw(" "));
            }
            let w = app.agent_weights.get(a.as_str()).copied().unwrap_or(1.0);
            let w_color = if w >= 1.2 {
                Color::Green
            } else if w >= 0.8 {
                Color::White
            } else {
                Color::Red
            };
            let prefix = if i == active_idx { "*" } else { "" };
            spans.push(Span::styled(format!("[{prefix}{a}"), Style::default()));
            spans.push(Span::styled(
                format!(" {w:.2}"),
                Style::default().fg(w_color),
            ));
            spans.push(Span::raw("]"));
        }
        Line::from(spans)
    };
    let agents_para = Paragraph::new(agents_line)
        .block(Block::default().borders(Borders::ALL).title(" μ Agents "));
    frame.render_widget(agents_para, top[1]);

    let center = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(rows[1]);

    let ghost_style = focus_border(app, FocusedPane::GhostStream);
    let ghost_area = center[0];
    let inner_height = ghost_area.height.saturating_sub(2) as usize; // subtract borders
    let scroll_y = if app.ghost_auto_scroll {
        app.ghost_scroll.saturating_sub(inner_height)
    } else {
        app.ghost_scroll
    };
    let ghost = Paragraph::new(app.streaming_buffer.as_str())
        .scroll((scroll_y as u16, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Ghost Stream ")
                .border_style(ghost_style),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(ghost, ghost_area);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(center[1]);

    let art_style = focus_border(app, FocusedPane::Artifacts);
    let artifact_items: Vec<ListItem> = app
        .artifacts
        .iter()
        .skip(app.artifact_scroll)
        .map(|snap| {
            let ver = format!("v{}", snap.version);
            let label = if snap.skeleton.is_empty() {
                format!(" {ver} {}", snap.name)
            } else {
                format!(" {ver} {}  {}", snap.name, snap.skeleton)
            };
            let color = if snap.diff_count == 0 {
                Color::DarkGray
            } else if snap.diff_count < 3 {
                Color::Green
            } else {
                Color::Yellow
            };
            ListItem::new(label).style(Style::default().fg(color))
        })
        .collect();
    let art_title = format!(" Artifacts ({}) ", app.artifacts.len());
    let artifacts_list = List::new(artifact_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(art_title)
            .border_style(art_style),
    );
    frame.render_widget(artifacts_list, right[0]);

    draw_entropy_heatmap(
        frame,
        app,
        right[1],
        focus_border(app, FocusedPane::EntropyMap),
    );

    let gauges = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[2]);

    let conv = app.convergence.clamp(0.0, 1.0);
    let conv_color = gauge_color(conv);
    let conv_gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Convergence "),
        )
        .gauge_style(Style::default().fg(conv_color))
        .ratio(conv);
    frame.render_widget(conv_gauge, gauges[0]);

    let cert = app.certainty.clamp(0.0, 1.0);
    let cert_color = gauge_color(cert);
    let cert_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Certainty "))
        .gauge_style(Style::default().fg(cert_color))
        .ratio(cert);
    frame.render_widget(cert_gauge, gauges[1]);

    let ev_style = focus_border(app, FocusedPane::Events);
    let visible: Vec<ListItem> = app
        .recent_events
        .iter()
        .skip(app.scroll_offset)
        .map(|e| {
            let color = if e.starts_with("Error:") || e.contains("PANIC") || e.contains("FAIL") {
                Color::Red
            } else if e.contains("[rollback]")
                || e.contains("[Warning")
                || e.contains("rate-limited")
            {
                Color::Yellow
            } else if e.starts_with("Turn ") || e.contains("Checkpoint") {
                Color::Green
            } else if e.contains("[convergence]")
                || e.contains("[write]")
                || e.contains("[sandbox]")
            {
                Color::Cyan
            } else {
                Color::White
            };
            ListItem::new(e.as_str()).style(Style::default().fg(color))
        })
        .collect();
    let ev_title = format!(
        " Δα Events ({}) [j/k scroll | g/G top/bottom] ",
        app.recent_events.len()
    );
    let events_list = List::new(visible).block(
        Block::default()
            .borders(Borders::ALL)
            .title(ev_title)
            .border_style(ev_style),
    );
    frame.render_widget(events_list, rows[3]);

    let fps_str = if app.fps > 0.0 {
        format!("{:.0}fps", app.fps)
    } else {
        "---fps".to_string()
    };
    let godview_str = format!(
        "ξ:{:.0}% σ:{:.0}%",
        app.godview_certainty * 100.0,
        app.godview_surprise * 100.0
    );
    let status = Paragraph::new(format!(
        " [Tab] Focus | [Ctrl+I] Inject | [j/k] Scroll | [g/G] Top/Bottom | [q] Quit | {godview_str} | {fps_str} "
    ));
    frame.render_widget(status, rows[4]);

    if app.showing_inject {
        draw_inject_dialog(frame, &app.inject_buffer);
    }
}

fn draw_entropy_heatmap(
    frame: &mut Frame,
    app: &App,
    area: ratatui::layout::Rect,
    border_style: Style,
) {
    if app.entropy_scores.is_empty() || app.agent_list.is_empty() {
        let hint = if app.agent_list.len() < 2 {
            " Waiting for 2+ agents to generate data..."
        } else {
            " Waiting for agents to modify shared artifacts..."
        };
        let placeholder = Paragraph::new(hint).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Entropy Heatmap ")
                .border_style(border_style),
        );
        frame.render_widget(placeholder, area);
        return;
    }

    let header_cells: Vec<Cell> = std::iter::once(
        Cell::from("Artifact").style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .chain(app.agent_list.iter().map(|a| {
        let label = if a.len() > 10 { &a[..10] } else { a.as_str() };
        Cell::from(label.to_string()).style(Style::default().add_modifier(Modifier::BOLD))
    }))
    .collect();
    let header = Row::new(header_cells).height(1);

    let rows: Vec<Row> = app
        .entropy_scores
        .iter()
        .skip(app.entropy_scroll)
        .map(|entry| {
            let artifact_label = if entry.artifact.len() > 14 {
                format!("{}...", &entry.artifact[..13])
            } else {
                entry.artifact.clone()
            };
            let mut cells: Vec<Cell> = vec![Cell::from(artifact_label)];
            for agent_id in &app.agent_list {
                let score = entry
                    .agents
                    .iter()
                    .find(|(a, _)| a == agent_id)
                    .map(|(_, s)| *s)
                    .unwrap_or(0.0);
                let color = heatmap_color(score);
                cells.push(Cell::from(format!("{:.2}", score)).style(Style::default().fg(color)));
            }
            Row::new(cells).height(1)
        })
        .collect();

    let agent_count = app.agent_list.len().max(1);
    let mut widths: Vec<Constraint> = vec![Constraint::Percentage(30)];
    for _ in 0..agent_count {
        widths.push(Constraint::Percentage(70 / agent_count as u16));
    }

    let total_scores: f64 = app
        .entropy_scores
        .iter()
        .flat_map(|e| e.agents.iter().map(|(_, s)| s))
        .sum();
    let score_count = app
        .entropy_scores
        .iter()
        .map(|e| e.agents.len())
        .sum::<usize>()
        .max(1);
    let avg_entropy = total_scores / score_count as f64;
    let hm_title = format!(" Entropy Heatmap [avg {avg_entropy:.2} | 0=agree 1=conflict] ");
    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .borders(Borders::ALL)
            .title(hm_title)
            .border_style(border_style),
    );
    frame.render_widget(table, area);
}

fn heatmap_color(score: f64) -> Color {
    if score > 0.7 {
        Color::Red
    } else if score > 0.3 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn gauge_color(ratio: f64) -> Color {
    if ratio > 0.8 {
        Color::Green
    } else if ratio > 0.5 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn focus_border(app: &App, pane: FocusedPane) -> Style {
    if app.focused_pane == pane {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    }
}
