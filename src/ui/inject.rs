use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Clear, Paragraph},
};

pub fn draw_inject_dialog(frame: &mut Frame, input_buffer: &str) {
    let r = frame.area();
    if r.width < 30 || r.height < 10 {
        return;
    }
    let area = centered_rect(60, 20, r);
    frame.render_widget(Clear, area);

    let dialog = Block::default()
        .borders(Borders::ALL)
        .title(" Neural Intercept (Enter: submit | ESC: cancel) ")
        .border_style(Style::default().fg(Color::Cyan));

    let inner = dialog.inner(area);
    frame.render_widget(dialog, area);

    let input_para = Paragraph::new(format!("> {input_buffer}_"))
        .style(Style::default().fg(Color::White));
    frame.render_widget(input_para, inner);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let pad_y = (100 - percent_y) / 2;
    let pad_x = (100 - percent_x) / 2;
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(pad_y),
            Constraint::Percentage(100 - pad_y * 2),
            Constraint::Percentage(pad_y),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(pad_x),
            Constraint::Percentage(100 - pad_x * 2),
            Constraint::Percentage(pad_x),
        ])
        .split(vertical[1])[1]
}
