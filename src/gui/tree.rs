use crate::config::CONFIG;
use egui_plot::{Line, Plot, PlotPoint, Points, Text};
use std::collections::{HashMap, HashSet};

struct TreePlotData {
    positions: HashMap<usize, PlotPoint>,
    edges: Vec<(usize, usize)>,
}

fn layout_bsp_tree(
    node: &crate::bsp::BspNode,
    depth: f64,
    x_min: f64,
    x_max: f64,
    data: &mut TreePlotData,
) {
    let self_x = (x_min + x_max) / 2.0;
    let self_point = PlotPoint { x: self_x, y: -depth };
    data.positions.insert(node.id, self_point);

    if let Some(ref front) = node.front {
        layout_bsp_tree(front, depth + 1.0, x_min, self_x, data);
        data.edges.push((node.id, front.id));
    }
    if let Some(ref back) = node.back {
        layout_bsp_tree(back, depth + 1.0, self_x, x_max, data);
        data.edges.push((node.id, back.id));
    }
}

pub fn draw_bsp_tree_window(
    ctx: &egui::Context,
    open: &mut bool,
    root: &crate::bsp::BspNode,
    selected: &mut Option<usize>,
) {
    egui::Window::new("BSP Tree")
        .open(open)
        .vscroll(true)
        .hscroll(true)
        .show(ctx, |ui| {
            let mut data = TreePlotData { positions: HashMap::new(), edges: Vec::new() };
            layout_bsp_tree(root, 0.0, 0.0, 1.0, &mut data);

            let mut path_ids = HashSet::new();
            if let Some(sel_id) = *selected {
                let mut path = Vec::new();
                if crate::bsp::find_node_path(root, sel_id, &mut path) {
                    for n in path { path_ids.insert(n.id); }
                }
            }

            let cfg = CONFIG.read().unwrap();
            let highlight_color = cfg.bsp_tree_path_color;
            let selected_color = cfg.bsp_tree_selected_color;

            let plot = Plot::new("bsp_tree_plot");
            let plot_resp = plot.show(ui, |plot_ui| {
                for &(a, b) in &data.edges {
                    if let (Some(&p1), Some(&p2)) = (data.positions.get(&a), data.positions.get(&b)) {
                        let pts = vec![[p1.x, p1.y], [p2.x, p2.y]];
                        let color = if path_ids.contains(&a) && path_ids.contains(&b) {
                            highlight_color
                        } else {
                            egui::Color32::LIGHT_GRAY
                        };
                        plot_ui.line(Line::new(pts).color(color));
                    }
                }
                for (&id, &pos) in &data.positions {
                    let color = if selected == &Some(id) {
                        selected_color
                    } else if path_ids.contains(&id) {
                        highlight_color
                    } else {
                        egui::Color32::LIGHT_BLUE
                    };
                    let pt = vec![[pos.x, pos.y]];
                    plot_ui.points(Points::new(pt).radius(4.0).color(color));
                    plot_ui.text(
                        Text::new(
                            pos,
                            egui::RichText::new(format!("{}", id)).size(cfg.bsp_tree_text_size),
                        )
                        .anchor(egui::Align2::CENTER_CENTER),
                    );
                }
                plot_ui.pointer_coordinate()
            });

            if plot_resp.response.clicked() {
                if let Some(pointer) = plot_resp.inner {
                    let mut best = None;
                    let mut best_dist = f64::INFINITY;
                    for (&id, &p) in &data.positions {
                        let dx = pointer.x - p.x;
                        let dy = pointer.y - p.y;
                        let d = dx * dx + dy * dy;
                        if d < best_dist {
                            best_dist = d;
                            best = Some(id);
                        }
                    }
                    if let Some(id) = best { *selected = Some(id); }
                }
            }
        });
}
