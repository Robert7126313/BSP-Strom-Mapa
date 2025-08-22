use crate::config::MAX_BSP_DEPTH;
use cgmath::InnerSpace;
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
    let self_point = PlotPoint {
        x: self_x,
        y: -depth,
    };
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

fn draw_bsp_tree_window(
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
            let mut data = TreePlotData {
                positions: HashMap::new(),
                edges: Vec::new(),
            };
            layout_bsp_tree(root, 0.0, 0.0, 1.0, &mut data);

            // Determine path from root to selected node
            let mut path_ids = HashSet::new();
            if let Some(sel_id) = *selected {
                let mut path = Vec::new();
                if crate::bsp::find_node_path(root, sel_id, &mut path) {
                    for n in path {
                        path_ids.insert(n.id);
                    }
                }
            }

            let highlight_color = egui::Color32::from_rgb(255, 200, 0);

            let plot = Plot::new("bsp_tree_plot");
            let plot_resp = plot.show(ui, |plot_ui| {
                for &(a, b) in &data.edges {
                    if let (Some(&p1), Some(&p2)) = (data.positions.get(&a), data.positions.get(&b))
                    {
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
                        egui::Color32::YELLOW
                    } else if path_ids.contains(&id) {
                        highlight_color
                    } else {
                        egui::Color32::LIGHT_BLUE
                    };
                    let pt = vec![[pos.x, pos.y]];
                    plot_ui.points(Points::new(pt).radius(4.0).color(color));
                    plot_ui.text(
                        Text::new(pos, format!("{}", id)).anchor(egui::Align2::CENTER_CENTER),
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
                    if let Some(id) = best {
                        *selected = Some(id);
                    }
                }
            }
        });
}

pub fn draw_left_panel(
    ctx: &egui::Context,
    mode: crate::camera::CamMode,
    loaded_file_name: &mut String,
    file_loading: &mut bool,
    tx_gui: &std::sync::mpsc::Sender<crate::Message>,
    rx: &std::sync::mpsc::Receiver<crate::Message>,
    current_cpu_mesh: &mut three_d::CpuMesh,
    current_triangles: &mut Vec<crate::bsp::Triangle>,
    bsp_root_preview: &mut Option<crate::bsp::BspNode>,
    selected_node: &mut Option<usize>,
    show_splitting_plane: &mut bool,
    use_gpu_culling: &mut bool,
    disable_culling: &mut bool,
    show_loaded_model: &mut bool,
    show_selected_model: &mut bool,
    show_camera_direction: &mut bool,
    spectator_state: &mut crate::camera::CameraState,
    third_person_state: &mut crate::camera::CameraState,
    cam: &mut crate::camera::FreeCamera,
    current_stats: &crate::bsp::BspStats,
    tree_window_open: &mut bool,
    branch_limit: &mut u32,
    limit_culling: &mut bool,
) {
    egui::SidePanel::left("tree").show(ctx, |side_ui| {
        egui::ScrollArea::vertical().show(side_ui, |ui| {
            ui.heading("BSP Strom");
            ui.label(format!("Režim: {:?}", mode));

            ui.separator();
            ui.heading("Načtení modelu");
            ui.label("Aktuální model:");
            ui.label(loaded_file_name.as_str());

            if ui.button("📁 Načíst nový model").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("GLTF/GLB files", &["gltf", "glb"])
                    .pick_file()
                {
                    *file_loading = true;
                    let path_clone = path.clone();
                    let file_name_clone = path.file_name().unwrap().to_string_lossy().into_owned();
                    let tx_gui_clone = tx_gui.clone();
                    std::thread::spawn(move || {
                        let (new_cpu_mesh, load_status) = crate::load_cpu_mesh(&path_clone);
                        let _ = tx_gui_clone.send(crate::Message::NewFile {
                            cpu_mesh: new_cpu_mesh,
                            file_name: file_name_clone,
                            load_status,
                            triangles: Vec::new(),
                            bsp_tree: crate::bsp::BspNode::new_leaf(Vec::new(), 0),
                        });
                    });
                }
            }

            if *file_loading {
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .text("Načítání modelu a stavba BSP stromu...")
                        .animate(true),
                );
            }

            if bsp_root_preview.is_none() {
                ui.separator();
                ui.label("Strom se staví…");
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .animate(true),
                );
                return;
            }

            ui.separator();
            ui.heading("Struktura BSP stromu");
            ui.add(egui::Slider::new(branch_limit, 1..=MAX_BSP_DEPTH).text("Limit větvení"));
            ui.checkbox(limit_culling, "Omezit culling podle slideru");
            ui.checkbox(show_splitting_plane, "Zobrazit dělící rovinu");
            if ui.button("Otevřít vizualizaci").clicked() {
                *tree_window_open = true;
            }

            if let Some(node_id) = *selected_node {
                if let Some(ref root) = *bsp_root_preview {
                    if let Some(node) = crate::bsp::find_node(root, node_id) {
                        ui.separator();
                        ui.heading("Vybraný uzel");
                        ui.label(format!("ID: {}", node.id));
                        ui.label(format!("Trojúhelníků: {}", node.triangles.len()));
                        if let Some(ref plane) = node.plane {
                            ui.label("Dělící rovina:");
                            ui.label(format!(
                                "Normála: ({:.2}, {:.2}, {:.2})",
                                plane.n.x, plane.n.y, plane.n.z
                            ));
                            ui.label(format!("Vzdálenost: {:.2}", plane.d));
                        } else {
                            ui.label("List (bez dělící roviny)");
                        }
                        ui.label("Obalový objem:");
                        ui.label(format!(
                            "Min: ({:.2}, {:.2}, {:.2})",
                            node.bounds.min.x, node.bounds.min.y, node.bounds.min.z
                        ));
                        ui.label(format!(
                            "Max: ({:.2}, {:.2}, {:.2})",
                            node.bounds.max.x, node.bounds.max.y, node.bounds.max.z
                        ));
                        if ui.button("Odznačit").clicked() {
                            *selected_node = None;
                        }
                    }
                }
            }

            ui.separator();
            ui.heading("Nastavení zobrazení");
            ui.checkbox(disable_culling, "Vypnout culling");
            ui.add_enabled_ui(!*disable_culling, |ui| {
                ui.checkbox(use_gpu_culling, "Použít GPU culling");
            });
            ui.checkbox(show_loaded_model, "Zobrazit načtený model");
            ui.checkbox(show_selected_model, "Zobrazit vybranou oblast");

            ui.separator();
            ui.heading("BSP Statistiky");
            ui.label(format!("Celkem uzlů: {}", current_stats.total_nodes));
            ui.label(format!(
                "Celkem trojúhelníků: {}",
                current_stats.total_triangles
            ));
            ui.label(format!("Navštíveno uzlů: {}", current_stats.nodes_visited));
            ui.label(format!(
                "Vykresleno trojúhelníků: {}",
                current_stats.triangles_rendered
            ));
            ui.label(format!(
                "Procházka efektivita: {:.1}%",
                if current_stats.total_nodes > 0 {
                    (current_stats.nodes_visited as f32 / current_stats.total_nodes as f32) * 100.0
                } else {
                    0.0
                }
            ));

            ui.separator();
            ui.heading("Mesh Info");
            ui.label(format!("Vrcholy: {}", current_cpu_mesh.positions.len()));
            match &current_cpu_mesh.indices {
                three_d_asset::Indices::U32(idx) => {
                    ui.label(format!("Indexy (U32): {}", idx.len()))
                }
                three_d_asset::Indices::U16(idx) => {
                    ui.label(format!("Indexy (U16): {}", idx.len()))
                }
                _ => ui.label("Indexy: žádné"),
            };

            ui.separator();
            ui.heading("Ovládání");
            ui.label("POHYB:");
            ui.label("• W - Dopředu");
            ui.label("• S - Dozadu");
            ui.label("• A - Doleva");
            ui.label("• D - Doprava");
            ui.label("• Space - Nahoru");
            ui.label("• C - Dolů");
            ui.label(format!("Rychlost: {:.1}", cam.speed));

            ui.separator();
            ui.label("ROZHLÍŽENÍ:");
            ui.label("• ↑ - Díváš se nahoru");
            ui.label("• ↓ - Díváš se dolů");
            ui.label("• ← - Otočit hlavu doleva");
            ui.label("• → - Otočit hlavu doprava");
            ui.label(format!(
                "Rychlost rozhlížení: {:.1}°/s",
                cam.look_speed * 180.0 / std::f32::consts::PI
            ));
            ui.add(egui::Slider::new(&mut cam.look_speed, 0.5..=5.0).text("Rychlost rozhlížení"));

            ui.separator();
            ui.label("OSTATNÍ:");
            ui.label("• F - Přepnout na režim Spectator");
            ui.label("• G - Přepnout na režim ThirdPerson");
            ui.label("• Home - Návrat na výchozí pozici");
            ui.label("• PageUp/PageDown - Upravit rychlost");

            ui.separator();
            ui.heading("Informace o kameře");
            ui.label(format!("Aktivní režim: {:?}", mode));
            ui.collapsing("Spectator kamera", |ui| {
                ui.label(format!(
                    "Pozice: ({:.1}, {:.1}, {:.1})",
                    spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z
                ));
                ui.label(format!(
                    "Směr (yaw): {:.1}°",
                    spectator_state.yaw * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!(
                    "Náklon (pitch): {:.1}°",
                    spectator_state.pitch * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!("Rychlost: {:.1}", spectator_state.speed));
            });
            ui.collapsing("ThirdPerson kamera", |ui| {
                ui.label(format!(
                    "Pozice: ({:.1}, {:.1}, {:.1})",
                    third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z
                ));
                ui.label(format!(
                    "Směr (yaw): {:.1}°",
                    third_person_state.yaw * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!(
                    "Náklon (pitch): {:.1}°",
                    third_person_state.pitch * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!("Rychlost: {:.1}", third_person_state.speed));
            });

            ui.label(format!(
                "Aktuální pozice kamery: ({:.1}, {:.1}, {:.1})",
                cam.pos.x, cam.pos.y, cam.pos.z
            ));
            ui.label(format!(
                "Vzdálenost mezi kamerami: {:.1}",
                (spectator_state.pos - third_person_state.pos).magnitude()
            ));
            ui.checkbox(show_camera_direction, "Zobrazit směr pohledu kamery");

            if let Ok(msg) = rx.try_recv() {
                match msg {
                    crate::Message::NewFile {
                        cpu_mesh,
                        file_name,
                        load_status: _,
                        triangles: _,
                        bsp_tree: _,
                    } => {
                        *current_cpu_mesh = cpu_mesh;
                        *loaded_file_name = file_name;
                        *file_loading = false;
                        *current_triangles = crate::bsp::cpu_mesh_to_triangles(current_cpu_mesh);
                        let mut next_id = 0;
                        *bsp_root_preview = Some(crate::bsp::build_bsp(
                            current_triangles,
                            0,
                            *branch_limit,
                            &mut next_id,
                        ));
                    }
                    _ => {}
                }
            }
        });
    });
    if *tree_window_open {
        if let Some(ref root) = *bsp_root_preview {
            draw_bsp_tree_window(ctx, tree_window_open, root, selected_node);
        } else {
            *tree_window_open = false;
        }
    }
}
