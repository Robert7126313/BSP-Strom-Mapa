use crate::config::CONFIG;
use cgmath::InnerSpace;
use egui::{CollapsingHeader, Grid};
use egui_plot::{Line, Plot, PlotPoint, Points, Text};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use three_d::{CpuTexture, Srgba, Texture2DRef};

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

            let cfg = CONFIG.lock().unwrap().clone();
            let highlight_color = cfg.bsp_tree_path_color;
            let selected_color = cfg.bsp_tree_selected_color;

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
                    if let Some(id) = best {
                        *selected = Some(id);
                    }
                }
            }
        });
}

pub fn draw_left_panel(
    ctx: &egui::Context,
    gl: &three_d::Context,
    mode: crate::camera::CamMode,
    loaded_file_name: &mut String,
    file_loading: &mut bool,
    tx_gui: &std::sync::mpsc::Sender<crate::Message>,
    rx: &std::sync::mpsc::Receiver<crate::Message>,
    current_cpu_mesh: &mut three_d::CpuMesh,
    current_triangles: &mut Vec<crate::bsp::Triangle>,
    current_texture: &mut Option<Texture2DRef>,
    bsp_root_preview: &mut Option<crate::bsp::BspNode>,
    selected_node: &mut Option<usize>,
    show_splitting_plane: &mut bool,
    disable_culling: &mut bool,
    show_loaded_model: &mut bool,
    show_selected_model: &mut bool,
    show_texture: &mut bool,
    show_spectator_marker: &mut bool,
    spectator_state: &mut crate::camera::CameraState,
    third_person_state: &mut crate::camera::CameraState,
    cam: &mut crate::camera::FreeCamera,
    current_stats: &crate::bsp::BspStats,
    config_window_open: &mut bool,
    tree_window_open: &mut bool,
    branch_limit: &mut u32,
    limit_culling: &mut bool,
    selected_node_help_open: &mut bool,
) {
    egui::SidePanel::left("tree").show(ctx, |side_ui| {
        egui::ScrollArea::vertical().show(side_ui, |ui| {
            ui.heading("BSP Strom");
            if ui.button("⚙️ Nastavení").clicked() {
                *config_window_open = true;
            }
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
                        let (new_cpu_mesh, new_texture, _load_status) =
                            crate::load_cpu_mesh(&path_clone);
                        let _ = tx_gui_clone.send(crate::Message::NewFile {
                            cpu_mesh: new_cpu_mesh,
                            texture: new_texture,
                            file_name: file_name_clone,
                        });
                    });
                }
            }
            if ui.button("📷 Načíst texturu").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Image", &["png", "jpg", "jpeg"])
                    .pick_file()
                {
                    if let Ok(tex) = three_d_asset::io::load_and_deserialize::<CpuTexture>(&path) {
                        *current_texture = Some(Texture2DRef::from_cpu_texture(gl, &tex));
                        *show_texture = true;
                    }
                }
            }
            ui.checkbox(show_texture, "Zobrazit texturu");

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
            let max_depth = CONFIG.lock().unwrap().max_bsp_depth;
            ui.add(egui::Slider::new(branch_limit, 1..=max_depth).text("Limit větvení"));
            ui.checkbox(limit_culling, "Omezit culling podle slideru");
            ui.checkbox(show_splitting_plane, "Zobrazit dělící rovinu");
            if ui.button("Otevřít vizualizaci").clicked() {
                *tree_window_open = true;
            }

            if let Some(node_id) = *selected_node {
                if let Some(ref root) = *bsp_root_preview {
                    if let Some(node) = crate::bsp::find_node(root, node_id) {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.heading("Vybraný uzel");
                            if ui.button("❓").on_hover_text("Vysvětlivky").clicked() {
                                *selected_node_help_open = true;
                            }
                        });
                        ui.label(format!("ID: {}", node.id));
                        ui.label(format!("Trojúhelníků: {}", node.subtree_triangles()));
                        // Additional contextual information about the selected node
                        // How deep in the tree this node is
                        let mut path = Vec::new();
                        let depth = if crate::bsp::find_node_path(root, node.id, &mut path) {
                            path.len().saturating_sub(1)
                        } else {
                            0
                        };
                        ui.label(format!("Hloubka: {}", depth));
                        // How many nodes were visited when this node was picked
                        ui.label(format!(
                            "Navštíveno uzlů při výběru: {}",
                            current_stats.nodes_to_selected
                        ));
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
                        if ui.button("Odznačit").clicked() {
                            *selected_node = None;
                        }
                    }
                }
            }

            ui.separator();
            ui.heading("Nastavení zobrazení");
            ui.checkbox(disable_culling, "Vypnout culling");
            ui.checkbox(show_loaded_model, "Zobrazit načtený model");
            ui.checkbox(show_selected_model, "Zobrazit vybranou oblast");

            ui.separator();
            ui.heading("BSP Statistiky");
            Grid::new("bsp_stats_grid")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Celkem uzlů");
                    ui.label(format!("{}", current_stats.total_nodes));
                    ui.end_row();

                    ui.label("Celkem trojúhelníků");
                    ui.label(format!("{}", current_stats.total_triangles));
                    ui.end_row();

                    ui.label("Navštíveno uzlů");
                    ui.label(format!("{}", current_stats.nodes_visited));
                    ui.end_row();

                    ui.label("Vykresleno trojúhelníků");
                    ui.label(format!("{}", current_stats.triangles_rendered));
                    ui.end_row();

                    ui.label("Procházka efektivita");
                    let efficiency = if current_stats.total_nodes > 0 {
                        (current_stats.nodes_visited as f32 / current_stats.total_nodes as f32)
                            * 100.0
                    } else {
                        0.0
                    };
                    ui.label(format!("{:.1}%", efficiency));
                    ui.end_row();
                });

            ui.separator();
            ui.heading("Mesh Info");
            Grid::new("mesh_info_grid").num_columns(2).show(ui, |ui| {
                ui.label("Vrcholy");
                ui.label(format!("{}", current_cpu_mesh.positions.len()));
                ui.end_row();

                ui.label("Indexy");
                match &current_cpu_mesh.indices {
                    three_d_asset::Indices::U32(idx) => {
                        ui.label(format!("U32: {}", idx.len()));
                    }
                    three_d_asset::Indices::U16(idx) => {
                        ui.label(format!("U16: {}", idx.len()));
                    }
                    _ => {
                        ui.label("žádné");
                    }
                }
                ui.end_row();
            });

            ui.separator();
            ui.heading("Ovládání");
            CollapsingHeader::new("Pohyb").show(ui, |ui| {
                ui.label("W - Dopředu");
                ui.label("S - Dozadu");
                ui.label("A - Doleva");
                ui.label("D - Doprava");
                ui.label("Space - Nahoru");
                ui.label("C - Dolů");
                ui.label(format!("Rychlost: {:.1}", cam.speed));
            });
            CollapsingHeader::new("Rozhlížení").show(ui, |ui| {
                ui.label("↑ - Díváš se nahoru");
                ui.label("↓ - Díváš se dolů");
                ui.label("← - Otočit hlavu doleva");
                ui.label("→ - Otočit hlavu doprava");
                ui.label(format!(
                    "Rychlost rozhlížení: {:.1}°/s",
                    cam.look_speed * 180.0 / std::f32::consts::PI
                ));
                ui.add(
                    egui::Slider::new(&mut cam.look_speed, 0.5..=5.0).text("Rychlost rozhlížení"),
                );
            });
            CollapsingHeader::new("Ostatní").show(ui, |ui| {
                ui.label("F - Přepnout na režim Spectator");
                ui.label("G - Přepnout na režim ThirdPerson");
                ui.label("Home - Návrat na výchozí pozici");
                ui.label("PageUp/PageDown - Upravit rychlost");
            });

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
            ui.checkbox(show_spectator_marker, "Zobrazit pozici a směr spectatoru");

            if let Ok(msg) = rx.try_recv() {
                match msg {
                    crate::Message::NewFile {
                        cpu_mesh,
                        texture,
                        file_name,
                    } => {
                        *current_cpu_mesh = cpu_mesh;
                        *loaded_file_name = file_name;
                        *file_loading = false;
                        *current_triangles = crate::bsp::cpu_mesh_to_triangles(current_cpu_mesh);
                        *current_texture = texture.map(|t| Texture2DRef::from_cpu_texture(gl, &t));
                        if current_texture.is_some() {
                            *show_texture = true;
                        }
                        let next_id = AtomicUsize::new(0);
                        *bsp_root_preview = Some(crate::bsp::build_bsp(
                            current_triangles,
                            0,
                            *branch_limit,
                            &next_id,
                        ));
                    }
                    _ => {}
                }
            }
        });
    });
    if *config_window_open {
        draw_config_window(
            ctx,
            config_window_open,
            cam,
            spectator_state,
            third_person_state,
        );
    }
    if *selected_node_help_open {
        egui::Window::new("Vysvětlivky - Vybraný uzel")
            .open(selected_node_help_open)
            .show(ctx, |ui| {
                ui.label("Tato sekce zobrazuje detaily o aktuálně vybraném uzlu v BSP stromu.");
                ui.separator();
                ui.label("• Hloubka udává, kolik hran vede od kořene k tomuto uzlu. Kořen má hloubku 0 a každá úroveň ji zvyšuje o 1.");
                ui.label("• Navštíveno uzlů při výběru je počet uzlů, které algoritmus prozkoumal, než našel tento uzel. Číslo bývá vyšší než hloubka, protože se kontrolují i jiné větve stromu.");
                ui.separator();
                ui.label("Praktický příklad:");
                ui.label("Uzel na hloubce 3 znamená cestu kořen → A → B → C. Pokud algoritmus při hledání zkontroluje ještě dvě vedlejší větve, může být celkový počet navštívených uzlů 5.");
            });
    }
    if *tree_window_open {
        if let Some(ref root) = *bsp_root_preview {
            draw_bsp_tree_window(ctx, tree_window_open, root, selected_node);
        } else {
            *tree_window_open = false;
        }
    }
}

fn draw_config_window(
    ctx: &egui::Context,
    open: &mut bool,
    cam: &mut crate::camera::FreeCamera,
    spectator_state: &mut crate::camera::CameraState,
    third_person_state: &mut crate::camera::CameraState,
) {
    egui::Window::new("Config")
        .open(open)
        .vscroll(true)
        .show(ctx, |ui| {
            let mut cfg = CONFIG.lock().unwrap();

            ui.heading("Colors & Lighting");
            Grid::new("color_settings").num_columns(2).show(ui, |ui| {
                ui.label("Background");
                ui.color_edit_button_rgb(&mut cfg.bg_color);
                ui.end_row();

                let mut color = egui::Color32::from_rgba_unmultiplied(
                    cfg.model_color.r,
                    cfg.model_color.g,
                    cfg.model_color.b,
                    cfg.model_color.a,
                );
                ui.label("Model");
                if ui.color_edit_button_srgba(&mut color).changed() {
                    cfg.model_color = Srgba::new(color.r(), color.g(), color.b(), color.a());
                }
                ui.end_row();

                let mut hcol = egui::Color32::from_rgba_unmultiplied(
                    cfg.highlight_color.r,
                    cfg.highlight_color.g,
                    cfg.highlight_color.b,
                    cfg.highlight_color.a,
                );
                ui.label("Highlight");
                if ui.color_edit_button_srgba(&mut hcol).changed() {
                    cfg.highlight_color = Srgba::new(hcol.r(), hcol.g(), hcol.b(), hcol.a());
                }
                ui.end_row();

                ui.label("Ambient color");
                let mut acol = egui::Color32::from_rgba_unmultiplied(
                    cfg.ambient_light_color.r,
                    cfg.ambient_light_color.g,
                    cfg.ambient_light_color.b,
                    cfg.ambient_light_color.a,
                );
                if ui.color_edit_button_srgba(&mut acol).changed() {
                    cfg.ambient_light_color = Srgba::new(acol.r(), acol.g(), acol.b(), acol.a());
                }
                ui.end_row();
            });

            ui.add(
                egui::Slider::new(&mut cfg.ambient_light_intensity, 0.0..=5.0)
                    .text("Ambient intensity"),
            );

            ui.separator();
            ui.heading("BSP Tree");
            ui.add(egui::Slider::new(&mut cfg.bsp_tree_text_size, 8.0..=32.0).text("Text size"));
            Grid::new("bsp_tree_colors").num_columns(2).show(ui, |ui| {
                ui.label("Path color");
                ui.color_edit_button_srgba(&mut cfg.bsp_tree_path_color);
                ui.end_row();

                ui.label("Selected color");
                ui.color_edit_button_srgba(&mut cfg.bsp_tree_selected_color);
                ui.end_row();
            });

            ui.separator();
            ui.heading("BSP Limits");
            ui.add(egui::Slider::new(&mut cfg.max_bsp_depth, 1..=64).text("Max depth"));
            let max_depth = cfg.max_bsp_depth;
            ui.add(
                egui::Slider::new(&mut cfg.default_branch_limit, 1..=max_depth)
                    .text("Default branch limit"),
            );
            ui.add(
                egui::Slider::new(&mut cfg.min_triangles_per_leaf, 1..=100)
                    .text("Min triangles per leaf"),
            );

            ui.separator();
            ui.heading("Camera");
            if ui
                .add(
                    egui::Slider::new(&mut cfg.default_camera_speed, 0.1..=20.0)
                        .text("Default speed"),
                )
                .changed()
            {
                cam.speed = cfg.default_camera_speed;
                spectator_state.speed = cfg.default_camera_speed;
                third_person_state.speed = cfg.default_camera_speed;
            }
            if ui
                .add(egui::Slider::new(&mut cfg.default_look_speed, 0.1..=10.0).text("Look speed"))
                .changed()
            {
                cam.look_speed = cfg.default_look_speed;
            }
            ui.add(egui::Slider::new(&mut cfg.pitch_limit, 0.1..=3.14).text("Pitch limit"));
            ui.add(egui::Slider::new(&mut cfg.default_fov_deg, 30.0..=120.0).text("FOV deg"));
            ui.add(egui::Slider::new(&mut cfg.near_plane, 0.01..=1.0).text("Near plane"));
            ui.add(egui::Slider::new(&mut cfg.far_plane, 10.0..=5000.0).text("Far plane"));
            ui.add(
                egui::Slider::new(&mut cfg.camera_switch_cooldown, 0.1..=10.0)
                    .text("Switch cooldown"),
            );
            ui.add(
                egui::Slider::new(&mut cfg.speed_adjustment_factor, 1.01..=5.0)
                    .text("Speed factor"),
            );

            ui.horizontal(|ui| {
                ui.label("Spectator pos");
                ui.add(egui::DragValue::new(&mut cfg.default_spectator_pos.x));
                ui.add(egui::DragValue::new(&mut cfg.default_spectator_pos.y));
                ui.add(egui::DragValue::new(&mut cfg.default_spectator_pos.z));
            });
            ui.horizontal(|ui| {
                ui.label("Third person pos");
                ui.add(egui::DragValue::new(&mut cfg.default_third_person_pos.x));
                ui.add(egui::DragValue::new(&mut cfg.default_third_person_pos.y));
                ui.add(egui::DragValue::new(&mut cfg.default_third_person_pos.z));
            });
            ui.add(
                egui::Slider::new(&mut cfg.camera_marker_scale, 0.01..=1.0).text("Marker scale"),
            );
            ui.add(
                egui::Slider::new(&mut cfg.direction_ray_thickness, 0.01..=1.0)
                    .text("Ray thickness"),
            );
            ui.add(egui::Slider::new(&mut cfg.direction_ray_length, 0.1..=10.0).text("Ray length"));
        });
}
