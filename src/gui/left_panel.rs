use crate::config::CONFIG;
use cgmath::InnerSpace;
use egui::{CollapsingHeader, Grid};
use std::sync::atomic::AtomicUsize;
use three_d::{CpuTexture, Texture2DRef};

use super::{config::draw_config_window, tree::draw_bsp_tree_window};

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
            ui.heading("BSP Tree");
            if ui.button("⚙️ Settings").clicked() {
                *config_window_open = true;
            }
            ui.label(format!("Mode: {:?}", mode));

            ui.separator();
            ui.heading("Load Model");
            ui.label("Current model:");
            ui.label(loaded_file_name.as_str());

            if ui.button("📁 Load new model").clicked() {
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
            if ui.button("📷 Load texture").clicked() {
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
            ui.checkbox(show_texture, "Show texture");

            if *file_loading {
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .text("Loading model and building BSP tree...")
                        .animate(true),
                );
            }

            if bsp_root_preview.is_none() {
                ui.separator();
                ui.label("Building tree…");
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .animate(true),
                );
                return;
            }

            ui.separator();
            ui.heading("BSP Tree Structure");
            let max_depth = CONFIG.read().unwrap().max_bsp_depth;
            ui.add(egui::Slider::new(branch_limit, 1..=max_depth).text("Branch limit"));
            ui.checkbox(limit_culling, "Limit culling by slider");
            ui.checkbox(show_splitting_plane, "Show splitting plane");
            if ui.button("Open visualization").clicked() {
                *tree_window_open = true;
            }

            if let Some(node_id) = *selected_node {
                if let Some(ref root) = *bsp_root_preview {
                    if let Some(node) = crate::bsp::find_node(root, node_id) {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.heading("Selected node");
                            if ui.button("❓").on_hover_text("Help").clicked() {
                                *selected_node_help_open = true;
                            }
                        });
                        ui.label(format!("ID: {}", node.id));
                        ui.label(format!("Triangles: {}", node.subtree_triangles()));
                        let mut path = Vec::new();
                        let depth = if crate::bsp::find_node_path(root, node.id, &mut path) {
                            path.len().saturating_sub(1)
                        } else {
                            0
                        };
                        ui.label(format!("Depth: {}", depth));
                        ui.label(format!(
                            "Nodes visited to select: {}",
                            current_stats.nodes_to_selected
                        ));
                        if let Some(ref plane) = node.plane {
                            ui.label("Splitting plane:");
                            ui.label(format!(
                                "Normal: ({:.2}, {:.2}, {:.2})",
                                plane.n.x, plane.n.y, plane.n.z
                            ));
                            ui.label(format!("Distance: {:.2}", plane.d));
                        } else {
                            ui.label("Leaf (no plane)");
                        }
                        if ui.button("Clear selection").clicked() {
                            *selected_node = None;
                        }
                    }
                }
            }

            ui.separator();
            ui.heading("Display Settings");
            ui.checkbox(disable_culling, "Disable culling");
            ui.checkbox(show_loaded_model, "Show loaded model");
            ui.checkbox(show_selected_model, "Show selected area");

            ui.separator();
            ui.heading("BSP Stats");
            Grid::new("bsp_stats_grid")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Total nodes");
                    ui.label(format!("{}", current_stats.total_nodes));
                    ui.end_row();

                    ui.label("Total triangles");
                    ui.label(format!("{}", current_stats.total_triangles));
                    ui.end_row();

                    ui.label("Nodes visited");
                    ui.label(format!("{}", current_stats.nodes_visited));
                    ui.end_row();

                    ui.label("Triangles rendered");
                    ui.label(format!("{}", current_stats.triangles_rendered));
                    ui.end_row();

                    ui.label("Traversal efficiency");
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
                ui.label("Vertices");
                ui.label(format!("{}", current_cpu_mesh.positions.len()));
                ui.end_row();

                ui.label("Indices");
                match &current_cpu_mesh.indices {
                    three_d_asset::Indices::U32(idx) => {
                        ui.label(format!("U32: {}", idx.len()));
                    }
                    three_d_asset::Indices::U16(idx) => {
                        ui.label(format!("U16: {}", idx.len()));
                    }
                    _ => {
                        ui.label("none");
                    }
                }
                ui.end_row();
            });

            ui.separator();
            ui.heading("Controls");
            CollapsingHeader::new("Movement").show(ui, |ui| {
                ui.label("W - Forward");
                ui.label("S - Backward");
                ui.label("A - Left");
                ui.label("D - Right");
                ui.label("Space - Up");
                ui.label("C - Down");
                ui.label(format!("Speed: {:.1}", cam.speed));
            });
            CollapsingHeader::new("Looking around").show(ui, |ui| {
                ui.label("↑ - Look up");
                ui.label("↓ - Look down");
                ui.label("← - Turn left");
                ui.label("→ - Turn right");
                ui.label(format!(
                    "Look speed: {:.1}°/s",
                    cam.look_speed * 180.0 / std::f32::consts::PI
                ));
                ui.add(
                    egui::Slider::new(&mut cam.look_speed, 0.5..=5.0).text("Look speed"),
                );
            });
            CollapsingHeader::new("Misc").show(ui, |ui| {
                ui.label("F - Switch to Spectator mode");
                ui.label("G - Switch to ThirdPerson mode");
                ui.label("Home - Reset to default position");
                ui.label("PageUp/PageDown - Adjust speed");
            });

            ui.separator();
            ui.heading("Camera Info");
            ui.label(format!("Active mode: {:?}", mode));
            ui.collapsing("Spectator camera", |ui| {
                ui.label(format!(
                    "Position: ({:.1}, {:.1}, {:.1})",
                    spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z
                ));
                ui.label(format!(
                    "Yaw: {:.1}°",
                    spectator_state.yaw * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!(
                    "Pitch: {:.1}°",
                    spectator_state.pitch * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!("Speed: {:.1}", spectator_state.speed));
            });
            ui.collapsing("Third-person camera", |ui| {
                ui.label(format!(
                    "Position: ({:.1}, {:.1}, {:.1})",
                    third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z
                ));
                ui.label(format!(
                    "Yaw: {:.1}°",
                    third_person_state.yaw * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!(
                    "Pitch: {:.1}°",
                    third_person_state.pitch * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!("Speed: {:.1}", third_person_state.speed));
            });

            ui.label(format!(
                "Current camera position: ({:.1}, {:.1}, {:.1})",
                cam.pos.x, cam.pos.y, cam.pos.z
            ));
            ui.label(format!(
                "Distance between cameras: {:.1}",
                (spectator_state.pos - third_person_state.pos).magnitude()
            ));

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
            cam,
            spectator_state,
            third_person_state,
            mode,
            show_spectator_marker,
            config_window_open,
        );
    }
    if *selected_node_help_open {
        egui::Window::new("Help - Selected node")
            .open(selected_node_help_open)
            .show(ctx, |ui| {
                ui.label("This section shows details about the currently selected node in the BSP tree.");
                ui.separator();
                ui.label("• Depth indicates how many edges lead from the root to this node. The root has depth 0 and each level increases it by 1.");
                ui.label("• Nodes visited to select is the number of nodes the algorithm examined before finding this node. The number can be higher than the depth because other branches are checked.");
                ui.separator();
                ui.label("Example:");
                ui.label("A node at depth 3 means path root → A → B → C. If the algorithm checks two side branches, the total visited nodes may be 5.");
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
