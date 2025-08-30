use crate::config::CONFIG;
use cgmath::InnerSpace;
use egui::{CollapsingHeader, Grid};
use log::error;
use std::sync::atomic::AtomicUsize;
use three_d::{CpuTexture, Texture2DRef};

use crate::lang::tr;

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
    let lang = { CONFIG.read().unwrap().language };
    egui::SidePanel::left("tree").show(ctx, |side_ui| {
        egui::ScrollArea::vertical().show(side_ui, |ui| {
            ui.heading(tr(lang, "BSP Tree", "BSP Strom"));
            if ui.button(tr(lang, "⚙️ Settings", "⚙️ Nastavení")).clicked() {
                *config_window_open = true;
            }
            ui.label(format!("{}: {:?}", tr(lang, "Mode", "Režim"), mode));

            ui.separator();
            ui.heading(tr(lang, "Load Model", "Načíst model"));
            ui.label(tr(lang, "Current model:", "Aktuální model:"));
            ui.label(loaded_file_name.as_str());

            if ui.button(tr(lang, "📁 Load new model", "📁 Načíst nový model")).clicked() {
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
            if ui.button(tr(lang, "📷 Load texture", "📷 Načíst texturu")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Image", &["png", "jpg", "jpeg"])
                    .pick_file()
                {
                    match three_d_asset::io::load_and_deserialize::<CpuTexture>(&path) {
                        Ok(tex) => {
                            // Drop the previous texture (if any) so the new one is always used
                            *current_texture = None;
                            *current_texture =
                                Some(Texture2DRef::from_cpu_texture(gl, &tex));
                            *show_texture = true;
                        }
                        Err(e) => {
                            error!(
                                "Failed to load texture {}: {}",
                                path.display(),
                                e
                            );
                        }
                    }
                }
            }
            ui.checkbox(show_texture, tr(lang, "Show texture", "Zobrazit texturu"));

            if *file_loading {
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .text(tr(lang, "Loading model and building BSP tree...", "Načítání modelu a stavba BSP stromu..."))
                        .animate(true),
                );
            }

            if bsp_root_preview.is_none() {
                ui.separator();
                ui.label(tr(lang, "Building tree…", "Stavba stromu…"));
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .animate(true),
                );
                return;
            }

            ui.separator();
            ui.heading(tr(lang, "BSP Tree Structure", "Struktura BSP stromu"));
            let max_depth = CONFIG.read().unwrap().max_bsp_depth;
            ui.add(egui::Slider::new(branch_limit, 1..=max_depth).text(tr(lang, "Branch limit", "Limit větví")));
            ui.checkbox(limit_culling, tr(lang, "Limit culling by slider", "Omezit culling posuvníkem"));
            ui.checkbox(show_splitting_plane, tr(lang, "Show splitting plane", "Zobrazit dělící rovinu"));
            if ui.button(tr(lang, "Open visualization", "Otevřít vizualizaci")).clicked() {
                *tree_window_open = true;
            }

            if let Some(node_id) = *selected_node {
                if let Some(ref root) = *bsp_root_preview {
                    if let Some(node) = crate::bsp::find_node(root, node_id) {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.heading(tr(lang, "Selected node", "Vybraný uzel"));
                            if ui
                                .button("❓")
                                .on_hover_text(tr(lang, "Help", "Nápověda"))
                                .clicked()
                            {
                                *selected_node_help_open = true;
                            }
                        });
                        ui.label(format!("{}: {}", tr(lang, "ID", "ID"), node.id));
                        ui.label(format!("{}: {}", tr(lang, "Triangles", "Trojúhelníky"), node.subtree_triangles()));
                        let mut path = Vec::new();
                        let depth = if crate::bsp::find_node_path(root, node.id, &mut path) {
                            path.len().saturating_sub(1)
                        } else {
                            0
                        };
                        ui.label(format!("{}: {}", tr(lang, "Depth", "Hloubka"), depth));
                        ui.label(format!(
                            "{}: {}",
                            tr(lang, "Nodes visited to select", "Navštívené uzly při výběru"),
                            current_stats.nodes_to_selected
                        ));
                        if let Some(ref plane) = node.plane {
                            ui.label(tr(lang, "Splitting plane:", "Dělící rovina:"));
                            ui.label(format!(
                                "{}: ({:.2}, {:.2}, {:.2})",
                                tr(lang, "Normal", "Normála"),
                                plane.n.x, plane.n.y, plane.n.z
                            ));
                            ui.label(format!("{}: {:.2}", tr(lang, "Distance", "Vzdálenost"), plane.d));
                        } else {
                            ui.label(tr(lang, "Leaf (no plane)", "List (bez roviny)"));
                        }
                        if ui.button(tr(lang, "Clear selection", "Zrušit výběr")).clicked() {
                            *selected_node = None;
                        }
                    }
                }
            }

            ui.separator();
            ui.heading(tr(lang, "Display Settings", "Zobrazovací nastavení"));
            ui.checkbox(disable_culling, tr(lang, "Disable culling", "Zakázat culling"));
            ui.checkbox(show_loaded_model, tr(lang, "Show loaded model", "Zobrazit načtený model"));
            ui.checkbox(show_selected_model, tr(lang, "Show selected area", "Zobrazit vybranou oblast"));

            ui.separator();
            ui.heading(tr(lang, "BSP Stats", "Statistiky BSP"));
            Grid::new("bsp_stats_grid")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label(tr(lang, "Total nodes", "Celkem uzlů"));
                    ui.label(format!("{}", current_stats.total_nodes));
                    ui.end_row();
                    ui.label(tr(lang, "Total triangles", "Celkem trojúhelníků"));
                    ui.label(format!("{}", current_stats.total_triangles));
                    ui.end_row();
                    ui.label(tr(lang, "Nodes visited", "Navštívené uzly"));
                    ui.label(format!("{}", current_stats.nodes_visited));
                    ui.end_row();
                    ui.label(tr(lang, "Triangles rendered", "Vykreslené trojúhelníky"));
                    ui.label(format!("{}", current_stats.triangles_rendered));
                    ui.end_row();
                    ui.label(tr(lang, "Traversal efficiency", "Efektivita průchodu"));
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
            ui.heading(tr(lang, "Mesh Info", "Informace o meshi"));
            Grid::new("mesh_info_grid").num_columns(2).show(ui, |ui| {
                ui.label(tr(lang, "Vertices", "Vrcholy"));
                ui.label(format!("{}", current_cpu_mesh.positions.len()));
                ui.end_row();

                ui.label(tr(lang, "Indices", "Indexy"));
                match &current_cpu_mesh.indices {
                    three_d_asset::Indices::U32(idx) => {
                        ui.label(format!("U32: {}", idx.len()));
                    }
                    three_d_asset::Indices::U16(idx) => {
                        ui.label(format!("U16: {}", idx.len()));
                    }
                    _ => {
                        ui.label(tr(lang, "none", "žádné"));
                    }
                }
                ui.end_row();
            });

            ui.separator();
            ui.heading(tr(lang, "Controls", "Ovládání"));
            CollapsingHeader::new(tr(lang, "Movement", "Pohyb")).show(ui, |ui| {
                ui.label(tr(lang, "W - Forward", "W - vpřed"));
                ui.label(tr(lang, "S - Backward", "S - vzad"));
                ui.label(tr(lang, "A - Left", "A - doleva"));
                ui.label(tr(lang, "D - Right", "D - doprava"));
                ui.label(tr(lang, "Space - Up", "Mezerník - nahoru"));
                ui.label(tr(lang, "C - Down", "C - dolů"));
                ui.label(format!("{}: {:.1}", tr(lang, "Speed", "Rychlost"), cam.speed));
            });
            CollapsingHeader::new(tr(lang, "Looking around", "Rozhlížení"))
                .show(ui, |ui| {
                    ui.label(tr(lang, "↑ - Look up", "↑ - dívat se nahoru"));
                    ui.label(tr(lang, "↓ - Look down", "↓ - dívat se dolů"));
                    ui.label(tr(lang, "← - Turn left", "← - otočit doleva"));
                    ui.label(tr(lang, "→ - Turn right", "→ - otočit doprava"));
                    ui.label(format!(
                        "{}: {:.1}°/s",
                        tr(lang, "Look speed", "Rychlost otáčení"),
                        cam.look_speed * 180.0 / std::f32::consts::PI
                    ));
                    ui.add(
                        egui::Slider::new(&mut cam.look_speed, 0.5..=5.0)
                            .text(tr(lang, "Look speed", "Rychlost otáčení")),
                    );
                });
            CollapsingHeader::new(tr(lang, "Misc", "Různé")).show(ui, |ui| {
                ui.label(tr(lang, "F - Switch to Spectator mode", "F - přepnout na režim diváka"));
                ui.label(tr(lang, "G - Switch to ThirdPerson mode", "G - přepnout na režim třetí osoby"));
                ui.label(tr(lang, "Home - Reset to default position", "Home - reset na výchozí pozici"));
                ui.label(tr(lang, "PageUp/PageDown - Adjust speed", "PageUp/PageDown - upravit rychlost"));
            });

            ui.separator();
            ui.heading(tr(lang, "Camera Info", "Informace o kameře"));
            ui.label(format!("{}: {:?}", tr(lang, "Active mode", "Aktivní režim"), mode));
            ui.collapsing(tr(lang, "Spectator camera", "Kamera diváka"), |ui| {
                ui.label(format!(
                    "{}: ({:.1}, {:.1}, {:.1})",
                    tr(lang, "Position", "Pozice"),
                    spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z
                ));
                ui.label(format!(
                    "{}: {:.1}°",
                    tr(lang, "Yaw", "Yaw"),
                    spectator_state.yaw * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!(
                    "{}: {:.1}°",
                    tr(lang, "Pitch", "Pitch"),
                    spectator_state.pitch * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!("{}: {:.1}", tr(lang, "Speed", "Rychlost"), spectator_state.speed));
            });
            ui.collapsing(tr(lang, "Third-person camera", "Kamera třetí osoby"), |ui| {
                ui.label(format!(
                    "{}: ({:.1}, {:.1}, {:.1})",
                    tr(lang, "Position", "Pozice"),
                    third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z
                ));
                ui.label(format!(
                    "{}: {:.1}°",
                    tr(lang, "Yaw", "Yaw"),
                    third_person_state.yaw * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!(
                    "{}: {:.1}°",
                    tr(lang, "Pitch", "Pitch"),
                    third_person_state.pitch * 180.0 / std::f32::consts::PI
                ));
                ui.label(format!("{}: {:.1}", tr(lang, "Speed", "Rychlost"), third_person_state.speed));
            });

            ui.label(format!(
                "{}: ({:.1}, {:.1}, {:.1})",
                tr(lang, "Current camera position", "Aktuální pozice kamery"),
                cam.pos.x, cam.pos.y, cam.pos.z
            ));
            ui.label(format!(
                "{}: {:.1}",
                tr(lang, "Distance between cameras", "Vzdálenost mezi kamerami"),
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
        egui::Window::new(tr(lang, "Help - Selected node", "Nápověda - Vybraný uzel"))
            .open(selected_node_help_open)
            .show(ctx, |ui| {
                ui.label(tr(
                    lang,
                    "This section shows details about the currently selected node in the BSP tree.",
                    "Tato část ukazuje detaily o aktuálně vybraném uzlu v BSP stromu.",
                ));
                ui.separator();
                ui.label(tr(
                    lang,
                    "• Depth indicates how many edges lead from the root to this node. The root has depth 0 and each level increases it by 1.",
                    "• Hloubka udává, kolik hran vede od kořene k tomuto uzlu. Kořen má hloubku 0 a každá úroveň ji zvýší o 1.",
                ));
                ui.label(tr(
                    lang,
                    "• Nodes visited to select is the number of nodes the algorithm examined before finding this node. The number can be higher than the depth because other branches are checked.",
                    "• Navštívené uzly při výběru je počet uzlů, které algoritmus prohlédl před nalezením tohoto uzlu. Číslo může být větší než hloubka, protože se kontrolují i jiné větve.",
                ));
                ui.separator();
                ui.label(tr(lang, "Example:", "Příklad:"));
                ui.label(tr(
                    lang,
                    "A node at depth 3 means path root → A → B → C. If the algorithm checks two side branches, the total visited nodes may be 5.",
                    "Uzel v hloubce 3 znamená cestu kořen → A → B → C. Pokud algoritmus zkontroluje dvě vedlejší větve, může být počet navštívených uzlů 5.",
                ));
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
