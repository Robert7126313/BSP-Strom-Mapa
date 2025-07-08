use cgmath::InnerSpace;

pub fn draw_left_panel(
    ctx: &egui::Context,
    mode: crate::camera::CamMode,
    loaded_file_name: &mut String,
    file_loading: &mut bool,
    tx_gui: &std::sync::mpsc::Sender<crate::Message>,
    rx: &std::sync::mpsc::Receiver<crate::Message>,
    current_cpu_mesh: &mut three_d::CpuMesh,
    current_triangles: &mut Vec<crate::bsp::Triangle>,
    bsp_root: &mut Option<crate::bsp::BspNode>,
    selected_node: &mut Option<usize>,
    show_splitting_plane: &mut bool,
    disable_culling: &mut bool,
    show_camera_direction: &mut bool,
    spectator_state: &mut crate::camera::CameraState,
    third_person_state: &mut crate::camera::CameraState,
    cam: &mut crate::camera::FreeCamera,
    current_stats: &crate::bsp::BspStats,
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
                    let file_name_clone = path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
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

            if bsp_root.is_none() {
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
            ui.checkbox(show_splitting_plane, "Zobrazit dělící rovinu");

            ui.separator();
            ui.heading("Nastavení zobrazení");
            ui.checkbox(disable_culling, "Zobrazit celý BSP strom (bez cullingu)");
            if *disable_culling {
                ui.label("Varování: Zobrazení celého stromu může zpomalit vykreslování.");
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                let root = bsp_root.as_ref().unwrap();
                crate::bsp::render_bsp_tree(ui, root, selected_node);
            });

            if let Some(node_id) = *selected_node {
                if let Some(ref root) = *bsp_root {
                    if let Some(node) = crate::bsp::find_node(root, node_id) {
                        ui.separator();
                        ui.heading("Vybraný uzel");
                        ui.label(format!("ID: {}", node.id));
                        ui.label(format!("Trojúhelníků: {}", node.triangles.len()));
                        if let Some(ref plane) = node.plane {
                            ui.label("Dělící rovina:");
                            ui.label(format!("Normála: ({:.2}, {:.2}, {:.2})", plane.n.x, plane.n.y, plane.n.z));
                            ui.label(format!("Vzdálenost: {:.2}", plane.d));
                        } else {
                            ui.label("List (bez dělící roviny)");
                        }
                        ui.label("Obalový objem:");
                        ui.label(format!("Min: ({:.2}, {:.2}, {:.2})", node.bounds.min.x, node.bounds.min.y, node.bounds.min.z));
                        ui.label(format!("Max: ({:.2}, {:.2}, {:.2})", node.bounds.max.x, node.bounds.max.y, node.bounds.max.z));
                        if ui.button("Odznačit").clicked() {
                            *selected_node = None;
                        }
                    }
                }
            }

            ui.separator();
            ui.heading("BSP Statistiky");
            ui.label(format!("Celkem uzlů: {}", current_stats.total_nodes));
            ui.label(format!("Celkem trojúhelníků: {}", current_stats.total_triangles));
            ui.label(format!("Navštíveno uzlů: {}", current_stats.nodes_visited));
            ui.label(format!("Vykresleno trojúhelníků: {}", current_stats.triangles_rendered));
            ui.label(format!("Procházka efektivita: {:.1}%", if current_stats.total_nodes > 0 { (current_stats.nodes_visited as f32 / current_stats.total_nodes as f32) * 100.0 } else { 0.0 }));

            ui.separator();
            ui.heading("Mesh Info");
            ui.label(format!("Vrcholy: {}", current_cpu_mesh.positions.len()));
            match &current_cpu_mesh.indices {
                three_d_asset::Indices::U32(idx) => ui.label(format!("Indexy (U32): {}", idx.len())),
                three_d_asset::Indices::U16(idx) => ui.label(format!("Indexy (U16): {}", idx.len())),
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
            ui.label(format!("Rychlost rozhlížení: {:.1}°/s", cam.look_speed * 180.0 / std::f32::consts::PI));
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
                ui.label(format!("Pozice: ({:.1}, {:.1}, {:.1})", spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z));
                ui.label(format!("Směr (yaw): {:.1}°", spectator_state.yaw * 180.0 / std::f32::consts::PI));
                ui.label(format!("Náklon (pitch): {:.1}°", spectator_state.pitch * 180.0 / std::f32::consts::PI));
                ui.label(format!("Rychlost: {:.1}", spectator_state.speed));
            });
            ui.collapsing("ThirdPerson kamera", |ui| {
                ui.label(format!("Pozice: ({:.1}, {:.1}, {:.1})", third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z));
                ui.label(format!("Směr (yaw): {:.1}°", third_person_state.yaw * 180.0 / std::f32::consts::PI));
                ui.label(format!("Náklon (pitch): {:.1}°", third_person_state.pitch * 180.0 / std::f32::consts::PI));
                ui.label(format!("Rychlost: {:.1}", third_person_state.speed));
            });

            ui.label(format!("Aktuální pozice kamery: ({:.1}, {:.1}, {:.1})", cam.pos.x, cam.pos.y, cam.pos.z));
            ui.label(format!("Vzdálenost mezi kamerami: {:.1}", (spectator_state.pos - third_person_state.pos).magnitude()));
            ui.checkbox(show_camera_direction, "Zobrazit směr pohledu kamery");

            ui.separator();
            ui.heading("Načtení modelu");
            ui.label("Aktuální model:");
            ui.label(loaded_file_name.as_str());
            if ui.button("Načíst nový model").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("GLTF/GLB files", &["gltf", "glb"])
                    .pick_file()
                {
                    *file_loading = true;
                    let path_clone = path.clone();
                    let file_name_clone = path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
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
                ui.label("Načítání nového modelu...");
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .desired_width(ui.available_width())
                        .animate(true),
                );
            }
            if let Ok(msg) = rx.try_recv() {
                match msg {
                    crate::Message::NewFile { cpu_mesh, file_name, load_status: _, triangles: _, bsp_tree: _ } => {
                        *current_cpu_mesh = cpu_mesh;
                        *loaded_file_name = file_name;
                        *file_loading = false;
                        *current_triangles = crate::bsp::cpu_mesh_to_triangles(current_cpu_mesh);
                        let mut next_id = 0;
                        *bsp_root = Some(crate::bsp::build_bsp(current_triangles, 0, &mut next_id));
                    }
                    _ => {}
                }
            }
            ui.label("Debug info:");
            ui.label(format!("Vrcholy: {}", current_cpu_mesh.positions.len()));
            match &current_cpu_mesh.indices {
                three_d_asset::Indices::U32(idx) => ui.label(format!("Indexy (U32): {}", idx.len())),
                three_d_asset::Indices::U16(idx) => ui.label(format!("Indexy (U16): {}", idx.len())),
                _ => ui.label("Indexy: žádné"),
            };
        });
    });
}
