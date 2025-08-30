use crate::config::{Config, CONFIG};
use three_d::Srgba;

pub fn draw_config_window(
    ctx: &egui::Context,
    cam: &mut crate::camera::FreeCamera,
    spectator_state: &mut crate::camera::CameraState,
    third_person_state: &mut crate::camera::CameraState,
    mode: crate::camera::CamMode,
    show_spectator_marker: &mut bool,
    open: &mut bool,
) {
    egui::Window::new("Config")
        .open(open)
        .vscroll(true)
        .show(ctx, |ui| {
            let mut cfg = CONFIG.write().unwrap();

            if ui.button("Load default").clicked() {
                let defaults = Config::default();
                cam.speed = defaults.camera_speed;
                cam.look_speed = defaults.look_speed;
                spectator_state.pos = defaults.default_spectator_pos;
                spectator_state.yaw = defaults.default_spectator_yaw;
                spectator_state.pitch = defaults.default_spectator_pitch;
                third_person_state.pos = defaults.default_third_person_pos;
                third_person_state.yaw = defaults.default_third_person_yaw;
                third_person_state.pitch = defaults.default_third_person_pitch;
                if mode == crate::camera::CamMode::Spectator {
                    cam.pos = spectator_state.pos;
                    cam.yaw = spectator_state.yaw;
                    cam.pitch = spectator_state.pitch;
                } else if mode == crate::camera::CamMode::ThirdPerson {
                    cam.pos = third_person_state.pos;
                    cam.yaw = third_person_state.yaw;
                    cam.pitch = third_person_state.pitch;
                }
                *show_spectator_marker = false;
                *cfg = defaults;
            }

            ui.heading("Colors & Lighting");
            ui.horizontal(|ui| {
                ui.label("Background");
                ui.color_edit_button_rgb(&mut cfg.bg_color);
            });

            let mut color = [
                cfg.model_color.r,
                cfg.model_color.g,
                cfg.model_color.b,
                cfg.model_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label("Model");
                if ui.color_edit_button_srgba_unmultiplied(&mut color).changed() {
                    cfg.model_color = Srgba::new(color[0], color[1], color[2], color[3]);
                }
            });

            let mut hcol = [
                cfg.highlight_color.r,
                cfg.highlight_color.g,
                cfg.highlight_color.b,
                cfg.highlight_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label("Highlight");
                if ui.color_edit_button_srgba_unmultiplied(&mut hcol).changed() {
                    cfg.highlight_color = Srgba::new(hcol[0], hcol[1], hcol[2], hcol[3]);
                }
            });

            let mut pcol = [
                cfg.splitting_plane_color.r,
                cfg.splitting_plane_color.g,
                cfg.splitting_plane_color.b,
                cfg.splitting_plane_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label("Splitting plane");
                if ui.color_edit_button_srgba_unmultiplied(&mut pcol).changed() {
                    cfg.splitting_plane_color = Srgba::new(pcol[0], pcol[1], pcol[2], pcol[3]);
                }
            });

            let mut mcol = [
                cfg.marker_color.r,
                cfg.marker_color.g,
                cfg.marker_color.b,
                cfg.marker_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label("Marker");
                if ui.color_edit_button_srgba_unmultiplied(&mut mcol).changed() {
                    cfg.marker_color = Srgba::new(mcol[0], mcol[1], mcol[2], mcol[3]);
                }
            });

            let mut dircol = [
                cfg.arrow_color.r,
                cfg.arrow_color.g,
                cfg.arrow_color.b,
                cfg.arrow_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label("Arrow");
                if ui
                    .color_edit_button_srgba_unmultiplied(&mut dircol)
                    .changed()
                {
                    cfg.arrow_color = Srgba::new(dircol[0], dircol[1], dircol[2], dircol[3]);
                }
            });

            ui.add(
                egui::Slider::new(&mut cfg.ambient_light_intensity, 0.0..=5.0)
                    .text("Ambient intensity"),
            );
            let mut acol = [
                cfg.ambient_light_color.r,
                cfg.ambient_light_color.g,
                cfg.ambient_light_color.b,
                cfg.ambient_light_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label("Ambient color");
                if ui.color_edit_button_srgba_unmultiplied(&mut acol).changed() {
                    cfg.ambient_light_color = Srgba::new(acol[0], acol[1], acol[2], acol[3]);
                }
            });

            ui.separator();
            ui.heading("BSP Tree");
            ui.add(egui::Slider::new(&mut cfg.bsp_tree_text_size, 8.0..=32.0).text("Text size"));
            ui.horizontal(|ui| {
                ui.label("Path color");
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut cfg.bsp_tree_path_color,
                    egui::color_picker::Alpha::OnlyBlend,
                );
            });
            ui.horizontal(|ui| {
                ui.label("Selected color");
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut cfg.bsp_tree_selected_color,
                    egui::color_picker::Alpha::OnlyBlend,
                );
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
                .add(egui::Slider::new(&mut cam.speed, 0.1..=20.0).text("Speed"))
                .changed()
            {
                cfg.camera_speed = cam.speed;
            }
            if ui
                .add(egui::Slider::new(&mut cam.look_speed, 0.1..=10.0).text("Look speed"))
                .changed()
            {
                cfg.look_speed = cam.look_speed;
            }
            ui.add(egui::Slider::new(&mut cfg.default_fov_deg, 30.0..=120.0).text("FOV deg"));
            ui.add(egui::Slider::new(&mut cfg.near_plane, 0.01..=10.0).text("Near plane"));
            ui.add(egui::Slider::new(&mut cfg.far_plane, 10.0..=5000.0).text("Far plane"));

            ui.separator();
            ui.heading("Spectator");
            ui.checkbox(show_spectator_marker, "Show spectator position and direction");

            ui.horizontal(|ui| {
                ui.label("Spectator pos");
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut spectator_state.pos.x))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut spectator_state.pos.y))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut spectator_state.pos.z))
                    .changed();
                if changed {
                    cfg.default_spectator_pos = spectator_state.pos;
                    if mode == crate::camera::CamMode::Spectator {
                        cam.pos = spectator_state.pos;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Spectator angle");
                let mut yaw_deg = spectator_state.yaw.to_degrees();
                let mut pitch_deg = spectator_state.pitch.to_degrees();
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut yaw_deg).suffix("°"))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut pitch_deg).suffix("°"))
                    .changed();
                if changed {
                    spectator_state.yaw = yaw_deg.to_radians();
                    spectator_state.pitch = pitch_deg.to_radians();
                    cfg.default_spectator_yaw = spectator_state.yaw;
                    cfg.default_spectator_pitch = spectator_state.pitch;
                    if mode == crate::camera::CamMode::Spectator {
                        cam.yaw = spectator_state.yaw;
                        cam.pitch = spectator_state.pitch;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Third person pos");
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut third_person_state.pos.x))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut third_person_state.pos.y))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut third_person_state.pos.z))
                    .changed();
                if changed {
                    cfg.default_third_person_pos = third_person_state.pos;
                    if mode == crate::camera::CamMode::ThirdPerson {
                        cam.pos = third_person_state.pos;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Third person angle");
                let mut yaw_deg = third_person_state.yaw.to_degrees();
                let mut pitch_deg = third_person_state.pitch.to_degrees();
                let mut changed = false;
                changed |= ui
                    .add(egui::DragValue::new(&mut yaw_deg).suffix("°"))
                    .changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut pitch_deg).suffix("°"))
                    .changed();
                if changed {
                    third_person_state.yaw = yaw_deg.to_radians();
                    third_person_state.pitch = pitch_deg.to_radians();
                    cfg.default_third_person_yaw = third_person_state.yaw;
                    cfg.default_third_person_pitch = third_person_state.pitch;
                    if mode == crate::camera::CamMode::ThirdPerson {
                        cam.yaw = third_person_state.yaw;
                        cam.pitch = third_person_state.pitch;
                    }
                }
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
