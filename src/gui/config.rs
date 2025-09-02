//! Runtime configuration window allowing tweaks to colors and camera settings.

use crate::config::{Config, CONFIG};
use three_d::Srgba;

use crate::lang::{tr, Language};

pub fn draw_config_window(
    ctx: &egui::Context,
    cam: &mut crate::camera::FreeCamera,
    spectator_state: &mut crate::camera::CameraState,
    third_person_state: &mut crate::camera::CameraState,
    mode: crate::camera::CamMode,
    show_spectator_marker: &mut bool,
    open: &mut bool,
) {
    let lang = { CONFIG.read().unwrap().language };
    egui::Window::new(tr(lang, "Config", "Nastavení"))
        .open(open)
        .vscroll(true)
        .show(ctx, |ui| {
            let mut cfg = CONFIG.write().unwrap();
            let lang = cfg.language;

            ui.horizontal(|ui| {
                ui.label(tr(lang, "Language", "Jazyk"));
                egui::ComboBox::from_id_source("language_select")
                    .selected_text(match cfg.language {
                        Language::English => "English",
                        Language::Czech => "Čeština",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut cfg.language, Language::English, "English");
                        ui.selectable_value(&mut cfg.language, Language::Czech, "Čeština");
                    });
            });

            if ui
                .button(tr(lang, "Load default", "Načíst výchozí"))
                .clicked()
            {
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

            ui.heading(tr(lang, "Colors & Lighting", "Barvy a osvětlení"));
            ui.horizontal(|ui| {
                ui.label(tr(lang, "Background", "Pozadí"));
                ui.color_edit_button_rgb(&mut cfg.bg_color);
            });

            let mut color = [
                cfg.model_color.r,
                cfg.model_color.g,
                cfg.model_color.b,
                cfg.model_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label(tr(lang, "Model", "Model"));
                if ui
                    .color_edit_button_srgba_unmultiplied(&mut color)
                    .changed()
                {
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
                ui.label(tr(lang, "Highlight", "Zvýraznění"));
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
                ui.label(tr(lang, "Splitting plane", "Dělící rovina"));
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
                ui.label(tr(lang, "Marker", "Značka"));
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
                ui.label(tr(lang, "Arrow", "Šipka"));
                if ui
                    .color_edit_button_srgba_unmultiplied(&mut dircol)
                    .changed()
                {
                    cfg.arrow_color = Srgba::new(dircol[0], dircol[1], dircol[2], dircol[3]);
                }
            });

            ui.add(
                egui::Slider::new(&mut cfg.ambient_light_intensity, 0.0..=5.0).text(tr(
                    lang,
                    "Ambient intensity",
                    "Intenzita ambientního osvětlení",
                )),
            );
            let mut acol = [
                cfg.ambient_light_color.r,
                cfg.ambient_light_color.g,
                cfg.ambient_light_color.b,
                cfg.ambient_light_color.a,
            ];
            ui.horizontal(|ui| {
                ui.label(tr(lang, "Ambient color", "Barva ambientního světla"));
                if ui.color_edit_button_srgba_unmultiplied(&mut acol).changed() {
                    cfg.ambient_light_color = Srgba::new(acol[0], acol[1], acol[2], acol[3]);
                }
            });

            ui.separator();
            ui.heading(tr(lang, "BSP Tree", "BSP Strom"));
            ui.add(
                egui::Slider::new(&mut cfg.bsp_tree_text_size, 8.0..=32.0).text(tr(
                    lang,
                    "Text size",
                    "Velikost textu",
                )),
            );
            ui.horizontal(|ui| {
                ui.label(tr(lang, "Path color", "Barva cesty"));
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut cfg.bsp_tree_path_color,
                    egui::color_picker::Alpha::OnlyBlend,
                );
            });
            ui.horizontal(|ui| {
                ui.label(tr(lang, "Selected color", "Barva vybraného"));
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut cfg.bsp_tree_selected_color,
                    egui::color_picker::Alpha::OnlyBlend,
                );
            });

            ui.separator();
            ui.heading(tr(lang, "BSP Limits", "Limity BSP"));
            ui.add(egui::Slider::new(&mut cfg.max_bsp_depth, 1..=64).text(tr(
                lang,
                "Max depth",
                "Maximální hloubka",
            )));
            let max_depth = cfg.max_bsp_depth;
            ui.add(
                egui::Slider::new(&mut cfg.default_branch_limit, 1..=max_depth).text(tr(
                    lang,
                    "Default branch limit",
                    "Výchozí limit větví",
                )),
            );
            ui.add(
                egui::Slider::new(&mut cfg.min_triangles_per_leaf, 1..=100).text(tr(
                    lang,
                    "Min triangles per leaf",
                    "Min trojúhelníků na list",
                )),
            );

            ui.separator();
            ui.heading(tr(lang, "Camera", "Kamera"));
            if ui
                .add(
                    egui::Slider::new(&mut cam.speed, 0.1..=20.0)
                        .text(tr(lang, "Speed", "Rychlost")),
                )
                .changed()
            {
                cfg.camera_speed = cam.speed;
            }
            if ui
                .add(egui::Slider::new(&mut cam.look_speed, 0.1..=10.0).text(tr(
                    lang,
                    "Look speed",
                    "Rychlost otáčení",
                )))
                .changed()
            {
                cfg.look_speed = cam.look_speed;
            }
            ui.add(
                egui::Slider::new(&mut cfg.default_fov_deg, 30.0..=120.0).text(tr(
                    lang,
                    "FOV deg",
                    "FOV stupně",
                )),
            );
            ui.add(egui::Slider::new(&mut cfg.near_plane, 0.01..=10.0).text(tr(
                lang,
                "Near plane",
                "Blízká rovina",
            )));
            ui.add(
                egui::Slider::new(&mut cfg.far_plane, 10.0..=5000.0).text(tr(
                    lang,
                    "Far plane",
                    "Vzdálená rovina",
                )),
            );

            ui.separator();
            ui.heading(tr(lang, "Spectator", "Divák"));
            ui.checkbox(
                show_spectator_marker,
                tr(
                    lang,
                    "Show spectator position and direction",
                    "Zobrazit pozici a směr diváka",
                ),
            );

            if ui
                .button(tr(
                    lang,
                    "Swap spectator and third person",
                    "Prohodit diváka a třetí osobu",
                ))
                .clicked()
            {
                std::mem::swap(spectator_state, third_person_state);
                if mode == crate::camera::CamMode::Spectator {
                    cam.pos = spectator_state.pos;
                    cam.yaw = spectator_state.yaw;
                    cam.pitch = spectator_state.pitch;
                } else if mode == crate::camera::CamMode::ThirdPerson {
                    cam.pos = third_person_state.pos;
                    cam.yaw = third_person_state.yaw;
                    cam.pitch = third_person_state.pitch;
                }
                cfg.default_spectator_pos = spectator_state.pos;
                cfg.default_spectator_yaw = spectator_state.yaw;
                cfg.default_spectator_pitch = spectator_state.pitch;
                cfg.default_third_person_pos = third_person_state.pos;
                cfg.default_third_person_yaw = third_person_state.yaw;
                cfg.default_third_person_pitch = third_person_state.pitch;
            }

            ui.horizontal(|ui| {
                ui.label(tr(lang, "Spectator pos", "Pozice diváka"));
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
                ui.label(tr(lang, "Spectator angle", "Úhel diváka"));
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
                ui.label(tr(lang, "Third person pos", "Pozice třetí osoby"));
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
                ui.label(tr(lang, "Third person angle", "Úhel třetí osoby"));
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
                egui::Slider::new(&mut cfg.camera_marker_scale, 0.01..=1.0).text(tr(
                    lang,
                    "Marker scale",
                    "Měřítko značky",
                )),
            );
            ui.add(
                egui::Slider::new(&mut cfg.direction_ray_thickness, 0.01..=1.0).text(tr(
                    lang,
                    "Ray thickness",
                    "Tloušťka paprsku",
                )),
            );
            ui.add(
                egui::Slider::new(&mut cfg.direction_ray_length, 0.1..=10.0).text(tr(
                    lang,
                    "Ray length",
                    "Délka paprsku",
                )),
            );
        });
}
