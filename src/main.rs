// SPDX-License-Identifier: MIT
// -----------------------------------------------------------------------------
// BSP Viewer – minimální demo pro three‑d 0.18.x
// -----------------------------------------------------------------------------
// Cargo.toml
// -----------------------------------------------------------------------------
// [package]
// name    = "bsp_viewer"
// version = "0.4.0"
// edition = "2021"
//
// [dependencies]
// anyhow        = "1"
// cgmath        = "0.18"
// egui          = "0.29"
// rfd           = "0.11"
// three-d       = { version = "0.18", features = ["window", "egui-gui"] }
// three-d-asset = "0.9"
// -----------------------------------------------------------------------------
// build:  $ cargo run --release
// -----------------------------------------------------------------------------
// DEMO FUNKCE: Neřeší načítání .glb (pro jednoduchost používá vestavěnou kouli).
// Pokud chceš importovat model.glb, přidej kód přes three-d‑asset::io::load
// a vytvoř Mesh::new(&context, &cpu_mesh).
// -----------------------------------------------------------------------------

use anyhow::Result;
use cgmath::{Deg, InnerSpace, Vector3};
use rfd::FileDialog;
use std::f32::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};
use three_d::*;

// ---------------- BSP placeholder ---------------------------------------- //

type Triangle = [u32; 3];
#[derive(Clone)]
#[allow(dead_code)]
struct BspNode { id: usize, tris: Vec<Triangle> }
fn build_bsp(tris: Vec<Triangle>) -> BspNode { BspNode { id: 0, tris } }

// ---------------- Free‑fly kamera ---------------------------------------- //

#[derive(Clone)]
struct FreeCamera {
    pos: Vector3<f32>,
    yaw: f32,
    pitch: f32,
    speed: f32,
}
impl FreeCamera {
    fn new(pos: Vector3<f32>) -> Self {
        // kamera směřuje podél -Z, takže je model v popředí
        Self { pos, yaw: -FRAC_PI_2, pitch: 0.0, speed: 4.0 }
    }
    fn dir(&self) -> Vector3<f32> {
        Vector3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos()).normalize()
    }
    fn right(&self) -> Vector3<f32> { self.dir().cross(Vector3::unit_y()).normalize() }

    fn update(&mut self, events: &[Event], dt: f32) {
        // rychlost PageUp/PageDown
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::PageUp, .. })) { self.speed *= 1.2; }
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::PageDown, .. })) { self.speed /= 1.2; }
        // otáčení (drž LMB)
        if let Some(Event::MouseMotion { delta, .. }) = events.iter().find(|e| matches!(e, Event::MouseMotion { .. })) {
            if events.iter().any(|e| matches!(e, Event::MousePress { button: MouseButton::Left, .. })) {
                self.yaw   -= delta.0 as f32 * 0.002;
                self.pitch = (self.pitch - delta.1 as f32 * 0.002).clamp(-1.5, 1.5);
            }
        }
        // pohyb
        let held = |k: Key| events.iter().any(|e| matches!(e, Event::KeyPress { kind, .. } if *kind == k));
        let mut v = Vector3::new(0.0, 0.0, 0.0);
        if held(Key::W) { v += self.dir(); }
        if held(Key::S) { v -= self.dir(); }
        if held(Key::A) { v -= self.right(); }
        if held(Key::D) { v += self.right(); }
        if held(Key::Space) { v += Vector3::unit_y(); }
        if held(Key::C) { v -= Vector3::unit_y(); } // "C" = dolů
        if v.magnitude2() > 0.0 { self.pos += v.normalize() * self.speed * dt; }
    }
    fn cam(&self, vp: Viewport) -> Camera {
        Camera::new_perspective(vp, self.pos, self.pos + self.dir(), Vector3::unit_y(), Deg(60.0), 0.1, 1000.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CamMode { Spectator, ThirdPerson }

// helper: načte CpuMesh z .glb nebo vrátí kouli
fn load_cpu_mesh(path: &Path) -> CpuMesh {
    if path.exists() {
        if let Ok(mut assets) = three_d_asset::io::load(&[path]) {
            if let Some(k) = assets.keys().find(|k| k.to_string_lossy().ends_with(".glb")).cloned() {
                if let Ok(model) = assets.deserialize::<three_d_asset::Model>(&k) {
                    if let Some(geom) = model.geometries.first() {
                        if let three_d_asset::geometry::Geometry::Triangles(tri) = &geom.geometry {
                            return tri.clone().into();
                        }
                    }
                }
            }
        }
    }
    CpuMesh::sphere(32)
}

// ---------------- Main --------------------------------------------------- //

fn main() -> Result<()> {
    // okno + GL
    let window = Window::new(WindowSettings { title: "BSP Viewer (three‑d 0.18)".into(), ..Default::default() })?;
    let context = window.gl();
    let mut gui = GUI::new(&context);

    // stavová proměnná: název aktuálního souboru a úspěšnost načtení
    let initial_path = Path::new("assets/model.glb");
    let mut loaded_file_name = if initial_path.exists() {
        initial_path.file_name().unwrap().to_string_lossy().into_owned()
    } else {
        "embedded sphere".to_string()
    };
    let mut file_loaded = initial_path.exists();

    // Načtení CPU mesh (model.glb nebo vestavěná koule)
    let cpu_mesh = load_cpu_mesh(initial_path);

    // Extrakce indexů z CpuMesh
    let tris = match &cpu_mesh.indices {
        Indices::U32(indices) => indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        Indices::U16(indices) => indices.chunks(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect(),
        _ => Vec::new(),
    };

    // stav pro vykreslovaný mesh
    let mut glb_path: Option<PathBuf> = None;
    let mut cpu_mesh = load_cpu_mesh(Path::new("assets/model.glb"));
    let material = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(100, 150, 255, 255), // Modrá barva aby byl model viditelný
        ..Default::default()
    });
    let mut model = Gm::new(Mesh::new(&context, &cpu_mesh), material.clone());
    let light = AmbientLight::new(&context, 1.0, Srgba::WHITE); // Zvýšit intenzitu světla

    let _bsp_root = build_bsp(tris);
    let mut cam = FreeCamera::new(Vector3::new(0.0, 2.0, 8.0)); // Posunout kameru dál
    let mut mode = CamMode::Spectator;

    window.render_loop(move |frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &frame_input.events;

        // --- GUI ---
        gui.update(&mut frame_input.events.clone(), frame_input.accumulated_time, frame_input.viewport, frame_input.device_pixel_ratio, |ctx| {
            egui::SidePanel::left("tree").show(ctx, |ui| {
                ui.heading("BSP strom (placeholder)");
                ui.label(format!("Režim: {:?}", mode));

                ui.separator();
                ui.heading("Ovládání");

                if mode == CamMode::Spectator {
                    ui.label("🎮 Spectator Mode Controls:");
                    ui.label("W/S - Pohyb dopředu/dozadu");
                    ui.label("A/D - Pohyb doleva/doprava");
                    ui.label("Space - Pohyb nahoru");
                    ui.label("C - Pohyb dolů");
                    ui.label("LMB + Mouse - Rozhlížení");
                    ui.label("PageUp/PageDown - Rychlost");
                    ui.label("F - Přepnout režim");
                } else {
                    ui.label("📷 Third Person Mode Controls:");
                    ui.label("W/S - Pohyb dopředu/dozadu");
                    ui.label("A/D - Pohyb doleva/doprava");
                    ui.label("Space - Pohyb nahoru");
                    ui.label("C - Pohyb dolů");
                    ui.label("LMB + Mouse - Rozhlížení");
                    ui.label("PageUp/PageDown - Rychlost");
                    ui.label("F - Přepnout režim");
                }

                ui.separator();
                ui.label(format!("Rychlost: {:.1}", cam.speed));
                ui.label(format!("Pozice: ({:.1}, {:.1}, {:.1})", cam.pos.x, cam.pos.y, cam.pos.z));

                ui.separator();
                // tlačítko pro výběr .glb souboru
                if ui.button("Vyber .glb soubor").clicked() {
                    if let Some(file) = FileDialog::new().add_filter("GLB", &["glb"]).pick_file() {
                        glb_path = Some(file.clone());
                        cpu_mesh = load_cpu_mesh(&file);
                        model = Gm::new(Mesh::new(&context, &cpu_mesh), material.clone());
                        // aktualizace stavu názvu a úspěšnosti
                        loaded_file_name = file.file_name().unwrap().to_string_lossy().into_owned();
                        file_loaded = file.exists();
                    }
                }

                ui.separator();
                // zobrazení názvu a stavu načtení
                ui.label(format!("Aktuální soubor: {}", loaded_file_name));
                ui.label(if file_loaded {
                    "Soubor byl načten úspěšně"
                } else {
                    "Používám vestavěnou kouli"
                });
            });
        });

        // --- ovládání ---
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::F, .. })) {
            mode = if mode == CamMode::Spectator { CamMode::ThirdPerson } else { CamMode::Spectator };
        }
        cam.update(events, dt);

        // --- vykreslení ---
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0)); // Tmavě šedé pozadí místo černého
        screen.render(&cam.cam(frame_input.viewport), &[&model], &[&light]);
        let _ = gui.render();
        FrameOutput::default()
    });

    Ok(())
}
