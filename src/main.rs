// SPDX-License-Identifier: MIT
// -----------------------------------------------------------------------------
// BSP Viewer – minimální demo pro three‑d 0.18.x
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
use three_d::asset::CpuMesh;
use three_d::window::*;
use three_d::*;

// ---------------- BSP placeholder ---------------------------------------- //

type Triangle = [u32; 3];
#[derive(Clone)]
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
    fn new(pos: Vector3<f32>) -> Self { Self { pos, yaw: 0.0, pitch: 0.0, speed: 4.0 } }
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

// ---------------- Main --------------------------------------------------- //

fn main() -> Result<()> {
    // okno + GL
    let window = Window::new(WindowSettings { title: "BSP Viewer (three‑d 0.18)".into(), ..Default::default() })?;
    let context = window.gl();
    let mut gui = GUI::new(&context);

    // demo mesh = koule
    let cpu_mesh = CpuMesh::sphere(32);
    let tris = cpu_mesh.indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let mesh = Mesh::new(&context, &cpu_mesh);

    let _bsp_root = build_bsp(tris);
    let mut cam = FreeCamera::new(Vector3::new(0.0, 1.0, 5.0));
    let mut mode = CamMode::Spectator;

    window.render_loop(move |frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &frame_input.events;

        // --- GUI ---
        gui.update(&mut frame_input.events.clone(), frame_input.accumulated_time, frame_input.viewport, frame_input.device_pixel_ratio, |ctx| {
            egui::SidePanel::left("tree").show(ctx, |ui| {
                ui.heading("BSP strom (placeholder)");
                ui.label(format!("Režim: {:?}", mode));
            });
        });

        // --- ovládání ---
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::Tab, .. })) {
            mode = if mode == CamMode::Spectator { CamMode::ThirdPerson } else { CamMode::Spectator };
        }
        cam.update(events, dt);

        // --- vykreslení ---
        let mut screen = frame_input.screen();
        screen.clear(ClearState::default());
        screen.render(&cam.cam(frame_input.viewport), &[&mesh], &[]);
        gui.render();
        FrameOutput::default()
    });

    Ok(())
}
