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
// DEMO FUNKCE: Neřeší načítání .glb (pro jednoduchost používá vestavěnou kouli).
// Pokud chceš importovat model.glb, přidej kód přes three-d‑asset::io::load
// a vytvoř Mesh::new(&context, &cpu_mesh).
// -----------------------------------------------------------------------------

use anyhow::Result;
use cgmath::{Deg, InnerSpace, Vector3};
use rfd::FileDialog;
use std::f32::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};
use three_d::*;

// ---------------- BSP Implementation -------------------------------------- //

#[derive(Clone, Debug)]
struct Triangle {
    a: Vector3<f32>,
    b: Vector3<f32>,
    c: Vector3<f32>,
}

#[derive(Clone, Debug)]
struct Plane {
    n: Vector3<f32>,  // normála
    d: f32,           // vzdálenost od počátku (ax+by+cz+d=0)
}

impl Plane {
    fn new(n: Vector3<f32>, point: Vector3<f32>) -> Self {
        let n = n.normalize();
        let d = -n.dot(point);
        Self { n, d }
    }

    fn side(&self, point: Vector3<f32>) -> f32 {
        self.n.dot(point) + self.d
    }

    fn classify(&self, point: Vector3<f32>) -> i32 {
        let dist = self.side(point);
        const EPSILON: f32 = 1e-6;
        if dist > EPSILON { 1 }        // front
        else if dist < -EPSILON { -1 } // back  
        else { 0 }                     // on plane
    }
}

struct BspNode {
    plane: Option<Plane>,
    front: Option<Box<BspNode>>,
    back: Option<Box<BspNode>>,
    triangles: Vec<Triangle>,
}

#[derive(Default)]
struct BspStats {
    nodes_visited: u32,
    triangles_rendered: u32,
    total_nodes: u32,
    total_triangles: u32,
}

impl BspNode {
    fn new_leaf(triangles: Vec<Triangle>) -> Self {
        Self {
            plane: None,
            front: None,
            back: None,
            triangles,
        }
    }

    fn new_node(plane: Plane, front: BspNode, back: BspNode) -> Self {
        Self {
            plane: Some(plane),
            front: Some(Box::new(front)),
            back: Some(Box::new(back)),
            triangles: Vec::new(),
        }
    }

    fn count_nodes(&self) -> u32 {
        1 + self.front.as_ref().map_or(0, |n| n.count_nodes()) + 
            self.back.as_ref().map_or(0, |n| n.count_nodes())
    }

    fn count_triangles(&self) -> u32 {
        self.triangles.len() as u32 + 
        self.front.as_ref().map_or(0, |n| n.count_triangles()) +
        self.back.as_ref().map_or(0, |n| n.count_triangles())
    }
}

fn plane_from_triangle(tri: &Triangle) -> Plane {
    let edge1 = tri.b - tri.a;
    let edge2 = tri.c - tri.a;
    let normal = edge1.cross(edge2).normalize();
    Plane::new(normal, tri.a)
}

fn triangle_center(tri: &Triangle) -> Vector3<f32> {
    (tri.a + tri.b + tri.c) / 3.0
}

fn build_bsp(triangles: Vec<Triangle>, depth: u32) -> BspNode {
    const MAX_DEPTH: u32 = 20;
    const MIN_TRIANGLES: usize = 50;

    if depth >= MAX_DEPTH || triangles.len() <= MIN_TRIANGLES {
        return BspNode::new_leaf(triangles);
    }

    if triangles.is_empty() {
        return BspNode::new_leaf(Vec::new());
    }

    // Výběr dělicí roviny - použijeme rovinu prvního trojúhelníku
    let splitting_plane = plane_from_triangle(&triangles[0]);

    let mut front_triangles = Vec::new();
    let mut back_triangles = Vec::new();

    // Klasifikace trojúhelníků podle střední pozice
    for triangle in triangles {
        let center = triangle_center(&triangle);
        let side = splitting_plane.classify(center);
        
        if side >= 0 {
            front_triangles.push(triangle);
        } else {
            back_triangles.push(triangle);
        }
    }

    // Rekurzivní stavba podstromů
    let front_node = build_bsp(front_triangles, depth + 1);
    let back_node = build_bsp(back_triangles, depth + 1);

    BspNode::new_node(splitting_plane, front_node, back_node)
}

fn traverse_bsp(
    node: &BspNode, 
    camera_pos: Vector3<f32>, 
    stats: &mut BspStats,
    visible_triangles: &mut Vec<Triangle>
) {
    stats.nodes_visited += 1;

    match &node.plane {
        None => {
            // List - přidej všechny trojúhelníky
            visible_triangles.extend(node.triangles.iter().cloned());
            stats.triangles_rendered += node.triangles.len() as u32;
        },
        Some(plane) => {
            // Vnitřní uzel - rozhodni o pořadí traversalu
            let camera_side = plane.side(camera_pos);
            
            let (near_node, far_node) = if camera_side > 0.0 {
                (&node.front, &node.back)
            } else {
                (&node.back, &node.front)
            };

            // Projdi nejdřív blízký uzel, pak vzdálený
            if let Some(near) = near_node {
                traverse_bsp(near, camera_pos, stats, visible_triangles);
            }
            if let Some(far) = far_node {
                traverse_bsp(far, camera_pos, stats, visible_triangles);
            }
        }
    }
}

fn cpu_mesh_to_triangles(cpu_mesh: &CpuMesh) -> Vec<Triangle> {
    let mut triangles = Vec::new();

    // Konverze z three-d Vec3 na cgmath Vector3
    let positions: Vec<Vector3<f32>> = cpu_mesh.positions.to_f32()
        .iter()
        .map(|pos| Vector3::new(pos.x, pos.y, pos.z))
        .collect();

    match &cpu_mesh.indices {
        Indices::U32(indices) => {
            for chunk in indices.chunks(3) {
                if chunk.len() == 3 {
                    let a = positions[chunk[0] as usize];
                    let b = positions[chunk[1] as usize];
                    let c = positions[chunk[2] as usize];
                    triangles.push(Triangle { a, b, c });
                }
            }
        },
        Indices::U16(indices) => {
            for chunk in indices.chunks(3) {
                if chunk.len() == 3 {
                    let a = positions[chunk[0] as usize];
                    let b = positions[chunk[1] as usize];
                    let c = positions[chunk[2] as usize];
                    triangles.push(Triangle { a, b, c });
                }
            }
        },
        _ => {}
    }

    triangles
}

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
    let mut cpu_mesh = load_cpu_mesh(initial_path);

    // Vytvoření BSP stromu
    let triangles = cpu_mesh_to_triangles(&cpu_mesh);
    let mut bsp_root = build_bsp(triangles, 0);
    let mut total_stats = BspStats {
        total_nodes: bsp_root.count_nodes(),
        total_triangles: bsp_root.count_triangles(),
        ..Default::default()
    };

    // stav pro vykreslovaný mesh
    let mut glb_path: Option<PathBuf> = None;
    let material = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(100, 150, 255, 255), // Modrá barva aby byl model viditelný
        ..Default::default()
    });
    let mut model = Gm::new(Mesh::new(&context, &cpu_mesh), material.clone());
    let light = AmbientLight::new(&context, 1.0, Srgba::WHITE); // Zvýšit intenzitu světla

    // před inicializací kamery přidáme mutable proměnné pro pozice obou režimů
    let mut spectator_pos    = Vector3::new(0.0, 2.0, 8.0);
    let mut third_person_pos = spectator_pos;
    let mut cam = FreeCamera::new(spectator_pos);
    let mut mode = CamMode::Spectator;

    window.render_loop(move |frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &frame_input.events;

        // BSP traversal pro aktuální pozici kamery
        let mut current_stats = BspStats {
            total_nodes: total_stats.total_nodes,
            total_triangles: total_stats.total_triangles,
            ..Default::default()
        };
        let mut visible_triangles = Vec::new();
        traverse_bsp(&bsp_root, cam.pos, &mut current_stats, &mut visible_triangles);

        // --- GUI ---
        gui.update(&mut frame_input.events.clone(), frame_input.accumulated_time, frame_input.viewport, frame_input.device_pixel_ratio, |ctx| {
            egui::SidePanel::left("tree").show(ctx, |ui| {
                ui.heading("BSP Strom");
                ui.label(format!("Režim: {:?}", mode));

                ui.separator();
                ui.heading("BSP Statistiky");
                ui.label(format!("Celkem uzlů: {}", current_stats.total_nodes));
                ui.label(format!("Celkem trojúhelníků: {}", current_stats.total_triangles));
                ui.label(format!("Navštíveno uzlů: {}", current_stats.nodes_visited));
                ui.label(format!("Vykresleno trojúhelníků: {}", current_stats.triangles_rendered));
                ui.label(format!("Procházka efektivita: {:.1}%", 
                    if current_stats.total_nodes > 0 {
                        (current_stats.nodes_visited as f32 / current_stats.total_nodes as f32) * 100.0
                    } else { 0.0 }));

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
                        
                        // Přestavění BSP stromu pro nový model  
                        let triangles = cpu_mesh_to_triangles(&cpu_mesh);
                        bsp_root = build_bsp(triangles, 0);
                        total_stats = BspStats {
                            total_nodes: bsp_root.count_nodes(),
                            total_triangles: bsp_root.count_triangles(),
                            ..Default::default()
                        };
                        
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
        // --- ovládání přepnutí režimu ---
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::F, .. })) {
            // ulož aktuální pozici do příslušné proměnné
            if mode == CamMode::Spectator {
                spectator_pos = cam.pos;
            } else {
                third_person_pos = cam.pos;
            }
            // přepni režim
            mode = if mode == CamMode::Spectator { CamMode::ThirdPerson } else { CamMode::Spectator };
            // obnov pozici nové kamery
            cam.pos = if mode == CamMode::Spectator {
                spectator_pos
            } else {
                third_person_pos
            };
        }
        cam.update(events, dt);

        // --- vykreslení ---
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0)); // Tmavě šedé pozladí místo černého
        screen.render(&cam.cam(frame_input.viewport), &[&model], &[&light]);
        let _ = gui.render();
        FrameOutput::default()
    });

    Ok(())
}
