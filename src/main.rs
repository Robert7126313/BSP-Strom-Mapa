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
// gltf           = "0.14"
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
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};
use three_d::*;

// před funkci main přidáme enum pro sledování stavu kláves
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum KeyState {
    Pressed,
    Released,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[derive(Hash)]
enum KeyCode {
    W,
    A,
    S,
    D,
    Up,
    Down,
    Left,
    Right,
    Space,
    C,
    PageUp,
    PageDown,
    F,
}

impl KeyCode {
    fn from_event(event: &Event) -> Option<Self> {
        use KeyCode::*;
        match event {
            Event::KeyPress { kind, .. } | Event::KeyRelease { kind, .. } => match kind {
                Key::W => Some(W),
                Key::A => Some(A),
                Key::S => Some(S),
                Key::D => Some(D),
                Key::ArrowUp => Some(Up),
                Key::ArrowDown => Some(Down),
                Key::ArrowLeft => Some(Left),
                Key::ArrowRight => Some(Right),
                Key::Space => Some(Space),
                Key::C => Some(C),
                Key::PageUp => Some(PageUp),
                Key::PageDown => Some(PageDown),
                Key::F => Some(F),
                _ => None,
            },
            _ => None,
        }
    }
}

// Struktura pro pohybové konstanty
struct MovementSettings {
    move_speed: f32,
    tilt_speed: f32,
    rotation_speed: f32,
}

impl Default for MovementSettings {
    fn default() -> Self {
        Self {
            move_speed: 5.0,
            tilt_speed: 1.0,
            rotation_speed: 1.0,
        }
    }
}

// Přidáme strukturu KeyState map pro sledování stavu kláves
struct InputManager {
    key_states: HashMap<KeyCode, KeyState>,
    movement_settings: MovementSettings,
}

impl Default for InputManager {
    fn default() -> Self {
        Self {
            key_states: HashMap::new(),
            movement_settings: MovementSettings::default(),
        }
    }
}

impl InputManager {
    fn new() -> Self {
        Self::default()
    }

    fn update_key_states(&mut self, events: &[Event]) {
        for event in events {
            if let Some(key_code) = KeyCode::from_event(event) {
                match event {
                    Event::KeyPress { .. } => {
                        self.key_states.insert(key_code, KeyState::Pressed);
                    }
                    Event::KeyRelease { .. } => {
                        self.key_states.insert(key_code, KeyState::Released);
                    }
                    _ => {}
                }
            }
        }
    }

    fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        self.key_states.get(&key_code).map_or(false, |state| *state == KeyState::Pressed)
    }

    // Získá vektor pohybu na základě aktuálního stavu kláves
    fn get_movement_vector(&self) -> Vector3<f32> {
        let mut move_vec = Vector3::new(0.0, 0.0, 0.0);

        // Dopředu/dozadu (W/S nebo šipky nahoru/dolů)
        if self.is_key_pressed(KeyCode::W) || self.is_key_pressed(KeyCode::Up) {
            move_vec.z += 1.0;
        }
        if self.is_key_pressed(KeyCode::S) || self.is_key_pressed(KeyCode::Down) {
            move_vec.z -= 1.0;
        }

        // Doleva/doprava (A/D)
        if self.is_key_pressed(KeyCode::A) {
            move_vec.x -= 1.0;
        }
        if self.is_key_pressed(KeyCode::D) {
            move_vec.x += 1.0;
        }

        // Nahoru/dolů (Space/C)
        if self.is_key_pressed(KeyCode::Space) {
            move_vec.y += 1.0;
        }
        if self.is_key_pressed(KeyCode::C) {
            move_vec.y -= 1.0;
        }

        // Normalizujeme vektor, pokud má nenulovou délku
        if move_vec.magnitude2() > 0.0 {
            move_vec = move_vec.normalize();
        }

        move_vec
    }

    // Získá hodnotu naklonění (tilt) na základě stavu kláves
    fn get_tilt_value(&self) -> f32 {
        let mut tilt = 0.0;

        if self.is_key_pressed(KeyCode::Left) {
            tilt -= 1.0;
        }
        if self.is_key_pressed(KeyCode::Right) {
            tilt += 1.0;
        }

        tilt
    }
}

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
    look_speed: f32,
}

impl FreeCamera {
    fn new(pos: Vector3<f32>) -> Self {
        // kamera směřuje podél -Z, takže je model v popředí
        Self { 
            pos, 
            yaw: -FRAC_PI_2, 
            pitch: 0.0, 
            speed: 4.0,
            look_speed: 2.0,
        }
    }

    fn dir(&self) -> Vector3<f32> {
        Vector3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos()).normalize()
    }

    fn right(&self) -> Vector3<f32> {
        self.dir().cross(Vector3::unit_y()).normalize()
    }

    fn update_smooth(&mut self, input_manager: &InputManager, dt: f32) {
        // Získání vektoru pohybu z InputManageru
        let raw_move_vec = input_manager.get_movement_vector();
        let tilt_value = input_manager.get_tilt_value();

        // Převedení abstraktního pohybového vektoru na reálný vektor v prostoru
        let mut v = Vector3::new(0.0, 0.0, 0.0);

        // Zpracování pohybu vpřed/vzad (Z složka vstupního vektoru)
        if raw_move_vec.z != 0.0 {
            v += self.dir() * raw_move_vec.z;
        }

        // Zpracování pohybu vlevo/vpravo (X složka vstupního vektoru)
        if raw_move_vec.x != 0.0 {
            v += self.right() * raw_move_vec.x;
        }

        // Zpracování pohybu nahoru/dolů (Y složka vstupního vektoru přímo na Y osu kamery)
        if raw_move_vec.y != 0.0 {
            v += Vector3::unit_y() * raw_move_vec.y;
        }

        // Aplikace pohybu s rychlostí a dt (pro nezávislost na snímkové frekvenci)
        if v.magnitude2() > 0.0 {
            self.pos += v * self.speed * dt;
        }

        // Zpracování naklonění hlavy (hodnota tilt_value)
        // Naklonění ovlivňuje yaw (otočení doleva/doprava)
        if tilt_value != 0.0 {
            self.yaw += tilt_value * self.look_speed * dt;
        }

        // PageUp/PageDown pro změnu rychlosti - tyto jsou zpracovány ve staré metodě update
    }

    fn update(&mut self, events: &[Event], dt: f32, _viewport: Viewport) {
        // rychlost PageUp/PageDown
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::PageUp, .. })) { 
            self.speed *= 1.2; 
        }
        if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::PageDown, .. })) { 
            self.speed /= 1.2; 
        }

        // Šipky pro rozhlížení kamery (look around)
        let held = |k: Key| events.iter().any(|e| matches!(e, Event::KeyPress { kind, .. } if *kind == k));
        
        let look_speed = self.look_speed * dt; // rychlost rotace šipkami
        
        // Arrow keys pro rozhlížení
        if held(Key::ArrowLeft) { self.yaw += look_speed; }
        if held(Key::ArrowRight) { self.yaw -= look_speed; }
        if held(Key::ArrowUp) { self.pitch = (self.pitch + look_speed).clamp(-1.5, 1.5); }
        if held(Key::ArrowDown) { self.pitch = (self.pitch - look_speed).clamp(-1.5, 1.5); }

        // pohyb klávesnicí - pouze WASD
        let mut v = Vector3::new(0.0, 0.0, 0.0);
        
        // WASD pro pohyb
        if held(Key::W) { v += self.dir(); }
        if held(Key::S) { v -= self.dir(); }
        if held(Key::A) { v -= self.right(); }
        if held(Key::D) { v += self.right(); }
        
        // Nahoru/dolů
        if held(Key::Space) { v += Vector3::unit_y(); }
        if held(Key::C) { v -= Vector3::unit_y(); } // "C" = dolů
        
        if v.magnitude2() > 0.0 { 
            self.pos += v.normalize() * self.speed * dt; 
        }
    }

    fn cam(&self, vp: Viewport) -> Camera {
        Camera::new_perspective(vp, self.pos, self.pos + self.dir(), Vector3::unit_y(), Deg(60.0), 0.1, 1000.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CamMode { Spectator, ThirdPerson }

// helper: načte CpuMesh z .glb/.gltf pomocí gltf crate
fn load_cpu_mesh(path: &Path) -> (CpuMesh, String) {
    println!("Pokouším se načíst: {}", path.display());

    if !path.exists() {
        println!("Soubor neexistuje: {}", path.display());
        return (CpuMesh::sphere(32), format!("Soubor neexistuje: {}", path.display()));
    }

    match std::fs::metadata(path) {
        Ok(metadata) => {
            println!("Velikost souboru: {} bytes", metadata.len());
            if metadata.len() == 0 {
                return (CpuMesh::sphere(32), "Soubor je prázdný".to_string());
            }
            if metadata.len() > 100_000_000 { // 100MB limit
                return (CpuMesh::sphere(32), "Soubor je příliš velký (>100MB)".to_string());
            }
        },
        Err(e) => {
            println!("Nelze přečíst metadata souboru: {}", e);
            return (CpuMesh::sphere(32), format!("Chyba metadata: {}", e));
        }
    }

    // Pokus načtení pomocí gltf crate
    match load_gltf_with_gltf_crate(path) {
        Ok(mesh) => {
            println!("✓ GLTF úspěšně načten pomocí gltf crate");
            return (mesh, "GLTF soubor úspěšně načten".to_string());
        },
        Err(e) => {
            println!("Chyba při načítání pomocí gltf crate: {}", e);
            return (CpuMesh::sphere(32), format!("Nepodařilo se načíst GLTF: {}", e));
        }
    }
}

fn load_gltf_with_gltf_crate(path: &Path) -> Result<CpuMesh> {
    println!("Načítám GLTF pomocí gltf crate...");
    
    let (document, buffers, _images) = gltf::import(path)?;
    
    println!("GLTF dokument načten:");
    println!("- Scény: {}", document.scenes().count());
    println!("- Uzly: {}", document.nodes().count());  
    println!("- Meshe: {}", document.meshes().count());
    println!("- Materiály: {}", document.materials().count());

    let mut all_positions = Vec::new();
    let mut all_indices = Vec::new();
    let mut vertex_offset = 0u32;

    // Projdi všechny meshe ve scéně
    for scene in document.scenes() {
        println!("Zpracovávám scénu: {:?}", scene.name());
        
        for node in scene.nodes() {
            process_node(&node, &buffers, &mut all_positions, &mut all_indices, &mut vertex_offset, cgmath::Matrix4::identity())?;
        }
    }

    if all_positions.is_empty() {
        anyhow::bail!("Žádné pozice nenalezeny v GLTF souboru");
    }

    println!("Celkem načteno {} vrcholů a {} indexů", all_positions.len(), all_indices.len());

    Ok(CpuMesh {
        positions: Positions::F32(all_positions),
        indices: if all_indices.is_empty() { 
            Indices::None 
        } else { 
            Indices::U32(all_indices) 
        },
        ..Default::default()
    })
}

fn process_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    all_positions: &mut Vec<Vec3>,
    all_indices: &mut Vec<u32>,
    vertex_offset: &mut u32,
    parent_transform: cgmath::Matrix4<f32>
) -> Result<()> {
    // Získej transformaci uzlu
    let transform_matrix = cgmath::Matrix4::from(node.transform().matrix());
    let current_transform = parent_transform * transform_matrix;

    println!("Zpracovávám uzel: {:?}", node.name());

    // Zpracuj mesh pokud existuje
    if let Some(mesh) = node.mesh() {
        println!("Zpracovávám mesh: {:?} s {} primitivy", mesh.name(), mesh.primitives().count());
        
        for primitive in mesh.primitives() {
            process_primitive(&primitive, buffers, all_positions, all_indices, vertex_offset, current_transform)?;
        }
    }

    // Rekurzivně zpracuj potomky
    for child in node.children() {
        process_node(&child, buffers, all_positions, all_indices, vertex_offset, current_transform)?;
    }

    Ok(())
}

fn process_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    all_positions: &mut Vec<Vec3>,
    all_indices: &mut Vec<u32>,
    vertex_offset: &mut u32,
    transform: cgmath::Matrix4<f32>
) -> Result<()> {
    println!("Zpracovávám primitiv s modem: {:?}", primitive.mode());

    // Pouze trojúhelníky
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        println!("Přeskakuji primitiv - není trojúhelníkový");
        return Ok(());
    }

    // Získej pozice vrcholů
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    
    if let Some(positions) = reader.read_positions() {
        let start_vertex_count = all_positions.len();
        
        // Přidej pozice s transformací
        for position in positions {
            let pos = cgmath::Vector4::new(position[0], position[1], position[2], 1.0);
            let transformed = transform * pos;
            all_positions.push(Vec3::new(transformed.x, transformed.y, transformed.z));
        }
        
        let vertex_count = all_positions.len() - start_vertex_count;
        println!("Přidáno {} vrcholů", vertex_count);

        // Získej indexy
        if let Some(indices) = reader.read_indices() {
            match indices {
                gltf::mesh::util::ReadIndices::U8(iter) => {
                    for idx in iter {
                        all_indices.push(idx as u32 + *vertex_offset);
                    }
                },
                gltf::mesh::util::ReadIndices::U16(iter) => {
                    for idx in iter {
                        all_indices.push(idx as u32 + *vertex_offset);
                    }
                },
                gltf::mesh::util::ReadIndices::U32(iter) => {
                    for idx in iter {
                        all_indices.push(idx + *vertex_offset);
                    }
                }
            }
            println!("Přidáno {} indexů", all_indices.len());
        } else {
            // Bez indexů - vytvoř sekvenčn��
            for i in (0..vertex_count).step_by(3) {
                if i + 2 < vertex_count {
                    all_indices.push(*vertex_offset + i as u32);
                    all_indices.push(*vertex_offset + i as u32 + 1);
                    all_indices.push(*vertex_offset + i as u32 + 2);
                }
            }
            println!("Vytvořeno {} sekvenčních indexů", (vertex_count / 3) * 3);
        }

        *vertex_offset += vertex_count as u32;
    } else {
        println!("Primitiv nemá pozice vrcholů");
    }

    Ok(())
}

// ---------------- Main --------------------------------------------------- //

fn main() -> Result<()> {
    // okno + GL
    let window = Window::new(WindowSettings { 
        title: "BSP Viewer (three‑d 0.18)".into(), 
        ..Default::default() 
    })?;
    let context = window.gl();
    let mut gui = GUI::new(&context);

    // stavová prom��nná: název aktuálního souboru a úspěšnost načtení
    let initial_path = Path::new("assets/model.glb");
    let (mut cpu_mesh, mut load_status) = load_cpu_mesh(initial_path);

    let mut loaded_file_name = if initial_path.exists() {
        initial_path.file_name().unwrap().to_string_lossy().into_owned()
    } else {
        "embedded sphere".to_string()
    };

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
    
    // Glow efekty pro pozice kamer
    let glow_mesh = CpuMesh::sphere(16);
    
    // Materiály pro glow efekty
    let spectator_glow_material = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(0, 255, 100, 200), // Zelená pro spectator
        ..Default::default()
    });
    
    let third_person_glow_material = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(255, 100, 0, 200), // Oranžová pro third person
        ..Default::default()
    });

    let mut spectator_glow = Gm::new(Mesh::new(&context, &glow_mesh), spectator_glow_material);
    let mut third_person_glow = Gm::new(Mesh::new(&context, &glow_mesh), third_person_glow_material);
    
    let light = AmbientLight::new(&context, 1.0, Srgba::WHITE); // Zvýšit intenzitu světla

    // před inicializací kamery přidáme mutable proměnné pro pozice obou režimů
    let mut spectator_pos    = Vector3::new(0.0, 2.0, 8.0);
    let mut third_person_pos = Vector3::new(5.0, 2.0, 8.0); // Jiná pozice pro lepší vizualizaci
    let mut cam = FreeCamera::new(spectator_pos);
    let mut mode = CamMode::Spectator;

    // Nastavení pozic glow efektů
    spectator_glow.set_transformation(Mat4::from_translation(vec3(
        spectator_pos.x, spectator_pos.y, spectator_pos.z
    )) * Mat4::from_scale(0.2)); // Malé koule
    
    third_person_glow.set_transformation(Mat4::from_translation(vec3(
        third_person_pos.x, third_person_pos.y, third_person_pos.z
    )) * Mat4::from_scale(0.2));

    // Inicializace InputManageru pro plynulé ovládání s více klávesami
    let mut input_manager = InputManager::new();

    window.render_loop(move |frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &frame_input.events;

        // Aktualizuj stav kláves v InputManageru
        input_manager.update_key_states(events);

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
                    ui.label("W/A/S/D - Pohyb (dopředu/doleva/dozadu/doprava)");
                    ui.label("↑/↓/←/→ - Rozhlížení (nahoru/dolů/doleva/doprava)");
                    ui.label("Space - Pohyb nahoru");
                    ui.label("C - Pohyb dolů");
                    ui.label("PageUp/PageDown - Rychlost");
                    ui.label("F - Přepnout režim");
                } else {
                    ui.label("📷 Third Person Mode Controls:");
                    ui.label("W/A/S/D - Pohyb (dopředu/doleva/dozadu/doprava)");
                    ui.label("↑/↓/←/→ - Rozhlížení (nahoru/dolů/doleva/doprava)");
                    ui.label("Space - Pohyb nahoru");
                    ui.label("C - Pohyb dolů");
                    ui.label("PageUp/PageDown - Rychlost");
                    ui.label("F - Přepnout režim");
                }

                ui.separator();
                ui.heading("Ovládání klávesnice");
                ui.label("POHYB - WASD:");
                ui.label("• W - Dopředu");
                ui.label("• S - Dozadu");
                ui.label("• A - Doleva");
                ui.label("• D - Doprava");
                ui.label("• Space - Nahoru");
                ui.label("• C - Dolů");
                ui.separator();
                ui.label("ROZHLÍŽENÍ - Šipky:");
                ui.label("• ↑ - Díváš se nahoru");
                ui.label("• ↓ - Díváš se dolů");
                ui.label("• ← - Otočit hlavu doleva");
                ui.label("• → - Otočit hlavu doprava");
                ui.label(format!("Rychlost rozhlížení: {:.1}°/s", cam.look_speed * 180.0 / std::f32::consts::PI));

                ui.separator();
                ui.label(format!("Rychlost: {:.1}", cam.speed));
                
                ui.add(egui::Slider::new(&mut cam.look_speed, 0.5..=5.0)
                    .text("Rychlost šipek"));

                ui.label(format!("Pozice: ({:.1}, {:.1}, {:.1})", cam.pos.x, cam.pos.y, cam.pos.z));

                // Detailní info o meshu
                ui.separator();
                ui.heading("Mesh Info");
                ui.label(format!("Vrcholy: {}", cpu_mesh.positions.len()));
                match &cpu_mesh.indices {
                    Indices::U32(idx) => ui.label(format!("Indexy (U32): {}", idx.len())),
                    Indices::U16(idx) => ui.label(format!("Indexy (U16): {}", idx.len())),
                    Indices::None => ui.label("Indexy: žádné"),
                    &three_d::Indices::U8(_) => todo!(),
                };

                ui.separator();
                ui.heading("Ovládání Info");
                ui.label("Jednoduché klávesnicové ovládání:");
                ui.label("• WASD - pohyb v prostoru");
                ui.label("• Šipky - rozhlížení kamery");
                ui.label("• PageUp/PageDown - rychlost pohybu");
                ui.label("• F - přepnutí režimu kamery");
                ui.label(format!("Yaw: {:.3}, Pitch: {:.3}", cam.yaw, cam.pitch));

                ui.separator();
                // tlačítko pro výběr .glb souboru
                if ui.button("Vyber .glb soubor").clicked() {
                    if let Some(file) = FileDialog::new()
                        .add_filter("3D Models", &["glb", "gltf"])
                        .pick_file()
                    {
                        glb_path = Some(file.clone());
                        let (new_mesh, new_status) = load_cpu_mesh(&file);
                        cpu_mesh = new_mesh;
                        load_status = new_status;

                        model = Gm::new(Mesh::new(&context, &cpu_mesh), material.clone());
                        
                        // Přestavění BSP stromu pro nový model  
                        let triangles = cpu_mesh_to_triangles(&cpu_mesh);
                        bsp_root = build_bsp(triangles, 0);
                        total_stats = BspStats {
                            total_nodes: bsp_root.count_nodes(),
                            total_triangles: bsp_root.count_triangles(),
                            ..Default::default()
                        };
                        
                        // aktualizace stavu názvu
                        loaded_file_name = file.file_name().unwrap().to_string_lossy().into_owned();
                    }
                }

                ui.separator();
                // zobrazení názvu a stavu načtení
                ui.label(format!("Aktuální soubor: {}", loaded_file_name));
                ui.label(format!("Stav: {}", load_status));
            });
        });

        // --- ovládání ---
        // --- ovládání přepnutí režimu ---
        if input_manager.is_key_pressed(KeyCode::F) {
            // ulož aktuální pozici do příslušné proměnné
            if mode == CamMode::Spectator {
                spectator_pos = cam.pos;
                // Aktualizuj pozici spectator glow
                spectator_glow.set_transformation(Mat4::from_translation(vec3(
                    spectator_pos.x, spectator_pos.y, spectator_pos.z
                )) * Mat4::from_scale(0.2));
            } else {
                third_person_pos = cam.pos;
                // Aktualizuj pozici third person glow
                third_person_glow.set_transformation(Mat4::from_translation(vec3(
                    third_person_pos.x, third_person_pos.y, third_person_pos.z
                )) * Mat4::from_scale(0.2));
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

        // Zpracování změny rychlosti pomocí PageUp/PageDown přes InputManager
        if input_manager.is_key_pressed(KeyCode::PageUp) {
            cam.speed *= 1.2;
            println!("Rychlost zvýšena na: {:.1}", cam.speed);
        }
        if input_manager.is_key_pressed(KeyCode::PageDown) {
            cam.speed /= 1.2;
            println!("Rychlost snížena na: {:.1}", cam.speed);
        }
        
        // Aktualizace kamery pomocí nové metody pro hladký pohyb
        cam.update_smooth(&input_manager, dt);

        // --- vykreslení ---
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0)); // Tmavě šedé pozladí místo černého

        // Vykresli hlavní model a glow efekty
        let mut objects_to_render: Vec<&dyn Object> = vec![&model];

        // Přidej glow koule pouze pokud nejsou na stejné pozici jako aktuální kamera
        let current_distance_to_spectator = (cam.pos - spectator_pos).magnitude();
        let current_distance_to_third_person = (cam.pos - third_person_pos).magnitude();

        // Zobraz spectator glow pouze pokud nejsme v spectator režimu nebo jsme daleko
        if mode != CamMode::Spectator || current_distance_to_spectator > 1.0 {
            objects_to_render.push(&spectator_glow);
        }

        // Zobraz third person glow pouze pokud nejsme v third person režimu nebo jsme daleko
        if mode != CamMode::ThirdPerson || current_distance_to_third_person > 1.0 {
            objects_to_render.push(&third_person_glow);
        }

        screen.render(&cam.cam(frame_input.viewport), &objects_to_render, &[&light]);
        let _ = gui.render();
        FrameOutput::default()
    });

    Ok(())
}
