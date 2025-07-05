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
// rayon          = "1.8"

// -----------------------------------------------------------------------------

use anyhow::Result;
use cgmath::{Deg, InnerSpace, Vector3};
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};
use three_d::*;
use rayon::prelude::*; // Add Rayon prelude for parallelization

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
    G,    // Přidána nová klávesa G pro přepnutí do ThirdPerson
    Home,  // Přidána nová klávesa Home
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
                Key::G => Some(G),  // Přiřazení klávesy G
                Key::Home => Some(Home),  // Přiřazení klávesy Home
                _ => None,
            },
            _ => None,
        }
    }
}

// Přidáme strukturu KeyState map pro sledování stavu kláves
struct InputManager {
    key_states: HashMap<KeyCode, KeyState>,
}

impl Default for InputManager {
    fn default() -> Self {
        Self {
            key_states: HashMap::new(),
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
    
    // fn is_key_released(&self, key_code: KeyCode) -> bool {
    //     self.key_states.get(&key_code).map_or(true, |state| *state == KeyState::Released)
    // }

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
        match dist {
            d if d > EPSILON => 1,     // front
            d if d < -EPSILON => -1,   // back
            _ => 0,                    // on plane
        }
    }
}

struct BspNode {
    id: usize,
    plane: Option<Plane>,
    front: Option<Box<BspNode>>,
    back: Option<Box<BspNode>>,
    triangles: Vec<Triangle>,
    bounds: BoundingBox,
}

#[derive(Default)]
struct BspStats {
    nodes_visited: u32,
    triangles_rendered: u32,
    total_nodes: u32,
    total_triangles: u32,
}

impl BspNode {
    fn new_leaf(triangles: Vec<Triangle>, id: usize) -> Self {
        Self {
            id,
            plane: None,
            front: None,
            back: None,
            triangles: triangles.clone(),
            bounds: BoundingBox::from_triangles(&triangles),
        }
    }

    fn new_node(plane: Plane, front: BspNode, back: BspNode, id: usize) -> Self {
        // Nejprve vytvoříme společný obalový objem, než přesuneme hodnoty do boxů
        let bounds = BoundingBox::encompass(&front.bounds, &back.bounds);

        Self {
            id,
            plane: Some(plane),
            front: Some(Box::new(front)),
            back: Some(Box::new(back)),
            triangles: Vec::new(),
            bounds,
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

// Upravená funkce build_bsp, která přiřazuje ID uzlům
fn build_bsp(triangles: &[Triangle], depth: u32, next_id: &mut usize) -> BspNode {
    const MAX_DEPTH: u32 = 20;
    const MIN_TRIANGLES: usize = 50;

    let my_id = *next_id;
    *next_id += 1;

    if depth >= MAX_DEPTH || triangles.len() <= MIN_TRIANGLES {
        return BspNode::new_leaf(triangles.to_vec(), my_id);
    }

    if triangles.is_empty() {
        return BspNode::new_leaf(Vec::new(), my_id);
    }

    // Výběr dělicí roviny - použijeme rovinu prvního trojúhelníku
    let splitting_plane = plane_from_triangle(&triangles[0]);

    // Paralelní klasifikace trojúhelníků pomocí Rayon
    let (front_triangles, back_triangles): (Vec<Triangle>, Vec<Triangle>) = triangles.par_iter()
        .cloned()
        .partition(|triangle| {
            let center = triangle_center(triangle);
            splitting_plane.classify(center) >= 0
        });

    // Rekurzivní stavba podstromů - můžeme paralelizovat, pokud jsme v horních úrovních stromu
    if depth < 5 {
        // Pro horní úrovně použijeme paralelní zpracování s předáním next_id
        let mut front_id = *next_id;
        *next_id += 1;
        let mut back_id = *next_id;
        *next_id += 1;
        
        let (front_node, back_node) = rayon::join(
            || build_bsp(&front_triangles, depth + 1, &mut front_id),
            || build_bsp(&back_triangles, depth + 1, &mut back_id)
        );
        
        // Aktualizujeme next_id na nejvyšší hodnotu z obou větví
        *next_id = front_id.max(back_id);
        
        BspNode::new_node(splitting_plane, front_node, back_node, my_id)
    } else {
        // Pro hlubší úrovně zůstaneme u sekvenčního zpracování
        let front_node = build_bsp(&front_triangles, depth + 1, next_id);
        let back_node = build_bsp(&back_triangles, depth + 1, next_id);
        BspNode::new_node(splitting_plane, front_node, back_node, my_id)
    }
}

// Funkce pro rekurzivní hledání uzlu podle ID
fn find_node(node: &BspNode, id: usize) -> Option<&BspNode> {
    if node.id == id {
        return Some(node);
    }
    if let Some(found) = node.front.as_deref().and_then(|f| find_node(f, id)) {
        return Some(found);
    }
    node.back.as_deref().and_then(|b| find_node(b, id))
}

// Funkce pro rekurzivní vykreslení stromu v UI a zpracování výběru uzlu
fn render_bsp_tree(ui: &mut egui::Ui, node: &BspNode, selected: &mut Option<usize>) {
    // build the label
    let is_leaf = node.plane.is_none();
    let triangle_count = node.triangles.len();
    let child_count = node.front.as_ref().map_or(0, |n| n.count_nodes())
                   + node.back.as_ref().map_or(0, |n| n.count_nodes());

    let is_selected = selected == &Some(node.id);
    let label = if is_leaf {
        if is_selected {
            format!("🔸 List {} ({} tris)", node.id, triangle_count)
        } else {
            format!("List {} ({} tris)", node.id, triangle_count)
        }
    } else {
        if is_selected {
            format!("🔸 Node {} ({} tris, {} children)", node.id, triangle_count, child_count)
        } else {
            format!("Node {} ({} tris, {} children)", node.id, triangle_count, child_count)
        }
    };

    // collapsible header
    let header = egui::CollapsingHeader::new(label)
        .id_source(node.id)
        .default_open(node.id == selected.unwrap_or(0)); // auto-open the selected node

    // draw the header
    let response = header.show(ui, |ui| {
        // small "select" button inside the collapsible content
        if ui.add(egui::SelectableLabel::new(selected == &Some(node.id), "▶ Select")).clicked() {
            *selected = Some(node.id);
        }

        // and recurse below *only if* this header is open
        if let Some(ref front) = node.front {
            ui.label("Front:");
            ui.indent("front", |ui| {
                render_bsp_tree(ui, front, selected);
            });
        }
        if let Some(ref back) = node.back {
            ui.label("Back:");
            ui.indent("back", |ui| {
                render_bsp_tree(ui, back, selected);
            });
        }
    });

    // if you want clicking the header itself to select:
    if response.header_response.clicked() {
        *selected = Some(node.id);
    }
}

// Funkce pro sběr všech trojúhelníků v podstromu
fn collect_triangles_in_subtree(node: &BspNode, triangles: &mut Vec<Triangle>) {
    triangles.extend(node.triangles.iter().cloned());
    if let Some(ref front) = node.front {
        collect_triangles_in_subtree(front, triangles);
    }
    if let Some(ref back) = node.back {
        collect_triangles_in_subtree(back, triangles);
    }
}

// Funkce pro vytvoření zvýrazněného meshe
fn create_highlight_mesh(triangles: &[Triangle], context: &Context) -> Gm<Mesh, ColorMaterial> {
    let positions: Vec<Vec3> = triangles.iter().flat_map(|tri| {
        vec![
            vec3(tri.a.x, tri.a.y, tri.a.z),
            vec3(tri.b.x, tri.b.y, tri.b.z),
            vec3(tri.c.x, tri.c.y, tri.c.z),
        ]
    }).collect();
    
    let indices: Vec<u32> = (0..triangles.len() as u32).flat_map(|i| {
        let base = i * 3;
        vec![base, base + 1, base + 2]
    }).collect();
    
    let cpu_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };
    
    let material = ColorMaterial::new_transparent(context, &CpuMaterial {
        albedo: Srgba::new(255, 50, 50, 150), // Červená s průhledností
        ..Default::default()
    });
    
    Gm::new(Mesh::new(context, &cpu_mesh), material)
}

// Funkce pro vytvoření meshe dělící roviny
fn create_plane_mesh(plane: &Plane, bounds: &BoundingBox, context: &Context) -> Gm<Mesh, ColorMaterial> {
    // Vypočítáme střed obalového objemu
    let center = (bounds.min + bounds.max) * 0.5;
    
    // Potřebujeme najít dva vektory kolmé na normálu roviny
    // Nejprve najdeme libovolný vektor kolmý na normálu
    let n = plane.n;
    let u = if n.x.abs() < n.y.abs() && n.x.abs() < n.z.abs() {
        Vector3::new(0.0, -n.z, n.y).normalize()
    } else if n.y.abs() < n.z.abs() {
        Vector3::new(-n.z, 0.0, n.x).normalize()
    } else {
        Vector3::new(-n.y, n.x, 0.0).normalize()
    };
    
    // Druhý vektor kolmý na normálu a první vektor
    let v = n.cross(u).normalize();
    
    // Velikost roviny - vycházíme z velikosti obalového objemu
    let extent = (bounds.max - bounds.min).magnitude() * 0.6;
    
    // Vytvoříme čtyři rohy roviny
    let corners = [
        center + (u + v) * extent,
        center + (u - v) * extent,
        center + (-u - v) * extent,
        center + (-u + v) * extent,
    ];
    
    // Vytvoříme pozice a indexy pro mesh
    let positions = vec![
        vec3(corners[0].x, corners[0].y, corners[0].z),
        vec3(corners[1].x, corners[1].y, corners[1].z),
        vec3(corners[2].x, corners[2].y, corners[2].z),
        vec3(corners[3].x, corners[3].y, corners[3].z),
    ];
    
    // Dva trojúhelníky pro čtyřúhelník
    let indices = vec![0, 1, 2, 2, 3, 0];
    
    let cpu_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };
    
    let material = ColorMaterial::new_transparent(context, &CpuMaterial {
        albedo: Srgba::new(200, 200, 50, 128), // Žlutá s průhledností
        ..Default::default()
    });
    
    Gm::new(Mesh::new(context, &cpu_mesh), material)
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

        // Zpracování naklonění hlavy nahoru/dolů (pitch)
        if input_manager.is_key_pressed(KeyCode::Up) {
            self.pitch = (self.pitch + self.look_speed * dt).clamp(-1.5, 1.5);
        }
        if input_manager.is_key_pressed(KeyCode::Down) {
            self.pitch = (self.pitch - self.look_speed * dt).clamp(-1.5, 1.5);
        }
    }

    // Starší verze update metody, která byla nahrazena update_smooth
    // fn update(&mut self, events: &[Event], dt: f32, _viewport: Viewport) {
    //     // rychlost PageUp/PageDown
    //     if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::PageUp, .. })) { 
    //         self.speed *= 1.2; 
    //     }
    //     if events.iter().any(|e| matches!(e, Event::KeyPress { kind: Key::PageDown, .. })) { 
    //         self.speed /= 1.2; 
    //     }

    //     // Šipky pro rozhlížení kamery (look around)
    //     let held = |k: Key| events.iter().any(|e| matches!(e, Event::KeyPress { kind, .. } if *kind == k));
        
    //     let look_speed = self.look_speed * dt; // rychlost rotace šipkami
        
    //     // Arrow keys pro rozhlížení
    //     if held(Key::ArrowLeft) { self.yaw += look_speed; }
    //     if held(Key::ArrowRight) { self.yaw -= look_speed; }
    //     if held(Key::ArrowUp) { self.pitch = (self.pitch + look_speed).clamp(-1.5, 1.5); }
    //     if held(Key::ArrowDown) { self.pitch = (self.pitch - look_speed).clamp(-1.5, 1.5); }

    //     // pohyb klávesnicí - pouze WASD
    //     let mut v = Vector3::new(0.0, 0.0, 0.0);
        
    //     // WASD pro pohyb
    //     if held(Key::W) { v += self.dir(); }
    //     if held(Key::S) { v -= self.dir(); }
    //     if held(Key::A) { v -= self.right(); }
    //     if held(Key::D) { v += self.right(); }
        
    //     // Nahoru/dolů
    //     if held(Key::Space) { v += Vector3::unit_y(); }
    //     if held(Key::C) { v -= Vector3::unit_y(); } // "C" = dolů
        
    //     if v.magnitude2() > 0.0 { 
    //         self.pos += v.normalize() * self.speed * dt; 
    //     }
    // }

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

// Funkce pro vytvoření meshe z viditelných trojúhelníků
fn create_visible_mesh(triangles: &[Triangle], context: &Context) -> Gm<Mesh, ColorMaterial> {
    // Paralelní zpracování pozic a indexů
    let triangles_count = triangles.len();
    
    // Paralelně vytvoříme pozice vrcholů
    let positions: Vec<Vec3> = triangles.par_iter().flat_map(|tri| {
        vec![
            vec3(tri.a.x, tri.a.y, tri.a.z),
            vec3(tri.b.x, tri.b.y, tri.b.z),
            vec3(tri.c.x, tri.c.y, tri.c.z)
        ]
    }).collect();
    
    // Paralelně vygenerujeme indexy
    let indices: Vec<u32> = (0..triangles_count as u32).into_par_iter().flat_map(|i| {
        let base_idx = i * 3;
        vec![base_idx, base_idx + 1, base_idx + 2]
    }).collect();

    // Vytvoření nového meshe
    let visible_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };
    
    // Vytvoření materiálu a modelu
    let material = ColorMaterial::new_opaque(context, &CpuMaterial {
        albedo: Srgba::new(100, 150, 255, 255),
        ..Default::default()
    });
    
    Gm::new(Mesh::new(context, &visible_mesh), material)
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

    // stavová proměnná: název aktuálního souboru a úspěšnost načtení
    let initial_path = Path::new("assets/model.glb");
    let (cpu_mesh, _load_status) = load_cpu_mesh(initial_path);

    let _loaded_file_name = if initial_path.exists() {
        initial_path.file_name().unwrap().to_string_lossy().into_owned()
    } else {
        "embedded sphere".to_string()
    };

    // Vytvoření triangles z CPU meshe
    let triangles = cpu_mesh_to_triangles(&cpu_mesh);
    
    // Inicializujeme počítadlo ID a vytvoříme BSP strom s unikátními ID
    let mut next_id = 0;
    let bsp_root = build_bsp(&triangles, 0, &mut next_id);
    
    let total_stats = BspStats {
        total_nodes: bsp_root.count_nodes(),
        total_triangles: bsp_root.count_triangles(),
        ..Default::default()
    };

    // stav pro vykreslovaný mesh
    let _glb_path: Option<PathBuf> = None;
    let material = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(100, 150, 255, 255), // Modrá barva aby byl model viditelný
        ..Default::default()
    });
    let _model = Gm::new(Mesh::new(&context, &cpu_mesh), material.clone());
    
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
    
    // Materiál pro směrový paprsek kamery
    let direction_material = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(255, 255, 0, 200), // Žlutá barva pro směrový paprsek
        ..Default::default()
    });

    let mut spectator_glow = Gm::new(Mesh::new(&context, &glow_mesh), spectator_glow_material);
    let mut third_person_glow = Gm::new(Mesh::new(&context, &glow_mesh), third_person_glow_material);
    
    // Vytvoření kuželu (cone) pro směrový indikátor kamery místo cylindru
    let direction_mesh = CpuMesh::cone(16);
    let mut camera_direction_ray = Gm::new(Mesh::new(&context, &direction_mesh), direction_material);
    
    let light = AmbientLight::new(&context, 1.0, Srgba::WHITE); // Zvýšit intenzitu světla

    // Nastavení výchozích pozic pro kamery (spawnpoint)
    let default_spectator_pos = Vector3::new(0.0, 2.0, 8.0);
    let default_third_person_pos = Vector3::new(5.0, 2.0, 8.0);

    // před inicializací kamery přidáme mutable proměnné pro stavy kamer obou režimů
    let mut cam = FreeCamera::new(default_spectator_pos);
    let mut spectator_state = CameraState::from_camera(&cam);
    let mut third_person_state = CameraState::new(default_third_person_pos); // Jiná pozice pro lepší vizualizaci
    let mut mode = CamMode::Spectator;
    
    // Proměnná pro sledování, zda zobrazit směr pohledu kamery
    let mut show_camera_direction = false;

    // Nastavení pozic glow efektů podle stavů kamer
    spectator_glow.set_transformation(Mat4::from_translation(vec3(
        spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z
    )) * Mat4::from_scale(0.2)); // Malé koule
    
    third_person_glow.set_transformation(Mat4::from_translation(vec3(
        third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z
    )) * Mat4::from_scale(0.2));

    // Inicializace InputManageru pro plynulé ovládání s více klávesami
    let mut input_manager = InputManager::new();

    // Přidání struktury pro sledování času přepnutí režimu
    let mut switch_delay = SwitchDelay::new(2.0); // 0.5 sekundy cooldown

    // ----------------------------------------------------------------------------
    // Stav pro interaktivní výběr BSP:
    // ----------------------------------------------------------------------------
    let mut selected_node: Option<usize> = None;
    let mut show_splitting_plane: bool = true;

    window.render_loop(move |frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &frame_input.events;

        // Aktualizuj stav kláves v InputManageru
        input_manager.update_key_states(events);

        // Vytvoření frustumu kamery pro view-culling
        let camera_obj = cam.cam(frame_input.viewport);
        
        // Použij správnou pozici pozorovatele pro traverzování BSP stromu
        let observer_position = match mode {
            CamMode::Spectator => cam.pos,  // V režimu Spectator používáme pozici kamery
            CamMode::ThirdPerson => spectator_state.pos,  // V režimu ThirdPerson používáme pozici Spectator kamery
        };
        
        // V třetí osobě vytvoříme frustum z pozice pozorovatele
        let frustum = if mode == CamMode::ThirdPerson {
            // Vytvoříme kameru z pozice spectator
            let spectator_dir = Vector3::new(
                spectator_state.yaw.cos() * spectator_state.pitch.cos(),
                spectator_state.pitch.sin(),
                spectator_state.yaw.sin() * spectator_state.pitch.cos()
            ).normalize();
            
            let spectator_camera = Camera::new_perspective(
                frame_input.viewport,
                spectator_state.pos,
                spectator_state.pos + spectator_dir,
                Vector3::unit_y(),
                Deg(60.0),
                0.1,
                1000.0
            );
            Frustum::from_camera(&spectator_camera)
        } else {
            Frustum::from_camera(&camera_obj)
        };

        // Volba způsobu cullingu - CPU nebo GPU - použití přejmenované funkce
        let mut current_stats = BspStats {
            total_nodes: total_stats.total_nodes,
            total_triangles: total_stats.total_triangles,
            ..Default::default()
        };

        // CPU culling - použijeme původní CPU implementaci
        let mut cpu_visible_triangles = Vec::new();
        traverse_bsp_with_frustum(&bsp_root, observer_position, &frustum, &mut current_stats, &mut cpu_visible_triangles);
        let visible_triangles = cpu_visible_triangles;

        // Vytvoř mesh z viditelných trojúhelníků pro vykreslení
        let visible_model = create_visible_mesh(&visible_triangles, &context);

        // --- GUI ---
        gui.update(&mut frame_input.events.clone(), frame_input.accumulated_time, frame_input.viewport, frame_input.device_pixel_ratio, |ctx| {
            egui::SidePanel::left("tree").show(ctx, |ui| {
                ui.heading("BSP Strom");
                ui.label(format!("Režim: {:?}", mode));

                // Přidáme sekci pro výběr uzlu BSP stromu
                ui.separator();
                ui.heading("Struktura BSP stromu");
                ui.checkbox(&mut show_splitting_plane, "Zobrazit dělící rovinu");
                
                // Použijeme scrollovatelné okno pro zobrazení stromu, aby nepřetekl
                egui::ScrollArea::vertical().show(ui, |ui| {
                    render_bsp_tree(ui, &bsp_root, &mut selected_node);
                });
                
                // Zobrazíme informace o vybraném uzlu
                if let Some(node_id) = selected_node {
                    if let Some(node) = find_node(&bsp_root, node_id) {
                        ui.separator();
                        ui.heading("Vybraný uzel");
                        ui.label(format!("ID: {}", node.id));
                        ui.label(format!("Trojúhelníků: {}", node.triangles.len()));
                        
                        if let Some(ref plane) = node.plane {
                            ui.label("Dělící rovina:");
                            ui.label(format!("Normála: ({:.2}, {:.2}, {:.2})", 
                                plane.n.x, plane.n.y, plane.n.z));
                            ui.label(format!("Vzdálenost: {:.2}", plane.d));
                        } else {
                            ui.label("List (bez dělící roviny)");
                        }
                        
                        ui.label("Obalový objem:");
                        ui.label(format!("Min: ({:.2}, {:.2}, {:.2})", 
                            node.bounds.min.x, node.bounds.min.y, node.bounds.min.z));
                        ui.label(format!("Max: ({:.2}, {:.2}, {:.2})", 
                            node.bounds.max.x, node.bounds.max.y, node.bounds.max.z));
                    }
                }

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
                ui.heading("Mesh Info");
                ui.label(format!("Vrcholy: {}", cpu_mesh.positions.len()));
                match &cpu_mesh.indices {
                    Indices::U32(idx) => ui.label(format!("Indexy (U32): {}", idx.len())),
                    Indices::U16(idx) => ui.label(format!("Indexy (U16): {}", idx.len())),
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
                ui.add(egui::Slider::new(&mut cam.look_speed, 0.5..=5.0)
                    .text("Rychlost rozhlížení"));

                ui.separator();
                ui.label("OSTATNÍ:");
                ui.label("• F - Přepnout na režim Spectator");
                ui.label("• G - Přepnout na režim ThirdPerson");
                ui.label("• Home - Návrat na výchozí pozici");
                ui.label("• PageUp/PageDown - Upravit rychlost");

                ui.separator();
                ui.heading("Informace o kameře");
                ui.label(format!("Aktivní režim: {:?}", mode));
                
                // Informace o Spectator kameře
                ui.collapsing("Spectator kamera", |ui| {
                    ui.label(format!("Pozice: ({:.1}, {:.1}, {:.1})",
                        spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z));
                    ui.label(format!("Směr (yaw): {:.1}°", spectator_state.yaw * 180.0 / std::f32::consts::PI));
                    ui.label(format!("Náklon (pitch): {:.1}°", spectator_state.pitch * 180.0 / std::f32::consts::PI));
                    ui.label(format!("Rychlost: {:.1}", spectator_state.speed));
                });
                
                // Informace o ThirdPerson kameře
                ui.collapsing("ThirdPerson kamera", |ui| {
                    ui.label(format!("Pozice: ({:.1}, {:.1}, {:.1})",
                        third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z));
                    ui.label(format!("Směr (yaw): {:.1}°", third_person_state.yaw * 180.0 / std::f32::consts::PI));
                    ui.label(format!("Náklon (pitch): {:.1}°", third_person_state.pitch * 180.0 / std::f32::consts::PI));
                    ui.label(format!("Rychlost: {:.1}", third_person_state.speed));
                });
                
                // Informace o aktuální kameře
                ui.label(format!("Aktuální pozice kamery: ({:.1}, {:.1}, {:.1})", 
                    cam.pos.x, cam.pos.y, cam.pos.z));
                ui.label(format!("Vzdálenost mezi kamerami: {:.1}", 
                    (spectator_state.pos - third_person_state.pos).magnitude()));
                
                ui.checkbox(&mut show_camera_direction, "Zobrazit směr pohledu kamery");
            });
        });

        // --- ovládání ---
        // --- ovládání přepnutí režimu pomocí kláves F a G ---
        let current_time = frame_input.accumulated_time;

        // Pomocná funkce pro přepínání režimů
        let mut switch_camera_mode = |target_mode: CamMode| {
            if switch_delay.can_switch(current_time) && mode != target_mode {
                match target_mode {
                    CamMode::Spectator => {
                        // Ulož aktuální pozici do ThirdPerson stavu
                        third_person_state = CameraState::from_camera(&cam);

                        // Přepni na Spectator režim a použij jeho stav
                        mode = CamMode::Spectator;
                        spectator_state.apply_to_camera(&mut cam);

                        println!("Přepnuto na režim: Spectator");
                    },
                    CamMode::ThirdPerson => {
                        // Ulož aktuální pozici do Spectator stavu
                        spectator_state = CameraState::from_camera(&cam);

                        // Přepni na ThirdPerson režim a použij jeho stav
                        mode = CamMode::ThirdPerson;
                        third_person_state.apply_to_camera(&mut cam);

                        println!("Přepnuto na režim: ThirdPerson");
                    }
                }

                // Zaznamenej čas posledního přepnutí
                switch_delay.record_switch(current_time);

                // Aktualizuj pozice glow značek
                spectator_glow.set_transformation(Mat4::from_translation(vec3(
                    spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z
                )) * Mat4::from_scale(0.2));
                
                third_person_glow.set_transformation(Mat4::from_translation(vec3(
                    third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z
                )) * Mat4::from_scale(0.2));
            }
        };

        // Klávesa F - přepnutí na Spectator režim
        if input_manager.is_key_pressed(KeyCode::F) {
            switch_camera_mode(CamMode::Spectator);
        }

        // Klávesa G - přepnutí na ThirdPerson režim
        if input_manager.is_key_pressed(KeyCode::G) {
            switch_camera_mode(CamMode::ThirdPerson);
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
        
        // Obsluha klávesy Home - návrat na výchozí pozici pro aktuální režim
        if input_manager.is_key_pressed(KeyCode::Home) {
            if mode == CamMode::Spectator {
                // Vytvoření nového stavu kamery s výchozí pozicí, ale aktuální rychlostí kamery
                let mut reset_state = CameraState::new(default_spectator_pos);
                reset_state.speed = cam.speed; // Zachová aktuální rychlost
                reset_state.apply_to_camera(&mut cam);
                println!("Kamera resetována na výchozí spectator pozici");
            } else { // ThirdPerson
                // Vytvoření nového stavu kamery s výchozí pozicí, ale aktuální rychlostí kamery
                let mut reset_state = CameraState::new(default_third_person_pos);
                reset_state.speed = cam.speed; // Zachová aktuální rychlost
                reset_state.apply_to_camera(&mut cam);
                println!("Kamera resetována na výchozí third person pozici");
            }
        }

        // Aktualizace kamery pomocí nové metody pro hladký pohyb
        cam.update_smooth(&input_manager, dt);
        
        // Aktualizace stavů kamer a značek podle aktuálního režimu
        if mode == CamMode::Spectator {
            // Aktualizuj stav aktuální kamery (Spectator)
            spectator_state = CameraState::from_camera(&cam);
            
            // Aktualizuj pozici značky aktuální kamery (Spectator)
            spectator_glow.set_transformation(Mat4::from_translation(vec3(
                spectator_state.pos.x, spectator_state.pos.y, spectator_state.pos.z
            )) * Mat4::from_scale(0.2));
            
            // Aktualizuj směrový paprsek pro spectator kameru
            if show_camera_direction {
                // Získáme směrový vektor kamery a nastavíme transformaci paprsku
                let dir = cam.dir();
                
                // Vytvoříme rotační matici, která natočí válec (který je standardně podél osy Y)
                // ve směru pohledu kamery
                
                // 1. Vypočítáme úhel mezi osou Y a směrovým vektorem kamery
                let y_axis = Vector3::unit_y();
                let angle = y_axis.dot(dir).acos();
                
                // 2. Vypočítáme osu rotace (kolmou na rovinu obsahující osu Y a směrový vektor)
                let rotation_axis = y_axis.cross(dir).normalize();
                
                // Vytvoření transformační matice pro válec
                let scale = 0.05; // tenký válec
                let length = 3.0; // délka paprsku
                
                // Vytvoření matice transformace
                let translation = Mat4::from_translation(vec3(
                    spectator_state.pos.x, 
                    spectator_state.pos.y, 
                    spectator_state.pos.z
                ));
                
                // Pokud je směrový vektor téměř rovnoběžný s osou Y, použijeme speciální zacházení
                let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01 {
                    // Pro případ kdy je vektor téměř rovnoběžný s osou Y
                    if dir.y > 0.0 {
                        Mat4::identity() // směr už je podél osy Y
                    } else {
                        // Rotace o 180° kolem osy X
                        Mat4::from_angle_x(Rad(std::f32::consts::PI))
                    }
                } else {
                    // Normální případ - rotace kolem vypočtené osy
                    Mat4::from_axis_angle(
                        vec3(rotation_axis.x, rotation_axis.y, rotation_axis.z),
                        Rad(angle)
                    )
                };
                
                // Měřítko - válec je standardně výšky 2.0, chceme jej natáhnout na délku `length`
                // a zúžit na šířku `scale`
                let scaling = Mat4::from_nonuniform_scale(scale, length / 2.0, scale);
                
                // Aplikujeme transformace v pořadí: měřítko, rotace, posun
                camera_direction_ray.set_transformation(translation * rotation * scaling);
            }
        } else {
            // Aktualizuj stav aktuální kamery (ThirdPerson)
            third_person_state = CameraState::from_camera(&cam);
            
            // Aktualizuj pozici značky aktuální kamery (ThirdPerson)
            third_person_glow.set_transformation(Mat4::from_translation(vec3(
                third_person_state.pos.x, third_person_state.pos.y, third_person_state.pos.z
            )) * Mat4::from_scale(0.2));
            
            // Když jsme v third person mode, zobrazíme směrový paprsek pro spectator kameru
            if show_camera_direction {
                // Získáme směrový vektor kamery a nastavíme transformaci paprsku
                // Tentokrát použijeme směr spectator kamery
                let dir = Vector3::new(
                    spectator_state.yaw.cos() * spectator_state.pitch.cos(),
                    spectator_state.pitch.sin(),
                    spectator_state.yaw.sin() * spectator_state.pitch.cos()
                ).normalize();
                
                // 1. Vypočítáme úhel mezi osou Y a směrovým vektorem kamery
                let y_axis = Vector3::unit_y();
                let angle = y_axis.dot(dir).acos();
                
                // 2. Vypočítáme osu rotace (kolmou na rovinu obsahující osu Y a směrový vektor)
                let rotation_axis = y_axis.cross(dir).normalize();
                
                // Vytvoření transformační matice pro válec
                let scale = 0.05; // tenký válec
                let length = 3.0; // délka paprsku
                
                // Vytvoření matice transformace
                let translation = Mat4::from_translation(vec3(
                    spectator_state.pos.x, 
                    spectator_state.pos.y, 
                    spectator_state.pos.z
                ));
                
                // Pokud je směrový vektor téměř rovnoběžný s osou Y, použijeme speciální zacházení
                let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01 {
                    // Pro případ kdy je vektor téměř rovnoběžný s osou Y
                    if dir.y > 0.0 {
                        Mat4::identity() // směr už je podél osy Y
                    } else {
                        // Rotace o 180° kolem osy X
                        Mat4::from_angle_x(Rad(std::f32::consts::PI))
                    }
                } else {
                    // Normální případ - rotace kolem vypočtené osy
                    Mat4::from_axis_angle(
                        vec3(rotation_axis.x, rotation_axis.y, rotation_axis.z),
                        Rad(angle)
                    )
                };
                
                // Měřítko - válec je standardně výšky 2.0, chceme jej natáhnout na délku `length`
                // a zúžit na šířku `scale`
                let scaling = Mat4::from_nonuniform_scale(scale, length / 2.0, scale);
                
                // Aplikujeme transformace v pořadí: měřítko, rotace, posun
                camera_direction_ray.set_transformation(translation * rotation * scaling);
            }
        }

        // --- vykreslení ---
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0)); // Tmavě šedé pozladí místo černého

        // Vytvoříme kolekci objektů k vykreslení
        let mut objects_to_render: Vec<&dyn Object> = vec![&visible_model];

        // Přidej glow koule pouze pokud nejsou na stejné pozici jako aktuální kamera
        let current_distance_to_spectator = (cam.pos - spectator_state.pos).magnitude();
        let current_distance_to_third_person = (cam.pos - third_person_state.pos).magnitude();

        // Zobraz spectator glow pouze pokud nejsme v spectator režimu nebo jsme daleko
        if mode != CamMode::Spectator || current_distance_to_spectator > 1.0 {
            objects_to_render.push(&spectator_glow);
        }

        // Zobraz third person glow pouze pokud nejsme v third person režimu nebo jsme daleko
        if mode != CamMode::ThirdPerson || current_distance_to_third_person > 1.0 {
            objects_to_render.push(&third_person_glow);
        }

        // Přidej směrový paprsek kamery
        if show_camera_direction {
            objects_to_render.push(&camera_direction_ray);
        }

        // Pokud je vybrán uzel, vytvoříme zvýrazněný mesh a případně mesh dělící roviny
        let mut maybe_highlight: Option<Gm<Mesh, ColorMaterial>> = None;
        let mut maybe_plane: Option<Gm<Mesh, ColorMaterial>> = None;

        if let Some(node_id) = selected_node {
            if let Some(node) = find_node(&bsp_root, node_id) {
                // Sbíráme všechny trojúhelníky v podstromu
                let mut highlight_triangles = Vec::new();
                collect_triangles_in_subtree(node, &mut highlight_triangles);
                
                // Vytvoříme zvýrazněný mesh
                if !highlight_triangles.is_empty() {
                    maybe_highlight = Some(create_highlight_mesh(&highlight_triangles, &context));
                }
                
                // Pokud má uzel dělící rovinu a je povoleno její zobrazení, vytvoříme její mesh
                if let Some(ref plane) = node.plane {
                    if show_splitting_plane {
                        maybe_plane = Some(create_plane_mesh(plane, &node.bounds, &context));
                    }
                }
            }
        }

        // Přidáme meshe do kolekce pro vykreslení
        if let Some(ref highlight) = maybe_highlight {
            objects_to_render.push(highlight);
        }
        if let Some(ref plane) = maybe_plane {
            objects_to_render.push(plane);
        }

        screen.render(&cam.cam(frame_input.viewport), &objects_to_render, &[&light]);
        let _ = gui.render();
        FrameOutput::default()
    });

    Ok(())
}

// Funkce pro převod CpuMesh na Triangle struktury
fn cpu_mesh_to_triangles(mesh: &CpuMesh) -> Vec<Triangle> {
    let mut triangles = Vec::new();

    // Získáme pozice vrcholů z meshe
    let positions = match &mesh.positions {
        Positions::F32(pos) => pos,
        _ => return Vec::new(), // Pokud nemáme F32 pozice, vrátíme prázdný vektor
    };

    // Zpracujeme indexy, pokud existují
    match &mesh.indices {
        Indices::U32(indices) => {
            // Pro každou trojici indexů vytvoříme trojúhelník
            for i in (0..indices.len()).step_by(3) {
                if i + 2 < indices.len() {
                    let a_idx = indices[i] as usize;
                    let b_idx = indices[i + 1] as usize;
                    let c_idx = indices[i + 2] as usize;

                    // Kontrola, zda indexy jsou v rozsahu
                    if a_idx < positions.len() && b_idx < positions.len() && c_idx < positions.len() {
                        let a = Vector3::new(positions[a_idx].x, positions[a_idx].y, positions[a_idx].z);
                        let b = Vector3::new(positions[b_idx].x, positions[b_idx].y, positions[b_idx].z);
                        let c = Vector3::new(positions[c_idx].x, positions[c_idx].y, positions[c_idx].z);

                        triangles.push(Triangle { a, b, c });
                    }
                }
            }
        },
        Indices::U16(indices) => {
            // Pro každou trojici indexů vytvoříme trojúhelník
            for i in (0..indices.len()).step_by(3) {
                if i + 2 < indices.len() {
                    let a_idx = indices[i] as usize;
                    let b_idx = indices[i + 1] as usize;
                    let c_idx = indices[i + 2] as usize;

                    // Kontrola, zda indexy jsou v rozsahu
                    if a_idx < positions.len() && b_idx < positions.len() && c_idx < positions.len() {
                        let a = Vector3::new(positions[a_idx].x, positions[a_idx].y, positions[a_idx].z);
                        let b = Vector3::new(positions[b_idx].x, positions[b_idx].y, positions[b_idx].z);
                        let c = Vector3::new(positions[c_idx].x, positions[c_idx].y, positions[c_idx].z);

                        triangles.push(Triangle { a, b, c });
                    }
                }
            }
        },
        Indices::None => {
            // Pokud nemáme indexy, předpokládáme, že pozice jsou přímo vrcholy trojúhelníků
            for i in (0..positions.len()).step_by(3) {
                if i + 2 < positions.len() {
                    let a = Vector3::new(positions[i].x, positions[i].y, positions[i].z);
                    let b = Vector3::new(positions[i + 1].x, positions[i + 1].y, positions[i + 1].z);
                    let c = Vector3::new(positions[i + 2].x, positions[i + 2].y, positions[i + 2].z);

                    triangles.push(Triangle { a, b, c });
                }
            }
        }
        _ => {
            // Přidáno pro pokrytí všech případů
            return Vec::new();
        }
    }

    triangles
}

// Funkce pro traverzování BSP stromu s frustum cullingem
fn traverse_bsp_with_frustum(
    node: &BspNode,
    observer_position: Vector3<f32>,
    frustum: &Frustum,
    stats: &mut BspStats,
    visible_triangles: &mut Vec<Triangle>
) {
    // Inkrementujeme počítadlo navštívených uzlů
    stats.nodes_visited += 1;

    // Nejprve zkontrolujeme, zda obalový objem uzlu protíná frustum
    let mut is_visible = true;

    // Testujeme proti všem rovinám frustumu
    for plane in &frustum.planes {
        if !node.bounds.intersects_plane(plane) {
            is_visible = false;
            break;
        }
    }

    if !is_visible {
        return; // Uzel je mimo frustum, končíme
    }

    // Přidáme trojúhelníky z tohoto uzlu do viditelných
    if !node.triangles.is_empty() {
        visible_triangles.extend(node.triangles.iter().cloned());
        stats.triangles_rendered += node.triangles.len() as u32;
    }

    // Pokud uzel není list, traverzujeme podstromy v závislosti na pozici pozorovatele
    if let Some(ref plane) = node.plane {
        let side = plane.classify(observer_position);

        if side >= 0 {
            // Pozorovatel je před rovinou, nejprve front, pak back
            if let Some(ref front) = node.front {
                traverse_bsp_with_frustum(front, observer_position, frustum, stats, visible_triangles);
            }
            if let Some(ref back) = node.back {
                traverse_bsp_with_frustum(back, observer_position, frustum, stats, visible_triangles);
            }
        } else {
            // Pozorovatel je za rovinou, nejprve back, pak front
            if let Some(ref back) = node.back {
                traverse_bsp_with_frustum(back, observer_position, frustum, stats, visible_triangles);
            }
            if let Some(ref front) = node.front {
                traverse_bsp_with_frustum(front, observer_position, frustum, stats, visible_triangles);
            }
        }
    }
}

// Funkce pro vytvoření materiálu a modelu z CPU meshe
fn create_material_and_model(context: &Context, cpu_mesh: &CpuMesh) -> (ColorMaterial, Gm<Mesh, ColorMaterial>) {
    let material = ColorMaterial::new_opaque(context, &CpuMaterial {
        albedo: Srgba::new(100, 150, 255, 255), // Modrá barva aby byl model viditelný
        ..Default::default()
    });
    let model = Gm::new(Mesh::new(context, cpu_mesh), material.clone());

    (material, model)
}

// Funkce pro vytvoření glow materiálu
fn create_glow_material(context: &Context, color: Srgba, opacity: u8) -> ColorMaterial {
    ColorMaterial::new_transparent(context, &CpuMaterial {
        albedo: Srgba::new(color.r, color.g, color.b, opacity),
        ..Default::default()
    })
}

// Funkce pro vytvoření směrového materiálu
fn create_direction_material(context: &Context, color: Srgba, opacity: u8) -> ColorMaterial {
    ColorMaterial::new_transparent(context, &CpuMaterial {
        albedo: Srgba::new(color.r, color.g, color.b, opacity),
        ..Default::default()
    })
}

// Funkce pro vytvoření směrového paprsku
fn create_direction_ray(context: &Context, position: Vector3<f32>, direction: Vector3<f32>, color: Srgba, opacity: u8, length: f32) -> Gm<Mesh, ColorMaterial> {
    let direction_material = create_direction_material(context, color, opacity);
    let direction_mesh = CpuMesh::cone(16);
    let mut direction_ray = Gm::new(Mesh::new(context, &direction_mesh), direction_material);

    // Vypočítáme úhel mezi osou Y a směrovým vektorem
    let y_axis = Vector3::unit_y();
    let angle = y_axis.dot(direction).acos();

    // Vypočítáme osu rotace (kolmou na rovinu obsahující osu Y a směrový vektor)
    let rotation_axis = y_axis.cross(direction).normalize();

    // Vytvoření transformační matice pro válec
    let scale = 0.05; // tenký válec
    let translation = Mat4::from_translation(position);

    // Pokud je směrový vektor téměř rovnoběžný s osou Y, použijeme speciální zacházení
    let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01 {
        // Pro případ kdy je vektor téměř rovnoběžný s osou Y
        if direction.y > 0.0 {
            Mat4::identity() // směr už je podél osy Y
        } else {
            // Rotace o 180° kolem osy X
            Mat4::from_angle_x(Rad(std::f32::consts::PI))
        }
    } else {
        // Normální případ - rotace kolem vypočtené osy
        Mat4::from_axis_angle(
            vec3(rotation_axis.x, rotation_axis.y, rotation_axis.z),
            Rad(angle)
        )
    };

    // Měřítko - válec je standardně výšky 2.0, chceme jej natáhnout na délku `length`
    // a zúžit na šířku `scale`
    let scaling = Mat4::from_nonuniform_scale(scale, length / 2.0, scale);

    // Aplikujeme transformace v pořadí: měřítko, rotace, posun
    direction_ray.set_transformation(translation * rotation * scaling);

    direction_ray
}

// Funkce pro resetování kamery na výchozí pozici
fn reset_camera_to_default(camera: &mut FreeCamera, default_position: Vector3<f32>, speed: f32) {
    // Vytvoření nového stavu kamery s výchozí pozicí, ale aktuální rychlostí kamery
    let mut reset_state = CameraState::new(default_position);
    reset_state.speed = speed; // Zachová aktuální rychlost
    reset_state.apply_to_camera(camera);
}

// Struktura pro sledování času přepnutí režimu kamery
struct SwitchDelay {
    last_switch_time: f64,
    cooldown: f64,
}

impl SwitchDelay {
    fn new(cooldown: f64) -> Self {
        Self {
            last_switch_time: 0.0,
            cooldown,
        }
    }

    fn can_switch(&self, current_time: f64) -> bool {
        current_time - self.last_switch_time >= self.cooldown
    }

    fn record_switch(&mut self, current_time: f64) {
        self.last_switch_time = current_time;
    }
}

// Přidání nové struktury CameraState pro ukládání stavu kamery
#[derive(Clone)]
struct CameraState {
    pos: Vector3<f32>,
    yaw: f32,
    pitch: f32,
    speed: f32,
}

impl CameraState {
    fn new(pos: Vector3<f32>) -> Self {
        Self {
            pos,
            yaw: -FRAC_PI_2, // výchozí směr
            pitch: 0.0,
            speed: 4.0,
        }
    }

    fn from_camera(camera: &FreeCamera) -> Self {
        Self {
            pos: camera.pos,
            yaw: camera.yaw,
            pitch: camera.pitch,
            speed: camera.speed,
        }
    }

    fn apply_to_camera(&self, camera: &mut FreeCamera) {
        camera.pos = self.pos;
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        camera.speed = self.speed;
    }
}

#[derive(Clone, Debug)]
struct BoundingBox {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

impl BoundingBox {
    fn new_empty() -> Self {
        Self {
            min: Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    fn from_triangle(tri: &Triangle) -> Self {
        let min = Vector3::new(
            tri.a.x.min(tri.b.x).min(tri.c.x),
            tri.a.y.min(tri.b.y).min(tri.c.y),
            tri.a.z.min(tri.b.z).min(tri.c.z),
        );
        let max = Vector3::new(
            tri.a.x.max(tri.b.x).max(tri.c.x),
            tri.a.y.max(tri.b.y).max(tri.c.y),
            tri.a.z.max(tri.b.z).max(tri.c.z),
        );
        BoundingBox { min, max }
    }

    fn from_triangles(triangles: &[Triangle]) -> Self {
        if triangles.is_empty() {
            return Self::new_empty();
        }

        let mut bounds = Self::from_triangle(&triangles[0]);
        for tri in triangles.iter().skip(1) {
            let tri_bounds = Self::from_triangle(tri);
            bounds = Self::encompass(&bounds, &tri_bounds);
        }
        bounds
    }

    fn encompass(box1: &Self, box2: &Self) -> Self {
        BoundingBox {
            min: Vector3::new(
                box1.min.x.min(box2.min.x),
                box1.min.y.min(box2.min.y),
                box1.min.z.min(box2.min.z),
            ),
            max: Vector3::new(
                box1.max.x.max(box2.max.x),
                box1.max.y.max(box2.max.y),
                box1.max.z.max(box2.max.z),
            ),
        }
    }

    /// Test against a single plane: return true if any part of the box is in front of the plane.
    fn intersects_plane(&self, plane: &Plane) -> bool {
        // compute the "positive vertex" for this plane's normal
        let p = Vector3::new(
            if plane.n.x >= 0.0 { self.max.x } else { self.min.x },
            if plane.n.y >= 0.0 { self.max.y } else { self.min.y },
            if plane.n.z >= 0.0 { self.max.z } else { self.min.z },
        );
        // if this farthest point is in front, the box may intersect or be in front
        plane.side(p) >= 0.0
    }
}

// Struktura pro reprezentaci frustumu kamery
struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    fn from_camera(camera: &Camera) -> Self {
        // Získáme view-projection matici
        let vp_matrix = camera.projection() * camera.view();

        // Převedeme na pole - Matrix4 nemá as_slice(), musíme použít jiný přístup
        let mat = [
            vp_matrix.x.x, vp_matrix.x.y, vp_matrix.x.z, vp_matrix.x.w,
            vp_matrix.y.x, vp_matrix.y.y, vp_matrix.y.z, vp_matrix.y.w,
            vp_matrix.z.x, vp_matrix.z.y, vp_matrix.z.z, vp_matrix.z.w,
            vp_matrix.w.x, vp_matrix.w.y, vp_matrix.w.z, vp_matrix.w.w,
        ];

        // Extrahujeme 6 rovin frustumu
        // Levá rovina
        let left = Plane {
            n: Vector3::new(
                mat[3] + mat[0],
                mat[7] + mat[4],
                mat[11] + mat[8],
            ).normalize(),
            d: (mat[15] + mat[12]) / (mat[3] + mat[0]).hypot((mat[7] + mat[4]).hypot(mat[11] + mat[8])),
        };

        // Pravá rovina
        let right = Plane {
            n: Vector3::new(
                mat[3] - mat[0],
                mat[7] - mat[4],
                mat[11] - mat[8],
            ).normalize(),
            d: (mat[15] - mat[12]) / (mat[3] - mat[0]).hypot((mat[7] - mat[4]).hypot(mat[11] - mat[8])),
        };

        // Spodní rovina
        let bottom = Plane {
            n: Vector3::new(
                mat[3] + mat[1],
                mat[7] + mat[5],
                mat[11] + mat[9],
            ).normalize(),
            d: (mat[15] + mat[13]) / (mat[3] + mat[1]).hypot((mat[7] + mat[5]).hypot(mat[11] + mat[9])),
        };

        // Horní rovina
        let top = Plane {
            n: Vector3::new(
                mat[3] - mat[1],
                mat[7] - mat[5],
                mat[11] - mat[9],
            ).normalize(),
            d: (mat[15] - mat[13]) / (mat[3] - mat[1]).hypot((mat[7] - mat[5]).hypot(mat[11] - mat[9])),
        };

        // Blízká rovina
        let near = Plane {
            n: Vector3::new(
                mat[3] + mat[2],
                mat[7] + mat[6],
                mat[11] + mat[10],
            ).normalize(),
            d: (mat[15] + mat[14]) / (mat[3] + mat[2]).hypot((mat[7] + mat[6]).hypot(mat[11] + mat[10])),
        };

        // Vzdálená rovina
        let far = Plane {
            n: Vector3::new(
                mat[3] - mat[2],
                mat[7] - mat[6],
                mat[11] - mat[10],
            ).normalize(),
            d: (mat[15] - mat[14]) / (mat[3] - mat[2]).hypot((mat[7] - mat[6]).hypot(mat[11] - mat[10])),
        };

        Frustum {
            planes: [left, right, bottom, top, near, far],
        }
    }
}
