// src/main.rs – „Hello‑BSP": načti .glb, postav BSP strom, klikem ukaž hloubku
// -----------------------------------------------------------------------------
// (c) 2025 – používá externí knihovny pro BSP, geometrii a optimalizaci
// -----------------------------------------------------------------------------

use anyhow::*;
use three_d::*;
use three_d_asset::{io, Model};
use std::result::Result as StdResult;
use std::result::Result::Ok as StdOk;

// Lepší error handling
use thiserror::Error;

// BSP a geometrické knihovny
use parry3d::shape::TriMesh;
use parry3d::query::{Ray, RayCast};
use nalgebra::{Point3 as NPoint3, Vector3 as NVector3, Isometry3};

#[derive(Error, Debug)]
pub enum BspError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    #[error("BSP construction failed: {0}")]
    BspConstructionError(String),
    #[error("Mesh processing failed: {0}")]
    MeshProcessingError(String),
}

// Jednoduchá AABB struktura pro prostorové dotazy
#[derive(Clone, Debug)]
pub struct AABB {
    min: [f32; 3],
    max: [f32; 3],
}

impl AABB {
    fn new() -> Self {
        AABB {
            min: [f32::INFINITY, f32::INFINITY, f32::INFINITY],
            max: [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        }
    }

    fn add_point(&mut self, x: f32, y: f32, z: f32) {
        self.min[0] = self.min[0].min(x);
        self.min[1] = self.min[1].min(y);
        self.min[2] = self.min[2].min(z);
        self.max[0] = self.max[0].max(x);
        self.max[1] = self.max[1].max(y);
        self.max[2] = self.max[2].max(z);
    }

    fn overlaps(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0] &&
        self.min[1] <= other.max[1] && self.max[1] >= other.min[1] &&
        self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }
}

// Jednoduchá trojúhelníková struktura
#[derive(Clone, Debug)]
pub struct Triangle {
    pub vertices: [[f32; 3]; 3],
    pub index: usize,
    pub bbox: AABB,
}

impl Triangle {
    fn new(vertices: [[f32; 3]; 3], index: usize) -> Self {
        let mut bbox = AABB::new();
        for vertex in &vertices {
            bbox.add_point(vertex[0], vertex[1], vertex[2]);
        }
        
        Triangle {
            vertices,
            index,
            bbox,
        }
    }
    
    fn centroid(&self) -> [f32; 3] {
        [
            (self.vertices[0][0] + self.vertices[1][0] + self.vertices[2][0]) / 3.0,
            (self.vertices[0][1] + self.vertices[1][1] + self.vertices[2][1]) / 3.0,
            (self.vertices[0][2] + self.vertices[1][2] + self.vertices[2][2]) / 3.0,
        ]
    }
}

pub struct BspScene {
    pub trimesh: TriMesh,
    pub triangles: Vec<Triangle>,
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
}

impl BspScene {
    pub fn new(positions: Vec<Vec3>, indices: Vec<u32>) -> StdResult<Self, BspError> {
        log::info!("Building BSP scene with {} vertices, {} triangles", 
                   positions.len(), indices.len() / 3);

        // Konverze na nalgebra typy pro parry3d
        let points: Vec<NPoint3<f32>> = positions.iter()
            .map(|v| NPoint3::new(v.x, v.y, v.z))
            .collect();
        
        let triangle_indices: Vec<[u32; 3]> = indices.chunks(3)
            .filter(|chunk| chunk.len() == 3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();

        // Vytvoř TriMesh pro collision detection
        let trimesh = TriMesh::new(points.clone(), triangle_indices.clone());

        // Vytvoř trojúhelníky pro vlastní prostorové dotazy
        let triangles: Vec<Triangle> = triangle_indices.iter().enumerate()
            .map(|(i, &[a, b, c])| {
                Triangle::new([
                    [positions[a as usize].x, positions[a as usize].y, positions[a as usize].z],
                    [positions[b as usize].x, positions[b as usize].y, positions[b as usize].z],
                    [positions[c as usize].x, positions[c as usize].y, positions[c as usize].z],
                ], i)
            })
            .collect();

        log::info!("Spatial index built with {} triangles", triangles.len());

        StdOk(BspScene {
            trimesh,
            triangles,
            positions,
            indices,
        })
    }

    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<(f32, usize, u32)> {
        let ray = Ray::new(
            NPoint3::new(origin.x, origin.y, origin.z),
            NVector3::new(direction.x, direction.y, direction.z)
        );

        // Isometry reprezentuje pozici a orientaci - potřebujeme ho pro cast_ray
        let identity = Isometry3::identity();
        
        // cast_ray vrací Option<f32> - vzdálenost průsečíku
        if let Some(toi) = self.trimesh.cast_ray(&identity, &ray, f32::MAX, true) {
            // Najdeme ID trojúhelníku pomocí lineárního vyhledávání
            let hit_point = ray.origin + ray.dir * toi;
            let hit_point_array = [hit_point.x, hit_point.y, hit_point.z];
            
            // Hledáme trojúhelník blízko průsečíku
            let search_radius = 0.1;
            let mut search_box = AABB::new();
            search_box.add_point(
                hit_point.x - search_radius, 
                hit_point.y - search_radius, 
                hit_point.z - search_radius
            );
            search_box.add_point(
                hit_point.x + search_radius, 
                hit_point.y + search_radius, 
                hit_point.z + search_radius
            );
            
            // Najdeme všechny trojúhelníky jejichž bbox se překrývá s vyhledávacím boxem
            let nearby_triangles: Vec<&Triangle> = self.triangles.iter()
                .filter(|tri| tri.bbox.overlaps(&search_box))
                .collect();
            
            // Vybereme ten s nejmenší vzdáleností k hit_point
            let triangle_id = nearby_triangles.iter()
                .map(|triangle| {
                    let centroid = triangle.centroid();
                    let distance_squared = 
                        (centroid[0] - hit_point_array[0]).powi(2) +
                        (centroid[1] - hit_point_array[1]).powi(2) +
                        (centroid[2] - hit_point_array[2]).powi(2);
                    (triangle.index, distance_squared)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            let depth = self.calculate_spatial_depth(origin);
            log::debug!("Raycast hit: triangle {}, distance {}, depth {}", 
                       triangle_id, toi, depth);
            Some((toi, triangle_id, depth))
        } else {
            None
        }
    }

    fn calculate_spatial_depth(&self, _point: Vec3) -> u32 {
        // Simulace BSP hloubky na základě prostorového dělení
        // Pro demonstraci - v reálné aplikaci by se použilo skutečné BSP dělení
        (self.positions.len() as f32).log2() as u32
    }

    pub fn get_triangles_in_region(&self, center: Vec3, radius: f32) -> Vec<usize> {
        let mut search_box = AABB::new();
        search_box.add_point(center.x - radius, center.y - radius, center.z - radius);
        search_box.add_point(center.x + radius, center.y + radius, center.z + radius);
        
        self.triangles.iter()
            .filter(|triangle| triangle.bbox.overlaps(&search_box))
            .map(|triangle| triangle.index)
            .collect()
    }
}

// Ray z kliknutí na obrazovce
fn screen_ray(camera: &Camera, vp: Viewport, pos: (f32, f32)) -> (Vec3, Vec3) {
    let ndc_x = (pos.0 / vp.width as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pos.1 / vp.height as f32) * 2.0;

    let inv_vp = (camera.projection() * camera.view()).invert().unwrap();
    let near_pt = inv_vp.transform_point(Point3::new(ndc_x, ndc_y, -1.0));
    let origin = camera.position();
    let dir = (vec3(near_pt.x, near_pt.y, near_pt.z) - origin).normalize();
    (origin, dir)
}

// Optimalizace meshe pomocí agresivnějšího podvzorkování
fn optimize_mesh(cpu_mesh: &CpuMesh, target_triangles: usize) -> StdResult<CpuMesh, BspError> {
    let positions = match &cpu_mesh.positions {
        Positions::F32(v) => v.clone(),
        Positions::F64(v) => v.iter().map(|&p| vec3(p.x as f32, p.y as f32, p.z as f32)).collect(),
    };
    
    let original_indices = match &cpu_mesh.indices {
        Indices::U32(v) => v.clone(),
        Indices::U16(v) => v.iter().map(|&x| x as u32).collect(),
        _ => return Err(BspError::MeshProcessingError("Invalid indices".to_string())),
    };

    let current_triangles = original_indices.len() / 3;
    
    // Vždy optimalizujeme pokud je více než target_triangles
    if current_triangles <= target_triangles {
        log::info!("Mesh has {} triangles (target: {}), keeping original",
                   current_triangles, target_triangles);
        return StdOk(cpu_mesh.clone());
    }

    // Agresivnější decimace pro velké modely
    let decimation_factor = (current_triangles / target_triangles).max(2);
    log::info!("Applying decimation factor {} to reduce {} triangles to ~{}",
               decimation_factor, current_triangles, current_triangles / decimation_factor);

    let optimized_indices: Vec<u32> = original_indices.chunks(3)
        .enumerate()
        .filter_map(|(i, triangle)| {
            if i % decimation_factor == 0 { Some(triangle) } else { None }
        })
        .flatten()
        .copied()
        .collect();

    let final_triangles = optimized_indices.len() / 3;
    log::info!("Mesh optimized: {} -> {} triangles", current_triangles, final_triangles);

    // Kontrola three-d limitu (přibližně 350,000 indexů)
    if optimized_indices.len() > 350000 {
        log::warn!("Still too many indices ({}), applying additional decimation", optimized_indices.len());
        let additional_factor = (optimized_indices.len() / 300000).max(2);
        let super_optimized: Vec<u32> = optimized_indices.chunks(3)
            .enumerate()
            .filter_map(|(i, triangle)| {
                if i % additional_factor == 0 { Some(triangle) } else { None }
            })
            .flatten()
            .copied()
            .collect();

        log::info!("Additional optimization: {} -> {} indices",
                   optimized_indices.len(), super_optimized.len());

        return StdOk(CpuMesh {
            positions: Positions::F32(positions),
            indices: Indices::U32(super_optimized),
            ..Default::default()
        });
    }

    StdOk(CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(optimized_indices),
        ..Default::default()
    })
}

fn load_model(model_path: &str) -> StdResult<CpuMesh, BspError> {
    log::info!("Loading model: {}", model_path);
    
    if !std::path::Path::new(model_path).exists() {
        return Err(BspError::ModelLoadError(format!("File {} does not exist", model_path)));
    }

    let mut assets = io::load(&[model_path])
        .map_err(|e| BspError::ModelLoadError(format!("Failed to load assets: {}", e)))?;

    let model_key = assets.keys()
        .find(|k| k.to_string_lossy().ends_with(".glb"))
        .or_else(|| assets.keys().next())
        .ok_or_else(|| BspError::ModelLoadError("No suitable model found in assets".to_string()))?
        .clone();

    let model = assets.deserialize::<Model>(&model_key)
        .map_err(|e| BspError::ModelLoadError(format!("Failed to deserialize model: {}", e)))?;

    log::info!("Model loaded: {} geometries", model.geometries.len());

    if model.geometries.is_empty() {
        return StdOk(create_test_mesh());
    }

    // Sekvenciální zpracov��ní pro lepší debugging
    let mut geometries_data = Vec::new();

    for (i, prim) in model.geometries.iter().enumerate() {
        log::info!("Processing geometry {}: name={:?}, material_index={:?}",
                   i, prim.name, prim.material_index);

        match &prim.geometry {
            three_d_asset::geometry::Geometry::Triangles(t) => {
                let mut cpu_mesh: CpuMesh = t.clone().into();

                // Zkontroluj, zda transformace prošla
                if let Err(e) = cpu_mesh.transform(prim.transformation) {
                    log::warn!("Failed to transform geometry {}: {:?}", i, e);
                    continue;
                }

                let positions = match &cpu_mesh.positions {
                    Positions::F32(v) => v.clone(),
                    Positions::F64(v) => v.iter().map(|&p| vec3(p.x as f32, p.y as f32, p.z as f32)).collect(),
                };

                let indices = match &cpu_mesh.indices {
                    Indices::U32(v) => v.clone(),
                    Indices::U16(v) => v.iter().map(|&x| x as u32).collect(),
                    Indices::None => {
                        // Vytvořit indexy pro triangle strip/fan
                        if positions.len() >= 3 {
                            (0..positions.len() as u32).collect()
                        } else {
                            continue;
                        }
                    },
                    &three_d::Indices::U8(_) => todo!(),
                };

                if positions.is_empty() || indices.is_empty() {
                    log::warn!("Empty geometry {} skipped", i);
                    continue;
                }

                log::info!("Geometry {} added: {} vertices, {} triangles",
                          i, positions.len(), indices.len() / 3);
                geometries_data.push((positions, indices));
            },
            three_d_asset::geometry::Geometry::Points(p) => {
                log::info!("Found points geometry {} with {} points (skipping for now)",
                          i, p.positions.len());
            },
        }
    }

    if geometries_data.is_empty() {
        log::warn!("No usable geometry found, using test mesh");
        return StdOk(create_test_mesh());
    }

    // Spojení všech geometrií
    let mut combined_positions = Vec::new();
    let mut combined_indices = Vec::new();
    let mut vertex_offset = 0u32;

    for (positions, indices) in geometries_data {
        combined_positions.extend(positions.clone());
        combined_indices.extend(indices.iter().map(|&idx| idx + vertex_offset));
        vertex_offset += positions.len() as u32;
    }

    log::info!("Combined mesh: {} vertices, {} triangles",
               combined_positions.len(), combined_indices.len() / 3);

    StdOk(CpuMesh {
        positions: Positions::F32(combined_positions),
        indices: Indices::U32(combined_indices),
        ..Default::default()
    })
}

struct FreeCameraController {
    move_speed: f32,
    look_sensitivity: f32,
    position: Vec3,
    pitch: f32,
    yaw: f32,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

impl FreeCameraController {
    fn new(initial_position: Vec3, initial_target: Vec3) -> Self {
        let forward = (initial_target - initial_position).normalize();
        let right = forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(forward).normalize();

        // Vypočítej počáteční pitch a yaw z forward vektoru
        let pitch = (-forward.y).asin();
        let yaw = forward.z.atan2(forward.x);

        FreeCameraController {
            move_speed: 10.0,
            look_sensitivity: 0.002,
            position: initial_position,
            pitch,
            yaw,
            forward,
            right,
            up,
        }
    }

    fn update_vectors(&mut self) {
        // Vypočítej forward vektor z pitch a yaw
        self.forward = vec3(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos()
        ).normalize();

        // Vypočítej right a up vektory
        self.right = self.forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
        self.up = self.right.cross(self.forward).normalize();
    }

    fn handle_input(&mut self, input: &mut FrameInput, delta_time: f32) {
        let move_distance = self.move_speed * delta_time;
        let rotation_speed = 1.5 * delta_time;
        
        // Původní pozice a orientace před změnami - pro vrácení změn v případě problémů
        let original_position = self.position;
        let original_forward = self.forward;
        let original_pitch = self.pitch;
        let original_yaw = self.yaw;
        
        // Kontrolujeme události kláves
        for event in &input.events {
            match event {
                Event::KeyPress { kind, .. } => {
                    match kind {
                        Key::W => self.position += self.forward * move_distance,
                        Key::S => self.position -= self.forward * move_distance,
                        Key::A => self.position -= self.right * move_distance,
                        Key::D => self.position += self.right * move_distance,
                        Key::ArrowUp => self.position += vec3(0.0, 1.0, 0.0) * move_distance,
                        Key::ArrowDown => self.position -= vec3(0.0, 1.0, 0.0) * move_distance,
                        Key::Q => self.yaw -= rotation_speed,
                        Key::E => self.yaw += rotation_speed,
                        Key::R => {
                            self.pitch += rotation_speed.min(0.1);
                            self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
                        },
                        Key::F => {
                            self.pitch -= rotation_speed.min(0.1);
                            self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
                        },
                        Key::Z => self.position += self.forward * (move_distance * 3.0),
                        Key::Home => {
                            // Reset kamery na výchozí pozici
                            println!("Resetuji kameru na výchozí pozici");
                            self.position = original_position;
                            self.pitch = original_pitch;
                            self.yaw = original_yaw;
                        },
                        _ => {}
                    }
                },
                _ => {}
            }
        }
        
        // Aktualizujeme vektory kamery po všech změnách najednou
        self.update_vectors();
        
        // Kód pro ovládání myší
        let mut mouse_motion = None;
        let mut right_mouse_pressed = false;
        
        for event in &input.events {
            match event {
                Event::MouseMotion { delta, .. } => {
                    mouse_motion = Some(*delta);
                }
                Event::MousePress { button: MouseButton::Right, .. } => {
                    right_mouse_pressed = true;
                }
                _ => {}
            }
        }
        
        if right_mouse_pressed {
            if let Some(delta) = mouse_motion {
                self.yaw += delta.0 * self.look_sensitivity;
                self.pitch -= delta.1 * self.look_sensitivity;
                
                // Omez pitch aby se kamera nepřetočila
                self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
                
                self.update_vectors();
            }
        }
    }

    fn apply_to_camera(&self, camera: &mut Camera) {
        // Použijeme set_view místo view matice
        camera.set_view(
            self.position,
            self.position + self.forward,
            self.up
        );
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    log::info!("Starting BSP-Strom-Mapa application");

    let window = Window::new(WindowSettings::default())?;
    let context = window.gl();

    // Načtení modelu s menším limitem na trojúhelníky
    let model_path = "assets/model.glb";
    println!("Pokus o načtení modelu z: {}", model_path);
    
    let absolute_path = std::path::Path::new(model_path).canonicalize()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| model_path.to_string());
    
    println!("Absolutní cesta k modelu: {}", absolute_path);
    
    // Kontrolujeme, zda soubor existuje
    if !std::path::Path::new(model_path).exists() {
        println!("VAROVÁNÍ: Soubor modelu nebyl nalezen! Bude použit výchozí model krychle.");
    }
    
    let cpu_mesh = load_model(model_path)
        .or_else(|e| {
            println!("Chyba při načítání modelu: {}. Použiji výchozí model krychle.", e);
            StdOk::<CpuMesh, BspError>(create_test_mesh())
        })?;

    // Snížíme limit na 100,000 trojúhelníků kvůli three-d limitům
    let optimized_mesh = optimize_mesh(&cpu_mesh, 100000)?;

    // Výpočet kamery podle bounding boxu
    let positions_vec = match &optimized_mesh.positions {
        Positions::F32(vec) => vec.clone(),
        Positions::F64(vec) => vec.iter().map(|&p| vec3(p.x as f32, p.y as f32, p.z as f32)).collect(),
    };

    let indices_vec = match &optimized_mesh.indices {
        Indices::U32(vec) => vec.clone(),
        Indices::U16(vec) => vec.iter().map(|&x| x as u32).collect(),
        _ => bail!("Invalid indices"),
    };

    // Postavení BSP scény
    let bsp_scene = BspScene::new(positions_vec.clone(), indices_vec.clone())?;
    log::info!("BSP scene constructed successfully");

    // Výpočet počáteční pozice kamery
    let (camera_position, camera_target) = if !positions_vec.is_empty() {
        let min = positions_vec.iter().fold(positions_vec[0], |acc, &pos| 
            vec3(acc.x.min(pos.x), acc.y.min(pos.y), acc.z.min(pos.z)));
        let max = positions_vec.iter().fold(positions_vec[0], |acc, &pos| 
            vec3(acc.x.max(pos.x), acc.y.max(pos.y), acc.z.max(pos.z)));
        
        let center = (min + max) * 0.5;
        let size = max - min;
        let radius = size.magnitude() * 0.5;
        
        // Ukládáme boundingBox pro pozdější použití
        let scene_min = min;
        let scene_max = max;
        let scene_center = center;
        let scene_radius = radius;
        
        println!("Scene bounding box: min={:?}, max={:?}, center={:?}, radius={:.2}", 
                min, max, center, radius);
        
        // Lepší umístění kamery pro viditelnost celé scény
        (center + vec3(0.0, radius * 0.5, radius * 2.5), center)
    } else {
        (vec3(4.0, 3.0, 4.0), vec3(0.0, 0.0, 0.0))
    };

    // Přidáme přímý výpis pozice kamery pro diagnostiku
    println!("Camera initial position: {:?}, target: {:?}", camera_position, camera_target);

    // GPU objekty
    let main_mesh = Mesh::new(&context, &optimized_mesh);
    let main_mat = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(200, 200, 200, 255), // Světlejší barva modelu
        ..Default::default()
    });
    let main_obj = Gm::new(main_mesh, main_mat);

    let highlight_mesh = Mesh::new(&context, &CpuMesh::default());
    let highlight_mat = ColorMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(255, 0, 0, 180),
        ..Default::default()
    });
    let mut highlight_obj = Gm::new(highlight_mesh, highlight_mat);

    // Kamera a free camera controller
    let mut camera = Camera::new_perspective(
        window.viewport(), camera_position, camera_target, vec3(0.0, 1.0, 0.0),
        degrees(55.0), 0.1, 10000.0  // Zvětšujeme far plane na 10000.0 místo 1000.0
    );

    let mut free_camera = FreeCameraController::new(camera_position, camera_target);
    let ambient = AmbientLight::new(&context, 1.0, Srgba::WHITE);

    let mut last_time = std::time::Instant::now();

    println!("Controls:");
    println!("WASD - Move forward/back/left/right");
    println!("Q/E - Rotate camera left/right");
    println!("R/F - Look up/down");
    println!("Arrow Up/Down - Move up/down");
    println!("Z - Move forward faster");
    println!("Left click - Ray cast to show BSP depth");

    // Render loop
    window.render_loop(move |mut input| {
        let current_time = std::time::Instant::now();
        let delta_time = (current_time - last_time).as_secs_f32();
        last_time = current_time;

        camera.set_viewport(input.viewport);

        // Update free camera
        free_camera.handle_input(&mut input, delta_time);
        free_camera.apply_to_camera(&mut camera);

        // Handle ray casting on left click
        for event in &input.events {
            if let Event::MousePress { button: MouseButton::Left, position, .. } = event {
                let (origin, dir) = screen_ray(&camera, input.viewport, (position.x, position.y));
                
                if let Some((distance, _triangle_id, depth)) = bsp_scene.raycast(origin, dir) {
                    println!("BSP depth = {} (distance: {:.2})", depth, distance);

                    // Najdi trojúhelníky v okolí
                    let nearby_triangles = bsp_scene.get_triangles_in_region(
                        origin + dir * distance, 2.0
                    );

                    if !nearby_triangles.is_empty() {
                        let mut sel = CpuMesh::default();
                        sel.positions = optimized_mesh.positions.clone();
                        
                        let highlight_indices: Vec<u32> = nearby_triangles.iter()
                            .flat_map(|&tri_idx| {
                                let base = tri_idx * 3;
                                if base + 2 < indices_vec.len() {
                                    vec![indices_vec[base], indices_vec[base + 1], indices_vec[base + 2]]
                                } else {
                                    vec![]
                                }
                            })
                            .collect();

                        sel.indices = Indices::U32(highlight_indices);
                        let new_highlight_mesh = Mesh::new(&context, &sel);
                        highlight_obj = Gm::new(new_highlight_mesh, highlight_obj.material.clone());

                        println!("Highlighted {} triangles", nearby_triangles.len());
                    }
                } else {
                    println!("Ray missed");
                }
            }
        }

        input.screen()
            .clear(ClearState::color_and_depth(0.8, 0.9, 1.0, 1.0, 1.0))
            .render(&camera, &[&main_obj, &highlight_obj], &[&ambient]);

        FrameOutput::default()
    });

    Ok(())
}

// Vytvoří jednoduchý test mesh (krychle)
fn create_test_mesh() -> CpuMesh {
    let positions = vec![
        // Přední strana
        vec3(-1.0, -1.0,  1.0), vec3( 1.0, -1.0,  1.0), vec3( 1.0,  1.0,  1.0), vec3(-1.0,  1.0,  1.0),
        // Zadní strana  
        vec3(-1.0, -1.0, -1.0), vec3(-1.0,  1.0, -1.0), vec3( 1.0,  1.0, -1.0), vec3( 1.0, -1.0, -1.0),
    ];
    
    let indices = vec![
        // Přední strana
        0, 1, 2,  2, 3, 0,
        // Zadní strana
        4, 5, 6,  6, 7, 4,
        // Levá strana
        4, 0, 3,  3, 5, 4,
        // Pravá strana
        1, 7, 6,  6, 2, 1,
        // Horní strana
        3, 2, 6,  6, 5, 3,
        // Dolní strana
        4, 7, 1,  1, 0, 4,
    ];

    CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    }
}
