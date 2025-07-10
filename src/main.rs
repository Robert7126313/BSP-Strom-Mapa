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
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use three_d::*;
use rayon::prelude::*; // Add Rayon prelude for parallelization

mod gpu_job;
mod shaders;
use crate::gpu_job::GpuJob;
use crate::input::{InputManager, KeyCode};
use crate::camera::{FreeCamera, CamMode, CameraState, SwitchDelay};
use crate::bsp::{BspNode, BspStats, Triangle, Frustum, build_bsp, find_node, find_deepest_node_containing_point, collect_triangles_in_subtree, create_plane_mesh, create_highlight_mesh, cpu_mesh_to_triangles, traverse_bsp_with_frustum, triangle_center};

mod input;
mod camera;
mod bsp;
mod gui;
// Message types for the channel
#[derive(Debug)]
enum Message {
    InitialTree(BspNode),
    NewFile {
        cpu_mesh: CpuMesh,
        triangles: Vec<Triangle>,
        file_name: String,
        load_status: String,
        bsp_tree: BspNode
    },
}




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
            vec3(tri.c.x, tri.c.y, tri.c.z),
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

fn gpu_cull_triangles(job: &GpuJob, tris: &[Triangle], frustum: &Frustum) -> Vec<Triangle> {
    let mut bytes = Vec::with_capacity(tris.len() * 3 * 16);
    for t in tris {
        for v in [&t.a, &t.b, &t.c] {
            bytes.extend_from_slice(&v.x.to_ne_bytes());
            bytes.extend_from_slice(&v.y.to_ne_bytes());
            bytes.extend_from_slice(&v.z.to_ne_bytes());
            bytes.extend_from_slice(&1f32.to_ne_bytes());
        }
    }
    unsafe { job.update_ssbo_data(&bytes); }

    let planes = frustum.as_vec4_array();
    let mut flat = [0f32; 24];
    for (i, p) in planes.iter().enumerate() {
        flat[i * 4] = p[0];
        flat[i * 4 + 1] = p[1];
        flat[i * 4 + 2] = p[2];
        flat[i * 4 + 3] = p[3];
    }
    unsafe {
        let loc = job.gl.get_uniform_location(job.prog, "frustum").unwrap();
        let count_loc = job.gl.get_uniform_location(job.prog, "num_tris").unwrap();
        job.gl.use_program(Some(job.prog));
        job.gl.uniform_4_f32_slice(Some(&loc), &flat);
        job.gl.uniform_1_u32(Some(&count_loc), tris.len() as u32);
        let groups = ((tris.len() as u32) + 63) / 64;
        job.dispatch(groups, 1, 1);
    }
    let out = unsafe { job.read_ssbo_u8(tris.len() * 4) };
    let mut result = Vec::new();
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        let flag = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if flag != 0 {
            result.push(tris[i].clone());
        }
    }
    result
}

// ---------------- Main --------------------------------------------------- //

fn main() -> Result<()> {
    println!("🚀 Spouštím BSP Viewer...");

    // okno + GL
    let window = Window::new(WindowSettings { 
        title: "BSP Viewer (three‑d 0.18)".into(), 
        ..Default::default() 
    })?;
    println!("✓ Okno vytvořeno");

    let context = window.gl();
    let mut gui = GUI::new(&context);
    println!("✓ GUI inicializováno");

    // stavová proměnná: název aktuálního souboru a úspěšnost načtení
    let initial_path = Path::new("assets/model.glb");
    println!("📁 Načítám model z: {}", initial_path.display());
    let (cpu_mesh, _load_status) = load_cpu_mesh(initial_path);
    println!("✓ Model načten");

    let mut loaded_file_name = if initial_path.exists() {
        initial_path.file_name().unwrap().to_string_lossy().into_owned()
    } else {
        "embedded sphere".to_string()
    };

    // Add state for file loading
    let mut current_cpu_mesh = cpu_mesh.clone();
    let mut current_triangles = cpu_mesh_to_triangles(&cpu_mesh);
    let mut file_loading = false;
    let mut gpu_job: Option<GpuJob> = None;

    // Vytvoření triangles z CPU meshe
    println!("🔺 Převádím mesh na trojúhelníky...");
    let triangles = cpu_mesh_to_triangles(&cpu_mesh);
    println!("✓ Převedeno {} trojúhelníků", triangles.len());

    // Inicializace GPU jobu pro frustum culling
    unsafe {
        let gl = &*context;
        let gpu_job = Some(GpuJob::new(
            gl,
            shaders::CULL_SHADER,
            triangles.len() * 3 * 16,
            triangles.len() * 4,
        ));
    }

    // Asynchronní stavba BSP stromu na pozadí
    println!("🌳 Spouštím stavbu BSP stromu na pozadí...");
    let mut bsp_root: Option<BspNode> = None;
    let triangles_clone = triangles.clone();
    let (tx, rx) = mpsc::channel();

    // Vytvoření klonu tx pro GUI
    let tx_gui = tx.clone();

    // Spuštění stavby BSP stromu v jiném vlákně
    thread::spawn(move || {
        let mut next_id = 0;
        let tree = build_bsp(&triangles_clone, 0, &mut next_id);
        println!("✓ BSP strom sestaven s {} uzly", tree.count_nodes());
        tx.send(Message::InitialTree(tree)).unwrap();
    });

    // Inicializujeme výchozí statistiky
    let mut total_stats = BspStats {
        total_nodes: 0,
        total_triangles: triangles.len() as u32,
        ..Default::default()
    };

    // Přidáme novou proměnnou pro vypnutí cullingu
    let mut disable_culling = false;
    // Volba pro GPU akceleraci frustum cullingu
    let mut use_gpu_culling = false;
    let mut hide_selected_area = false;

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
    
    let ambient_light = AmbientLight::new(&context, 1.0, Srgba::WHITE); // Zvýšit intenzitu světla

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

        // Zkontroluj, zda background thread dokončil stavbu BSP stromu
        if let Ok(message) = rx.try_recv() {
            match message {
                Message::InitialTree(tree) => {
                    total_stats.total_nodes = tree.count_nodes();
                    bsp_root = Some(tree);
                    println!("✅ BSP strom úspěšně načten do GUI!");
                }
                Message::NewFile { cpu_mesh: new_cpu_mesh, triangles: new_triangles, file_name, load_status: _, bsp_tree } => {
                    current_cpu_mesh = new_cpu_mesh;
                    current_triangles = new_triangles;
                    loaded_file_name = file_name;
                    file_loading = false;
                    bsp_root = Some(bsp_tree);
                    total_stats.total_nodes = bsp_root.as_ref().unwrap().count_nodes();
                    total_stats.total_triangles = current_triangles.len() as u32;
                    unsafe {
                        let gl = &*context;
                        gpu_job = Some(GpuJob::new(
                            gl,
                            shaders::CULL_SHADER,
                            current_triangles.len() * 3 * 16,
                            current_triangles.len() * 4,
                        ));
                    }
                    println!("✅ Nový model a BSP strom načteny!");
                }
            }
        }

        // Aktualizuj stav kláves v InputManageru
        input_manager.update_key_states(events);

        // Vytvoření frustumu kamery pro view-culling
        let camera_obj = cam.cam(frame_input.viewport);

        // Zpracuj kliknutí myší pro výběr uzlu
        let mut click_position = None;
        for event in events {
            if let Event::MousePress { button: MouseButton::Left, position, .. } = event {
                click_position = Some(*position);
            }
        }
        if let Some(pos) = click_position {
            if let Some(ref root) = bsp_root {
                let pick_mesh = Mesh::new(&context, &current_cpu_mesh);
                if let Some(hit) = three_d::pick(&context, &camera_obj, pos, [&pick_mesh]) {
                    let p = Vector3::new(hit.position.x, hit.position.y, hit.position.z);
                    if let Some(node) = find_deepest_node_containing_point(root, p) {
                        selected_node = Some(node.id);
                    }
                }
            }
        }
        
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

        // Výběr culling metody
        let visible_triangles = if disable_culling {
            // If culling is disabled, render all triangles without traversing
            // the BSP tree each frame.
            current_stats.nodes_visited = current_stats.total_nodes;
            current_stats.triangles_rendered = current_stats.total_triangles;
            current_triangles.clone()
        } else if use_gpu_culling {
            if let Some(ref job) = gpu_job {
                gpu_cull_triangles(job, &current_triangles, &frustum)
            } else {
                Vec::new()
            }
        } else {
            let mut tris = Vec::new();
            if let Some(ref root) = bsp_root {
                traverse_bsp_with_frustum(root, observer_position, &frustum, &mut current_stats, &mut tris);
            }
            tris
        };

        // 1) Shromáždění trojúhelníků z vybraného podstromu
        let mut picked_tris = Vec::new();
        if let Some(sel_id) = selected_node {
            if let Some(ref root) = bsp_root {
                if let Some(node) = find_node(root, sel_id) {
                    collect_triangles_in_subtree(node, &mut picked_tris);
                }
            }
        }

        // Pomocná funkce pro kvantizaci středu trojúhelníku
        fn quantized_center(tri: &Triangle) -> (i32, i32, i32) {
            let c = triangle_center(tri);
            ((c.x * 1000.0) as i32, (c.y * 1000.0) as i32, (c.z * 1000.0) as i32)
        }

        use std::collections::HashSet;
        let picked_centers: HashSet<_> = picked_tris.iter().map(|t| quantized_center(t)).collect();

        let mut normal_tris = Vec::with_capacity(visible_triangles.len());
        let mut highlight_tris = Vec::with_capacity(picked_tris.len());
        if hide_selected_area {
            for tri in visible_triangles.into_iter() {
                let c = quantized_center(&tri);
                if !picked_centers.contains(&c) {
                    normal_tris.push(tri);
                }
            }
        } else {
            for tri in visible_triangles.into_iter() {
                let c = quantized_center(&tri);
                if picked_centers.contains(&c) {
                    highlight_tris.push(tri);
                } else {
                    normal_tris.push(tri);
                }
            }
        }

        let base_model = create_visible_mesh(&normal_tris, &context);
        let highlight_model = if !highlight_tris.is_empty() && !hide_selected_area {
            Some(create_highlight_mesh(&highlight_tris, &context))
        } else {
            None
        };

        // --- GUI ---
        gui.update(&mut frame_input.events.clone(), frame_input.accumulated_time, frame_input.viewport, frame_input.device_pixel_ratio, |ctx| {
            crate::gui::draw_left_panel(
                ctx,
                mode,
                &mut loaded_file_name,
                &mut file_loading,
                &tx_gui,
                &rx,
                &mut current_cpu_mesh,
                &mut current_triangles,
                &mut bsp_root,
                &mut selected_node,
                &mut show_splitting_plane,
                &mut disable_culling,
                &mut use_gpu_culling,
                &mut hide_selected_area,
                &mut show_camera_direction,
                &mut spectator_state,
                &mut third_person_state,
                &mut cam,
                &current_stats,
            );
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
                
                // Měřítko - válec - válec je standardně výšky 2.0, chceme jej natáhnout na délku `length`
                // a zúžit na šířku `scale`
                let scaling = Mat4::from_nonuniform_scale(scale, length / 2.0, scale);
                
                // Aplikujeme transformace v pořadí: měřítko, rotace, posun
                camera_direction_ray.set_transformation(translation * rotation * scaling);
            }
        }

        // --- vykreslení ---
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0));
        let mut objects_to_render: Vec<&dyn Object> = Vec::new();
        objects_to_render.push(&base_model);
        // ... další objekty ...
        if let Some(ref h) = highlight_model {
            objects_to_render.push(h);
        }
        // --- ZOBRAZENÍ DĚLÍCÍ ROVINY ---
        let mut splitting_plane_mesh = None;
        if show_splitting_plane {
            if let Some(sel_id) = selected_node {
                if let Some(ref root) = bsp_root {
                    if let Some(node) = find_node(root, sel_id) {
                        if let Some(ref plane) = node.plane {
                            // Vytvoř mesh dělící roviny pro vybraný uzel
                            splitting_plane_mesh = Some(create_plane_mesh(plane, &node.bounds, &context));
                        }
                    }
                }
            }
        }
        if let Some(ref plane_mesh) = splitting_plane_mesh {
            objects_to_render.push(plane_mesh);
        }
        // ... další objekty ...
        screen.render(&cam.cam(frame_input.viewport), &objects_to_render, &[&ambient_light]);
        let _ = gui.render();
        FrameOutput::default()
    });

    Ok(())
}

// Funkce pro převod CpuMesh na Triangle struktury
