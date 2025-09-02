// SPDX-License-Identifier: MIT
//! Application entry point orchestrating rendering, input and BSP updates.
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
use cgmath::{Deg, InnerSpace, Vector3, Vector4};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, mpsc};
use std::thread;
use three_d::*; // Add Rayon prelude for parallelization

use crate::bsp::{
    build_bsp, collect_triangles_in_subtree, cpu_mesh_to_triangles, create_highlight_mesh,
    create_plane_mesh, find_deepest_node_containing_point, find_node, traverse_bsp_with_frustum,
    BspNode, BspStats, Frustum, Triangle,
};
use crate::camera::{CamMode, CameraState, FreeCamera};
use crate::config::CONFIG;
use crate::geometry::triangle_center;
use crate::input::{InputManager, KeyCode};
use crate::loader::load_cpu_mesh;
use log::info;

mod bsp;
mod camera;
mod config; // global constants
mod geometry;
mod gui;
mod input;
mod lang;
mod loader;
// Message types for the channel
#[derive(Debug)]
enum Message {
    InitialTree(BspNode),
    NewFile {
        cpu_mesh: CpuMesh,
        texture: Option<CpuTexture>,
        file_name: String,
    },
}

// Funkce pro vytvoření meshe z viditelných trojúhelníků
#[allow(dead_code)]
fn create_visible_mesh_old(
    triangles: &[Triangle],
    context: &Context,
) -> Gm<Mesh, PhysicalMaterial> {
    // Paralelní zpracování pozic a indexů
    let triangles_count = triangles.len();

    // Předalokujeme pozice vrcholů a vyplníme je paralelně
    let mut positions = vec![vec3(0.0, 0.0, 0.0); triangles_count * 3];
    positions
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, chunk)| {
            let tri = &triangles[i];
            chunk[0] = vec3(tri.a.x, tri.a.y, tri.a.z);
            chunk[1] = vec3(tri.b.x, tri.b.y, tri.b.z);
            chunk[2] = vec3(tri.c.x, tri.c.y, tri.c.z);
        });

    // Předalokujeme indexy a naplníme je paralelně
    let mut indices = vec![0u32; triangles_count * 3];
    indices
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, chunk)| {
            let base = i as u32 * 3;
            chunk[0] = base;
            chunk[1] = base + 1;
            chunk[2] = base + 2;
        });

    // Vytvoření nového meshe
    let visible_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };

    // Vytvoření materiálu a modelu
    let model_color = CONFIG.read().unwrap().model_color;
    let cpu_material = CpuMaterial {
        albedo: model_color,
        ..Default::default()
    };
    let material = if model_color.a < 255 {
        PhysicalMaterial::new_transparent(context, &cpu_material)
    } else {
        PhysicalMaterial::new_opaque(context, &cpu_material)
    };

    Gm::new(Mesh::new(context, &visible_mesh), material)
}

fn create_visible_mesh(
    triangles: &[Triangle],
    context: &Context,
    texture: Option<&Texture2DRef>,
) -> Gm<Mesh, PhysicalMaterial> {
    let triangles_count = triangles.len();
    let mut positions = vec![vec3(0.0, 0.0, 0.0); triangles_count * 3];
    let mut uvs = vec![vec2(0.0, 0.0); triangles_count * 3];
    positions
        .par_chunks_mut(3)
        .zip(uvs.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, (p_chunk, uv_chunk))| {
            let tri = &triangles[i];
            p_chunk[0] = vec3(tri.a.x, tri.a.y, tri.a.z);
            p_chunk[1] = vec3(tri.b.x, tri.b.y, tri.b.z);
            p_chunk[2] = vec3(tri.c.x, tri.c.y, tri.c.z);
            uv_chunk[0] = vec2(tri.uv_a.x, tri.uv_a.y);
            uv_chunk[1] = vec2(tri.uv_b.x, tri.uv_b.y);
            uv_chunk[2] = vec2(tri.uv_c.x, tri.uv_c.y);
        });
    let mut indices = vec![0u32; triangles_count * 3];
    indices
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, chunk)| {
            let base = i as u32 * 3;
            chunk[0] = base;
            chunk[1] = base + 1;
            chunk[2] = base + 2;
        });
    let visible_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        uvs: Some(uvs),
        ..Default::default()
    };
    let model_color = CONFIG.read().unwrap().model_color;
    let is_transparent = model_color.a < 255;
    let render_states = if is_transparent {
        RenderStates {
            blend: Blend::TRANSPARENCY,
            ..Default::default()
        }
    } else {
        RenderStates::default()
    };
    let material = PhysicalMaterial {
        albedo: model_color,
        albedo_texture: texture.cloned(),
        render_states,
        is_transparent,
        ..Default::default()
    };
    Gm::new(Mesh::new(context, &visible_mesh), material)
}

// ---------------- Main --------------------------------------------------- //

fn main() -> Result<()> {
    env_logger::init();
    info!("🚀 Launching BSP Viewer...");

    // window + GL
    let window = Window::new(WindowSettings {
        title: "BSP Viewer (three‑d 0.18)".into(),
        ..Default::default()
    })?;
    info!("✓ Window created");

    let context = window.gl();
    let mut gui = GUI::new(&context);
    info!("✓ GUI initialized");

    // Clone configuration so we don't hold a read lock for the entire program,
    // which would block the settings window from acquiring a write lock.
    let cfg = CONFIG.read().unwrap().clone();

    // state variable: name of current file and load status
    let initial_path = Path::new("assets/model.glb");
    info!("📁 Loading model from: {}", initial_path.display());
    let (cpu_mesh, initial_texture, _load_status) = load_cpu_mesh(initial_path);
    info!("✓ Model loaded");

    let mut loaded_file_name = String::new();
    let mut loaded_texture_name = String::new();

    if initial_path.exists() {
        loaded_file_name = initial_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
    } else {
        loaded_file_name = "embedded sphere".to_string();
    }

    // Add state for file loading
    let mut current_cpu_mesh = cpu_mesh.clone();
    let mut current_triangles = cpu_mesh_to_triangles(&cpu_mesh);
    let mut current_texture = initial_texture.map(|t| Texture2DRef::from_cpu_texture(&context, &t));
    let mut show_texture = current_texture.is_some();
    let mut file_loading = false;

    // Create triangles from CPU mesh
    info!("🔺 Converting mesh to triangles...");
    let triangles = cpu_mesh_to_triangles(&cpu_mesh);
    info!("✓ Converted {} triangles", triangles.len());

    // Asynchronous BSP tree build on background thread
    info!("🌳 Building BSP tree in background...");
    let mut bsp_root_full: Option<BspNode> = None;
    let mut bsp_root_preview: Option<BspNode> = None;
    let triangles_clone = triangles.clone();
    let (tx, rx) = mpsc::channel();

    // Vytvoření klonu tx pro GUI
    let tx_gui = tx.clone();

    // Spuštění stavby BSP stromu v jiném vlákně
    let branch_limit_clone = cfg.max_bsp_depth;
    thread::spawn(move || {
        let next_id = AtomicUsize::new(0);
        let tree = build_bsp(&triangles_clone, 0, branch_limit_clone, &next_id);
        info!("✓ BSP tree built with {} nodes", tree.count_nodes());
        tx.send(Message::InitialTree(tree)).unwrap();
    });

    // Inicializujeme výchozí statistiky
    let mut total_stats = BspStats {
        total_nodes: 0,
        total_triangles: triangles.len() as u32,
        ..Default::default()
    };

    let mut disable_culling = false;
    let mut show_loaded_model = true;
    let mut show_selected_model = true;
    let mut tree_window_open = false;
    let mut selected_node_help_open = false;
    let mut config_window_open = false;
    let mut branch_limit = cfg.default_branch_limit;
    let mut last_branch_limit = branch_limit;
    let mut last_default_branch_limit = cfg.default_branch_limit;
    let mut limit_culling = false;

    // stav pro vykreslovaný mesh
    let _glb_path: Option<PathBuf> = None;
    let cpu_material = CpuMaterial {
        albedo: cfg.model_color,
        ..Default::default()
    };
    let material = if cfg.model_color.a < 255 {
        ColorMaterial::new_transparent(&context, &cpu_material)
    } else {
        ColorMaterial::new_opaque(&context, &cpu_material)
    };
    let _model = Gm::new(Mesh::new(&context, &cpu_mesh), material.clone());

    // Glow efekty pro pozice kamer
    let glow_mesh = CpuMesh::sphere(16);

    // Materiály pro glow efekty
    let spectator_cpu_material = CpuMaterial {
        albedo: cfg.marker_color,
        ..Default::default()
    };
    let spectator_glow_material = if cfg.marker_color.a < 255 {
        ColorMaterial::new_transparent(&context, &spectator_cpu_material)
    } else {
        ColorMaterial::new_opaque(&context, &spectator_cpu_material)
    };

    let third_person_cpu_material = CpuMaterial {
        albedo: cfg.marker_color,
        ..Default::default()
    };
    let third_person_glow_material = if cfg.marker_color.a < 255 {
        ColorMaterial::new_transparent(&context, &third_person_cpu_material)
    } else {
        ColorMaterial::new_opaque(&context, &third_person_cpu_material)
    };

    // Materiály pro směrový indikátor kamery
    let direction_cpu_material = CpuMaterial {
        albedo: cfg.arrow_color,
        ..Default::default()
    };
    let shaft_material = if cfg.arrow_color.a < 255 {
        ColorMaterial::new_transparent(&context, &direction_cpu_material)
    } else {
        ColorMaterial::new_opaque(&context, &direction_cpu_material)
    };
    let head_material = if cfg.arrow_color.a < 255 {
        ColorMaterial::new_transparent(&context, &direction_cpu_material)
    } else {
        ColorMaterial::new_opaque(&context, &direction_cpu_material)
    };
    let tip_cpu_material = CpuMaterial {
        albedo: Srgba::new(255, 255, 80, 255),
        ..Default::default()
    };
    let tip_material = ColorMaterial::new_opaque(&context, &tip_cpu_material);

    let mut spectator_glow = Gm::new(Mesh::new(&context, &glow_mesh), spectator_glow_material);
    let mut third_person_glow =
        Gm::new(Mesh::new(&context, &glow_mesh), third_person_glow_material);

    // Vytvoření směrové šipky kamery: válec (tělo), kužel (hlava) a koule (špička)
    let shaft_mesh = CpuMesh::cylinder(16);
    let head_mesh = CpuMesh::cone(24);
    let tip_mesh = CpuMesh::sphere(10);
    let mut camera_arrow_shaft = Gm::new(Mesh::new(&context, &shaft_mesh), shaft_material);
    let mut camera_arrow_head = Gm::new(Mesh::new(&context, &head_mesh), head_material);
    let mut camera_arrow_tip = Gm::new(Mesh::new(&context, &tip_mesh), tip_material);

    let mut ambient_light = AmbientLight::new(
        &context,
        cfg.ambient_light_intensity,
        cfg.ambient_light_color,
    ); // Zvýšit intenzitu světla

    // Nastavení výchozích pozic pro kamery (spawnpoint)
    // před inicializací kamery přidáme mutable proměnné pro stavy kamer obou režimů
    let mut cam = FreeCamera::new(cfg.default_spectator_pos);
    let mut spectator_state = CameraState::from_camera(&cam);
    let mut third_person_state = CameraState::new(
        cfg.default_third_person_pos,
        cfg.default_third_person_yaw,
        cfg.default_third_person_pitch,
    ); // Jiná pozice pro lepší vizualizaci
    let mut mode = CamMode::Spectator;

    // Proměnná pro zobrazení značky spectator kamery
    let mut show_spectator_marker = false;

    // Nastavení pozic glow efektů podle stavů kamer
    spectator_glow.set_transformation(
        Mat4::from_translation(vec3(
            spectator_state.pos.x,
            spectator_state.pos.y,
            spectator_state.pos.z,
        )) * Mat4::from_scale(cfg.camera_marker_scale),
    ); // Malé koule

    third_person_glow.set_transformation(
        Mat4::from_translation(vec3(
            third_person_state.pos.x,
            third_person_state.pos.y,
            third_person_state.pos.z,
        )) * Mat4::from_scale(cfg.camera_marker_scale),
    );

    // Inicializace InputManageru pro plynulé ovládání s více klávesami
    let mut input_manager = InputManager::new();

    // ----------------------------------------------------------------------------
    // Stav pro interaktivní výběr BSP:
    // ----------------------------------------------------------------------------
    let mut selected_node: Option<usize> = None;
    // Počet uzlů navštívených při posledním hledání vybraného uzlu
    let mut last_pick_visits: u32 = 0;
    let mut show_splitting_plane: bool = true;

    window.render_loop(move |mut frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &mut frame_input.events;
        let cfg = CONFIG.read().unwrap().clone();
        let mut arrow_tip_position: Option<Vector3<f32>> = None;

        // Apply dynamic configuration updates
        ambient_light.intensity = cfg.ambient_light_intensity;
        ambient_light.color = cfg.ambient_light_color;
        spectator_glow.material.color = cfg.marker_color;
        third_person_glow.material.color = cfg.marker_color;
        camera_arrow_shaft.material.color = cfg.arrow_color;
        camera_arrow_head.material.color = cfg.arrow_color;

        // Synchronize branch limit with configuration changes
        if branch_limit > cfg.max_bsp_depth {
            branch_limit = cfg.max_bsp_depth;
        }
        if cfg.default_branch_limit != last_default_branch_limit {
            branch_limit = cfg.default_branch_limit.min(cfg.max_bsp_depth);
            last_default_branch_limit = cfg.default_branch_limit;
        }

        // Zkontroluj, zda background thread dokončil stavbu BSP stromu
        if let Ok(message) = rx.try_recv() {
            match message {
                Message::InitialTree(tree) => {
                    total_stats.total_nodes = tree.count_nodes();
                    bsp_root_full = Some(tree);
                    let next_id = AtomicUsize::new(0);
                    bsp_root_preview =
                        Some(build_bsp(&current_triangles, 0, branch_limit, &next_id));
                    info!("✅ BSP tree loaded into GUI");
                }
                Message::NewFile {
                    cpu_mesh: new_cpu_mesh,
                    texture: new_tex,
                    file_name,
                } => {
                    current_cpu_mesh = new_cpu_mesh;
                    loaded_file_name = file_name;
                    file_loading = false;
                    current_triangles = cpu_mesh_to_triangles(&current_cpu_mesh);
                    current_texture = new_tex.map(|t| Texture2DRef::from_cpu_texture(&context, &t));
                    if current_texture.is_some() {
                        show_texture = true;
                    }
                    let next_id = AtomicUsize::new(0);
                    bsp_root_full = Some(build_bsp(
                        &current_triangles,
                        0,
                        cfg.max_bsp_depth,
                        &next_id,
                    ));
                    total_stats.total_nodes = bsp_root_full.as_ref().unwrap().count_nodes();
                    let next_id = AtomicUsize::new(0);
                    bsp_root_preview =
                        Some(build_bsp(&current_triangles, 0, branch_limit, &next_id));
                    total_stats.total_triangles = current_triangles.len() as u32;
                    info!("✅ New model and BSP tree loaded");
                }
            }
        }

        // Vytvoření frustumu kamery pro view-culling
        let camera_obj = cam.cam(frame_input.viewport);

        // Použij správnou pozici pozorovatele pro traverzování BSP stromu
        let observer_position = match mode {
            CamMode::Spectator => cam.pos, // V režimu Spectator používáme pozici kamery
            CamMode::ThirdPerson => spectator_state.pos, // V režimu ThirdPerson používáme pozici Spectator kamery
        };

        // V třetí osobě vytvoříme frustum z pozice pozorovatele
        let frustum = if mode == CamMode::ThirdPerson {
            // Vytvoříme kameru z pozice spectator
            let spectator_dir = Vector3::new(
                spectator_state.yaw.cos() * spectator_state.pitch.cos(),
                spectator_state.pitch.sin(),
                spectator_state.yaw.sin() * spectator_state.pitch.cos(),
            )
            .normalize();

            let spectator_camera = Camera::new_perspective(
                frame_input.viewport,
                spectator_state.pos,
                spectator_state.pos + spectator_dir,
                Vector3::unit_y(),
                Deg(60.0),
                0.1,
                1000.0,
            );
            Frustum::from_camera(&spectator_camera)
        } else {
            Frustum::from_camera(&camera_obj)
        };

        // Výpočet statistik a cullingu na CPU
        let mut current_stats = BspStats {
            total_nodes: if limit_culling {
                bsp_root_preview
                    .as_ref()
                    .map(|n| n.count_nodes())
                    .unwrap_or(0)
            } else {
                total_stats.total_nodes
            },
            total_triangles: total_stats.total_triangles,
            ..Default::default()
        };

        let visible_triangles = if disable_culling {
            current_stats.camera.nodes_visited = current_stats.total_nodes;
            current_stats.camera.triangles_rendered = current_triangles.len() as u32;
            current_stats.camera.vertices_rendered = current_triangles.len() as u32 * 3;
            current_triangles.clone()
        } else {
            let mut tris = Vec::new();
            let root = if limit_culling {
                bsp_root_preview.as_ref()
            } else {
                bsp_root_full.as_ref()
            };
            if let Some(r) = root {
                traverse_bsp_with_frustum(
                    r,
                    observer_position,
                    &frustum,
                    &mut current_stats.camera,
                    &mut tris,
                );
            }
            tris
        };

        current_stats.nodes_to_selected = last_pick_visits;

        // --- GUI ---
        gui.update(
            events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |ctx| {
                crate::gui::draw_left_panel(
                    ctx,
                    &context,
                    mode,
                    &mut loaded_file_name,
                    &mut loaded_texture_name,
                    &mut file_loading,
                    &tx_gui,
                    &rx,
                    &mut current_cpu_mesh,
                    &mut current_triangles,
                    &mut current_texture,
                    &mut bsp_root_preview,
                    &mut selected_node,
                    &mut show_splitting_plane,
                    &mut disable_culling,
                    &mut show_loaded_model,
                    &mut show_selected_model,
                    &mut show_texture,
                    &mut show_spectator_marker,
                    &mut spectator_state,
                    &mut third_person_state,
                    &mut cam,
                    &current_stats,
                    &mut config_window_open,
                    &mut tree_window_open,
                    &mut branch_limit,
                    &mut limit_culling,
                    &mut selected_node_help_open,
                );
            },
        );

        if branch_limit != last_branch_limit {
            // Update the default branch limit in the global configuration so that
            // the slider in the configuration window stays in sync with the one in
            // the left control panel.
            CONFIG.write().unwrap().default_branch_limit = branch_limit;
            last_default_branch_limit = branch_limit;

            let next_id = AtomicUsize::new(0);
            bsp_root_preview = Some(build_bsp(&current_triangles, 0, branch_limit, &next_id));
            selected_node = None;
            last_branch_limit = branch_limit;
        }

        // Reload configuration so input handling reflects any changes made in the GUI
        let cfg = CONFIG.read().unwrap().clone();

        // Aktualizuj stav kláves v InputManageru
        input_manager.update_key_states(events);

        // Handle mouse clicks: cast a ray into the scene and select the
        // deepest BSP node containing the hit point. The selected ID is
        // reflected in the left control panel.
        if !gui.context().wants_pointer_input() {
            let mut click_position = None;
            for event in events.iter() {
                if let Event::MousePress {
                    button: MouseButton::Left,
                    position,
                    ..
                } = event
                {
                    click_position = Some(*position);
                }
            }
            if let Some(pos) = click_position {
                if let Some(ref root) = bsp_root_preview {
                    let pick_mesh = Mesh::new(&context, &current_cpu_mesh);
                    if let Some(hit) = three_d::pick(&context, &camera_obj, pos, [&pick_mesh]) {
                        let p = Vector3::new(hit.position.x, hit.position.y, hit.position.z);
                        let mut visited = 0;
                        if let Some(node) =
                            find_deepest_node_containing_point(root, p, &mut visited)
                        {
                            selected_node = Some(node.id);
                        }
                        last_pick_visits = visited;
                    }
                }
            }
        }

        // 1) Shromáždění trojúhelníků z vybraného podstromu
        let mut picked_tris = Vec::new();
        if let Some(sel_id) = selected_node {
            if let Some(ref root) = bsp_root_preview {
                if let Some(node) = find_node(root, sel_id) {
                    collect_triangles_in_subtree(node, &mut picked_tris);
                }
            }
        }

        // Pomocná funkce pro kvantizaci středu trojúhelníku
        fn quantized_center(tri: &Triangle) -> (i32, i32, i32) {
            let c = triangle_center(tri);
            (
                (c.x * 1000.0) as i32,
                (c.y * 1000.0) as i32,
                (c.z * 1000.0) as i32,
            )
        }

        fn world_to_screen(cam: &Camera, p: Vector3<f32>, viewport: Viewport) -> Option<[f32; 2]> {
            let clip: Vector4<f32> =
                cam.projection() * cam.view() * Vector4::new(p.x, p.y, p.z, 1.0);
            if clip.w.abs() < 1e-6 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            if ndc.z < -1.0 || ndc.z > 1.0 {
                return None;
            }
            let x = (ndc.x * 0.5 + 0.5) * viewport.width as f32;
            let y = (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.height as f32;
            Some([x, y])
        }

        use std::collections::HashSet;
        let picked_centers: HashSet<_> = picked_tris.iter().map(|t| quantized_center(t)).collect();

        let mut normal_tris = Vec::with_capacity(visible_triangles.len());
        let mut highlight_tris = Vec::with_capacity(picked_tris.len());
        for tri in visible_triangles.into_iter() {
            let is_selected = picked_centers.contains(&quantized_center(&tri));
            if is_selected {
                if show_selected_model {
                    highlight_tris.push(tri);
                } else if show_loaded_model {
                    normal_tris.push(tri);
                }
            } else if show_loaded_model {
                normal_tris.push(tri);
            }
        }

        let base_model = if !normal_tris.is_empty() {
            Some(create_visible_mesh(
                &normal_tris,
                &context,
                if show_texture {
                    current_texture.as_ref()
                } else {
                    None
                },
            ))
        } else {
            None
        };
        let highlight_model = if !highlight_tris.is_empty() {
            Some(create_highlight_mesh(&highlight_tris, &context))
        } else {
            None
        };

        // --- ovládání ---
        // --- ovládání přepnutí režimu pomocí kláves F a G ---

        // Pomocná funkce pro přepínání režimů
        let mut switch_camera_mode = |target_mode: CamMode| {
            if mode != target_mode {
                match target_mode {
                    CamMode::Spectator => {
                        // Ulož aktuální pozici do ThirdPerson stavu
                        third_person_state = CameraState::from_camera(&cam);

                        // Přepni na Spectator režim a použij jeho stav
                        mode = CamMode::Spectator;
                        spectator_state.apply_to_camera(&mut cam);

                        info!("Switched to Spectator mode");
                    }
                    CamMode::ThirdPerson => {
                        // Ulož aktuální pozici do Spectator stavu
                        spectator_state = CameraState::from_camera(&cam);

                        // Přepni na ThirdPerson režim a použij jeho stav
                        mode = CamMode::ThirdPerson;
                        third_person_state.apply_to_camera(&mut cam);

                        info!("Switched to ThirdPerson mode");
                    }
                }

                // Aktualizuj pozice glow značek
                spectator_glow.set_transformation(
                    Mat4::from_translation(vec3(
                        spectator_state.pos.x,
                        spectator_state.pos.y,
                        spectator_state.pos.z,
                    )) * Mat4::from_scale(cfg.camera_marker_scale),
                );

                third_person_glow.set_transformation(
                    Mat4::from_translation(vec3(
                        third_person_state.pos.x,
                        third_person_state.pos.y,
                        third_person_state.pos.z,
                    )) * Mat4::from_scale(cfg.camera_marker_scale),
                );
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
        const SPEED_STEP: f32 = 0.5;
        if input_manager.is_key_pressed(KeyCode::PageUp) {
            cam.speed += SPEED_STEP;
            // Keep configuration in sync with runtime speed adjustments
            CONFIG.write().unwrap().camera_speed = cam.speed;
            info!("Speed increased to: {:.1}", cam.speed);
        }
        if input_manager.is_key_pressed(KeyCode::PageDown) {
            cam.speed = (cam.speed - SPEED_STEP).max(0.1);
            // Keep configuration in sync with runtime speed adjustments
            CONFIG.write().unwrap().camera_speed = cam.speed;
            info!("Speed decreased to: {:.1}", cam.speed);
        }

        // Obsluha klávesy Home - návrat na výchozí pozici pro aktuální režim
        if input_manager.is_key_pressed(KeyCode::Home) {
            // Always reset to the hard-coded defaults rather than values from the config UI
            let defaults = crate::config::Config::default();
            if mode == CamMode::Spectator {
                // Vytvoření nového stavu kamery s výchozí pozicí, ale aktuální rychlostí kamery
                let mut reset_state = CameraState::new(
                    defaults.default_spectator_pos,
                    defaults.default_spectator_yaw,
                    defaults.default_spectator_pitch,
                );
                reset_state.speed = cam.speed; // Zachová aktuální rychlost
                reset_state.apply_to_camera(&mut cam);
                info!("Camera reset to default spectator position");
            } else {
                // ThirdPerson
                // Vytvoření nového stavu kamery s výchozí pozicí, ale aktuální rychlostí kamery
                let mut reset_state = CameraState::new(
                    defaults.default_third_person_pos,
                    defaults.default_third_person_yaw,
                    defaults.default_third_person_pitch,
                );
                reset_state.speed = cam.speed; // Zachová aktuální rychlost
                reset_state.apply_to_camera(&mut cam);
                info!("Camera reset to default third person position");
            }
        }

        // Aktualizace kamery pomocí nové metody pro hladký pohyb
        cam.update_smooth(&input_manager, dt);

        // Aktualizace stavů kamer a značek podle aktuálního režimu
        if mode == CamMode::Spectator {
            // Aktualizuj stav aktuální kamery (Spectator)
            spectator_state = CameraState::from_camera(&cam);

            // Aktualizuj pozici značky aktuální kamery (Spectator)
            spectator_glow.set_transformation(
                Mat4::from_translation(vec3(
                    spectator_state.pos.x,
                    spectator_state.pos.y,
                    spectator_state.pos.z,
                )) * Mat4::from_scale(cfg.camera_marker_scale),
            );

            // Aktualizuj směrovou šipku pro spectator kameru
            if show_spectator_marker {
                let dir = cam.dir().normalize();
                let base_pos = spectator_state.pos;
                let length = cfg.direction_ray_length;
                let head_len = length * 0.25;
                let shaft_len = length - head_len;
                let shaft_radius = cfg.direction_ray_thickness;
                let head_radius = shaft_radius * 2.0;
                let tip_radius = shaft_radius * 1.5;

                // Rotace z osy X do směru dir
                let x_axis = Vector3::unit_x();
                let angle = x_axis.dot(dir).acos();
                let rotation_axis = x_axis.cross(dir).normalize();
                let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01
                {
                    if dir.x > 0.0 {
                        Mat4::identity()
                    } else {
                        Mat4::from_angle_y(Rad(std::f32::consts::PI))
                    }
                } else {
                    Mat4::from_axis_angle(
                        vec3(rotation_axis.x, rotation_axis.y, rotation_axis.z),
                        Rad(angle),
                    )
                };

                // Tělo šipky
                let shaft_tr = Mat4::from_translation(vec3(base_pos.x, base_pos.y, base_pos.z));
                let shaft_scale =
                    Mat4::from_nonuniform_scale(shaft_len, shaft_radius, shaft_radius);
                camera_arrow_shaft.set_transformation(shaft_tr * rotation * shaft_scale);

                // Hlava šipky
                let head_pos = base_pos + dir * shaft_len;
                let head_tr = Mat4::from_translation(vec3(head_pos.x, head_pos.y, head_pos.z));
                let head_scale = Mat4::from_nonuniform_scale(head_len, head_radius, head_radius);
                camera_arrow_head.set_transformation(head_tr * rotation * head_scale);

                // Špička
                let tip_pos = base_pos + dir * length;
                let tip_tr = Mat4::from_translation(vec3(tip_pos.x, tip_pos.y, tip_pos.z));
                let tip_scale = Mat4::from_scale(tip_radius);
                camera_arrow_tip.set_transformation(tip_tr * tip_scale);

                arrow_tip_position = Some(tip_pos);
            }
        } else {
            // Aktualizuj stav aktuální kamery (ThirdPerson)
            third_person_state = CameraState::from_camera(&cam);

            // Aktualizuj pozici značky aktuální kamery (ThirdPerson)
            third_person_glow.set_transformation(
                Mat4::from_translation(vec3(
                    third_person_state.pos.x,
                    third_person_state.pos.y,
                    third_person_state.pos.z,
                )) * Mat4::from_scale(cfg.camera_marker_scale),
            );

            // Když jsme v third person mode, zobrazíme směrovou šipku pro spectator kameru
            if show_spectator_marker {
                spectator_glow.set_transformation(
                    Mat4::from_translation(vec3(
                        spectator_state.pos.x,
                        spectator_state.pos.y,
                        spectator_state.pos.z,
                    )) * Mat4::from_scale(cfg.camera_marker_scale),
                );

                let dir = Vector3::new(
                    spectator_state.yaw.cos() * spectator_state.pitch.cos(),
                    spectator_state.pitch.sin(),
                    spectator_state.yaw.sin() * spectator_state.pitch.cos(),
                )
                .normalize();

                let base_pos = spectator_state.pos;
                let length = cfg.direction_ray_length;
                let head_len = length * 0.25;
                let shaft_len = length - head_len;
                let shaft_radius = cfg.direction_ray_thickness;
                let head_radius = shaft_radius * 2.0;
                let tip_radius = shaft_radius * 1.5;

                // Rotace z osy X do směru dir
                let x_axis = Vector3::unit_x();
                let angle = x_axis.dot(dir).acos();
                let rotation_axis = x_axis.cross(dir).normalize();
                let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01
                {
                    if dir.x > 0.0 {
                        Mat4::identity()
                    } else {
                        Mat4::from_angle_y(Rad(std::f32::consts::PI))
                    }
                } else {
                    Mat4::from_axis_angle(
                        vec3(rotation_axis.x, rotation_axis.y, rotation_axis.z),
                        Rad(angle),
                    )
                };

                // Tělo šipky
                let shaft_tr = Mat4::from_translation(vec3(base_pos.x, base_pos.y, base_pos.z));
                let shaft_scale =
                    Mat4::from_nonuniform_scale(shaft_len, shaft_radius, shaft_radius);
                camera_arrow_shaft.set_transformation(shaft_tr * rotation * shaft_scale);

                // Hlava
                let head_pos = base_pos + dir * shaft_len;
                let head_tr = Mat4::from_translation(vec3(head_pos.x, head_pos.y, head_pos.z));
                let head_scale = Mat4::from_nonuniform_scale(head_len, head_radius, head_radius);
                camera_arrow_head.set_transformation(head_tr * rotation * head_scale);

                // Špička
                let tip_pos = base_pos + dir * length;
                let tip_tr = Mat4::from_translation(vec3(tip_pos.x, tip_pos.y, tip_pos.z));
                let tip_scale = Mat4::from_scale(tip_radius);
                camera_arrow_tip.set_transformation(tip_tr * tip_scale);

                arrow_tip_position = Some(tip_pos);
            }
        }

        // Vykreslení labelu s pozicí špičky šipky
        let render_cam = cam.cam(frame_input.viewport);
        if show_spectator_marker && mode == CamMode::ThirdPerson {
            if let Some(tip_pos) = arrow_tip_position {
                if let Some([sx, sy]) = world_to_screen(&render_cam, tip_pos, frame_input.viewport)
                {
                    let txt = format!("({:.2}, {:.2}, {:.2})", tip_pos.x, tip_pos.y, tip_pos.z);
                    let painter = gui.context().layer_painter(egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("arrow_label"),
                    ));
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(sx + 8.0, sy - 18.0),
                            egui::vec2(120.0, 18.0),
                        ),
                        6.0,
                        egui::Color32::from_black_alpha(160),
                    );
                    painter.text(
                        egui::pos2(sx + 14.0, sy - 14.0),
                        egui::Align2::LEFT_CENTER,
                        txt,
                        egui::FontId::monospace(13.0),
                        egui::Color32::WHITE,
                    );
                }
            }
        }

        // --- vykreslení ---
        let screen = frame_input.screen();
        // Clear the screen using the configured background color
        screen.clear(ClearState::color_and_depth(
            cfg.bg_color[0],
            cfg.bg_color[1],
            cfg.bg_color[2],
            1.0,
            1.0,
        ));
        let mut objects_to_render: Vec<&dyn Object> = Vec::new();
        if let Some(ref base) = base_model {
            objects_to_render.push(base);
        }
        if let Some(ref h) = highlight_model {
            objects_to_render.push(h);
        }
        // --- ZOBRAZENÍ DĚLÍCÍ ROVINY ---
        let mut splitting_plane_mesh = None;
        if show_splitting_plane {
            if let Some(sel_id) = selected_node {
                if let Some(ref root) = bsp_root_preview {
                    if let Some(node) = find_node(root, sel_id) {
                        if let Some(ref plane) = node.plane {
                            // Vytvoř mesh dělící roviny pro vybraný uzel
                            splitting_plane_mesh =
                                Some(create_plane_mesh(plane, &node.bounds, &context));
                        }
                    }
                }
            }
        }
        if let Some(ref plane_mesh) = splitting_plane_mesh {
            objects_to_render.push(plane_mesh);
        }
        if show_spectator_marker && mode == CamMode::ThirdPerson {
            objects_to_render.push(&spectator_glow);
            objects_to_render.push(&camera_arrow_shaft);
            objects_to_render.push(&camera_arrow_head);
            objects_to_render.push(&camera_arrow_tip);
        }
        // ... další objekty ...
        screen.render(&render_cam, &objects_to_render, &[&ambient_light]);
        let _ = gui.render();
        FrameOutput::default()
    });

    Ok(())
}

// Funkce pro převod CpuMesh na Triangle struktury
