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
use crate::config::{Config, CONFIG};
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

struct AppState {
    gui: GUI,
    cfg: Config,
    loaded_file_name: String,
    loaded_texture_name: String,
    current_cpu_mesh: CpuMesh,
    current_triangles: Vec<Triangle>,
    current_texture: Option<Texture2DRef>,
    show_texture: bool,
    file_loading: bool,
    bsp_root_full: Option<BspNode>,
    bsp_root_preview: Option<BspNode>,
    tx: mpsc::Sender<Message>,
    rx: mpsc::Receiver<Message>,
    total_stats: BspStats,
    disable_culling: bool,
    show_loaded_model: bool,
    show_selected_model: bool,
    tree_window_open: bool,
    selected_node_help_open: bool,
    config_window_open: bool,
    branch_limit: u32,
    last_branch_limit: u32,
    last_default_branch_limit: u32,
    limit_culling: bool,
    spectator_glow: Gm<Mesh, ColorMaterial>,
    third_person_glow: Gm<Mesh, ColorMaterial>,
    camera_arrow_shaft: Gm<Mesh, ColorMaterial>,
    camera_arrow_head: Gm<Mesh, ColorMaterial>,
    camera_arrow_tip: Gm<Mesh, ColorMaterial>,
    ambient_light: AmbientLight,
    cam: FreeCamera,
    spectator_state: CameraState,
    third_person_state: CameraState,
    mode: CamMode,
    show_spectator_marker: bool,
    input_manager: InputManager,
    selected_node: Option<usize>,
    last_pick_visits: u32,
    show_splitting_plane: bool,
}

fn init(context: &Context) -> AppState {
    let cfg = CONFIG.read().unwrap().clone();

    let initial_path = Path::new("assets/model.glb");
    info!("📁 Loading model from: {}", initial_path.display());
    let (cpu_mesh, initial_texture, _load_status) = load_cpu_mesh(initial_path);
    info!("✓ Model loaded");

    let loaded_file_name = if initial_path.exists() {
        initial_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    } else {
        "embedded sphere".to_string()
    };

    let current_triangles = cpu_mesh_to_triangles(&cpu_mesh);
    let triangles_clone = current_triangles.clone();
    let (tx, rx) = mpsc::channel();
    let tx_clone = tx.clone();
    let branch_limit_clone = cfg.max_bsp_depth;

    thread::spawn(move || {
        let next_id = AtomicUsize::new(0);
        let tree = build_bsp(&triangles_clone, 0, branch_limit_clone, &next_id);
        info!("✓ BSP tree built with {} nodes", tree.count_nodes());
        tx_clone.send(Message::InitialTree(tree)).unwrap();
    });

    let glow_mesh = CpuMesh::sphere(16);
    let spectator_cpu_material = CpuMaterial {
        albedo: cfg.marker_color,
        ..Default::default()
    };
    let spectator_glow_material = if cfg.marker_color.a < 255 {
        ColorMaterial::new_transparent(context, &spectator_cpu_material)
    } else {
        ColorMaterial::new_opaque(context, &spectator_cpu_material)
    };

    let third_person_cpu_material = CpuMaterial {
        albedo: cfg.marker_color,
        ..Default::default()
    };
    let third_person_glow_material = if cfg.marker_color.a < 255 {
        ColorMaterial::new_transparent(context, &third_person_cpu_material)
    } else {
        ColorMaterial::new_opaque(context, &third_person_cpu_material)
    };

    let direction_cpu_material = CpuMaterial {
        albedo: cfg.arrow_color,
        ..Default::default()
    };
    let shaft_material = if cfg.arrow_color.a < 255 {
        ColorMaterial::new_transparent(context, &direction_cpu_material)
    } else {
        ColorMaterial::new_opaque(context, &direction_cpu_material)
    };
    let head_material = if cfg.arrow_color.a < 255 {
        ColorMaterial::new_transparent(context, &direction_cpu_material)
    } else {
        ColorMaterial::new_opaque(context, &direction_cpu_material)
    };
    let tip_cpu_material = CpuMaterial {
        albedo: Srgba::new(255, 255, 80, 255),
        ..Default::default()
    };
    let tip_material = ColorMaterial::new_opaque(context, &tip_cpu_material);

    let spectator_glow = Gm::new(Mesh::new(context, &glow_mesh), spectator_glow_material);
    let third_person_glow = Gm::new(Mesh::new(context, &glow_mesh), third_person_glow_material);

    let shaft_mesh = CpuMesh::cylinder(16);
    let head_mesh = CpuMesh::cone(24);
    let tip_mesh = CpuMesh::sphere(10);

    let cam = FreeCamera::new(cfg.default_spectator_pos);
    let spectator_state = CameraState::from_camera(&cam);
    let third_person_state = CameraState::new(
        cfg.default_third_person_pos,
        cfg.default_third_person_yaw,
        cfg.default_third_person_pitch,
    );

    let current_texture = initial_texture.map(|t| Texture2DRef::from_cpu_texture(context, &t));
    let show_texture = current_texture.is_some();

    AppState {
        gui: GUI::new(context),
        cfg: cfg.clone(),
        loaded_file_name,
        loaded_texture_name: String::new(),
        current_cpu_mesh: cpu_mesh.clone(),
        current_triangles,
        current_texture,
        show_texture,
        file_loading: false,
        bsp_root_full: None,
        bsp_root_preview: None,
        tx,
        rx,
        total_stats: BspStats {
            total_nodes: 0,
            total_triangles: cpu_mesh_to_triangles(&cpu_mesh).len() as u32,
            ..Default::default()
        },
        disable_culling: false,
        show_loaded_model: true,
        show_selected_model: true,
        tree_window_open: false,
        selected_node_help_open: false,
        config_window_open: false,
        branch_limit: cfg.default_branch_limit,
        last_branch_limit: cfg.default_branch_limit,
        last_default_branch_limit: cfg.default_branch_limit,
        limit_culling: false,
        spectator_glow,
        third_person_glow,
        camera_arrow_shaft: Gm::new(Mesh::new(context, &shaft_mesh), shaft_material),
        camera_arrow_head: Gm::new(Mesh::new(context, &head_mesh), head_material),
        camera_arrow_tip: Gm::new(Mesh::new(context, &tip_mesh), tip_material),
        ambient_light: AmbientLight::new(
            context,
            cfg.ambient_light_intensity,
            cfg.ambient_light_color,
        ),
        cam,
        spectator_state,
        third_person_state,
        mode: CamMode::Spectator,
        show_spectator_marker: false,
        input_manager: InputManager::new(),
        selected_node: None,
        last_pick_visits: 0,
        show_splitting_plane: true,
    }
}

fn handle_input(state: &mut AppState, events: &mut [Event], context: &Context, camera_obj: &Camera) {
    state.input_manager.update_key_states(events);

    if !state.gui.context().wants_pointer_input() {
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
            if let Some(ref root) = state.bsp_root_preview {
                let pick_mesh = Mesh::new(context, &state.current_cpu_mesh);
                if let Some(hit) = three_d::pick(context, camera_obj, pos, [&pick_mesh]) {
                    let p = Vector3::new(hit.position.x, hit.position.y, hit.position.z);
                    let mut visited = 0;
                    if let Some(node) =
                        find_deepest_node_containing_point(root, p, &mut visited)
                    {
                        state.selected_node = Some(node.id);
                    }
                    state.last_pick_visits = visited;
                }
            }
        }
    }

    let mut switch_camera_mode = |target_mode: CamMode| {
        if state.mode != target_mode {
            match target_mode {
                CamMode::Spectator => {
                    state.third_person_state = CameraState::from_camera(&state.cam);
                    state.mode = CamMode::Spectator;
                    state.spectator_state.apply_to_camera(&mut state.cam);
                    info!("Switched to Spectator mode");
                }
                CamMode::ThirdPerson => {
                    state.spectator_state = CameraState::from_camera(&state.cam);
                    state.mode = CamMode::ThirdPerson;
                    state.third_person_state.apply_to_camera(&mut state.cam);
                    info!("Switched to ThirdPerson mode");
                }
            }
            state.spectator_glow.set_transformation(
                Mat4::from_translation(state.spectator_state.pos)
                    * Mat4::from_scale(state.cfg.camera_marker_scale),
            );
            state.third_person_glow.set_transformation(
                Mat4::from_translation(state.third_person_state.pos)
                    * Mat4::from_scale(state.cfg.camera_marker_scale),
            );
        }
    };

    if state.input_manager.is_key_pressed(KeyCode::F) {
        switch_camera_mode(CamMode::Spectator);
    }
    if state.input_manager.is_key_pressed(KeyCode::G) {
        switch_camera_mode(CamMode::ThirdPerson);
    }

    const SPEED_STEP: f32 = 0.5;
    if state.input_manager.is_key_pressed(KeyCode::PageUp) {
        state.cam.speed += SPEED_STEP;
        CONFIG.write().unwrap().camera_speed = state.cam.speed;
        info!("Speed increased to: {:.1}", state.cam.speed);
    }
    if state.input_manager.is_key_pressed(KeyCode::PageDown) {
        state.cam.speed = (state.cam.speed - SPEED_STEP).max(0.1);
        CONFIG.write().unwrap().camera_speed = state.cam.speed;
        info!("Speed decreased to: {:.1}", state.cam.speed);
    }

    if state.input_manager.is_key_pressed(KeyCode::Home) {
        let defaults = crate::config::Config::default();
        if state.mode == CamMode::Spectator {
            let mut reset_state = CameraState::new(
                defaults.default_spectator_pos,
                defaults.default_spectator_yaw,
                defaults.default_spectator_pitch,
            );
            reset_state.speed = state.cam.speed;
            reset_state.apply_to_camera(&mut state.cam);
            info!("Camera reset to default spectator position");
        } else {
            let mut reset_state = CameraState::new(
                defaults.default_third_person_pos,
                defaults.default_third_person_yaw,
                defaults.default_third_person_pitch,
            );
            reset_state.speed = state.cam.speed;
            reset_state.apply_to_camera(&mut state.cam);
            info!("Camera reset to default third person position");
        }
    }
}

fn update_gui(state: &mut AppState, frame_input: &FrameInput, context: &Context, current_stats: &BspStats) {
    state.gui.update(
        &mut frame_input.events.clone(),
        frame_input.accumulated_time,
        frame_input.viewport,
        frame_input.device_pixel_ratio,
        |ctx| {
            crate::gui::draw_left_panel(
                ctx,
                context,
                state.mode,
                &mut state.loaded_file_name,
                &mut state.loaded_texture_name,
                &mut state.file_loading,
                &state.tx,
                &state.rx,
                &mut state.current_cpu_mesh,
                &mut state.current_triangles,
                &mut state.current_texture,
                &mut state.bsp_root_preview,
                &mut state.selected_node,
                &mut state.show_splitting_plane,
                &mut state.disable_culling,
                &mut state.show_loaded_model,
                &mut state.show_selected_model,
                &mut state.show_texture,
                &mut state.show_spectator_marker,
                &mut state.spectator_state,
                &mut state.third_person_state,
                &mut state.cam,
                current_stats,
                &mut state.config_window_open,
                &mut state.tree_window_open,
                &mut state.branch_limit,
                &mut state.limit_culling,
                &mut state.selected_node_help_open,
            );
        },
    );
}

fn update_camera(state: &mut AppState, dt: f32) -> Option<Vector3<f32>> {
    state.cam.update_smooth(&state.input_manager, dt);

    if state.mode == CamMode::Spectator {
        state.spectator_state = CameraState::from_camera(&state.cam);
        state.spectator_glow.set_transformation(
            Mat4::from_translation(state.spectator_state.pos)
                * Mat4::from_scale(state.cfg.camera_marker_scale),
        );
    } else {
        state.third_person_state = CameraState::from_camera(&state.cam);
        state.third_person_glow.set_transformation(
            Mat4::from_translation(state.third_person_state.pos)
                * Mat4::from_scale(state.cfg.camera_marker_scale),
        );
    }

    if state.show_spectator_marker {
        let state_to_show = if state.mode == CamMode::ThirdPerson {
            // In third-person mode, the arrow shows the direction of the spectator camera.
            &state.spectator_state
        } else {
            // In spectator mode, the arrow shows the direction of the third-person camera.
            &state.third_person_state
        };
        Some(update_camera_arrow(
            state_to_show,
            &state.cfg,
            &mut state.camera_arrow_shaft,
            &mut state.camera_arrow_head,
            &mut state.camera_arrow_tip,
        ))
    } else {
        None
    }
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

fn render_scene(
    screen: &RenderTarget,
    context: &Context,
    state: &AppState,
    base_model: Option<&Gm<Mesh, PhysicalMaterial>>,
    highlight_model: Option<&Gm<Mesh, PhysicalMaterial>>,
) {
    let render_cam = state.cam.cam(screen.viewport());
    let mut frame = screen.clear(ClearState::color_and_depth(
        state.cfg.bg_color[0],
        state.cfg.bg_color[1],
        state.cfg.bg_color[2],
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

    let mut splitting_plane_mesh = None;
    if state.show_splitting_plane {
        if let Some(sel_id) = state.selected_node {
            if let Some(ref root) = state.bsp_root_preview {
                if let Some(node) = find_node(root, sel_id) {
                    if let Some(ref plane) = node.plane {
                        splitting_plane_mesh =
                            Some(create_plane_mesh(plane, &node.bounds, context));
                    }
                }
            }
        }
    }
    if let Some(ref plane_mesh) = splitting_plane_mesh {
        objects_to_render.push(plane_mesh);
    }

    if state.show_spectator_marker {
        if state.mode == CamMode::ThirdPerson { // We are in 3rd person, show spectator marker
            objects_to_render.push(&state.spectator_glow);
            objects_to_render.push(&state.camera_arrow_shaft);
            objects_to_render.push(&state.camera_arrow_head);
            objects_to_render.push(&state.camera_arrow_tip);
        } else { // We are in spectator, show 3rd person marker
            objects_to_render.push(&state.third_person_glow);
        }
    }

    frame.render(&render_cam, &objects_to_render, &[&state.ambient_light]);
}

fn process_visible_triangles(
    visible_triangles: Vec<Triangle>,
    state: &AppState,
    context: &Context,
) -> (
    Option<Gm<Mesh, PhysicalMaterial>>,
    Option<Gm<Mesh, PhysicalMaterial>>,
) {
    let mut picked_tris = Vec::new();
    if let Some(sel_id) = state.selected_node {
        if let Some(ref root) = state.bsp_root_preview {
            if let Some(node) = find_node(root, sel_id) {
                collect_triangles_in_subtree(node, &mut picked_tris);
            }
        }
    }

    fn quantized_center(tri: &Triangle) -> (i32, i32, i32) {
        let c = triangle_center(tri);
        (
            (c.x * 1000.0) as i32,
            (c.y * 1000.0) as i32,
            (c.z * 1000.0) as i32,
        )
    }

    use std::collections::HashSet;
    let picked_centers: HashSet<_> = picked_tris.iter().map(|t| quantized_center(t)).collect();

    let mut normal_tris = Vec::with_capacity(visible_triangles.len());
    let mut highlight_tris = Vec::with_capacity(picked_tris.len());
    for tri in visible_triangles.into_iter() {
        let is_selected = picked_centers.contains(&quantized_center(&tri));
        if is_selected {
            if state.show_selected_model {
                highlight_tris.push(tri);
            } else if state.show_loaded_model {
                normal_tris.push(tri);
            }
        } else if state.show_loaded_model {
            normal_tris.push(tri);
        }
    }

    let base_model = if !normal_tris.is_empty() {
        Some(create_visible_mesh(
            &normal_tris,
            context,
            if state.show_texture {
                state.current_texture.as_ref()
            } else {
                None
            },
        ))
    } else {
        None
    };
    let highlight_model = if !highlight_tris.is_empty() {
        Some(create_highlight_mesh(&highlight_tris, context))
    } else {
        None
    };

    (base_model, highlight_model)
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
    let mut state = init(&context);

    window.render_loop(move |mut frame_input| {
        let dt = frame_input.elapsed_time as f32 / 1000.0;
        let events = &mut frame_input.events;
        state.cfg = CONFIG.read().unwrap().clone();

        // Apply dynamic configuration updates
        state.ambient_light.intensity = state.cfg.ambient_light_intensity;
        state.ambient_light.color = state.cfg.ambient_light_color;
        state.spectator_glow.material.color = state.cfg.marker_color;
        state.third_person_glow.material.color = state.cfg.marker_color;
        state.camera_arrow_shaft.material.color = state.cfg.arrow_color;
        state.camera_arrow_head.material.color = state.cfg.arrow_color;

        // Synchronize branch limit with configuration changes
        if state.branch_limit > state.cfg.max_bsp_depth {
            state.branch_limit = state.cfg.max_bsp_depth;
        }
        if state.cfg.default_branch_limit != state.last_default_branch_limit {
            state.branch_limit = state.cfg.default_branch_limit.min(state.cfg.max_bsp_depth);
            state.last_default_branch_limit = state.cfg.default_branch_limit;
        }

        // Zkontroluj, zda background thread dokončil stavbu BSP stromu
        if let Ok(message) = state.rx.try_recv() {
            match message {
                Message::InitialTree(tree) => {
                    state.total_stats.total_nodes = tree.count_nodes();
                    state.bsp_root_full = Some(tree);
                    let next_id = AtomicUsize::new(0);
                    state.bsp_root_preview =
                        Some(build_bsp(&state.current_triangles, 0, state.branch_limit, &next_id));
                    info!("✅ BSP tree loaded into GUI");
                }
                Message::NewFile {
                    cpu_mesh: new_cpu_mesh,
                    texture: new_tex,
                    file_name,
                } => {
                    state.current_cpu_mesh = new_cpu_mesh;
                    state.loaded_file_name = file_name;
                    state.file_loading = false;
                    state.current_triangles = cpu_mesh_to_triangles(&state.current_cpu_mesh);
                    state.current_texture = new_tex.map(|t| Texture2DRef::from_cpu_texture(&context, &t));
                    if state.current_texture.is_some() {
                        state.show_texture = true;
                    }
                    let next_id = AtomicUsize::new(0);
                    state.bsp_root_full = Some(build_bsp(
                        &state.current_triangles,
                        0,
                        state.cfg.max_bsp_depth,
                        &next_id,
                    ));
                    state.total_stats.total_nodes = state.bsp_root_full.as_ref().unwrap().count_nodes();
                    let next_id = AtomicUsize::new(0);
                    state.bsp_root_preview =
                        Some(build_bsp(&state.current_triangles, 0, state.branch_limit, &next_id));
                    state.total_stats.total_triangles = state.current_triangles.len() as u32;
                    info!("✅ New model and BSP tree loaded");
                }
            }
        }

        // Vytvoření frustumu kamery pro view-culling
        let camera_obj = state.cam.cam(frame_input.viewport);

        handle_input(&mut state, events, &context, &camera_obj);

        // Použij správnou pozici pozorovatele pro traverzování BSP stromu
        let observer_position = match state.mode {
            CamMode::Spectator => state.cam.pos, // V režimu Spectator používáme pozici kamery
            CamMode::ThirdPerson => state.spectator_state.pos, // V režimu ThirdPerson používáme pozici Spectator kamery
        };

        // V třetí osobě vytvoříme frustum z pozice pozorovatele
        let frustum = if state.mode == CamMode::ThirdPerson {
            // Vytvoříme kameru z pozice spectator
            let spectator_camera = Camera::new_perspective(
                frame_input.viewport,
                state.spectator_state.pos,
                state.spectator_state.pos + state.spectator_state.dir(),
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
            total_nodes: if state.limit_culling {
                state.bsp_root_preview
                    .as_ref()
                    .map(|n| n.count_nodes())
                    .unwrap_or(0)
            } else {
                state.total_stats.total_nodes
            },
            total_triangles: state.total_stats.total_triangles,
            ..Default::default()
        };

        let visible_triangles = if state.disable_culling {
            current_stats.camera.nodes_visited = current_stats.total_nodes;
            current_stats.camera.triangles_rendered = state.current_triangles.len() as u32;
            current_stats.camera.vertices_rendered = state.current_triangles.len() as u32 * 3;
            state.current_triangles.clone()
        } else {
            let mut tris = Vec::new();
            let root = if state.limit_culling {
                state.bsp_root_preview.as_ref()
            } else {
                state.bsp_root_full.as_ref()
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

        current_stats.nodes_to_selected = state.last_pick_visits;

        update_gui(&mut state, &frame_input, &context, &current_stats);

        if state.branch_limit != state.last_branch_limit {
            // Update the default branch limit in the global configuration so that
            // the slider in the configuration window stays in sync with the one in
            // the left control panel.
            CONFIG.write().unwrap().default_branch_limit = state.branch_limit;
            state.last_default_branch_limit = state.branch_limit;

            let next_id = AtomicUsize::new(0);
            state.bsp_root_preview = Some(build_bsp(&state.current_triangles, 0, state.branch_limit, &next_id));
            state.selected_node = None;
            state.last_branch_limit = state.branch_limit;
        }

        // Reload configuration so input handling reflects any changes made in the GUI
        state.cfg = CONFIG.read().unwrap().clone();

        let (base_model, highlight_model) =
            process_visible_triangles(visible_triangles, &state, &context);

        let arrow_tip_position = update_camera(&mut state, dt);

        // Vykreslení labelu s pozicí špičky šipky
        let render_cam = state.cam.cam(frame_input.viewport);
        if state.show_spectator_marker && state.mode == CamMode::ThirdPerson {
            if let Some(tip_pos) = arrow_tip_position {
                if let Some([sx, sy]) = world_to_screen(&render_cam, tip_pos, frame_input.viewport)
                {
                    let txt = format!("({:.2}, {:.2}, {:.2})", tip_pos.x, tip_pos.y, tip_pos.z);
                    let painter = state.gui.context().layer_painter(egui::LayerId::new(
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

        render_scene(&frame_input.screen(), &context, &state, base_model.as_ref(), Option::from(&highlight_model));
        let _ = state.gui.render();
        FrameOutput::default()
    });

    Ok(())
}

fn update_camera_arrow(
    state: &CameraState,
    cfg: &Config,
    shaft: &mut Gm<Mesh, ColorMaterial>,
    head: &mut Gm<Mesh, ColorMaterial>,
    tip: &mut Gm<Mesh, ColorMaterial>,
) -> Vector3<f32> {
    let dir = state.dir();
    let base_pos = state.pos;
    let length = cfg.direction_ray_length;
    let head_len = length * 0.25;
    let shaft_len = length - head_len;
    let shaft_radius = cfg.direction_ray_thickness;
    let head_radius = shaft_radius * 2.0;
    let tip_radius = shaft_radius * 1.5;

    // Rotation from X-axis to dir
    let x_axis = Vector3::unit_x();
    let angle = x_axis.dot(dir).acos();
    let rotation_axis = x_axis.cross(dir).normalize();
    let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01 {
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

    // Arrow body
    let shaft_tr = Mat4::from_translation(base_pos);
    let shaft_scale = Mat4::from_nonuniform_scale(shaft_len, shaft_radius, shaft_radius);
    shaft.set_transformation(shaft_tr * rotation * shaft_scale);

    // Arrow head
    let head_pos = base_pos + dir * shaft_len;
    let head_tr = Mat4::from_translation(head_pos);
    let head_scale = Mat4::from_nonuniform_scale(head_len, head_radius, head_radius);
    head.set_transformation(head_tr * rotation * head_scale);

    // Arrow tip
    let tip_pos = base_pos + dir * length;
    let tip_tr = Mat4::from_translation(tip_pos);
    let tip_scale = Mat4::from_scale(tip_radius);
    tip.set_transformation(tip_tr * tip_scale);

    tip_pos
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
