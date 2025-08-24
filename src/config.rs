//! Central place for user‑tunable parameters.
//! Values here can be adjusted at runtime through the configuration window.

use cgmath::Vector3;
use egui::Color32;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use three_d::Srgba;

/// Collection of runtime configuration options.
#[derive(Clone)]
pub struct Config {
    // ---------------------------------------------------------------------
    // Colors & lighting
    // ---------------------------------------------------------------------
    pub bg_color: [f32; 3],
    pub model_color: Srgba,
    pub highlight_color: Srgba,
    pub plane_color: Srgba,
    pub spectator_glow_color: Srgba,
    pub third_person_glow_color: Srgba,
    pub direction_ray_color: Srgba,
    pub ambient_light_intensity: f32,
    pub ambient_light_color: Srgba,

    // ---------------------------------------------------------------------
    // BSP tree visualization
    // ---------------------------------------------------------------------
    pub bsp_tree_text_size: f32,
    pub bsp_tree_path_color: Color32,
    pub bsp_tree_selected_color: Color32,

    // ---------------------------------------------------------------------
    // BSP tree limits
    // ---------------------------------------------------------------------
    pub max_bsp_depth: u32,
    pub default_branch_limit: u32,
    pub min_triangles_per_leaf: usize,

    // ---------------------------------------------------------------------
    // Camera configuration
    // ---------------------------------------------------------------------
    pub default_camera_speed: f32,
    pub default_look_speed: f32,
    pub pitch_limit: f32,
    pub default_fov_deg: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub camera_switch_cooldown: f64,
    pub speed_adjustment_factor: f32,
    pub default_spectator_pos: Vector3<f32>,
    pub default_third_person_pos: Vector3<f32>,
    pub camera_marker_scale: f32,
    pub direction_ray_thickness: f32,
    pub direction_ray_length: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bg_color: [0.1, 0.1, 0.1],
            model_color: Srgba::new(100, 150, 255, 255),
            highlight_color: Srgba::new(255, 50, 50, 150),
            plane_color: Srgba::new(200, 200, 50, 128),
            spectator_glow_color: Srgba::new(0, 255, 100, 200),
            third_person_glow_color: Srgba::new(255, 100, 0, 200),
            direction_ray_color: Srgba::new(255, 255, 0, 200),
            ambient_light_intensity: 1.0,
            ambient_light_color: Srgba::WHITE,
            bsp_tree_text_size: 16.0,
            bsp_tree_path_color: Color32::from_rgb(255, 200, 0),
            bsp_tree_selected_color: Color32::YELLOW,
            max_bsp_depth: 16,
            default_branch_limit: 7,
            min_triangles_per_leaf: 20,
            default_camera_speed: 4.0,
            default_look_speed: 2.0,
            pitch_limit: 1.5,
            default_fov_deg: 60.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            camera_switch_cooldown: 2.0,
            speed_adjustment_factor: 1.2,
            default_spectator_pos: Vector3::new(0.0, 2.0, 8.0),
            default_third_person_pos: Vector3::new(5.0, 2.0, 8.0),
            camera_marker_scale: 0.2,
            direction_ray_thickness: 0.05,
            direction_ray_length: 3.0,
        }
    }
}

/// Global mutable configuration accessible across modules.
pub static CONFIG: Lazy<Mutex<Config>> = Lazy::new(|| Mutex::new(Config::default()));

