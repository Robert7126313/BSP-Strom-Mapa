//! Central place for user-tunable parameters.
//! Values here can be adjusted at runtime through the configuration window.

use cgmath::Vector3;
use egui::Color32;
use once_cell::sync::Lazy;
use std::f32::consts::FRAC_PI_2;
use std::sync::RwLock;
use three_d::Srgba;

use crate::lang::Language;

/// Collection of runtime configuration options.
#[derive(Clone)]
pub struct Config {
    // ---------------------------------------------------------------------
    // General
    // ---------------------------------------------------------------------
    /// The UI language.
    pub language: Language,

    // ---------------------------------------------------------------------
    // Colors & lighting
    // ---------------------------------------------------------------------
    /// The background color of the 3D view.
    pub bg_color: [f32; 3],
    /// The color of the 3D model.
    pub model_color: Srgba,
    /// The color of highlighted triangles.
    pub highlight_color: Srgba,
    /// The color of the splitting plane visualization.
    pub splitting_plane_color: Srgba,
    /// The color of the camera position markers.
    pub marker_color: Srgba,
    /// The color of the camera direction arrow.
    pub arrow_color: Srgba,
    /// The intensity of the ambient light.
    pub ambient_light_intensity: f32,
    /// The color of the ambient light.
    pub ambient_light_color: Srgba,

    // ---------------------------------------------------------------------
    // BSP tree visualization
    // ---------------------------------------------------------------------
    /// The font size for the BSP tree view.
    pub bsp_tree_text_size: f32,
    /// The color of the path to the selected node in the BSP tree view.
    pub bsp_tree_path_color: Color32,
    /// The color of the selected node in the BSP tree view.
    pub bsp_tree_selected_color: Color32,

    // ---------------------------------------------------------------------
    // BSP tree limits
    // ---------------------------------------------------------------------
    /// The maximum depth of the BSP tree.
    pub max_bsp_depth: u32,
    /// The default branch limit for the BSP tree preview.
    pub default_branch_limit: u32,
    /// The minimum number of triangles per leaf node.
    pub min_triangles_per_leaf: usize,

    // ---------------------------------------------------------------------
    // Camera configuration
    // ---------------------------------------------------------------------
    /// The movement speed of the camera.
    pub camera_speed: f32,
    /// The look speed of the camera.
    pub look_speed: f32,
    /// The default field of view in degrees.
    pub default_fov_deg: f32,
    /// The near clipping plane.
    pub near_plane: f32,
    /// The far clipping plane.
    pub far_plane: f32,
    /// The default position of the spectator camera.
    pub default_spectator_pos: Vector3<f32>,
    /// The default position of the third-person camera.
    pub default_third_person_pos: Vector3<f32>,
    /// The default yaw of the spectator camera.
    pub default_spectator_yaw: f32,
    /// The default pitch of the spectator camera.
    pub default_spectator_pitch: f32,
    /// The default yaw of the third-person camera.
    pub default_third_person_yaw: f32,
    /// The default pitch of the third-person camera.
    pub default_third_person_pitch: f32,
    /// The scale of the camera position markers.
    pub camera_marker_scale: f32,
    /// The thickness of the camera direction ray.
    pub direction_ray_thickness: f32,
    /// The length of the camera direction ray.
    pub direction_ray_length: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            language: Language::English,
            bg_color: [0.1, 0.1, 0.1],
            model_color: Srgba::new(100, 150, 255, 255),
            highlight_color: Srgba::new(255, 50, 50, 150),
            splitting_plane_color: Srgba::new(50, 50, 255, 150),
            marker_color: Srgba::new(50, 255, 50, 150),
            arrow_color: Srgba::new(255, 255, 50, 150),
            ambient_light_intensity: 1.0,
            ambient_light_color: Srgba::WHITE,
            bsp_tree_text_size: 16.0,
            bsp_tree_path_color: Color32::from_rgb(255, 200, 0),
            bsp_tree_selected_color: Color32::YELLOW,
            max_bsp_depth: 16,
            default_branch_limit: 7,
            min_triangles_per_leaf: 20,
            camera_speed: 4.0,
            look_speed: 2.0,
            default_fov_deg: 60.0,
            near_plane: 0.1,
            far_plane: 1000.0,
            default_spectator_pos: Vector3::new(0.0, 2.0, 8.0),
            default_third_person_pos: Vector3::new(5.0, 2.0, 8.0),
            default_spectator_yaw: -FRAC_PI_2,
            default_spectator_pitch: 0.0,
            default_third_person_yaw: -FRAC_PI_2,
            default_third_person_pitch: 0.0,
            camera_marker_scale: 0.2,
            direction_ray_thickness: 0.05,
            direction_ray_length: 3.0,
        }
    }
}

/// Global mutable configuration accessible across modules.
pub static CONFIG: Lazy<RwLock<Config>> = Lazy::new(|| RwLock::new(Config::default()));
