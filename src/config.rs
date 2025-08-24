//! Central place for user‑tunable parameters.
//! Values here can be quickly adjusted without digging
//! through the rest of the code base.

use three_d::Srgba;

/// Background color of the main render target (RGB 0.0‑1.0).
pub const BG_COLOR: (f32, f32, f32) = (0.1, 0.1, 0.1);

/// Color used for the loaded model.
pub const MODEL_COLOR: Srgba = Srgba::new(100, 150, 255, 255);

/// Color used to highlight the currently selected geometry.
pub const HIGHLIGHT_COLOR: Srgba = Srgba::new(255, 50, 50, 150);

/// Color of the optional splitting plane preview.
pub const PLANE_COLOR: Srgba = Srgba::new(200, 200, 50, 128);

/// Color used for the spectator camera marker.
pub const SPECTATOR_GLOW_COLOR: Srgba = Srgba::new(0, 255, 100, 200);

/// Color used for the third person camera marker.
pub const THIRD_PERSON_GLOW_COLOR: Srgba = Srgba::new(255, 100, 0, 200);

/// Color used for the camera direction indicator.
pub const DIRECTION_RAY_COLOR: Srgba = Srgba::new(255, 255, 0, 200);

/// Intensity of the ambient light in the scene.
pub const AMBIENT_LIGHT_INTENSITY: f32 = 1.0;

/// Color of the ambient light in the scene.
pub const AMBIENT_LIGHT_COLOR: Srgba = Srgba::WHITE;

/// Maximum allowed depth when building the BSP tree.
pub const MAX_BSP_DEPTH: u32 = 16;

/// Default depth limit used when initially building the BSP tree.
pub const DEFAULT_BRANCH_LIMIT: u32 = 7;

/// Minimum number of triangles inside a leaf before we stop splitting.
pub const MIN_TRIANGLES_PER_LEAF: usize = 20;
