//! Central place for user‑tunable parameters.
//! Values here can be quickly adjusted without digging
//! through the rest of the code base.

use three_d::Srgba;

/// Background color of the main render target (RGB 0.0‑1.0).
pub const BG_COLOR: (f32, f32, f32) = (0.1, 0.1, 0.1);

/// Color used to highlight the currently selected geometry.
pub const HIGHLIGHT_COLOR: Srgba = Srgba::new(255, 50, 50, 150);

/// Color of the optional splitting plane preview.
pub const PLANE_COLOR: Srgba = Srgba::new(200, 200, 50, 128);

/// Maximum allowed depth when building the BSP tree.
pub const MAX_BSP_DEPTH: u32 = 16;

/// Minimum number of triangles inside a leaf before we stop splitting.
pub const MIN_TRIANGLES_PER_LEAF: usize = 20;
