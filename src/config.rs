//! Central place for user‑tunable parameters.
//! Values here can be quickly adjusted without digging
//! through the rest of the code base.
//!
//! Centrální místo pro parametry, které může uživatel snadno měnit
//! bez nutnosti procházet zbytek kódu.

use three_d::Srgba;

/// Background color of the main render target (RGB 0.0‑1.0).
/// Barva pozadí hlavního render targetu (RGB 0.0‑1.0).
pub const BG_COLOR: (f32, f32, f32) = (0.1, 0.1, 0.1);

/// Color used for the loaded model.
/// Barva nahraného modelu.
pub const MODEL_COLOR: Srgba = Srgba::new(100, 150, 255, 255);

/// Color used to highlight the currently selected geometry.
/// Barva zvýraznění aktuálně vybraných trojúhelníků.
pub const HIGHLIGHT_COLOR: Srgba = Srgba::new(255, 50, 50, 150);

/// Color of the optional splitting plane preview.
/// Barva náhledové dělicí roviny.
pub const PLANE_COLOR: Srgba = Srgba::new(200, 200, 50, 128);

/// Color of the glowing marker for the Spectator camera.
/// Barva svítící značky pro kameru v režimu Spectator.
pub const SPECTATOR_GLOW_COLOR: Srgba = Srgba::new(0, 255, 100, 200);

/// Color of the marker indicating the third‑person camera.
/// Barva značky kamery ve třetí osobě.
pub const THIRD_PERSON_GLOW_COLOR: Srgba = Srgba::new(255, 100, 0, 200);

/// Color of the ray that shows the camera's forward direction.
/// Barva paprsku znázorňujícího směr kamery.
pub const DIRECTION_RAY_COLOR: Srgba = Srgba::new(255, 255, 0, 200);

/// Intensity multiplier for ambient light (higher = brighter scene).
/// Koeficient intenzity ambientního světla (vyšší = světlejší scéna).
pub const AMBIENT_LIGHT_INTENSITY: f32 = 1.0;

/// Color tint of the ambient light.
/// Barva (odstín) ambientního světla.
pub const AMBIENT_LIGHT_COLOR: Srgba = Srgba::WHITE;

/// Maximum allowed depth when building the BSP tree.
/// Maximální povolená hloubka při stavbě BSP stromu.
pub const MAX_BSP_DEPTH: u32 = 16;

/// Default depth limit used when initially building the BSP tree.
/// Výchozí limit hloubky při počáteční stavbě BSP stromu.
pub const DEFAULT_BRANCH_LIMIT: u32 = 7;

/// Minimum number of triangles inside a leaf before we stop splitting.
/// Minimální počet trojúhelníků v listu, než přestaneme dělit.
pub const MIN_TRIANGLES_PER_LEAF: usize = 20;
