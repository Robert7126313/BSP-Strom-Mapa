//! Central place for user‑tunable parameters.
//! Values here can be quickly adjusted without digging
//! through the rest of the code base.

use cgmath::Vector3;
use three_d::Srgba;

// -----------------------------------------------------------------------------
// Colors & lighting
// -----------------------------------------------------------------------------

/// Background color of the main render target (RGB 0.0‑1.0).
/// Barva pozadí hlavního renderovacího cíle (RGB 0.0‑1.0).
pub const BG_COLOR: (f32, f32, f32) = (0.1, 0.1, 0.1);

/// Color used for the loaded model.
/// Barva načteného modelu.
pub const MODEL_COLOR: Srgba = Srgba::new(100, 150, 255, 255);

/// Color used to highlight the currently selected geometry.
/// Barva zvýraznění aktuálně vybrané geometrie.
pub const HIGHLIGHT_COLOR: Srgba = Srgba::new(255, 50, 50, 150);

/// Color of the optional splitting plane preview.
/// Barva volitelného náhledu dělící roviny.
pub const PLANE_COLOR: Srgba = Srgba::new(200, 200, 50, 128);

/// Color used for the spectator camera marker.
/// Barva značky spectator kamery.
pub const SPECTATOR_GLOW_COLOR: Srgba = Srgba::new(0, 255, 100, 200);

/// Color used for the third person camera marker.
/// Barva značky third person kamery.
pub const THIRD_PERSON_GLOW_COLOR: Srgba = Srgba::new(255, 100, 0, 200);

/// Color used for the camera direction indicator.
/// Barva paprsku indikujícího směr kamery.
pub const DIRECTION_RAY_COLOR: Srgba = Srgba::new(255, 255, 0, 200);

/// Intensity of the ambient light in the scene.
/// Intenzita ambientního světla ve scéně.
pub const AMBIENT_LIGHT_INTENSITY: f32 = 1.0;

/// Color of the ambient light in the scene.
/// Barva ambientního světla ve scéně.
pub const AMBIENT_LIGHT_COLOR: Srgba = Srgba::WHITE;

// -----------------------------------------------------------------------------
// BSP tree limits
// -----------------------------------------------------------------------------

/// Maximum allowed depth when building the BSP tree.
/// Maximální povolená hloubka při tvorbě BSP stromu.
pub const MAX_BSP_DEPTH: u32 = 16;

/// Default depth limit used when initially building the BSP tree.
/// Výchozí limit hloubky použitý při počáteční stavbě BSP stromu.
pub const DEFAULT_BRANCH_LIMIT: u32 = 7;

/// Minimum number of triangles inside a leaf before we stop splitting.
/// Minimální počet trojúhelníků v listu, než přestaneme dělit.
pub const MIN_TRIANGLES_PER_LEAF: usize = 20;

// -----------------------------------------------------------------------------
// Camera configuration
// -----------------------------------------------------------------------------

/// Default movement speed for the free camera (units per second).
/// Výchozí rychlost pohybu volné kamery (jednotky za sekundu).
pub const DEFAULT_CAMERA_SPEED: f32 = 4.0;

/// Base angular speed when turning the camera (radians per second).
/// Základní úhlová rychlost otáčení kamery (radiany za sekundu).
pub const DEFAULT_LOOK_SPEED: f32 = 2.0;

/// Maximum absolute pitch angle from the horizontal.
/// Maximální absolutní úhel naklonění od horizontu.
pub const PITCH_LIMIT: f32 = 1.5;

/// Field of view for the perspective camera in degrees.
/// Zorné pole perspektivní kamery ve stupních.
pub const DEFAULT_FOV_DEG: f32 = 60.0;

/// Near clipping plane distance.
/// Vzdálenost přední ořezové roviny.
pub const NEAR_PLANE: f32 = 0.1;

/// Far clipping plane distance.
/// Vzdálenost zadní ořezové roviny.
pub const FAR_PLANE: f32 = 1000.0;

/// Minimum seconds between camera mode switches.
/// Minimální čas v sekundách mezi přepnutími režimu kamery.
pub const CAMERA_SWITCH_COOLDOWN: f64 = 2.0;

/// Multiplicative factor used when adjusting camera speed.
/// Násobící faktor použitý při úpravě rychlosti kamery.
pub const SPEED_ADJUSTMENT_FACTOR: f32 = 1.2;

/// Default starting position for the spectator camera.
/// Výchozí startovní pozice spectator kamery.
pub const DEFAULT_SPECTATOR_POS: Vector3<f32> = Vector3::new(0.0, 2.0, 8.0);

/// Default starting position for the third‑person camera.
/// Výchozí startovní pozice third‑person kamery.
pub const DEFAULT_THIRD_PERSON_POS: Vector3<f32> = Vector3::new(5.0, 2.0, 8.0);

/// Uniform scale applied to camera marker spheres.
/// Jednotné měřítko aplikované na koule označující kamery.
pub const CAMERA_MARKER_SCALE: f32 = 0.2;

/// Radius of the cylindrical indicator showing camera direction.
/// Poloměr válcového indikátoru směru kamery.
pub const DIRECTION_RAY_THICKNESS: f32 = 0.05;

/// Length of the direction indicator ray.
/// Délka paprsku indikujícího směr.
pub const DIRECTION_RAY_LENGTH: f32 = 3.0;
