# Function Reference

Quick index of key functions and methods in the project. Each entry links to the
source code so you can jump directly to the implementation.

## `src/camera.rs`
- [`FreeCamera::new`](src/camera.rs#L22-L29) – initialize a free camera with
  default parameters.
- [`FreeCamera::update_smooth`](src/camera.rs#L45-L71) – update position and
  orientation based on keyboard input.
- [`FreeCamera::cam`](src/camera.rs#L73-L83) – construct a `three_d` camera from
  the current state.
- [`SwitchDelay::can_switch`](src/camera.rs#L101-L103) – check if camera mode
  can be toggled.
- [`CameraState::new`](src/camera.rs#L117-L120) – snapshot of camera parameters.

## `src/bsp.rs`
- [`build_bsp`](src/bsp.rs#L293-L331) – construct the BSP tree and assign node
  IDs.
- [`find_node`](src/bsp.rs#L334-L342) – locate a node by its identifier.
- [`find_node_path`](src/bsp.rs#L344-L367) – collect the path from root to a
  specific node.
- [`find_deepest_node_containing_point`](src/bsp.rs#L369-L389) – find the deepest
  node that contains a given point.
- [`collect_triangles_in_subtree`](src/bsp.rs#L465-L477) – gather triangles from
  a subtree.
- [`create_highlight_mesh`](src/bsp.rs#L480-L514) – build a colored mesh for the
  selected triangles.
- [`create_plane_mesh`](src/bsp.rs#L517-L573) – generate a mesh representing the
  splitting plane of a node.
- [`cpu_mesh_to_triangles`](src/bsp.rs#L579-L686) – convert a `CpuMesh` to a
  triangle list.
- [`traverse_bsp_with_frustum`](src/bsp.rs#L689-L750) – traverse the BSP tree
  while performing frustum culling.

## `src/gui.rs`
- [`layout_bsp_tree`](src/gui.rs#L13-L35) – compute plot positions for BSP
  nodes.
- [`draw_left_panel`](src.gui.rs#L119-L143) – render the left control panel and
  handle UI interactions.

## `src/input.rs`
- [`InputManager::update_key_states`](src/input.rs#L76-L90) – update internal
  key state map from event list.
- [`InputManager::get_movement_vector`](src/input.rs#L98-L126) – translate key
  presses into a normalized movement vector.
- [`InputManager::get_tilt_value`](src/input.rs#L129-L139) – compute horizontal
  tilt from arrow keys.

## `src/main.rs`
- [`load_cpu_mesh`](src/main.rs#L61-L109) – load a GLTF/GLB file into a CPU
  mesh and optional texture with basic validation.
- [`load_gltf_with_gltf_crate`](src/main.rs#L111-L168) – helper using the
  `gltf` crate to parse meshes and textures.
