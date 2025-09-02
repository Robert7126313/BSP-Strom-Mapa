# Function Reference

Quick index of key functions and methods in the project. Each entry links to the
source code so you can jump directly to the implementation.

## `src/camera.rs`
- [`FreeCamera::new`](src/camera.rs#L19-L27) – initialize a free camera with
  default parameters.
- [`FreeCamera::update_smooth`](src/camera.rs#L43-L72) – update position and
  orientation based on keyboard input.
- [`FreeCamera::cam`](src/camera.rs#L74-L85) – construct a `three_d` camera from
  the current state.
- [`CameraState::new`](src/camera.rs#L103-L111) – create a camera state from
  position and orientation.

## `src/bsp.rs`
- [`build_bsp`](src/bsp.rs#L250-L289) – construct the BSP tree and assign node
  IDs.
- [`find_node`](src/bsp.rs#L292-L300) – locate a node by its identifier.
- [`find_node_path`](src/bsp.rs#L304-L325) – collect the path from root to a
  specific node.
- [`find_deepest_node_containing_point`](src/bsp.rs#L327-L347) – find the deepest
  node that contains a given point.
- [`collect_triangles_in_subtree`](src/bsp.rs#L350-L360) – gather triangles from
  a subtree.
- [`create_highlight_mesh`](src/bsp.rs#L365-L403) – build a colored mesh for the
  selected triangles.
- [`create_plane_mesh`](src/bsp.rs#L406-L466) – generate a mesh representing the
  splitting plane of a node.
- [`cpu_mesh_to_triangles`](src/bsp.rs#L469-L559) – convert a `CpuMesh` to a
  triangle list.
- [`traverse_bsp_with_frustum`](src/bsp.rs#L562-L774) – traverse the BSP tree
  while performing frustum culling.
 
## `src/geometry.rs`
- [`BoundingBox::from_triangles`](src/geometry.rs#L81-L98) – build an axis-aligned bounding box for a list of triangles.
- [`Frustum::from_camera`](src/geometry.rs#L138-L175) – extract six clipping planes from a `three_d` camera.
- [`triangle_center`](src/geometry.rs#L182-L184) – compute the centroid of a triangle.

## `src/gui/left_panel.rs`
- [`draw_left_panel`](src/gui/left_panel.rs#L14-L507) – render the left control panel and handle UI interactions.

## `src/gui/tree.rs`
- [`layout_bsp_tree`](src/gui/tree.rs#L14-L33) – compute plot positions for BSP nodes.
- [`draw_bsp_tree_window`](src/gui/tree.rs#L35-L113) – show an interactive BSP tree view.

## `src/gui/config.rs`
- [`draw_config_window`](src/gui/config.rs#L8-L302) – runtime configuration UI for colors and camera settings.

## `src/input.rs`
- [`InputManager::update_key_states`](src/input.rs#L78-L92) – update internal
  key state map from event list.
- [`InputManager::get_movement_vector`](src/input.rs#L100-L129) – translate key
  presses into a normalized movement vector.
- [`InputManager::get_tilt_value`](src/input.rs#L131-L142) – compute horizontal
  tilt from arrow keys.

## `src/loader.rs`
- [`load_cpu_mesh`](src/loader.rs#L8-L55) – load a GLTF/GLB file with basic validation and logging.
- [`load_gltf_with_gltf_crate`](src/loader.rs#L57-L113) – helper using the `gltf` crate to parse meshes and textures.

## `src/config.rs`
- [`Config`](src/config.rs#L13-L64) – collection of runtime configuration options.
- [`CONFIG`](src/config.rs#L102-L103) – global mutable configuration store.

## `src/lang.rs`
- [`Language`](src/lang.rs#L3-L7) – supported UI languages.
- [`tr`](src/lang.rs#L9-L14) – translate a pair of strings based on selected language.
