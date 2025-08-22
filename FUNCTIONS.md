# Function Reference / Přehled funkcí

## src/main.rs

### `load_cpu_mesh` (line 60)
- EN: Loads a mesh from a GLTF/GLB file and returns a status message.
- CZ: Načte mesh ze souboru GLTF/GLB a vrátí stavovou zprávu.

### `load_gltf_with_gltf_crate` (line 107)
- EN: Parses a GLTF file using the `gltf` crate and builds a `CpuMesh`.
- CZ: Zpracuje soubor GLTF pomocí knihovny `gltf` a vytvoří `CpuMesh`.

### `process_node` (line 159)
- EN: Recursively walks GLTF nodes, collecting vertex data.
- CZ: Rekurzivně prochází uzly GLTF a sbírá data o vrcholech.

### `process_primitive` (line 208)
- EN: Extracts vertices and indices from a single GLTF primitive.
- CZ: Získá vrcholy a indexy z jedné GLTF primitivy.

### `create_visible_mesh` (line 281)
- EN: Creates a renderable mesh from a triangle list.
- CZ: Vytvoří vykreslitelný mesh ze seznamu trojúhelníků.

### `main` (line 330)
- EN: Application entry point; sets up window, loads models and renders the scene.
- CZ: Vstupní bod aplikace; nastaví okno, načte modely a vykreslí scénu.

### `quantized_center` (line 703 inside `main`)
- EN: Helper that quantizes a triangle's center for comparison.
- CZ: Pomocná funkce kvantizující střed trojúhelníku pro porovnání.

## src/camera.rs

### `FreeCamera::new` (line 19)
- EN: Creates a free camera at the given position.
- CZ: Vytvoří volnou kameru na zadané pozici.

### `FreeCamera::dir` (line 23)
- EN: Returns the forward direction vector of the camera.
- CZ: Vrátí směrový vektor kamery.

### `FreeCamera::right` (line 32)
- EN: Computes the camera's right vector.
- CZ: Vypočítá pravý vektor kamery.

### `FreeCamera::update_smooth` (line 36)
- EN: Updates position and orientation from input with smoothing.
- CZ: Aktualizuje pozici a orientaci podle vstupu s vyhlazením.

### `FreeCamera::cam` (line 64)
- EN: Builds a perspective `Camera` object from the current state.
- CZ: Vytvoří perspektivní objekt `Camera` z aktuálního stavu.

### `CamMode` (enum) – modes Spectator/ThirdPerson.

### `SwitchDelay::new` (line 89)
- EN: Constructs a switch delay timer with cooldown.
- CZ: Vytvoří časovač prodlevy přepnutí s čekací dobou.

### `SwitchDelay::can_switch` (line 92)
- EN: Checks if enough time passed to switch modes.
- CZ: Ověří, zda uplynula doba pro přepnutí režimu.

### `SwitchDelay::record_switch` (line 95)
- EN: Records the moment of a mode switch.
- CZ: Zapíše čas provedeného přepnutí režimu.

### `CameraState::new` (line 109)
- EN: Builds default camera state at position.
- CZ: Vytvoří výchozí stav kamery na pozici.

### `CameraState::from_camera` (line 113)
- EN: Captures state from an existing `FreeCamera`.
- CZ: Získá stav ze stávající `FreeCamera`.

### `CameraState::apply_to_camera` (line 117)
- EN: Applies stored state to a `FreeCamera`.
- CZ: Aplikuje uložený stav na `FreeCamera`.

### `reset_camera_to_default` (line 125)
- EN: Resets a camera to a default position and speed.
- CZ: Resetuje kameru na výchozí pozici a rychlost.

## src/input.rs

### `KeyCode::from_event` (line 35)
- EN: Maps engine events to internal key codes.
- CZ: Mapuje události enginu na interní klávesové kódy.

### `InputManager::default` (line 66)
- EN: Creates an empty key state map.
- CZ: Vytvoří prázdnou mapu stavů kláves.

### `InputManager::new` (line 72)
- EN: Constructs a new input manager.
- CZ: Vytvoří nový správce vstupu.

### `InputManager::update_key_states` (line 76)
- EN: Updates pressed/released states from events.
- CZ: Aktualizuje stavy stisknutých/puštěných kláves z událostí.

### `InputManager::is_key_pressed` (line 92)
- EN: Tests whether a key is currently pressed.
- CZ: Ověří, zda je klávesa aktuálně stisknutá.

### `InputManager::get_movement_vector` (line 98)
- EN: Calculates normalized movement vector from key states.
- CZ: Vypočítá normalizovaný vektor pohybu ze stavů kláves.

### `InputManager::get_tilt_value` (line 129)
- EN: Returns left/right tilt input.
- CZ: Vrátí vstup pro naklonění doleva/doprava.

## src/gui.rs

### `layout_bsp_tree` (line 13)
- EN: Computes 2D positions for BSP nodes for plotting.
- CZ: Vypočítá 2D pozice BSP uzlů pro vykreslení.

### `draw_bsp_tree_window` (line 37)
- EN: Shows an interactive window with the BSP tree plot.
- CZ: Zobrazí interaktivní okno s grafem BSP stromu.

### `draw_left_panel` (line 119)
- EN: Builds the main GUI side panel with controls and stats.
- CZ: Vytvoří hlavní boční panel GUI s ovládáním a statistikami.

## src/bsp.rs

### `Plane::new` (line 27)
- EN: Creates a plane from normal and point.
- CZ: Vytvoří rovinu z normály a bodu.

### `Plane::side` (line 33)
- EN: Computes signed distance of a point from the plane.
- CZ: Spočítá podepsanou vzdálenost bodu od roviny.

### `Plane::classify` (line 37)
- EN: Classifies a point as front, back or on the plane.
- CZ: Určí, zda je bod před, za nebo na rovině.

### `BspNode::new_leaf` (line 71)
- EN: Builds a leaf node storing triangles.
- CZ: Vytvoří listový uzel uchovávající trojúhelníky.

### `BspNode::new_node` (line 84)
- EN: Creates an internal node with splitting plane and children.
- CZ: Vytvoří vnitřní uzel s dělící rovinou a potomky.

### `BspNode::count_nodes` (line 104)
- EN: Returns total number of nodes in subtree.
- CZ: Vrátí celkový počet uzlů v podstromu.

### `BspNode::count_triangles` (line 109)
- EN: Counts triangles contained in subtree.
- CZ: Spočítá trojúhelníky v podstromu.

### `BspNode::subtree_triangles` (line 115)
- EN: Reads cached triangle count of subtree.
- CZ: Vrátí uložený počet trojúhelníků v podstromu.

### `plane_from_triangle` (line 120)
- EN: Derives a plane from a triangle.
- CZ: Odvodí rovinu z trojúhelníku.

### `Vector3Ext::map2` (line 129/135)
- EN: Element-wise combines two vectors with a function.
- CZ: Prvkově kombinuje dva vektory pomocí funkce.

### `triangle_center` (line 143)
- EN: Computes the centroid of a triangle.
- CZ: Vypočítá těžiště trojúhelníku.

### `bucketed_sah_plane` (line 148)
- EN: Chooses a splitting plane using bucketed SAH heuristic.
- CZ: Zvolí dělící rovinu pomocí heuristiky bucketed SAH.

### `build_bsp` (line 293)
- EN: Recursively builds a BSP tree with unique node IDs.
- CZ: Rekurzivně vytvoří BSP strom s unikátními ID uzlů.

### `find_node` (line 334)
- EN: Finds a node by ID in the BSP tree.
- CZ: Najde uzel podle ID v BSP stromu.

### `find_node_path` (line 346)
- EN: Collects path from root to node with given ID.
- CZ: Získá cestu od kořene k uzlu s daným ID.

### `find_deepest_node_containing_point` (line 369)
- EN: Finds deepest node whose bounding box contains a point.
- CZ: Najde nejhlubší uzel, jehož obal obsahuje bod.

### `render_bsp_tree` (line 392)
- EN: Renders BSP tree structure in an `egui` UI.
- CZ: Vykreslí strukturu BSP stromu v rozhraní `egui`.

### `collect_triangles_in_subtree` (line 465)
- EN: Gathers all triangles from a subtree.
- CZ: Shromáždí všechny trojúhelníky z podstromu.

### `create_highlight_mesh` (line 480)
- EN: Builds a transparent mesh to highlight selected triangles.
- CZ: Vytvoří průhledný mesh pro zvýraznění vybraných trojúhelníků.

### `create_plane_mesh` (line 517)
- EN: Generates a mesh representing a splitting plane.
- CZ: Vygeneruje mesh představující dělící rovinu.

### `cpu_mesh_to_triangles` (line 579)
- EN: Converts `CpuMesh` geometry into a list of triangles.
- CZ: Převede geometrii `CpuMesh` na seznam trojúhelníků.

### `traverse_bsp_with_frustum` (line 689)
- EN: Traverses BSP tree with frustum culling and collects visible triangles.
- CZ: Prochází BSP strom s frustum cullingem a sbírá viditelné trojúhelníky.

### `create_material_and_model` (line 845)
- EN: Creates a default material and model from CPU mesh.
- CZ: Vytvoří výchozí materiál a model z CPU meshe.

### `create_glow_material` (line 862)
- EN: Produces a transparent material for glow effects.
- CZ: Vytvoří průhledný materiál pro glow efekty.

### `create_direction_material` (line 873)
- EN: Produces a transparent material for direction rays.
- CZ: Vytvoří průhledný materiál pro směrové paprsky.

### `create_direction_ray` (line 884)
- EN: Builds a cone mesh representing a camera direction ray.
- CZ: Vytvoří kuželový mesh představující směrový paprsek kamery.

### `BoundingBox::new_empty` (line 941)
- EN: Returns an empty bounding box.
- CZ: Vrátí prázdný obalový box.

### `BoundingBox::contains` (line 948)
- EN: Tests whether a point lies inside the box.
- CZ: Ověří, zda bod leží uvnitř boxu.

### `BoundingBox::from_triangle` (line 957)
- EN: Builds a bounding box around a single triangle.
- CZ: Vytvoří obalový box kolem jednoho trojúhelníku.

### `BoundingBox::from_triangles` (line 971)
- EN: Builds a bounding box around multiple triangles.
- CZ: Vytvoří obalový box kolem více trojúhelníků.

### `BoundingBox::encompass` (line 990)
- EN: Combines two bounding boxes into one encompassing box.
- CZ: Sloučí dva obalové boxy do jednoho.

### `BoundingBox::intersects_plane` (line 1006)
- EN: Checks if the box intersects a plane.
- CZ: Ověří, zda box protíná rovinu.

### `BoundingBox::surface_area` (line 1030)
- EN: Calculates surface area of the box.
- CZ: Vypočítá povrch boxu.

### `Frustum::from_camera` (line 1045)
- EN: Creates view frustum planes from a camera.
- CZ: Vytvoří roviny pohledového frustumu z kamery.

### `Frustum::as_vec4_array` (line 1117)
- EN: Converts frustum planes to an array of Vec4s.
- CZ: Převede roviny frustumu na pole `Vec4`.

### `test_frustum` (line 1164)
- EN: Helper constructing a test frustum (test module).
- CZ: Pomocná funkce vytvářející testovací frustum (modul testů).

### `frustum_culling_skips_outside_triangles` (line 1178)
- EN: Unit test verifying frustum culling.
- CZ: Jednotkový test ověřující frustum culling.

