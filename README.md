# BSP-Strom-Mapa

BSP-Strom-Mapa is a simple viewer written in Rust for visualising 3D models and
inspecting the Binary Space Partition (BSP) tree that is built for them.  It
uses the [three-d](https://github.com/asny/three-d) graphics library together
with `egui` for the user interface.

## Features

- Loads `GLTF`/`GLB` files and converts them to a simple triangle list.
- Builds a BSP tree in a background thread so the UI stays responsive.
- CPU-based frustum culling of triangles.
- Two camera modes – *Spectator* and *ThirdPerson* – with smooth controls.
- Interactive UI for exploring the BSP structure and basic statistics.

## Building

This project is managed with Cargo.  To compile it simply run

```bash
cargo build
```

For the best performance, run the viewer in release mode:

```bash
cargo run --release
```

The binary opens a window showing the loaded scene.  On the first start the
example model from `assets/model.glb` is loaded.  New models can be loaded via
the UI panel on the left.

## Controls

The viewer can be controlled either with the keyboard or through the left panel.
Below is a list of the most important keys:

| Key                 | Action                                   |
|---------------------|-------------------------------------------|
| **W/A/S/D**         | Move forward/left/back/right              |
| **Space / C**       | Move up/down                              |
| **Arrow keys**      | Look around                               |
| **PageUp/PageDown** | Increase/decrease movement speed          |
| **F / G**           | Switch between Spectator and ThirdPerson  |
| **Home**            | Reset the current camera position         |

The left panel also exposes a checkbox to disable culling and options to view
the BSP tree.  When a node is selected, its splitting plane and triangles are
highlighted in the 3D view.

## Configuration

User-tunable parameters are collected in [`src/config.rs`](src/config.rs). Colors
and lighting can be customized by editing the constants in this file, e.g.:

- `HIGHLIGHT_COLOR` – accent color used for selections, camera markers and the
  direction indicator.
- `AMBIENT_LIGHT_INTENSITY` and `AMBIENT_LIGHT_COLOR` – ambient light settings.
- `DEFAULT_FOV_DEG` – default field of view for the camera.
- `camera_speed` and `look_speed` – movement speed and turning sensitivity.

