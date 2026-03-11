//! GLTF/GLB mesh loading with basic validation and logging.

use anyhow::Result;
use log::{error, info, warn};
use std::path::Path;
use three_d::*;

/// Load a CpuMesh and optional texture from a GLTF/GLB file.
///
/// This function performs basic validation, such as checking if the file exists
/// and if its size is within a reasonable limit. It then attempts to load the
/// file using the `gltf` crate. If loading fails, it returns a default sphere
/// mesh and an error message.
pub fn load_cpu_mesh(path: &Path) -> (CpuMesh, Option<CpuTexture>, String) {
    info!("Attempting to load: {}", path.display());

    if !path.exists() {
        error!("File does not exist: {}", path.display());
        return (
            CpuMesh::sphere(32),
            None,
            format!("File does not exist: {}", path.display()),
        );
    }

    match std::fs::metadata(path) {
        Ok(metadata) => {
            info!("File size: {} bytes", metadata.len());
            if metadata.len() == 0 {
                return (CpuMesh::sphere(32), None, "File is empty".to_string());
            }
            if metadata.len() > 100_000_000 {
                return (
                    CpuMesh::sphere(32),
                    None,
                    "File too large (>100MB)".to_string(),
                );
            }
        }
        Err(e) => {
            error!("Failed to read file metadata: {}", e);
            return (CpuMesh::sphere(32), None, format!("Metadata error: {}", e));
        }
    }

    match load_gltf_with_gltf_crate(path) {
        Ok((mesh, texture)) => {
            info!("✓ GLTF loaded using gltf crate");
            (mesh, texture, "GLTF file loaded successfully".to_string())
        }
        Err(e) => {
            error!("Failed to load with gltf crate: {}", e);
            (
                CpuMesh::sphere(32),
                None,
                format!("Could not load GLTF: {}", e),
            )
        }
    }
}

/// Helper function to load a GLTF file using the `gltf` crate.
///
/// This function recursively processes the nodes in the GLTF scene, extracting
/// vertex positions, indices, and UV coordinates. It also extracts the first
/// texture it finds.
fn load_gltf_with_gltf_crate(path: &Path) -> Result<(CpuMesh, Option<CpuTexture>)> {
    info!("Loading GLTF via gltf crate...");
    let (document, buffers, images) = gltf::import(path)?;

    info!(
        "GLTF document loaded: scenes {} nodes {} meshes {} materials {}",
        document.scenes().count(),
        document.nodes().count(),
        document.meshes().count(),
        document.materials().count()
    );

    let mut all_positions = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_uvs = Vec::new();
    let mut vertex_offset = 0u32;
    let mut texture: Option<CpuTexture> = None;

    for scene in document.scenes() {
        info!("Processing scene: {:?}", scene.name());
        for node in scene.nodes() {
            process_node(
                &node,
                &buffers,
                &images,
                &mut all_positions,
                &mut all_indices,
                &mut all_uvs,
                &mut vertex_offset,
                &mut texture,
                cgmath::Matrix4::identity(),
            )?;
        }
    }

    if all_positions.is_empty() {
        anyhow::bail!("No vertex positions found in GLTF");
    }

    info!(
        "Loaded {} vertices and {} indices",
        all_positions.len(),
        all_indices.len()
    );

    let mesh = CpuMesh {
        positions: Positions::F32(all_positions),
        indices: if all_indices.is_empty() {
            Indices::None
        } else {
            Indices::U32(all_indices)
        },
        uvs: if all_uvs.is_empty() {
            None
        } else {
            Some(all_uvs)
        },
        ..Default::default()
    };
    Ok((mesh, texture))
}

/// Recursively processes a node in the GLTF scene.
fn process_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    all_positions: &mut Vec<Vec3>,
    all_indices: &mut Vec<u32>,
    all_uvs: &mut Vec<Vec2>,
    vertex_offset: &mut u32,
    texture: &mut Option<CpuTexture>,
    parent_transform: cgmath::Matrix4<f32>,
) -> Result<()> {
    let transform_matrix = cgmath::Matrix4::from(node.transform().matrix());
    let current_transform = parent_transform * transform_matrix;
    info!("Processing node: {:?}", node.name());

    if let Some(mesh) = node.mesh() {
        info!(
            "Processing mesh: {:?} with {} primitives",
            mesh.name(),
            mesh.primitives().count()
        );
        for primitive in mesh.primitives() {
            process_primitive(
                &primitive,
                buffers,
                images,
                all_positions,
                all_indices,
                all_uvs,
                vertex_offset,
                texture,
                current_transform,
            )?;
        }
    }

    for child in node.children() {
        process_node(
            &child,
            buffers,
            images,
            all_positions,
            all_indices,
            all_uvs,
            vertex_offset,
            texture,
            current_transform,
        )?;
    }
    Ok(())
}

/// Processes a primitive in a GLTF mesh.
fn process_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    all_positions: &mut Vec<Vec3>,
    all_indices: &mut Vec<u32>,
    all_uvs: &mut Vec<Vec2>,
    vertex_offset: &mut u32,
    texture: &mut Option<CpuTexture>,
    transform: cgmath::Matrix4<f32>,
) -> Result<()> {
    info!("Processing primitive with mode: {:?}", primitive.mode());
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        warn!("Skipping primitive - not triangles");
        return Ok(());
    }

    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    if let Some(positions) = reader.read_positions() {
        let start_vertex_count = all_positions.len();
        for position in positions {
            let pos = cgmath::Vector4::new(position[0], position[1], position[2], 1.0);
            let transformed = transform * pos;
            all_positions.push(Vec3::new(transformed.x, transformed.y, transformed.z));
        }
        let vertex_count = all_positions.len() - start_vertex_count;
        info!("Added {} vertices", vertex_count);

        if let Some(tex) = reader.read_tex_coords(0) {
            for tc in tex.into_f32() {
                all_uvs.push(vec2(tc[0], 1.0 - tc[1]));
            }
        } else {
            all_uvs.extend(std::iter::repeat(vec2(0.0, 0.0)).take(vertex_count));
        }

        if let Some(indices) = reader.read_indices() {
            match indices {
                gltf::mesh::util::ReadIndices::U8(iter) => {
                    for idx in iter {
                        all_indices.push(idx as u32 + *vertex_offset);
                    }
                }
                gltf::mesh::util::ReadIndices::U16(iter) => {
                    for idx in iter {
                        all_indices.push(idx as u32 + *vertex_offset);
                    }
                }
                gltf::mesh::util::ReadIndices::U32(iter) => {
                    for idx in iter {
                        all_indices.push(idx + *vertex_offset);
                    }
                }
            }
            info!("Added {} indices", all_indices.len());
        } else {
            for i in (0..vertex_count).step_by(3) {
                if i + 2 < vertex_count {
                    all_indices.push(*vertex_offset + i as u32);
                    all_indices.push(*vertex_offset + i as u32 + 1);
                    all_indices.push(*vertex_offset + i as u32 + 2);
                }
            }
            info!("Generated {} sequential indices", (vertex_count / 3) * 3);
        }
        *vertex_offset += vertex_count as u32;
    } else {
        warn!("Primitive has no vertex positions");
    }

    if texture.is_none() {
        if let Some(info) = primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_texture()
        {
            let tex = info.texture();
            let image = &images[tex.source().index()];
            let data = match image.format {
                gltf::image::Format::R8G8B8A8 => TextureData::RgbaU8(
                    image
                        .pixels
                        .chunks(4)
                        .map(|c| [c[0], c[1], c[2], c[3]])
                        .collect(),
                ),
                gltf::image::Format::R8G8B8 => {
                    TextureData::RgbU8(image.pixels.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
                }
                _ => TextureData::RgbaU8(Vec::new()),
            };
            if !matches!(data, TextureData::RgbaU8(ref v) if v.is_empty()) {
                *texture = Some(CpuTexture {
                    name: "embedded".into(),
                    data,
                    width: image.width,
                    height: image.height,
                    ..Default::default()
                });
            }
        }
    }
    Ok(())
}
