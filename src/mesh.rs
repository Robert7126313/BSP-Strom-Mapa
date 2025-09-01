// SPDX-License-Identifier: MIT
//! Mesh construction helpers extracted from `main` for clarity.

use rayon::prelude::*;
use three_d::*;

use crate::bsp::Triangle;
use crate::config::CONFIG;

/// Build a mesh from the provided triangles, applying the configured color and
/// optional texture. Transparency is handled automatically based on the model
/// color's alpha channel.
pub fn create_visible_mesh(
    triangles: &[Triangle],
    context: &Context,
    texture: Option<&Texture2DRef>,
) -> Gm<Mesh, PhysicalMaterial> {
    let triangles_count = triangles.len();
    let mut positions = vec![vec3(0.0, 0.0, 0.0); triangles_count * 3];
    let mut uvs = vec![vec2(0.0, 0.0); triangles_count * 3];

    positions
        .par_chunks_mut(3)
        .zip(uvs.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, (p_chunk, uv_chunk))| {
            let tri = &triangles[i];
            p_chunk[0] = vec3(tri.a.x, tri.a.y, tri.a.z);
            p_chunk[1] = vec3(tri.b.x, tri.b.y, tri.b.z);
            p_chunk[2] = vec3(tri.c.x, tri.c.y, tri.c.z);
            uv_chunk[0] = vec2(tri.uv_a.x, tri.uv_a.y);
            uv_chunk[1] = vec2(tri.uv_b.x, tri.uv_b.y);
            uv_chunk[2] = vec2(tri.uv_c.x, tri.uv_c.y);
        });

    let mut indices = vec![0u32; triangles_count * 3];
    indices
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, chunk)| {
            let base = i as u32 * 3;
            chunk[0] = base;
            chunk[1] = base + 1;
            chunk[2] = base + 2;
        });

    let visible_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        uvs: Some(uvs),
        ..Default::default()
    };

    let model_color = CONFIG.read().unwrap().model_color;
    let is_transparent = model_color.a < 255;
    let render_states = if is_transparent {
        RenderStates {
            blend: Blend::TRANSPARENCY,
            ..Default::default()
        }
    } else {
        RenderStates::default()
    };

    let material = PhysicalMaterial {
        albedo: model_color,
        albedo_texture: texture.cloned(),
        render_states,
        is_transparent,
        ..Default::default()
    };

    Gm::new(Mesh::new(context, &visible_mesh), material)
}
