// SPDX-License-Identifier: MIT
//! Core BSP tree construction, traversal and visualization utilities.
use cgmath::{Vector2, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use three_d::*;

// Configuration values (colors and tree limits)
use crate::config::CONFIG;
use crate::geometry::triangle_center;
pub use crate::geometry::{BoundingBox, Frustum, Plane, Triangle};

// ---------------- BSP Implementation -------------------------------------- //

#[derive(Debug)]
pub struct BspNode {
    pub id: usize,
    pub plane: Option<Plane>,
    pub front: Option<Box<BspNode>>,
    pub back: Option<Box<BspNode>>,
    pub triangles: Vec<Triangle>,
    pub bounds: BoundingBox,
    node_count: u32,   // Cache the total number of nodes in this subtree
    subtree_tris: u32, // Cache the total number of triangles in this subtree
}

#[derive(Default)]
pub struct CameraStats {
    pub nodes_visited: u32,
    pub triangles_rendered: u32,
    pub vertices_rendered: u32,
}

#[derive(Default)]
pub struct BspStats {
    pub total_nodes: u32,
    pub total_triangles: u32,
    /// Statistics tied to the active camera.
    pub camera: CameraStats,
    /// Number of nodes visited during the last node selection.
    pub nodes_to_selected: u32,
}

impl BspNode {
    pub fn new_leaf(triangles: Vec<Triangle>, id: usize) -> Self {
        Self {
            id,
            plane: None,
            front: None,
            back: None,
            triangles: triangles.clone(),
            bounds: BoundingBox::from_triangles(&triangles),
            node_count: 1,                        // Leaf nodes count as 1
            subtree_tris: triangles.len() as u32, // Cache the triangle count
        }
    }

    fn new_node(plane: Plane, front: BspNode, back: BspNode, id: usize) -> Self {
        // Calculate the node count and triangle count before moving the nodes into boxes
        let total_count = 1 + front.node_count + back.node_count;
        let total_tris = front.subtree_tris + back.subtree_tris;

        // Compute combined bounding box before moving nodes into boxes
        let bounds = BoundingBox::encompass(&front.bounds, &back.bounds);

        Self {
            id,
            plane: Some(plane),
            front: Some(Box::new(front)),
            back: Some(Box::new(back)),
            triangles: Vec::new(),
            bounds,
            node_count: total_count,  // Use the cached count
            subtree_tris: total_tris, // Cache the total triangle count in subtree
        }
    }

    pub fn count_nodes(&self) -> u32 {
        1 + self.front.as_ref().map_or(0, |n| n.count_nodes())
            + self.back.as_ref().map_or(0, |n| n.count_nodes())
    }

    pub fn subtree_triangles(&self) -> u32 {
        self.subtree_tris
    }
}

// před funkci triangle_center přidáme trait extension pro Vector3
trait Vector3Ext<S> {
    fn map2<F>(self, other: Self, f: F) -> Self
    where
        F: Fn(S, S) -> S;
}

impl Vector3Ext<f32> for Vector3<f32> {
    fn map2<F>(self, other: Self, f: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        Vector3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}

/// Bucketed SAH for O(n + k) split – much faster than the original O(n²) SAH.
fn bucketed_sah_plane(tris: &[Triangle], buckets: usize) -> Plane {
    // 1) Parent bounding box and surface area
    let parent_bb = BoundingBox::from_triangles(tris);
    let parent_sa = parent_bb.surface_area();

    // 2) Compute centroids and extent (SoA)
    let mut mins = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut maxs = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut centroid_x = Vec::with_capacity(tris.len());
    let mut centroid_y = Vec::with_capacity(tris.len());
    let mut centroid_z = Vec::with_capacity(tris.len());
    for t in tris.iter() {
        let c = triangle_center(t);
        mins = mins.map2(c, |a, b| a.min(b));
        maxs = maxs.map2(c, |a, b| a.max(b));
        centroid_x.push(c.x);
        centroid_y.push(c.y);
        centroid_z.push(c.z);
    }

    let extent = maxs - mins;

    // Handle degenerate case – all centroids at the same position
    if extent.x < 1e-6 && extent.y < 1e-6 && extent.z < 1e-6 {
        // Fallback na střed parent BB
        let center = (parent_bb.min + parent_bb.max) * 0.5;
        return Plane::new(Vector3::unit_x(), center);
    }

    // 3) Pick axis with largest extent
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // If extent on chosen axis is nearly zero, use fallback
    if extent[axis] < 1e-6 {
        let center = (parent_bb.min + parent_bb.max) * 0.5;
        let normal = match axis {
            0 => Vector3::unit_x(),
            1 => Vector3::unit_y(),
            _ => Vector3::unit_z(),
        };
        return Plane::new(normal, center);
    }

    // 4) Prepare buckets
    #[derive(Clone)]
    struct Bucket {
        count: usize,
        bb: BoundingBox,
    }

    let mut buckets_data = vec![
        Bucket {
            count: 0,
            bb: BoundingBox::new_empty()
        };
        buckets
    ];

    // 5) Single pass: assign each triangle to a bucket
    let centroid_axis = match axis {
        0 => &centroid_x,
        1 => &centroid_y,
        _ => &centroid_z,
    };
    for (i, tri) in tris.iter().enumerate() {
        let c_axis = centroid_axis[i];
        let t = ((c_axis - mins[axis]) / extent[axis] * (buckets as f32))
            .floor()
            .clamp(0.0, (buckets - 1) as f32) as usize;
        let b = &mut buckets_data[t];
        b.count += 1;
        b.bb = BoundingBox::encompass(&b.bb, &BoundingBox::from_triangle(tri));
    }

    // 6) Prefix/suffix calculations
    let mut left_counts = vec![0; buckets];
    let mut left_bbs = vec![BoundingBox::new_empty(); buckets];
    let mut acc_bb = BoundingBox::new_empty();
    let mut acc_cnt = 0;

    for i in 0..buckets {
        acc_cnt += buckets_data[i].count;
        acc_bb = BoundingBox::encompass(&acc_bb, &buckets_data[i].bb);
        left_counts[i] = acc_cnt;
        left_bbs[i] = acc_bb.clone();
    }

    let mut right_counts = vec![0; buckets];
    let mut right_bbs = vec![BoundingBox::new_empty(); buckets];
    let mut acc_bb2 = BoundingBox::new_empty();
    let mut acc_cnt2 = 0;

    for j in (0..buckets).rev() {
        acc_cnt2 += buckets_data[j].count;
        acc_bb2 = BoundingBox::encompass(&acc_bb2, &buckets_data[j].bb);
        right_counts[j] = acc_cnt2;
        right_bbs[j] = acc_bb2.clone();
    }

    // 7) Find best split between buckets i and i+1
    let mut best_cost = f32::INFINITY;
    let mut best_i = 0;

    for i in 0..buckets - 1 {
        let nl = left_counts[i] as f32;
        let nr = right_counts[i + 1] as f32;
        if nl == 0.0 || nr == 0.0 {
            continue;
        }

        let cost = if parent_sa > 0.0 {
            (left_bbs[i].surface_area() / parent_sa) * nl
                + (right_bbs[i + 1].surface_area() / parent_sa) * nr
        } else {
            nl + nr
        };

        if cost < best_cost {
            best_cost = cost;
            best_i = i;
        }
    }

    // 8) Compute split position between buckets best_i and best_i+1
    let split_norm = (best_i as f32 + 1.0) / buckets as f32;
    let mut split_point = mins;
    split_point[axis] = mins[axis] + split_norm * extent[axis];

    // 9) Return splitting plane
    let normal = match axis {
        0 => Vector3::unit_x(),
        1 => Vector3::unit_y(),
        _ => Vector3::unit_z(),
    };

    Plane::new(normal, split_point)
}

// Upravená funkce build_bsp, která přiřazuje ID uzlům
pub fn build_bsp(
    triangles: &[Triangle],
    depth: u32,
    max_depth: u32,
    next_id: &AtomicUsize,
) -> BspNode {
    let my_id = next_id.fetch_add(1, Ordering::SeqCst);

    let min_tris = CONFIG.read().unwrap().min_triangles_per_leaf;
    if depth >= max_depth || triangles.len() <= min_tris {
        return BspNode::new_leaf(triangles.to_vec(), my_id);
    }

    if triangles.is_empty() {
        return BspNode::new_leaf(Vec::new(), my_id);
    }

    // Použij bucketed SAH algoritmus místo původního SAH - O(n + K) složitost
    let splitting_plane = bucketed_sah_plane(triangles, 16);

    // Paralelní klasifikace trojúhelníků pomocí Rayon
    let (front_triangles, back_triangles): (Vec<Triangle>, Vec<Triangle>) =
        triangles.par_iter().cloned().partition(|triangle| {
            let center = triangle_center(triangle);
            splitting_plane.classify(center) >= 0
        });

    // ✂️ degenerate split → leaf
    if front_triangles.is_empty() || back_triangles.is_empty() {
        return BspNode::new_leaf(triangles.to_vec(), my_id);
    }

    // Rekurzivní stavba podstromů v paralelních větvích
    let (front_node, back_node) = rayon::join(
        || build_bsp(&front_triangles, depth + 1, max_depth, next_id),
        || build_bsp(&back_triangles, depth + 1, max_depth, next_id),
    );

    BspNode::new_node(splitting_plane, front_node, back_node, my_id)
}

// Funkce pro rekurzivní hledání uzlu podle ID
pub fn find_node(node: &BspNode, id: usize) -> Option<&BspNode> {
    if node.id == id {
        return Some(node);
    }
    if let Some(found) = node.front.as_deref().and_then(|f| find_node(f, id)) {
        return Some(found);
    }
    node.back.as_deref().and_then(|b| find_node(b, id))
}

/// Fills `path` with pointers from the root down *to* the node with `target_id`.
/// Returns true if found.
pub fn find_node_path<'a>(
    node: &'a BspNode,
    target_id: usize,
    path: &mut Vec<&'a BspNode>,
) -> bool {
    if node.id == target_id {
        path.push(node);
        return true;
    }
    for child in node
        .front
        .as_deref()
        .into_iter()
        .chain(node.back.as_deref())
    {
        if find_node_path(child, target_id, path) {
            path.push(node);
            return true;
        }
    }
    false
}

pub fn find_deepest_node_containing_point<'a>(
    node: &'a BspNode,
    point: Vector3<f32>,
    visited: &mut u32,
) -> Option<&'a BspNode> {
    *visited += 1;
    if !node.bounds.contains(point) {
        return None;
    }
    if let Some(ref front) = node.front {
        if let Some(n) = find_deepest_node_containing_point(front, point, visited) {
            return Some(n);
        }
    }
    if let Some(ref back) = node.back {
        if let Some(n) = find_deepest_node_containing_point(back, point, visited) {
            return Some(n);
        }
    }
    Some(node)
}

// Funkce pro sběr všech trojúhelníků v podstromu
pub fn collect_triangles_in_subtree(node: &BspNode, triangles: &mut Vec<Triangle>) {
    // Iterativní varianta pro lepší výkon a menší stack usage
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        triangles.extend(n.triangles.iter().cloned());
        if let Some(ref front) = n.front {
            stack.push(front);
        }
        if let Some(ref back) = n.back {
            stack.push(back);
        }
    }
}

// Funkce pro vytvoření zvýrazněného meshe
pub fn create_highlight_mesh(
    triangles: &[Triangle],
    context: &Context,
) -> Gm<Mesh, PhysicalMaterial> {
    let positions: Vec<Vec3> = triangles
        .iter()
        .flat_map(|tri| {
            vec![
                vec3(tri.a.x, tri.a.y, tri.a.z),
                vec3(tri.b.x, tri.b.y, tri.b.z),
                vec3(tri.c.x, tri.c.y, tri.c.z),
            ]
        })
        .collect();

    let indices: Vec<u32> = (0..triangles.len() as u32)
        .flat_map(|i| {
            let base = i * 3;
            vec![base, base + 1, base + 2]
        })
        .collect();

    let cpu_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };

    let highlight_color = CONFIG.read().unwrap().highlight_color;
    let material = PhysicalMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: highlight_color,
            ..Default::default()
        },
    );

    Gm::new(Mesh::new(context, &cpu_mesh), material)
}

// Funkce pro vytvoření meshe dělící roviny
pub fn create_plane_mesh(
    plane: &Plane,
    bounds: &BoundingBox,
    context: &Context,
) -> Gm<Mesh, PhysicalMaterial> {
    // Vypočítáme střed obalového objemu
    let center = (bounds.min + bounds.max) * 0.5;

    // Potřebujeme najít dva vektory kolmé na normálu roviny
    // Nejprve najdeme libovolný vektor kolmý na normálu
    let n = plane.n;
    let u = if n.x.abs() < n.y.abs() && n.x.abs() < n.z.abs() {
        Vector3::new(0.0, -n.z, n.y).normalize()
    } else if n.y.abs() < n.z.abs() {
        Vector3::new(-n.z, 0.0, n.x).normalize()
    } else {
        Vector3::new(-n.y, n.x, 0.0).normalize()
    };

    // Druhý vektor kolmý na normálu a první vektor
    let v = n.cross(u).normalize();

    // Velikost roviny - vycházíme z velikosti obalového objemu
    let extent = (bounds.max - bounds.min).magnitude() * 0.6;

    // Vytvoříme čtyři rohy roviny
    let corners = [
        center + (u + v) * extent,
        center + (u - v) * extent,
        center + (-u - v) * extent,
        center + (-u + v) * extent,
    ];

    // Vytvoříme pozice a indexy pro mesh
    let positions = vec![
        vec3(corners[0].x, corners[0].y, corners[0].z),
        vec3(corners[1].x, corners[1].y, corners[1].z),
        vec3(corners[2].x, corners[2].y, corners[2].z),
        vec3(corners[3].x, corners[3].y, corners[3].z),
    ];

    // Dva trojúlníky pro čtyřúhelník
    let indices = vec![0, 1, 2, 2, 3, 0];

    let cpu_mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };

    let plane_color = CONFIG.read().unwrap().splitting_plane_color;
    let material = PhysicalMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: plane_color,
            ..Default::default()
        },
    );

    Gm::new(Mesh::new(context, &cpu_mesh), material)
}

// ---------------- Free‑fly kamera ---------------------------------------- //
pub fn cpu_mesh_to_triangles(mesh: &CpuMesh) -> Vec<Triangle> {
    // Získáme pozice vrcholů z meshe
    let positions = match &mesh.positions {
        Positions::F32(pos) => pos,
        _ => return Vec::new(), // Pokud nemáme F32 pozice, vrátíme prázdný vektor
    };
    let uvs = mesh.uvs.as_ref();

    match &mesh.indices {
        Indices::U32(indices) => {
            let tri_count = indices.len() / 3;
            let mut tris = Vec::with_capacity(tri_count);
            tris.par_extend(indices.par_chunks(3).filter_map(|chunk| {
                if chunk.len() < 3 {
                    return None;
                }
                let a_idx = chunk[0] as usize;
                let b_idx = chunk[1] as usize;
                let c_idx = chunk[2] as usize;

                if a_idx < positions.len() && b_idx < positions.len() && c_idx < positions.len() {
                    let default_uv = Vector2::new(0.0, 0.0);
                    Some(Triangle {
                        a: Vector3::new(positions[a_idx].x, positions[a_idx].y, positions[a_idx].z),
                        b: Vector3::new(positions[b_idx].x, positions[b_idx].y, positions[b_idx].z),
                        c: Vector3::new(positions[c_idx].x, positions[c_idx].y, positions[c_idx].z),
                        uv_a: uvs.map_or(default_uv, |u| u[a_idx]),
                        uv_b: uvs.map_or(default_uv, |u| u[b_idx]),
                        uv_c: uvs.map_or(default_uv, |u| u[c_idx]),
                    })
                } else {
                    None
                }
            }));
            tris
        }
        Indices::U16(indices) => {
            let tri_count = indices.len() / 3;
            let mut tris = Vec::with_capacity(tri_count);
            tris.par_extend(indices.par_chunks(3).filter_map(|chunk| {
                if chunk.len() < 3 {
                    return None;
                }
                let a_idx = chunk[0] as usize;
                let b_idx = chunk[1] as usize;
                let c_idx = chunk[2] as usize;

                if a_idx < positions.len() && b_idx < positions.len() && c_idx < positions.len() {
                    let default_uv = Vector2::new(0.0, 0.0);
                    Some(Triangle {
                        a: Vector3::new(positions[a_idx].x, positions[a_idx].y, positions[a_idx].z),
                        b: Vector3::new(positions[b_idx].x, positions[b_idx].y, positions[b_idx].z),
                        c: Vector3::new(positions[c_idx].x, positions[c_idx].y, positions[c_idx].z),
                        uv_a: uvs.map_or(default_uv, |u| u[a_idx]),
                        uv_b: uvs.map_or(default_uv, |u| u[b_idx]),
                        uv_c: uvs.map_or(default_uv, |u| u[c_idx]),
                    })
                } else {
                    None
                }
            }));
            tris
        }
        Indices::None => {
            let tri_count = positions.len() / 3;
            let mut tris = Vec::with_capacity(tri_count);
            let default_uv = Vector2::new(0.0, 0.0);
            tris.par_extend(
                positions
                    .par_chunks(3)
                    .enumerate()
                    .filter_map(|(i, chunk)| {
                        if chunk.len() < 3 {
                            return None;
                        }
                        let base = i * 3;
                        Some(Triangle {
                            a: Vector3::new(chunk[0].x, chunk[0].y, chunk[0].z),
                            b: Vector3::new(chunk[1].x, chunk[1].y, chunk[1].z),
                            c: Vector3::new(chunk[2].x, chunk[2].y, chunk[2].z),
                            uv_a: uvs.map_or(default_uv, |u| u[base]),
                            uv_b: uvs.map_or(default_uv, |u| u[base + 1]),
                            uv_c: uvs.map_or(default_uv, |u| u[base + 2]),
                        })
                    }),
            );
            tris
        }
        _ => Vec::new(), // Přidáno pro pokrytí všech případů
    }
}

// Funkce pro traverzování BSP stromu s frustum cullingem
pub fn traverse_bsp_with_frustum(
    node: &BspNode,
    observer_position: Vector3<f32>,
    frustum: &Frustum,
    stats: &mut CameraStats,
    visible_triangles: &mut Vec<Triangle>,
) {
    stats.nodes_visited += 1;

    // Nejprve zkontrolujeme, zda obalový objem uzlu protíná frustum
    let mut is_visible = true;

    // Testujeme proti všem rovinám frustumu
    for plane in &frustum.planes {
        if !node.bounds.intersects_plane(plane) {
            is_visible = false;
            break;
        }
    }

    if !is_visible {
        return;
    }

    // Pokud je list a nemá trojúhelníky, ukonči dříve
    if node.triangles.is_empty() && node.plane.is_none() {
        return;
    }

    // Přidáme trojúhelníky z tohoto uzlu do viditelných
    if !node.triangles.is_empty() {
        visible_triangles.extend(node.triangles.iter().cloned());
        let tris = node.triangles.len() as u32;
        stats.triangles_rendered += tris;
        stats.vertices_rendered += tris * 3;
    }

    // Pokud uzel není list, traverzujeme podstromy v závislosti na pozici pozorovatele
    if let Some(ref plane) = node.plane {
        let side = plane.classify(observer_position);

        if side >= 0 {
            // Pozorovatel je před rovinou, nejprve front, pak back
            match (node.front.as_ref(), node.back.as_ref()) {
                (Some(front), Some(back)) => {
                    let (mut front_stats, mut front_tris, mut back_stats, mut back_tris) = (
                        CameraStats::default(),
                        Vec::new(),
                        CameraStats::default(),
                        Vec::new(),
                    );
                    rayon::join(
                        || {
                            traverse_bsp_with_frustum(
                                front,
                                observer_position,
                                frustum,
                                &mut front_stats,
                                &mut front_tris,
                            );
                        },
                        || {
                            traverse_bsp_with_frustum(
                                back,
                                observer_position,
                                frustum,
                                &mut back_stats,
                                &mut back_tris,
                            );
                        },
                    );
                    stats.nodes_visited += front_stats.nodes_visited + back_stats.nodes_visited;
                    stats.triangles_rendered +=
                        front_stats.triangles_rendered + back_stats.triangles_rendered;
                    stats.vertices_rendered +=
                        front_stats.vertices_rendered + back_stats.vertices_rendered;
                    visible_triangles.extend(front_tris);
                    visible_triangles.extend(back_tris);
                }
                (Some(front), None) => {
                    traverse_bsp_with_frustum(
                        front,
                        observer_position,
                        frustum,
                        stats,
                        visible_triangles,
                    );
                }
                (None, Some(back)) => {
                    traverse_bsp_with_frustum(
                        back,
                        observer_position,
                        frustum,
                        stats,
                        visible_triangles,
                    );
                }
                _ => {}
            }
        } else {
            // Pozorovatel je za rovinou, nejprve back, pak front
            match (node.back.as_ref(), node.front.as_ref()) {
                (Some(back), Some(front)) => {
                    let (mut back_stats, mut back_tris, mut front_stats, mut front_tris) = (
                        CameraStats::default(),
                        Vec::new(),
                        CameraStats::default(),
                        Vec::new(),
                    );
                    rayon::join(
                        || {
                            traverse_bsp_with_frustum(
                                back,
                                observer_position,
                                frustum,
                                &mut back_stats,
                                &mut back_tris,
                            );
                        },
                        || {
                            traverse_bsp_with_frustum(
                                front,
                                observer_position,
                                frustum,
                                &mut front_stats,
                                &mut front_tris,
                            );
                        },
                    );
                    stats.nodes_visited += back_stats.nodes_visited + front_stats.nodes_visited;
                    stats.triangles_rendered +=
                        back_stats.triangles_rendered + front_stats.triangles_rendered;
                    stats.vertices_rendered +=
                        back_stats.vertices_rendered + front_stats.vertices_rendered;
                    visible_triangles.extend(back_tris);
                    visible_triangles.extend(front_tris);
                }
                (Some(back), None) => {
                    traverse_bsp_with_frustum(
                        back,
                        observer_position,
                        frustum,
                        stats,
                        visible_triangles,
                    );
                }
                (None, Some(front)) => {
                    traverse_bsp_with_frustum(
                        front,
                        observer_position,
                        frustum,
                        stats,
                        visible_triangles,
                    );
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Vector2, Vector3};

    fn test_frustum() -> Frustum {
        Frustum {
            planes: [
                Plane::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(-2.0, 0.0, 0.0)),
                Plane::new(Vector3::new(-1.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0)),
                Plane::new(Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -2.0, 0.0)),
                Plane::new(Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 2.0, 0.0)),
                Plane::new(Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, -2.0)),
                Plane::new(Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 0.0, 2.0)),
            ],
        }
    }

    #[test]
    fn frustum_culling_skips_outside_triangles() {
        let inside = Triangle {
            a: Vector3::new(1.0, 0.0, 0.0),
            b: Vector3::new(1.0, 1.0, 0.0),
            c: Vector3::new(1.0, 0.0, 1.0),
            uv_a: Vector2::new(0.0, 0.0),
            uv_b: Vector2::new(0.0, 0.0),
            uv_c: Vector2::new(0.0, 0.0),
        };

        let outside = Triangle {
            a: Vector3::new(-10.0, 0.0, 0.0),
            b: Vector3::new(-10.0, 1.0, 0.0),
            c: Vector3::new(-10.0, 0.0, 1.0),
            uv_a: Vector2::new(0.0, 0.0),
            uv_b: Vector2::new(0.0, 0.0),
            uv_c: Vector2::new(0.0, 0.0),
        };

        let front = BspNode::new_leaf(vec![inside.clone()], 2);
        let back = BspNode::new_leaf(vec![outside.clone()], 3);
        let root = BspNode::new_node(
            Plane::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
            front,
            back,
            1,
        );

        let frustum = test_frustum();
        let mut stats = CameraStats::default();
        let mut visible = Vec::new();

        traverse_bsp_with_frustum(
            &root,
            Vector3::new(0.0, 0.0, 0.0),
            &frustum,
            &mut stats,
            &mut visible,
        );

        assert_eq!(visible.len(), 1);
        assert!(visible.contains(&inside));
        assert!(!visible.contains(&outside));
    }
}
