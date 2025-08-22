// SPDX-License-Identifier: MIT
// BSP and geometry utilities
use cgmath::Vector3;
use rayon::prelude::*;
use three_d::*;

// Configuration values (colors and tree limits)
use crate::config::{HIGHLIGHT_COLOR, MIN_TRIANGLES_PER_LEAF, MODEL_COLOR, PLANE_COLOR};

// ---------------- BSP Implementation -------------------------------------- //

#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    pub a: Vector3<f32>,
    pub b: Vector3<f32>,
    pub c: Vector3<f32>,
}

#[derive(Clone, Debug)]
pub struct Plane {
    pub n: Vector3<f32>, // normála
    pub d: f32,          // vzdálenost od počátku (ax+by+cz+d=0)
}

impl Plane {
    fn new(n: Vector3<f32>, point: Vector3<f32>) -> Self {
        let n = n.normalize();
        let d = -n.dot(point);
        Self { n, d }
    }

    fn side(&self, point: Vector3<f32>) -> f32 {
        self.n.dot(point) + self.d
    }

    fn classify(&self, point: Vector3<f32>) -> i32 {
        let dist = self.side(point);
        const EPSILON: f32 = 1e-6;
        match dist {
            d if d > EPSILON => 1,   // front
            d if d < -EPSILON => -1, // back
            _ => 0,                  // on plane
        }
    }
}

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
pub struct BspStats {
    pub nodes_visited: u32,
    pub triangles_rendered: u32,
    pub total_nodes: u32,
    pub total_triangles: u32,
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

        // Nejprve vytvoříme společný obalový objem, než přesuneme hodnoty do boxů
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

    fn count_triangles(&self) -> u32 {
        self.triangles.len() as u32
            + self.front.as_ref().map_or(0, |n| n.count_triangles())
            + self.back.as_ref().map_or(0, |n| n.count_triangles())
    }
}

fn plane_from_triangle(tri: &Triangle) -> Plane {
    let edge1 = tri.b - tri.a;
    let edge2 = tri.c - tri.a;
    let normal = edge1.cross(edge2).normalize();
    Plane::new(normal, tri.a)
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

pub fn triangle_center(tri: &Triangle) -> Vector3<f32> {
    (tri.a + tri.b + tri.c) / 3.0
}

/// Bucketovaná SAH pro O(n + K) split - mnohem rychlejší než původní O(n²) SAH
fn bucketed_sah_plane(tris: &[Triangle], buckets: usize) -> Plane {
    // 1) Parent BB a SA
    let parent_bb = BoundingBox::from_triangles(tris);
    let parent_sa = parent_bb.surface_area();

    // 2) Spočti centroidy a rozsah (SoA)
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

    // Ošetření degenerovaného případu - všechny centroidy na stejném místě
    if extent.x < 1e-6 && extent.y < 1e-6 && extent.z < 1e-6 {
        // Fallback na střed parent BB
        let center = (parent_bb.min + parent_bb.max) * 0.5;
        return Plane::new(Vector3::unit_x(), center);
    }

    // 3) Výběr osy podle největší extent
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Pokud je extent na vybrané ose téměř nulový, použij fallback
    if extent[axis] < 1e-6 {
        let center = (parent_bb.min + parent_bb.max) * 0.5;
        let normal = match axis {
            0 => Vector3::unit_x(),
            1 => Vector3::unit_y(),
            _ => Vector3::unit_z(),
        };
        return Plane::new(normal, center);
    }

    // 4) Příprava bucketů
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

    // 5) Jediný průchod: přiřaď každý trojúhelník do bucketu
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

    // 6) Prefix/suffix výpočty
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

    // 7) Najdi nejlepší rozdělení mezi buckety i a i+1
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

    // 8) Vypočti pozici split-point mezi buckety best_i a best_i+1
    let split_norm = (best_i as f32 + 1.0) / buckets as f32;
    let mut split_point = mins;
    split_point[axis] = mins[axis] + split_norm * extent[axis];

    // 9) Vrať rovinu
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
    next_id: &mut usize,
) -> BspNode {
    let my_id = *next_id;
    *next_id += 1;

    if depth >= max_depth || triangles.len() <= MIN_TRIANGLES_PER_LEAF {
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

    // Rekurzivní stavba podstromů - use sequential processing to fix ID assignment
    let front_node = build_bsp(&front_triangles, depth + 1, max_depth, next_id);
    let back_node = build_bsp(&back_triangles, depth + 1, max_depth, next_id);

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
) -> Option<&'a BspNode> {
    if !node.bounds.contains(point) {
        return None;
    }
    if let Some(ref front) = node.front {
        if let Some(n) = find_deepest_node_containing_point(front, point) {
            return Some(n);
        }
    }
    if let Some(ref back) = node.back {
        if let Some(n) = find_deepest_node_containing_point(back, point) {
            return Some(n);
        }
    }
    Some(node)
}

// Funkce pro rekurzivní vykreslení stromu v UI a zpracování výběru uzlu
pub fn render_bsp_tree(ui: &mut egui::Ui, node: &BspNode, selected: &mut Option<usize>) {
    // build the label
    let is_leaf = node.plane.is_none();
    let local_tris = node.triangles.len();
    // total tris in this subtree (using cached value)
    let subtree_tris = node.subtree_tris as usize;
    // number of children nodes
    let child_count = node.front.as_ref().map_or(0, |n| n.node_count - 1)
        + node.back.as_ref().map_or(0, |n| n.node_count - 1);

    let is_selected = selected == &Some(node.id);
    let label = if is_leaf {
        // leaf: show only local triangles
        if is_selected {
            format!("🔸 Leaf {} ({} tris)", node.id, local_tris)
        } else {
            format!("Leaf {} ({} tris)", node.id, local_tris)
        }
    } else {
        // interior: show total subtree triangles
        if is_selected {
            format!(
                "🔸 Node {} ({} tris subtree, {} children)",
                node.id, subtree_tris, child_count
            )
        } else {
            format!(
                "Node {} ({} tris subtree, {} children)",
                node.id, subtree_tris, child_count
            )
        }
    };

    // collapsible header
    let header = egui::CollapsingHeader::new(label)
        .id_salt(node.id) // Aktualizace zastaralé metody id_source na id_salt
        .default_open(node.id == selected.unwrap_or(0)); // auto-open the selected node

    // draw the header
    let response = header.show(ui, |ui| {
        // small "select" button inside the collapsible content
        if ui
            .add(egui::SelectableLabel::new(
                selected == &Some(node.id),
                "▶ Select",
            ))
            .clicked()
        {
            *selected = Some(node.id);
        }

        // and recurse below *only if* this header is open
        if let Some(ref front) = node.front {
            ui.label("Front:");
            ui.indent("front", |ui| {
                render_bsp_tree(ui, front, selected);
            });
        }
        if let Some(ref back) = node.back {
            ui.label("Back:");
            ui.indent("back", |ui| {
                render_bsp_tree(ui, back, selected);
            });
        }
    });

    // if you want clicking the header itself to select:
    if response.header_response.clicked() {
        *selected = Some(node.id);
    }
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
pub fn create_highlight_mesh(triangles: &[Triangle], context: &Context) -> Gm<Mesh, ColorMaterial> {
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

    let material = ColorMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: HIGHLIGHT_COLOR, // configured highlight color
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
) -> Gm<Mesh, ColorMaterial> {
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

    let material = ColorMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: PLANE_COLOR, // configured plane color
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

    match &mesh.indices {
        Indices::U32(indices) => indices
            .par_chunks(3)
            .filter_map(|chunk| {
                if chunk.len() < 3 {
                    return None;
                }
                let a_idx = chunk[0] as usize;
                let b_idx = chunk[1] as usize;
                let c_idx = chunk[2] as usize;

                if a_idx < positions.len() && b_idx < positions.len() && c_idx < positions.len() {
                    Some(Triangle {
                        a: Vector3::new(positions[a_idx].x, positions[a_idx].y, positions[a_idx].z),
                        b: Vector3::new(positions[b_idx].x, positions[b_idx].y, positions[b_idx].z),
                        c: Vector3::new(positions[c_idx].x, positions[c_idx].y, positions[c_idx].z),
                    })
                } else {
                    None
                }
            })
            .collect(),
        Indices::U16(indices) => indices
            .par_chunks(3)
            .filter_map(|chunk| {
                if chunk.len() < 3 {
                    return None;
                }
                let a_idx = chunk[0] as usize;
                let b_idx = chunk[1] as usize;
                let c_idx = chunk[2] as usize;

                if a_idx < positions.len() && b_idx < positions.len() && c_idx < positions.len() {
                    Some(Triangle {
                        a: Vector3::new(positions[a_idx].x, positions[a_idx].y, positions[a_idx].z),
                        b: Vector3::new(positions[b_idx].x, positions[b_idx].y, positions[b_idx].z),
                        c: Vector3::new(positions[c_idx].x, positions[c_idx].y, positions[c_idx].z),
                    })
                } else {
                    None
                }
            })
            .collect(),
        Indices::None => positions
            .par_chunks(3)
            .filter_map(|chunk| {
                if chunk.len() < 3 {
                    return None;
                }
                Some(Triangle {
                    a: Vector3::new(chunk[0].x, chunk[0].y, chunk[0].z),
                    b: Vector3::new(chunk[1].x, chunk[1].y, chunk[1].z),
                    c: Vector3::new(chunk[2].x, chunk[2].y, chunk[2].z),
                })
            })
            .collect(),
        _ => Vec::new(), // Přidáno pro pokrytí všech případů
    }
}

// Funkce pro traverzování BSP stromu s frustum cullingem
pub fn traverse_bsp_with_frustum(
    node: &BspNode,
    observer_position: Vector3<f32>,
    frustum: &Frustum,
    stats: &mut BspStats,
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
        stats.triangles_rendered += node.triangles.len() as u32;
    }

    // Pokud uzel není list, traverzujeme podstromy v závislosti na pozici pozorovatele
    if let Some(ref plane) = node.plane {
        let side = plane.classify(observer_position);

        if side >= 0 {
            // Pozorovatel je před rovinou, nejprve front, pak back
            if let Some(ref front) = node.front {
                traverse_bsp_with_frustum(
                    front,
                    observer_position,
                    frustum,
                    stats,
                    visible_triangles,
                );
            }
            if let Some(ref back) = node.back {
                traverse_bsp_with_frustum(
                    back,
                    observer_position,
                    frustum,
                    stats,
                    visible_triangles,
                );
            }
        } else {
            // Pozorovatel je za rovinou, nejprve back, pak front
            if let Some(ref back) = node.back {
                traverse_bsp_with_frustum(
                    back,
                    observer_position,
                    frustum,
                    stats,
                    visible_triangles,
                );
            }
            if let Some(ref front) = node.front {
                traverse_bsp_with_frustum(
                    front,
                    observer_position,
                    frustum,
                    stats,
                    visible_triangles,
                );
            }
        }
    }
}

// Funkce pro vytvoření materiálu a modelu z CPU meshe
fn create_material_and_model(
    context: &Context,
    cpu_mesh: &CpuMesh,
) -> (ColorMaterial, Gm<Mesh, ColorMaterial>) {
    let material = ColorMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: MODEL_COLOR, // Modrá barva aby byl model viditelný
            ..Default::default()
        },
    );
    let model = Gm::new(Mesh::new(context, cpu_mesh), material.clone());

    (material, model)
}

// Funkce pro vytvoření glow materiálu
fn create_glow_material(context: &Context, color: Srgba, opacity: u8) -> ColorMaterial {
    ColorMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: Srgba::new(color.r, color.g, color.b, opacity),
            ..Default::default()
        },
    )
}

// Funkce pro vytvoření směrového materiálu
fn create_direction_material(context: &Context, color: Srgba, opacity: u8) -> ColorMaterial {
    ColorMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: Srgba::new(color.r, color.g, color.b, opacity),
            ..Default::default()
        },
    )
}

// Funkce pro vytvoření směrového paprsku
fn create_direction_ray(
    context: &Context,
    position: Vector3<f32>,
    direction: Vector3<f32>,
    color: Srgba,
    opacity: u8,
    length: f32,
) -> Gm<Mesh, ColorMaterial> {
    let direction_material = create_direction_material(context, color, opacity);
    let direction_mesh = CpuMesh::cone(16);
    let mut direction_ray = Gm::new(Mesh::new(context, &direction_mesh), direction_material);

    // Vypočítáme úhel mezi osou Y a směrovým vektorem
    let y_axis = Vector3::unit_y();
    let angle = y_axis.dot(direction).acos();

    // Vypočítáme osu rotace (kolmou na rovinu obsahující osu Y a směrový vektor)
    let rotation_axis = y_axis.cross(direction).normalize();

    // Vytvoření transformační matice pro válec
    let scale = 0.05; // tenký válec
    let translation = Mat4::from_translation(position);

    // Pokud je směrový vektor téměř rovnoběžný s osou Y, použijeme speciální zacházení
    let rotation = if angle.abs() < 0.01 || (std::f32::consts::PI - angle).abs() < 0.01 {
        // Pro případ kdy je vektor téměř rovnoběžný s osou Y
        if direction.y > 0.0 {
            Mat4::identity() // směr už je podél osy Y
        } else {
            // Rotace o 180° kolem osy X
            Mat4::from_angle_x(Rad(std::f32::consts::PI))
        }
    } else {
        // Normální případ - rotace kolem vypočtené osy
        Mat4::from_axis_angle(
            vec3(rotation_axis.x, rotation_axis.y, rotation_axis.z),
            Rad(angle),
        )
    };

    // Měřítko - válec - válec je standardně výšky 2.0, chceme jej natáhnout na délku `length`
    // a zúžit na šířku `scale`
    let scaling = Mat4::from_nonuniform_scale(scale, length / 2.0, scale);

    // Aplikujeme transformace v pořadí: měřítko, rotace, posun
    direction_ray.set_transformation(translation * rotation * scaling);

    direction_ray
}

#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl BoundingBox {
    fn new_empty() -> Self {
        Self {
            min: Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn contains(&self, point: Vector3<f32>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    fn from_triangle(tri: &Triangle) -> Self {
        let min = Vector3::new(
            tri.a.x.min(tri.b.x).min(tri.c.x),
            tri.a.y.min(tri.b.y).min(tri.c.y),
            tri.a.z.min(tri.b.z).min(tri.c.z),
        );
        let max = Vector3::new(
            tri.a.x.max(tri.b.x).max(tri.c.x),
            tri.a.y.max(tri.b.y).max(tri.c.y),
            tri.a.z.max(tri.b.z).max(tri.c.z),
        );
        BoundingBox { min, max }
    }

    fn from_triangles(triangles: &[Triangle]) -> Self {
        if triangles.is_empty() {
            return Self::new_empty();
        }
        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for tri in triangles {
            for v in [&tri.a, &tri.b, &tri.c] {
                min.x = min.x.min(v.x);
                min.y = min.y.min(v.y);
                min.z = min.z.min(v.z);
                max.x = max.x.max(v.x);
                max.y = max.y.max(v.y);
                max.z = max.z.max(v.z);
            }
        }
        BoundingBox { min, max }
    }

    fn encompass(box1: &Self, box2: &Self) -> Self {
        BoundingBox {
            min: Vector3::new(
                box1.min.x.min(box2.min.x),
                box1.min.y.min(box2.min.y),
                box1.min.z.min(box2.min.z),
            ),
            max: Vector3::new(
                box1.max.x.max(box2.max.x),
                box1.max.y.max(box2.max.y),
                box1.max.z.max(box2.max.z),
            ),
        }
    }

    /// Test against a single plane: return true if any part of the box is in front of the plane.
    fn intersects_plane(&self, plane: &Plane) -> bool {
        // compute the "positive vertex" for this plane's normal
        let p = Vector3::new(
            if plane.n.x >= 0.0 {
                self.max.x
            } else {
                self.min.x
            },
            if plane.n.y >= 0.0 {
                self.max.y
            } else {
                self.min.y
            },
            if plane.n.z >= 0.0 {
                self.max.z
            } else {
                self.min.z
            },
        );
        // if this farthest point is in front, the box may intersect or be in front
        plane.side(p) >= 0.0
    }

    /// Výpočet povrchové plochy bounding boxu pro SAH
    fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0; // prázdný nebo neplatný box
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

// Struktura pro reprezentaci frustumu kamery
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    pub fn from_camera(camera: &Camera) -> Self {
        // Získáme view-projection matici
        let vp_matrix = camera.projection() * camera.view();

        // Převedeme na pole - Matrix4 nemá as_slice(), musíme použít jiný přístup
        let mat = [
            vp_matrix.x.x,
            vp_matrix.x.y,
            vp_matrix.x.z,
            vp_matrix.x.w,
            vp_matrix.y.x,
            vp_matrix.y.y,
            vp_matrix.y.z,
            vp_matrix.y.w,
            vp_matrix.z.x,
            vp_matrix.z.y,
            vp_matrix.z.z,
            vp_matrix.z.w,
            vp_matrix.w.x,
            vp_matrix.w.y,
            vp_matrix.w.z,
            vp_matrix.w.w,
        ];

        // Extrahujeme 6 rovin frustumu
        // Levá rovina
        let left = Plane {
            n: Vector3::new(mat[3] + mat[0], mat[7] + mat[4], mat[11] + mat[8]).normalize(),
            d: (mat[15] + mat[12])
                / (mat[3] + mat[0]).hypot((mat[7] + mat[4]).hypot(mat[11] + mat[8])),
        };

        // Pravá rovina
        let right = Plane {
            n: Vector3::new(mat[3] - mat[0], mat[7] - mat[4], mat[11] - mat[8]).normalize(),
            d: (mat[15] - mat[12])
                / (mat[3] - mat[0]).hypot((mat[7] - mat[4]).hypot(mat[11] - mat[8])),
        };

        // Spodní rovina
        let bottom = Plane {
            n: Vector3::new(mat[3] + mat[1], mat[7] + mat[5], mat[11] + mat[9]).normalize(),
            d: (mat[15] + mat[13])
                / (mat[3] + mat[1]).hypot((mat[7] + mat[5]).hypot(mat[11] + mat[9])),
        };

        // Horní rovina
        let top = Plane {
            n: Vector3::new(mat[3] - mat[1], mat[7] - mat[5], mat[11] - mat[9]).normalize(),
            d: (mat[15] - mat[13])
                / (mat[3] - mat[1]).hypot((mat[7] - mat[5]).hypot(mat[11] - mat[9])),
        };

        // Blízká rovina
        let near = Plane {
            n: Vector3::new(mat[3] + mat[2], mat[7] + mat[6], mat[11] + mat[10]).normalize(),
            d: (mat[15] + mat[14])
                / (mat[3] + mat[2]).hypot((mat[7] + mat[6]).hypot(mat[11] + mat[10])),
        };

        // Vzdálená rovina
        let far = Plane {
            n: Vector3::new(mat[3] - mat[2], mat[7] - mat[6], mat[11] - mat[10]).normalize(),
            d: (mat[15] - mat[14])
                / (mat[3] - mat[2]).hypot((mat[7] - mat[6]).hypot(mat[11] - mat[10])),
        };

        Frustum {
            planes: [left, right, bottom, top, near, far],
        }
    }

    pub fn as_vec4_array(&self) -> [[f32; 4]; 6] {
        [
            [
                self.planes[0].n.x,
                self.planes[0].n.y,
                self.planes[0].n.z,
                self.planes[0].d,
            ],
            [
                self.planes[1].n.x,
                self.planes[1].n.y,
                self.planes[1].n.z,
                self.planes[1].d,
            ],
            [
                self.planes[2].n.x,
                self.planes[2].n.y,
                self.planes[2].n.z,
                self.planes[2].d,
            ],
            [
                self.planes[3].n.x,
                self.planes[3].n.y,
                self.planes[3].n.z,
                self.planes[3].d,
            ],
            [
                self.planes[4].n.x,
                self.planes[4].n.y,
                self.planes[4].n.z,
                self.planes[4].d,
            ],
            [
                self.planes[5].n.x,
                self.planes[5].n.y,
                self.planes[5].n.z,
                self.planes[5].d,
            ],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Vector3;

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
        };

        let outside = Triangle {
            a: Vector3::new(-10.0, 0.0, 0.0),
            b: Vector3::new(-10.0, 1.0, 0.0),
            c: Vector3::new(-10.0, 0.0, 1.0),
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
        let mut stats = BspStats::default();
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
