use cgmath::{InnerSpace, Vector2, Vector3};
use three_d::Camera;

#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    pub a: Vector3<f32>,
    pub b: Vector3<f32>,
    pub c: Vector3<f32>,
    pub uv_a: Vector2<f32>,
    pub uv_b: Vector2<f32>,
    pub uv_c: Vector2<f32>,
}

#[derive(Clone, Debug)]
pub struct Plane {
    pub n: Vector3<f32>,
    pub d: f32,
}

impl Plane {
    pub fn new(n: Vector3<f32>, point: Vector3<f32>) -> Self {
        let n = n.normalize();
        let d = -n.dot(point);
        Self { n, d }
    }

    pub fn side(&self, point: Vector3<f32>) -> f32 {
        self.n.dot(point) + self.d
    }

    pub fn classify(&self, point: Vector3<f32>) -> i32 {
        let dist = self.side(point);
        const EPSILON: f32 = 1e-6;
        match dist {
            d if d > EPSILON => 1,
            d if d < -EPSILON => -1,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl BoundingBox {
    pub fn new_empty() -> Self {
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

    pub fn from_triangle(tri: &Triangle) -> Self {
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

    pub fn from_triangles(triangles: &[Triangle]) -> Self {
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

    pub fn encompass(box1: &Self, box2: &Self) -> Self {
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

    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        let p = Vector3::new(
            if plane.n.x >= 0.0 { self.max.x } else { self.min.x },
            if plane.n.y >= 0.0 { self.max.y } else { self.min.y },
            if plane.n.z >= 0.0 { self.max.z } else { self.min.z },
        );
        plane.side(p) >= 0.0
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    pub fn from_camera(camera: &Camera) -> Self {
        let vp_matrix = camera.projection() * camera.view();
        let mat = [
            vp_matrix.x.x, vp_matrix.x.y, vp_matrix.x.z, vp_matrix.x.w,
            vp_matrix.y.x, vp_matrix.y.y, vp_matrix.y.z, vp_matrix.y.w,
            vp_matrix.z.x, vp_matrix.z.y, vp_matrix.z.z, vp_matrix.z.w,
            vp_matrix.w.x, vp_matrix.w.y, vp_matrix.w.z, vp_matrix.w.w,
        ];

        let left = Plane {
            n: Vector3::new(mat[3] + mat[0], mat[7] + mat[4], mat[11] + mat[8]).normalize(),
            d: (mat[15] + mat[12]) / (mat[3] + mat[0]).hypot((mat[7] + mat[4]).hypot(mat[11] + mat[8])),
        };
        let right = Plane {
            n: Vector3::new(mat[3] - mat[0], mat[7] - mat[4], mat[11] - mat[8]).normalize(),
            d: (mat[15] - mat[12]) / (mat[3] - mat[0]).hypot((mat[7] - mat[4]).hypot(mat[11] - mat[8])),
        };
        let bottom = Plane {
            n: Vector3::new(mat[3] + mat[1], mat[7] + mat[5], mat[11] + mat[9]).normalize(),
            d: (mat[15] + mat[13]) / (mat[3] + mat[1]).hypot((mat[7] + mat[5]).hypot(mat[11] + mat[9])),
        };
        let top = Plane {
            n: Vector3::new(mat[3] - mat[1], mat[7] - mat[5], mat[11] - mat[9]).normalize(),
            d: (mat[15] - mat[13]) / (mat[3] - mat[1]).hypot((mat[7] - mat[5]).hypot(mat[11] - mat[9])),
        };
        let near = Plane {
            n: Vector3::new(mat[3] + mat[2], mat[7] + mat[6], mat[11] + mat[10]).normalize(),
            d: (mat[15] + mat[14]) / (mat[3] + mat[2]).hypot((mat[7] + mat[6]).hypot(mat[11] + mat[10])),
        };
        let far = Plane {
            n: Vector3::new(mat[3] - mat[2], mat[7] - mat[6], mat[11] - mat[10]).normalize(),
            d: (mat[15] - mat[14]) / (mat[3] - mat[2]).hypot((mat[7] - mat[6]).hypot(mat[11] - mat[10])),
        };

        Self {
            planes: [left, right, bottom, top, near, far],
        }
    }

    pub fn intersects(&self, bbox: &BoundingBox) -> bool {
        self.planes.iter().all(|p| bbox.intersects_plane(p))
    }
}

pub fn triangle_center(tri: &Triangle) -> Vector3<f32> {
    (tri.a + tri.b + tri.c) / 3.0
}
