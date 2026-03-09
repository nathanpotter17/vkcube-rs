//! Procedural world generation for the chunk streaming test.
//!
//! Each chunk is a 64×64 world-unit tile containing:
//!   - A ground plane quad colored by chunk coordinate
//!   - 3–8 scattered objects (cubes, pyramids, columns) at positions
//!     determined by a simple hash of the chunk coordinate
//!
//! Everything is deterministic – regenerating a chunk at the same coord
//! always produces identical geometry, so evicted chunks can be re-loaded
//! without storing anything on disk.
//!
//! Phase 1: vertices now include normals and UVs for PBR shading.

use crate::scene::{Mesh, Vertex, CHUNK_SIZE};

// ====================================================================
//  Simple deterministic hash (no rand crate needed)
// ====================================================================

struct ChunkRng {
    state: u64,
}

impl ChunkRng {
    fn new(cx: i32, cz: i32) -> Self {
        let a = cx as u64;
        let b = cz as u64;
        let mut s = a.wrapping_mul(0x9E3779B97F4A7C15)
            ^ b.wrapping_mul(0x517CC1B727220A95);
        s = s.wrapping_add(0x6A09E667F3BCC908);
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    fn range_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ====================================================================
//  Chunk colour palette
// ====================================================================

fn chunk_ground_color(cx: i32, cz: i32) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 8] = [
        [0.25, 0.55, 0.22],
        [0.30, 0.50, 0.18],
        [0.35, 0.58, 0.25],
        [0.22, 0.48, 0.20],
        [0.40, 0.56, 0.28],
        [0.28, 0.52, 0.24],
        [0.32, 0.60, 0.22],
        [0.26, 0.46, 0.26],
    ];

    let idx = ((cx.wrapping_mul(7)) ^ (cz.wrapping_mul(13)))
        .unsigned_abs() as usize
        % PALETTE.len();
    PALETTE[idx]
}

// ====================================================================
//  Geometry primitives (now with normals + UVs)
// ====================================================================

/// Ground plane quad on XZ at y=0, with normal pointing UP (+Y).
fn make_ground_plane(cx: i32, cz: i32) -> Mesh {
    let x0 = cx as f32 * CHUNK_SIZE;
    let z0 = cz as f32 * CHUNK_SIZE;
    let x1 = x0 + CHUNK_SIZE;
    let z1 = z0 + CHUNK_SIZE;
    let color = chunk_ground_color(cx, cz);
    let normal = [0.0, 1.0, 0.0];

    let vertices = vec![
        Vertex::full([x0, 0.0, z0], normal, [0.0, 0.0], color),
        Vertex::full([x1, 0.0, z0], normal, [1.0, 0.0], color),
        Vertex::full([x1, 0.0, z1], normal, [1.0, 1.0], color),
        Vertex::full([x0, 0.0, z1], normal, [0.0, 1.0], color),
    ];
    let indices = vec![0, 2, 1, 0, 3, 2];

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
        material_id: 0, // default material
    }
}

/// Unit cube centered at origin with per-face normals and colors.
fn make_cube(base_color: [f32; 3]) -> Mesh {
    let tint = |r: f32, g: f32, b: f32| -> [f32; 3] {
        [
            (base_color[0] * 0.5 + r * 0.5).min(1.0),
            (base_color[1] * 0.5 + g * 0.5).min(1.0),
            (base_color[2] * 0.5 + b * 0.5).min(1.0),
        ]
    };

    let positions: [[f32; 3]; 8] = [
        [-0.5, 0.0, 0.5],  [ 0.5, 0.0, 0.5],
        [ 0.5, 1.0, 0.5],  [-0.5, 1.0, 0.5],
        [ 0.5, 0.0, -0.5], [-0.5, 0.0, -0.5],
        [-0.5, 1.0, -0.5], [ 0.5, 1.0, -0.5],
    ];

    let face_normals: [[f32; 3]; 6] = [
        [ 0.0,  0.0,  1.0],  // front
        [ 0.0,  0.0, -1.0],  // back
        [ 0.0,  1.0,  0.0],  // top
        [ 0.0, -1.0,  0.0],  // bottom
        [ 1.0,  0.0,  0.0],  // right
        [-1.0,  0.0,  0.0],  // left
    ];

    let face_colors = [
        tint(1.0, 0.3, 0.3),
        tint(0.3, 1.0, 0.3),
        tint(0.3, 0.3, 1.0),
        tint(1.0, 1.0, 0.3),
        tint(1.0, 0.3, 1.0),
        tint(0.3, 1.0, 1.0),
    ];

    let face_data: [([usize; 4], usize); 6] = [
        ([0, 1, 2, 3], 0),  // front
        ([4, 5, 6, 7], 1),  // back
        ([3, 2, 7, 6], 2),  // top
        ([5, 4, 1, 0], 3),  // bottom
        ([1, 4, 7, 2], 4),  // right
        ([5, 0, 3, 6], 5),  // left
    ];

    let face_uvs: [[f32; 2]; 4] = [
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (face_idx, color_idx) in face_data {
        let base = vertices.len() as u32;
        let normal = face_normals[color_idx];
        let color = face_colors[color_idx];

        for (vi, &idx) in face_idx.iter().enumerate() {
            vertices.push(Vertex::full(positions[idx], normal, face_uvs[vi], color));
        }
        indices.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
        material_id: 0,
    }
}

/// Four-sided pyramid with per-face normals.
fn make_pyramid(color: [f32; 3], height: f32) -> Mesh {
    let s = 0.5f32;
    let apex = [0.0, height, 0.0];

    let bl = [-s, 0.0, -s];
    let br = [ s, 0.0, -s];
    let fr = [ s, 0.0,  s];
    let fl = [-s, 0.0,  s];

    let c0 = color;
    let c1 = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8];
    let c2 = [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6];
    let c3 = [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Compute normals for each triangular face.
    let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3]); 4] = [
        (fl, fr, apex, c0),
        (fr, br, apex, c1),
        (br, bl, apex, c2),
        (bl, fl, apex, c3),
    ];

    for (a, b, c, col) in &faces {
        let base = vertices.len() as u32;
        let edge1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let edge2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let normal = normalize_vec3(cross_vec3(edge1, edge2));

        vertices.push(Vertex::full(*a, normal, [0.0, 1.0], *col));
        vertices.push(Vertex::full(*b, normal, [1.0, 1.0], *col));
        vertices.push(Vertex::full(*c, normal, [0.5, 0.0], *col));
        indices.extend([base, base + 1, base + 2]);
    }

    // Base quad.
    let base_col = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];
    let base_normal = [0.0, -1.0, 0.0];
    let base_start = vertices.len() as u32;
    vertices.push(Vertex::full(bl, base_normal, [0.0, 0.0], base_col));
    vertices.push(Vertex::full(br, base_normal, [1.0, 0.0], base_col));
    vertices.push(Vertex::full(fr, base_normal, [1.0, 1.0], base_col));
    vertices.push(Vertex::full(fl, base_normal, [0.0, 1.0], base_col));
    indices.extend([
        base_start, base_start + 2, base_start + 1,
        base_start, base_start + 3, base_start + 2,
    ]);

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
        material_id: 0,
    }
}

/// Octagonal column with per-face normals.
fn make_column(color: [f32; 3], height: f32, radius: f32) -> Mesh {
    const SIDES: usize = 8;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let top_color = [
        (color[0] * 1.2).min(1.0),
        (color[1] * 1.2).min(1.0),
        (color[2] * 1.2).min(1.0),
    ];

    let mut bottom_ring = Vec::new();
    let mut top_ring = Vec::new();

    for i in 0..SIDES {
        let angle = (i as f32 / SIDES as f32) * std::f32::consts::TAU;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        bottom_ring.push([x, 0.0, z]);
        top_ring.push([x, height, z]);
    }

    // Side quads with outward normals.
    for i in 0..SIDES {
        let j = (i + 1) % SIDES;
        let base = vertices.len() as u32;

        // Compute face normal: cross product of two edges.
        let mid_x = (bottom_ring[i][0] + bottom_ring[j][0]) * 0.5;
        let mid_z = (bottom_ring[i][2] + bottom_ring[j][2]) * 0.5;
        let len = (mid_x * mid_x + mid_z * mid_z).sqrt();
        let normal = if len > 0.0 {
            [mid_x / len, 0.0, mid_z / len]
        } else {
            [1.0, 0.0, 0.0]
        };

        let u0 = i as f32 / SIDES as f32;
        let u1 = (i + 1) as f32 / SIDES as f32;

        vertices.push(Vertex::full(bottom_ring[j], normal, [u1, 1.0], color));
        vertices.push(Vertex::full(bottom_ring[i], normal, [u0, 1.0], color));
        vertices.push(Vertex::full(top_ring[i],    normal, [u0, 0.0], color));
        vertices.push(Vertex::full(top_ring[j],    normal, [u1, 0.0], color));

        indices.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    // Top cap (fan from center), normal pointing UP.
    let top_normal = [0.0, 1.0, 0.0];
    let center_top = vertices.len() as u32;
    vertices.push(Vertex::full([0.0, height, 0.0], top_normal, [0.5, 0.5], top_color));
    for i in 0..SIDES {
        let j = (i + 1) % SIDES;
        let bi = vertices.len() as u32;
        let ui = (i as f32 / SIDES as f32) * std::f32::consts::TAU;
        let uj = ((i + 1) as f32 / SIDES as f32) * std::f32::consts::TAU;
        vertices.push(Vertex::full(top_ring[i], top_normal, [ui.cos()*0.5+0.5, ui.sin()*0.5+0.5], top_color));
        vertices.push(Vertex::full(top_ring[j], top_normal, [uj.cos()*0.5+0.5, uj.sin()*0.5+0.5], top_color));
        indices.extend([center_top, bi + 1, bi]);
    }

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
        material_id: 0,
    }
}

// ====================================================================
//  Math helpers (local to worldgen)
// ====================================================================

fn cross_vec3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len > 0.0 { [v[0]/len, v[1]/len, v[2]/len] } else { [0.0, 1.0, 0.0] }
}

// ====================================================================
//  Public API
// ====================================================================

#[derive(Debug, Clone, Copy)]
enum ObjectKind {
    Cube,
    Pyramid,
    Column,
}

pub fn generate_chunk_meshes(cx: i32, cz: i32) -> Vec<Mesh> {
    let mut rng = ChunkRng::new(cx, cz);
    let mut meshes = Vec::new();

    // 1. Ground plane.
    meshes.push(make_ground_plane(cx, cz));

    // 2. Scattered objects: 3–8 per chunk.
    let object_count = 3 + rng.range_usize(6);

    let world_x = cx as f32 * CHUNK_SIZE;
    let world_z = cz as f32 * CHUNK_SIZE;

    let kinds = [ObjectKind::Cube, ObjectKind::Pyramid, ObjectKind::Column];

    for _ in 0..object_count {
        let kind = kinds[rng.range_usize(kinds.len())];

        let margin = 2.0;
        let lx = rng.range_f32(margin, CHUNK_SIZE - margin);
        let lz = rng.range_f32(margin, CHUNK_SIZE - margin);

        let px = world_x + lx;
        let pz = world_z + lz;

        let scale = rng.range_f32(0.8, 3.0);

        let r = rng.range_f32(0.3, 1.0);
        let g = rng.range_f32(0.3, 1.0);
        let b = rng.range_f32(0.3, 1.0);
        let color = [r, g, b];

        let mut mesh = match kind {
            ObjectKind::Cube => make_cube(color),
            ObjectKind::Pyramid => {
                let h = rng.range_f32(1.0, 2.5);
                make_pyramid(color, h)
            }
            ObjectKind::Column => {
                let h = rng.range_f32(1.5, 4.0);
                let rad = rng.range_f32(0.2, 0.6);
                make_column(color, h, rad)
            }
        };

        mesh.transform = [
            [scale, 0.0, 0.0, 0.0],
            [0.0, scale, 0.0, 0.0],
            [0.0, 0.0, scale, 0.0],
            [px, 0.0, pz, 1.0],
        ];

        meshes.push(mesh);
    }

    meshes
}