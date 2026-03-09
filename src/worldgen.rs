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

use crate::scene::{Mesh, Vertex, CHUNK_SIZE};

// ====================================================================
//  Simple deterministic hash (no rand crate needed)
// ====================================================================

/// A trivially simple splitmix-style PRNG seeded from two i32s.
struct ChunkRng {
    state: u64,
}

impl ChunkRng {
    fn new(cx: i32, cz: i32) -> Self {
        // Combine coords into a single seed with bit mixing.
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

    /// Random f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Random f32 in [lo, hi).
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Random usize in [0, n).
    fn range_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ====================================================================
//  Chunk colour palette
// ====================================================================

/// Pick a ground-plane colour from the chunk coordinate so neighbouring
/// chunks are visually distinct.  Uses a small hand-tuned palette and
/// indexes by `(cx ^ cz)` to get a checkerboard-ish pattern.
fn chunk_ground_color(cx: i32, cz: i32) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 8] = [
        [0.25, 0.55, 0.22], // forest green
        [0.30, 0.50, 0.18], // olive
        [0.35, 0.58, 0.25], // grass
        [0.22, 0.48, 0.20], // dark green
        [0.40, 0.56, 0.28], // lime tint
        [0.28, 0.52, 0.24], // mid green
        [0.32, 0.60, 0.22], // bright green
        [0.26, 0.46, 0.26], // sage
    ];

    let idx = ((cx.wrapping_mul(7)) ^ (cz.wrapping_mul(13)))
        .unsigned_abs() as usize
        % PALETTE.len();
    PALETTE[idx]
}

// ====================================================================
//  Geometry primitives
// ====================================================================

/// A flat quad on the XZ plane at y=0, covering one full chunk.
fn make_ground_plane(cx: i32, cz: i32) -> Mesh {
    let x0 = cx as f32 * CHUNK_SIZE;
    let z0 = cz as f32 * CHUNK_SIZE;
    let x1 = x0 + CHUNK_SIZE;
    let z1 = z0 + CHUNK_SIZE;
    let color = chunk_ground_color(cx, cz);

    let vertices = vec![
        Vertex::new([x0, 0.0, z0], color),
        Vertex::new([x1, 0.0, z0], color),
        Vertex::new([x1, 0.0, z1], color),
        Vertex::new([x0, 0.0, z1], color),
    ];
    // Wind CCW from above so the normal points UP (+Y) toward the camera.
    //   0──1       Tri 1: 0→2→1  (bottom-left → top-right → bottom-right)
    //   │╲ │       Tri 2: 0→3→2  (bottom-left → top-left  → top-right)
    //   3──2
    let indices = vec![0, 2, 1, 0, 3, 2];

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
    }
}

/// Unit cube centered at origin, with per-face colors tinted by `base`.
fn make_cube(base_color: [f32; 3]) -> Mesh {
    let tint = |r: f32, g: f32, b: f32| -> [f32; 3] {
        [
            (base_color[0] * 0.5 + r * 0.5).min(1.0),
            (base_color[1] * 0.5 + g * 0.5).min(1.0),
            (base_color[2] * 0.5 + b * 0.5).min(1.0),
        ]
    };

    let positions: [[f32; 3]; 8] = [
        [-0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
        [-0.5, 1.0, 0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, -0.5],
        [-0.5, 1.0, -0.5],
        [0.5, 1.0, -0.5],
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
        ([0, 1, 2, 3], 0),
        ([4, 5, 6, 7], 1),
        ([3, 2, 7, 6], 2),
        ([5, 4, 1, 0], 3),
        ([1, 4, 7, 2], 4),
        ([5, 0, 3, 6], 5),
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (face_idx, color_idx) in face_data {
        let base = vertices.len() as u32;
        for &idx in face_idx.iter() {
            vertices.push(Vertex::new(positions[idx], face_colors[color_idx]));
        }
        indices.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
    }
}

/// Four-sided pyramid with base at y=0 and apex at y=height.
fn make_pyramid(color: [f32; 3], height: f32) -> Mesh {
    let s = 0.5f32; // half-width of base
    let apex = [0.0, height, 0.0];

    let bl = [-s, 0.0, -s];
    let br = [s, 0.0, -s];
    let fr = [s, 0.0, s];
    let fl = [-s, 0.0, s];

    // Slightly different shade per face for depth cue.
    let c0 = color;
    let c1 = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8];
    let c2 = [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6];
    let c3 = [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Four triangular faces.
    let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3]); 4] = [
        (fl, fr, apex, c0),  // front
        (fr, br, apex, c1),  // right
        (br, bl, apex, c2),  // back
        (bl, fl, apex, c3),  // left
    ];

    for (a, b, c, col) in &faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex::new(*a, *col));
        vertices.push(Vertex::new(*b, *col));
        vertices.push(Vertex::new(*c, *col));
        indices.extend([base, base + 1, base + 2]);
    }

    // Base quad (two tris).  Wound so normal faces UP (consistent with
    // the ground plane — visible from above through any gaps).
    let base_col = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];
    let base_start = vertices.len() as u32;
    vertices.push(Vertex::new(bl, base_col));
    vertices.push(Vertex::new(br, base_col));
    vertices.push(Vertex::new(fr, base_col));
    vertices.push(Vertex::new(fl, base_col));
    indices.extend([
        base_start,
        base_start + 2,
        base_start + 1,
        base_start,
        base_start + 3,
        base_start + 2,
    ]);

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
    }
}

/// Hexagonal column (approximated as 8-sided prism).
fn make_column(color: [f32; 3], height: f32, radius: f32) -> Mesh {
    const SIDES: usize = 8;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let top_color = [
        (color[0] * 1.2).min(1.0),
        (color[1] * 1.2).min(1.0),
        (color[2] * 1.2).min(1.0),
    ];
    let side_color = color;

    // Generate ring of points at y=0 and y=height.
    let mut bottom_ring = Vec::new();
    let mut top_ring = Vec::new();

    for i in 0..SIDES {
        let angle = (i as f32 / SIDES as f32) * std::f32::consts::TAU;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        bottom_ring.push([x, 0.0, z]);
        top_ring.push([x, height, z]);
    }

    // Side quads.  Vertex order swapped (j before i) so the outward
    // face normal points away from the cylinder centre.
    for i in 0..SIDES {
        let j = (i + 1) % SIDES;
        let base = vertices.len() as u32;

        vertices.push(Vertex::new(bottom_ring[j], side_color));
        vertices.push(Vertex::new(bottom_ring[i], side_color));
        vertices.push(Vertex::new(top_ring[i], side_color));
        vertices.push(Vertex::new(top_ring[j], side_color));

        indices.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    // Top cap (fan from center).  Reversed winding so normal faces UP.
    let center_top = vertices.len() as u32;
    vertices.push(Vertex::new([0.0, height, 0.0], top_color));
    for i in 0..SIDES {
        let j = (i + 1) % SIDES;
        let bi = vertices.len() as u32;
        vertices.push(Vertex::new(top_ring[i], top_color));
        vertices.push(Vertex::new(top_ring[j], top_color));
        indices.extend([center_top, bi + 1, bi]);
    }

    Mesh {
        vertices,
        indices,
        transform: crate::scene::identity_matrix(),
    }
}

// ====================================================================
//  Public API
// ====================================================================

/// Object type placed in a chunk.
#[derive(Debug, Clone, Copy)]
enum ObjectKind {
    Cube,
    Pyramid,
    Column,
}

/// Generate all meshes for a single chunk.  The first mesh is always the
/// ground plane; the rest are scattered objects.
pub fn generate_chunk_meshes(cx: i32, cz: i32) -> Vec<Mesh> {
    let mut rng = ChunkRng::new(cx, cz);
    let mut meshes = Vec::new();

    // 1. Ground plane.
    meshes.push(make_ground_plane(cx, cz));

    // 2. Scattered objects: 3–8 per chunk.
    let object_count = 3 + rng.range_usize(6); // 3..=8

    let world_x = cx as f32 * CHUNK_SIZE;
    let world_z = cz as f32 * CHUNK_SIZE;

    let kinds = [ObjectKind::Cube, ObjectKind::Pyramid, ObjectKind::Column];

    for _ in 0..object_count {
        let kind = kinds[rng.range_usize(kinds.len())];

        // Random position within the chunk, with a 2-unit margin from
        // edges to avoid z-fighting with neighbours.
        let margin = 2.0;
        let lx = rng.range_f32(margin, CHUNK_SIZE - margin);
        let lz = rng.range_f32(margin, CHUNK_SIZE - margin);

        let px = world_x + lx;
        let pz = world_z + lz;

        // Random scale.
        let scale = rng.range_f32(0.8, 3.0);

        // Random color (warm/cool biased by coord for variety).
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

        // Build transform: translate to (px, 0, pz) and scale.
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