use ash::vk;
use std::collections::HashMap;
use crate::memory::{BufferHandle, TransferTicket};

// ===== Vertex & UBO =====

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, color }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UniformBufferObject {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

// ===== Camera =====

pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position: [0.0, 3.0, 0.0],
            target: [1.0, 3.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            aspect,
            near: 0.1,
            far: 800.0,
        }
    }

    pub fn get_view_matrix(&self) -> [[f32; 4]; 4] {
        let f = normalize(sub3(self.target, self.position));
        let s = normalize(cross3(f, self.up));
        let u = cross3(s, f);

        [
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [
                -dot3(s, self.position),
                -dot3(u, self.position),
                dot3(f, self.position),
                1.0,
            ],
        ]
    }

    pub fn get_projection_matrix(&self) -> [[f32; 4]; 4] {
        let fov_rad = self.fov.to_radians();
        let f = 1.0 / (fov_rad / 2.0).tan();

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, -f, 0.0, 0.0],
            [0.0, 0.0, self.far / (self.near - self.far), -1.0],
            [
                0.0,
                0.0,
                (self.near * self.far) / (self.near - self.far),
                0.0,
            ],
        ]
    }

    pub fn update_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn rotate_around_target(&mut self, delta_x: f32, delta_y: f32) {
        let radius = length3(sub3(self.position, self.target));
        let mut theta = (self.position[2] - self.target[2])
            .atan2(self.position[0] - self.target[0]);
        let mut phi = ((self.position[1] - self.target[1]) / radius).asin();
        theta += delta_x * 0.01;
        phi = (phi + delta_y * 0.01).clamp(-1.5, 1.5);
        self.position[0] = self.target[0] + radius * phi.cos() * theta.cos();
        self.position[1] = self.target[1] + radius * phi.sin();
        self.position[2] = self.target[2] + radius * phi.cos() * theta.sin();
    }

    /// Gribb-Hartmann frustum plane extraction.
    pub fn extract_frustum_planes(&self) -> [[f32; 4]; 6] {
        // multiply_matrices(A, B) computes B*A in our column-major layout,
        // so pass (view, proj) to get Proj * View.
        let vp = multiply_matrices(
            self.get_view_matrix(),
            self.get_projection_matrix(),
        );

        let row = |r: usize| [vp[0][r], vp[1][r], vp[2][r], vp[3][r]];
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let add = |a: [f32; 4], b: [f32; 4]| [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]];
        let norm = |mut p: [f32; 4]| {
            let len = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
            if len > 0.0 { p[0] /= len; p[1] /= len; p[2] /= len; p[3] /= len; }
            p
        };

        [
            norm(add(r3, r0)),
            norm(sub(r3, r0)),
            norm(add(r3, r1)),
            norm(sub(r3, r1)),
            norm(add(r3, r2)),
            norm(sub(r3, r2)),
        ]
    }
}

// ===== Chunk coordinate =====

pub type ChunkCoord = (i32, i32);
pub const CHUNK_SIZE: f32 = 64.0;
pub const WORLD_GRID_RADIUS: i32 = 3;
pub const MAX_STREAM_STARTS_PER_FRAME: usize = 4;

// ===== Load State =====

#[derive(Debug)]
pub enum ChunkLoadState {
    Unloaded,
    Streaming {
        vertex_ticket: TransferTicket,
        index_ticket: TransferTicket,
    },
    Ready,
}

// ===== Chunk =====

pub struct Chunk {
    pub coord: ChunkCoord,
    pub load_state: ChunkLoadState,

    pub vertex_handle: Option<BufferHandle>,
    pub index_handle: Option<BufferHandle>,

    /// Raw VkBuffer objects for cmd_bind_vertex_buffers / cmd_bind_index_buffer.
    pub vertex_vk_buffer: Option<vk::Buffer>,
    pub index_vk_buffer: Option<vk::Buffer>,

    pub meshes: Vec<Mesh>,

    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],

    pub total_vertex_count: u32,
    pub total_index_count: u32,
}

impl Chunk {
    pub fn new(coord: ChunkCoord, meshes: Vec<Mesh>) -> Self {
        let mut total_vertex_count = 0u32;
        let mut total_index_count = 0u32;
        let mut aabb_min = [f32::MAX; 3];
        let mut aabb_max = [f32::MIN; 3];

        for mesh in &meshes {
            total_vertex_count += mesh.vertices.len() as u32;
            total_index_count += mesh.indices.len() as u32;
            for v in &mesh.vertices {
                let wp = transform_point(mesh.transform, v.position);
                for i in 0..3 {
                    aabb_min[i] = aabb_min[i].min(wp[i]);
                    aabb_max[i] = aabb_max[i].max(wp[i]);
                }
            }
        }

        if meshes.is_empty() {
            let cx = coord.0 as f32 * CHUNK_SIZE;
            let cz = coord.1 as f32 * CHUNK_SIZE;
            aabb_min = [cx, -1.0, cz];
            aabb_max = [cx + CHUNK_SIZE, 1.0, cz + CHUNK_SIZE];
        }

        Self {
            coord,
            load_state: ChunkLoadState::Unloaded,
            vertex_handle: None,
            index_handle: None,
            vertex_vk_buffer: None,
            index_vk_buffer: None,
            meshes,
            aabb_min,
            aabb_max,
            total_vertex_count,
            total_index_count,
        }
    }

    pub fn is_visible(&self, frustum_planes: &[[f32; 4]; 6]) -> bool {
        for plane in frustum_planes {
            let px = if plane[0] >= 0.0 { self.aabb_max[0] } else { self.aabb_min[0] };
            let py = if plane[1] >= 0.0 { self.aabb_max[1] } else { self.aabb_min[1] };
            let pz = if plane[2] >= 0.0 { self.aabb_max[2] } else { self.aabb_min[2] };
            if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
                return false;
            }
        }
        true
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.load_state, ChunkLoadState::Ready)
    }

    pub fn is_unloaded(&self) -> bool {
        matches!(self.load_state, ChunkLoadState::Unloaded)
    }
}

// ===== Mesh =====

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub transform: [[f32; 4]; 4],
}

impl Mesh {
    pub fn create_cube() -> Self {
        let positions = [
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
        ];
        let face_colors = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let face_data = [
            ([0,1,2,3], 0), ([4,5,6,7], 1), ([3,2,7,6], 2),
            ([5,4,1,0], 3), ([1,4,7,2], 4), ([5,0,3,6], 5),
        ];
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (face_idx, color_idx) in face_data {
            let base = vertices.len() as u32;
            for &idx in face_idx.iter() {
                vertices.push(Vertex::new(positions[idx], face_colors[color_idx]));
            }
            indices.extend([base, base+1, base+2, base+2, base+3, base]);
        }
        Self { vertices, indices, transform: identity_matrix() }
    }
}

// ===== Scene =====

pub struct Scene {
    pub chunks: HashMap<ChunkCoord, Chunk>,
    pub camera: Camera,
    pub rotation: f32,
    pub frame_number: u64,
}

impl Scene {
    pub fn new(aspect: f32) -> Self {
        let mut chunks = HashMap::new();

        for cx in -WORLD_GRID_RADIUS..=WORLD_GRID_RADIUS {
            for cz in -WORLD_GRID_RADIUS..=WORLD_GRID_RADIUS {
                let meshes = crate::worldgen::generate_chunk_meshes(cx, cz);
                let chunk = Chunk::new((cx, cz), meshes);
                chunks.insert((cx, cz), chunk);
            }
        }

        println!(
            "[Scene] Generated {} chunks ({}×{} grid), all Unloaded",
            chunks.len(),
            WORLD_GRID_RADIUS * 2 + 1,
            WORLD_GRID_RADIUS * 2 + 1,
        );

        Self {
            chunks,
            camera: Camera::new(aspect),
            rotation: 0.0,
            frame_number: 0,
        }
    }

    pub fn update(&mut self, delta_time: f32) {
        self.rotation += delta_time * 30.0_f32.to_radians(); // ~30°/sec → full 360° in 12 sec
        self.frame_number += 1;

        // Stand at the origin, 3 units above ground, rotate to look
        // outward horizontally.  This ensures many chunks are behind
        // the camera at any given moment and get frustum-culled.
        self.camera.position = [0.0, 3.0, 0.0];
        self.camera.target = [
            self.rotation.cos(),
            3.0,
            self.rotation.sin(),
        ];
    }

    /// Unloaded chunks sorted nearest-to-camera first.
    pub fn unloaded_chunks_by_distance(&self) -> Vec<ChunkCoord> {
        let cam_chunk = self.camera_chunk();
        let mut coords: Vec<ChunkCoord> = self
            .chunks
            .iter()
            .filter(|(_, c)| c.is_unloaded())
            .map(|(&coord, _)| coord)
            .collect();
        coords.sort_by_key(|&(x, z)| {
            let dx = (x - cam_chunk.0).abs();
            let dz = (z - cam_chunk.1).abs();
            dx * dx + dz * dz
        });
        coords
    }

    pub fn camera_chunk(&self) -> ChunkCoord {
        (
            (self.camera.position[0] / CHUNK_SIZE).floor() as i32,
            (self.camera.position[2] / CHUNK_SIZE).floor() as i32,
        )
    }

    pub fn visible_ready_chunks(&self, frustum: &[[f32; 4]; 6]) -> Vec<&Chunk> {
        self.chunks.values().filter(|c| c.is_ready() && c.is_visible(frustum)).collect()
    }

    pub fn get_ubo(&self) -> UniformBufferObject {
        UniformBufferObject {
            model: identity_matrix(),
            view: self.camera.get_view_matrix(),
            proj: self.camera.get_projection_matrix(),
        }
    }
}

// ===== Math helpers =====

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len > 0.0 { [v[0]/len, v[1]/len, v[2]/len] } else { v }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

fn length3(v: [f32; 3]) -> f32 {
    (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()
}

pub fn identity_matrix() -> [[f32; 4]; 4] {
    [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]
}

pub fn create_rotation_matrix(angle: f32, axis: [f32; 3]) -> [[f32; 4]; 4] {
    let axis = normalize(axis);
    let s = angle.sin();
    let c = angle.cos();
    let oc = 1.0 - c;
    [
        [oc*axis[0]*axis[0]+c, oc*axis[0]*axis[1]-axis[2]*s, oc*axis[2]*axis[0]+axis[1]*s, 0.0],
        [oc*axis[0]*axis[1]+axis[2]*s, oc*axis[1]*axis[1]+c, oc*axis[1]*axis[2]-axis[0]*s, 0.0],
        [oc*axis[2]*axis[0]-axis[1]*s, oc*axis[1]*axis[2]+axis[0]*s, oc*axis[2]*axis[2]+c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[allow(dead_code)]
fn create_translation_matrix(t: [f32; 3]) -> [[f32; 4]; 4] {
    [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[t[0],t[1],t[2],1.0]]
}

pub fn multiply_matrices(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 { for j in 0..4 { for k in 0..4 { r[i][j] += a[i][k] * b[k][j]; } } }
    r
}

fn transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0]*p[0] + m[1][0]*p[1] + m[2][0]*p[2] + m[3][0],
        m[0][1]*p[0] + m[1][1]*p[1] + m[2][1]*p[2] + m[3][1],
        m[0][2]*p[0] + m[1][2]*p[1] + m[2][2]*p[2] + m[3][2],
    ]
}