use ash::vk;
use std::sync::Arc;

// Simple 3D scene components

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
            position: [3.0, 3.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            aspect,
            near: 0.1,
            far: 100.0,
        }
    }

    pub fn get_view_matrix(&self) -> [[f32; 4]; 4] {
        // Simple look-at matrix
        let f = normalize(sub3(self.target, self.position));
        let s = normalize(cross3(f, self.up));
        let u = cross3(s, f);

        [
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [-dot3(s, self.position), -dot3(u, self.position), dot3(f, self.position), 1.0],
        ]
    }

    pub fn get_projection_matrix(&self) -> [[f32; 4]; 4] {
        let fov_rad = self.fov.to_radians();
        let f = 1.0 / (fov_rad / 2.0).tan();

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, -f, 0.0, 0.0],  // Vulkan Y-flip
            [0.0, 0.0, self.far / (self.near - self.far), -1.0],
            [0.0, 0.0, (self.near * self.far) / (self.near - self.far), 0.0],
        ]
    }

    pub fn update_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn rotate_around_target(&mut self, delta_x: f32, delta_y: f32) {
        // Simple orbit camera
        let radius = length3(sub3(self.position, self.target));
        
        // Convert to spherical coordinates
        let mut theta = (self.position[2] - self.target[2]).atan2(self.position[0] - self.target[0]);
        let mut phi = ((self.position[1] - self.target[1]) / radius).asin();
        
        // Apply deltas
        theta += delta_x * 0.01;
        phi = (phi + delta_y * 0.01).clamp(-1.5, 1.5);
        
        // Convert back to Cartesian
        self.position[0] = self.target[0] + radius * phi.cos() * theta.cos();
        self.position[1] = self.target[1] + radius * phi.sin();
        self.position[2] = self.target[2] + radius * phi.cos() * theta.sin();
    }
}

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub camera: Camera,
    rotation: f32,
}

impl Scene {
    pub fn new(aspect: f32) -> Self {
        let mut meshes = Vec::new();
        
        // Create a colorful cube at origin
        let mut cube = Mesh::create_cube();
        cube.transform = identity_matrix();
        meshes.push(cube);
        
        Self {
            meshes,
            camera: Camera::new(aspect),
            rotation: 0.0,
        }
    }

    pub fn update(&mut self, delta_time: f32) {
        // Animate rotation
        self.rotation += delta_time * 45.0_f32.to_radians();
        
        // Update mesh transform - rotate on Y and X axes
        if self.meshes.len() > 0 {
            let y_rotation = create_rotation_matrix(self.rotation, [0.0, 1.0, 0.0]);
            let x_rotation = create_rotation_matrix(self.rotation * 0.7, [1.0, 0.0, 0.0]);
            self.meshes[0].transform = multiply_matrices(y_rotation, x_rotation);
        }
    }

    pub fn get_ubo(&self) -> UniformBufferObject {
        UniformBufferObject {
            model: [[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]],
            view: self.camera.get_view_matrix(),
            proj: self.camera.get_projection_matrix(),
        }
    }
}

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
            [1.0, 0.0, 0.0], // front - red
            [0.0, 1.0, 0.0], // back - green  
            [0.0, 0.0, 1.0], // top - blue
            [1.0, 1.0, 0.0], // bottom - yellow
            [1.0, 0.0, 1.0], // right - magenta
            [0.0, 1.0, 1.0], // left - cyan
        ];
        
        let face_data = [
            ([0,1,2,3], 0), // front
            ([4,5,6,7], 1), // back
            ([3,2,7,6], 2), // top
            ([5,4,1,0], 3), // bottom
            ([1,4,7,2], 4), // right
            ([5,0,3,6], 5), // left
        ];
        
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        for (face_idx, color_idx) in face_data {
            let base = vertices.len() as u32;
            for &idx in face_idx.iter() {
                vertices.push(Vertex {
                    position: positions[idx],
                    color: face_colors[color_idx],
                });
            }
            indices.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        Self {
            vertices,
            indices,
            transform: identity_matrix(),
        }
    }
}

// Helper math functions
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn create_rotation_matrix(angle: f32, axis: [f32; 3]) -> [[f32; 4]; 4] {
    let axis = normalize(axis);
    let s = angle.sin();
    let c = angle.cos();
    let oc = 1.0 - c;
    
    [
        [oc * axis[0] * axis[0] + c, oc * axis[0] * axis[1] - axis[2] * s, oc * axis[2] * axis[0] + axis[1] * s, 0.0],
        [oc * axis[0] * axis[1] + axis[2] * s, oc * axis[1] * axis[1] + c, oc * axis[1] * axis[2] - axis[0] * s, 0.0],
        [oc * axis[2] * axis[0] - axis[1] * s, oc * axis[1] * axis[2] + axis[0] * s, oc * axis[2] * axis[2] + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn create_translation_matrix(translation: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [translation[0], translation[1], translation[2], 1.0],
    ]
}

fn multiply_matrices(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}