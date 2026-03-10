use ash::vk;

// ===== Vertex =====

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0], color }
    }

    pub fn with_normal(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, normal, uv: [0.0, 0.0], color }
    }

    pub fn full(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 3]) -> Self {
        Self { position, normal, uv, color }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(0).format(vk::Format::R32G32B32_SFLOAT).offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(1).format(vk::Format::R32G32B32_SFLOAT).offset(12),
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(2).format(vk::Format::R32G32_SFLOAT).offset(24),
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(3).format(vk::Format::R32G32B32_SFLOAT).offset(32),
        ]
    }
}

// ===== UBO =====

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UniformBufferObject {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub material_id: u32,
    pub _pad: [u32; 3],
}

impl UniformBufferObject {
    pub fn new(
        model: [[f32; 4]; 4], view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4], material_id: u32,
    ) -> Self {
        Self { model, view, proj, material_id, _pad: [0; 3] }
    }
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

const ORBIT_RADIUS: f32 = 25.0;
const ORBIT_HEIGHT: f32 = 18.0;
const TARGET_Y: f32 = 1.0;
const ORBIT_SPEED_RAD_PER_SEC: f32 = 0.08;

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position: [ORBIT_RADIUS, ORBIT_HEIGHT, 0.0],
            target: [0.0, TARGET_Y, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0, aspect, near: 0.1, far: 800.0,
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
            [-dot3(s, self.position), -dot3(u, self.position), dot3(f, self.position), 1.0],
        ]
    }

    pub fn get_projection_matrix(&self) -> [[f32; 4]; 4] {
        let fov_rad = self.fov.to_radians();
        let f = 1.0 / (fov_rad / 2.0).tan();
        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, -f, 0.0, 0.0],
            [0.0, 0.0, self.far / (self.near - self.far), -1.0],
            [0.0, 0.0, (self.near * self.far) / (self.near - self.far), 0.0],
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

    pub fn extract_frustum_planes(&self) -> [[f32; 4]; 6] {
        let vp = multiply_matrices(self.get_view_matrix(), self.get_projection_matrix());
        let row = |r: usize| [vp[0][r], vp[1][r], vp[2][r], vp[3][r]];
        let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
        let add = |a: [f32;4], b: [f32;4]| [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]];
        let sub = |a: [f32;4], b: [f32;4]| [a[0]-b[0],a[1]-b[1],a[2]-b[2],a[3]-b[3]];
        let norm = |mut p: [f32;4]| {
            let len = (p[0]*p[0]+p[1]*p[1]+p[2]*p[2]).sqrt();
            if len > 0.0 { p[0]/=len; p[1]/=len; p[2]/=len; p[3]/=len; }
            p
        };
        [
            norm(add(r3, r0)), norm(sub(r3, r0)),
            norm(add(r3, r1)), norm(sub(r3, r1)),
            norm(add(r3, r2)), norm(sub(r3, r2)),
        ]
    }
}

// ===== Scene =====

pub struct Scene {
    pub camera: Camera,
    pub rotation: f32,
    pub frame_number: u64,
    prev_camera_pos: [f32; 3],
}

impl Scene {
    pub fn new(aspect: f32) -> Self {
        let camera = Camera::new(aspect);
        let pos = camera.position;
        Self { camera, rotation: 0.0, frame_number: 0, prev_camera_pos: pos }
    }

    pub fn update(&mut self, delta_time: f32) {
        self.prev_camera_pos = self.camera.position;
        self.rotation += delta_time * ORBIT_SPEED_RAD_PER_SEC;
        self.frame_number += 1;
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let center_x = 32.0;
        let center_z = 32.0;
        self.camera.position = [
            center_x + ORBIT_RADIUS * cos_r, ORBIT_HEIGHT,
            center_z + ORBIT_RADIUS * sin_r,
        ];
        self.camera.target = [center_x, TARGET_Y, center_z];
    }

    pub fn camera_velocity_xz(&self) -> [f32; 2] {
        [
            self.camera.position[0] - self.prev_camera_pos[0],
            self.camera.position[2] - self.prev_camera_pos[2],
        ]
    }

    pub fn get_ubo(&self, material_id: u32) -> UniformBufferObject {
        UniformBufferObject::new(
            identity_matrix(), self.camera.get_view_matrix(),
            self.camera.get_projection_matrix(), material_id,
        )
    }
}

// ===== Math helpers =====

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if len > 0.0 { [v[0]/len, v[1]/len, v[2]/len] } else { v }
}

fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn dot3(a: [f32;3], b: [f32;3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }

fn sub3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }

fn length3(v: [f32; 3]) -> f32 { (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt() }

pub fn identity_matrix() -> [[f32;4];4] {
    [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]
}

pub fn create_rotation_matrix(angle: f32, axis: [f32; 3]) -> [[f32; 4]; 4] {
    let axis = normalize(axis);
    let s = angle.sin(); let c = angle.cos(); let oc = 1.0 - c;
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

pub fn multiply_matrices(a: [[f32;4];4], b: [[f32;4];4]) -> [[f32;4];4] {
    let mut r = [[0.0;4];4];
    for i in 0..4 { for j in 0..4 { for k in 0..4 { r[i][j] += a[i][k]*b[k][j]; } } }
    r
}