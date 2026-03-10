use ash::vk;

// ===== Vertex =====

/// Phase 4: Extended vertex layout (60 bytes).
///
/// Added `tangent: [f32; 4]` (xyz = tangent direction, w = handedness ±1)
/// after `normal` for TBN basis construction. Bitangent is reconstructed
/// in the vertex shader as `cross(N, T.xyz) * T.w`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],   // offset 0,  location 0
    pub normal: [f32; 3],     // offset 12, location 1
    pub tangent: [f32; 4],    // offset 24, location 2  [Phase 4]
    pub uv: [f32; 2],         // offset 40, location 3
    pub color: [f32; 3],      // offset 48, location 4
}

const _: () = assert!(std::mem::size_of::<Vertex>() == 60);

impl Vertex {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0], color }
    }

    pub fn with_normal(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, normal, tangent: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0], color }
    }

    pub fn full(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 3]) -> Self {
        Self { position, normal, tangent: [1.0, 0.0, 0.0, 1.0], uv, color }
    }

    /// Full constructor including explicit tangent vector.
    pub fn with_tangent(
        position: [f32; 3], normal: [f32; 3], tangent: [f32; 4],
        uv: [f32; 2], color: [f32; 3],
    ) -> Self {
        Self { position, normal, tangent, uv, color }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 5] {
        [
            // location 0: vec3 position @ offset 0
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(0).format(vk::Format::R32G32B32_SFLOAT).offset(0),
            // location 1: vec3 normal @ offset 12
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(1).format(vk::Format::R32G32B32_SFLOAT).offset(12),
            // location 2: vec4 tangent @ offset 24  [Phase 4]
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(2).format(vk::Format::R32G32B32A32_SFLOAT).offset(24),
            // location 3: vec2 uv @ offset 40
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(3).format(vk::Format::R32G32_SFLOAT).offset(40),
            // location 4: vec3 color @ offset 48
            vk::VertexInputAttributeDescription::default()
                .binding(0).location(4).format(vk::Format::R32G32B32_SFLOAT).offset(48),
        ]
    }
}

// ===== Per-Draw UBO (Phase 4: slimmed — view/proj moved to set 0 GlobalUbo) =====

/// Per-draw dynamic UBO pushed into the ring buffer per draw call.
///
/// Phase 4: removed redundant `view` and `proj` matrices which duplicate
/// the per-frame UBO at set 0 binding 0.  Shadow and probe passes now
/// push a per-face GlobalUbo and rebind set 0 instead.
///
/// Layout (80 bytes, std140):
///   mat4 model       (64 bytes)
///   uint materialId  (4 bytes)
///   uint _pad[3]     (12 bytes)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PerDrawUbo {
    pub model: [[f32; 4]; 4],
    pub material_id: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<PerDrawUbo>() == 80);

impl PerDrawUbo {
    pub fn new(model: [[f32; 4]; 4], material_id: u32) -> Self {
        Self { model, material_id, _pad: [0; 3] }
    }
}

/// Backward-compatible alias.  Old code that references `UniformBufferObject`
/// can migrate incrementally.
pub type UniformBufferObject = PerDrawUbo;

// ===== Input =====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputAction {
    SpawnLight,
    SpawnGeometry,
}

/// Per-frame input state built by main.rs from SDL2 events.
#[derive(Debug, Clone)]
pub struct InputState {
    /// W — move along camera forward.
    pub move_forward: bool,
    /// S — move opposite camera forward.
    pub move_back: bool,
    /// A — strafe left.
    pub move_left: bool,
    /// D — strafe right.
    pub move_right: bool,
    /// E — ascend (world +Y).
    pub move_up: bool,
    /// Q — descend (world -Y).
    pub move_down: bool,
    /// Left Shift — multiply speed.
    pub fast: bool,
    /// Right mouse button held — enables mouse look.
    pub mouse_look: bool,
    /// Accumulated mouse X delta this frame (pixels).
    pub mouse_dx: f32,
    /// Accumulated mouse Y delta this frame (pixels).
    pub mouse_dy: f32,
    /// Scroll wheel delta (positive = up/forward).
    pub scroll_y: f32,
    /// Discrete actions triggered by single key presses this frame.
    pub actions: Vec<InputAction>,
}

impl InputState {
    pub fn empty() -> Self {
        Self {
            move_forward: false, move_back: false,
            move_left: false, move_right: false,
            move_up: false, move_down: false,
            fast: false, mouse_look: false,
            mouse_dx: 0.0, mouse_dy: 0.0,
            scroll_y: 0.0, actions: Vec::new(),
        }
    }
}

// ===== Camera (UE5-style fly mode) =====

/// Fly-camera speed defaults.
const FLY_START_POS: [f32; 3] = [0.0, 12.0, 30.0];
/// Initial yaw: π = looking toward -Z (toward world origin from +Z side).
const FLY_START_YAW: f32 = std::f32::consts::PI;
/// Slight downward tilt so ground plane is visible on start.
const FLY_START_PITCH: f32 = -0.15;
const FLY_DEFAULT_SPEED: f32 = 20.0;
const FLY_FAST_MULTIPLIER: f32 = 3.0;
const FLY_LOOK_SENSITIVITY: f32 = 0.003;
const FLY_SCROLL_SPEED_STEP: f32 = 2.0;
const FLY_MIN_SPEED: f32 = 1.0;
const FLY_MAX_SPEED: f32 = 200.0;

pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    /// Horizontal rotation (radians). 0 = +Z, π = -Z.
    pub yaw: f32,
    /// Vertical rotation (radians). Clamped ±~89°.
    pub pitch: f32,
    /// Base movement speed (m/s). Adjusted by scroll wheel.
    pub move_speed: f32,
    /// Mouse-look sensitivity (radians per pixel).
    pub look_sensitivity: f32,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        let fwd = fly_forward(FLY_START_YAW, FLY_START_PITCH);
        Self {
            position: FLY_START_POS,
            target: add3(FLY_START_POS, fwd),
            up: [0.0, 1.0, 0.0],
            yaw: FLY_START_YAW,
            pitch: FLY_START_PITCH,
            move_speed: FLY_DEFAULT_SPEED,
            look_sensitivity: FLY_LOOK_SENSITIVITY,
            fov: 60.0, aspect, near: 0.1, far: 200.0,
        }
    }

    /// Per-frame update from input.  Mirrors UE5 viewport fly controls:
    /// RMB+mouse = look, WASD = move in camera space, Q/E = world vertical,
    /// Shift = fast, scroll = adjust base speed.
    pub fn update_fly(&mut self, input: &InputState, dt: f32) {
        // ---- Mouse look (only while RMB held) ----
        if input.mouse_look {
            self.yaw -= input.mouse_dx * self.look_sensitivity;
            self.pitch -= input.mouse_dy * self.look_sensitivity;
            self.pitch = self.pitch.clamp(-1.553, 1.553); // ±89°
        }

        // ---- Scroll wheel: adjust base speed ----
        if input.scroll_y.abs() > 0.01 {
            self.move_speed = (self.move_speed + input.scroll_y * FLY_SCROLL_SPEED_STEP)
                .clamp(FLY_MIN_SPEED, FLY_MAX_SPEED);
        }

        // ---- Movement ----
        let fwd = fly_forward(self.yaw, self.pitch);
        let right = normalize(cross3(fwd, self.up));

        let speed = self.move_speed
            * if input.fast { FLY_FAST_MULTIPLIER } else { 1.0 }
            * dt;

        let mut delta = [0.0f32; 3];

        // Forward/back move along full camera forward (including pitch).
        if input.move_forward { delta = add3(delta, scale3(fwd, speed)); }
        if input.move_back    { delta = add3(delta, scale3(fwd, -speed)); }

        // Strafe moves along camera right (horizontal).
        if input.move_right { delta = add3(delta, scale3(right, speed)); }
        if input.move_left  { delta = add3(delta, scale3(right, -speed)); }

        // Vertical movement in world space (+Y up).
        if input.move_up   { delta[1] += speed; }
        if input.move_down { delta[1] -= speed; }

        self.position = add3(self.position, delta);
        self.target = add3(self.position, fwd);
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

/// Camera forward vector from yaw/pitch (spherical → cartesian).
/// yaw=0 → +Z, yaw=π → -Z.  pitch>0 → up, pitch<0 → down.
fn fly_forward(yaw: f32, pitch: f32) -> [f32; 3] {
    let cp = pitch.cos();
    [yaw.sin() * cp, pitch.sin(), yaw.cos() * cp]
}

// ===== Scene =====

pub struct Scene {
    pub camera: Camera,
    pub frame_number: u64,
    prev_camera_pos: [f32; 3],
}

impl Scene {
    pub fn new(aspect: f32) -> Self {
        let camera = Camera::new(aspect);
        let pos = camera.position;
        Self { camera, frame_number: 0, prev_camera_pos: pos }
    }

    /// Per-frame update.  Drives camera from input state.
    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.prev_camera_pos = self.camera.position;
        self.camera.update_fly(input, dt);
        self.frame_number += 1;
    }

    pub fn camera_velocity_xz(&self) -> [f32; 2] {
        [
            self.camera.position[0] - self.prev_camera_pos[0],
            self.camera.position[2] - self.prev_camera_pos[2],
        ]
    }

    /// Phase 4: returns PerDrawUbo (model + material_id only).
    pub fn get_ubo(&self, material_id: u32) -> PerDrawUbo {
        PerDrawUbo::new(identity_matrix(), material_id)
    }
}

// ===== Math helpers =====

pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if len > 0.0 { [v[0]/len, v[1]/len, v[2]/len] } else { v }
}

pub fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

pub fn dot3(a: [f32;3], b: [f32;3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }

pub fn sub3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }

pub fn add3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }

pub fn scale3(v: [f32;3], s: f32) -> [f32;3] { [v[0]*s, v[1]*s, v[2]*s] }

pub fn length3(v: [f32; 3]) -> f32 { (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt() }

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