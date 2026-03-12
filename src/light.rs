//! Phase 2: Lighting Infrastructure
//!
//! Provides:
//! - `Light` / `GpuLight`: CPU and GPU light representations.
//! - `LightManager`: register/unregister by sector, frustum cull, SSBO upload.
//! - `ShadowBudgetManager`: scoring, slot assignment with hysteresis.
//! - `ShadowAtlas`: cube map array for point light shadows, per-face
//!   framebuffers, view matrix generation.
//! - `ClusterParams`: UBO for the cluster assignment compute pass.
//!
//! Light SSBO layout (set 0, binding 1):
//!   [light_count: u32, _pad: [u32;3], GpuLight × N]
//!
//! Cluster SSBO (set 0, binding 2):
//!   [GpuCluster × total_clusters]
//!
//! Light index SSBO (set 0, binding 3):
//!   [global_count: u32, _pad: [u32;3], indices: u32 × capacity]

use ash::{vk, Device};
use std::collections::HashMap;

use crate::memory::{GpuAllocator, ImageHandle, MemoryLocation};
use crate::world::SectorCoord;

// ====================================================================
//  Constants
// ====================================================================

/// Maximum active lights uploaded to the GPU per frame.
pub const MAX_LIGHTS: usize = 4096;

/// Maximum lights tested per cluster (hard cap in compute shader).
pub const MAX_LIGHTS_PER_CLUSTER: usize = 128;

/// Number of point light shadow slots (cube map slices).
pub const MAX_SHADOW_SLOTS: usize = 32;

/// Shadow map resolution per face (pixels).
pub const SHADOW_MAP_SIZE: u32 = 256;

/// Near plane for shadow projection.
pub const SHADOW_NEAR: f32 = 0.1;

/// Cluster grid dimensions for 1080p with logarithmic depth slicing.
pub const CLUSTER_X: u32 = 16;
pub const CLUSTER_Y: u32 = 9;
pub const CLUSTER_Z: u32 = 32; // Phase 8C.3: was 24, finer depth binning
pub const TOTAL_CLUSTERS: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;

/// Max entries in the global light index buffer (clusters × max per cluster).
/// Over-provisioned; actual usage is typically 10-20% of this.
pub const LIGHT_INDEX_CAPACITY: u32 = TOTAL_CLUSTERS * MAX_LIGHTS_PER_CLUSTER as u32;

/// Hysteresis bonus score for lights retaining shadow slots across frames.
const SHADOW_HYSTERESIS_BONUS: f32 = 2.0;

pub const SUN_COLOR: [f32; 3] = [1.0, 0.95, 0.85]; 
pub const SUN_INTENSITY: f32 = 2.5;
pub const SUN_DIR: [f32; 3] = [0.6, -0.4, 0.5];
/// Resolution of the sun shadow map (single 2D texture).
pub const SUN_SHADOW_SIZE: u32 = 2048;

/// Half-width of the sun shadow orthographic projection (meters).
/// Covers 300m × 300m — matches the streaming radius.
pub const SUN_SHADOW_EXTENT: f32 = 300.0;

/// Far plane for the sun shadow orthographic projection.
pub const SUN_SHADOW_FAR: f32 = 600.0;

/// Near plane.
pub const SUN_SHADOW_NEAR: f32 = 1.0;

// ====================================================================
//  Light Types
// ====================================================================

/// Light type discriminant (matches GPU bitfield bits 0-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightType {
    Point = 0,
    Spot = 1,
    Directional = 2,
}

/// Phase 8C.1: Light update category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightCategory {
    /// Updated per-frame. Eligible for real-time shadows.
    Dynamic,
    /// Uploaded once when sector loads. No shadows. Never updated.
    /// Removed when sector evicts. Skips shadow sampling in shader.
    Baked,
}

/// CPU-side light definition.  Registered per-sector, culled per-frame.
#[derive(Debug, Clone)]
pub struct Light {
    pub light_type: LightType,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    /// Effective radius (point/spot); 0.0 for directional.
    pub radius: f32,
    /// Attenuation exponent (default 2.0 = inverse-square).
    pub falloff: f32,
    /// Cosine of outer cone angle (spot lights).  1.0 = point.
    pub cos_outer_angle: f32,
    /// Cosine of inner cone angle (spot lights).  1.0 = point.
    pub cos_inner_angle: f32,
    /// Can this light cast shadows?
    pub shadow_capable: bool,
    /// Phase 8C.1: Light update category (dynamic vs baked).
    pub category: LightCategory,
    /// Sector that registered this light (for bulk unregister).
    pub(crate) source_sector: SectorCoord,
}

impl Light {
    /// Create a point light at `position`.
    pub fn point(
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
    ) -> Self {
        Self {
            light_type: LightType::Point,
            position,
            direction: [0.0, -1.0, 0.0],
            color,
            intensity,
            radius,
            falloff: 2.0,
            cos_outer_angle: -1.0,
            cos_inner_angle: -1.0,
            shadow_capable: true,
            category: LightCategory::Dynamic,
            source_sector: (0, 0),
        }
    }

    /// Create a directional light (e.g. sun).
    pub fn directional(
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
        .sqrt();
        let d = if len > 0.0 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            light_type: LightType::Directional,
            position: [0.0; 3],
            direction: d,
            color,
            intensity,
            radius: 0.0,
            falloff: 1.0,
            cos_outer_angle: -1.0,
            cos_inner_angle: -1.0,
            shadow_capable: false,
            category: LightCategory::Dynamic,
            source_sector: (0, 0),
        }
    }

    /// Create a spot light.
    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
        inner_angle_deg: f32,
        outer_angle_deg: f32,
    ) -> Self {
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
        .sqrt();
        let d = if len > 0.0 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            light_type: LightType::Spot,
            position,
            direction: d,
            color,
            intensity,
            radius,
            falloff: 2.0,
            cos_outer_angle: outer_angle_deg.to_radians().cos(),
            cos_inner_angle: inner_angle_deg.to_radians().cos(),
            shadow_capable: true,
            category: LightCategory::Dynamic,
            source_sector: (0, 0),
        }
    }

    /// Phase 8C.1: Create a baked point light. No shadows, no per-frame updates.
    pub fn baked_point(
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
    ) -> Self {
        Self {
            light_type: LightType::Point,
            position,
            direction: [0.0, -1.0, 0.0],
            color,
            intensity,
            radius,
            falloff: 2.0,
            cos_outer_angle: -1.0,
            cos_inner_angle: -1.0,
            shadow_capable: false,
            category: LightCategory::Baked,
            source_sector: (0, 0),
        }
    }
}

// ====================================================================
//  GPU-side structs (must match GLSL layout exactly)
// ====================================================================

/// GPU light struct — 64 bytes, std430 layout.
///
/// ```glsl
/// struct GpuLight {
///     vec4  position_radius;      // xyz = pos, w = radius
///     vec4  direction_cos_outer;   // xyz = dir, w = cos(outer_angle)
///     vec4  color_intensity;       // xyz = color, w = intensity
///     uint  type_flags;            // bits 0-1: type, bit 2: shadow_capable, bit 3: baked (8C.1)
///     uint  shadow_index;          // cube map slice (0xFFFFFFFF = none)
///     float falloff;
///     float cos_inner_angle;
/// };
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuLight {
    pub position_radius: [f32; 4],
    pub direction_cos_outer: [f32; 4],
    pub color_intensity: [f32; 4],
    pub type_flags: u32,
    pub shadow_index: u32,
    pub falloff: f32,
    pub cos_inner_angle: f32,
}

const _: () = assert!(std::mem::size_of::<GpuLight>() == 64);

impl GpuLight {
    pub fn from_light(light: &Light, shadow_index: Option<u32>) -> Self {
        // Phase 8C.1: bit 3 encodes baked status
        let type_flags = light.light_type as u32
            | if light.shadow_capable { 1 << 2 } else { 0 }
            | if light.category == LightCategory::Baked { 1 << 3 } else { 0 };

        Self {
            position_radius: [
                light.position[0],
                light.position[1],
                light.position[2],
                light.radius,
            ],
            direction_cos_outer: [
                light.direction[0],
                light.direction[1],
                light.direction[2],
                light.cos_outer_angle,
            ],
            color_intensity: [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ],
            type_flags,
            shadow_index: shadow_index.unwrap_or(u32::MAX),
            falloff: light.falloff,
            cos_inner_angle: light.cos_inner_angle,
        }
    }
}

/// Light SSBO header (16 bytes, precedes the GpuLight array).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LightSsboHeader {
    pub light_count: u32,
    pub _pad: [u32; 3],
}

/// GPU cluster entry — 8 bytes, std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuCluster {
    pub offset: u32,
    pub count: u32,
}

/// Light index SSBO header (16 bytes, precedes the u32 index array).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LightIndexHeader {
    pub global_count: u32,
    pub _pad: [u32; 3],
}

/// Cluster params UBO — std140 layout.
///
/// ```glsl
/// layout(set = 0, binding = 4) uniform ClusterParams {
///     mat4  view;
///     mat4  proj;
///     mat4  inv_proj;
///     uvec4 grid_size;   // x, y, z, total
///     vec4  z_params;    // near, far, log_ratio, _pad
///     uvec2 screen_size;
///     uint  light_count;
///     uint  _pad;
/// };
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClusterParamsUbo {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub grid_size: [u32; 4],
    pub z_params: [f32; 4],
    pub screen_size: [u32; 2],
    pub light_count: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<ClusterParamsUbo>() == 240);

impl ClusterParamsUbo {
    pub fn new(
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        near: f32,
        far: f32,
        screen_w: u32,
        screen_h: u32,
        light_count: u32,
    ) -> Self {
        let inv_proj = invert_projection(&proj);
        let log_ratio = (far / near).ln() / CLUSTER_Z as f32;

        Self {
            view,
            proj,
            inv_proj,
            grid_size: [CLUSTER_X, CLUSTER_Y, CLUSTER_Z, TOTAL_CLUSTERS],
            z_params: [near, far, log_ratio, 0.0],
            screen_size: [screen_w, screen_h],
            light_count,
            _pad: 0,
        }
    }
}

/// Push constant for shadow pass — 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShadowPushConstants {
    pub light_pos: [f32; 3],
    pub light_radius: f32,
}

// ====================================================================
//  LightManager
// ====================================================================

/// CPU-side light management.
///
/// Lights are registered per-sector.  Each frame, the manager frustum-culls,
/// sorts by distance, caps at `MAX_LIGHTS`, and produces a `Vec<GpuLight>`
/// for SSBO upload.
pub struct LightManager {
    /// All registered lights.
    lights: Vec<Light>,
    /// Sector → index range in `lights`.
    sector_ranges: HashMap<SectorCoord, Vec<usize>>,
    /// Per-frame output after cull (indices into `lights`).
    active_indices: Vec<usize>,
    /// Per-frame GPU data ready for upload.
    gpu_lights: Vec<GpuLight>,
    /// Active count after cull.
    active_count: u32,
}

impl LightManager {
    pub fn new() -> Self {
        Self {
            lights: Vec::with_capacity(1024),
            sector_ranges: HashMap::new(),
            active_indices: Vec::with_capacity(MAX_LIGHTS),
            gpu_lights: Vec::with_capacity(MAX_LIGHTS),
            active_count: 0,
        }
    }

    /// Register lights from a sector.  Returns indices into the global
    /// light array (for optional external tracking).
    pub fn register(&mut self, coord: SectorCoord, lights: Vec<Light>) -> Vec<usize> {
        let mut indices = Vec::with_capacity(lights.len());
        for mut light in lights {
            light.source_sector = coord;
            let idx = self.lights.len();
            self.lights.push(light);
            indices.push(idx);
        }
        self.sector_ranges.insert(coord, indices.clone());
        indices
    }

    /// Bulk-remove all lights registered by a sector.
    ///
    /// Uses swap-remove for O(1) per light.  Updates sector_ranges to
    /// reflect swapped indices.
    pub fn unregister(&mut self, coord: SectorCoord) {
        let Some(indices) = self.sector_ranges.remove(&coord) else {
            return;
        };

        // Sort descending so swap-removes don't invalidate earlier indices.
        let mut sorted = indices;
        sorted.sort_unstable_by(|a, b| b.cmp(a));

        for idx in sorted {
            if idx >= self.lights.len() {
                continue;
            }

            // swap_remove: moves last element into `idx`.
            self.lights.swap_remove(idx);

            // If we moved a light, update its sector's index mapping.
            if idx < self.lights.len() {
                let moved_sector = self.lights[idx].source_sector;
                if let Some(mapping) = self.sector_ranges.get_mut(&moved_sector) {
                    // The moved light was at position `self.lights.len()` (before remove),
                    // now it's at `idx`.
                    if let Some(pos) = mapping.iter().position(|&i| i == self.lights.len()) {
                        mapping[pos] = idx;
                    }
                }
            }
        }
    }

    /// Per-frame cull and sort.  Returns the number of active lights.
    ///
    /// Pipeline:
    /// 1. Frustum cull (sphere-vs-frustum for point/spot; directional always passes).
    /// 2. Distance sort (nearest camera first).
    /// 3. Budget cap at `MAX_LIGHTS`.
    pub fn cull_and_sort(
        &mut self,
        camera_pos: [f32; 3],
        frustum_planes: &[[f32; 4]; 6],
        shadow_assignments: &HashMap<usize, u32>,
    ) -> u32 {
        self.active_indices.clear();
        self.gpu_lights.clear();

        // 1. Frustum cull.
        for (i, light) in self.lights.iter().enumerate() {
            if light.light_type == LightType::Directional {
                // Directional lights always pass.
                self.active_indices.push(i);
                continue;
            }

            // Sphere-frustum test.
            let pos = light.position;
            let r = light.radius;
            let mut visible = true;
            for plane in frustum_planes {
                let dist = plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3];
                if dist < -r {
                    visible = false;
                    break;
                }
            }
            if visible {
                self.active_indices.push(i);
            }
        }

        // 2. Distance sort (nearest first).
        self.active_indices.sort_unstable_by(|&a, &b| {
            let la = &self.lights[a];
            let lb = &self.lights[b];

            // Directional lights always first.
            if la.light_type == LightType::Directional {
                return std::cmp::Ordering::Less;
            }
            if lb.light_type == LightType::Directional {
                return std::cmp::Ordering::Greater;
            }

            let da = dist_sq(camera_pos, la.position);
            let db = dist_sq(camera_pos, lb.position);
            da.total_cmp(&db)
        });

        // 3. Budget cap.
        self.active_indices.truncate(MAX_LIGHTS);

        // 4. Build GPU array.
        for &idx in &self.active_indices {
            let light = &self.lights[idx];
            let shadow_slot = shadow_assignments.get(&idx).copied();
            self.gpu_lights.push(GpuLight::from_light(light, shadow_slot));
        }

        self.active_count = self.gpu_lights.len() as u32;
        self.active_count
    }

    /// Zero-allocation SSBO header for direct upload.
    /// Pair with [`gpu_lights()`] to write header + body into mapped
    /// GPU memory without an intermediate `Vec<u8>` allocation.
    #[inline]
    pub fn ssbo_header(&self) -> LightSsboHeader {
        LightSsboHeader { light_count: self.active_count, _pad: [0; 3] }
    }

    /// Immutable slice of the per-frame GPU light array (post-cull).
    /// Use with [`ssbo_header()`] for zero-alloc SSBO upload.
    #[inline]
    pub fn gpu_lights(&self) -> &[GpuLight] {
        &self.gpu_lights
    }

    /// Raw bytes for the light SSBO upload (header + GpuLight array).
    ///
    /// Allocating variant retained for diagnostics / tooling.  The hot
    /// path should use [`ssbo_header()`] + [`gpu_lights()`] via
    /// `FrameLightingBuffers::upload_lights_direct()` instead.
    #[allow(dead_code)]
    pub fn ssbo_bytes(&self) -> Vec<u8> {
        let header = LightSsboHeader {
            light_count: self.active_count,
            _pad: [0; 3],
        };

        let header_bytes = std::mem::size_of::<LightSsboHeader>();
        let body_bytes = self.gpu_lights.len() * std::mem::size_of::<GpuLight>();
        let mut buf = Vec::with_capacity(header_bytes + body_bytes);

        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(&header as *const _ as *const u8, header_bytes)
        });
        if !self.gpu_lights.is_empty() {
            buf.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    self.gpu_lights.as_ptr() as *const u8,
                    body_bytes,
                )
            });
        }

        buf
    }

    /// Active light count after last cull.
    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Active indices (into `self.lights`) after last cull.
    pub fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }

    /// Access a light by global index.
    pub fn get(&self, index: usize) -> Option<&Light> {
        self.lights.get(index)
    }

    /// Mutable access to a light by global index.
    /// Used for per-frame position updates on dynamic lights.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Light> {
        self.lights.get_mut(index)
    }

    /// Total registered lights.
    pub fn total_count(&self) -> usize {
        self.lights.len()
    }
}

// ====================================================================
//  ShadowBudgetManager
// ====================================================================

/// Assigns shadow atlas slots to the highest-priority lights each frame.
///
/// Scoring: inverse distance² × projected screen radius + hysteresis.
/// Directional sun always keeps its CSM cascades (not competing for
/// point-light slots).
///
/// Phase 8C.2: Includes per-slot cache state for temporal shadow caching.
/// Phase 8C.5: Adaptive budget driven by profiler feedback.
pub struct ShadowBudgetManager {
    /// Current slot assignments: slot index → global light index.
    slot_to_light: [Option<usize>; MAX_SHADOW_SLOTS],
    /// Reverse: global light index → slot index.
    light_to_slot: HashMap<usize, u32>,
    /// Phase 8C.2: Per-slot cache state for temporal shadow caching.
    slot_cache: Vec<ShadowSlotCache>,
    /// Phase 8C.5: Target shadow pass time budget (ms). Default: 4.0.
    pub time_budget_ms: f32,
    /// Phase 8C.5: Previous frame's measured shadow pass time (from GpuProfiler).
    pub last_shadow_time_ms: f32,
    /// Phase 8C.5: Current effective max slots (may be less than MAX_SHADOW_SLOTS).
    effective_max_slots: usize,
}

/// Phase 8C.2: Per-slot shadow cache state.
#[derive(Debug, Clone)]
pub struct ShadowSlotCache {
    /// Global light index occupying this slot (None = empty).
    pub light_idx: Option<usize>,
    /// Frame number when this slot was last rendered.
    pub last_rendered_frame: u64,
    /// Position hash at time of last render.
    pub position_hash: u64,
    /// Radius at time of last render.
    pub cached_radius: f32,
    /// True if slot content is stale and needs re-render.
    pub dirty: bool,
}

impl ShadowSlotCache {
    pub fn empty() -> Self {
        Self {
            light_idx: None,
            last_rendered_frame: 0,
            position_hash: 0,
            cached_radius: 0.0,
            dirty: true,
        }
    }
}

/// Hash a light position for cache invalidation.
/// Uses bit-cast to u64 for exact equality (no floating point tolerance).
fn hash_position(pos: [f32; 3]) -> u64 {
    let a = (pos[0].to_bits() as u64) ^ ((pos[1].to_bits() as u64) << 20);
    let b = (pos[2].to_bits() as u64) << 40;
    a ^ b
}

impl ShadowBudgetManager {
    pub fn new() -> Self {
        Self {
            slot_to_light: [None; MAX_SHADOW_SLOTS],
            light_to_slot: HashMap::new(),
            slot_cache: (0..MAX_SHADOW_SLOTS)
                .map(|_| ShadowSlotCache::empty())
                .collect(),
            time_budget_ms: 4.0,
            last_shadow_time_ms: 0.0,
            effective_max_slots: MAX_SHADOW_SLOTS,
        }
    }

    /// Phase 8C.5: Called each frame BEFORE assign(). Adjusts effective_max_slots
    /// based on last frame's measured shadow time.
    pub fn adapt_budget(&mut self) {
        if self.last_shadow_time_ms > self.time_budget_ms * 1.2 {
            // Over budget by >20%: reduce slots.
            self.effective_max_slots = (self.effective_max_slots.saturating_sub(1)).max(4);
        } else if self.last_shadow_time_ms < self.time_budget_ms * 0.6
            && self.effective_max_slots < MAX_SHADOW_SLOTS
        {
            // Under budget by >40%: allow one more slot.
            self.effective_max_slots += 1;
        }
    }

    /// Phase 8C.5: Feed the profiler's shadow pass time for next frame's adaptation.
    pub fn set_last_shadow_time(&mut self, ms: f32) {
        self.last_shadow_time_ms = ms;
    }

    /// Current effective max slots (for debug overlay).
    pub fn effective_max_slots(&self) -> usize {
        self.effective_max_slots
    }

    /// Score and assign shadow slots for the current frame's active lights.
    ///
    /// Returns a map: global light index → shadow slot.
    pub fn assign(
        &mut self,
        light_manager: &LightManager,
        camera_pos: [f32; 3],
    ) -> HashMap<usize, u32> {
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        for &idx in light_manager.active_indices() {
            let Some(light) = light_manager.get(idx) else { continue };

            // Only shadow-capable point/spot lights compete for slots.
            // Phase 8C.1: baked lights never get shadow slots.
            if !light.shadow_capable
                || light.light_type == LightType::Directional
                || light.category == LightCategory::Baked
            {
                continue;
            }

            let d2 = dist_sq(camera_pos, light.position).max(1.0);

            // Score: inverse distance² × radius (screen-space proxy).
            let mut score = (light.radius * light.radius) / d2;

            // Hysteresis: bonus for lights already holding a slot.
            if self.light_to_slot.contains_key(&idx) {
                score += SHADOW_HYSTERESIS_BONUS;
            }

            candidates.push((idx, score));
        }

        // Sort descending by score.
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Clear previous assignments.
        let prev_light_to_slot = std::mem::take(&mut self.light_to_slot);
        self.slot_to_light = [None; MAX_SHADOW_SLOTS];

        // Assign top N candidates.
        let mut next_slot: u32 = 0;
        let mut assignments = HashMap::new();

        for (light_idx, _score) in candidates {
            // Phase 8C.5: use adaptive effective_max_slots instead of MAX_SHADOW_SLOTS.
            if next_slot as usize >= self.effective_max_slots {
                break;
            }

            // Prefer re-using the light's previous slot (reduces shadow
            // flicker from re-rendering a different atlas slice).
            let slot = if let Some(&prev) = prev_light_to_slot.get(&light_idx) {
                if (prev as usize) < self.effective_max_slots
                    && self.slot_to_light[prev as usize].is_none()
                {
                    prev
                } else {
                    // Previous slot taken or out of budget; find next free.
                    while (next_slot as usize) < self.effective_max_slots
                        && self.slot_to_light[next_slot as usize].is_some()
                    {
                        next_slot += 1;
                    }
                    if next_slot as usize >= self.effective_max_slots {
                        break;
                    }
                    next_slot
                }
            } else {
                while (next_slot as usize) < self.effective_max_slots
                    && self.slot_to_light[next_slot as usize].is_some()
                {
                    next_slot += 1;
                }
                if next_slot as usize >= self.effective_max_slots {
                    break;
                }
                next_slot
            };

            self.slot_to_light[slot as usize] = Some(light_idx);
            self.light_to_slot.insert(light_idx, slot);
            assignments.insert(light_idx, slot);
        }

        assignments
    }

    /// Iterate (slot, global_light_index) for all assigned lights.
    pub fn assigned_slots(&self) -> impl Iterator<Item = (u32, usize)> + '_ {
        self.slot_to_light
            .iter()
            .enumerate()
            .filter_map(|(slot, opt)| opt.map(|idx| (slot as u32, idx)))
    }

    /// Number of active shadow slots this frame.
    pub fn active_shadow_count(&self) -> u32 {
        self.slot_to_light.iter().filter(|s| s.is_some()).count() as u32
    }

    // ================================================================
    //  Phase 8C.2: Shadow cache methods
    // ================================================================

    /// Update cache state after shadow slot assignment.
    /// Call AFTER assign() each frame.
    pub fn update_cache(
        &mut self,
        light_manager: &LightManager,
        current_frame: u64,
    ) {
        for (slot, opt_light) in self.slot_to_light.iter().enumerate() {
            let cache = &mut self.slot_cache[slot];

            match opt_light {
                None => {
                    // Slot freed — mark empty.
                    *cache = ShadowSlotCache::empty();
                }
                Some(light_idx) => {
                    let Some(light) = light_manager.get(*light_idx) else {
                        cache.dirty = true;
                        continue;
                    };

                    let pos_hash = hash_position(light.position);

                    if cache.light_idx != Some(*light_idx)
                        || cache.position_hash != pos_hash
                        || (cache.cached_radius - light.radius).abs() > 0.001
                    {
                        // Light changed or slot reassigned — must re-render.
                        cache.light_idx = Some(*light_idx);
                        cache.position_hash = pos_hash;
                        cache.cached_radius = light.radius;
                        cache.dirty = true;
                    }
                    // If nothing changed, dirty stays false from last clear.
                }
            }
        }
    }

    /// Check if a shadow slot needs re-rendering this frame.
    pub fn is_slot_dirty(&self, slot: u32) -> bool {
        self.slot_cache.get(slot as usize).map_or(true, |c| c.dirty)
    }

    /// Mark a slot as clean after rendering. Call after shadow pass completes.
    pub fn mark_slot_clean(&mut self, slot: u32, frame: u64) {
        if let Some(cache) = self.slot_cache.get_mut(slot as usize) {
            cache.dirty = false;
            cache.last_rendered_frame = frame;
        }
    }

    /// Mark all slots dirty (e.g., after sector load/unload).
    pub fn invalidate_all(&mut self) {
        for cache in &mut self.slot_cache {
            cache.dirty = true;
        }
    }
}

// ====================================================================
//  ShadowAtlas — cube map array for point light shadows
// ====================================================================

/// Manages the shadow cube map array texture and per-face framebuffers.
///
/// Layout: VkImage with `CUBE_COMPATIBLE` and
/// `arrayLayers = MAX_SHADOW_SLOTS * 6`.  Each slot has 6 faces.
/// Sampled as `samplerCubeArray` in the PBR shader.
pub struct ShadowAtlas {
    pub image: vk::Image,
    pub image_handle: ImageHandle,
    /// Dedicated VkDeviceMemory backing the cube map array.
    pub memory: vk::DeviceMemory,
    /// View for shader sampling (CUBE_ARRAY).
    pub sampling_view: vk::ImageView,
    /// Per-face views for framebuffer attachment (TYPE_2D, one per layer).
    pub face_views: Vec<vk::ImageView>,
    /// Per-face framebuffers (MAX_SHADOW_SLOTS * 6).
    pub face_framebuffers: Vec<vk::Framebuffer>,
    /// Sampler for shadow comparison.
    pub shadow_sampler: vk::Sampler,
}

impl ShadowAtlas {
    /// Allocate the shadow cube map array and create all per-face
    /// framebuffers for the given shadow render pass.
    ///
    /// Performs an initial layout transition of ALL layers from
    /// `UNDEFINED` → `SHADER_READ_ONLY_OPTIMAL` so that unrendered
    /// faces are in a valid layout when sampled by the PBR shader.
    pub fn new(
        device: &Device,
        allocator: &mut GpuAllocator,
        shadow_render_pass: vk::RenderPass,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let total_layers = (MAX_SHADOW_SLOTS * 6) as u32;

        // ---- Create cube map array image ----
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(total_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

        let image = unsafe { device.create_image(&image_info, None)? };
        let mem_req = unsafe { device.get_image_memory_requirements(image) };

        // Sub-allocate from pool.
        let mem_type = allocator.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // We need to bind memory manually since GpuAllocator::create_image
        // creates its own VkImage. We'll use a raw pool_alloc.
        // For simplicity, allocate a dedicated block for the atlas.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        // Create a dummy ImageHandle for tracking (we manage cleanup ourselves).
        let image_handle = ImageHandle(u64::MAX);

        // ---- Initial layout transition: UNDEFINED → SHADER_READ_ONLY_OPTIMAL ----
        //
        // All layers must be in a valid layout before the PBR shader
        // samples the cube array.  Faces that are not rendered to by the
        // shadow pass would otherwise remain UNDEFINED and trigger
        // validation errors.
        unsafe {
            let cmd_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = device.allocate_command_buffers(&cmd_info)?[0];

            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: total_layers,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );

            device.end_command_buffer(cmd)?;

            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            device.queue_wait_idle(queue)?;

            device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        }

        // ---- Sampling view (CUBE_ARRAY) ----
        let sampling_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::CUBE_ARRAY)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: total_layers,
                    }),
                None,
            )?
        };

        // ---- Per-face image views + framebuffers ----
        let mut face_views = Vec::with_capacity(total_layers as usize);
        let mut face_framebuffers = Vec::with_capacity(total_layers as usize);

        for layer in 0..total_layers {
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::D32_SFLOAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: layer,
                            layer_count: 1,
                        }),
                    None,
                )?
            };
            face_views.push(view);

            let attachments = [view];
            let fb = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(shadow_render_pass)
                        .attachments(&attachments)
                        .width(SHADOW_MAP_SIZE)
                        .height(SHADOW_MAP_SIZE)
                        .layers(1),
                    None,
                )?
            };
            face_framebuffers.push(fb);
        }

        // ---- Shadow comparison sampler ----
        let shadow_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .compare_enable(false)
                    .min_lod(0.0)
                    .max_lod(1.0)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE),
                None,
            )?
        };

        let total_mb =
            (SHADOW_MAP_SIZE as u64 * SHADOW_MAP_SIZE as u64 * 4 * total_layers as u64)
                / (1024 * 1024);
        println!(
            "[ShadowAtlas] {}×{}, {} slots ({}×6 layers), ~{} MB",
            SHADOW_MAP_SIZE,
            SHADOW_MAP_SIZE,
            MAX_SHADOW_SLOTS,
            MAX_SHADOW_SLOTS,
            total_mb,
        );

        Ok(Self {
            image,
            image_handle,
            memory,
            sampling_view,
            face_views,
            face_framebuffers,
            shadow_sampler,
        })
    }

    /// Framebuffer for a specific slot + face.
    ///
    /// `face`: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
    pub fn framebuffer(&self, slot: u32, face: u32) -> vk::Framebuffer {
        self.face_framebuffers[(slot * 6 + face) as usize]
    }

    /// Destroy all Vulkan resources.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            for &fb in &self.face_framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            for &view in &self.face_views {
                device.destroy_image_view(view, None);
            }
            device.destroy_image_view(self.sampling_view, None);
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

// ====================================================================
//  SunShadow — 2D depth map for directional light
// ====================================================================

/// Manages a single 2D depth texture for directional (sun) shadows.
///
/// The sun shadow uses the same depth-only render pass as the point
/// light cube maps, but writes standard hardware depth (no linear distance).
/// Uses an orthographic projection centered on the camera, covering
/// `SUN_SHADOW_EXTENT` meters in each direction.
pub struct SunShadow {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub depth_view: vk::ImageView,
    /// View for shader sampling (TYPE_2D).
    pub sampling_view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub sampler: vk::Sampler,
}

impl SunShadow {
    /// Create the sun shadow map resources.
    ///
    /// Reuses the provided shadow render pass (depth-only,
    /// SHADER_READ_ONLY → SHADER_READ_ONLY).
    pub fn new(
        device: &Device,
        allocator: &mut GpuAllocator,
        shadow_render_pass: vk::RenderPass,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = SUN_SHADOW_SIZE;

        // ---- Create 2D depth image ----
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D { width: size, height: size, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None)? };
        let mem_req = unsafe { device.get_image_memory_requirements(image) };

        let mem_type = allocator.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        // ---- Initial layout transition: UNDEFINED → SHADER_READ_ONLY_OPTIMAL ----
        unsafe {
            let cmd_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = device.allocate_command_buffers(&cmd_info)?[0];
            device.begin_command_buffer(cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &[], &[],
                std::slice::from_ref(&barrier));

            device.end_command_buffer(cmd)?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        }

        // ---- Depth view (framebuffer attachment) ----
        let depth_view = unsafe { device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }), None)? };

        // ---- Sampling view (same as depth_view for 2D) ----
        let sampling_view = depth_view;

        // ---- Framebuffer ----
        let fb = unsafe { device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(shadow_render_pass)
                .attachments(std::slice::from_ref(&depth_view))
                .width(size)
                .height(size)
                .layers(1), None)? };

        // ---- Shadow sampler (linear, clamp-to-border white) ----
        let sampler = unsafe { device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .compare_enable(false)
                .min_lod(0.0)
                .max_lod(1.0)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE), None)? };

        let mb = (mem_req.size + 1024 * 1024 - 1) / (1024 * 1024);
        println!("[SunShadow] {}×{} D32_SFLOAT, ~{} MB", size, size, mb);

        Ok(Self {
            image, memory, depth_view, sampling_view, framebuffer: fb, sampler,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            // depth_view == sampling_view, destroy once.
            device.destroy_image_view(self.depth_view, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Compute the sun's orthographic view + projection matrices for shadow rendering.
///
/// The orthographic frustum is centered on `camera_pos` projected along the
/// sun direction, covering `SUN_SHADOW_EXTENT` meters in each direction.
pub fn compute_sun_shadow_matrices(
    sun_direction: [f32; 3],
    camera_pos: [f32; 3],
) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
    // Light "eye" = camera center offset far along opposite of sun direction.
    let eye = [
        camera_pos[0] - sun_direction[0] * SUN_SHADOW_FAR * 0.5,
        camera_pos[1] - sun_direction[1] * SUN_SHADOW_FAR * 0.5,
        camera_pos[2] - sun_direction[2] * SUN_SHADOW_FAR * 0.5,
    ];
    let target = camera_pos;
    let up = if sun_direction[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };

    let view = look_at(eye, target, up);

    // Orthographic projection (Vulkan convention: Y-flip, Z=[0,1]).
    let half = SUN_SHADOW_EXTENT;
    let near = SUN_SHADOW_NEAR;
    let far = SUN_SHADOW_FAR;

    let proj = [
        [1.0 / half, 0.0, 0.0, 0.0],
        [0.0, -1.0 / half, 0.0, 0.0],                  // Y-flip for Vulkan
        [0.0, 0.0, -1.0 / (far - near), 0.0],          // NEGATIVE Z-scale
        [0.0, 0.0, -near / (far - near), 1.0],         // Z-offset in row 3
    ];

    (view, proj)
}

// ====================================================================
//  Cube map face view/projection matrices
// ====================================================================

/// Returns `(view_matrix, projection_matrix)` for a cube map face
/// rendered from `light_pos` with the given `radius` as far plane.
pub fn cube_face_matrices(
    light_pos: [f32; 3],
    radius: f32,
    face: u32,
) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
    let proj = perspective_fov_cube(90.0_f32.to_radians(), 1.0, SHADOW_NEAR, radius);

    let (target_offset, up): ([f32; 3], [f32; 3]) = match face {
        0 => ([ 1.0,  0.0,  0.0], [0.0, -1.0,  0.0]), // +X
        1 => ([-1.0,  0.0,  0.0], [0.0, -1.0,  0.0]), // -X
        2 => ([ 0.0,  1.0,  0.0], [0.0,  0.0,  1.0]), // +Y
        3 => ([ 0.0, -1.0,  0.0], [0.0,  0.0, -1.0]), // -Y
        4 => ([ 0.0,  0.0,  1.0], [0.0, -1.0,  0.0]), // +Z
        5 => ([ 0.0,  0.0, -1.0], [0.0, -1.0,  0.0]), // -Z
        _ => unreachable!(),
    };

    let target = [
        light_pos[0] + target_offset[0],
        light_pos[1] + target_offset[1],
        light_pos[2] + target_offset[2],
    ];

    let view = look_at(light_pos, target, up);
    (view, proj)
}

/// Perspective projection for cube map face rendering.
/// NO Y-flip — cube map sampling hardware expects OpenGL face orientation.
fn perspective_fov_cube(fov_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_rad / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],           // ← positive f, NOT -f
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (near * far) / (near - far), 0.0],
    ]
}

// ====================================================================
//  Math helpers (local to this module)
// ====================================================================

fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Look-at view matrix.
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub3(target, eye));
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

/// Invert a perspective projection matrix (column-major, Vulkan convention).
/// Only valid for standard perspective projections (no shear).
fn invert_projection(p: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // For a standard perspective matrix:
    //   [a  0  0  0]       [1/a  0   0     0  ]
    //   [0  b  0  0]  -->  [ 0  1/b  0     0  ]
    //   [0  0  c -1]       [ 0   0   0    1/e ]
    //   [0  0  e  0]       [ 0   0  -1    c/e ]
    let a = p[0][0];
    let b = p[1][1];
    let c = p[2][2];
    let e = p[3][2];

    let ia = if a.abs() > 1e-12 { 1.0 / a } else { 0.0 };
    let ib = if b.abs() > 1e-12 { 1.0 / b } else { 0.0 };
    let ie = if e.abs() > 1e-12 { 1.0 / e } else { 0.0 };

    [
        [ia, 0.0, 0.0, 0.0],
        [0.0, ib, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, ie, c * ie],
    ]
}