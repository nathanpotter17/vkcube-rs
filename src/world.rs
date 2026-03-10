//! Open-world spatial architecture + procedural generation.
//!
//! **Spatial systems:**
//! - `Sector` — 256 m streaming region.  Owns shared vertex/index buffer
//!   handles; objects store offsets into these shared buffers.
//! - `RenderObject` — per-entity renderable with AABB, LOD, offsets.
//! - `SpatialGrid` — 64 m uniform grid for broad-phase frustum culling.
//! - `DrawCommand` — fully resolved draw with offsets for cmd_draw_indexed.
//! - `World` — orchestrates sectors, objects, spatial index, per-frame culling.
//!
//! **Procedural generation:**
//! - `ObjectDescriptor` — CPU-side object data before GPU upload.
//! - `generate_sector_objects()` — non-uniform scatter across 256 m sector.
//!
//! Material palette (must match renderer.rs MaterialLibrary):
//!   0=default 1=ground 2=polished_metal 3=rough_stone 4=copper
//!   5=ceramic_red 6=ceramic_blue 7=gold 8=rubber 9=marble
//!   10=emissive_warm 11=emissive_cool

use ash::vk;
use std::collections::HashMap;

use crate::memory::BufferHandle;
use crate::scene::{identity_matrix, Vertex};

// ====================================================================
//  Constants
// ====================================================================

pub const SECTOR_SIZE: f32 = 256.0;
pub const SPATIAL_CELL_SIZE: f32 = 64.0;
pub const GROUND_TILE_SIZE: f32 = 64.0;
pub const MAX_LOD_LEVELS: usize = 4;

const DEFAULT_LOD_THRESHOLDS_SQ: [f32; MAX_LOD_LEVELS] = [
    6_400.0, 40_000.0, 250_000.0, f32::MAX,
];

pub const STREAMING_RADIUS: f32 = 200.0;
pub const EVICTION_RADIUS: f32 = 250.0;
pub const MAX_SECTOR_STARTS_PER_FRAME: usize = 2;

/// Extra padding added to sector content bounds to prevent border-mesh
/// pop-in/out.  Matches plan §6.4 "Expanded AABB for Border Meshes".
pub const SECTOR_BORDER_PADDING: f32 = 4.0;

// ====================================================================
//  Sector
// ====================================================================

pub type SectorCoord = (i32, i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectorState {
    Unloaded,
    Streaming,
    Ready,
    /// Upload failed — don't retry until camera moves away and back.
    Failed,
}

pub struct Sector {
    pub coord: SectorCoord,
    pub state: SectorState,
    pub objects: Vec<RenderObjectId>,
    pub priority: f32,
    /// Shared vertex buffer handle for all objects in this sector.
    pub vertex_handle: Option<BufferHandle>,
    /// Shared index buffer handle for all objects in this sector.
    pub index_handle: Option<BufferHandle>,
    /// Actual world-space AABB of all geometry in this sector (XYZ).
    /// `None` when sector has no registered objects (Unloaded/Streaming).
    /// Includes SECTOR_BORDER_PADDING margin for border meshes.
    pub content_bounds: Option<Aabb>,
}

impl Sector {
    pub fn new(coord: SectorCoord) -> Self {
        Self {
            coord, state: SectorState::Unloaded, objects: Vec::new(),
            priority: 0.0, vertex_handle: None, index_handle: None,
            content_bounds: None,
        }
    }

    /// Theoretical center of the 256 m grid cell.  Used only as a fallback
    /// when no content bounds are available (e.g. before objects are loaded).
    #[inline]
    pub fn world_center(&self) -> [f32; 2] {
        [
            self.coord.0 as f32 * SECTOR_SIZE + SECTOR_SIZE * 0.5,
            self.coord.1 as f32 * SECTOR_SIZE + SECTOR_SIZE * 0.5,
        ]
    }

    /// Expand the sector's content AABB to include `object_aabb`.
    /// Adds SECTOR_BORDER_PADDING margin on each axis.
    pub fn grow_bounds(&mut self, object_aabb: &Aabb) {
        match self.content_bounds {
            Some(ref mut cb) => {
                for i in 0..3 {
                    cb.min[i] = cb.min[i].min(object_aabb.min[i] - SECTOR_BORDER_PADDING);
                    cb.max[i] = cb.max[i].max(object_aabb.max[i] + SECTOR_BORDER_PADDING);
                }
            }
            None => {
                self.content_bounds = Some(Aabb::new(
                    [
                        object_aabb.min[0] - SECTOR_BORDER_PADDING,
                        object_aabb.min[1] - SECTOR_BORDER_PADDING,
                        object_aabb.min[2] - SECTOR_BORDER_PADDING,
                    ],
                    [
                        object_aabb.max[0] + SECTOR_BORDER_PADDING,
                        object_aabb.max[1] + SECTOR_BORDER_PADDING,
                        object_aabb.max[2] + SECTOR_BORDER_PADDING,
                    ],
                ));
            }
        }
    }

    /// Minimum squared distance from `point_xz` (camera XZ) to the sector's
    /// actual content footprint.  Returns 0.0 if the camera is inside.
    ///
    /// Falls back to theoretical 256 m cell center-point distance when
    /// `content_bounds` is `None` (sector has no loaded geometry yet).
    #[inline]
    pub fn distance_sq_to_point_xz(&self, point_xz: [f32; 2]) -> f32 {
        if let Some(ref cb) = self.content_bounds {
            // Project to XZ plane — cb.min[0]/max[0] = X, cb.min[2]/max[2] = Z.
            let dx = (cb.min[0] - point_xz[0]).max(0.0).max(point_xz[0] - cb.max[0]);
            let dz = (cb.min[2] - point_xz[1]).max(0.0).max(point_xz[1] - cb.max[2]);
            dx * dx + dz * dz
        } else {
            // No content bounds — fall back to center-point distance.
            let c = self.world_center();
            let dx = c[0] - point_xz[0];
            let dz = c[1] - point_xz[1];
            dx * dx + dz * dz
        }
    }

    /// Compute streaming priority.  Uses actual content bounds when available;
    /// falls back to theoretical 256 m AABB for sectors not yet loaded.
    pub fn compute_priority(
        &mut self, camera_xz: [f32; 2], camera_velocity_xz: [f32; 2],
        frustum: &[[f32; 4]; 6],
    ) {
        // ---- Distance component (inverse quadratic) ----
        let dist_sq = self.distance_sq_to_point_xz(camera_xz);
        let mut p = 1.0 / (dist_sq + 1.0);

        // ---- Frustum visibility boost (2×) ----
        let visible = if let Some(ref cb) = self.content_bounds {
            aabb_visible(frustum, cb.min, cb.max)
        } else {
            let center = self.world_center();
            let half = SECTOR_SIZE * 0.5;
            aabb_visible(frustum,
                [center[0] - half, -100.0, center[1] - half],
                [center[0] + half, 1000.0, center[1] + half],
            )
        };
        if visible { p *= 2.0; }

        // ---- Velocity prediction ----
        let center = if let Some(ref cb) = self.content_bounds {
            [(cb.min[0] + cb.max[0]) * 0.5, (cb.min[2] + cb.max[2]) * 0.5]
        } else {
            self.world_center()
        };
        let dx = center[0] - camera_xz[0];
        let dz = center[1] - camera_xz[1];
        let vel_len_sq = camera_velocity_xz[0] * camera_velocity_xz[0]
            + camera_velocity_xz[1] * camera_velocity_xz[1];
        if vel_len_sq > 0.001 {
            let dir_dist = (dx * dx + dz * dz).sqrt().max(0.001);
            let vel_len = vel_len_sq.sqrt();
            let dot = (camera_velocity_xz[0] * dx / dir_dist
                + camera_velocity_xz[1] * dz / dir_dist) / vel_len;
            p *= 0.5 + (dot + 1.0) * 0.5;
        }
        self.priority = p;
    }
}

// ====================================================================
//  RenderObject + MeshRange
// ====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderObjectId(pub u32);

/// Mesh data for one LOD level.  References the sector's shared buffers
/// via VkBuffer handles plus per-object offsets.  No per-object
/// BufferHandle — the sector owns the allocation.
#[derive(Debug, Clone, Copy)]
pub struct MeshRange {
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    /// Index into the shared index buffer (in indices, not bytes).
    pub first_index: u32,
    /// Number of indices to draw.
    pub index_count: u32,
    /// Base vertex added to each index value.
    pub vertex_offset: i32,
}

#[derive(Debug, Clone)]
pub struct LodChain {
    pub levels: [Option<MeshRange>; MAX_LOD_LEVELS],
    pub thresholds_sq: [f32; MAX_LOD_LEVELS],
    pub level_count: u8,
}

impl LodChain {
    pub fn single(range: MeshRange) -> Self {
        let mut levels = [None; MAX_LOD_LEVELS];
        levels[0] = Some(range);
        Self { levels, thresholds_sq: DEFAULT_LOD_THRESHOLDS_SQ, level_count: 1 }
    }

    #[inline]
    pub fn select(&self, dist_sq: f32) -> Option<&MeshRange> {
        for i in 0..self.level_count as usize {
            if dist_sq < self.thresholds_sq[i] {
                return self.levels[i].as_ref();
            }
        }
        self.levels[(self.level_count as usize).saturating_sub(1)].as_ref()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self { Self { min, max } }

    #[inline]
    pub fn distance_sq_to_point(&self, p: [f32; 3]) -> f32 {
        let dx = (self.min[0]-p[0]).max(0.0).max(p[0]-self.max[0]);
        let dy = (self.min[1]-p[1]).max(0.0).max(p[1]-self.max[1]);
        let dz = (self.min[2]-p[2]).max(0.0).max(p[2]-self.max[2]);
        dx*dx + dy*dy + dz*dz
    }

    #[inline]
    pub fn is_visible(&self, frustum: &[[f32; 4]; 6]) -> bool {
        aabb_visible(frustum, self.min, self.max)
    }

    pub fn from_vertices(verts: &[Vertex], transform: &[[f32; 4]; 4]) -> Self {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in verts {
            let wp = transform_point(*transform, v.position);
            for i in 0..3 { min[i] = min[i].min(wp[i]); max[i] = max[i].max(wp[i]); }
        }
        Self { min, max }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderFlags(pub u32);
impl RenderFlags {
    pub const NONE: Self = Self(0);
    pub const SHADOW_CASTER: Self = Self(1 << 0);
    pub const TERRAIN: Self = Self(1 << 1);
    pub const FOLIAGE: Self = Self(1 << 2);
    pub const STATIC: Self = Self(1 << 3);
    pub const TRANSPARENT: Self = Self(1 << 4);
    #[inline] pub fn contains(self, f: Self) -> bool { (self.0 & f.0) == f.0 }
}
impl std::ops::BitOr for RenderFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}

pub struct RenderObject {
    pub sector: SectorCoord,
    pub bounds: Aabb,
    pub lod: LodChain,
    pub transform: [[f32; 4]; 4],
    pub material_id: u32,
    pub flags: RenderFlags,
    pub id: RenderObjectId,
    pub alive: bool,
}

// ====================================================================
//  DrawCommand
// ====================================================================

#[derive(Debug, Clone, Copy)]
pub struct DrawCommand {
    pub object_id: RenderObjectId,
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub transform: [[f32; 4]; 4],
    pub material_id: u32,
    pub distance_sq: f32,
    pub flags: RenderFlags,
}

// ====================================================================
//  SpatialGrid
// ====================================================================

pub type GridCell = (i32, i32);

pub struct SpatialGrid {
    cell_size: f32, inv_cell_size: f32,
    cells: HashMap<GridCell, Vec<RenderObjectId>>,
    object_cells: HashMap<RenderObjectId, Vec<GridCell>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size, inv_cell_size: 1.0 / cell_size,
            cells: HashMap::with_capacity(4096),
            object_cells: HashMap::with_capacity(8192),
        }
    }
    #[inline] fn to_cell(&self, v: f32) -> i32 { (v * self.inv_cell_size).floor() as i32 }

    pub fn insert(&mut self, id: RenderObjectId, bounds: &Aabb) {
        let (mnx, mnz) = (self.to_cell(bounds.min[0]), self.to_cell(bounds.min[2]));
        let (mxx, mxz) = (self.to_cell(bounds.max[0]), self.to_cell(bounds.max[2]));
        let mut occ = Vec::new();
        for cx in mnx..=mxx { for cz in mnz..=mxz {
            self.cells.entry((cx,cz)).or_default().push(id);
            occ.push((cx,cz));
        }}
        self.object_cells.insert(id, occ);
    }

    pub fn remove(&mut self, id: RenderObjectId) {
        if let Some(cells) = self.object_cells.remove(&id) {
            for c in cells { if let Some(l) = self.cells.get_mut(&c) { l.retain(|&o| o != id); } }
        }
    }

    pub fn query_frustum(&self, frustum: &[[f32;4];6], xz: &[f32;4]) -> Vec<RenderObjectId> {
        let (mnx,mnz) = (self.to_cell(xz[0]), self.to_cell(xz[1]));
        let (mxx,mxz) = (self.to_cell(xz[2]), self.to_cell(xz[3]));
        let cap = ((mxx-mnx+1)*(mxz-mnz+1)) as usize;
        let mut res = Vec::with_capacity(cap.min(256)*30);
        let mut seen: HashMap<u32,()> = HashMap::with_capacity(cap.min(256)*30);
        for cx in mnx..=mxx { for cz in mnz..=mxz {
            let x0 = cx as f32*self.cell_size;
            let z0 = cz as f32*self.cell_size;
            if !aabb_visible(frustum, [x0,-500.0,z0], [x0+self.cell_size,5000.0,z0+self.cell_size]) { continue; }
            if let Some(objs) = self.cells.get(&(cx,cz)) {
                for &o in objs { if seen.insert(o.0,()).is_none() { res.push(o); } }
            }
        }}
        res
    }

    pub fn object_count(&self) -> usize { self.object_cells.len() }
}

// ====================================================================
//  World
// ====================================================================

pub struct World {
    pub sectors: HashMap<SectorCoord, Sector>,
    pub objects: Vec<RenderObject>,
    free_ids: Vec<u32>, next_id: u32,
    pub spatial: SpatialGrid,
    pub opaque_draws: Vec<DrawCommand>,
    pub transparent_draws: Vec<DrawCommand>,
    pub shadow_draws: Vec<DrawCommand>,
}

impl World {
    pub fn new() -> Self {
        Self {
            sectors: HashMap::with_capacity(256),
            objects: Vec::with_capacity(4096),
            free_ids: Vec::new(), next_id: 0,
            spatial: SpatialGrid::new(SPATIAL_CELL_SIZE),
            opaque_draws: Vec::with_capacity(4096),
            transparent_draws: Vec::with_capacity(256),
            shadow_draws: Vec::with_capacity(2048),
        }
    }

    pub fn alloc_id(&mut self) -> RenderObjectId {
        if let Some(r) = self.free_ids.pop() { RenderObjectId(r) }
        else { let id = self.next_id; self.next_id += 1; RenderObjectId(id) }
    }

    pub fn add_object(
        &mut self, sector: SectorCoord, bounds: Aabb, lod: LodChain,
        transform: [[f32;4];4], material_id: u32, flags: RenderFlags,
    ) -> RenderObjectId {
        let id = self.alloc_id();
        let idx = id.0 as usize;
        let obj = RenderObject {
            sector, bounds, lod, transform, material_id, flags, id, alive: true,
        };
        if idx >= self.objects.len() {
            self.objects.resize_with(idx+1, || RenderObject {
                sector:(0,0), bounds: Aabb::new([0.0;3],[0.0;3]),
                lod: LodChain { levels:[None;MAX_LOD_LEVELS], thresholds_sq: DEFAULT_LOD_THRESHOLDS_SQ, level_count:0 },
                transform: identity_matrix(), material_id:0, flags: RenderFlags::NONE,
                id: RenderObjectId(u32::MAX), alive: false,
            });
        }
        self.spatial.insert(id, &bounds);
        self.objects[idx] = obj;
        if let Some(sec) = self.sectors.get_mut(&sector) {
            sec.objects.push(id);
            // Grow sector content bounds to include this object's AABB.
            sec.grow_bounds(&bounds);
        }
        id
    }

    pub fn evict_sector(&mut self, coord: SectorCoord) -> Vec<RenderObjectId> {
        let ids = if let Some(sec) = self.sectors.get_mut(&coord) {
            let ids = std::mem::take(&mut sec.objects);
            sec.state = SectorState::Unloaded;
            sec.vertex_handle = None;
            sec.index_handle = None;
            sec.content_bounds = None;  // Reset — will be rebuilt on reload.
            ids
        } else { Vec::new() };
        for &id in &ids {
            self.spatial.remove(id);
            if let Some(o) = self.objects.get_mut(id.0 as usize) { o.alive = false; }
            self.free_ids.push(id.0);
        }
        ids
    }

    pub fn cull_and_select_lod(
        &mut self, camera_pos: [f32;3], frustum: &[[f32;4];6], xz: &[f32;4],
    ) {
        self.opaque_draws.clear();
        self.transparent_draws.clear();
        self.shadow_draws.clear();

        for oid in self.spatial.query_frustum(frustum, xz) {
            let idx = oid.0 as usize;
            if idx >= self.objects.len() { continue; }
            let obj = &self.objects[idx];
            if !obj.alive || !obj.bounds.is_visible(frustum) { continue; }

            let dist_sq = obj.bounds.distance_sq_to_point(camera_pos);
            let mesh = match obj.lod.select(dist_sq) { Some(m) => *m, None => continue };

            let cmd = DrawCommand {
                object_id: oid,
                vertex_buffer: mesh.vertex_buffer, index_buffer: mesh.index_buffer,
                first_index: mesh.first_index, index_count: mesh.index_count,
                vertex_offset: mesh.vertex_offset,
                transform: obj.transform, material_id: obj.material_id,
                distance_sq: dist_sq, flags: obj.flags,
            };
            if obj.flags.contains(RenderFlags::TRANSPARENT) {
                self.transparent_draws.push(cmd);
            } else {
                self.opaque_draws.push(cmd);
            }
            if obj.flags.contains(RenderFlags::SHADOW_CASTER) {
                self.shadow_draws.push(cmd);
            }
        }
        self.transparent_draws.sort_unstable_by(|a,b| b.distance_sq.partial_cmp(&a.distance_sq).unwrap());
        self.opaque_draws.sort_unstable_by(|a,b| a.distance_sq.partial_cmp(&b.distance_sq).unwrap());
    }

    pub fn update_sector_grid(&mut self, camera_xz: [f32;2], r: f32) {
        let cx = (camera_xz[0]/SECTOR_SIZE).floor() as i32;
        let cz = (camera_xz[1]/SECTOR_SIZE).floor() as i32;
        let rc = (r/SECTOR_SIZE).ceil() as i32;
        let r_sq = r * r;
        for sx in (cx-rc)..=(cx+rc) { for sz in (cz-rc)..=(cz+rc) {
            // For sectors already known, use actual content bounds if available.
            // For new sectors, use theoretical center — conservative for discovery.
            let dist_sq = if let Some(sec) = self.sectors.get(&(sx, sz)) {
                sec.distance_sq_to_point_xz(camera_xz)
            } else {
                let x0 = sx as f32 * SECTOR_SIZE;
                let x1 = x0 + SECTOR_SIZE;
                let z0 = sz as f32 * SECTOR_SIZE;
                let z1 = z0 + SECTOR_SIZE;
                let dx = (x0 - camera_xz[0]).max(0.0).max(camera_xz[0] - x1);
                let dz = (z0 - camera_xz[1]).max(0.0).max(camera_xz[1] - z1);
                dx * dx + dz * dz
            };
            if dist_sq > r_sq { continue; }
            self.sectors.entry((sx,sz)).or_insert_with(|| Sector::new((sx,sz)));
        }}
    }

    pub fn prioritized_unloaded_sectors(
        &mut self, cam: [f32;2], vel: [f32;2], frustum: &[[f32;4];6],
    ) -> Vec<SectorCoord> {
        for s in self.sectors.values_mut() {
            if s.state == SectorState::Unloaded { s.compute_priority(cam, vel, frustum); }
        }
        let mut v: Vec<_> = self.sectors.iter()
            .filter(|(_,s)| s.state == SectorState::Unloaded)
            .map(|(&c,s)| (c,s.priority)).collect();
        v.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        v.into_iter().map(|(c,_)| c).collect()
    }

    pub fn total_objects(&self) -> usize { self.spatial.object_count() }
    pub fn ready_sector_count(&self) -> usize {
        self.sectors.values().filter(|s| s.state == SectorState::Ready).count()
    }
    pub fn streaming_sector_count(&self) -> usize {
        self.sectors.values().filter(|s| s.state == SectorState::Streaming).count()
    }
}

// ====================================================================
//  Frustum helpers
// ====================================================================

#[inline]
pub fn aabb_visible(frustum: &[[f32;4];6], mn: [f32;3], mx: [f32;3]) -> bool {
    for p in frustum {
        let px = if p[0]>=0.0 {mx[0]} else {mn[0]};
        let py = if p[1]>=0.0 {mx[1]} else {mn[1]};
        let pz = if p[2]>=0.0 {mx[2]} else {mn[2]};
        if p[0]*px + p[1]*py + p[2]*pz + p[3] < 0.0 { return false; }
    }
    true
}

pub fn frustum_aabb_xz(pos: [f32;3], far: f32) -> [f32;4] {
    [pos[0]-far, pos[2]-far, pos[0]+far, pos[2]+far]
}

fn transform_point(m: [[f32;4];4], p: [f32;3]) -> [f32;3] {
    [
        m[0][0]*p[0]+m[1][0]*p[1]+m[2][0]*p[2]+m[3][0],
        m[0][1]*p[0]+m[1][1]*p[1]+m[2][1]*p[2]+m[3][1],
        m[0][2]*p[0]+m[1][2]*p[1]+m[2][2]*p[2]+m[3][2],
    ]
}

// ####################################################################
//  PROCEDURAL GENERATION
// ####################################################################

/// CPU-side object data before GPU upload.
pub struct ObjectDescriptor {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub transform: [[f32; 4]; 4],
    pub material_id: u32,
    pub flags: RenderFlags,
    pub bounds: Aabb,
}

// ---- Ground colour palette ----

pub fn tile_ground_color(cx: i32, cz: i32) -> [f32; 3] {
    const P: [[f32;3];8] = [[0.15; 3]; 8];
    P[((cx.wrapping_mul(7))^(cz.wrapping_mul(13))).unsigned_abs() as usize % P.len()]
}

// ---- Geometry primitives ----

pub struct RawMesh { pub vertices: Vec<Vertex>, pub indices: Vec<u32>, pub transform: [[f32;4];4], pub material_id: u32 }

pub fn make_ground_plane(cx: i32, cz: i32) -> RawMesh {
    let x0 = cx as f32*GROUND_TILE_SIZE; let z0 = cz as f32*GROUND_TILE_SIZE;
    let x1 = x0+GROUND_TILE_SIZE; let z1 = z0+GROUND_TILE_SIZE;
    let c = tile_ground_color(cx,cz); let n = [0.0,1.0,0.0];
    RawMesh {
        vertices: vec![
            Vertex::full([x0,0.0,z0],n,[0.0,0.0],c), Vertex::full([x1,0.0,z0],n,[1.0,0.0],c),
            Vertex::full([x1,0.0,z1],n,[1.0,1.0],c), Vertex::full([x0,0.0,z1],n,[0.0,1.0],c),
        ],
        indices: vec![0,2,1,0,3,2],
        transform: identity_matrix(), material_id: 1,
    }
}

pub fn make_cube(base_color: [f32;3], material_id: u32) -> RawMesh {
    let t = |r:f32,g:f32,b:f32| [(base_color[0]*0.5+r*0.5).min(1.0),(base_color[1]*0.5+g*0.5).min(1.0),(base_color[2]*0.5+b*0.5).min(1.0)];
    let p: [[f32;3];8] = [[-0.5,0.0,0.5],[0.5,0.0,0.5],[0.5,1.0,0.5],[-0.5,1.0,0.5],
        [0.5,0.0,-0.5],[-0.5,0.0,-0.5],[-0.5,1.0,-0.5],[0.5,1.0,-0.5]];
    let fn_: [[f32;3];6] = [[0.0,0.0,1.0],[0.0,0.0,-1.0],[0.0,1.0,0.0],[0.0,-1.0,0.0],[1.0,0.0,0.0],[-1.0,0.0,0.0]];
    let fc = [t(1.0,0.3,0.3),t(0.3,1.0,0.3),t(0.3,0.3,1.0),t(1.0,1.0,0.3),t(1.0,0.3,1.0),t(0.3,1.0,1.0)];
    let fd: [([usize;4],usize);6] = [([0,1,2,3],0),([4,5,6,7],1),([3,2,7,6],2),([5,4,1,0],3),([1,4,7,2],4),([5,0,3,6],5)];
    let uv: [[f32;2];4] = [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]];
    let (mut vs, mut is) = (Vec::new(), Vec::new());
    for (fi,ci) in fd { let b = vs.len() as u32;
        for (vi,&pi) in fi.iter().enumerate() { vs.push(Vertex::full(p[pi],fn_[ci],uv[vi],fc[ci])); }
        is.extend([b,b+1,b+2,b+2,b+3,b]);
    }
    RawMesh { vertices: vs, indices: is, transform: identity_matrix(), material_id }
}

pub fn make_pyramid(color: [f32;3], height: f32, material_id: u32) -> RawMesh {
    let s=0.5; let apex=[0.0,height,0.0];
    let bl=[-s,0.0,-s]; let br=[s,0.0,-s]; let fr=[s,0.0,s]; let fl=[-s,0.0,s];
    let (mut vs, mut is) = (Vec::new(), Vec::new());
    for (a,b,c,col) in [(fl,fr,apex,color),(fr,br,apex,[color[0]*0.8,color[1]*0.8,color[2]*0.8]),
        (br,bl,apex,[color[0]*0.6,color[1]*0.6,color[2]*0.6]),(bl,fl,apex,[color[0]*0.7,color[1]*0.7,color[2]*0.7])] {
        let ba = vs.len() as u32;
        let e1=[b[0]-a[0],b[1]-a[1],b[2]-a[2]]; let e2=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];
        let n = normalize_vec3(cross_vec3(e1,e2));
        vs.push(Vertex::full(a,n,[0.0,1.0],col)); vs.push(Vertex::full(b,n,[1.0,1.0],col));
        vs.push(Vertex::full(c,n,[0.5,0.0],col)); is.extend([ba,ba+1,ba+2]);
    }
    let bc = [color[0]*0.5,color[1]*0.5,color[2]*0.5]; let bn=[0.0,-1.0,0.0]; let bs=vs.len()as u32;
    vs.push(Vertex::full(bl,bn,[0.0,0.0],bc)); vs.push(Vertex::full(br,bn,[1.0,0.0],bc));
    vs.push(Vertex::full(fr,bn,[1.0,1.0],bc)); vs.push(Vertex::full(fl,bn,[0.0,1.0],bc));
    is.extend([bs,bs+2,bs+1,bs,bs+3,bs+2]);
    RawMesh { vertices: vs, indices: is, transform: identity_matrix(), material_id }
}

pub fn make_column(color: [f32;3], height: f32, radius: f32, material_id: u32) -> RawMesh {
    const S: usize = 8;
    let (mut vs, mut is) = (Vec::new(), Vec::new());
    let tc = [(color[0]*1.2).min(1.0),(color[1]*1.2).min(1.0),(color[2]*1.2).min(1.0)];
    let (mut br, mut tr) = (Vec::new(), Vec::new());
    for i in 0..S {
        let a = (i as f32/S as f32)*std::f32::consts::TAU;
        br.push([a.cos()*radius,0.0,a.sin()*radius]);
        tr.push([a.cos()*radius,height,a.sin()*radius]);
    }
    for i in 0..S {
        let j=(i+1)%S; let b=vs.len()as u32;
        let mx=(br[i][0]+br[j][0])*0.5; let mz=(br[i][2]+br[j][2])*0.5;
        let l=(mx*mx+mz*mz).sqrt(); let n=if l>0.0{[mx/l,0.0,mz/l]}else{[1.0,0.0,0.0]};
        let u0=i as f32/S as f32; let u1=(i+1)as f32/S as f32;
        vs.push(Vertex::full(br[j],n,[u1,1.0],color)); vs.push(Vertex::full(br[i],n,[u0,1.0],color));
        vs.push(Vertex::full(tr[i],n,[u0,0.0],color)); vs.push(Vertex::full(tr[j],n,[u1,0.0],color));
        is.extend([b,b+1,b+2,b+2,b+3,b]);
    }
    let tn=[0.0,1.0,0.0]; let ct=vs.len()as u32;
    vs.push(Vertex::full([0.0,height,0.0],tn,[0.5,0.5],tc));
    for i in 0..S {
        let j=(i+1)%S; let bi=vs.len()as u32;
        let ui=(i as f32/S as f32)*std::f32::consts::TAU;
        let uj=((i+1)as f32/S as f32)*std::f32::consts::TAU;
        vs.push(Vertex::full(tr[i],tn,[ui.cos()*0.5+0.5,ui.sin()*0.5+0.5],tc));
        vs.push(Vertex::full(tr[j],tn,[uj.cos()*0.5+0.5,uj.sin()*0.5+0.5],tc));
        is.extend([ct,bi+1,bi]);
    }
    RawMesh { vertices: vs, indices: is, transform: identity_matrix(), material_id }
}

pub fn cross_vec3(a:[f32;3],b:[f32;3])->[f32;3]{[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
pub fn normalize_vec3(v:[f32;3])->[f32;3]{let l=(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();if l>0.0{[v[0]/l,v[1]/l,v[2]/l]}else{[0.0,1.0,0.0]}}

#[allow(dead_code)]
const OBJ_MATS: [u32; 8] = [2,3,4,5,6,7,8,9];

// ====================================================================
//  Public API: generate ObjectDescriptors for a 256 m sector
// ====================================================================

/// Demo scene: 4 ground tiles meeting at the world origin.
/// Only sectors (-1,-1), (-1,0), (0,-1), (0,0) produce content;
/// all other sectors return empty.
///
/// Scene starts with ground only — geometry is spawned interactively via keybinds.
pub fn generate_sector_objects(sector: SectorCoord) -> Vec<ObjectDescriptor> {
    const DEMO_SECTORS: [(i32,i32); 4] = [(-1,-1), (0,-1), (-1,0), (0,0)];
    if !DEMO_SECTORS.contains(&sector) {
        return Vec::new();
    }

    let flags = RenderFlags::STATIC | RenderFlags::SHADOW_CASTER;
    let mut result = Vec::new();

    // ---- One ground tile per sector ----
    let raw = make_ground_plane(sector.0, sector.1);
    let bounds = Aabb::from_vertices(&raw.vertices, &raw.transform);
    result.push(ObjectDescriptor {
        vertices: raw.vertices, indices: raw.indices, transform: raw.transform,
        material_id: raw.material_id, flags, bounds,
    });

    result
}

// ---- Demo helper: push transformed object into result ----

pub fn push_cube(result: &mut Vec<ObjectDescriptor>, flags: RenderFlags,
             pos: [f32; 3], scale: f32, y_rot: f32, color: [f32; 3], mat: u32) {
    let raw = make_cube(color, mat);
    let transform = demo_transform(pos, scale, y_rot);
    let bounds = Aabb::from_vertices(&raw.vertices, &transform);
    result.push(ObjectDescriptor {
        vertices: raw.vertices, indices: raw.indices, transform,
        material_id: mat, flags, bounds,
    });
}

pub fn push_pyramid(result: &mut Vec<ObjectDescriptor>, flags: RenderFlags,
                pos: [f32; 3], scale: f32, height: f32, y_rot: f32,
                color: [f32; 3], mat: u32) {
    let raw = make_pyramid(color, height, mat);
    let transform = demo_transform(pos, scale, y_rot);
    let bounds = Aabb::from_vertices(&raw.vertices, &transform);
    result.push(ObjectDescriptor {
        vertices: raw.vertices, indices: raw.indices, transform,
        material_id: mat, flags, bounds,
    });
}

pub fn push_column(result: &mut Vec<ObjectDescriptor>, flags: RenderFlags,
               pos: [f32; 3], scale: f32, height: f32, radius: f32,
               y_rot: f32, color: [f32; 3], mat: u32) {
    let raw = make_column(color, height, radius, mat);
    let transform = demo_transform(pos, scale, y_rot);
    let bounds = Aabb::from_vertices(&raw.vertices, &transform);
    result.push(ObjectDescriptor {
        vertices: raw.vertices, indices: raw.indices, transform,
        material_id: mat, flags, bounds,
    });
}

/// Build a Y-axis rotation + uniform scale + translation matrix.
pub fn demo_transform(pos: [f32; 3], scale: f32, y_rot: f32) -> [[f32; 4]; 4] {
    let cos = y_rot.cos();
    let sin = y_rot.sin();
    [
        [scale * cos,  0.0, scale * sin, 0.0],
        [0.0,          scale, 0.0,       0.0],
        [-scale * sin, 0.0, scale * cos, 0.0],
        [pos[0],       pos[1], pos[2],   1.0],
    ]
}