//! Phase 8B: GPU-Driven Culling Resources + Mega Buffers
//!
//! Replaces CPU-side frustum culling with a compute shader dispatch.
//! Objects stored in a persistent SSBO; culled results written to
//! indirect command buffers consumed by `cmd_draw_indexed_indirect_count`.
//!
//! Phase 8B additions:
//! - MegaBuffers: single 128 MB VB + 64 MB IB with free-list sub-allocator
//! - BufferGroup eliminated: all geometry in mega buffers, single bind per pass
//! - Cull shader uses single atomic counter (buffer_group always 0)
//!
//! Key invariants:
//! - `flush_dirty()` MUST be called AFTER `wait_for_fences()` (§3.3)
//! - Pre-cull barrier must include VERTEX_SHADER|FRAGMENT_SHADER src (§3.4)
//! - HiZ image initialized to SHADER_READ_ONLY_OPTIMAL on frame 0 (§3.5)

use ash::{vk, Device};
use std::ptr::NonNull;

use crate::memory::{
    BufferAllocation, BufferHandle, GpuAllocator, MemoryContext, MemoryLocation,
    MAX_FRAMES_IN_FLIGHT,
};
use crate::world::RenderFlags;

// ====================================================================
//  Constants
// ====================================================================

pub const MAX_GPU_OBJECTS: u32 = 65_536;
pub const MAX_INDIRECT_DRAWS: u32 = 65_536;
/// Size of VkDrawIndexedIndirectCommand (20 bytes)
pub const INDIRECT_COMMAND_STRIDE: u32 = 20;
/// Phase 8B: single mega buffer group (retained for descriptor layout compat)
pub const MAX_BUFFER_GROUPS: u32 = 256;

/// Phase 8B: Mega vertex buffer capacity (128 MB)
pub const MEGA_VB_SIZE: u64 = 128 * 1024 * 1024;
/// Phase 8B: Mega index buffer capacity (64 MB)
pub const MEGA_IB_SIZE: u64 = 64 * 1024 * 1024;
pub const VERTEX_STRIDE: u64 = 60;
// ====================================================================
//  GpuObjectData — 128 bytes, std430
// ====================================================================

/// Per-object data stored in the persistent SSBO.
/// Vertex shaders index via `gl_InstanceIndex` (== firstInstance from indirect cmd).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuObjectData {
    /// Model-to-world transformation matrix (64 bytes)
    pub model: [[f32; 4]; 4],
    /// World-space AABB minimum (w unused) — 16 bytes
    pub aabb_min: [f32; 4],
    /// World-space AABB maximum (w unused) — 16 bytes
    pub aabb_max: [f32; 4],
    /// Index into the mega index buffer (in indices, not bytes)
    pub first_index: u32,
    /// Number of indices to draw; 0 = slot empty (skip in cull shader)
    pub index_count: u32,
    /// Base vertex added to each index value (offset into mega vertex buffer)
    pub vertex_offset: i32,
    /// Material ID for fragment shader lookup
    pub material_id: u32,
    /// Phase 8B: always 0 (single mega buffer). Retained for SSBO layout compat.
    pub buffer_group: u32,
    /// RenderFlags bits (SHADOW_CASTER, TRANSPARENT, etc.)
    pub flags: u32,
    /// LOD selection bias
    pub lod_bias: f32,
    /// Padding for 128-byte alignment
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<GpuObjectData>() == 128);

impl GpuObjectData {
    pub fn new(
        model: [[f32; 4]; 4],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        first_index: u32,
        index_count: u32,
        vertex_offset: i32,
        material_id: u32,
        flags: RenderFlags,
    ) -> Self {
        Self {
            model,
            aabb_min: [aabb_min[0], aabb_min[1], aabb_min[2], 0.0],
            aabb_max: [aabb_max[0], aabb_max[1], aabb_max[2], 0.0],
            first_index,
            index_count,
            vertex_offset,
            material_id,
            buffer_group: 0, // Phase 8B: always mega buffer group 0
            flags: flags.0,
            lod_bias: 0.0,
            _pad: 0,
        }
    }

    /// Create a zeroed/dead slot (index_count = 0 signals skip in cull shader)
    pub fn dead() -> Self {
        Self::default()
    }
}

// ====================================================================
//  CullPushConstants — 112 bytes
// ====================================================================

/// Push constants for the cull compute shader dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CullPushConstants {
    /// Six frustum planes in world space (Ax + By + Cz + D format)
    pub frustum_planes: [[f32; 4]; 6],
    /// Camera world position
    pub camera_pos: [f32; 3],
    /// Total number of objects to process
    pub total_objects: u32,
}

const _: () = assert!(std::mem::size_of::<CullPushConstants>() == 112);

impl CullPushConstants {
    pub fn new(frustum: &[[f32; 4]; 6], camera_pos: [f32; 3], total_objects: u32) -> Self {
        Self {
            frustum_planes: *frustum,
            camera_pos,
            total_objects,
        }
    }
}

// ====================================================================
//  Phase 8B: MegaBuffers — single VB + IB with free-list sub-allocator
// ====================================================================

/// A contiguous free byte region within a mega buffer.
#[derive(Debug, Clone, Copy)]
struct MegaFreeRegion {
    offset: u64,
    size: u64,
}

/// Tracks a sub-allocation from the mega buffers for one sector's geometry.
#[derive(Debug, Clone, Copy)]
pub struct MegaAlloc {
    /// Byte offset into the mega vertex buffer
    pub vertex_offset_bytes: u64,
    /// Byte size of the vertex data
    pub vertex_size_bytes: u64,
    /// Byte offset into the mega index buffer
    pub index_offset_bytes: u64,
    /// Byte size of the index data
    pub index_size_bytes: u64,
}

impl MegaAlloc {
    /// First vertex index (in vertices, not bytes) for this allocation.
    /// Used as the base vertex_offset for objects within this sector.
    pub fn base_vertex(&self) -> i32 {
        debug_assert!(
            self.vertex_offset_bytes % VERTEX_STRIDE == 0,
            "vertex_offset_bytes {} is not aligned to VERTEX_STRIDE {}",
            self.vertex_offset_bytes, VERTEX_STRIDE,
        );
        (self.vertex_offset_bytes / VERTEX_STRIDE) as i32
    }

    /// First index position (in indices, not bytes) for this allocation.
    /// Used as the base first_index for objects within this sector.
    pub fn base_index(&self) -> u32 {
        debug_assert!(
            self.index_offset_bytes % 4 == 0,
            "index_offset_bytes {} is not aligned to sizeof(u32)",
            self.index_offset_bytes,
        );
        (self.index_offset_bytes / 4) as u32
    }
}

/// Single 128 MB vertex buffer + 64 MB index buffer with coalescing free-list.
///
/// Mirrors the `PoolBlock` allocator pattern from `memory.rs` but operates
/// on sub-regions of a single VkBuffer rather than VkDeviceMemory.
pub struct MegaBuffers {
    pub vertex_buffer: vk::Buffer,
    pub vertex_handle: BufferHandle,
    vertex_capacity: u64,
    vertex_free: Vec<MegaFreeRegion>,

    pub index_buffer: vk::Buffer,
    pub index_handle: BufferHandle,
    index_capacity: u64,
    index_free: Vec<MegaFreeRegion>,
}

impl MegaBuffers {
    pub fn new(
        allocator: &mut GpuAllocator,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Vertex buffer: VERTEX_BUFFER + TRANSFER_DST (staging uploads)
        let vb_alloc = allocator.create_buffer(
            MEGA_VB_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        // Index buffer: INDEX_BUFFER + TRANSFER_DST (staging uploads)
        let ib_alloc = allocator.create_buffer(
            MEGA_IB_SIZE,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        println!(
            "[MegaBuffers] VB={:.1} MB, IB={:.1} MB",
            MEGA_VB_SIZE as f64 / (1024.0 * 1024.0),
            MEGA_IB_SIZE as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            vertex_buffer: vb_alloc.buffer,
            vertex_handle: vb_alloc.handle,
            vertex_capacity: MEGA_VB_SIZE,
            vertex_free: vec![MegaFreeRegion { offset: 0, size: MEGA_VB_SIZE }],

            index_buffer: ib_alloc.buffer,
            index_handle: ib_alloc.handle,
            index_capacity: MEGA_IB_SIZE,
            index_free: vec![MegaFreeRegion { offset: 0, size: MEGA_IB_SIZE }],
        })
    }

    /// Sub-allocate `vertex_bytes` from VB and `index_bytes` from IB.
    /// Alignment: vertices to 64 bytes (cache line), indices to 4 bytes.
    /// Returns None if either allocation fails (mega buffer full).
    pub fn alloc(
        &mut self,
        vertex_bytes: u64,
        index_bytes: u64,
    ) -> Option<MegaAlloc> {
        let v_off = Self::alloc_region(&mut self.vertex_free, vertex_bytes, VERTEX_STRIDE)?;
        match Self::alloc_region(&mut self.index_free, index_bytes, 4) {
            Some(i_off) => Some(MegaAlloc {
                vertex_offset_bytes: v_off,
                vertex_size_bytes: vertex_bytes,
                index_offset_bytes: i_off,
                index_size_bytes: index_bytes,
            }),
            None => {
                // Roll back vertex allocation
                Self::free_region(&mut self.vertex_free, v_off, vertex_bytes);
                None
            }
        }
    }

    /// Free a previously allocated region, returning it to the free list.
    pub fn free(&mut self, alloc: &MegaAlloc) {
        Self::free_region(
            &mut self.vertex_free,
            alloc.vertex_offset_bytes,
            alloc.vertex_size_bytes,
        );
        Self::free_region(
            &mut self.index_free,
            alloc.index_offset_bytes,
            alloc.index_size_bytes,
        );
    }

    /// Current usage stats (vertex_used, vertex_total, index_used, index_total).
    pub fn usage(&self) -> (u64, u64, u64, u64) {
        let vb_free: u64 = self.vertex_free.iter().map(|r| r.size).sum();
        let ib_free: u64 = self.index_free.iter().map(|r| r.size).sum();
        (
            self.vertex_capacity - vb_free,
            self.vertex_capacity,
            self.index_capacity - ib_free,
            self.index_capacity,
        )
    }

    /// First-fit allocation with alignment. Returns byte offset or None.
    fn alloc_region(
        free_list: &mut Vec<MegaFreeRegion>,
        size: u64,
        alignment: u64,
    ) -> Option<u64> {
        for i in 0..free_list.len() {
            let region = &free_list[i];
            let aligned = align_up_general(region.offset, alignment);
            let padding = aligned - region.offset;
            let needed = padding + size;

            if region.size < needed {
                continue;
            }

            let remaining = region.size - needed;

            if padding > 0 && remaining > 0 {
                // Split: padding region stays, allocated in middle, remainder after
                free_list[i].size = padding;
                free_list.insert(
                    i + 1,
                    MegaFreeRegion { offset: aligned + size, size: remaining },
                );
            } else if padding > 0 {
                // Only padding remains
                free_list[i].size = padding;
            } else if remaining > 0 {
                // Remainder stays in-place
                free_list[i] = MegaFreeRegion { offset: aligned + size, size: remaining };
            } else {
                // Exact fit — remove region
                free_list.remove(i);
            }

            return Some(aligned);
        }
        None
    }

    /// Return a region to the free list with coalescing of adjacent regions.
    fn free_region(free_list: &mut Vec<MegaFreeRegion>, offset: u64, size: u64) {
        let pos = free_list.partition_point(|r| r.offset < offset);
        free_list.insert(pos, MegaFreeRegion { offset, size });

        // Coalesce right
        if pos + 1 < free_list.len() {
            let cur_end = free_list[pos].offset + free_list[pos].size;
            if cur_end == free_list[pos + 1].offset {
                free_list[pos].size += free_list[pos + 1].size;
                free_list.remove(pos + 1);
            }
        }
        // Coalesce left
        if pos > 0 {
            let prev_end = free_list[pos - 1].offset + free_list[pos - 1].size;
            if prev_end == free_list[pos].offset {
                free_list[pos - 1].size += free_list[pos].size;
                free_list.remove(pos);
            }
        }
    }

    pub fn destroy(&mut self, allocator: &mut GpuAllocator) {
        allocator.free_buffer(self.vertex_handle);
        allocator.free_buffer(self.index_handle);
    }
}

// ====================================================================
//  GpuCullResources
// ====================================================================

pub struct GpuCullResources {
    device: Device,

    // ---- Object SSBO (persistent, 128B × MAX_GPU_OBJECTS) ----
    pub object_ssbo: vk::Buffer,
    pub object_ssbo_handle: BufferHandle,
    /// ReBAR path: Some(ptr) if GpuMapped allocation succeeded; None → staging fallback
    object_ssbo_mapped: Option<NonNull<u8>>,
    /// Set true after host writes; cleared after pre-cull barrier is recorded
    pub needs_host_barrier: bool,

    // ---- Indirect command buffers (double-buffered) ----
    pub opaque_cmds: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
    pub opaque_cmds_handles: [BufferHandle; MAX_FRAMES_IN_FLIGHT],
    pub shadow_cmds: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
    pub shadow_cmds_handles: [BufferHandle; MAX_FRAMES_IN_FLIGHT],

    // ---- Atomic draw count buffers (one u32 per group, double-buffered) ----
    // Phase 8B: only index 0 is used (single mega buffer group)
    pub opaque_counts: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
    pub opaque_counts_handles: [BufferHandle; MAX_FRAMES_IN_FLIGHT],
    pub shadow_counts: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
    pub shadow_counts_handles: [BufferHandle; MAX_FRAMES_IN_FLIGHT],

    // ---- Group base offsets (prefix sum, read by cull shader) ----
    // Phase 8B: retained for descriptor layout compat; group_bases[0] = 0 always
    pub group_bases: vk::Buffer,
    pub group_bases_handle: BufferHandle,

    // ---- Compute pipeline ----
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],

    // ---- Object SSBO descriptor for graphics pipelines (set 3) ----
    pub object_ssbo_set_layout: vk::DescriptorSetLayout,
    pub object_ssbo_set: vk::DescriptorSet,

    // ---- Dirty object queue ----
    pub dirty_objects: Vec<(u32, GpuObjectData)>,
    /// Tier 1 fix: O(1) lookup — object ID → index in dirty_objects vec.
    /// Eliminates O(n) linear scan in queue_dirty.
    dirty_index: std::collections::HashMap<u32, usize>,

    // ---- Phase 8B: Mega Buffers ----
    pub mega: MegaBuffers,

    /// Total number of alive objects (next available slot)
    pub total_alive: u32,

    /// Debug assertion: fence was waited before flush_dirty
    #[cfg(debug_assertions)]
    fence_waited_this_frame: bool,

    /// CPU-side mirror of SSBO data for read-modify-write workflows
    pub object_mirror: Vec<GpuObjectData>,
}

/// Align `value` up to the next multiple of `alignment`.
/// Works for ANY alignment (not just power-of-2), unlike `memory::align_up`.
#[inline(always)]
const fn align_up_general(value: u64, alignment: u64) -> u64 {
    let remainder = value % alignment;
    if remainder == 0 { value } else { value + alignment - remainder }
}

// Safety: GpuCullResources contains raw pointers but they're either null
// or point to persistently mapped GPU memory that outlives the struct.
unsafe impl Send for GpuCullResources {}

impl GpuCullResources {
    pub fn new(
        device: &Device,
        allocator: &mut GpuAllocator,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ssbo_size = (MAX_GPU_OBJECTS as u64) * 128;
        let indirect_size = (MAX_INDIRECT_DRAWS as u64) * (INDIRECT_COMMAND_STRIDE as u64);
        let counts_size = (MAX_BUFFER_GROUPS as u64) * 4;
        let group_bases_size = (MAX_BUFFER_GROUPS as u64) * 4;

        // ---- Allocate Object SSBO (try ReBAR first) ----
        let (object_ssbo, object_ssbo_handle, object_ssbo_mapped) = {
            match allocator.create_buffer(
                ssbo_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::CpuToGpu,
            ) {
                Ok(alloc) if alloc.mapped_ptr.is_some() => {
                    println!("[GpuCull] Object SSBO allocated via ReBAR ({} bytes)", ssbo_size);
                    (alloc.buffer, alloc.handle, alloc.mapped_ptr)
                }
                Ok(alloc) => {
                    println!("[GpuCull] Object SSBO allocated CpuToGpu but not mapped ({})", ssbo_size);
                    (alloc.buffer, alloc.handle, None)
                }
                Err(_) => {
                    let alloc = allocator.create_buffer(
                        ssbo_size,
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                        MemoryLocation::GpuOnly,
                    )?;
                    println!("[GpuCull] Object SSBO allocated GpuOnly (staging required)");
                    (alloc.buffer, alloc.handle, None)
                }
            }
        };

        // ---- Allocate indirect command buffers (double-buffered) ----
        let mut opaque_cmds = [vk::Buffer::null(); MAX_FRAMES_IN_FLIGHT];
        let mut opaque_cmds_handles = [BufferHandle(0); MAX_FRAMES_IN_FLIGHT];
        let mut shadow_cmds = [vk::Buffer::null(); MAX_FRAMES_IN_FLIGHT];
        let mut shadow_cmds_handles = [BufferHandle(0); MAX_FRAMES_IN_FLIGHT];

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let alloc = allocator.create_buffer(
                indirect_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )?;
            opaque_cmds[i] = alloc.buffer;
            opaque_cmds_handles[i] = alloc.handle;

            let alloc2 = allocator.create_buffer(
                indirect_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )?;
            shadow_cmds[i] = alloc2.buffer;
            shadow_cmds_handles[i] = alloc2.handle;
        }

        // ---- Allocate count buffers (double-buffered) ----
        let mut opaque_counts = [vk::Buffer::null(); MAX_FRAMES_IN_FLIGHT];
        let mut opaque_counts_handles = [BufferHandle(0); MAX_FRAMES_IN_FLIGHT];
        let mut shadow_counts = [vk::Buffer::null(); MAX_FRAMES_IN_FLIGHT];
        let mut shadow_counts_handles = [BufferHandle(0); MAX_FRAMES_IN_FLIGHT];

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let alloc = allocator.create_buffer(
                counts_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )?;
            opaque_counts[i] = alloc.buffer;
            opaque_counts_handles[i] = alloc.handle;

            let alloc2 = allocator.create_buffer(
                counts_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )?;
            shadow_counts[i] = alloc2.buffer;
            shadow_counts_handles[i] = alloc2.handle;
        }

        // ---- Allocate group base offsets buffer ----
        let group_bases_alloc = allocator.create_buffer(
            group_bases_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        let group_bases = group_bases_alloc.buffer;
        let group_bases_handle = group_bases_alloc.handle;

        // ---- Create cull compute descriptor set layout ----
        let cull_bindings = [
            // binding 0: GpuObjectData[] (read)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // binding 1: DrawIndexedIndirectCommand[] opaque (write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // binding 2: opaque group counts (read/write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // binding 3: DrawIndexedIndirectCommand[] shadow (write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // binding 4: shadow group counts (read/write)
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // binding 5: group_base_offsets[] (read) — Phase 8B: [0]=0 always
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&cull_bindings),
                None,
            )?
        };

        // ---- Create object SSBO descriptor set layout for graphics (set 3) ----
        let ssbo_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let object_ssbo_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&ssbo_binding)),
                None,
            )?
        };

        // ---- Create pipeline layout with push constants ----
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CullPushConstants>() as u32);

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_constant_range)),
                None,
            )?
        };

        // ---- Create descriptor pool ----
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(6 * MAX_FRAMES_IN_FLIGHT as u32 + 1),
        ];

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32 + 1)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        // ---- Allocate and write cull descriptor sets ----
        let mut descriptor_sets = [vk::DescriptorSet::null(); MAX_FRAMES_IN_FLIGHT];
        let layouts: Vec<_> = (0..MAX_FRAMES_IN_FLIGHT).map(|_| descriptor_set_layout).collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        for (i, set) in sets.into_iter().enumerate() {
            descriptor_sets[i] = set;
        }

        // Write descriptor sets
        for frame in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_infos = [
                vk::DescriptorBufferInfo::default()
                    .buffer(object_ssbo)
                    .offset(0)
                    .range(ssbo_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(opaque_cmds[frame])
                    .offset(0)
                    .range(indirect_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(opaque_counts[frame])
                    .offset(0)
                    .range(counts_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(shadow_cmds[frame])
                    .offset(0)
                    .range(indirect_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(shadow_counts[frame])
                    .offset(0)
                    .range(counts_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(group_bases)
                    .offset(0)
                    .range(group_bases_size),
            ];

            let writes: Vec<_> = buffer_infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[frame])
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // ---- Allocate object SSBO descriptor set for graphics ----
        let ssbo_set_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&object_ssbo_set_layout));

        let object_ssbo_set = unsafe { device.allocate_descriptor_sets(&ssbo_set_alloc)?[0] };

        let ssbo_info = vk::DescriptorBufferInfo::default()
            .buffer(object_ssbo)
            .offset(0)
            .range(ssbo_size);

        let ssbo_write = vk::WriteDescriptorSet::default()
            .dst_set(object_ssbo_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&ssbo_info));

        unsafe { device.update_descriptor_sets(std::slice::from_ref(&ssbo_write), &[]) };

        // ---- Load and create cull compute pipeline ----
        let pipeline = Self::create_cull_pipeline(device, pipeline_layout)?;

        // ---- Phase 8B: Create mega buffers ----
        let mega = MegaBuffers::new(allocator)?;

        println!(
            "[GpuCull] Phase 8B: SSBO={:.2}MB, IndirectCmds={:.2}MB×2, MegaVB={:.0}MB, MegaIB={:.0}MB",
            ssbo_size as f64 / (1024.0 * 1024.0),
            indirect_size as f64 / (1024.0 * 1024.0),
            MEGA_VB_SIZE as f64 / (1024.0 * 1024.0),
            MEGA_IB_SIZE as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            device: device.clone(),
            object_ssbo,
            object_ssbo_handle,
            object_ssbo_mapped,
            needs_host_barrier: false,
            opaque_cmds,
            opaque_cmds_handles,
            shadow_cmds,
            shadow_cmds_handles,
            opaque_counts,
            opaque_counts_handles,
            shadow_counts,
            shadow_counts_handles,
            group_bases,
            group_bases_handle,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            object_ssbo_set_layout,
            object_ssbo_set,
            dirty_objects: Vec::with_capacity(1024),
            dirty_index: std::collections::HashMap::with_capacity(1024),
            mega,
            total_alive: 0,
            #[cfg(debug_assertions)]
            fence_waited_this_frame: false,
            object_mirror: vec![GpuObjectData::default(); MAX_GPU_OBJECTS as usize],
        })
    }

    fn create_cull_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        let shader_code = std::fs::read("shaders/compiled/cull.comp.spv")?;
        let shader_code_aligned: Vec<u32> = shader_code
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let shader_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&shader_code_aligned),
                None,
            )?
        };

        let entry_name = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(pipeline)
    }

    #[cfg(debug_assertions)]
    pub fn assert_fence_waited(&mut self) {
        self.fence_waited_this_frame = true;
    }

    #[cfg(debug_assertions)]
    pub fn reset_fence_flag(&mut self) {
        self.fence_waited_this_frame = false;
    }

    /// Flush all pending GpuObjectData writes to the GPU SSBO.
    ///
    /// # Safety invariant
    /// MUST be called AFTER `wait_for_fences()` for the current frame slot.
    pub fn flush_dirty(&mut self, memory_ctx: &mut MemoryContext) {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.fence_waited_this_frame,
            "flush_dirty called before wait_for_fences — SSBO race condition!"
        );

        if self.dirty_objects.is_empty() {
            return;
        }

        if let Some(base) = self.object_ssbo_mapped {
            for &(id, ref data) in &self.dirty_objects {
                let offset = id as usize * 128;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data as *const GpuObjectData as *const u8,
                        base.as_ptr().add(offset),
                        128,
                    );
                }
            }
            self.needs_host_barrier = true;
        } else {
            self.flush_dirty_staged(memory_ctx);
            self.needs_host_barrier = false;
        }

        self.dirty_objects.clear();
        self.dirty_index.clear();
    }

    /// Batches all dirty entries into a single staging region + single transfer submit.
    fn flush_dirty_staged(&mut self, memory_ctx: &mut MemoryContext) {
        if self.dirty_objects.is_empty() {
            return;
        }

        let entry_size = 128usize;
        let total_bytes = self.dirty_objects.len() * entry_size;

        let mut staging_data = Vec::with_capacity(total_bytes);
        let mut copy_regions = Vec::with_capacity(self.dirty_objects.len());

        for (i, &(id, ref data)) in self.dirty_objects.iter().enumerate() {
            let src_offset_in_staging = (i * entry_size) as u64;
            let dst_offset_in_ssbo = (id as u64) * (entry_size as u64);

            let bytes = unsafe {
                std::slice::from_raw_parts(
                    data as *const GpuObjectData as *const u8,
                    entry_size,
                )
            };
            staging_data.extend_from_slice(bytes);

            copy_regions.push(vk::BufferCopy {
                src_offset: src_offset_in_staging,
                dst_offset: dst_offset_in_ssbo,
                size: entry_size as u64,
            });
        }

        let transfer = &mut memory_ctx.transfer;

        assert!(
            staging_data.len() as u64 <= transfer.staging_size,
            "Batched SSBO flush {} B exceeds staging belt {} B",
            staging_data.len(),
            transfer.staging_size,
        );

        if transfer.staging_offset + staging_data.len() as u64 > transfer.staging_size {
            transfer.staging_offset = 0;
        }
        let staging_base = transfer.staging_offset;

        unsafe {
            std::ptr::copy_nonoverlapping(
                staging_data.as_ptr(),
                transfer.staging_mapped.as_ptr().add(staging_base as usize),
                staging_data.len(),
            );
        }
        transfer.staging_offset += crate::memory::align_up(staging_data.len() as u64, 64);

        for region in &mut copy_regions {
            region.src_offset += staging_base;
        }

        let cmd = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(transfer.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            transfer.device.allocate_command_buffers(&info)
                .expect("Failed to allocate transfer cmd buffer")[0]
        };

        unsafe {
            transfer.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).expect("Failed to begin transfer cmd buffer");

            transfer.device.cmd_copy_buffer(
                cmd,
                transfer.staging_buffer,
                self.object_ssbo,
                &copy_regions,
            );

            if transfer.is_dedicated {
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(self.object_ssbo)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(transfer.queue_family_index)
                    .dst_queue_family_index(transfer.graphics_family_index);

                transfer.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&barrier),
                    &[],
                );
            }

            transfer.device.end_command_buffer(cmd)
                .expect("Failed to end transfer cmd buffer");
        }

        let timeline_value = transfer.submit_timeline(cmd)
            .expect("Failed to submit batched SSBO transfer");

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&transfer.timeline_semaphore))
            .values(std::slice::from_ref(&timeline_value));

        unsafe {
            transfer.device.wait_semaphores(&wait_info, u64::MAX)
                .expect("Failed to wait for batched SSBO transfer");
        }
    }

    /// Queue an object for SSBO update. Updates CPU mirror.
    /// Tier 1 fix: O(1) deduplication via HashMap index.
    /// Old code: `.iter_mut().find()` was O(n) per call → O(n²) per sector stream.
    pub fn queue_dirty(&mut self, id: u32, data: GpuObjectData) {
        if (id as usize) < self.object_mirror.len() {
            self.object_mirror[id as usize] = data;
        }

        if let Some(&idx) = self.dirty_index.get(&id) {
            // Update in-place — O(1)
            self.dirty_objects[idx].1 = data;
        } else {
            let idx = self.dirty_objects.len();
            self.dirty_objects.push((id, data));
            self.dirty_index.insert(id, idx);
        }
    }

    /// Get a copy of object data from the CPU mirror.
    pub fn get_object_data(&self, id: u32) -> GpuObjectData {
        if (id as usize) < self.object_mirror.len() {
            self.object_mirror[id as usize]
        } else {
            GpuObjectData::default()
        }
    }

    /// Phase 8B: Upload group base offsets inline. group_bases[0] = 0 always.
    /// Retained for cull shader descriptor layout compatibility.
    pub fn update_group_base_offsets_inline(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
    ) {
        // Phase 8B: all objects in group 0, base offset is 0
        let base_offsets = [0u32; MAX_BUFFER_GROUPS as usize];
        let byte_size = (MAX_BUFFER_GROUPS as usize) * 4;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                base_offsets.as_ptr() as *const u8,
                byte_size,
            )
        };

        unsafe {
            device.cmd_update_buffer(cmd, self.group_bases, 0, bytes);
        }
    }

    pub fn destroy(&mut self, allocator: &mut GpuAllocator) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_set_layout(self.object_ssbo_set_layout, None);
        }

        allocator.free_buffer(self.object_ssbo_handle);
        allocator.free_buffer(self.group_bases_handle);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            allocator.free_buffer(self.opaque_cmds_handles[i]);
            allocator.free_buffer(self.shadow_cmds_handles[i]);
            allocator.free_buffer(self.opaque_counts_handles[i]);
            allocator.free_buffer(self.shadow_counts_handles[i]);
        }

        // Phase 8B: destroy mega buffers
        self.mega.destroy(allocator);
    }
}