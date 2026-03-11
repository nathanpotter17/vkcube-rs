//! Phase 8A: GPU-Driven Culling Resources
//!
//! Replaces CPU-side frustum culling with a compute shader dispatch.
//! Objects stored in a persistent SSBO; culled results written to
//! indirect command buffers consumed by `cmd_draw_indexed_indirect_count`.
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
/// Maximum buffer groups for Phase 8A (per-sector VB/IB still exist)
pub const MAX_BUFFER_GROUPS: u32 = 256;

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
    /// Index into the shared index buffer (in indices, not bytes)
    pub first_index: u32,
    /// Number of indices to draw; 0 = slot empty (skip in cull shader)
    pub index_count: u32,
    /// Base vertex added to each index value
    pub vertex_offset: i32,
    /// Material ID for fragment shader lookup
    pub material_id: u32,
    /// VB/IB pair index (Phase 8A); always 0 in Phase 8B
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
        buffer_group: u32,
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
            buffer_group,
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
//  BufferGroup — VB/IB pair tracking for Phase 8A
// ====================================================================

/// Tracks a VB/IB pair for grouped indirect dispatch.
/// Phase 8B eliminates this when mega buffers merge all geometry.
#[derive(Clone, Copy, Debug)]
pub struct BufferGroup {
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    /// Base offset into the indirect command buffer for this group
    pub cmd_base_offset: u32,
    /// Index of this group (for count buffer offset calculation)
    pub group_index: u32,
    /// Number of draws written by cull shader (read back from count buffer)
    pub draw_count: u32,
    /// Whether this group is active (has geometry)
    pub active: bool,
}

impl Default for BufferGroup {
    fn default() -> Self {
        Self {
            vertex_buffer: vk::Buffer::null(),
            index_buffer: vk::Buffer::null(),
            cmd_base_offset: 0,
            group_index: 0,
            draw_count: 0,
            active: false,
        }
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
    pub opaque_counts: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
    pub opaque_counts_handles: [BufferHandle; MAX_FRAMES_IN_FLIGHT],
    pub shadow_counts: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
    pub shadow_counts_handles: [BufferHandle; MAX_FRAMES_IN_FLIGHT],

    // ---- Group base offsets (prefix sum, read by cull shader) ----
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
    /// Objects needing SSBO re-upload this frame.
    /// Populated by add_object / transform updates.
    /// Flushed in flush_dirty(), called AFTER wait_for_fences (§3.3).
    pub dirty_objects: Vec<(u32, GpuObjectData)>,

    // ---- Buffer groups for Phase 8A ----
    pub buffer_groups: Vec<BufferGroup>,

    /// Total number of alive objects (next available slot)
    pub total_alive: u32,

    /// Debug assertion: fence was waited before flush_dirty
    #[cfg(debug_assertions)]
    fence_waited_this_frame: bool,
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
        // ReBAR = Resizable BAR: CPU can directly write to GPU VRAM
        let (object_ssbo, object_ssbo_handle, object_ssbo_mapped) = {
            // First try HOST_VISIBLE | DEVICE_LOCAL (ReBAR)
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
                    // Fallback to GpuOnly + staging transfers
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
            // binding 5: group_base_offsets[] (read)
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
                .descriptor_count(6 * MAX_FRAMES_IN_FLIGHT as u32 + 1), // cull sets + ssbo set
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

        // Write object SSBO to graphics descriptor set
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

        println!(
            "[GpuCull] Initialized: SSBO={:.2}MB, IndirectCmds={:.2}MB×2, Counts={:.2}KB×2",
            ssbo_size as f64 / (1024.0 * 1024.0),
            indirect_size as f64 / (1024.0 * 1024.0),
            counts_size as f64 / 1024.0,
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
            buffer_groups: Vec::with_capacity(MAX_BUFFER_GROUPS as usize),
            total_alive: 0,
            #[cfg(debug_assertions)]
            fence_waited_this_frame: false,
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

    /// Mark that the fence was waited this frame (debug assertion helper).
    #[cfg(debug_assertions)]
    pub fn assert_fence_waited(&mut self) {
        self.fence_waited_this_frame = true;
    }

    /// Reset the fence-waited flag at frame start (debug assertion helper).
    #[cfg(debug_assertions)]
    pub fn reset_fence_flag(&mut self) {
        self.fence_waited_this_frame = false;
    }

    /// Flush all pending GpuObjectData writes to the GPU SSBO.
    ///
    /// # Safety invariant
    /// MUST be called AFTER `wait_for_fences()` for the current frame slot.
    /// Calling before fence wait risks writing into a slot still in use by GPU.
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
            // ReBAR path: direct memcpy, no staging buffer, no transfer queue
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
            self.needs_host_barrier = true; // signal: use HOST_WRITE barrier pre-cull
        } else {
            // Staging path: batch all dirty entries into staging + transfer
            self.flush_dirty_staged(memory_ctx);
            self.needs_host_barrier = false; // signal: use TRANSFER_WRITE barrier pre-cull
        }

        self.dirty_objects.clear();
    }

    /// Staging-based upload for non-ReBAR systems
    fn flush_dirty_staged(&mut self, memory_ctx: &mut MemoryContext) {
        // For simplicity, upload each dirty object individually through the transfer queue.
        // A more optimal implementation would batch into a single staging buffer region.
        for &(id, ref data) in &self.dirty_objects {
            let offset = id as u64 * 128;
            let bytes = unsafe {
                std::slice::from_raw_parts(data as *const GpuObjectData as *const u8, 128)
            };
            // Use transfer queue for async upload
            let _ = memory_ctx.transfer.upload_region(
                bytes,
                self.object_ssbo,
                offset,
            );
        }
    }

    /// Queue an object for SSBO update. Called by World::add_object.
    pub fn queue_dirty(&mut self, id: u32, data: GpuObjectData) {
        // Check if already queued and update in place
        if let Some(entry) = self.dirty_objects.iter_mut().find(|(oid, _)| *oid == id) {
            entry.1 = data;
        } else {
            self.dirty_objects.push((id, data));
        }
    }

    /// Get a copy of object data (for modification + re-queue)
    pub fn get_object_data(&self, _id: u32) -> GpuObjectData {
        // In a full implementation, this would read from a CPU-side mirror
        // or directly from the mapped SSBO. For now return default.
        GpuObjectData::default()
    }

    /// Register or update a buffer group. Returns the group index.
    pub fn register_buffer_group(
        &mut self,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) -> u32 {
        // Check if this VB/IB pair already exists
        for (i, group) in self.buffer_groups.iter_mut().enumerate() {
            if group.vertex_buffer == vertex_buffer && group.index_buffer == index_buffer {
                group.active = true;
                group.group_index = i as u32;
                return i as u32;
            }
        }

        // Find an inactive slot or add new
        for (i, group) in self.buffer_groups.iter_mut().enumerate() {
            if !group.active {
                group.vertex_buffer = vertex_buffer;
                group.index_buffer = index_buffer;
                group.active = true;
                group.group_index = i as u32;
                return i as u32;
            }
        }

        // Add new group
        let idx = self.buffer_groups.len() as u32;
        self.buffer_groups.push(BufferGroup {
            vertex_buffer,
            index_buffer,
            cmd_base_offset: 0,
            group_index: idx,
            draw_count: 0,
            active: true,
        });
        idx
    }

    /// Deactivate a buffer group (on sector eviction)
    pub fn deactivate_buffer_group(&mut self, vertex_buffer: vk::Buffer) {
        for group in &mut self.buffer_groups {
            if group.vertex_buffer == vertex_buffer {
                group.active = false;
                group.draw_count = 0;
            }
        }
    }

    /// Compute and upload group base offsets (prefix sum).
    /// Call this before the cull dispatch each frame.
    pub fn update_group_base_offsets(&mut self, memory_ctx: &mut MemoryContext) {
        let mut base_offsets: Vec<u32> = Vec::with_capacity(MAX_BUFFER_GROUPS as usize);
        let mut running_offset = 0u32;

        for group in &mut self.buffer_groups {
            group.cmd_base_offset = running_offset;
            base_offsets.push(running_offset);
            // Reserve space for max draws per group (conservative)
            // In practice, use previous frame's count + some headroom
            running_offset += MAX_INDIRECT_DRAWS / MAX_BUFFER_GROUPS.max(1);
        }

        // Pad to MAX_BUFFER_GROUPS
        while base_offsets.len() < MAX_BUFFER_GROUPS as usize {
            base_offsets.push(running_offset);
        }

        // Upload via transfer queue
        let bytes = unsafe {
            std::slice::from_raw_parts(
                base_offsets.as_ptr() as *const u8,
                base_offsets.len() * 4,
            )
        };
        let _ = memory_ctx.transfer.upload_region(bytes, self.group_bases, 0);
    }

    /// Get active buffer groups for draw dispatch
    pub fn active_groups(&self) -> impl Iterator<Item = &BufferGroup> {
        self.buffer_groups.iter().filter(|g| g.active)
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
    }
}