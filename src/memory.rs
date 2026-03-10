use ash::{vk, Device};
use std::collections::HashMap;
use std::ptr::NonNull;

// ===== Public Constants =====

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;
pub const ENABLE_VALIDATION: bool = cfg!(debug_assertions);

/// Pool block size.  64 MB gives good TLB coverage per VkDeviceMemory
/// while keeping the total number of Vulkan allocations well under the
/// ~4 096 driver limit on most ICDs.
const POOL_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

/// Ring-buffer capacity for per-frame transient data (uniforms, dynamic
/// vertices).  4 MB per frame supports ~20 000 draws × 192-byte UBOs.
const RING_BUFFER_SIZE: u64 = 4 * 1024 * 1024;

/// Frames allowed in flight simultaneously.  Must match the renderer.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Staging-belt size for CPU → GPU transfers.  The transfer queue uses a
/// ring of staging memory so multiple uploads can be in-flight at once.
const STAGING_BUFFER_SIZE: u64 = 64 * 1024 * 1024;

/// Maximum VRAM budget ratio.  When pool allocations exceed this fraction
/// of the device-local heap, the budget system starts evicting.
const VRAM_BUDGET_RATIO: f64 = 0.85;

// ===== Memory Location =====

/// High-level placement hint that maps to Vulkan memory-property flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// `DEVICE_LOCAL` – fastest GPU access, not CPU-visible.
    GpuOnly,
    /// `HOST_VISIBLE | HOST_COHERENT` – CPU-writable, GPU-readable.
    CpuToGpu,
}

impl MemoryLocation {
    fn required_flags(self) -> vk::MemoryPropertyFlags {
        match self {
            Self::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            Self::CpuToGpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        }
    }
}

// ===== Handles =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub(crate) u64);

// ===== Helpers =====

/// Align `value` up to the next multiple of `alignment` (power-of-2).
#[inline(always)]
const fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Format a byte count with the most readable unit (B / KB / MB / GB).
fn fmt_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ====================================================================
//  Pool Block – one VkDeviceMemory carved up by a coalescing free-list
// ====================================================================

#[derive(Debug, Clone)]
struct FreeRegion {
    offset: u64,
    size: u64,
}

/// Represents a single large `VkDeviceMemory` allocation.
///
/// Sub-allocation uses a sorted free-list with first-fit search and
/// immediate coalescing on free.  This gives O(n) alloc in the number
/// of free regions – perfectly adequate for the few hundred buffers a
/// typical frame touches.
struct PoolBlock {
    memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: Option<NonNull<u8>>,
    free_regions: Vec<FreeRegion>,
    total_free: u64,
}

unsafe impl Send for PoolBlock {}

impl PoolBlock {
    fn new(
        device: &Device,
        size: u64,
        memory_type_index: u32,
        map: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        let mapped_ptr = if map {
            let ptr = unsafe {
                device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
            };
            NonNull::new(ptr as *mut u8)
        } else {
            None
        };

        Ok(Self {
            memory,
            size,
            mapped_ptr,
            free_regions: vec![FreeRegion { offset: 0, size }],
            total_free: size,
        })
    }

    /// First-fit sub-allocation respecting `alignment`.
    fn alloc(
        &mut self,
        size: u64,
        alignment: u64,
    ) -> Option<(u64, Option<NonNull<u8>>)> {
        for i in 0..self.free_regions.len() {
            let region = &self.free_regions[i];
            let aligned = align_up(region.offset, alignment);
            let padding = aligned - region.offset;
            let needed = padding + size;

            if region.size < needed {
                continue;
            }

            let remaining = region.size - needed;

            if padding > 0 && remaining > 0 {
                self.free_regions[i].size = padding;
                self.free_regions.insert(
                    i + 1,
                    FreeRegion { offset: aligned + size, size: remaining },
                );
            } else if padding > 0 {
                self.free_regions[i].size = padding;
            } else if remaining > 0 {
                self.free_regions[i] =
                    FreeRegion { offset: aligned + size, size: remaining };
            } else {
                self.free_regions.remove(i);
            }

            self.total_free -= needed;

            let mapped = self.mapped_ptr.map(|base| unsafe {
                NonNull::new_unchecked(base.as_ptr().add(aligned as usize))
            });
            return Some((aligned, mapped));
        }
        None
    }

    /// Return a region to the free-list and coalesce neighbours.
    fn free(&mut self, offset: u64, size: u64) {
        self.total_free += size;

        let pos = self.free_regions.partition_point(|r| r.offset < offset);
        self.free_regions.insert(pos, FreeRegion { offset, size });

        // Coalesce right.
        if pos + 1 < self.free_regions.len() {
            let cur_end =
                self.free_regions[pos].offset + self.free_regions[pos].size;
            if cur_end == self.free_regions[pos + 1].offset {
                self.free_regions[pos].size += self.free_regions[pos + 1].size;
                self.free_regions.remove(pos + 1);
            }
        }
        // Coalesce left.
        if pos > 0 {
            let prev_end = self.free_regions[pos - 1].offset
                + self.free_regions[pos - 1].size;
            if prev_end == self.free_regions[pos].offset {
                self.free_regions[pos - 1].size += self.free_regions[pos].size;
                self.free_regions.remove(pos);
            }
        }
    }
}

// ====================================================================
//  GpuAllocator – VMA-style pool allocator
// ====================================================================
#[derive(Clone)] // this is safe, vk::buffer is actually just a handle
struct AllocationRecord {
    buffer: vk::Buffer,
    usage: vk::BufferUsageFlags,
    memory_type_index: u32,
    block_index: usize,
    offset: u64,
    size: u64,
}

/// Tracks a VkImage sub-allocated from a pool block.
struct ImageAllocationRecord {
    image: vk::Image,
    view: Option<vk::ImageView>,
    memory_type_index: u32,
    block_index: usize,
    offset: u64,
    size: u64,
    format: vk::Format,
    extent: vk::Extent3D,
    mip_levels: u32,
}

/// Returned from [`GpuAllocator::create_buffer`].
pub struct BufferAllocation {
    pub handle: BufferHandle,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
    /// Non-null only for `CpuToGpu` allocations.
    pub mapped_ptr: Option<NonNull<u8>>,
}

/// Returned from [`GpuAllocator::create_image`].
pub struct ImageAllocation {
    pub handle: ImageHandle,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
}

/// VMA-style GPU memory allocator.
pub struct GpuAllocator {
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pools: HashMap<u32, Vec<PoolBlock>>,
    allocations: HashMap<BufferHandle, AllocationRecord>,
    image_allocations: HashMap<ImageHandle, ImageAllocationRecord>,
    next_handle: u64,
    next_image_handle: u64,
}

impl GpuAllocator {
    pub fn new(
        device: Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        for i in 0..memory_properties.memory_heap_count {
            let heap = memory_properties.memory_heaps[i as usize];
            let kind = if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                "VRAM"
            } else {
                "System"
            };
            println!(
                "[GpuAllocator] Heap {}: {:.2} GB ({})",
                i,
                heap.size as f64 / (1024.0 * 1024.0 * 1024.0),
                kind,
            );
        }

        Self {
            device,
            memory_properties,
            pools: HashMap::new(),
            allocations: HashMap::new(),
            image_allocations: HashMap::new(),
            next_handle: 1,
            next_image_handle: 1,
        }
    }

    /// Create a `VkBuffer` backed by sub-allocated pool memory.
    pub fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let mem_req =
            unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let memory_type_index =
            self.find_memory_type(mem_req.memory_type_bits, location.required_flags())?;
        let should_map = location == MemoryLocation::CpuToGpu;

        let (block_index, offset, mapped_ptr) = self.pool_alloc(
            memory_type_index,
            mem_req.size,
            mem_req.alignment,
            should_map,
        )?;

        let memory =
            self.pools.get(&memory_type_index).unwrap()[block_index].memory;
        unsafe {
            self.device.bind_buffer_memory(buffer, memory, offset)?;
        }

        let handle = BufferHandle(self.next_handle);
        self.next_handle += 1;

        self.allocations.insert(
            handle,
            AllocationRecord {
                buffer,
                usage,
                memory_type_index,
                block_index,
                offset,
                size: mem_req.size,
            },
        );

        Ok(BufferAllocation { handle, buffer, memory, offset, size: mem_req.size, mapped_ptr })
    }

    /// Create a `VkImage` + `VkImageView` backed by sub-allocated pool memory.
    ///
    /// The image is created in `UNDEFINED` layout. The caller must
    /// transition it before use (typically via a pipeline barrier in
    /// the transfer or graphics command buffer).
    pub fn create_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        location: MemoryLocation,
    ) -> Result<ImageAllocation, Box<dyn std::error::Error>> {
        let extent = vk::Extent3D { width, height, depth: 1 };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { self.device.create_image(&image_info, None)? };
        let mem_req = unsafe { self.device.get_image_memory_requirements(image) };

        let memory_type_index =
            self.find_memory_type(mem_req.memory_type_bits, location.required_flags())?;
        // Images are GPU-only in practice; never map.
        let should_map = false;

        let (block_index, offset, _mapped_ptr) = self.pool_alloc(
            memory_type_index,
            mem_req.size,
            mem_req.alignment,
            should_map,
        )?;

        let memory =
            self.pools.get(&memory_type_index).unwrap()[block_index].memory;
        unsafe {
            self.device.bind_image_memory(image, memory, offset)?;
        }

        // Determine aspect mask from format.
        let aspect = if format == vk::Format::D32_SFLOAT
            || format == vk::Format::D16_UNORM
            || format == vk::Format::D32_SFLOAT_S8_UINT
            || format == vk::Format::D24_UNORM_S8_UINT
        {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: mip_levels,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?
        };

        let handle = ImageHandle(self.next_image_handle);
        self.next_image_handle += 1;

        self.image_allocations.insert(
            handle,
            ImageAllocationRecord {
                image,
                view: Some(view),
                memory_type_index,
                block_index,
                offset,
                size: mem_req.size,
                format,
                extent,
                mip_levels,
            },
        );

        Ok(ImageAllocation {
            handle,
            image,
            view,
            memory,
            offset,
            size: mem_req.size,
            format,
            extent,
            mip_levels,
        })
    }

    /// Destroy a buffer and return its memory to the pool.
    pub fn free_buffer(&mut self, handle: BufferHandle) {
        if let Some(rec) = self.allocations.remove(&handle) {
            unsafe { self.device.destroy_buffer(rec.buffer, None) }
            if let Some(blocks) = self.pools.get_mut(&rec.memory_type_index) {
                if rec.block_index < blocks.len() {
                    blocks[rec.block_index].free(rec.offset, rec.size);
                }
            }
        }
    }

    /// Destroy an image (and its view) and return memory to the pool.
    pub fn free_image(&mut self, handle: ImageHandle) {
        if let Some(rec) = self.image_allocations.remove(&handle) {
            unsafe {
                if let Some(view) = rec.view {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image(rec.image, None);
            }
            if let Some(blocks) = self.pools.get_mut(&rec.memory_type_index) {
                if rec.block_index < blocks.len() {
                    blocks[rec.block_index].free(rec.offset, rec.size);
                }
            }
        }
    }

    /// Total bytes allocated across all pool blocks.
    pub fn total_allocated(&self) -> u64 {
        self.pools.values().flat_map(|bs| bs.iter()).map(|b| b.size).sum()
    }

    /// Total bytes actually used (allocated minus free).
    pub fn total_used(&self) -> u64 {
        self.pools
            .values()
            .flat_map(|bs| bs.iter())
            .map(|b| b.size - b.total_free)
            .sum()
    }

    /// Number of live buffer handles.
    pub fn live_buffer_count(&self) -> usize {
        self.allocations.len()
    }

    /// Number of live image handles.
    pub fn live_image_count(&self) -> usize {
        self.image_allocations.len()
    }

    /// Print a compact utilisation summary to stdout.
    pub fn print_stats(&self) {
        let mut total_used: u64 = 0;
        let mut total_cap: u64 = 0;
        let mut n_blocks: usize = 0;

        for (ty, blocks) in &self.pools {
            for blk in blocks {
                let used = blk.size - blk.total_free;
                total_used += used;
                total_cap += blk.size;
                n_blocks += 1;
                println!(
                    "  [type {}] block {}  used {}  free {}  regions {}",
                    ty,
                    fmt_bytes(blk.size),
                    fmt_bytes(used),
                    fmt_bytes(blk.total_free),
                    blk.free_regions.len(),
                );
            }
        }
        println!(
            "[GpuAllocator] {} blocks, {}/{} used, {} live buffers, {} live images",
            n_blocks,
            fmt_bytes(total_used),
            fmt_bytes(total_cap),
            self.allocations.len(),
            self.image_allocations.len(),
        );
    }

    /// Phase 5 (§5.4): Defragment one pool block.
    ///
    /// Selects the block with the highest free-region count (most
    /// fragmented), allocates a fresh block of the same memory type,
    /// copies all live sub-allocations into the new block, and updates
    /// all `BufferHandle` → `vk::Buffer` mappings in the handle table.
    ///
    /// **Caller contract:**
    /// - Call only when no in-flight commands reference buffers in the
    ///   fragmented block (after fence wait or during device idle).
    /// - Apply the returned `BufferRemap` to all `MeshRange` entries in
    ///   `World.objects[*].lod.levels[*]` via `World::apply_buffer_remap`.
    ///   `DrawCommand` lists rebuild from `MeshRange` each frame, so they
    ///   pick up new handles automatically.
    ///
    /// Returns `Ok(None)` if no block needs defragmentation (all blocks
    /// have ≤ 2 free regions).
    pub fn defragment_one_block(
        &mut self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Option<HashMap<vk::Buffer, vk::Buffer>>, Box<dyn std::error::Error>> {
        // ---- Find the most fragmented block ----
        let mut worst_type: Option<u32> = None;
        let mut worst_block_idx: usize = 0;
        let mut worst_frag_count: usize = 0;

        for (&mem_type, blocks) in &self.pools {
            for (bi, block) in blocks.iter().enumerate() {
                // ≤ 2 free regions = at most one hole.  Not worth moving.
                if block.free_regions.len() > worst_frag_count
                    && block.free_regions.len() > 2
                {
                    worst_type = Some(mem_type);
                    worst_block_idx = bi;
                    worst_frag_count = block.free_regions.len();
                }
            }
        }

        let mem_type = match worst_type {
            Some(t) => t,
            None => return Ok(None),
        };

        // ---- Snapshot live allocations in the target block ----
        //
        // Clone safety: all fields are Copy.  The snapshot is read-only.
        // Originals are overwritten (not removed) via the same key below.
        let to_move: Vec<(BufferHandle, AllocationRecord)> = self.allocations
            .iter()
            .filter(|(_, rec)| {
                rec.memory_type_index == mem_type
                    && rec.block_index == worst_block_idx
            })
            .map(|(&handle, rec)| (handle, rec.clone()))
            .collect();

        if to_move.is_empty() {
            return Ok(None);
        }

        let old_block_size = self.pools[&mem_type][worst_block_idx].size;
        let should_map = self.pools[&mem_type][worst_block_idx].mapped_ptr.is_some();

        // ---- Allocate fresh block ----
        let new_block = PoolBlock::new(
            &self.device, old_block_size, mem_type, should_map,
        )?;
        let blocks = self.pools.get_mut(&mem_type).unwrap();
        let new_block_idx = blocks.len();
        blocks.push(new_block);

        // ---- Create new buffers + sub-allocate in new block ----
        //
        // Collect all state needed for the post-copy record update.
        struct MovedBuffer {
            handle: BufferHandle,
            old_buffer: vk::Buffer,
            old_offset: u64,
            old_size: u64,
            new_buffer: vk::Buffer,
            new_offset: u64,
            new_mapped_ptr: Option<NonNull<u8>>,
            usage: vk::BufferUsageFlags,
        }

        let mut moved: Vec<MovedBuffer> = Vec::with_capacity(to_move.len());

        for (handle, old_rec) in &to_move {
            let blocks = self.pools.get_mut(&mem_type).unwrap();
            let (new_offset, new_mapped_ptr) = blocks[new_block_idx]
                .alloc(old_rec.size, 256) // §5.4: 256-byte minimum alloc class
                .ok_or("Defrag: sub-alloc in new block failed")?;

            let new_memory = blocks[new_block_idx].memory;

            // Recreate with original usage + TRANSFER_SRC/DST for the copy.
            let recreate_usage = old_rec.usage
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;

            let buffer_info = vk::BufferCreateInfo::default()
                .size(old_rec.size)
                .usage(recreate_usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let new_buffer = unsafe {
                self.device.create_buffer(&buffer_info, None)?
            };
            unsafe {
                self.device.bind_buffer_memory(new_buffer, new_memory, new_offset)?;
            }

            moved.push(MovedBuffer {
                handle: *handle,
                old_buffer: old_rec.buffer,
                old_offset: old_rec.offset,
                old_size: old_rec.size,
                new_buffer,
                new_offset,
                new_mapped_ptr,
                usage: old_rec.usage,
            });
        }

        // ---- Record and submit copy commands ----
        let cmd = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&info)?[0];
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            cmd
        };

        for m in &moved {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(m.old_size);
            unsafe {
                self.device.cmd_copy_buffer(
                    cmd, m.old_buffer, m.new_buffer,
                    std::slice::from_ref(&region),
                );
            }
        }

        unsafe {
            self.device.end_command_buffer(cmd)?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd));
            self.device.queue_submit(
                queue, std::slice::from_ref(&submit), vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(queue)?;
            self.device.free_command_buffers(
                command_pool, std::slice::from_ref(&cmd),
            );
        }

        // ---- GPU copy complete — update records, build remap ----
        //
        // For each moved buffer:
        //   1. Destroy old VkBuffer (safe: queue_wait_idle done)
        //   2. Free old sub-allocation in old block
        //   3. Overwrite AllocationRecord IN-PLACE (same BufferHandle key)
        //   4. Insert into remap table for caller
        //
        // Step 3 is what the prior version was missing.  BufferHandle
        // remains valid — external code looking up by handle finds the
        // new buffer.

        let mut remap: HashMap<vk::Buffer, vk::Buffer> =
            HashMap::with_capacity(moved.len());
        let blocks = self.pools.get_mut(&mem_type).unwrap();

        for m in &moved {
            // 1. Destroy old VkBuffer.
            unsafe { self.device.destroy_buffer(m.old_buffer, None); }

            // 2. Return old sub-allocation to old block's free list.
            blocks[worst_block_idx].free(m.old_offset, m.old_size);

            // 3. Overwrite the record — same key, new buffer/offset/block.
            if let Some(rec) = self.allocations.get_mut(&m.handle) {
                rec.buffer = m.new_buffer;
                rec.block_index = new_block_idx;
                rec.offset = m.new_offset;
                // size, usage, memory_type_index — unchanged.
            }

            // 4. Remap: callers holding raw vk::Buffer must translate.
            remap.insert(m.old_buffer, m.new_buffer);
        }

        let frag_after = blocks[worst_block_idx].free_regions.len();
        println!(
            "[GpuAllocator] Defragmented block (type {}, idx {}): \
             moved {} buffers, frag regions {} → {}",
            mem_type, worst_block_idx, moved.len(),
            worst_frag_count, frag_after,
        );

        Ok(Some(remap))
    }

    /// Phase 5 (§5.5): Query VK_EXT_memory_budget for actual
    /// driver-reported budget values.
    ///
    /// Returns `(budget_bytes, usage_bytes)` for the device-local heap.
    /// Returns `None` if the extension is unavailable or no device-local
    /// heap is found.
    pub fn query_memory_budget(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Option<(u64, u64)> {
        let mut budget_props =
            vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default()
            .push_next(&mut budget_props);

        unsafe {
            instance.get_physical_device_memory_properties2(
                physical_device, &mut props2,
            );
        }

        for i in 0..props2.memory_properties.memory_heap_count as usize {
            let heap = props2.memory_properties.memory_heaps[i];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                return Some((
                    budget_props.heap_budget[i],
                    budget_props.heap_usage[i],
                ));
            }
        }

        None
    }

    // ---- internals ----

    fn pool_alloc(
        &mut self,
        memory_type_index: u32,
        size: u64,
        alignment: u64,
        map: bool,
    ) -> Result<(usize, u64, Option<NonNull<u8>>), Box<dyn std::error::Error>> {
        let blocks = self.pools.entry(memory_type_index).or_default();

        for (i, block) in blocks.iter_mut().enumerate() {
            if block.total_free >= size {
                if let Some((offset, mapped)) = block.alloc(size, alignment) {
                    return Ok((i, offset, mapped));
                }
            }
        }

        let block_size = POOL_BLOCK_SIZE.max(align_up(size * 2, 1024 * 1024));
        let new_block =
            PoolBlock::new(&self.device, block_size, memory_type_index, map)?;
        let idx = blocks.len();
        blocks.push(new_block);

        let (offset, mapped) = blocks[idx]
            .alloc(size, alignment)
            .ok_or("Sub-alloc failed on fresh block")?;

        println!(
            "[GpuAllocator] New {} MB block for memory type {} (now {} blocks)",
            block_size / (1024 * 1024),
            memory_type_index,
            idx + 1,
        );

        Ok((idx, offset, mapped))
    }

    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }
        Err("No suitable memory type".into())
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        unsafe {
            // Destroy all live buffers.
            for (_, rec) in self.allocations.drain() {
                self.device.destroy_buffer(rec.buffer, None);
            }
            // Destroy all live images.
            for (_, rec) in self.image_allocations.drain() {
                if let Some(view) = rec.view {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image(rec.image, None);
            }
            // Free all pool blocks.
            for (_, blocks) in self.pools.drain() {
                for block in blocks {
                    if block.mapped_ptr.is_some() {
                        self.device.unmap_memory(block.memory);
                    }
                    self.device.free_memory(block.memory, None);
                }
            }
        }
    }
}

// ====================================================================
//  RingBuffer – per-frame bump allocator for transient data
// ====================================================================

/// Triple-buffered ring allocator for uniform and dynamic-vertex data.
pub struct RingBuffer {
    pub buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    mapped_base: NonNull<u8>,
    #[allow(dead_code)]
    total_size: u64,
    frame_size: u64,
    min_alignment: u64,
    current_frame: usize,
    frame_offset: u64,
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

/// A slice of ring-buffer memory returned by [`RingBuffer::push`].
pub struct RingSlice {
    pub offset: u64,
    pub mapped_ptr: NonNull<u8>,
    pub size: u64,
}

impl RingBuffer {
    pub fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        min_alignment: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let total_size = RING_BUFFER_SIZE;
        let frame_size = total_size / MAX_FRAMES_IN_FLIGHT as u64;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(total_size)
            .usage(
                vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

        let required = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_type = (0..memory_properties.memory_type_count)
            .find(|&i| {
                (mem_req.memory_type_bits & (1 << i)) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or("No HOST_VISIBLE|HOST_COHERENT memory for ring buffer")?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? }

        let raw = unsafe {
            device.map_memory(memory, 0, total_size, vk::MemoryMapFlags::empty())?
        };
        let mapped_base =
            NonNull::new(raw as *mut u8).ok_or("Failed to map ring buffer")?;

        println!(
            "[RingBuffer] {} MB total, {} MB/frame, alignment {}",
            total_size / (1024 * 1024),
            frame_size / (1024 * 1024),
            min_alignment,
        );

        Ok(Self {
            buffer,
            memory,
            mapped_base,
            total_size,
            frame_size,
            min_alignment: min_alignment.max(64),
            current_frame: 0,
            frame_offset: 0,
        })
    }

    pub fn begin_frame(&mut self, frame_index: usize) {
        self.current_frame = frame_index;
        self.frame_offset = 0;
    }

    pub fn push(&mut self, size: u64) -> Option<RingSlice> {
        let aligned = align_up(size, self.min_alignment);
        let base = self.current_frame as u64 * self.frame_size;

        if self.frame_offset + aligned > self.frame_size {
            return None;
        }

        let global = base + self.frame_offset;
        let ptr = unsafe {
            NonNull::new_unchecked(
                self.mapped_base.as_ptr().add(global as usize),
            )
        };
        self.frame_offset += aligned;

        Some(RingSlice { offset: global, mapped_ptr: ptr, size })
    }

    pub fn push_data<T: Copy>(&mut self, data: &T) -> Option<RingSlice> {
        let n = std::mem::size_of::<T>() as u64;
        let slice = self.push(n)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data as *const T as *const u8,
                slice.mapped_ptr.as_ptr(),
                n as usize,
            );
        }
        Some(slice)
    }

    /// Push a byte slice (useful for SSBO uploads).
    pub fn push_bytes(&mut self, data: &[u8]) -> Option<RingSlice> {
        let slice = self.push(data.len() as u64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                slice.mapped_ptr.as_ptr(),
                data.len(),
            );
        }
        Some(slice)
    }
}

// ====================================================================
//  TransferQueue – async GPU uploads via timeline semaphore
// ====================================================================

/// Identifies a single in-flight transfer.  The renderer polls completed
/// tickets each frame to promote chunk load states from Streaming→Ready.
#[derive(Debug, Clone)]
pub struct TransferTicket {
    /// The destination buffer that is being uploaded to.
    pub dst_buffer_handle: Option<BufferHandle>,
    /// The destination image that is being uploaded to (mutually exclusive with buffer).
    pub dst_image_handle: Option<ImageHandle>,
    /// Timeline semaphore value that will be signalled when the copy
    /// command finishes on the transfer queue.
    pub timeline_value: u64,
}

// Keep backward compat: dst_handle as an alias
impl TransferTicket {
    /// Convenience accessor for buffer uploads (backward compat).
    pub fn dst_handle(&self) -> Option<BufferHandle> {
        self.dst_buffer_handle
    }
}

/// Async transfer engine.  Owns a dedicated command pool on the transfer
/// queue family, a HOST_VISIBLE staging ring, and a timeline semaphore
/// for CPU-side completion polling.
pub struct TransferQueue {
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    timeline_semaphore: vk::Semaphore,
    /// Monotonically increasing; bumped once per `upload_async` call.
    next_timeline: u64,
    /// Staging ring: single persistently-mapped buffer used round-robin.
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_mapped: NonNull<u8>,
    staging_size: u64,
    /// Current write offset into the staging ring.
    staging_offset: u64,
    /// Queue family index for ownership transfer barriers.
    pub queue_family_index: u32,
    /// The graphics queue family (for release/acquire barriers).
    pub graphics_family_index: u32,
    /// Whether transfer and graphics are on separate families.
    pub is_dedicated: bool,
}

unsafe impl Send for TransferQueue {}

impl TransferQueue {
    pub fn new(
        device: Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        transfer_queue: vk::Queue,
        transfer_family: u32,
        graphics_family: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let is_dedicated = transfer_family != graphics_family;

        // Command pool for the transfer queue family.
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(
                        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | vk::CommandPoolCreateFlags::TRANSIENT,
                    )
                    .queue_family_index(transfer_family),
                None,
            )?
        };

        // Timeline semaphore (Vulkan 1.2+).
        let mut timeline_info =
            vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);

        let timeline_semaphore = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info),
                None,
            )?
        };

        // HOST_VISIBLE staging ring.
        let staging_info = vk::BufferCreateInfo::default()
            .size(STAGING_BUFFER_SIZE)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { device.create_buffer(&staging_info, None)? };
        let staging_req =
            unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let required = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let staging_type = (0..memory_properties.memory_type_count)
            .find(|&i| {
                (staging_req.memory_type_bits & (1 << i)) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or("No memory type for transfer staging buffer")?;

        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_req.size)
            .memory_type_index(staging_type);
        let staging_memory = unsafe { device.allocate_memory(&staging_alloc, None)? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0)? }

        let raw = unsafe {
            device.map_memory(
                staging_memory,
                0,
                STAGING_BUFFER_SIZE,
                vk::MemoryMapFlags::empty(),
            )?
        };
        let staging_mapped =
            NonNull::new(raw as *mut u8).ok_or("Failed to map transfer staging")?;

        println!(
            "[TransferQueue] family {} (dedicated: {}), staging {} MB, timeline semaphore ready",
            transfer_family,
            is_dedicated,
            STAGING_BUFFER_SIZE / (1024 * 1024),
        );

        Ok(Self {
            device,
            queue: transfer_queue,
            command_pool,
            timeline_semaphore,
            next_timeline: 1,
            staging_buffer,
            staging_memory,
            staging_mapped,
            staging_size: STAGING_BUFFER_SIZE,
            staging_offset: 0,
            queue_family_index: transfer_family,
            graphics_family_index: graphics_family,
            is_dedicated,
        })
    }

    /// Copy `data` into a staging region and submit a transfer command
    /// that copies it into `dst_buffer`.  Returns a ticket whose
    /// `timeline_value` the caller can poll with [`is_complete`].
    pub fn upload_buffer_async(
        &mut self,
        data: &[u8],
        dst_buffer: vk::Buffer,
        dst_handle: BufferHandle,
    ) -> Result<TransferTicket, Box<dyn std::error::Error>> {
        let size = data.len() as u64;
        assert!(
            size <= self.staging_size,
            "Upload {} B exceeds transfer staging belt {} B",
            size,
            self.staging_size,
        );

        // Wrap around the staging ring if needed.
        if self.staging_offset + size > self.staging_size {
            self.staging_offset = 0;
        }

        // Copy data into the staging region.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_mapped.as_ptr().add(self.staging_offset as usize),
                data.len(),
            );
        }

        let staging_offset = self.staging_offset;
        self.staging_offset += align_up(size, 64);

        // Record the copy command.
        let cmd = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.device.allocate_command_buffers(&info)?[0]
        };

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let region = vk::BufferCopy::default()
                .src_offset(staging_offset)
                .dst_offset(0)
                .size(size);
            self.device.cmd_copy_buffer(
                cmd,
                self.staging_buffer,
                dst_buffer,
                std::slice::from_ref(&region),
            );

            // If using a dedicated transfer family, insert a release
            // barrier so the graphics queue can acquire ownership.
            if self.is_dedicated {
                let barrier = vk::BufferMemoryBarrier::default()
                    .buffer(dst_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.graphics_family_index);

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&barrier),
                    &[],
                );
            }

            self.device.end_command_buffer(cmd)?;
        }

        self.submit_timeline(cmd)
            .map(|timeline_value| TransferTicket {
                dst_buffer_handle: Some(dst_handle),
                dst_image_handle: None,
                timeline_value,
            })
    }

    /// Upload pixel data to a VkImage via the staging ring.
    ///
    /// The image must be in `UNDEFINED` or `PREINITIALIZED` layout.
    /// After the copy the image will be in `TRANSFER_DST_OPTIMAL` layout.
    /// The caller must transition to `SHADER_READ_ONLY_OPTIMAL` on the
    /// graphics queue (an acquire barrier if dedicated transfer, or a
    /// simple layout transition otherwise).
    pub fn upload_image_async(
        &mut self,
        data: &[u8],
        dst_image: vk::Image,
        dst_handle: ImageHandle,
        width: u32,
        height: u32,
    ) -> Result<TransferTicket, Box<dyn std::error::Error>> {
        let size = data.len() as u64;
        assert!(
            size <= self.staging_size,
            "Image upload {} B exceeds staging belt {} B",
            size,
            self.staging_size,
        );

        if self.staging_offset + size > self.staging_size {
            self.staging_offset = 0;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_mapped.as_ptr().add(self.staging_offset as usize),
                data.len(),
            );
        }

        let staging_offset = self.staging_offset;
        self.staging_offset += align_up(size, 64);

        let cmd = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.device.allocate_command_buffers(&info)?[0]
        };

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // Transition UNDEFINED → TRANSFER_DST_OPTIMAL.
            let barrier_to_transfer = vk::ImageMemoryBarrier::default()
                .image(dst_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier_to_transfer),
            );

            // Copy staging buffer → image.
            let region = vk::BufferImageCopy::default()
                .buffer_offset(staging_offset)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width, height, depth: 1 });

            self.device.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );

            // If dedicated transfer, release ownership.
            // The graphics queue must acquire and transition to
            // SHADER_READ_ONLY_OPTIMAL.
            if self.is_dedicated {
                let release = vk::ImageMemoryBarrier::default()
                    .image(dst_image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.graphics_family_index)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&release),
                );
            } else {
                // Same family: just transition directly.
                let barrier_to_shader = vk::ImageMemoryBarrier::default()
                    .image(dst_image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier_to_shader),
                );
            }

            self.device.end_command_buffer(cmd)?;
        }

        self.submit_timeline(cmd)
            .map(|timeline_value| TransferTicket {
                dst_buffer_handle: None,
                dst_image_handle: Some(dst_handle),
                timeline_value,
            })
    }

    /// Backward-compatible wrapper: upload buffer async.
    pub fn upload_async(
        &mut self,
        data: &[u8],
        dst_buffer: vk::Buffer,
        dst_handle: BufferHandle,
    ) -> Result<TransferTicket, Box<dyn std::error::Error>> {
        self.upload_buffer_async(data, dst_buffer, dst_handle)
    }

    /// Submit a recorded command buffer with timeline signal.
    /// Returns the timeline value.
    fn submit_timeline(
        &mut self,
        cmd: vk::CommandBuffer,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let timeline_value = self.next_timeline;
        self.next_timeline += 1;

        let binding = [timeline_value];
        let mut timeline_submit = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&binding);

        let signal_sems = [self.timeline_semaphore];
        let cmd_bufs = [cmd];

        let submit = vk::SubmitInfo::default()
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_sems)
            .push_next(&mut timeline_submit);

        unsafe {
            self.device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )?;
        }

        Ok(timeline_value)
    }

    /// Check whether a particular transfer has finished on the GPU.
    pub fn is_complete(&self, ticket: &TransferTicket) -> bool {
        let current = unsafe {
            self.device
                .get_semaphore_counter_value(self.timeline_semaphore)
                .unwrap_or(0)
        };
        current >= ticket.timeline_value
    }

    /// Block until a specific timeline value is reached.
    pub fn wait_for(&self, ticket: &TransferTicket, timeout_ns: u64) -> Result<(), vk::Result> {
        let binding = [ticket.timeline_value];
        let ts = [self.timeline_semaphore];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&ts)
            .values(&binding);
        unsafe { self.device.wait_semaphores(&wait_info, timeout_ns) }
    }

    /// The timeline semaphore, for the renderer to use as a wait
    /// semaphore on the graphics queue if needed.
    pub fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline_semaphore
    }

    /// Drain the command pool.
    pub fn reset_pool(&self) {
        unsafe {
            let _ = self.device.reset_command_pool(
                self.command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            );
        }
    }
}

impl Drop for TransferQueue {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_semaphore(self.timeline_semaphore, None);
            self.device.unmap_memory(self.staging_memory);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);
        }
    }
}

// ====================================================================
//  MemoryBudget – VRAM tracking + LRU eviction
// ====================================================================

/// Per-handle usage tracking for LRU eviction.
#[derive(Debug, Clone)]
struct UsageRecord {
    handle: BufferHandle,
    size: u64,
    last_used_frame: u64,
}

/// Tracks VRAM consumption against the device-local heap budget and
/// provides LRU eviction for streamable assets.
pub struct MemoryBudget {
    budget_bytes: u64,
    #[allow(dead_code)]
    device_local_heap: u32,
    usage: HashMap<BufferHandle, UsageRecord>,
}

impl MemoryBudget {
    pub fn new(
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        let (heap_idx, heap_size) = (0..memory_properties.memory_heap_count as usize)
            .find_map(|i| {
                let heap = memory_properties.memory_heaps[i];
                heap.flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                    .then_some((i as u32, heap.size))
            })
            .unwrap_or((0, 256 * 1024 * 1024));

        let budget_bytes = (heap_size as f64 * VRAM_BUDGET_RATIO) as u64;

        println!(
            "[MemoryBudget] Heap {} = {}, budget = {} ({:.0}%)",
            heap_idx,
            fmt_bytes(heap_size),
            fmt_bytes(budget_bytes),
            VRAM_BUDGET_RATIO * 100.0,
        );

        Self {
            budget_bytes,
            device_local_heap: heap_idx,
            usage: HashMap::new(),
        }
    }

    pub fn track(&mut self, handle: BufferHandle, size: u64, current_frame: u64) {
        self.usage.insert(
            handle,
            UsageRecord { handle, size, last_used_frame: current_frame },
        );
    }

    pub fn touch(&mut self, handle: BufferHandle, current_frame: u64) {
        if let Some(rec) = self.usage.get_mut(&handle) {
            rec.last_used_frame = current_frame;
        }
    }

    pub fn untrack(&mut self, handle: BufferHandle) {
        self.usage.remove(&handle);
    }

    pub fn is_over_budget(&self, current_pool_usage: u64) -> bool {
        current_pool_usage > self.budget_bytes
    }

    pub fn evict_lru(
        &mut self,
        current_pool_usage: u64,
        current_frame: u64,
        min_age_frames: u64,
    ) -> Vec<BufferHandle> {
        if !self.is_over_budget(current_pool_usage) {
            return Vec::new();
        }

        let mut candidates: Vec<&UsageRecord> = self
            .usage
            .values()
            .filter(|r| current_frame.saturating_sub(r.last_used_frame) >= min_age_frames)
            .collect();
        candidates.sort_by_key(|r| r.last_used_frame);

        let mut freed = 0u64;
        let excess = current_pool_usage.saturating_sub(self.budget_bytes);
        let mut evicted = Vec::new();

        for rec in candidates {
            if freed >= excess {
                break;
            }
            freed += rec.size;
            evicted.push(rec.handle);
        }

        for &h in &evicted {
            self.usage.remove(&h);
        }

        if !evicted.is_empty() {
            println!(
                "[MemoryBudget] Evicting {} handles, freeing ~{}",
                evicted.len(),
                fmt_bytes(freed),
            );
        }

        evicted
    }

    pub fn tracked_bytes(&self) -> u64 {
        self.usage.values().map(|r| r.size).sum()
    }
}

// ====================================================================
//  MemoryContext – public interface that owns everything
// ====================================================================

/// Owns all GPU memory resources: pool allocator, ring buffer, transfer
/// queue, memory budget, and the legacy staging belt for synchronous
/// init-time uploads.
pub struct MemoryContext {
    device: Device,
    pub allocator: GpuAllocator,
    pub ring: RingBuffer,
    pub transfer: TransferQueue,
    pub budget: MemoryBudget,

    // Legacy synchronous staging belt
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_mapped: NonNull<u8>,
    staging_size: u64,
}

unsafe impl Send for MemoryContext {}

impl MemoryContext {
    pub fn new(
        device: Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        min_ubo_alignment: u64,
        transfer_queue: vk::Queue,
        transfer_family: u32,
        graphics_family: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let allocator = GpuAllocator::new(device.clone(), memory_properties);
        let ring =
            RingBuffer::new(&device, &memory_properties, min_ubo_alignment.max(256))?;

        let transfer = TransferQueue::new(
            device.clone(),
            &memory_properties,
            transfer_queue,
            transfer_family,
            graphics_family,
        )?;

        let budget = MemoryBudget::new(&memory_properties);

        // ---- legacy reusable staging buffer ----
        let staging_info = vk::BufferCreateInfo::default()
            .size(STAGING_BUFFER_SIZE)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { device.create_buffer(&staging_info, None)? };
        let staging_req =
            unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let required = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let staging_type = (0..memory_properties.memory_type_count)
            .find(|&i| {
                (staging_req.memory_type_bits & (1 << i)) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or("No memory type for staging buffer")?;

        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_req.size)
            .memory_type_index(staging_type);
        let staging_memory = unsafe { device.allocate_memory(&staging_alloc, None)? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0)? }

        let raw = unsafe {
            device.map_memory(
                staging_memory,
                0,
                STAGING_BUFFER_SIZE,
                vk::MemoryMapFlags::empty(),
            )?
        };
        let staging_mapped =
            NonNull::new(raw as *mut u8).ok_or("Failed to map staging")?;

        println!(
            "[MemoryContext] Legacy staging belt: {} MB",
            STAGING_BUFFER_SIZE / (1024 * 1024),
        );

        Ok(Self {
            device,
            allocator,
            ring,
            transfer,
            budget,
            staging_buffer,
            staging_memory,
            staging_mapped,
            staging_size: STAGING_BUFFER_SIZE,
        })
    }

    // ----------------------------------------------------------------
    //  Async buffer upload (for streaming chunks)
    // ----------------------------------------------------------------

    pub fn upload_async(
        &mut self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<(BufferAllocation, TransferTicket), Box<dyn std::error::Error>> {
        let alloc = self.allocator.create_buffer(
            data.len() as u64,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        let ticket = self.transfer.upload_buffer_async(
            data,
            alloc.buffer,
            alloc.handle,
        )?;

        Ok((alloc, ticket))
    }

    pub fn upload_async_typed<T: Copy>(
        &mut self,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<(BufferAllocation, TransferTicket), Box<dyn std::error::Error>> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
            )
        };
        self.upload_async(bytes, usage)
    }

    // ----------------------------------------------------------------
    //  Async image upload (Phase 1: texture streaming)
    // ----------------------------------------------------------------

    /// Allocate a device-local VkImage and kick off an async transfer.
    /// Returns the `ImageAllocation` and a `TransferTicket`.
    /// After the ticket completes, the image is in
    /// `SHADER_READ_ONLY_OPTIMAL` (or `TRANSFER_DST_OPTIMAL` if
    /// dedicated transfer — caller must issue acquire barrier).
    pub fn upload_image_async(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(ImageAllocation, TransferTicket), Box<dyn std::error::Error>> {
        let alloc = self.allocator.create_image(
            width,
            height,
            format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            1, // mip_levels
            MemoryLocation::GpuOnly,
        )?;

        let ticket = self.transfer.upload_image_async(
            data,
            alloc.image,
            alloc.handle,
            width,
            height,
        )?;

        Ok((alloc, ticket))
    }

    // ----------------------------------------------------------------
    //  Synchronous upload path (for init-time data)
    // ----------------------------------------------------------------

    pub fn create_buffer_with_data(
        &mut self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
        let size = data.len() as u64;
        assert!(
            size <= self.staging_size,
            "Upload {} B exceeds staging belt {} B",
            size,
            self.staging_size,
        );

        let alloc = self.allocator.create_buffer(
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_mapped.as_ptr(),
                data.len(),
            );
        }

        unsafe {
            let cmd_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&cmd_info)?[0];

            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(
                cmd,
                self.staging_buffer,
                alloc.buffer,
                std::slice::from_ref(&region),
            );

            self.device.end_command_buffer(cmd)?;

            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.device
                .queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            self.device.queue_wait_idle(queue)?;

            self.device
                .free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        }

        Ok(alloc)
    }

    pub fn create_typed_buffer<T: Copy>(
        &mut self,
        data: &[T],
        usage: vk::BufferUsageFlags,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
            )
        };
        self.create_buffer_with_data(bytes, usage, command_pool, queue)
    }

    // ----------------------------------------------------------------
    //  Staging buffer accessors (for bulk uploads in gi.rs, etc.)
    // ----------------------------------------------------------------

    /// Raw staging buffer handle (for vkCmdCopyBufferToImage).
    pub fn staging_buffer(&self) -> vk::Buffer { self.staging_buffer }

    /// Staging buffer capacity in bytes.
    pub fn staging_size(&self) -> u64 { self.staging_size }

    /// Raw mapped pointer into the staging buffer.
    pub fn staging_ptr(&self) -> *mut u8 { self.staging_mapped.as_ptr() }

    /// Create a device-local VkImage and upload pixel data synchronously.
    /// Returns the image in `SHADER_READ_ONLY_OPTIMAL` layout.
    pub fn create_image_with_data(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<ImageAllocation, Box<dyn std::error::Error>> {
        let size = data.len() as u64;
        assert!(size <= self.staging_size);

        let alloc = self.allocator.create_image(
            width,
            height,
            format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            1,
            MemoryLocation::GpuOnly,
        )?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_mapped.as_ptr(),
                data.len(),
            );

            let cmd_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&cmd_info)?[0];

            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // UNDEFINED → TRANSFER_DST_OPTIMAL
            let barrier = vk::ImageMemoryBarrier::default()
                .image(alloc.image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width, height, depth: 1 });

            self.device.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer,
                alloc.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );

            // TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
            let barrier2 = vk::ImageMemoryBarrier::default()
                .image(alloc.image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier2),
            );

            self.device.end_command_buffer(cmd)?;

            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.device
                .queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            self.device.queue_wait_idle(queue)?;

            self.device
                .free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        }

        Ok(alloc)
    }
    
    /// Phase 4 (§4.6): Generate a full mip chain for a 2D image via a blit chain.
    ///
    /// Each mip level is blitted from the previous at half resolution.
    /// The image must be in `TRANSFER_DST_OPTIMAL` layout on entry.
    /// On exit, all mip levels are in `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// The image must have been created with `TRANSFER_SRC | TRANSFER_DST`
    /// usage flags. The format must support `FORMAT_FEATURE_BLIT_SRC`
    /// and `FORMAT_FEATURE_BLIT_DST` (all uncompressed color formats do).
    pub fn generate_mipmaps(
        &self,
        image: vk::Image,
        width: u32,
        height: u32,
        mip_levels: u32,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if mip_levels <= 1 {
            // Single-mip image — just transition to shader-read.
            unsafe {
                let cmd = self.begin_one_time_cmd(command_pool)?;
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[],
                    std::slice::from_ref(&barrier),
                );
                self.end_one_time_cmd(cmd, command_pool, queue)?;
            }
            return Ok(());
        }

        unsafe {
            let cmd = self.begin_one_time_cmd(command_pool)?;

            let mut mip_width = width as i32;
            let mut mip_height = height as i32;

            for level in 1..mip_levels {
                // Transition mip level (level-1) from TRANSFER_DST → TRANSFER_SRC
                // so it can be the blit source.
                let barrier_to_src = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[], &[],
                    std::slice::from_ref(&barrier_to_src),
                );

                // Blit from (level-1) to (level).
                let next_width = (mip_width / 2).max(1);
                let next_height = (mip_height / 2).max(1);

                let blit = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: mip_width, y: mip_height, z: 1 },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: next_width, y: next_height, z: 1 },
                    ],
                };

                self.device.cmd_blit_image(
                    cmd,
                    image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&blit),
                    vk::Filter::LINEAR,
                );

                // Transition mip level (level-1) from TRANSFER_SRC → SHADER_READ_ONLY
                let barrier_to_shader = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: level - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[],
                    std::slice::from_ref(&barrier_to_shader),
                );

                mip_width = next_width;
                mip_height = next_height;
            }

            // Transition the last mip level from TRANSFER_DST → SHADER_READ_ONLY.
            let final_barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_levels - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[],
                std::slice::from_ref(&final_barrier),
            );

            self.end_one_time_cmd(cmd, command_pool, queue)?;
        }

        Ok(())
    }

    /// Compute the number of mip levels for a 2D image.
    pub fn compute_mip_levels(width: u32, height: u32) -> u32 {
        ((width.max(height) as f32).log2().floor() as u32) + 1
    }

    /// Allocate a device-local image with mip chain, upload base mip,
    /// generate remaining mips via blit, and return in SHADER_READ_ONLY.
    ///
    /// The image is created with SAMPLED | TRANSFER_DST | TRANSFER_SRC usage.
    pub fn create_image_with_mipmaps(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<ImageAllocation, Box<dyn std::error::Error>> {
        let mip_levels = Self::compute_mip_levels(width, height);
        let size = data.len() as u64;
        assert!(size <= self.staging_size);

        let alloc = self.allocator.create_image(
            width, height, format,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            mip_levels,
            MemoryLocation::GpuOnly,
        )?;

        // Copy data into staging buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.staging_mapped.as_ptr(),
                data.len(),
            );
        }

        unsafe {
            let cmd = self.begin_one_time_cmd(command_pool)?;

            // Transition ALL mip levels to TRANSFER_DST_OPTIMAL.
            let barrier = vk::ImageMemoryBarrier::default()
                .image(alloc.image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[],
                std::slice::from_ref(&barrier),
            );

            // Copy staging → base mip (level 0).
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width, height, depth: 1 });
            self.device.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer,
                alloc.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );

            self.device.end_command_buffer(cmd)?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd));
            self.device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            self.device.queue_wait_idle(queue)?;
            self.device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        }

        // Now generate the mip chain (transitions all levels to SHADER_READ_ONLY).
        self.generate_mipmaps(alloc.image, width, height, mip_levels, command_pool, queue)?;

        Ok(alloc)
    }

    // ---- One-time command buffer helpers ----

    fn begin_one_time_cmd(
        &self,
        command_pool: vk::CommandPool,
    ) -> Result<vk::CommandBuffer, Box<dyn std::error::Error>> {
        unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&info)?[0];
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            Ok(cmd)
        }
    }

    fn end_one_time_cmd(
        &self,
        cmd: vk::CommandBuffer,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.end_command_buffer(cmd)?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd));
            self.device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            self.device.queue_wait_idle(queue)?;
            self.device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        }
        Ok(())
    }
}

impl Drop for MemoryContext {
    fn drop(&mut self) {
        unsafe {
            // Ring buffer.
            self.device.unmap_memory(self.ring.memory);
            self.device.destroy_buffer(self.ring.buffer, None);
            self.device.free_memory(self.ring.memory, None);

            // Legacy staging belt.
            self.device.unmap_memory(self.staging_memory);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);

            // TransferQueue::drop runs automatically.
            // GpuAllocator::drop runs after this method.
        }
    }
}