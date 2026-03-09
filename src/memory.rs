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

/// Staging-belt size for CPU → GPU transfers at init time.
const STAGING_BUFFER_SIZE: u64 = 32 * 1024 * 1024;

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

// ===== Handle =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

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
    ///
    /// Returns `(aligned_offset, optional_mapped_ptr)`.  Alignment padding
    /// between the region start and the aligned offset is kept as its own
    /// free region so no memory is permanently lost.
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

            //  ┌─padding─┬──size──┬─remaining─┐
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

struct AllocationRecord {
    buffer: vk::Buffer,
    memory_type_index: u32,
    block_index: usize,
    offset: u64,
    size: u64,
}

/// VMA-style GPU memory allocator.
///
/// Maintains a small number of large `VkDeviceMemory` blocks (64 MB each)
/// and sub-allocates individual `VkBuffer` regions within them.  This
/// keeps the total `vkAllocateMemory` call count under ~20 regardless of
/// how many buffers the engine creates, while giving full control over
/// data locality within each block.
pub struct GpuAllocator {
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pools: HashMap<u32, Vec<PoolBlock>>,
    allocations: HashMap<BufferHandle, AllocationRecord>,
    next_handle: u64,
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
            next_handle: 1,
        }
    }

    /// Create a `VkBuffer` backed by sub-allocated pool memory.
    pub fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<BufferAllocation, Box<dyn std::error::Error>> {
        // Create a temporary VkBuffer to query memory requirements.
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
                memory_type_index,
                block_index,
                offset,
                size: mem_req.size,
            },
        );

        Ok(BufferAllocation { handle, buffer, memory, offset, size: mem_req.size, mapped_ptr })
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
            "[GpuAllocator] {} blocks, {}/{} used, {} live buffers",
            n_blocks,
            fmt_bytes(total_used),
            fmt_bytes(total_cap),
            self.allocations.len(),
        );
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

        // Try existing blocks first.
        for (i, block) in blocks.iter_mut().enumerate() {
            if block.total_free >= size {
                if let Some((offset, mapped)) = block.alloc(size, alignment) {
                    return Ok((i, offset, mapped));
                }
            }
        }

        // Allocate a new block.
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

    fn find_memory_type(
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
            for (_, rec) in self.allocations.drain() {
                self.device.destroy_buffer(rec.buffer, None);
            }
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
///
/// The underlying `VkBuffer` is divided into `MAX_FRAMES_IN_FLIGHT` equal
/// segments.  Each frame bumps forward inside its own segment, so there
/// is zero contention between the CPU writing frame N and the GPU still
/// consuming frame N−1.
///
/// Combined with `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`, per-draw
/// uniform data is addressed by a dynamic offset at bind time – no
/// descriptor rewrites, no stalls, and the entire frame's uniform data
/// sits in a handful of contiguous cache lines.
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
    /// Byte offset from the start of the ring buffer's `VkBuffer`.
    pub offset: u64,
    /// CPU-mapped pointer for writing data.
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
                    | vk::BufferUsageFlags::VERTEX_BUFFER,
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

    /// Reset the write cursor for a new frame.  Call **after** the fence
    /// for `frame_index` has been waited on.
    pub fn begin_frame(&mut self, frame_index: usize) {
        self.current_frame = frame_index;
        self.frame_offset = 0;
    }

    /// Reserve `size` bytes in the current frame's segment.
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

    /// Push a `Copy` value and write it into the ring.
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
}

// ====================================================================
//  MemoryContext – public interface that owns everything
// ====================================================================

/// Owns all GPU memory resources: pool allocator, ring buffer, and the
/// reusable staging belt.  Handed to the renderer at construction.
pub struct MemoryContext {
    device: Device,
    pub allocator: GpuAllocator,
    pub ring: RingBuffer,
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
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let allocator = GpuAllocator::new(device.clone(), memory_properties);
        let ring =
            RingBuffer::new(&device, &memory_properties, min_ubo_alignment.max(256))?;

        // ---- reusable staging buffer ----
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
            "[MemoryContext] Staging belt: {} MB",
            STAGING_BUFFER_SIZE / (1024 * 1024),
        );

        Ok(Self {
            device,
            allocator,
            ring,
            staging_buffer,
            staging_memory,
            staging_mapped,
            staging_size: STAGING_BUFFER_SIZE,
        })
    }

    /// Create a device-local buffer and upload `data` synchronously via
    /// the staging belt.
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

    /// Create a device-local buffer from a typed slice.
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
}

impl Drop for MemoryContext {
    fn drop(&mut self) {
        unsafe {
            // Ring buffer.
            self.device.unmap_memory(self.ring.memory);
            self.device.destroy_buffer(self.ring.buffer, None);
            self.device.free_memory(self.ring.memory, None);

            // Staging belt.
            self.device.unmap_memory(self.staging_memory);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);

            // GpuAllocator::drop runs after this method, destroying all
            // remaining VkBuffers and freeing every pool VkDeviceMemory.
        }
    }
}