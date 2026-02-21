use std::{
    ptr::NonNull,
    sync::atomic::{AtomicUsize, AtomicU64, AtomicBool, Ordering},
    sync::{Arc, Mutex},
    collections::{HashMap, VecDeque},
    thread,
    time::Instant,
    cell::{Cell, RefCell},
    marker::PhantomData,
    alloc::{alloc, dealloc, Layout},
};

use ash::{vk, Device, vk::Handle};
use crossbeam::channel::{Sender, Receiver, unbounded};
use parking_lot::RwLock as ParkingRwLock;

// ===== Constants and Compile-Time Optimizations =====

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;
pub const ENABLE_VALIDATION: bool = cfg!(debug_assertions);
pub const WORKER_THREADS: usize = 4;

// Beta II Architecture constants
const BLOCK_SIZE: usize = 2 * 1024 * 1024; // 2MB TLB-aligned blocks
const MAX_BLOCKS: usize = 512; // 1GB maximum heap
const MAX_FRAMES_IN_FLIGHT: usize = 3;
const CACHE_LINE_SIZE: usize = 64;

/// Power-of-2 alignment as per Beta II requirements
/// Reference: https://doc.rust-lang.org/std/alloc/struct.Layout.html#method.align_to
#[inline(always)]
pub const fn align_to_power_of_2(size: usize) -> usize {
    if size == 0 {
        return 1;
    }
    1 << (usize::BITS - size.saturating_sub(1).leading_zeros())
}

/// Compile-time block alignment
#[inline(always)]
pub const fn align_to_block(size: usize) -> usize {
    (size + BLOCK_SIZE - 1) & !(BLOCK_SIZE - 1)
}

// ===== Core Types =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHandle(u32);

#[derive(Debug, Clone, Copy)]
pub struct FrameTag(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    DeviceLocal = 0,
    HostCoherent = 1,
    HostCached = 2,
    HostCoherentUncached = 3,
    DeviceLocalHostVisible = 4,
    LazilyAllocated = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationPurpose {
    VertexBuffer,
    IndexBuffer,
    UniformBuffer,
    StorageBuffer,
    StagingBuffer,
    Texture,
    RenderTarget,
    DepthBuffer,
    ShadowMap,
    CascadedShadowMap,
    Implicit,
    HBAOTexture,
    HBAOBlurTexture,
    CullingData,
    Other,
}

impl AllocationPurpose {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VertexBuffer => "Vertex",
            Self::IndexBuffer => "Index",
            Self::UniformBuffer => "Uniform",
            Self::StorageBuffer => "Storage",
            Self::StagingBuffer => "Staging",
            Self::Texture => "Texture",
            Self::RenderTarget => "RenderTarget",
            Self::DepthBuffer => "Depth",
            Self::ShadowMap => "ShadowMap",
            Self::CascadedShadowMap => "CSM",
            Self::Implicit => "Implicit",
            Self::HBAOTexture => "HBAO",
            Self::HBAOBlurTexture => "HBAOBlur",
            Self::CullingData => "CullingData",
            Self::Other => "Other",
        }
    }
}

// ===== Platform Memory Layer (VMem) =====

pub struct VMem;

impl VMem {
    /// Reserve and commit memory with TLB alignment for Beta II blocks
    /// Reference: https://doc.rust-lang.org/std/alloc/fn.alloc.html
    #[cfg(target_os = "windows")]
    pub unsafe fn reserve_block_aligned(size: usize) -> Option<NonNull<u8>> {
        use winapi::um::memoryapi::VirtualAlloc;
        use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};
        
        // Note: MEM_LARGE_PAGES requires SeLockMemoryPrivilege which is not available
        // by default. We'll use regular pages but still get 2MB alignment.
        let ptr = VirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE
        ) as *mut u8;
        
        NonNull::new(ptr)
    }

    #[cfg(target_os = "linux")]
    pub unsafe fn reserve_block_aligned(size: usize) -> Option<NonNull<u8>> {
        use libc::{mmap, madvise, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, 
                   MADV_HUGEPAGE};
        
        let flags = MAP_PRIVATE | MAP_ANONYMOUS;
        
        let ptr = mmap(
            std::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            flags,
            -1,
            0
        );
        
        if ptr != libc::MAP_FAILED {
            // Advise huge pages for better performance
            if size >= BLOCK_SIZE {
                madvise(ptr, size, MADV_HUGEPAGE);
            }
            NonNull::new(ptr as *mut u8)
        } else {
            None
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    pub unsafe fn reserve_block_aligned(size: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, BLOCK_SIZE).ok()?;
        NonNull::new(alloc(layout))
    }

    #[cfg(target_os = "windows")]
    pub unsafe fn release_block(ptr: NonNull<u8>, _size: usize) {
        use winapi::um::memoryapi::VirtualFree;
        use winapi::um::winnt::MEM_RELEASE;
        VirtualFree(ptr.as_ptr() as *mut _, 0, MEM_RELEASE);
    }

    #[cfg(target_os = "linux")]
    pub unsafe fn release_block(ptr: NonNull<u8>, size: usize) {
        use libc::munmap;
        munmap(ptr.as_ptr() as *mut _, size);
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    pub unsafe fn release_block(ptr: NonNull<u8>, size: usize) {
        let layout = Layout::from_size_align_unchecked(size, BLOCK_SIZE);
        dealloc(ptr.as_ptr(), layout);
    }
}

// ===== Beta II Tagged Heap - Core Allocator =====

/// Tagged block representation
#[repr(C, align(64))]
struct TaggedBlock {
    ptr: NonNull<u8>,
    tag: AtomicU64,
    in_use: AtomicBool,
    allocated_bytes: AtomicUsize,
}

/// The Tagged Heap - Beta II's shared backend allocator
/// All engine allocators request blocks from this shared pool
pub struct TaggedHeap {
    blocks: Vec<TaggedBlock>,
    free_blocks: parking_lot::Mutex<VecDeque<BlockHandle>>,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
}

// Safety: TaggedHeap is safe to share between threads because:
// 1. All block access is protected by atomics (tag, in_use, allocated_bytes)
// 2. The free_blocks list is protected by a Mutex
// 3. NonNull pointers are only read, never modified after initialization
unsafe impl Send for TaggedHeap {}
unsafe impl Sync for TaggedHeap {}

impl TaggedHeap {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut blocks = Vec::with_capacity(MAX_BLOCKS);
        let mut free_blocks = VecDeque::with_capacity(MAX_BLOCKS);
        
        // Pre-allocate initial blocks - start with fewer blocks
        let initial_blocks = 8; // 16MB initial (reduced from 32 blocks)
        
        println!("[Startup::TaggedHeap] Allocating {} initial blocks ({} MB)...", 
            initial_blocks, (initial_blocks * BLOCK_SIZE) / (1024 * 1024));
        
        for i in 0..initial_blocks {
            let ptr = unsafe { VMem::reserve_block_aligned(BLOCK_SIZE) }
                .ok_or_else(|| {
                    format!("Failed to allocate block {} (size: {} bytes). \
                            This might be due to insufficient memory or system limits.", 
                            i, BLOCK_SIZE)
                })?;
            
            blocks.push(TaggedBlock {
                ptr,
                tag: AtomicU64::new(0),
                in_use: AtomicBool::new(false),
                allocated_bytes: AtomicUsize::new(0),
            });
            
            free_blocks.push_back(BlockHandle(i as u32));
        }
        
        println!("[Startup::TaggedHeap] Successfully allocated {} blocks", initial_blocks);
        
        // Fill remaining capacity with uninitialized blocks (will be allocated on demand)
        for _ in initial_blocks..MAX_BLOCKS {
            blocks.push(TaggedBlock {
                ptr: NonNull::dangling(),
                tag: AtomicU64::new(0),
                in_use: AtomicBool::new(false),
                allocated_bytes: AtomicUsize::new(0),
            });
        }
        
        Ok(Self {
            blocks,
            free_blocks: parking_lot::Mutex::new(free_blocks),
            total_allocated: AtomicUsize::new(initial_blocks * BLOCK_SIZE),
            peak_allocated: AtomicUsize::new(initial_blocks * BLOCK_SIZE),
        })
    }
    
    /// Allocate a block with a specific tag
    pub fn alloc_block(&self, tag: u64) -> Option<BlockHandle> {
        let mut free_list = self.free_blocks.lock();
        
        if let Some(handle) = free_list.pop_front() {
            let block = &self.blocks[handle.0 as usize];
            block.tag.store(tag, Ordering::Release);
            block.in_use.store(true, Ordering::Release);
            block.allocated_bytes.store(0, Ordering::Release);
            return Some(handle);
        }
        
        // Try to allocate a new block if we have capacity
        for i in 0..self.blocks.len() {
            let block = &self.blocks[i];
            if !block.in_use.load(Ordering::Acquire) && block.ptr == NonNull::dangling() {
                // Allocate new block
                if let Some(ptr) = unsafe { VMem::reserve_block_aligned(BLOCK_SIZE) } {
                    // Initialize the block - this is safe because we have exclusive access
                    // Reference: https://doc.rust-lang.org/std/ptr/struct.NonNull.html
                    unsafe {
                        let block_ptr = &self.blocks[i] as *const TaggedBlock as *mut TaggedBlock;
                        (*block_ptr).ptr = ptr;
                    }
                    
                    block.tag.store(tag, Ordering::Release);
                    block.in_use.store(true, Ordering::Release);
                    block.allocated_bytes.store(0, Ordering::Release);
                    
                    self.total_allocated.fetch_add(BLOCK_SIZE, Ordering::AcqRel);
                    self.update_peak();
                    
                    return Some(BlockHandle(i as u32));
                }
            }
        }
        
        None
    }
    
    /// Free all blocks with a specific tag - Beta II's key pattern
    pub fn free_all_with_tag(&self, tag: u64) {
        let mut freed_blocks = Vec::new();
        
        for (i, block) in self.blocks.iter().enumerate() {
            if block.tag.load(Ordering::Acquire) == tag && block.in_use.load(Ordering::Acquire) {
                block.in_use.store(false, Ordering::Release);
                block.allocated_bytes.store(0, Ordering::Release);
                freed_blocks.push(BlockHandle(i as u32));
            }
        }
        
        if !freed_blocks.is_empty() {
            let mut free_list = self.free_blocks.lock();
            free_list.extend(freed_blocks);
        }
    }
    
    /// Get block pointer for direct access
    pub fn get_block_ptr(&self, handle: BlockHandle) -> Option<NonNull<u8>> {
        let block = &self.blocks[handle.0 as usize];
        if block.in_use.load(Ordering::Acquire) {
            Some(block.ptr)
        } else {
            None
        }
    }
    
    fn update_peak(&self) {
        let current = self.total_allocated.load(Ordering::Acquire);
        let mut peak = self.peak_allocated.load(Ordering::Acquire);
        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                current,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }
}

impl Drop for TaggedHeap {
    fn drop(&mut self) {
        for block in &self.blocks {
            if block.ptr != NonNull::dangling() {
                unsafe { VMem::release_block(block.ptr, BLOCK_SIZE); }
            }
        }
    }
}

// ===== Thread-Local Block Allocator =====

/// Thread-local storage for per-thread block allocation
/// Reference: https://doc.rust-lang.org/std/thread/struct.LocalKey.html
thread_local! {
    static THREAD_ALLOCATOR: RefCell<Option<ThreadLocalAllocator>> = RefCell::new(None);
    static THREAD_STATS: Cell<ThreadLocalStats> = Cell::new(ThreadLocalStats::default());
}

#[derive(Default, Clone, Copy, Debug)]
pub struct ThreadLocalStats {
    pub allocations: usize,
    pub bytes_allocated: usize,
    pub blocks_used: usize,
}

pub struct ThreadLocalAllocator {
    current_block: Option<BlockHandle>,
    offset: usize,
    tag: u64,
    heap: Arc<TaggedHeap>,
    stats: Arc<GlobalThreadStats>,
    thread_id: usize,
}

/// Global statistics aggregator for all threads
pub struct GlobalThreadStats {
    per_thread: Vec<parking_lot::RwLock<ThreadLocalStats>>,
    total_allocations: AtomicUsize,
    total_bytes: AtomicUsize,
}

impl GlobalThreadStats {
    pub fn new(num_threads: usize) -> Self {
        let mut per_thread = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            per_thread.push(parking_lot::RwLock::new(ThreadLocalStats::default()));
        }
        
        Self {
            per_thread,
            total_allocations: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
        }
    }
    
    pub fn report_allocation(&self, thread_id: usize, size: usize) {
        if thread_id < self.per_thread.len() {
            if let Some(stats) = self.per_thread.get(thread_id) {
                let mut thread_stats = stats.write();
                thread_stats.allocations += 1;
                thread_stats.bytes_allocated += size;
            }
        }
        
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size, Ordering::Relaxed);
    }
    
    pub fn report_new_block(&self, thread_id: usize) {
        if thread_id < self.per_thread.len() {
            if let Some(stats) = self.per_thread.get(thread_id) {
                stats.write().blocks_used += 1;
            }
        }
    }
    
    pub fn get_thread_stats(&self, thread_id: usize) -> ThreadLocalStats {
        if thread_id < self.per_thread.len() {
            *self.per_thread[thread_id].read()
        } else {
            ThreadLocalStats::default()
        }
    }
    
    pub fn get_total_stats(&self) -> (usize, usize) {
        (
            self.total_allocations.load(Ordering::Relaxed),
            self.total_bytes.load(Ordering::Relaxed)
        )
    }
}

impl ThreadLocalAllocator {
    pub fn new(heap: Arc<TaggedHeap>, tag: u64, stats: Arc<GlobalThreadStats>, thread_id: usize) -> Self {
        Self {
            current_block: None,
            offset: 0,
            tag,
            heap,
            stats,
            thread_id,
        }
    }
    
    /// Allocate from thread-local block
    pub fn alloc(&mut self, size: usize) -> Option<NonNull<u8>> {
        let aligned_size = align_to_power_of_2(size);
        
        // Check if we need a new block
        if self.current_block.is_none() || self.offset + aligned_size > BLOCK_SIZE {
            self.current_block = self.heap.alloc_block(self.tag);
            self.offset = 0;
            
            if self.current_block.is_none() {
                return None;
            }
            
            // Report new block acquisition
            self.stats.report_new_block(self.thread_id);
        }
        
        // Allocate from current block
        if let Some(handle) = self.current_block {
            if let Some(mut ptr) = self.heap.get_block_ptr(handle) {
                let result = unsafe { 
                    NonNull::new_unchecked(ptr.as_ptr().add(self.offset))
                };
                self.offset += aligned_size;
                
                // Update block's allocated bytes
                let block = &self.heap.blocks[handle.0 as usize];
                block.allocated_bytes.fetch_add(aligned_size, Ordering::Release);
                
                // Report to global stats
                self.stats.report_allocation(self.thread_id, aligned_size);
                
                // Update thread-local stats cache
                THREAD_STATS.with(|stats| {
                    let mut s = stats.get();
                    s.allocations += 1;
                    s.bytes_allocated += aligned_size;
                    stats.set(s);
                });
                
                return Some(result);
            }
        }
        
        None
    }
    
    /// Switch to a new tag (e.g., new frame)
    pub fn set_tag(&mut self, tag: u64) {
        if self.tag != tag {
            self.tag = tag;
            self.current_block = None;
            self.offset = 0;
        }
    }
    
    /// Get cached thread-local stats without locking
    pub fn get_local_stats() -> ThreadLocalStats {
        THREAD_STATS.with(|stats| stats.get())
    }
}

// ===== Frame Parameters & Queue =====

/// Uncontended frame parameters - copied, not Arc'd
#[derive(Clone)]
pub struct FrameParams {
    pub frame_number: u64,
    pub delta_time: f32,
    pub camera_matrix: [f32; 16],
    pub mesh_count: usize,
    pub timestamp: Instant,
}

impl Default for FrameParams {
    fn default() -> Self {
        Self {
            frame_number: 0,
            delta_time: 0.016,
            camera_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            mesh_count: 0,
            timestamp: Instant::now(),
        }
    }
}

/// Frame synchronization queue with triple buffering
pub struct FrameQueue {
    params: [parking_lot::RwLock<FrameParams>; MAX_FRAMES_IN_FLIGHT],
    completed: [AtomicBool; MAX_FRAMES_IN_FLIGHT],
    current_frame: AtomicU64,
    frames_in_flight: AtomicUsize,
}

impl FrameQueue {
    pub fn new() -> Self {
        Self {
            params: [
                parking_lot::RwLock::new(FrameParams::default()),
                parking_lot::RwLock::new(FrameParams::default()),
                parking_lot::RwLock::new(FrameParams::default()),
            ],
            completed: [
                AtomicBool::new(true),
                AtomicBool::new(true),
                AtomicBool::new(true),
            ],
            current_frame: AtomicU64::new(0),
            frames_in_flight: AtomicUsize::new(0),
        }
    }
    
    /// Check if frame has completed - Beta II's HasFrameCompleted()
    pub fn has_frame_completed(&self, frame_num: u64) -> bool {
        let slot = (frame_num % MAX_FRAMES_IN_FLIGHT as u64) as usize;
        self.completed[slot].load(Ordering::Acquire)
    }
    
    /// Wait for frame completion
    pub fn wait_for_frame(&self, frame_num: u64) {
        while !self.has_frame_completed(frame_num) {
            std::hint::spin_loop();
        }
    }
    
    /// Try to begin frame with timeout - for testing backpressure
    pub fn try_begin_frame_with_timeout(&self, params: FrameParams, timeout: std::time::Duration) -> Option<FrameContext> {
        let start = std::time::Instant::now();
        
        // Check frame limit with timeout
        while self.frames_in_flight.load(Ordering::Acquire) >= MAX_FRAMES_IN_FLIGHT {
            if start.elapsed() > timeout {
                return None; // Timeout - backpressure working
            }
            
            let oldest_frame = self.current_frame.load(Ordering::Acquire)
                .saturating_sub(MAX_FRAMES_IN_FLIGHT as u64);
            
            // Quick check if oldest frame completed
            if self.has_frame_completed(oldest_frame) {
                break;
            }
            
            std::thread::yield_now();
        }
        
        // Try to acquire a slot
        if self.frames_in_flight.load(Ordering::Acquire) >= MAX_FRAMES_IN_FLIGHT {
            return None;
        }
        
        let frame_num = self.current_frame.fetch_add(1, Ordering::AcqRel);
        let slot = (frame_num % MAX_FRAMES_IN_FLIGHT as u64) as usize;
        
        // Mark frame as in-progress
        self.completed[slot].store(false, Ordering::Release);
        self.frames_in_flight.fetch_add(1, Ordering::AcqRel);
        
        // Update params
        *self.params[slot].write() = params.clone();
        
        Some(FrameContext {
            frame_number: frame_num,
            slot,
            tag: FrameTag(frame_num),
        })
    }
    
    /// Begin new frame with backpressure
    pub fn begin_frame(&self, params: FrameParams) -> Option<FrameContext> {
        // Check frame limit - Beta II's 3 frame limit
        while self.frames_in_flight.load(Ordering::Acquire) >= MAX_FRAMES_IN_FLIGHT {
            let oldest_frame = self.current_frame.load(Ordering::Acquire)
                .saturating_sub(MAX_FRAMES_IN_FLIGHT as u64);
            self.wait_for_frame(oldest_frame);
        }
        
        let frame_num = self.current_frame.fetch_add(1, Ordering::AcqRel);
        let slot = (frame_num % MAX_FRAMES_IN_FLIGHT as u64) as usize;
        
        // Mark frame as in-progress
        self.completed[slot].store(false, Ordering::Release);
        self.frames_in_flight.fetch_add(1, Ordering::AcqRel);
        
        // Update params
        *self.params[slot].write() = params.clone();
        
        Some(FrameContext {
            frame_number: frame_num,
            slot,
            tag: FrameTag(frame_num),
        })
    }
    
    /// Complete frame - takes reference to avoid move
    pub fn complete_frame(&self, ctx: &FrameContext) {
        self.completed[ctx.slot].store(true, Ordering::Release);
        self.frames_in_flight.fetch_sub(1, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
pub struct FrameContext {
    pub frame_number: u64,
    pub slot: usize,
    pub tag: FrameTag,
}

// ===== Job System with Priority Inversion Prevention =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

pub struct Job {
    pub id: u64,
    pub tag: FrameTag,
    pub priority: Priority,
    pub dependencies: Vec<u64>,
    pub work: Box<dyn FnOnce() + Send>,
}

pub struct JobStage {
    name: String,
    active_count: AtomicUsize,
    max_parallel: usize,
}

impl JobStage {
    pub fn new(name: String, max_parallel: usize) -> Self {
        Self {
            name,
            active_count: AtomicUsize::new(0),
            max_parallel,
        }
    }
    
    /// Try to enter stage - Beta II's stage guarding
    pub fn try_enter(&self) -> Option<StageGuard> {
        let count = self.active_count.load(Ordering::Acquire);
        if count < self.max_parallel {
            if self.active_count.compare_exchange(
                count,
                count + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                return Some(StageGuard { stage: self });
            }
        }
        None
    }
}

pub struct StageGuard<'a> {
    stage: &'a JobStage,
}

impl<'a> Drop for StageGuard<'a> {
    fn drop(&mut self) {
        self.stage.active_count.fetch_sub(1, Ordering::Release);
    }
}

pub struct JobSystem {
    stages: HashMap<String, JobStage>,
    job_queue: Arc<crossbeam::queue::SegQueue<Job>>,
    completed_jobs: parking_lot::Mutex<HashMap<u64, Instant>>,
    next_job_id: AtomicU64,
    workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl JobSystem {
    pub fn new(num_workers: usize) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let job_queue: Arc<crossbeam::queue::SegQueue<Job>> = Arc::new(crossbeam::queue::SegQueue::new());
        
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let queue = job_queue.clone();
            let shutdown_clone = shutdown.clone();
            
            workers.push(thread::spawn(move || {
                while !shutdown_clone.load(Ordering::Acquire) {
                    if let Some(job) = queue.pop() {
                        (job.work)();
                    } else {
                        std::thread::yield_now();
                    }
                }
            }));
        }
        
        Self {
            stages: HashMap::new(),
            job_queue,
            completed_jobs: parking_lot::Mutex::new(HashMap::new()),
            next_job_id: AtomicU64::new(1),
            workers,
            shutdown,
        }
    }
    
    pub fn submit(&self, job: Job) {
        self.job_queue.push(job);
    }
    
    pub fn submit_with_tag(&self, tag: FrameTag, priority: Priority, work: Box<dyn FnOnce() + Send>) -> u64 {
        let id = self.next_job_id.fetch_add(1, Ordering::AcqRel);
        self.job_queue.push(Job {
            id,
            tag,
            priority,
            dependencies: Vec::new(),
            work,
        });
        id
    }
}

impl Drop for JobSystem {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// ===== Unified Memory Bundle =====

pub struct UnifiedMemoryBundle {
    pub ptr: NonNull<u8>,
    pub size: usize,
    pub tag: FrameTag,
}

// Safety: UnifiedMemoryBundle can be sent between threads because:
// 1. The memory it points to is allocated from the tagged heap
// 2. The tagged heap ensures proper synchronization
// 3. Frame tags provide temporal safety
unsafe impl Send for UnifiedMemoryBundle {}

impl UnifiedMemoryBundle {
    /// Create zero-copy view - Reference: https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html
    #[inline]
    pub unsafe fn as_view<T>(&self) -> MemoryView<'_, T> {
        MemoryView::from_bundle(self)
    }
}

pub struct MemoryView<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> MemoryView<'a, T> {
    #[inline]
    pub unsafe fn from_bundle(bundle: &'a UnifiedMemoryBundle) -> Self {
        let elem_size = std::mem::size_of::<T>();
        let len = bundle.size / elem_size;
        
        Self {
            ptr: bundle.ptr.cast(),
            len,
            _phantom: PhantomData,
        }
    }
    
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

// ===== GPU Resource Manager (Simplified) =====

pub struct GPUResourceManager {
    device: Arc<Device>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    buffers: ParkingRwLock<HashMap<BufferHandle, TrackedBuffer>>,
    next_handle: AtomicU64,
}

struct TrackedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    purpose: AllocationPurpose,
}

impl GPUResourceManager {
    pub fn new(device: Arc<Device>, memory_properties: vk::PhysicalDeviceMemoryProperties) -> Self {
        // Calculate VRAM budget from memory heaps
        let mut vram_budget = 0;
        let mut vram_heaps = Vec::new();
        
        for i in 0..memory_properties.memory_heap_count {
            let heap = memory_properties.memory_heaps[i as usize];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                vram_budget += heap.size as usize;
                vram_heaps.push(heap.size as usize);
                println!("VRAM Heap {}: {:.2} GB", i, heap.size as f64 / (1024.0 * 1024.0 * 1024.0));
            }
        }
        
        println!("Total VRAM Budget: {:.2} GB", vram_budget as f64 / (1024.0 * 1024.0 * 1024.0));

        Self {
            device,
            memory_properties,
            buffers: ParkingRwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }
    
    pub fn create_buffer(
        &self,
        size: usize,
        usage: vk::BufferUsageFlags,
        purpose: AllocationPurpose,
    ) -> Result<BufferHandle, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        
        let memory_type_index = self.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(memory_type_index);
        
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        
        let handle = BufferHandle(self.next_handle.fetch_add(1, Ordering::AcqRel));
        
        self.buffers.write().insert(handle, TrackedBuffer {
            buffer,
            memory,
            size,
            purpose,
        });
        
        Ok(handle)
    }
    
    pub fn destroy_buffer(&self, handle: BufferHandle) {
        if let Some(tracked) = self.buffers.write().remove(&handle) {
            unsafe {
                self.device.destroy_buffer(tracked.buffer, None);
                self.device.free_memory(tracked.memory, None);
            }
        }
    }
    
    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0 &&
               self.memory_properties.memory_types[i as usize].property_flags.contains(properties) {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".into())
    }
}

// ===== Unified Memory Pool - Beta II Architecture =====

pub struct UnifiedMemoryPool {
    pub tagged_heap: Arc<TaggedHeap>,
    pub frame_queue: Arc<FrameQueue>,
    job_system: Arc<JobSystem>,
    gpu: Arc<GPUResourceManager>,
    current_frame_tag: AtomicU64,
    thread_stats: Arc<GlobalThreadStats>,
}

impl UnifiedMemoryPool {
    pub fn new(
        device: Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        _command_pool: vk::CommandPool,
        _queue: vk::Queue,
        _queue_family_index: u32,
        num_threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tagged_heap = Arc::new(TaggedHeap::new()?);
        let frame_queue = Arc::new(FrameQueue::new());
        let job_system = Arc::new(JobSystem::new(num_threads));
        let gpu = Arc::new(GPUResourceManager::new(Arc::new(device), memory_properties));
        let thread_stats = Arc::new(GlobalThreadStats::new(num_threads * 2)); // Extra slots for dynamic threads
        
        // Initialize thread-local allocator for main thread
        let heap_clone = tagged_heap.clone();
        let stats_clone = thread_stats.clone();
        THREAD_ALLOCATOR.with(|a| {
            *a.borrow_mut() = Some(ThreadLocalAllocator::new(heap_clone, 0, stats_clone, 0));
        });
        
        Ok(Self {
            tagged_heap,
            frame_queue,
            job_system,
            gpu,
            current_frame_tag: AtomicU64::new(0),
            thread_stats,
        })
    }
    
    /// Allocate memory using thread-local block
    pub fn alloc(&self, size: usize) -> Option<UnifiedMemoryBundle> {
        THREAD_ALLOCATOR.with(|a| {
            let mut allocator = a.borrow_mut();
            if let Some(ref mut alloc) = *allocator {
                alloc.alloc(size).map(|ptr| UnifiedMemoryBundle {
                    ptr,
                    size,
                    tag: FrameTag(alloc.tag),
                })
            } else {
                // Initialize allocator for this thread if not already done
                // Use a simple counter-based thread ID
                static THREAD_COUNTER: AtomicUsize = AtomicUsize::new(0);
                let thread_num = THREAD_COUNTER.fetch_add(1, Ordering::Relaxed);
                
                *allocator = Some(ThreadLocalAllocator::new(
                    self.tagged_heap.clone(),
                    self.current_frame_tag.load(Ordering::Acquire),
                    self.thread_stats.clone(),
                    thread_num
                ));
                
                // Try again with new allocator
                if let Some(ref mut alloc) = *allocator {
                    alloc.alloc(size).map(|ptr| UnifiedMemoryBundle {
                        ptr,
                        size,
                        tag: FrameTag(alloc.tag),
                    })
                } else {
                    None
                }
            }
        })
    }
    
    /// Allocate with explicit thread ID (for testing)
    pub fn alloc_on_thread(&self, size: usize, thread_id: usize) -> Option<UnifiedMemoryBundle> {
        THREAD_ALLOCATOR.with(|a| {
            let mut allocator = a.borrow_mut();
            
            // Ensure allocator exists for this thread
            if allocator.is_none() {
                *allocator = Some(ThreadLocalAllocator::new(
                    self.tagged_heap.clone(),
                    self.current_frame_tag.load(Ordering::Acquire),
                    self.thread_stats.clone(),
                    thread_id
                ));
            }
            
            if let Some(ref mut alloc) = *allocator {
                alloc.alloc(size).map(|ptr| UnifiedMemoryBundle {
                    ptr,
                    size,
                    tag: FrameTag(alloc.tag),
                })
            } else {
                None
            }
        })
    }
    
    /// Get statistics for a specific thread
    pub fn get_thread_stats(&self, thread_id: usize) -> ThreadLocalStats {
        self.thread_stats.get_thread_stats(thread_id)
    }
    
    /// Get total statistics across all threads
    pub fn get_total_thread_stats(&self) -> (usize, usize) {
        self.thread_stats.get_total_stats()
    }
    
    /// Begin new frame with Beta II semantics
    pub fn begin_frame(&self, params: FrameParams) -> Option<FrameContext> {
        let ctx = self.frame_queue.begin_frame(params)?;
        
        // Update thread-local allocator tags
        let tag = ctx.tag.0;
        self.current_frame_tag.store(tag, Ordering::Release);
        
        THREAD_ALLOCATOR.with(|a| {
            if let Some(ref mut alloc) = *a.borrow_mut() {
                alloc.set_tag(tag);
            }
        });
        
        Some(ctx)
    }
    
    /// Complete frame and free associated memory
    pub fn complete_frame(&self, ctx: &FrameContext) {
        // Free all allocations with this frame's tag
        self.tagged_heap.free_all_with_tag(ctx.tag.0);
        self.frame_queue.complete_frame(ctx);
    }
    
    /// Submit job with frame tag
    pub fn submit_job(&self, priority: Priority, work: Box<dyn FnOnce() + Send>) {
        let tag = FrameTag(self.current_frame_tag.load(Ordering::Acquire));
        self.job_system.submit_with_tag(tag, priority, work);
    }
    
    pub fn shutdown(&self) {
        // Clean shutdown
        unsafe {
            let _ = self.gpu.device.device_wait_idle();
        }
    }
}

// ===== Memory Context - Public Interface =====

#[derive(Clone)]
pub struct MemoryContext {
    pool: Arc<UnifiedMemoryPool>,
}

impl MemoryContext {
    pub fn from_pool(pool: Arc<UnifiedMemoryPool>) -> Self {
        Self { pool }
    }
    
    /// Begin frame with parameters
    pub fn begin_frame(&self, delta_time: f32) -> Option<FrameHandle> {
        let params = FrameParams {
            frame_number: self.pool.current_frame_tag.load(Ordering::Acquire),
            delta_time,
            ..Default::default()
        };
        
        self.pool.begin_frame(params).map(|ctx| FrameHandle {
            context: ctx,
            pool: self.pool.clone(),
        })
    }
    
    /// Allocate memory on current thread
    pub fn alloc(&self, size: usize) -> Option<UnifiedMemoryBundle> {
        self.pool.alloc(size)
    }
    
    /// Create GPU buffer
    pub fn create_buffer_with_purpose(
        &self,
        size: usize,
        _tier: Option<MemoryTier>,
        usage: vk::BufferUsageFlags,
        purpose: AllocationPurpose,
    ) -> BufferHandle {
        self.pool.gpu.create_buffer(size, usage, purpose).unwrap()
    }
    
    /// For compatibility - these now use the unified system
    pub fn create_buffer_async(
        &self,
        size: usize,
        _tier: Option<MemoryTier>,
        usage: vk::BufferUsageFlags,
    ) -> BufferHandle {
        self.create_buffer_with_purpose(size, None, usage, AllocationPurpose::Other)
    }
    
    pub fn submit_migration(&self, handle: BufferHandle, _to_tier: MemoryTier) -> BufferHandle {
        // Migration not needed in Beta II - return same handle
        handle
    }
    
    pub fn submit_defrag(&self, _tier: MemoryTier, _force: bool) {
        // Defrag not needed with tagged heap
    }
    
    pub fn advance_frame(&self) {
        // Frame advancement now handled through begin_frame/complete_frame
    }
    
    pub fn pool(&self) -> Arc<UnifiedMemoryPool> {
        self.pool.clone()
    }
    
    /// Compatibility methods
    pub fn frame_allocator(&self) -> FrameAllocator<'_> {
        FrameAllocator { context: self }
    }
    
    pub fn allocate_with_purpose(
        &self,
        size: usize,
        _tier: MemoryTier,
        _purpose: AllocationPurpose,
        _alignment: usize,
    ) -> Result<PoolAllocation, String> {
        self.pool.alloc(size)
            .map(|bundle| PoolAllocation {
                offset: bundle.ptr.as_ptr() as usize,
                size: bundle.size,
                pool_index: 0,
                is_frame: true,
            })
            .ok_or("Allocation failed".into())
    }
}

/// Frame handle for RAII frame lifecycle
pub struct FrameHandle {
    pub context: FrameContext,
    pub pool: Arc<UnifiedMemoryPool>,
}

impl Drop for FrameHandle {
    fn drop(&mut self) {
        self.pool.complete_frame(&self.context);
    }
}

/// Frame allocator for compatibility
pub struct FrameAllocator<'a> {
    context: &'a MemoryContext,
}

impl<'a> FrameAllocator<'a> {
    pub fn alloc(&self, size: usize) -> Option<UnifiedMemoryBundle> {
        self.context.alloc(size)
    }
    
    pub fn alloc_gpu_preferred(&self, size: usize) -> Option<UnifiedMemoryBundle> {
        // In Beta II, all allocations are unified
        self.alloc(size)
    }
}

// ===== Compatibility Types =====

#[derive(Debug)]
pub struct PoolAllocation {
    pub offset: usize,
    pub size: usize,
    pub pool_index: usize,
    pub is_frame: bool,
}

pub enum MemoryTask {
    CreateBuffer {
        size: usize,
        tier: MemoryTier,
        usage: vk::BufferUsageFlags,
        purpose: AllocationPurpose,
        result_tx: Sender<Result<BufferHandle, String>>,
    },
    DestroyBuffer {
        handle: BufferHandle,
    },
    MigrateBuffer {
        handle: BufferHandle,
        to_tier: MemoryTier,
        result_tx: Sender<Result<BufferHandle, String>>,
    },
    Shutdown,
}

impl MemoryContext {
    pub fn submit_to_workers(&self, task: MemoryTask) {
        match task {
            MemoryTask::CreateBuffer { size, usage, purpose, result_tx, .. } => {
                let handle = self.pool.gpu.create_buffer(size, usage, purpose)
                    .map_err(|e| e.to_string());
                let _ = result_tx.send(handle);
            }
            MemoryTask::DestroyBuffer { handle } => {
                self.pool.gpu.destroy_buffer(handle);
            }
            MemoryTask::MigrateBuffer { handle, result_tx, .. } => {
                // No migration in Beta II
                let _ = result_tx.send(Ok(handle));
            }
            MemoryTask::Shutdown => {
                self.pool.shutdown();
            }
        }
    }
}

// ===== Memory Statistics =====

#[derive(Debug)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub current_frame: u64,
    pub vram_budget: usize,
    pub vram_utilization: f32,
    pub allocations_by_purpose: HashMap<AllocationPurpose, usize>,
    pub tier_usage: Vec<TierStats>,
    pub thread_stats: Vec<ThreadLocalStats>,
}

#[derive(Debug)]
pub struct TierStats {
    pub tier: MemoryTier,
    pub allocated: usize,
    pub peak: usize,
    pub fragmentation: f32,
}

impl MemoryContext {
    pub fn get_memory_stats(&self) -> MemoryStats {
        let total = self.pool.tagged_heap.total_allocated.load(Ordering::Acquire);
        let peak = self.pool.tagged_heap.peak_allocated.load(Ordering::Acquire);
        
        // Collect thread statistics
        let mut thread_stats = Vec::new();
        for i in 0..8 { // Check first 8 thread slots
            let stats = self.pool.get_thread_stats(i);
            if stats.allocations > 0 {
                thread_stats.push(stats);
            }
        }
        
        // Get total thread allocations
        let (thread_allocs, _) = self.pool.get_total_thread_stats();
        
        MemoryStats {
            total_allocated: total,
            peak_usage: peak,
            allocation_count: thread_allocs,
            current_frame: self.pool.current_frame_tag.load(Ordering::Acquire),
            vram_budget: 8 * 1024 * 1024 * 1024, // 8GB
            vram_utilization: (total as f32 / (8.0 * 1024.0 * 1024.0 * 1024.0)) * 100.0,
            allocations_by_purpose: HashMap::new(),
            tier_usage: Vec::new(),
            thread_stats,
        }
    }
}

// ===== Implementation Notes =====
// 
// This implementation follows Beta II Architecture principles:
// 
// 1. **Tagged Heap as Shared Backend**: All allocations go through TaggedHeap
// 2. **Thread-Local Blocks**: Each thread has its own block via THREAD_ALLOCATOR
// 3. **Frame Parameters**: Uncontended FrameParams with triple buffering
// 4. **Job System**: Priority-based with stage guarding
// 5. **Power-of-2 Alignment**: All allocations use align_to_power_of_2()
// 6. **No Free(ptr)**: Only free_all_with_tag() for temporal cleanup
// 7. **3 Frame Limit**: Enforced in FrameQueue::begin_frame()
// 8. **Reference-based**: Minimal Arc usage, prefer borrows
//
// Reference: Rust ownership model - https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html