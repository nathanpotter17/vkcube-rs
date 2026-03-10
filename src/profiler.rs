//! Phase 7 §7.1: GPU Timestamp Profiling
//!
//! `GpuProfiler` wraps a `VkQueryPool` with `VK_QUERY_TYPE_TIMESTAMP`.
//! Each render pass writes begin + end timestamps via `vkCmdWriteTimestamp`.
//! Results are read non-blocking at frame start, converted to milliseconds
//! via `VkPhysicalDeviceLimits::timestampPeriod`, and stored in a 128-frame
//! ring buffer for moving-average display.
//!
//! Double-buffered: slot 0 and slot 1 alternate per frame (matching
//! `MAX_FRAMES_IN_FLIGHT = 2`).  After `wait_for_fences`, the current
//! slot's previous results are guaranteed complete, so reads never block.

use ash::{vk, Device};
use crate::memory::MAX_FRAMES_IN_FLIGHT;

// ====================================================================
//  Constants
// ====================================================================

/// Number of named render passes profiled.
pub const PASS_COUNT: usize = 6;

/// Rolling history depth for moving-average computation.
const HISTORY_SIZE: usize = 128;

// ====================================================================
//  PassId — named profiling scopes
// ====================================================================

/// Identifies a profiled render pass.  Ordinal doubles as array index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum PassId {
    Shadow   = 0,
    Depth    = 1,
    Hbao     = 2,
    Cluster  = 3,
    Lighting = 4,
    Post     = 5,
}

impl PassId {
    pub const ALL: [PassId; PASS_COUNT] = [
        PassId::Shadow,
        PassId::Depth,
        PassId::Hbao,
        PassId::Cluster,
        PassId::Lighting,
        PassId::Post,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PassId::Shadow   => "Shadow",
            PassId::Depth    => "Depth",
            PassId::Hbao     => "HBAO",
            PassId::Cluster  => "Cluster",
            PassId::Lighting => "Lighting",
            PassId::Post     => "Post",
        }
    }
}

// ====================================================================
//  GpuProfiler
// ====================================================================

pub struct GpuProfiler {
    query_pool: vk::QueryPool,

    /// Nanoseconds per timestamp tick (from `VkPhysicalDeviceLimits`).
    timestamp_period_ns: f32,

    /// Bitmask derived from `timestampValidBits` — applied to raw
    /// timestamp values before subtraction to avoid wrap-around.
    timestamp_mask: u64,

    /// 2 × PASS_COUNT queries per frame slot.
    queries_per_frame: u32,

    /// Per-pass timings in milliseconds, ring-buffered.
    /// `history[cursor][pass]` = ms for that frame/pass.
    history: [[f32; PASS_COUNT]; HISTORY_SIZE],
    history_cursor: usize,
    history_count: usize,

    /// Summed per-pass totals for the current history window.
    /// Updated incrementally to avoid re-scanning the ring.
    running_sum: [f64; PASS_COUNT],

    /// Latest per-pass ms (most recent frame that was read).
    latest: [f32; PASS_COUNT],

    /// Counts read_results calls. Slots aren't safe to read until each
    /// has been through at least one full record+submit cycle.
    frames_elapsed: usize,

    /// Whether the profiler is enabled (avoids queries when toggled off).
    enabled: bool,
}

impl GpuProfiler {
    /// Create the profiler.
    ///
    /// `timestamp_period` — `VkPhysicalDeviceLimits::timestampPeriod`
    ///   (nanoseconds per tick).
    /// `timestamp_valid_bits` — from `VkQueueFamilyProperties::timestampValidBits`
    ///   for the graphics queue family.  Pass 0 to disable masking (unlikely).
    pub fn new(
        device: &Device,
        timestamp_period: f32,
        timestamp_valid_bits: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let queries_per_frame = (PASS_COUNT * 2) as u32;
        let total_queries = queries_per_frame * MAX_FRAMES_IN_FLIGHT as u32;

        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(total_queries);

        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };

        // we reset query stats via reset_queries()

        let timestamp_mask = if timestamp_valid_bits == 0 || timestamp_valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << timestamp_valid_bits) - 1
        };

        println!(
            "[GpuProfiler] Created: {} queries, period={:.2} ns/tick, valid_bits={}, mask=0x{:X}",
            total_queries, timestamp_period, timestamp_valid_bits, timestamp_mask,
        );

        Ok(Self {
            query_pool,
            timestamp_period_ns: timestamp_period,
            timestamp_mask,
            queries_per_frame,
            history: [[0.0; PASS_COUNT]; HISTORY_SIZE],
            history_cursor: 0,
            history_count: 0,
            running_sum: [0.0; PASS_COUNT],
            latest: [0.0; PASS_COUNT],
            frames_elapsed: 0,
            enabled: true,
        })
    }

    /// Insert a begin-timestamp for `pass` into the command buffer.
    ///
    /// # Safety
    /// `cmd` must be in recording state.  `frame_slot` must be
    /// `current_frame % MAX_FRAMES_IN_FLIGHT`.
    pub fn begin_pass(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        pass: PassId,
        frame_slot: usize,
    ) {
        if !self.enabled { return; }
        let query = self.query_index(frame_slot, pass, false);
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                query,
            );
        }
    }

    /// Insert an end-timestamp for `pass` into the command buffer.
    ///
    /// # Safety
    /// `cmd` must be in recording state.
    pub fn end_pass(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        pass: PassId,
        frame_slot: usize,
    ) {
        if !self.enabled { return; }
        let query = self.query_index(frame_slot, pass, true);
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                query,
            );
        }
    }

    /// Reset the query range for `frame_slot` inside a command buffer.
    ///
    /// Must be called BEFORE any `begin_pass`/`end_pass` for this slot
    /// in the same command buffer.  Typically the first operation after
    /// `vkBeginCommandBuffer`.
    pub fn reset_queries(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        if !self.enabled { return; }
        let first = frame_slot as u32 * self.queries_per_frame;
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, first, self.queries_per_frame);
        }
    }

    /// Read timestamp results for `frame_slot` from the previous frame.
    ///
    /// Call this AFTER `vkWaitForFences` for the current frame slot,
    /// BEFORE recording new commands.  The fence guarantees these
    /// query results are available.
    ///
    /// Returns `true` if results were successfully read and history
    /// was updated.
    pub fn read_results(&mut self, device: &Device, frame_slot: usize) -> bool {
        if !self.enabled { return false; }

        // Each slot must be recorded+submitted once before its queries
        // are valid.  With MAX_FRAMES_IN_FLIGHT=2, slot 0 is first
        // submitted in frame 1, slot 1 in frame 2.  Both are safe to
        // read starting frame 3 (frames_elapsed >= 2).
        if self.frames_elapsed < MAX_FRAMES_IN_FLIGHT {
            self.frames_elapsed += 1;
            return false;
        }

        let first = frame_slot as u32 * self.queries_per_frame;
        let count = self.queries_per_frame;

        // 2 u64 per query: value + availability.  We use WAIT flag = 0
        // (non-blocking) combined with WITH_AVAILABILITY so we can
        // detect partial results.
        let mut timestamps = vec![0u64; count as usize];

        let result = unsafe {
            device.get_query_pool_results(
                self.query_pool,
                first,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64,
            )
        };

        if result.is_err() {
            // VK_NOT_READY or other issue — skip this frame.
            return false;
        }

        // Compute per-pass durations in milliseconds.
        let mut frame_ms = [0.0f32; PASS_COUNT];
        for pass in PassId::ALL {
            let bi = (pass as usize) * 2;
            let ei = bi + 1;
            if bi >= timestamps.len() || ei >= timestamps.len() { continue; }

            let begin = timestamps[bi] & self.timestamp_mask;
            let end = timestamps[ei] & self.timestamp_mask;

            // Guard against wrap-around: if end < begin, the timestamp
            // counter wrapped within the valid-bit range.
            let delta = if end >= begin {
                end - begin
            } else {
                // Wrapped: add the range covered by valid bits.
                (self.timestamp_mask - begin) + end + 1
            };

            let ns = delta as f64 * self.timestamp_period_ns as f64;
            frame_ms[pass as usize] = (ns / 1_000_000.0) as f32;
        }

        // Subtract the oldest entry from running sums (if history is full).
        if self.history_count >= HISTORY_SIZE {
            let old = &self.history[self.history_cursor];
            for p in 0..PASS_COUNT {
                self.running_sum[p] -= old[p] as f64;
            }
        }

        // Write new entry and update sums.
        self.history[self.history_cursor] = frame_ms;
        for p in 0..PASS_COUNT {
            self.running_sum[p] += frame_ms[p] as f64;
        }
        self.latest = frame_ms;

        self.history_cursor = (self.history_cursor + 1) % HISTORY_SIZE;
        if self.history_count < HISTORY_SIZE {
            self.history_count += 1;
        }

        true
    }

    /// Moving-average time in milliseconds for a specific pass.
    pub fn average_ms(&self, pass: PassId) -> f32 {
        if self.history_count == 0 { return 0.0; }
        (self.running_sum[pass as usize] / self.history_count as f64) as f32
    }

    /// Most recent frame's time in milliseconds for a specific pass.
    pub fn latest_ms(&self, pass: PassId) -> f32 {
        self.latest[pass as usize]
    }

    /// Sum of all pass averages — approximate total GPU frame time.
    pub fn total_average_ms(&self) -> f32 {
        PassId::ALL.iter().map(|&p| self.average_ms(p)).sum()
    }

    /// Number of history samples collected.
    pub fn sample_count(&self) -> usize {
        self.history_count
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // ---- Internal helpers ----

    /// Map (frame_slot, pass, is_end) to a query pool index.
    fn query_index(&self, frame_slot: usize, pass: PassId, is_end: bool) -> u32 {
        let base = frame_slot as u32 * self.queries_per_frame;
        base + (pass as u32) * 2 + if is_end { 1 } else { 0 }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_query_pool(self.query_pool, None); }
        println!("[GpuProfiler] Destroyed");
    }
}
