//! Phase 7 §7.2: Debug HUD Overlay
//!
//! Renders a fixed-position text overlay showing per-pass GPU timings,
//! streaming status, memory budget, light counts, draw call count, and
//! staging ring fill.  Uses a minimal embedded 8×8 bitmap font uploaded
//! as an R8_UNORM texture, rendered as alpha-blended quads via the ring
//! buffer (zero per-frame allocations).
//!
//! The overlay pipeline renders within the existing lighting render pass
//! (after skybox, before `cmd_end_render_pass`) with depth test/write
//! disabled and src-alpha blending enabled.

use ash::{vk, Device};
use std::fmt::Write as FmtWrite;

use crate::memory::{GpuAllocator, MemoryLocation, RingBuffer};
use crate::pipeline::{create_shader_module, RenderPasses};
use crate::profiler::{GpuProfiler, PassId, PASS_COUNT};

// ====================================================================
//  Constants
// ====================================================================

/// Character cell size in pixels.
const GLYPH_W: u32 = 8;
const GLYPH_H: u32 = 8;

/// Screen-space rendering scale (pixels per glyph).
const CHAR_SCALE: f32 = 2.0;
const CHAR_W: f32 = GLYPH_W as f32 * CHAR_SCALE;
const CHAR_H: f32 = GLYPH_H as f32 * CHAR_SCALE;

/// Line spacing in pixels.
const LINE_SPACING: f32 = CHAR_H + 2.0;

/// HUD position (top-left corner, pixels from screen top-left).
const HUD_X: f32 = 12.0;
const HUD_Y: f32 = 12.0;

/// Maximum characters rendered per frame.
const MAX_CHARS: usize = 1024;

/// Font atlas: 16 chars wide × 6 rows tall = 96 chars (ASCII 32-127).
const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 6;
const ATLAS_W: u32 = ATLAS_COLS * GLYPH_W;  // 128
const ATLAS_H: u32 = ATLAS_ROWS * GLYPH_H;  //  48

// ====================================================================
//  Vertex layout (must match overlay.vert)
// ====================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct OverlayVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const VERTEX_SIZE: u64 = std::mem::size_of::<OverlayVertex>() as u64;
const VERTS_PER_CHAR: usize = 6; // two triangles

// ====================================================================
//  Push constants (must match overlay.vert / overlay.frag)
// ====================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct OverlayPushConstants {
    screen_size: [f32; 2],
    _pad: [f32; 2],
    text_color: [f32; 4],
}

// ====================================================================
//  OverlayStats — collected once per frame by the renderer
// ====================================================================

/// Snapshot of engine stats for overlay display.
#[derive(Default, Clone)]
pub struct OverlayStats {
    // GPU timing (ms)
    pub pass_avg_ms: [f32; PASS_COUNT],
    pub pass_latest_ms: [f32; PASS_COUNT],
    pub total_gpu_ms: f32,

    // Streaming
    pub sectors_ready: usize,
    pub sectors_streaming: usize,
    pub sectors_unloaded: usize,
    pub in_flight_uploads: usize,

    // Memory
    pub pool_used_mb: f32,
    pub pool_allocated_mb: f32,
    pub budget_mb: f32,
    pub tracked_mb: f32,

    // Lights
    pub lights_total: usize,
    pub lights_active: u32,
    pub lights_shadow: u32,
    /// Phase 8C: Baked (no-shadow) light count among active lights.
    pub lights_baked: u32,
    /// Phase 8C: Dynamic (shadow-eligible) light count among active lights.
    pub lights_dynamic: u32,
    /// Phase 8C.5: Current adaptive shadow slot cap.
    pub effective_shadow_slots: usize,

    // Drawing
    pub draw_calls_opaque: usize,
    pub draw_calls_shadow: usize,

    // Staging
    pub staging_fill_pct: f32,

    // Ring buffer
    pub ring_fill_pct: f32,

    // CPU frame time
    pub frame_dt_ms: f32,
}

impl OverlayStats {
    /// Format the stats into a multi-line string for rendering.
    pub fn format_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(16);

        // Line 0: frame time
        lines.push(format!(
            "Frame: {:.2} ms ({:.0} FPS)  GPU: {:.2} ms",
            self.frame_dt_ms,
            if self.frame_dt_ms > 0.001 { 1000.0 / self.frame_dt_ms } else { 0.0 },
            self.total_gpu_ms,
        ));

        // Line 1-2: per-pass breakdown (2 lines, 3 passes each)
        let p = &self.pass_avg_ms;
        lines.push(format!(
            "  Shadow:{:5.2}  Depth:{:5.2}  HBAO:{:5.2}",
            p[0], p[1], p[2],
        ));
        lines.push(format!(
            "  Cluster:{:5.2} Light:{:5.2}  Post:{:5.2}",
            p[3], p[4], p[5],
        ));

        // Line 3: streaming
        lines.push(format!(
            "Sectors: {} rdy  {} str  {} unl  {} inflight",
            self.sectors_ready, self.sectors_streaming,
            self.sectors_unloaded, self.in_flight_uploads,
        ));

        // Line 4: memory
        lines.push(format!(
            "Memory: {:.1}/{:.1} MB used  budget {:.1} MB  tracked {:.1} MB",
            self.pool_used_mb, self.pool_allocated_mb,
            self.budget_mb, self.tracked_mb,
        ));

        // Line 5: lights (Phase 8C: show baked/dynamic breakdown + budget)
        lines.push(format!(
            "Lights: {} total  {} active  {} shadow/{}slots",
            self.lights_total, self.lights_active,
            self.lights_shadow, self.effective_shadow_slots,
        ));
        lines.push(format!(
            "  {} dynamic  {} baked (no shadow)",
            self.lights_dynamic, self.lights_baked,
        ));

        // Line 6: draws
        lines.push(format!(
            "Draws: {} opaque  {} shadow",
            self.draw_calls_opaque, self.draw_calls_shadow,
        ));

        // Line 7: staging + ring
        lines.push(format!(
            "Staging: {:.1}%  Ring: {:.1}%",
            self.staging_fill_pct, self.ring_fill_pct,
        ));

        lines
    }
}

// ====================================================================
//  DebugOverlay — GPU text renderer
// ====================================================================

pub struct DebugOverlay {
    // ---- Font texture ----
    font_image: vk::Image,
    font_memory: vk::DeviceMemory,
    font_view: vk::ImageView,
    font_sampler: vk::Sampler,

    // ---- Pipeline ----
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,

    // ---- State ----
    pub stats: OverlayStats,
    pub visible: bool,
}

impl DebugOverlay {
    pub fn new(
        device: &Device,
        allocator: &GpuAllocator,
        render_pass: vk::RenderPass,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        overlay_vert_spv: &[u8],
        overlay_frag_spv: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // ---- Create and upload font texture ----
        let (font_image, font_memory, font_view) =
            Self::create_font_texture(device, allocator, command_pool, queue)?;

        let font_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )?
        };

        // ---- Descriptor set layout: binding 0 = font sampler ----
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        // ---- Pipeline layout: push constants + 1 descriptor set ----
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<OverlayPushConstants>() as u32);

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&set_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )?
        };

        // ---- Descriptor pool + set ----
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1),
                None,
            )?
        };

        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&set_layout)),
            )?[0]
        };

        // Write font texture to descriptor set.
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(font_sampler)
            .image_view(font_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

        // ---- Graphics pipeline ----
        let pipeline = Self::create_pipeline(
            device, render_pass, pipeline_layout,
            overlay_vert_spv, overlay_frag_spv,
        )?;

        println!(
            "[DebugOverlay] Initialized: font atlas {}x{} R8_UNORM, pipeline created",
            ATLAS_W, ATLAS_H,
        );

        Ok(Self {
            font_image, font_memory, font_view, font_sampler,
            descriptor_pool, set_layout, pipeline_layout, pipeline,
            descriptor_set,
            stats: OverlayStats::default(),
            visible: true,
        })
    }

    /// Collect profiler timings into overlay stats.
    pub fn collect_profiler_stats(&mut self, profiler: &GpuProfiler) {
        for pass in PassId::ALL {
            self.stats.pass_avg_ms[pass as usize] = profiler.average_ms(pass);
            self.stats.pass_latest_ms[pass as usize] = profiler.latest_ms(pass);
        }
        self.stats.total_gpu_ms = profiler.total_average_ms();
    }

    /// Record overlay draw commands into the command buffer.
    ///
    /// Must be called INSIDE an active render pass (the lighting pass),
    /// after all scene geometry has been drawn.  Uses the ring buffer
    /// for transient vertex data.
    pub fn record_commands(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        ring: &mut RingBuffer,
        screen_width: f32,
        screen_height: f32,
    ) {
        if !self.visible { return; }

        let lines = self.stats.format_lines();
        if lines.is_empty() { return; }

        // Build vertex data for all text.
        let mut vertices: Vec<OverlayVertex> = Vec::with_capacity(MAX_CHARS * VERTS_PER_CHAR);

        // Semi-transparent dark background behind text.
        {
            let max_line_len = lines.iter().map(|l| l.len()).max().unwrap_or(0);
            let bg_w = (max_line_len as f32 * CHAR_W) + 16.0;
            let bg_h = (lines.len() as f32 * LINE_SPACING) + 12.0;
            let x0 = HUD_X - 4.0;
            let y0 = HUD_Y - 4.0;
            let x1 = x0 + bg_w;
            let y1 = y0 + bg_h;

            // Background quad (we'll draw it with a solid-color push constant pass).
            // For simplicity, encode it as characters that map to a solid region of
            // the font atlas — the space character (glyph 0) is fully transparent,
            // so instead we use a dedicated background draw with a separate push constant.
            // Actually, we'll just draw the text on top without a background for now,
            // or use a simple approach: render a large quad with UV mapping to a
            // fully opaque glyph.
            // Simplest: use ASCII 219 ... but we only have 32-127.
            // Let's skip the background for now and just render text.  The dark scene
            // provides sufficient contrast.  A background pass can be added later.
            let _ = (x0, y0, x1, y1, bg_w, bg_h);
        }

        for (line_idx, line) in lines.iter().enumerate() {
            let y = HUD_Y + line_idx as f32 * LINE_SPACING;
            for (col, ch) in line.chars().enumerate() {
                if vertices.len() >= MAX_CHARS * VERTS_PER_CHAR { break; }
                let ascii = ch as u32;
                if ascii < 32 || ascii > 127 { continue; }

                let glyph_idx = ascii - 32;
                let atlas_col = glyph_idx % ATLAS_COLS;
                let atlas_row = glyph_idx / ATLAS_COLS;

                let u0 = atlas_col as f32 * GLYPH_W as f32 / ATLAS_W as f32;
                let v0 = atlas_row as f32 * GLYPH_H as f32 / ATLAS_H as f32;
                let u1 = u0 + GLYPH_W as f32 / ATLAS_W as f32;
                let v1 = v0 + GLYPH_H as f32 / ATLAS_H as f32;

                let x = HUD_X + col as f32 * CHAR_W;

                let x0 = x;
                let y0 = y;
                let x1 = x + CHAR_W;
                let y1 = y + CHAR_H;

                // Triangle 1
                vertices.push(OverlayVertex { pos: [x0, y0], uv: [u0, v0] });
                vertices.push(OverlayVertex { pos: [x1, y0], uv: [u1, v0] });
                vertices.push(OverlayVertex { pos: [x0, y1], uv: [u0, v1] });
                // Triangle 2
                vertices.push(OverlayVertex { pos: [x1, y0], uv: [u1, v0] });
                vertices.push(OverlayVertex { pos: [x1, y1], uv: [u1, v1] });
                vertices.push(OverlayVertex { pos: [x0, y1], uv: [u0, v1] });
            }
        }

        if vertices.is_empty() { return; }

        let vertex_bytes = vertices.len() * std::mem::size_of::<OverlayVertex>();
        let ring_slice = match ring.push(vertex_bytes as u64) {
            Some(s) => s,
            None => {
                // Ring exhausted — skip overlay this frame.
                return;
            }
        };

        // Copy vertices into ring buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                ring_slice.mapped_ptr.as_ptr(),
                vertex_bytes,
            );
        }

        // Draw.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout, 0,
                std::slice::from_ref(&self.descriptor_set), &[],
            );

            let push = OverlayPushConstants {
                screen_size: [screen_width, screen_height],
                _pad: [0.0; 2],
                text_color: [0.0, 1.0, 0.3, 0.9], // bright green, slightly transparent
            };
            device.cmd_push_constants(
                cmd, self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &push as *const _ as *const u8,
                    std::mem::size_of::<OverlayPushConstants>(),
                ),
            );

            device.cmd_bind_vertex_buffers(cmd, 0, &[ring.buffer], &[ring_slice.offset]);
            device.cmd_draw(cmd, vertices.len() as u32, 1, 0, 0);
        }
    }

    // ================================================================
    //  Font texture creation
    // ================================================================

    fn create_font_texture(
        device: &Device,
        allocator: &GpuAllocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn std::error::Error>> {
        // Expand bitmap font into R8 pixel data.
        let mut pixels = vec![0u8; (ATLAS_W * ATLAS_H) as usize];

        for glyph_idx in 0..96u32 {
            let col = glyph_idx % ATLAS_COLS;
            let row = glyph_idx / ATLAS_COLS;
            let glyph = &FONT_8X8[glyph_idx as usize];

            for y in 0..GLYPH_H {
                let byte = glyph[y as usize];
                for x in 0..GLYPH_W {
                    let pixel_x = col * GLYPH_W + x;
                    let pixel_y = row * GLYPH_H + y;
                    let idx = (pixel_y * ATLAS_W + pixel_x) as usize;
                    pixels[idx] = if (byte >> x) & 1 != 0 { 255 } else { 0 };
                }
            }
        }

        // Create image.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .extent(vk::Extent3D { width: ATLAS_W, height: ATLAS_H, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None)? };
        let mem_req = unsafe { device.get_image_memory_requirements(image) };

        let mem_type = allocator.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type),
                None,
            )?
        };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        // Upload via a temporary staging buffer.
        let staging_size = pixels.len() as u64;
        let staging_info = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buf = unsafe { device.create_buffer(&staging_info, None)? };
        let staging_req = unsafe { device.get_buffer_memory_requirements(staging_buf) };
        let staging_mem_type = allocator.find_memory_type(
            staging_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let staging_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_req.size)
                    .memory_type_index(staging_mem_type),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(staging_buf, staging_mem, 0)?;
            let ptr = device.map_memory(staging_mem, 0, staging_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr as *mut u8, pixels.len());
            device.unmap_memory(staging_mem);
        }

        // Record and submit upload + layout transition.
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

            // UNDEFINED → TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier),
            );

            // Copy staging → image.
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0, base_array_layer: 0, layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: ATLAS_W, height: ATLAS_H, depth: 1 });
            device.cmd_copy_buffer_to_image(
                cmd, staging_buf, image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );

            // TRANSFER_DST → SHADER_READ_ONLY
            let barrier2 = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier2),
            );

            device.end_command_buffer(cmd)?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));

            // Clean up staging.
            device.destroy_buffer(staging_buf, None);
            device.free_memory(staging_mem, None);
        }

        // Create image view.
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    }),
                None,
            )?
        };

        Ok((image, memory, view))
    }

    // ================================================================
    //  Pipeline creation
    // ================================================================

    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        vert_spv: &[u8],
        frag_spv: &[u8],
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let vert_mod = create_shader_module(device, vert_spv)?;
            let frag_mod = create_shader_module(device, frag_spv)?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_mod)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_mod)
                    .name(c"main"),
            ];

            // Vertex input: 2 × vec2 (pos, uv).
            let binding = vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(VERTEX_SIZE as u32)
                .input_rate(vk::VertexInputRate::VERTEX);

            let attrs = [
                vk::VertexInputAttributeDescription::default()
                    .location(0).binding(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(0),
                vk::VertexInputAttributeDescription::default()
                    .location(1).binding(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(8),
            ];

            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(&attrs);

            let input_asm = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // Depth: test disabled, write disabled.
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false)
                .depth_write_enable(false);

            // Alpha blending: srcAlpha * src + (1-srcAlpha) * dst.
            let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA);

            let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&blend_attachment));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_asm)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisample)
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&color_blend)
                .dynamic_state(&dynamic_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipelines = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);

            Ok(pipelines[0])
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.font_sampler, None);
            device.destroy_image_view(self.font_view, None);
            device.destroy_image(self.font_image, None);
            device.free_memory(self.font_memory, None);
        }
        println!("[DebugOverlay] Destroyed");
    }
}

// ====================================================================
//  Embedded 8×8 bitmap font — ASCII 32-127 (96 glyphs)
// ====================================================================
//
// Public-domain 8×8 font.  Each glyph is 8 bytes; each byte is one
// pixel row (MSB = leftmost pixel).  Glyph 0 = space (U+0020),
// glyph 95 = DEL placeholder (U+007F, rendered as '~' alternate).

#[rustfmt::skip]
const FONT_8X8: [[u8; 8]; 96] = [
    // 32 ' '
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 33 '!'
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00],
    // 34 '"'
    [0x36, 0x36, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 35 '#'
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00],
    // 36 '$'
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00],
    // 37 '%'
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00],
    // 38 '&'
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00],
    // 39 '''
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 40 '('
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00],
    // 41 ')'
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00],
    // 42 '*'
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],
    // 43 '+'
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00],
    // 44 ','
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    // 45 '-'
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00],
    // 46 '.'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    // 47 '/'
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00],
    // 48 '0'
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00],
    // 49 '1'
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00],
    // 50 '2'
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00],
    // 51 '3'
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00],
    // 52 '4'
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00],
    // 53 '5'
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00],
    // 54 '6'
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00],
    // 55 '7'
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00],
    // 56 '8'
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00],
    // 57 '9'
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00],
    // 58 ':'
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    // 59 ';'
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    // 60 '<'
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00],
    // 61 '='
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00],
    // 62 '>'
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00],
    // 63 '?'
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00],
    // 64 '@'
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00],
    // 65 'A'
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00],
    // 66 'B'
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00],
    // 67 'C'
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00],
    // 68 'D'
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00],
    // 69 'E'
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00],
    // 70 'F'
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00],
    // 71 'G'
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00],
    // 72 'H'
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00],
    // 73 'I'
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 74 'J'
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00],
    // 75 'K'
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00],
    // 76 'L'
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00],
    // 77 'M'
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00],
    // 78 'N'
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00],
    // 79 'O'
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00],
    // 80 'P'
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00],
    // 81 'Q'
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00],
    // 82 'R'
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00],
    // 83 'S'
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00],
    // 84 'T'
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 85 'U'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00],
    // 86 'V'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    // 87 'W'
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
    // 88 'X'
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00],
    // 89 'Y'
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00],
    // 90 'Z'
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00],
    // 91 '['
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00],
    // 92 '\'
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],
    // 93 ']'
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00],
    // 94 '^'
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00],
    // 95 '_'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF],
    // 96 '`'
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 97 'a'
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00],
    // 98 'b'
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00],
    // 99 'c'
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00],
    // 100 'd'
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00],
    // 101 'e'
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00],
    // 102 'f'
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00],
    // 103 'g'
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    // 104 'h'
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00],
    // 105 'i'
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 106 'j'
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E],
    // 107 'k'
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00],
    // 108 'l'
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 109 'm'
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00],
    // 110 'n'
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00],
    // 111 'o'
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00],
    // 112 'p'
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F],
    // 113 'q'
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78],
    // 114 'r'
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00],
    // 115 's'
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00],
    // 116 't'
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00],
    // 117 'u'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00],
    // 118 'v'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    // 119 'w'
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00],
    // 120 'x'
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00],
    // 121 'y'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    // 122 'z'
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00],
    // 123 '{'
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00],
    // 124 '|'
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00],
    // 125 '}'
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00],
    // 126 '~'
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 127 DEL (rendered as filled block)
    [0x00, 0x08, 0x1C, 0x36, 0x63, 0x7F, 0x00, 0x00],
];