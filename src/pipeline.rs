//! Pipeline Management (Phase 1)
//!
//! Centralizes shader module loading, render pass creation, and
//! graphics pipeline construction for the multi-pass renderer.
//!
//! Pass structure:
//!   1. Depth pre-pass  (depth-only, enables early-Z)
//!   2. Lighting pass   (PBR shading with clustered lights)
//!   3. Post-process    (tone mapping, gamma — later: HBAO, bloom, TAA)
//!
//! Shadow pass and cluster compute are future additions (Phase 2).

use ash::{vk, Device};
use crate::scene::Vertex;
use crate::memory::MAX_FRAMES_IN_FLIGHT;

// ====================================================================
//  Shader Utilities
// ====================================================================

/// Align raw SPIR-V bytes to u32 words for VkShaderModule creation.
pub fn align_shader_code(code: &[u8]) -> Vec<u32> {
    code.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load a SPIR-V shader module from raw bytes.
pub fn create_shader_module(
    device: &Device,
    spv: &[u8],
) -> Result<vk::ShaderModule, vk::Result> {
    let code = align_shader_code(spv);
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&info, None) }
}

// ====================================================================
//  Descriptor Set Layouts (Phase 1 architecture)
// ====================================================================

/// Create the four descriptor set layouts for the Phase 1 pipeline.
///
/// Returns `[set0, set1, set2, set3]`:
///   - Set 0: Per-frame globals (view/proj UBO, material SSBO)
///   - Set 1: Bindless textures (created by TextureManager, passed in)
///   - Set 2: Shadow maps (placeholder — empty for Phase 1)
///   - Set 3: Per-draw dynamic UBO (model + material_id, ring buffer)
pub struct DescriptorLayouts {
    pub per_frame: vk::DescriptorSetLayout,      // set 0
    pub bindless_textures: vk::DescriptorSetLayout, // set 1 (owned by TextureManager)
    pub shadow_maps: vk::DescriptorSetLayout,     // set 2
    pub per_draw: vk::DescriptorSetLayout,        // set 3
}

impl DescriptorLayouts {
    pub fn new(
        device: &Device,
        bindless_textures_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // ---- Set 0: Per-frame globals ----
        //
        // Binding 0: View/Proj UBO (updated once per frame)
        // Binding 5: Material SSBO (updated when materials change)
        let set0_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let per_frame = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&set0_bindings),
                None,
            )?
        };

        // ---- Set 2: Shadow maps (empty placeholder for Phase 1) ----
        let shadow_maps = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default(),
                None,
            )?
        };

        // ---- Set 3: Per-draw dynamic UBO ----
        let set3_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let per_draw = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&set3_binding)),
                None,
            )?
        };

        Ok(Self {
            per_frame,
            bindless_textures: bindless_textures_layout,
            shadow_maps,
            per_draw,
        })
    }

    /// All four layouts in order, for pipeline layout creation.
    pub fn all(&self) -> [vk::DescriptorSetLayout; 4] {
        [
            self.per_frame,
            self.bindless_textures,
            self.shadow_maps,
            self.per_draw,
        ]
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            // Note: bindless_textures layout is owned by TextureManager.
            device.destroy_descriptor_set_layout(self.per_frame, None);
            device.destroy_descriptor_set_layout(self.shadow_maps, None);
            device.destroy_descriptor_set_layout(self.per_draw, None);
        }
    }
}

// ====================================================================
//  Render Passes
// ====================================================================

/// All render passes used by the multi-pass pipeline.
pub struct RenderPasses {
    /// Depth pre-pass: renders depth-only into the main depth buffer.
    pub depth_prepass: vk::RenderPass,
    /// Main lighting pass: reads depth (early-Z), writes color.
    pub lighting: vk::RenderPass,
}

impl RenderPasses {
    pub fn new(
        device: &Device,
        color_format: vk::Format,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let depth_prepass = Self::create_depth_prepass(device)?;
        let lighting = Self::create_lighting_pass(device, color_format)?;

        Ok(Self {
            depth_prepass,
            lighting,
        })
    }

    /// Depth-only render pass.
    ///
    /// Single attachment: D32_SFLOAT.
    /// Initial layout UNDEFINED → final DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    fn create_depth_prepass(
        device: &Device,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        let info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe { Ok(device.create_render_pass(&info, None)?) }
    }

    /// Main lighting render pass.
    ///
    /// Two attachments: color (swapchain format) + depth (D32_SFLOAT).
    /// Depth is loaded (from pre-pass), not cleared.
    fn create_lighting_pass(
        device: &Device,
        color_format: vk::Format,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        let attachments = [
            // Color attachment — swapchain image.
            vk::AttachmentDescription::default()
                .format(color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            // Depth attachment — reused from depth pre-pass.
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            );

        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe { Ok(device.create_render_pass(&info, None)?) }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_render_pass(self.depth_prepass, None);
            device.destroy_render_pass(self.lighting, None);
        }
    }
}

// ====================================================================
//  Pipeline Definitions
// ====================================================================

/// All graphics pipelines used by the multi-pass renderer.
pub struct Pipelines {
    pub depth_prepass: vk::Pipeline,
    pub lighting: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub cache: vk::PipelineCache,
}

impl Pipelines {
    /// Create all graphics pipelines.
    ///
    /// `vert_spv` / `frag_spv`: PBR vertex/fragment SPIR-V.
    /// `depth_vert_spv` / `depth_frag_spv`: depth-only vertex/fragment.
    pub fn new(
        device: &Device,
        layouts: &DescriptorLayouts,
        passes: &RenderPasses,
        vert_spv: &[u8],
        frag_spv: &[u8],
        depth_vert_spv: &[u8],
        depth_frag_spv: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Pipeline cache (empty for now — could persist to disk).
        let cache = unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default(),
                None,
            )?
        };

        // Pipeline layout uses all four descriptor set layouts.
        let set_layouts = layouts.all();
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts),
                None,
            )?
        };

        // ---- Depth pre-pass pipeline ----
        let depth_prepass = Self::create_depth_pipeline(
            device,
            cache,
            layout,
            passes.depth_prepass,
            depth_vert_spv,
            depth_frag_spv,
        )?;

        // ---- Lighting pipeline ----
        let lighting = Self::create_lighting_pipeline(
            device,
            cache,
            layout,
            passes.lighting,
            vert_spv,
            frag_spv,
        )?;

        Ok(Self {
            depth_prepass,
            lighting,
            layout,
            cache,
        })
    }

    fn create_depth_pipeline(
        device: &Device,
        cache: vk::PipelineCache,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        vert_spv: &[u8],
        frag_spv: &[u8],
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let vert_module = create_shader_module(device, vert_spv)?;
            let frag_module = create_shader_module(device, frag_spv)?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_module)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(c"main"),
            ];

            let binding = Vertex::binding_description();
            let attributes = Vertex::attribute_descriptions();
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(&attributes);

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            // Depth pre-pass: no color writes, depth write enabled.
            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);

            // No color blend attachment for depth-only pass.
            let color_blending = vk::PipelineColorBlendStateCreateInfo::default();

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&color_blending)
                .dynamic_state(&dynamic_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipelines = device
                .create_graphics_pipelines(cache, &[pipeline_info], None)
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            Ok(pipelines[0])
        }
    }

    fn create_lighting_pipeline(
        device: &Device,
        cache: vk::PipelineCache,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        vert_spv: &[u8],
        frag_spv: &[u8],
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let vert_module = create_shader_module(device, vert_spv)?;
            let frag_module = create_shader_module(device, frag_spv)?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_module)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(c"main"),
            ];

            let binding = Vertex::binding_description();
            let attributes = Vertex::attribute_descriptions();
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(&attributes);

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // Lighting pass: depth test enabled, depth write DISABLED
            // (already written by depth pre-pass). Compare EQUAL for
            // perfect early-Z rejection.
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

            let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false);
            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&blend_attachment));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&color_blending)
                .dynamic_state(&dynamic_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipelines = device
                .create_graphics_pipelines(cache, &[pipeline_info], None)
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            Ok(pipelines[0])
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.depth_prepass, None);
            device.destroy_pipeline(self.lighting, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline_cache(self.cache, None);
        }
    }
}

// ====================================================================
//  Framebuffers
// ====================================================================

/// Framebuffers for the multi-pass pipeline.
pub struct PassFramebuffers {
    /// One per swapchain image: depth-only (depth attachment only).
    pub depth_prepass: Vec<vk::Framebuffer>,
    /// One per swapchain image: color + depth.
    pub lighting: Vec<vk::Framebuffer>,
}

impl PassFramebuffers {
    pub fn new(
        device: &Device,
        passes: &RenderPasses,
        swapchain_image_views: &[vk::ImageView],
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut depth_prepass = Vec::new();
        let mut lighting = Vec::new();

        for &color_view in swapchain_image_views {
            // Depth pre-pass: only depth attachment.
            let depth_attach = [depth_view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(passes.depth_prepass)
                .attachments(&depth_attach)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            depth_prepass.push(unsafe { device.create_framebuffer(&fb_info, None)? });

            // Lighting pass: color + depth.
            let light_attach = [color_view, depth_view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(passes.lighting)
                .attachments(&light_attach)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            lighting.push(unsafe { device.create_framebuffer(&fb_info, None)? });
        }

        Ok(Self {
            depth_prepass,
            lighting,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            for &fb in &self.depth_prepass {
                device.destroy_framebuffer(fb, None);
            }
            for &fb in &self.lighting {
                device.destroy_framebuffer(fb, None);
            }
        }
    }
}

// ====================================================================
//  Per-Frame Descriptor Sets
// ====================================================================

/// Descriptor pool and sets for per-frame globals (set 0) and
/// per-draw dynamic UBOs (set 3).
pub struct FrameDescriptors {
    pub pool: vk::DescriptorPool,
    /// Per-frame global descriptor sets (one per frame in flight).
    pub per_frame_sets: Vec<vk::DescriptorSet>,
    /// Per-draw dynamic UBO descriptor sets (one per frame in flight).
    pub per_draw_sets: Vec<vk::DescriptorSet>,
}

impl FrameDescriptors {
    pub fn new(
        device: &Device,
        layouts: &DescriptorLayouts,
        ring_buffer: vk::Buffer,
        ubo_range: u64,
        material_ssbo: vk::Buffer,
        material_ssbo_size: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = MAX_FRAMES_IN_FLIGHT as u32;

        // Pool: we need N uniform-buffer-dynamic sets (per-draw) and
        // N sets with uniform-buffer + storage-buffer (per-frame).
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(n),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(n),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(n * 2); // N per-frame + N per-draw

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Allocate per-frame sets (set 0).
        let per_frame_layouts = vec![layouts.per_frame; MAX_FRAMES_IN_FLIGHT];
        let per_frame_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&per_frame_layouts),
            )?
        };

        // Allocate per-draw sets (set 3).
        let per_draw_layouts = vec![layouts.per_draw; MAX_FRAMES_IN_FLIGHT];
        let per_draw_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&per_draw_layouts),
            )?
        };

        // Write initial descriptors.
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            // Per-frame set 0: binding 0 = view/proj UBO (ring buffer).
            // We use a fixed-size range; the actual data is uploaded
            // each frame via ring buffer push.
            let global_ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(ring_buffer)
                .offset(0)
                .range(ubo_range);

            // Per-frame set 0: binding 5 = material SSBO.
            let material_info = vk::DescriptorBufferInfo::default()
                .buffer(material_ssbo)
                .offset(0)
                .range(material_ssbo_size);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&global_ubo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&material_info)),
            ];

            unsafe { device.update_descriptor_sets(&writes, &[]) };

            // Per-draw set 3: binding 0 = dynamic UBO (ring buffer).
            let per_draw_info = vk::DescriptorBufferInfo::default()
                .buffer(ring_buffer)
                .offset(0)
                .range(ubo_range);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(per_draw_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(std::slice::from_ref(&per_draw_info));

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        Ok(Self {
            pool,
            per_frame_sets,
            per_draw_sets,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}