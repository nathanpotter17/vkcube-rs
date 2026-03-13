//! Pipeline Management (Phase 1 + Phase 2 + Phase 3 + Phase 8A)
//!
//! Pass structure (Phase 2):
//!   1. Shadow pass     (per shadow-casting light × 6 cube faces)
//!   2. Depth pre-pass  (depth-only, enables early-Z)
//!   3. Cluster compute (light → cluster assignment)
//!   4. Lighting pass   (PBR + clustered shading + shadows + GI)
//!
//! Descriptor Set 0 (per-frame globals):
//!   binding 0:  GlobalUbo            (UNIFORM_BUFFER_DYNAMIC)
//!   binding 1:  Light SSBO           (STORAGE_BUFFER)
//!   binding 2:  Cluster SSBO         (STORAGE_BUFFER)
//!   binding 3:  Light Index SSBO     (STORAGE_BUFFER)
//!   binding 4:  ClusterParamsUbo     (UNIFORM_BUFFER_DYNAMIC)
//!   binding 5:  Material SSBO        (STORAGE_BUFFER)
//!   binding 6:  SH Probe SSBO        (STORAGE_BUFFER)            [Phase 3]
//!   binding 7:  ProbeGridParams UBO  (UNIFORM_BUFFER_DYNAMIC)    [Phase 3]
//!   binding 8:  BRDF LUT             (COMBINED_IMAGE_SAMPLER)    [Phase 3]
//!   binding 9:  Irradiance Cube Map  (COMBINED_IMAGE_SAMPLER)    [Phase 3]
//!   binding 10: Pre-filtered Env Map (COMBINED_IMAGE_SAMPLER)    [Phase 3]
//!
//! Descriptor Set 1: Bindless textures (from TextureManager)
//! Descriptor Set 2: Shadow maps (cube map array sampler)
//! Descriptor Set 3: Object SSBO (Phase 8A: replaces per-draw dynamic UBO)
//!   binding 0: STORAGE_BUFFER — GpuObjectData[] indexed by gl_InstanceIndex
//!
//! Push constants (16 bytes): ShadowPushConstants (shadow pass only).

use ash::{vk, Device};
use std::ptr::NonNull;

use crate::light::{
    self, ClusterParamsUbo, GpuCluster, GpuLight, LightIndexHeader, LightSsboHeader,
    ShadowPushConstants, LIGHT_INDEX_CAPACITY, MAX_LIGHTS, MAX_SHADOW_SLOTS,
    SHADOW_MAP_SIZE, TOTAL_CLUSTERS,
};
use crate::memory::{BufferAllocation, BufferHandle, GpuAllocator, MemoryLocation,
                     MAX_FRAMES_IN_FLIGHT};
use crate::scene::Vertex;

// ====================================================================
//  Shader Utilities
// ====================================================================

pub fn align_shader_code(code: &[u8]) -> Vec<u32> {
    code.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn create_shader_module(
    device: &Device,
    spv: &[u8],
) -> Result<vk::ShaderModule, vk::Result> {
    let code = align_shader_code(spv);
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&info, None) }
}

// ====================================================================
//  Descriptor Set Layouts
// ====================================================================

pub struct DescriptorLayouts {
    pub per_frame: vk::DescriptorSetLayout,         // set 0
    pub bindless_textures: vk::DescriptorSetLayout,  // set 1 (TextureManager-owned)
    pub shadow_maps: vk::DescriptorSetLayout,        // set 2
    pub per_draw: vk::DescriptorSetLayout,           // set 3 (now object SSBO layout)
}

impl DescriptorLayouts {
    pub fn new(
        device: &Device,
        bindless_textures_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // ---- Set 0: Per-frame globals ----
        //
        // Binding 0: GlobalUbo               (dynamic UBO)
        // Binding 1: Light SSBO              (storage buffer)
        // Binding 2: Cluster SSBO            (storage buffer)
        // Binding 3: Light Index SSBO        (storage buffer)
        // Binding 4: ClusterParamsUbo        (dynamic UBO)
        // Binding 5: Material SSBO           (storage buffer)
        // Binding 6: SH Probe SSBO           (storage buffer)           [Phase 3]
        // Binding 7: ProbeGridParams UBO     (dynamic UBO)              [Phase 3]
        // Binding 8: BRDF LUT               (combined image sampler)    [Phase 3]
        // Binding 9: Irradiance cube map    (combined image sampler)    [Phase 3]
        // Binding 10: Pre-filtered env map  (combined image sampler)    [Phase 3]
        // Binding 11: Screen-space AO texture (HBAO output)             [Phase 6]
        // Binding 12: DirectionalShadowData  (dynamic UBO)              [CSM]
        let set0_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Phase 3: SH Probe SSBO
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Phase 3: ProbeGridParams dynamic UBO
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Phase 3: BRDF LUT
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Phase 3: Irradiance cube map
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Phase 3: Pre-filtered env map
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Phase 6: Screen-space AO texture (HBAO output)
            vk::DescriptorSetLayoutBinding::default()
                .binding(11)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // CSM: Cascade shadow data (DirectionalShadowData) — dynamic UBO
            vk::DescriptorSetLayoutBinding::default()
                .binding(12)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
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

        // ---- Set 2: Shadow maps ----
        //
        // Binding 0: samplerCubeArray for point light shadow cube maps.
        // Binding 1: sampler2D for directional (sun) shadow map.
        let set2_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let shadow_maps = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&set2_bindings),
                None,
            )?
        };

        // ---- Set 3: Object SSBO (Phase 8A: replaces per-draw dynamic UBO) ----
        //
        // The persistent GpuObjectData SSBO is bound once per pass.
        // Vertex shaders index via gl_InstanceIndex (== firstInstance from indirect cmd).
        let set3_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)  // Changed from UNIFORM_BUFFER_DYNAMIC
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            );

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
            device.destroy_descriptor_set_layout(self.per_frame, None);
            // bindless_textures owned by TextureManager.
            device.destroy_descriptor_set_layout(self.shadow_maps, None);
            device.destroy_descriptor_set_layout(self.per_draw, None);
        }
    }
}

// ====================================================================
//  Render Passes
// ====================================================================

pub struct RenderPasses {
    pub depth_prepass: vk::RenderPass,
    pub lighting: vk::RenderPass,
    /// Shadow pass: depth-only into a single cube map face layer.
    pub shadow: vk::RenderPass,
    /// Probe capture pass: HDR color + depth into a cubemap face (Phase 3).
    pub probe_capture: vk::RenderPass,
}

impl RenderPasses {
    pub fn new(
        device: &Device,
        color_format: vk::Format,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let depth_prepass = Self::create_depth_prepass(device)?;
        let lighting = Self::create_lighting_pass(device, color_format)?;
        let shadow = Self::create_shadow_pass(device)?;
        let probe_capture = Self::create_probe_capture_pass(device)?;

        Ok(Self {
            depth_prepass,
            lighting,
            shadow,
            probe_capture,
        })
    }

    /// Depth pre-pass render pass with G-buffer normal output.
    ///
    /// Attachment 0: R16G16_SFLOAT normal (octahedral encoded view-space normal).
    ///   CLEAR + STORE, UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
    /// Attachment 1: D32_SFLOAT depth.
    ///   CLEAR + STORE, UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    fn create_depth_prepass(
        device: &Device,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        let attachments = [
            // Attachment 0: normal (R16G16_SFLOAT)
            vk::AttachmentDescription::default()
                .format(vk::Format::R16G16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            // Attachment 1: depth (D32_SFLOAT)
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
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
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );

        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));
        Ok(unsafe { device.create_render_pass(&info, None)? })
    }

    fn create_lighting_pass(
        device: &Device,
        color_format: vk::Format,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
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
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            );

        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));
        Ok(unsafe { device.create_render_pass(&info, None)? })
    }

    fn create_shadow_pass(
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
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref);

        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];

        let info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);
        Ok(unsafe { device.create_render_pass(&info, None)? })
    }

    /// Probe capture pass: HDR color + depth into a cubemap face (Phase 3).
    fn create_probe_capture_pass(
        device: &Device,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        let attachments = [
            // Color attachment: HDR R16G16B16A16_SFLOAT (must match gi.rs ProbeBakeTarget).
            vk::AttachmentDescription::default()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            // Depth attachment.
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
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
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));
        Ok(unsafe { device.create_render_pass(&info, None)? })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_render_pass(self.depth_prepass, None);
            device.destroy_render_pass(self.lighting, None);
            device.destroy_render_pass(self.shadow, None);
            device.destroy_render_pass(self.probe_capture, None);
        }
    }
}

// ====================================================================
//  Pipelines
// ====================================================================

pub struct Pipelines {
    pub depth_prepass: vk::Pipeline,
    pub lighting: vk::Pipeline,
    pub shadow: vk::Pipeline,
    pub cluster_compute: vk::Pipeline,
    /// Phase 3: Probe cubemap capture pipeline (full PBR, all-lights, HDR output).
    pub probe_capture: vk::Pipeline,
    /// Sun directional shadow pipeline (depth-only with hardware z, ortho projection).
    pub sun_shadow: vk::Pipeline,
    /// HDR skybox background pipeline (procedural cube, depth=1.0, no vertex buffer).
    pub skybox: vk::Pipeline,
    /// Graphics pipeline layout (all four sets + push constants).
    pub layout: vk::PipelineLayout,
    /// Compute pipeline layout (set 0 only).
    pub compute_layout: vk::PipelineLayout,
    pub cache: vk::PipelineCache,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        layouts: &DescriptorLayouts,
        passes: &RenderPasses,
        vert_spv: &[u8],
        frag_spv: &[u8],
        depth_vert_spv: &[u8],
        depth_frag_spv: &[u8],
        shadow_vert_spv: &[u8],
        shadow_frag_spv: &[u8],
        cluster_comp_spv: &[u8],
        probe_capture_frag_spv: &[u8],
        skybox_vert_spv: &[u8],
        skybox_frag_spv: &[u8],
        sun_shadow_vert_spv: &[u8],
        sun_shadow_frag_spv: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cache = unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default(),
                None,
            )?
        };

        // ---- Push constant range (shadow pass) ----
        let push_range = vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            )
            .offset(0)
            .size(std::mem::size_of::<ShadowPushConstants>() as u32);

        // ---- Graphics pipeline layout (all 4 sets + push constants) ----
        let set_layouts = layouts.all();
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )?
        };

        // ---- Compute pipeline layout (set 0 only) ----
        let compute_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&layouts.per_frame)),
                None,
            )?
        };

        // ---- Depth pre-pass pipeline (with G-buffer normal output) ----
        let depth_prepass = Self::create_depth_normal_pipeline(
            device, cache, layout, passes.depth_prepass,
            depth_vert_spv, depth_frag_spv,
        )?;

        // ---- Lighting pipeline ----
        let lighting = Self::create_lighting_pipeline(
            device, cache, layout, passes.lighting,
            vert_spv, frag_spv,
        )?;

        // ---- Shadow pipeline ----
        let shadow = Self::create_shadow_pipeline(
            device, cache, layout, passes.shadow,
            shadow_vert_spv, shadow_frag_spv,
        )?;

        // ---- Cluster compute pipeline ----
        let cluster_compute = Self::create_compute_pipeline(
            device, cache, compute_layout, cluster_comp_spv,
        )?;

        // ---- Probe capture pipeline (Phase 3) ----
        let probe_capture = Self::create_probe_capture_pipeline(
            device, cache, layout, passes.probe_capture,
            vert_spv, probe_capture_frag_spv,
        )?;

        // ---- Sun shadow pipeline (depth-only with hardware z, ortho) ----
        // Uses dedicated sun_shadow shaders - no color output, no validation warning.
        // Orthographic projection naturally produces linear depth ideal for shadow comparison.
        let sun_shadow = Self::create_depth_pipeline(
            device, cache, layout, passes.shadow,
            sun_shadow_vert_spv, sun_shadow_frag_spv,
        )?;

        // ---- Skybox pipeline ----
        let skybox = Self::create_skybox_pipeline(
            device, cache, layout, passes.lighting,
            skybox_vert_spv, skybox_frag_spv,
        )?;

        Ok(Self {
            depth_prepass,
            lighting,
            shadow,
            cluster_compute,
            probe_capture,
            sun_shadow,
            skybox,
            layout,
            compute_layout,
            cache,
        })
    }

    // ---- Pipeline builders ----

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

    fn create_depth_normal_pipeline(
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
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            // Color attachment 0: normal RG output.  Write R+G only.
            let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(
                    vk::ColorComponentFlags::R | vk::ColorComponentFlags::G,
                )
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

    /// Shadow pipeline: depth-only with push constants, no color output.
    /// Front-face NONE (no culling) to handle both faces of thin geometry.
    fn create_shadow_pipeline(
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
            // No backface culling for shadow maps (prevents light leaking).
            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(true)
                .depth_bias_constant_factor(1.25)
                .depth_bias_slope_factor(1.75);
            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
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

    /// Probe capture pipeline: simplified PBR for cubemap rendering (Phase 3).
    /// Uses the same vertex shader as lighting, but a simplified fragment shader.
    fn create_probe_capture_pipeline(
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
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
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

    /// Skybox pipeline: procedural cube, depth test only (no write), renders last.
    fn create_skybox_pipeline(
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

            // No vertex input — cube is procedural (gl_VertexIndex).
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::FRONT)       // We're inside the cube
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)                  // Don't write depth
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL); // Pass where depth=1.0 (cleared)
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

    fn create_compute_pipeline(
        device: &Device,
        cache: vk::PipelineCache,
        layout: vk::PipelineLayout,
        comp_spv: &[u8],
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let comp_module = create_shader_module(device, comp_spv)?;

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(comp_module)
                .name(c"main");

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);

            let pipelines = device
                .create_compute_pipelines(cache, &[pipeline_info], None)
                .map_err(|(_, e)| e)?;
            device.destroy_shader_module(comp_module, None);
            Ok(pipelines[0])
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.depth_prepass, None);
            device.destroy_pipeline(self.lighting, None);
            device.destroy_pipeline(self.shadow, None);
            device.destroy_pipeline(self.cluster_compute, None);
            device.destroy_pipeline(self.probe_capture, None);
            device.destroy_pipeline(self.sun_shadow, None);
            device.destroy_pipeline(self.skybox, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline_layout(self.compute_layout, None);
            device.destroy_pipeline_cache(self.cache, None);
        }
    }
}

// ====================================================================
//  Framebuffers (depth + lighting only; shadow FBs are in ShadowAtlas)
// ====================================================================

pub struct PassFramebuffers {
    pub depth_prepass: Vec<vk::Framebuffer>,
    pub lighting: Vec<vk::Framebuffer>,
}

impl PassFramebuffers {
    pub fn new(
        device: &Device,
        passes: &RenderPasses,
        swapchain_image_views: &[vk::ImageView],
        depth_view: vk::ImageView,
        normal_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut depth_prepass = Vec::new();
        let mut lighting = Vec::new();

        for &color_view in swapchain_image_views {
            // Depth pre-pass: attachment 0 = normal (R16G16_SFLOAT),
            //                  attachment 1 = depth (D32_SFLOAT).
            let depth_attach = [normal_view, depth_view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(passes.depth_prepass)
                .attachments(&depth_attach)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            depth_prepass.push(unsafe { device.create_framebuffer(&fb_info, None)? });

            let light_attach = [color_view, depth_view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(passes.lighting)
                .attachments(&light_attach)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            lighting.push(unsafe { device.create_framebuffer(&fb_info, None)? });
        }

        Ok(Self { depth_prepass, lighting })
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
//  Per-Frame Lighting Buffers
// ====================================================================

/// GPU buffers for the per-frame lighting data (one set per frame in flight).
pub struct FrameLightingBuffers {
    /// Light SSBO — HOST_COHERENT, memcpy'd each frame.
    pub light_ssbo_handles: Vec<BufferHandle>,
    pub light_ssbo_buffers: Vec<vk::Buffer>,
    pub light_ssbo_ptrs: Vec<NonNull<u8>>,
    pub light_ssbo_size: u64,

    /// Cluster SSBO — DEVICE_LOCAL, written by compute shader.
    pub cluster_ssbo_handles: Vec<BufferHandle>,
    pub cluster_ssbo_buffers: Vec<vk::Buffer>,
    pub cluster_ssbo_size: u64,

    /// Light index SSBO — DEVICE_LOCAL, written by compute shader.
    pub index_ssbo_handles: Vec<BufferHandle>,
    pub index_ssbo_buffers: Vec<vk::Buffer>,
    pub index_ssbo_size: u64,
}

impl FrameLightingBuffers {
    pub fn new(
        allocator: &mut GpuAllocator,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let light_ssbo_size = (std::mem::size_of::<LightSsboHeader>()
            + MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64;
        let cluster_ssbo_size =
            (TOTAL_CLUSTERS as usize * std::mem::size_of::<GpuCluster>()) as u64;
        let index_ssbo_size = (std::mem::size_of::<LightIndexHeader>()
            + LIGHT_INDEX_CAPACITY as usize * std::mem::size_of::<u32>()) as u64;

        let mut light_handles = Vec::new();
        let mut light_buffers = Vec::new();
        let mut light_ptrs = Vec::new();

        let mut cluster_handles = Vec::new();
        let mut cluster_buffers = Vec::new();

        let mut index_handles = Vec::new();
        let mut index_buffers = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            // Light SSBO: HOST_COHERENT for CPU writes.
            let alloc = allocator.create_buffer(
                light_ssbo_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            let ptr = alloc.mapped_ptr.ok_or("Light SSBO not mapped")?;
            light_handles.push(alloc.handle);
            light_buffers.push(alloc.buffer);
            light_ptrs.push(ptr);

            // Cluster SSBO: DEVICE_LOCAL, compute writes + fragment reads.
            let alloc = allocator.create_buffer(
                cluster_ssbo_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
            )?;
            cluster_handles.push(alloc.handle);
            cluster_buffers.push(alloc.buffer);

            // Light index SSBO: DEVICE_LOCAL, compute writes + fragment reads + CPU fill.
            let alloc = allocator.create_buffer(
                index_ssbo_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,  // <-- needed for vkCmdFillBuffer
                MemoryLocation::GpuOnly,
            )?;
            index_handles.push(alloc.handle);
            index_buffers.push(alloc.buffer);
        }

        println!(
            "[FrameLightingBuffers] Per-frame: light SSBO {:.1}K, cluster SSBO {:.1}K, index SSBO {:.1}K",
            light_ssbo_size as f64 / 1024.0,
            cluster_ssbo_size as f64 / 1024.0,
            index_ssbo_size as f64 / 1024.0,
        );

        Ok(Self {
            light_ssbo_handles: light_handles,
            light_ssbo_buffers: light_buffers,
            light_ssbo_ptrs: light_ptrs,
            light_ssbo_size,
            cluster_ssbo_handles: cluster_handles,
            cluster_ssbo_buffers: cluster_buffers,
            cluster_ssbo_size,
            index_ssbo_handles: index_handles,
            index_ssbo_buffers: index_buffers,
            index_ssbo_size,
        })
    }

    /// Upload light data for the current frame (CPU → HOST_COHERENT memcpy).
    pub fn upload_lights(&self, frame: usize, data: &[u8]) {
        let max = self.light_ssbo_size as usize;
        let len = data.len().min(max);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.light_ssbo_ptrs[frame].as_ptr(),
                len,
            );
        }
    }

    /// Zero-allocation light SSBO upload.  Writes the 16-byte header
    /// and the `GpuLight` array directly into the HOST_COHERENT mapped
    /// pointer — no intermediate `Vec<u8>` needed.
    ///
    /// Replaces `upload_lights(&self.light_manager.ssbo_bytes())` on the
    /// hot path.
    pub fn upload_lights_direct(
        &self,
        frame: usize,
        header: &LightSsboHeader,
        lights: &[GpuLight],
    ) {
        let dst = self.light_ssbo_ptrs[frame].as_ptr();
        let max = self.light_ssbo_size as usize;
        let header_bytes = std::mem::size_of::<LightSsboHeader>();
        let body_bytes = (lights.len() * std::mem::size_of::<GpuLight>()).min(max - header_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(
                header as *const LightSsboHeader as *const u8,
                dst,
                header_bytes,
            );
            if body_bytes > 0 {
                std::ptr::copy_nonoverlapping(
                    lights.as_ptr() as *const u8,
                    dst.add(header_bytes),
                    body_bytes,
                );
            }
        }
    }

    pub fn destroy(&self, allocator: &mut GpuAllocator) {
        for &h in &self.light_ssbo_handles {
            allocator.free_buffer(h);
        }
        for &h in &self.cluster_ssbo_handles {
            allocator.free_buffer(h);
        }
        for &h in &self.index_ssbo_handles {
            allocator.free_buffer(h);
        }
    }
}

// ====================================================================
//  Per-Frame Descriptor Sets
// ====================================================================

/// Per-frame descriptor sets.
///
/// Phase 8A: `per_draw_sets` removed — the object SSBO descriptor set is now
/// owned and managed by `GpuCullResources` instead of here.
pub struct FrameDescriptors {
    pub pool: vk::DescriptorPool,
    pub per_frame_sets: Vec<vk::DescriptorSet>,
    pub shadow_map_sets: Vec<vk::DescriptorSet>,
    // Phase 8A: per_draw_sets removed — now managed by GpuCullResources
}

impl FrameDescriptors {
    /// Create per-frame descriptor sets.
    ///
    /// Phase 8A: `per_draw_ubo_range` parameter removed — the per-draw dynamic
    /// UBO is replaced by the persistent object SSBO managed by GpuCullResources.
    pub fn new(
        device: &Device,
        layouts: &DescriptorLayouts,
        ring_buffer: vk::Buffer,
        global_ubo_range: u64,
        cluster_params_range: u64,
        // Phase 8A: per_draw_ubo_range removed
        material_ssbo: vk::Buffer,
        material_ssbo_size: u64,
        lighting: &FrameLightingBuffers,
        shadow_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        // CSM: cascade shadow (set 2, binding 1) — sampler2DArrayShadow
        cascade_shadow_view: vk::ImageView,
        cascade_shadow_sampler: vk::Sampler,
        // CSM: cascade shadow UBO (set 0, binding 12)
        cascade_shadow_ubo_range: u64,
        // Phase 3: GI resources
        probe_ssbo: vk::Buffer,
        probe_ssbo_size: u64,
        probe_grid_params_range: u64,
        brdf_lut_view: vk::ImageView,
        brdf_lut_sampler: vk::Sampler,
        irradiance_view: vk::ImageView,
        irradiance_sampler: vk::Sampler,
        prefiltered_view: vk::ImageView,
        prefiltered_sampler: vk::Sampler,
        ao_screen_view: vk::ImageView,
        ao_screen_sampler: vk::Sampler,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n = MAX_FRAMES_IN_FLIGHT as u32;

        // Phase 8A: Pool sizes updated — removed per-draw UNIFORM_BUFFER_DYNAMIC
        let pool_sizes = [
            // Set 0: 4× UNIFORM_BUFFER_DYNAMIC (global + cluster + probe grid + cascade shadow)
            //       + 5× STORAGE_BUFFER (light, cluster, index, material, probe)
            //       + 4× COMBINED_IMAGE_SAMPLER (brdf, irradiance, prefiltered, ao) per frame
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(n * 4),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n * 5),
            // Set 0 GI samplers (4) + Set 2 shadow samplers (2: cube + sun)
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(n * 6),
            // Phase 8A: Removed per-draw UNIFORM_BUFFER_DYNAMIC pool size
            // (object SSBO now managed by GpuCullResources)
        ];

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(n * 2), // N per-frame + N shadow (per_draw removed)
                None,
            )?
        };

        // Allocate per-frame sets (set 0).
        let per_frame_layouts = vec![layouts.per_frame; MAX_FRAMES_IN_FLIGHT];
        let per_frame_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&per_frame_layouts),
            )?
        };

        // Allocate shadow map sets (set 2).
        let shadow_layouts = vec![layouts.shadow_maps; MAX_FRAMES_IN_FLIGHT];
        let shadow_map_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&shadow_layouts),
            )?
        };

        // Phase 8A: per_draw_sets allocation removed — now in GpuCullResources

        // Write descriptors.
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            // ---- Set 0 ----
            let global_ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(ring_buffer)
                .offset(0)
                .range(global_ubo_range);

            let light_ssbo_info = vk::DescriptorBufferInfo::default()
                .buffer(lighting.light_ssbo_buffers[i])
                .offset(0)
                .range(lighting.light_ssbo_size);

            let cluster_ssbo_info = vk::DescriptorBufferInfo::default()
                .buffer(lighting.cluster_ssbo_buffers[i])
                .offset(0)
                .range(lighting.cluster_ssbo_size);

            let index_ssbo_info = vk::DescriptorBufferInfo::default()
                .buffer(lighting.index_ssbo_buffers[i])
                .offset(0)
                .range(lighting.index_ssbo_size);

            let cluster_params_info = vk::DescriptorBufferInfo::default()
                .buffer(ring_buffer)
                .offset(0)
                .range(cluster_params_range);

            let material_info = vk::DescriptorBufferInfo::default()
                .buffer(material_ssbo)
                .offset(0)
                .range(material_ssbo_size);

            // Phase 3: GI descriptor infos.
            let probe_ssbo_info = vk::DescriptorBufferInfo::default()
                .buffer(probe_ssbo)
                .offset(0)
                .range(probe_ssbo_size);

            let probe_grid_params_info = vk::DescriptorBufferInfo::default()
                .buffer(ring_buffer)
                .offset(0)
                .range(probe_grid_params_range);

            let brdf_lut_info = vk::DescriptorImageInfo::default()
                .sampler(brdf_lut_sampler)
                .image_view(brdf_lut_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let irradiance_info = vk::DescriptorImageInfo::default()
                .sampler(irradiance_sampler)
                .image_view(irradiance_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let prefiltered_info = vk::DescriptorImageInfo::default()
                .sampler(prefiltered_sampler)
                .image_view(prefiltered_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            
            let ao_screen_info = vk::DescriptorImageInfo::default()
                .sampler(ao_screen_sampler)
                .image_view(ao_screen_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            // CSM: cascade shadow UBO info (set 0, binding 12)
            let cascade_shadow_ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(ring_buffer)
                .offset(0)
                .range(cascade_shadow_ubo_range);

            let set0_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&global_ubo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&light_ssbo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&cluster_ssbo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&index_ssbo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&cluster_params_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&material_info)),
                // Phase 3: GI writes
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&probe_ssbo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&probe_grid_params_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(8)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&brdf_lut_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(9)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&irradiance_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(10)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&prefiltered_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(11)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&ao_screen_info)),
                // CSM: cascade shadow data (binding 12, dynamic UBO)
                vk::WriteDescriptorSet::default()
                    .dst_set(per_frame_sets[i])
                    .dst_binding(12)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&cascade_shadow_ubo_info)),
            ];
            unsafe { device.update_descriptor_sets(&set0_writes, &[]) };

            // ---- Set 2: Shadow maps ----
            let shadow_info = vk::DescriptorImageInfo::default()
                .sampler(shadow_sampler)
                .image_view(shadow_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            // CSM: 2D array + comparison sampler (sampler2DArrayShadow in shader)
            let cascade_shadow_img_info = vk::DescriptorImageInfo::default()
                .sampler(cascade_shadow_sampler)
                .image_view(cascade_shadow_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let shadow_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_map_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&shadow_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(shadow_map_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&cascade_shadow_img_info)),
            ];
            unsafe { device.update_descriptor_sets(&shadow_writes, &[]) };

            // Phase 8A: per_draw_sets writing removed — now in GpuCullResources
        }

        Ok(Self {
            pool,
            per_frame_sets,
            shadow_map_sets,
            // Phase 8A: per_draw_sets removed
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}