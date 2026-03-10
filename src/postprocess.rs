//! Phase 6 §6.1: Horizon-Based Ambient Occlusion
//!
//! `HbaoPass` owns all GPU resources for the HBAO compute pass and
//! its separable bilateral blur.  Integrates between the depth pre-pass
//! and the cluster-assign compute pass in the render graph.
//!
//! Resources owned:
//! - AO output image (`R8_UNORM`, full resolution)
//! - AO temp image (intermediate for separable blur)
//! - HBAO compute pipeline + descriptor set + layout
//! - Blur compute pipeline + descriptor sets (H/V) + layout
//! - Params UBO (host-coherent, updated on resize)
//! - Depth sampler (for sampling the depth pre-pass output)
//! - AO sampler (for PBR fragment shader to read final AO)
//!
//! The final AO image view + sampler are exposed for binding to
//! set 0, binding 11 (`COMBINED_IMAGE_SAMPLER`) in the PBR pass.

use ash::{vk, Device};
use crate::memory::{GpuAllocator, MemoryContext, MemoryLocation, BufferHandle};
use crate::pipeline::create_shader_module;

// ====================================================================
//  Constants
// ====================================================================

/// HBAO default parameters.
const HBAO_RADIUS: f32 = 0.5;       // World-space sampling radius (metres)
const HBAO_BIAS: f32 = 0.1;         // Angle bias to reduce self-occlusion
const HBAO_INTENSITY: f32 = 1.0;    // AO strength multiplier (was 1.5)
const HBAO_MAX_DISTANCE: f32 = 1.5; // Maximum influence distance (was 2.0)

/// Bilateral blur sharpness (depth-weight exponent).
const BLUR_SHARPNESS: f32 = 300.0;  // was 1000.0 — softer edge transitions

// ====================================================================
//  UBO layout — must match hbao.comp HBAOParams
// ====================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct HbaoParamsUbo {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    resolution: [f32; 4], // xy: screen size, zw: 1/screen size
    params: [f32; 4],     // x: radius, y: bias, z: intensity, w: max_distance
}

// ====================================================================
//  Push-constant layout — must match hbao_blur.comp BlurParams
// ====================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct BlurPushConstants {
    direction: [f32; 2],
    sharpness: f32,
    _pad: f32,
    resolution: [f32; 4], // xy: screen size, zw: 1/screen size
}

// ====================================================================
//  HbaoPass
// ====================================================================

pub struct HbaoPass {
    // ---- AO images ----
    ao_image: vk::Image,
    ao_memory: vk::DeviceMemory,
    /// Storage image view (R8, GENERAL layout) for compute write.
    ao_storage_view: vk::ImageView,

    ao_temp_image: vk::Image,
    ao_temp_memory: vk::DeviceMemory,
    ao_temp_storage_view: vk::ImageView,

    /// Sampled view for PBR fragment shader (binding 11).
    pub ao_sampled_view: vk::ImageView,
    /// Sampler for the AO result.
    pub ao_sampler: vk::Sampler,

    // ---- Depth sampling ----
    depth_sampler: vk::Sampler,

    // ---- Params UBO ----
    params_handle: BufferHandle,
    params_buffer: vk::Buffer,
    params_ptr: std::ptr::NonNull<u8>,

    // ---- HBAO compute pipeline ----
    hbao_pipeline: vk::Pipeline,
    hbao_pipeline_layout: vk::PipelineLayout,
    hbao_set_layout: vk::DescriptorSetLayout,
    hbao_set: vk::DescriptorSet,

    // ---- Blur compute pipeline ----
    blur_pipeline: vk::Pipeline,
    blur_pipeline_layout: vk::PipelineLayout,
    blur_set_layout: vk::DescriptorSetLayout,
    /// Horizontal blur: reads ao_image, writes ao_temp.
    blur_h_set: vk::DescriptorSet,
    /// Vertical blur: reads ao_temp, writes ao_image.
    blur_v_set: vk::DescriptorSet,

    // ---- Common ----
    descriptor_pool: vk::DescriptorPool,
    width: u32,
    height: u32,
}

impl HbaoPass {
    /// Create a new HBAO pass.  Must be called after the depth image
    /// is created (with `SAMPLED` usage) and after shader SPVs are
    /// available.
    ///
    /// `depth_image_view` — the engine's shared depth image view
    ///   (D32_SFLOAT, DEPTH aspect).
    /// `normal_image_view` — G-buffer view-space normal (R16G16_SFLOAT,
    ///   octahedral encoded, written by depth pre-pass).
    /// `proj` / `inv_proj` — camera projection matrices.
    pub fn new(
        device: &Device,
        allocator: &mut GpuAllocator,
        depth_image_view: vk::ImageView,
        normal_image_view: vk::ImageView,
        extent: vk::Extent2D,
        proj: [[f32; 4]; 4],
        inv_proj: [[f32; 4]; 4],
        hbao_comp_spv: &[u8],
        blur_comp_spv: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let w = extent.width;
        let h = extent.height;

        // ---- Samplers ----

        let depth_sampler = unsafe {
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

        let ao_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )?
        };

        // ---- AO images ----

        let (ao_image, ao_memory, ao_storage_view, ao_sampled_view) =
            Self::create_ao_image(device, allocator, w, h)?;
        let (ao_temp_image, ao_temp_memory, ao_temp_storage_view, ao_temp_sampled) =
            Self::create_ao_image(device, allocator, w, h)?;
        unsafe { device.destroy_image_view(ao_temp_sampled, None) };

        // ---- Params UBO ----

        let ubo_size = std::mem::size_of::<HbaoParamsUbo>() as u64;
        let alloc = allocator.create_buffer(
            ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        let params_handle = alloc.handle;
        let params_buffer = alloc.buffer;
        let params_ptr = alloc.mapped_ptr.ok_or("HBAO UBO not mapped")?;

        // Write initial params.
        Self::write_params(params_ptr, w, h, proj, inv_proj);

        // ---- Descriptor pool ----
        // Need: 3 sets total (hbao, blur_h, blur_v)
        // HBAO set:   1 COMBINED_IMAGE_SAMPLER (depth) + 1 STORAGE_IMAGE (ao)
        //           + 1 UBO + 1 COMBINED_IMAGE_SAMPLER (normal)
        // Blur H set: 2 STORAGE_IMAGE (in, out) + 1 COMBINED_IMAGE_SAMPLER (depth)
        // Blur V set: 2 STORAGE_IMAGE (in, out) + 1 COMBINED_IMAGE_SAMPLER (depth)
        // Totals:     4 COMBINED_IMAGE_SAMPLER, 5 STORAGE_IMAGE, 1 UNIFORM_BUFFER

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(4),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(5),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(3),
                None,
            )?
        };

        // ---- HBAO descriptor set layout ----
        // binding 0: sampler2D depthTexture    (COMBINED_IMAGE_SAMPLER)
        // binding 1: image2D  aoTexture        (STORAGE_IMAGE)
        // binding 2: uniform  HBAOParams       (UNIFORM_BUFFER)
        // binding 3: sampler2D normalTexture   (COMBINED_IMAGE_SAMPLER)

        let hbao_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let hbao_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&hbao_bindings),
                None,
            )?
        };

        // ---- Blur descriptor set layout ----
        // binding 0: image2D inputAO     (STORAGE_IMAGE, readonly)
        // binding 1: image2D outputAO    (STORAGE_IMAGE, writeonly)
        // binding 2: sampler2D depth     (COMBINED_IMAGE_SAMPLER)

        let blur_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let blur_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&blur_bindings),
                None,
            )?
        };

        // ---- Allocate descriptor sets ----

        let hbao_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&hbao_set_layout)),
            )?[0]
        };

        let blur_layouts = [blur_set_layout, blur_set_layout];
        let blur_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&blur_layouts),
            )?
        };
        let blur_h_set = blur_sets[0];
        let blur_v_set = blur_sets[1];

        // ---- Write descriptors ----

        Self::write_hbao_descriptors(
            device, hbao_set,
            depth_image_view, depth_sampler,
            normal_image_view, depth_sampler, // reuse NEAREST sampler for normals
            ao_storage_view,
            params_buffer, std::mem::size_of::<HbaoParamsUbo>() as u64,
        );
        Self::write_blur_descriptors(
            device, blur_h_set,
            ao_storage_view, ao_temp_storage_view,
            depth_image_view, depth_sampler,
        );
        Self::write_blur_descriptors(
            device, blur_v_set,
            ao_temp_storage_view, ao_storage_view,
            depth_image_view, depth_sampler,
        );

        // ---- Pipeline layouts ----

        let hbao_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&hbao_set_layout)),
                None,
            )?
        };

        let blur_push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<BlurPushConstants>() as u32);

        let blur_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&blur_set_layout))
                    .push_constant_ranges(std::slice::from_ref(&blur_push_range)),
                None,
            )?
        };

        // ---- Compute pipelines ----

        let hbao_pipeline = Self::create_compute_pipeline(
            device, hbao_pipeline_layout, hbao_comp_spv,
        )?;
        let blur_pipeline = Self::create_compute_pipeline(
            device, blur_pipeline_layout, blur_comp_spv,
        )?;

        println!(
            "[HbaoPass] Initialized: {}×{} AO texture, radius={:.2}m, \
             bias={:.2}, intensity={:.1}",
            w, h, HBAO_RADIUS, HBAO_BIAS, HBAO_INTENSITY,
        );

        Ok(Self {
            ao_image, ao_memory, ao_storage_view,
            ao_temp_image, ao_temp_memory, ao_temp_storage_view,
            ao_sampled_view, ao_sampler,
            depth_sampler,
            params_handle, params_buffer, params_ptr,
            hbao_pipeline, hbao_pipeline_layout, hbao_set_layout, hbao_set,
            blur_pipeline, blur_pipeline_layout, blur_set_layout,
            blur_h_set, blur_v_set,
            descriptor_pool,
            width: w, height: h,
        })
    }

    // ================================================================
    //  Command recording — called from Renderer::render()
    // ================================================================

    /// Record HBAO compute + bilateral blur into `cmd`.
    ///
    /// **Pre-condition:** depth image is in
    /// `DEPTH_STENCIL_READ_ONLY_OPTIMAL` layout (caller inserts the
    /// barrier after the depth pre-pass ends).  Normal image is in
    /// `SHADER_READ_ONLY_OPTIMAL` (caller transitions after depth pass).
    ///
    /// **Post-condition:** AO image is in `SHADER_READ_ONLY_OPTIMAL`,
    /// ready for the PBR fragment shader.  Depth image remains in
    /// `DEPTH_STENCIL_READ_ONLY_OPTIMAL`; caller transitions it back
    /// to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` before the lighting pass.
    pub unsafe fn dispatch(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
    ) {
        let groups_x = (self.width + 7) / 8;
        let groups_y = (self.height + 7) / 8;

        // ---- Transition AO images → GENERAL for compute write ----

        let ao_barriers = [
            vk::ImageMemoryBarrier::default()
                .image(self.ao_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }),
            vk::ImageMemoryBarrier::default()
                .image(self.ao_temp_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }),
        ];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[], &[], &ao_barriers,
        );

        // ---- HBAO compute dispatch ----

        device.cmd_bind_pipeline(
            cmd, vk::PipelineBindPoint::COMPUTE, self.hbao_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE,
            self.hbao_pipeline_layout, 0,
            std::slice::from_ref(&self.hbao_set), &[],
        );
        device.cmd_dispatch(cmd, groups_x, groups_y, 1);

        // ---- Barrier: HBAO write → blur H read ----

        let ao_read_barrier = vk::ImageMemoryBarrier::default()
            .image(self.ao_image)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            });
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[], &[], std::slice::from_ref(&ao_read_barrier),
        );

        // ---- Blur horizontal: ao_image → ao_temp ----

        let res = [
            self.width as f32, self.height as f32,
            1.0 / self.width as f32, 1.0 / self.height as f32,
        ];
        let blur_h_push = BlurPushConstants {
            direction: [1.0, 0.0],
            sharpness: BLUR_SHARPNESS,
            _pad: 0.0,
            resolution: res,
        };

        device.cmd_bind_pipeline(
            cmd, vk::PipelineBindPoint::COMPUTE, self.blur_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE,
            self.blur_pipeline_layout, 0,
            std::slice::from_ref(&self.blur_h_set), &[],
        );
        device.cmd_push_constants(
            cmd, self.blur_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE, 0,
            std::slice::from_raw_parts(
                &blur_h_push as *const _ as *const u8,
                std::mem::size_of::<BlurPushConstants>(),
            ),
        );
        device.cmd_dispatch(cmd, groups_x, groups_y, 1);

        // ---- Barrier: blur H write → blur V read ----
        // ao_temp: write → read.  ao_image: read → write.

        let blur_hv_barriers = [
            vk::ImageMemoryBarrier::default()
                .image(self.ao_temp_image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }),
            vk::ImageMemoryBarrier::default()
                .image(self.ao_image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }),
        ];
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[], &[], &blur_hv_barriers,
        );

        // ---- Blur vertical: ao_temp → ao_image ----

        let blur_v_push = BlurPushConstants {
            direction: [0.0, 1.0],
            sharpness: BLUR_SHARPNESS,
            _pad: 0.0,
            resolution: res,
        };

        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE,
            self.blur_pipeline_layout, 0,
            std::slice::from_ref(&self.blur_v_set), &[],
        );
        device.cmd_push_constants(
            cmd, self.blur_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE, 0,
            std::slice::from_raw_parts(
                &blur_v_push as *const _ as *const u8,
                std::mem::size_of::<BlurPushConstants>(),
            ),
        );
        device.cmd_dispatch(cmd, groups_x, groups_y, 1);

        // ---- Final transition: ao_image → SHADER_READ_ONLY ----

        let ao_final_barrier = vk::ImageMemoryBarrier::default()
            .image(self.ao_image)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            });
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[], &[], std::slice::from_ref(&ao_final_barrier),
        );
    }

    // ================================================================
    //  Resize — called on swapchain recreate
    // ================================================================

    /// Recreate AO images and update descriptors for new resolution.
    /// The depth and normal image views are passed because they are
    /// also recreated during swapchain resize.
    pub fn resize(
        &mut self,
        device: &Device,
        allocator: &mut GpuAllocator,
        depth_image_view: vk::ImageView,
        normal_image_view: vk::ImageView,
        extent: vk::Extent2D,
        proj: [[f32; 4]; 4],
        inv_proj: [[f32; 4]; 4],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { device.device_wait_idle()? };

        let w = extent.width;
        let h = extent.height;

        // Destroy old AO images.
        Self::destroy_ao_image(device, self.ao_image, self.ao_memory,
                               self.ao_storage_view, self.ao_sampled_view);
        Self::destroy_ao_image(device, self.ao_temp_image, self.ao_temp_memory,
                               self.ao_temp_storage_view, vk::ImageView::null());

        // Recreate.
        let (ai, am, asv, asmv) = Self::create_ao_image(device, allocator, w, h)?;
        self.ao_image = ai;
        self.ao_memory = am;
        self.ao_storage_view = asv;
        self.ao_sampled_view = asmv;

        let (ti, tm, tsv, temp_sampled) = Self::create_ao_image(device, allocator, w, h)?;
        unsafe { device.destroy_image_view(temp_sampled, None) };
        self.ao_temp_image = ti;
        self.ao_temp_memory = tm;
        self.ao_temp_storage_view = tsv;

        self.width = w;
        self.height = h;

        // Update UBO.
        Self::write_params(self.params_ptr, w, h, proj, inv_proj);

        // Re-write descriptors with new views.
        Self::write_hbao_descriptors(
            device, self.hbao_set,
            depth_image_view, self.depth_sampler,
            normal_image_view, self.depth_sampler,
            self.ao_storage_view,
            self.params_buffer,
            std::mem::size_of::<HbaoParamsUbo>() as u64,
        );
        Self::write_blur_descriptors(
            device, self.blur_h_set,
            self.ao_storage_view, self.ao_temp_storage_view,
            depth_image_view, self.depth_sampler,
        );
        Self::write_blur_descriptors(
            device, self.blur_v_set,
            self.ao_temp_storage_view, self.ao_storage_view,
            depth_image_view, self.depth_sampler,
        );

        println!("[HbaoPass] Resized to {}×{}", w, h);
        Ok(())
    }

    // ================================================================
    //  Depth barrier helpers — called from Renderer
    // ================================================================

    /// Transition depth from DEPTH_STENCIL_ATTACHMENT → READ_ONLY
    /// for HBAO sampling.  Call after depth pre-pass ends.
    pub unsafe fn barrier_depth_to_read(
        device: &Device,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .image(depth_image)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            });
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[], &[], std::slice::from_ref(&barrier),
        );
    }

    /// Transition depth from READ_ONLY → DEPTH_STENCIL_ATTACHMENT
    /// for the lighting pass.  Call after HBAO dispatch completes.
    pub unsafe fn barrier_depth_to_attachment(
        device: &Device,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .image(depth_image)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            });
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::DependencyFlags::empty(),
            &[], &[], std::slice::from_ref(&barrier),
        );
    }

    // ================================================================
    //  Cleanup
    // ================================================================

    pub fn destroy(&mut self, device: &Device, allocator: &mut GpuAllocator) {
        unsafe {
            device.destroy_pipeline(self.hbao_pipeline, None);
            device.destroy_pipeline(self.blur_pipeline, None);
            device.destroy_pipeline_layout(self.hbao_pipeline_layout, None);
            device.destroy_pipeline_layout(self.blur_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.hbao_set_layout, None);
            device.destroy_descriptor_set_layout(self.blur_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_sampler(self.depth_sampler, None);
            device.destroy_sampler(self.ao_sampler, None);
        }
        allocator.free_buffer(self.params_handle);
        Self::destroy_ao_image(
            device, self.ao_image, self.ao_memory,
            self.ao_storage_view, self.ao_sampled_view,
        );
        Self::destroy_ao_image(
            device, self.ao_temp_image, self.ao_temp_memory,
            self.ao_temp_storage_view, vk::ImageView::null(),
        );
    }

    // ================================================================
    //  Internal helpers
    // ================================================================

    /// Create a full-resolution R8_UNORM image for AO storage.
    /// Returns (image, memory, storage_view, sampled_view).
    fn create_ao_image(
        device: &Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::ImageView), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED,
            )
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

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        };

        let storage_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .subresource_range(subresource),
                None,
            )?
        };

        let sampled_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .subresource_range(subresource),
                None,
            )?
        };

        Ok((image, memory, storage_view, sampled_view))
    }

    fn destroy_ao_image(
        device: &Device,
        image: vk::Image,
        memory: vk::DeviceMemory,
        storage_view: vk::ImageView,
        sampled_view: vk::ImageView,
    ) {
        unsafe {
            device.destroy_image_view(storage_view, None);
            if sampled_view != vk::ImageView::null() {
                device.destroy_image_view(sampled_view, None);
            }
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
    }

    fn write_params(
        ptr: std::ptr::NonNull<u8>,
        w: u32, h: u32,
        proj: [[f32; 4]; 4],
        inv_proj: [[f32; 4]; 4],
    ) {
        let ubo = HbaoParamsUbo {
            proj,
            inv_proj,
            resolution: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
            params: [HBAO_RADIUS, HBAO_BIAS, HBAO_INTENSITY, HBAO_MAX_DISTANCE],
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &ubo as *const _ as *const u8,
                ptr.as_ptr(),
                std::mem::size_of::<HbaoParamsUbo>(),
            );
        }
    }

    fn write_hbao_descriptors(
        device: &Device,
        set: vk::DescriptorSet,
        depth_view: vk::ImageView,
        depth_sampler: vk::Sampler,
        normal_view: vk::ImageView,
        normal_sampler: vk::Sampler,
        ao_storage_view: vk::ImageView,
        params_buffer: vk::Buffer,
        params_size: u64,
    ) {
        let depth_info = vk::DescriptorImageInfo::default()
            .sampler(depth_sampler)
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        let normal_info = vk::DescriptorImageInfo::default()
            .sampler(normal_sampler)
            .image_view(normal_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let ao_info = vk::DescriptorImageInfo::default()
            .image_view(ao_storage_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(params_buffer)
            .offset(0)
            .range(params_size);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&ao_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&normal_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn write_blur_descriptors(
        device: &Device,
        set: vk::DescriptorSet,
        input_view: vk::ImageView,
        output_view: vk::ImageView,
        depth_view: vk::ImageView,
        depth_sampler: vk::Sampler,
    ) {
        let input_info = vk::DescriptorImageInfo::default()
            .image_view(input_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let output_info = vk::DescriptorImageInfo::default()
            .image_view(output_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let depth_info = vk::DescriptorImageInfo::default()
            .sampler(depth_sampler)
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn create_compute_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        spv: &[u8],
    ) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let module = create_shader_module(device, spv)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipelines = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?;
            device.destroy_shader_module(module, None);
            Ok(pipelines[0])
        }
    }
}