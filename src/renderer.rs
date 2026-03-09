use ash::{vk, Device};
use crate::device::DeviceContext;
use crate::memory::{BufferAllocation, MemoryContext, MAX_FRAMES_IN_FLIGHT};
use crate::scene::{Scene, Vertex, UniformBufferObject};

pub struct Renderer {
    device: Device,
    memory_ctx: MemoryContext,

    // Scene
    scene: Scene,

    // Pool-allocated GPU buffers (lifetime = entire renderer)
    vertex_buffer: BufferAllocation,
    index_buffer: BufferAllocation,

    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Render pass & framebuffers
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    // Command buffers
    command_buffers: Vec<vk::CommandBuffer>,

    // Synchronisation
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    current_frame: usize,
}

impl Renderer {
    pub fn new(
        device_ctx: &DeviceContext,
        mut memory_ctx: MemoryContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = device_ctx.device.clone();

        // Scene
        let aspect = device_ctx.swapchain_extent.width as f32
            / device_ctx.swapchain_extent.height as f32;
        let scene = Scene::new(aspect);

        // ---- upload mesh data via the pool allocator + staging belt ----

        let mut all_vertices: Vec<Vertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        for mesh in &scene.meshes {
            all_vertices.extend_from_slice(&mesh.vertices);
            all_indices.extend_from_slice(&mesh.indices);
        }

        let vertex_buffer = memory_ctx.create_typed_buffer(
            &all_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            device_ctx.command_pool,
            device_ctx.queue,
        )?;

        let index_buffer = memory_ctx.create_typed_buffer(
            &all_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
            device_ctx.command_pool,
            device_ctx.queue,
        )?;

        // ---- pipeline ----

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        let render_pass =
            Self::create_render_pass(&device, device_ctx.surface_format.format)?;
        let (pipeline, pipeline_layout) =
            Self::create_pipeline(&device, render_pass, descriptor_set_layout)?;

        // ---- framebuffers ----

        let framebuffers = Self::create_framebuffers(
            &device,
            render_pass,
            &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view,
            device_ctx.swapchain_extent,
        )?;

        // ---- descriptors (UNIFORM_BUFFER_DYNAMIC → ring buffer) ----

        let (descriptor_pool, descriptor_sets) =
            Self::create_descriptor_sets(&device, descriptor_set_layout, &memory_ctx)?;

        // ---- command buffers ----

        let command_buffers =
            Self::allocate_command_buffers(&device, device_ctx.command_pool)?;

        // ---- sync objects ----

        let mut image_available = Vec::new();
        let mut render_finished = Vec::new();
        let mut in_flight_fences = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available.push(
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
                render_finished.push(
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
                in_flight_fences.push(device.create_fence(
                    &vk::FenceCreateInfo::default()
                        .flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
            }
        }

        memory_ctx.allocator.print_stats();

        Ok(Self {
            device,
            memory_ctx,
            scene,
            vertex_buffer,
            index_buffer,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            render_pass,
            framebuffers,
            command_buffers,
            image_available,
            render_finished,
            in_flight_fences,
            current_frame: 0,
        })
    }

    // ================================================================
    //  Frame rendering
    // ================================================================

    pub fn render(
        &mut self,
        device_ctx: &DeviceContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            // 1. Wait for the in-flight fence of *this* frame slot.
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            )?;

            // 2. Advance scene.
            self.scene.update(0.016);

            // 3. Reset the ring-buffer cursor for this frame.  The fence
            //    wait above guarantees the GPU is done reading the old data
            //    in this segment.
            self.memory_ctx.ring.begin_frame(self.current_frame);

            // 4. Acquire the next swapchain image.
            let (image_index, _) = device_ctx.swapchain_loader.acquire_next_image(
                device_ctx.swapchain,
                u64::MAX,
                self.image_available[self.current_frame],
                vk::Fence::null(),
            )?;

            self.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            // 5. Record commands.
            let cmd = self.command_buffers[self.current_frame];
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: device_ctx.swapchain_extent,
                })
                .clear_values(&clear_values);

            self.device
                .cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            // Viewport & scissor (dynamic state).
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: device_ctx.swapchain_extent.width as f32,
                height: device_ctx.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: device_ctx.swapchain_extent,
            };
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Bind vertex + index buffers once for the entire frame.
            self.device.cmd_bind_vertex_buffers(
                cmd,
                0,
                &[self.vertex_buffer.buffer],
                &[0],
            );
            self.device.cmd_bind_index_buffer(
                cmd,
                self.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            // ---- per-mesh draws via ring-buffer dynamic offsets ----

            let mut vertex_offset: i32 = 0;
            let mut index_offset: u32 = 0;

            for mesh in &self.scene.meshes {
                let ubo = UniformBufferObject {
                    model: mesh.transform,
                    view: self.scene.camera.get_view_matrix(),
                    proj: self.scene.camera.get_projection_matrix(),
                };

                // Push this draw's UBO into the ring buffer.  Each mesh
                // lands at a different, correctly-aligned offset inside the
                // same VkBuffer – no descriptor rewrite needed.
                let ring_slice = self
                    .memory_ctx
                    .ring
                    .push_data(&ubo)
                    .expect("Ring buffer overflow – increase RING_BUFFER_SIZE");

                // Bind the descriptor set with a dynamic offset pointing at
                // this draw's UBO region.
                let dynamic_offset = ring_slice.offset as u32;
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[self.current_frame]],
                    &[dynamic_offset],
                );

                self.device.cmd_draw_indexed(
                    cmd,
                    mesh.indices.len() as u32,
                    1,
                    index_offset,
                    vertex_offset,
                    0,
                );

                vertex_offset += mesh.vertices.len() as i32;
                index_offset += mesh.indices.len() as u32;
            }

            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            // 6. Submit.
            let wait_sems = [self.image_available[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_sems = [self.render_finished[self.current_frame]];
            let cmd_bufs = [cmd];

            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(&signal_sems);

            self.device.queue_submit(
                device_ctx.queue,
                &[submit],
                self.in_flight_fences[self.current_frame],
            )?;

            // 7. Present.
            let swapchains = [device_ctx.swapchain];
            let image_indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_sems)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            device_ctx
                .swapchain_loader
                .queue_present(device_ctx.queue, &present)?;

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        Ok(())
    }

    // ================================================================
    //  Swapchain recreation
    // ================================================================

    pub fn recreate_framebuffers(
        &mut self,
        device_ctx: &DeviceContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.device_wait_idle()?;

            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }

            self.framebuffers = Self::create_framebuffers(
                &self.device,
                self.render_pass,
                &device_ctx.swapchain_image_views,
                device_ctx.depth_image_view,
                device_ctx.swapchain_extent,
            )?;

            self.scene.camera.update_aspect(
                device_ctx.swapchain_extent.width,
                device_ctx.swapchain_extent.height,
            );
        }
        Ok(())
    }

    // ================================================================
    //  Descriptor setup  (UNIFORM_BUFFER_DYNAMIC)
    // ================================================================

    fn create_descriptor_set_layout(
        device: &Device,
    ) -> Result<vk::DescriptorSetLayout, Box<dyn std::error::Error>> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&binding));

        unsafe { Ok(device.create_descriptor_set_layout(&info, None)?) }
    }

    fn create_descriptor_sets(
        device: &Device,
        layout: vk::DescriptorSetLayout,
        memory_ctx: &MemoryContext,
    ) -> Result<(vk::DescriptorPool, Vec<vk::DescriptorSet>), Box<dyn std::error::Error>>
    {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let layouts = vec![layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        // Each set points at the ring buffer with offset 0 and
        // range = sizeof(UBO).  The *actual* offset per draw is supplied
        // as a dynamic offset in cmd_bind_descriptor_sets.
        for &set in &sets {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(memory_ctx.ring.buffer)
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as u64);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(std::slice::from_ref(&buf_info));

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) }
        }

        Ok((pool, sets))
    }

    // ================================================================
    //  Pipeline
    // ================================================================

    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn std::error::Error>> {
        unsafe {
            let vert_spv = include_bytes!("../shaders/compiled/basic.vert.spv");
            let frag_spv = include_bytes!("../shaders/compiled/basic.frag.spv");

            let vert_code = align_shader_code(vert_spv);
            let frag_code = align_shader_code(frag_spv);

            let vert_module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vert_code),
                None,
            )?;
            let frag_module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code),
                None,
            )?;

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

            // Vertex input
            let binding = vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX);

            let attributes = [
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(0),
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(12),
            ];

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
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false);

            let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false);

            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&blend_attachment));

            let dynamic_states =
                [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            let pipeline_layout =
                device.create_pipeline_layout(&layout_info, None)?;

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
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipelines = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            Ok((pipelines[0], pipeline_layout))
        }
    }

    // ================================================================
    //  Render pass & framebuffers
    // ================================================================

    fn create_render_pass(
        device: &Device,
        format: vk::Format,
    ) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        unsafe {
            let attachments = [
                vk::AttachmentDescription::default()
                    .format(format)
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
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags::empty())
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );

            let rp_info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(std::slice::from_ref(&subpass))
                .dependencies(std::slice::from_ref(&dependency));

            Ok(device.create_render_pass(&rp_info, None)?)
        }
    }

    fn create_framebuffers(
        device: &Device,
        render_pass: vk::RenderPass,
        image_views: &[vk::ImageView],
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>, Box<dyn std::error::Error>> {
        image_views
            .iter()
            .map(|&view| {
                let attachments = [view, depth_view];
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn allocate_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
        unsafe { Ok(device.allocate_command_buffers(&info)?) }
    }
}

// ====================================================================
//  Helpers
// ====================================================================

fn align_shader_code(code: &[u8]) -> Vec<u32> {
    code.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ====================================================================
//  Cleanup
// ====================================================================

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Synchronisation primitives.
            for &f in &self.in_flight_fences {
                self.device.destroy_fence(f, None);
            }
            for &s in &self.render_finished {
                self.device.destroy_semaphore(s, None);
            }
            for &s in &self.image_available {
                self.device.destroy_semaphore(s, None);
            }

            // Descriptors.
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Framebuffers.
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }

            // Pipeline.
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            // vertex_buffer, index_buffer, ring buffer, staging belt, and
            // all pool VkDeviceMemory are destroyed when `memory_ctx` drops
            // (which happens automatically after this method returns).
        }
    }
}