use ash::{vk, Device};
use crate::device::DeviceContext;
use crate::memory::{MemoryContext, MAX_FRAMES_IN_FLIGHT};
use crate::scene::{
    Scene, Vertex, UniformBufferObject, ChunkLoadState, ChunkCoord,
    MAX_STREAM_STARTS_PER_FRAME,
};

pub struct Renderer {
    device: Device,
    memory_ctx: MemoryContext,

    // Scene
    scene: Scene,

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
    global_frame: u64,
}

impl Renderer {
    pub fn new(
        device_ctx: &DeviceContext,
        memory_ctx: MemoryContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = device_ctx.device.clone();

        let aspect = device_ctx.swapchain_extent.width as f32
            / device_ctx.swapchain_extent.height as f32;
        let scene = Scene::new(aspect);

        // All chunks start Unloaded.  The render loop will stream them in
        // over the first few frames via upload_async.

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

        // ---- descriptors ----

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

        println!(
            "[Renderer] Initialized.  {} chunks pending stream.",
            scene.chunks.len(),
        );

        Ok(Self {
            device,
            memory_ctx,
            scene,
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
            global_frame: 0,
        })
    }

    // ================================================================
    //  Streaming: initiate + poll + evict
    // ================================================================

    /// Kick off async uploads for up to N unloaded chunks this frame.
    fn begin_streaming_chunks(&mut self) {
        let coords = self.scene.unloaded_chunks_by_distance();
        let to_start: Vec<ChunkCoord> = coords
            .into_iter()
            .take(MAX_STREAM_STARTS_PER_FRAME)
            .collect();

        for coord in to_start {
            // Flatten this chunk's mesh data into contiguous vertex and
            // index arrays.
            let chunk = &self.scene.chunks[&coord];
            let mut all_verts: Vec<Vertex> = Vec::new();
            let mut all_indices: Vec<u32> = Vec::new();
            for mesh in &chunk.meshes {
                all_verts.extend_from_slice(&mesh.vertices);
                all_indices.extend_from_slice(&mesh.indices);
            }

            if all_verts.is_empty() {
                // Empty chunk — mark Ready immediately.
                if let Some(c) = self.scene.chunks.get_mut(&coord) {
                    c.load_state = ChunkLoadState::Ready;
                }
                continue;
            }

            // Upload vertex data.
            let vresult = self.memory_ctx.upload_async_typed(
                &all_verts,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            );

            let (valloc, vticket) = match vresult {
                Ok(v) => v,
                Err(e) => {
                    println!("[Renderer] Vertex upload failed for ({},{}): {}", coord.0, coord.1, e);
                    continue;
                }
            };

            // Upload index data.
            let iresult = self.memory_ctx.upload_async_typed(
                &all_indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            );

            let (ialloc, iticket) = match iresult {
                Ok(v) => v,
                Err(e) => {
                    // Clean up the vertex allocation we already made.
                    self.memory_ctx.allocator.free_buffer(valloc.handle);
                    println!("[Renderer] Index upload failed for ({},{}): {}", coord.0, coord.1, e);
                    continue;
                }
            };

            // Store handles + VkBuffers on the chunk.
            if let Some(c) = self.scene.chunks.get_mut(&coord) {
                c.vertex_handle = Some(valloc.handle);
                c.index_handle = Some(ialloc.handle);
                c.vertex_vk_buffer = Some(valloc.buffer);
                c.index_vk_buffer = Some(ialloc.buffer);
                c.load_state = ChunkLoadState::Streaming {
                    vertex_ticket: vticket.clone(),
                    index_ticket: iticket.clone(),
                };

                // Register with budget tracker.
                self.memory_ctx.budget.track(valloc.handle, valloc.size, self.global_frame);
                self.memory_ctx.budget.track(ialloc.handle, ialloc.size, self.global_frame);
            }
        }
    }

    /// Promote completed Streaming → Ready.
    fn poll_streaming_chunks(&mut self) {
        let to_promote: Vec<ChunkCoord> = self
            .scene
            .chunks
            .iter()
            .filter_map(|(&coord, chunk)| {
                if let ChunkLoadState::Streaming {
                    ref vertex_ticket,
                    ref index_ticket,
                } = chunk.load_state
                {
                    let done = self.memory_ctx.transfer.is_complete(vertex_ticket)
                        && self.memory_ctx.transfer.is_complete(index_ticket);
                    done.then_some(coord)
                } else {
                    None
                }
            })
            .collect();

        for coord in to_promote {
            if let Some(chunk) = self.scene.chunks.get_mut(&coord) {
                chunk.load_state = ChunkLoadState::Ready;
                println!(
                    "[Renderer] Chunk ({},{}) → Ready  (frame {})",
                    coord.0, coord.1, self.global_frame,
                );
            }
        }
    }

    /// Evict over-budget chunks.
    fn run_eviction(&mut self) {
        let pool_usage = self.memory_ctx.allocator.total_used();
        let evicted = self.memory_ctx.budget.evict_lru(
            pool_usage,
            self.global_frame,
            120, // ~2 sec at 60 fps
        );

        for handle in evicted {
            for chunk in self.scene.chunks.values_mut() {
                let owns = chunk.vertex_handle == Some(handle)
                    || chunk.index_handle == Some(handle);
                if owns {
                    if let Some(vh) = chunk.vertex_handle.take() {
                        self.memory_ctx.allocator.free_buffer(vh);
                        self.memory_ctx.budget.untrack(vh);
                    }
                    if let Some(ih) = chunk.index_handle.take() {
                        self.memory_ctx.allocator.free_buffer(ih);
                        self.memory_ctx.budget.untrack(ih);
                    }
                    chunk.vertex_vk_buffer = None;
                    chunk.index_vk_buffer = None;
                    chunk.load_state = ChunkLoadState::Unloaded;
                    println!(
                        "[Renderer] Evicted chunk ({},{})",
                        chunk.coord.0, chunk.coord.1,
                    );
                    break;
                }
            }
        }
    }

    // ================================================================
    //  Frame rendering
    // ================================================================

    pub fn render(
        &mut self,
        device_ctx: &DeviceContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.global_frame += 1;

        // Pre-frame bookkeeping.
        self.begin_streaming_chunks();
        self.poll_streaming_chunks();
        self.run_eviction();

        // (Stats logged after frustum cull below.)

        unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            )?;

            self.scene.update(0.016);
            self.memory_ctx.ring.begin_frame(self.current_frame);

            let (image_index, _) = device_ctx.swapchain_loader.acquire_next_image(
                device_ctx.swapchain,
                u64::MAX,
                self.image_available[self.current_frame],
                vk::Fence::null(),
            )?;

            self.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            let cmd = self.command_buffers[self.current_frame];
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.05, 0.05, 0.08, 1.0],
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

            // ---- frustum cull + draw per-chunk ----

            let frustum = self.scene.camera.extract_frustum_planes();
            let view_mat = self.scene.camera.get_view_matrix();
            let proj_mat = self.scene.camera.get_projection_matrix();

            // Collect drawable coords (Ready + visible).
            let drawable: Vec<ChunkCoord> = self
                .scene
                .chunks
                .iter()
                .filter(|(_, c)| c.is_ready() && c.is_visible(&frustum))
                .map(|(&coord, _)| coord)
                .collect();

            // ---- per-frame frustum cull stats (every 60 frames ≈ 1 sec) ----
            if self.global_frame % 60 == 0 {
                let total = self.scene.chunks.len();
                let ready = self.scene.chunks.values().filter(|c| c.is_ready()).count();
                let streaming = self.scene.chunks.values()
                    .filter(|c| matches!(c.load_state, ChunkLoadState::Streaming { .. })).count();
                let unloaded = self.scene.chunks.values().filter(|c| c.is_unloaded()).count();
                let visible = drawable.len();
                let culled = ready - visible;

                // Collect the coords that were culled for display.
                let culled_coords: Vec<ChunkCoord> = self
                    .scene
                    .chunks
                    .iter()
                    .filter(|(_, c)| c.is_ready() && !c.is_visible(&frustum))
                    .map(|(&coord, _)| coord)
                    .collect();

                let cam = self.scene.camera.position;
                let tgt = self.scene.camera.target;
                let dir = [tgt[0] - cam[0], tgt[1] - cam[1], tgt[2] - cam[2]];
                let heading_deg = dir[2].atan2(dir[0]).to_degrees();

                println!(
                    "[Frame {:>5}] heading: {:>6.1}°  visible: {:>2}/{}  culled: {:>2}  streaming: {}  unloaded: {}",
                    self.global_frame,
                    heading_deg,
                    visible,
                    total,
                    culled,
                    streaming,
                    unloaded,
                );
                if !culled_coords.is_empty() && culled_coords.len() <= 20 {
                    let coords_str: Vec<String> = culled_coords.iter()
                        .map(|(x, z)| format!("({},{})", x, z))
                        .collect();
                    println!("           culled: {}", coords_str.join(" "));
                }
            }

            let mut chunks_drawn = 0u32;
            let mut meshes_drawn = 0u32;

            for coord in &drawable {
                let chunk = &self.scene.chunks[coord];

                let vk_vb = match chunk.vertex_vk_buffer {
                    Some(b) => b,
                    None => continue,
                };
                let vk_ib = match chunk.index_vk_buffer {
                    Some(b) => b,
                    None => continue,
                };

                // Touch LRU.
                if let Some(vh) = chunk.vertex_handle {
                    self.memory_ctx.budget.touch(vh, self.global_frame);
                }
                if let Some(ih) = chunk.index_handle {
                    self.memory_ctx.budget.touch(ih, self.global_frame);
                }

                // Bind this chunk's vertex + index buffers.
                self.device.cmd_bind_vertex_buffers(cmd, 0, &[vk_vb], &[0]);
                self.device.cmd_bind_index_buffer(cmd, vk_ib, 0, vk::IndexType::UINT32);

                // Draw each mesh with its own model transform.
                let mut vertex_offset: i32 = 0;
                let mut index_offset: u32 = 0;

                for mesh in &chunk.meshes {
                    let ubo = UniformBufferObject {
                        model: mesh.transform,
                        view: view_mat,
                        proj: proj_mat,
                    };

                    let ring_slice = self
                        .memory_ctx
                        .ring
                        .push_data(&ubo)
                        .expect("Ring buffer overflow – increase RING_BUFFER_SIZE");

                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.descriptor_sets[self.current_frame]],
                        &[ring_slice.offset as u32],
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
                    meshes_drawn += 1;
                }

                chunks_drawn += 1;
            }

            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            // Submit.
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

            // Present.
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
    //  Descriptor setup
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
                &vk::ShaderModuleCreateInfo::default().code(&vert_code), None,
            )?;
            let frag_module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code), None,
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

            let binding = vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX);
            let attributes = [
                vk::VertexInputAttributeDescription::default()
                    .binding(0).location(0)
                    .format(vk::Format::R32G32B32_SFLOAT).offset(0),
                vk::VertexInputAttributeDescription::default()
                    .binding(0).location(1)
                    .format(vk::Format::R32G32B32_SFLOAT).offset(12),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(&attributes);

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1).scissor_count(1);

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

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

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
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
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
                .attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let depth_ref = vk::AttachmentReference::default()
                .attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::slice::from_ref(&color_ref))
                .depth_stencil_attachment(&depth_ref);
            let dependency = vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL).dst_subpass(0)
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
        image_views.iter().map(|&view| {
            let attachments = [view, depth_view];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width).height(extent.height).layers(1);
            unsafe { device.create_framebuffer(&info, None) }
        }).collect::<Result<Vec<_>, _>>().map_err(Into::into)
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

fn fmt_pool_usage(mem: &MemoryContext) -> String {
    let used = mem.allocator.total_used();
    let total = mem.allocator.total_allocated();
    let bufs = mem.allocator.live_buffer_count();
    format!("{}/{} ({} bufs)", fmt_bytes(used), fmt_bytes(total), bufs)
}

fn fmt_bytes(bytes: u64) -> String {
    const MB: u64 = 1024 * 1024;
    const KB: u64 = 1024;
    if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ====================================================================
//  Cleanup
// ====================================================================

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            for &f in &self.in_flight_fences {
                self.device.destroy_fence(f, None);
            }
            for &s in &self.render_finished {
                self.device.destroy_semaphore(s, None);
            }
            for &s in &self.image_available {
                self.device.destroy_semaphore(s, None);
            }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }

            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            // All chunk VkBuffers, ring buffer, staging belt, transfer
            // queue destroyed when memory_ctx drops.
        }
    }
}