use ash::{vk, Device};
use crate::device::DeviceContext;
use crate::material::{MaterialLibrary, MaterialSsbo, MaterialData};
use crate::memory::{MemoryContext, MAX_FRAMES_IN_FLIGHT};
use crate::pipeline::{
    DescriptorLayouts, FrameDescriptors, PassFramebuffers, Pipelines,
    RenderPasses,
};
use crate::scene::{
    Scene, Vertex, UniformBufferObject, ChunkLoadState, ChunkCoord,
    MAX_STREAM_STARTS_PER_FRAME,
};
use crate::texture::TextureManager;

/// Per-frame global UBO uploaded to set 0 binding 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct GlobalUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 4], // xyz = position, w = time
}

pub struct Renderer {
    device: Device,
    memory_ctx: MemoryContext,

    // Scene
    scene: Scene,

    // Material system
    material_library: MaterialLibrary,
    material_ssbo: MaterialSsbo,

    // Texture system
    texture_manager: TextureManager,

    // Pipeline infrastructure
    descriptor_layouts: DescriptorLayouts,
    render_passes: RenderPasses,
    pipelines: Pipelines,
    framebuffers: PassFramebuffers,
    frame_descriptors: FrameDescriptors,

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
        mut memory_ctx: MemoryContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = device_ctx.device.clone();

        let aspect = device_ctx.swapchain_extent.width as f32
            / device_ctx.swapchain_extent.height as f32;
        let scene = Scene::new(aspect);

        // ---- Material system ----

        let mut material_library = MaterialLibrary::new();

        // Register a few default materials for worldgen objects.
        material_library.add(
            "ground",
            MaterialData {
                roughness: 0.9,
                metallic: 0.0,
                ..Default::default()
            },
        );
        material_library.add(
            "object",
            MaterialData {
                roughness: 0.5,
                metallic: 0.1,
                ..Default::default()
            },
        );

        let material_ssbo = MaterialSsbo::new(&mut memory_ctx)?;
        material_ssbo.upload(&material_library);
        material_library.clear_dirty();

        // ---- Texture manager ----

        let texture_manager = TextureManager::new(
            &device,
            &mut memory_ctx,
            device_ctx.command_pool,
            device_ctx.queue,
        )?;

        // ---- Descriptor layouts (uses texture manager's layout for set 1) ----

        let descriptor_layouts = DescriptorLayouts::new(
            &device,
            texture_manager.descriptor_set_layout,
        )?;

        // ---- Render passes ----

        let render_passes = RenderPasses::new(
            &device,
            device_ctx.surface_format.format,
        )?;

        // ---- Shaders & Pipelines ----

        let vert_spv = include_bytes!("../shaders/compiled/basic.vert.spv");
        let frag_spv = include_bytes!("../shaders/compiled/basic.frag.spv");

        // Depth pre-pass uses dedicated minimal shaders for maximum
        // early-Z throughput.  The depth fragment shader is empty —
        // only the automatic depth write matters.
        let depth_vert_spv: &[u8] = include_bytes!("../shaders/compiled/depth.vert.spv");
        let depth_frag_spv: &[u8] = include_bytes!("../shaders/compiled/depth.frag.spv");

        let pipelines = Pipelines::new(
            &device,
            &descriptor_layouts,
            &render_passes,
            vert_spv,
            frag_spv,
            depth_vert_spv,
            depth_frag_spv,
        )?;

        // ---- Framebuffers ----

        let framebuffers = PassFramebuffers::new(
            &device,
            &render_passes,
            &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view,
            device_ctx.swapchain_extent,
        )?;

        // ---- Frame descriptors (set 0 + set 3) ----

        let ubo_size = std::mem::size_of::<UniformBufferObject>() as u64;
        let material_ssbo_size =
            (material_library.count() * std::mem::size_of::<MaterialData>()) as u64;

        let frame_descriptors = FrameDescriptors::new(
            &device,
            &descriptor_layouts,
            memory_ctx.ring.buffer,
            ubo_size,
            material_ssbo.buffer,
            material_ssbo_size.max(128),
        )?;

        // ---- Command buffers ----

        let command_buffers =
            Self::allocate_command_buffers(&device, device_ctx.command_pool)?;

        // ---- Sync objects ----

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
            "[Renderer] Initialized.  {} chunks pending stream.  {} materials registered.",
            scene.chunks.len(),
            material_library.count(),
        );

        Ok(Self {
            device,
            memory_ctx,
            scene,
            material_library,
            material_ssbo,
            texture_manager,
            descriptor_layouts,
            render_passes,
            pipelines,
            framebuffers,
            frame_descriptors,
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

    fn begin_streaming_chunks(&mut self) {
        let coords = self.scene.unloaded_chunks_by_distance();
        let to_start: Vec<ChunkCoord> = coords
            .into_iter()
            .take(MAX_STREAM_STARTS_PER_FRAME)
            .collect();

        for coord in to_start {
            let chunk = &self.scene.chunks[&coord];
            let mut all_verts: Vec<Vertex> = Vec::new();
            let mut all_indices: Vec<u32> = Vec::new();
            for mesh in &chunk.meshes {
                all_verts.extend_from_slice(&mesh.vertices);
                all_indices.extend_from_slice(&mesh.indices);
            }

            if all_verts.is_empty() {
                if let Some(c) = self.scene.chunks.get_mut(&coord) {
                    c.load_state = ChunkLoadState::Ready;
                }
                continue;
            }

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

            let iresult = self.memory_ctx.upload_async_typed(
                &all_indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            );

            let (ialloc, iticket) = match iresult {
                Ok(v) => v,
                Err(e) => {
                    self.memory_ctx.allocator.free_buffer(valloc.handle);
                    println!("[Renderer] Index upload failed for ({},{}): {}", coord.0, coord.1, e);
                    continue;
                }
            };

            if let Some(c) = self.scene.chunks.get_mut(&coord) {
                c.vertex_handle = Some(valloc.handle);
                c.index_handle = Some(ialloc.handle);
                c.vertex_vk_buffer = Some(valloc.buffer);
                c.index_vk_buffer = Some(ialloc.buffer);
                c.load_state = ChunkLoadState::Streaming {
                    vertex_ticket: vticket.clone(),
                    index_ticket: iticket.clone(),
                };

                self.memory_ctx.budget.track(valloc.handle, valloc.size, self.global_frame);
                self.memory_ctx.budget.track(ialloc.handle, ialloc.size, self.global_frame);
            }
        }
    }

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

    fn run_eviction(&mut self) {
        let pool_usage = self.memory_ctx.allocator.total_used();
        let evicted = self.memory_ctx.budget.evict_lru(
            pool_usage,
            self.global_frame,
            120,
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
    //  Frame rendering — multi-pass
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
        self.texture_manager.poll_pending(&self.memory_ctx);

        // Re-upload materials if dirty.
        if self.material_library.is_dirty() {
            self.material_ssbo.upload(&self.material_library);
            self.material_library.clear_dirty();
        }

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

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: device_ctx.swapchain_extent.width as f32,
                height: device_ctx.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: device_ctx.swapchain_extent,
            };

            // ---- Frustum cull ----

            let frustum = self.scene.camera.extract_frustum_planes();
            let view_mat = self.scene.camera.get_view_matrix();
            let proj_mat = self.scene.camera.get_projection_matrix();

            let drawable: Vec<ChunkCoord> = self
                .scene
                .chunks
                .iter()
                .filter(|(_, c)| c.is_ready() && c.is_visible(&frustum))
                .map(|(&coord, _)| coord)
                .collect();

            // ---- Stats (every 60 frames) ----

            if self.global_frame % 60 == 0 {
                let total = self.scene.chunks.len();
                let ready = self.scene.chunks.values().filter(|c| c.is_ready()).count();
                let streaming = self.scene.chunks.values()
                    .filter(|c| matches!(c.load_state, ChunkLoadState::Streaming { .. })).count();
                let unloaded = self.scene.chunks.values().filter(|c| c.is_unloaded()).count();
                let visible = drawable.len();
                let culled = ready.saturating_sub(visible);

                println!(
                    "[Frame {:>5}] visible: {:>2}/{}  culled: {:>2}  streaming: {}  unloaded: {}  textures: {}/{}",
                    self.global_frame,
                    visible, total, culled, streaming, unloaded,
                    self.texture_manager.active_count(),
                    self.texture_manager.pending_count(),
                );
            }

            // ============================================================
            //  PASS 1: Depth Pre-Pass
            // ============================================================

            let depth_clear = [vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }];

            let depth_rp = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_passes.depth_prepass)
                .framebuffer(self.framebuffers.depth_prepass[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: device_ctx.swapchain_extent,
                })
                .clear_values(&depth_clear);

            self.device
                .cmd_begin_render_pass(cmd, &depth_rp, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.depth_prepass,
            );
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Bind set 0 (per-frame globals) and set 1 (bindless textures)
            // for the depth prepass.  Even though the depth shader only
            // needs the model/view/proj from set 3, the pipeline layout
            // declares all four sets and the shader may statically
            // reference set 0 bindings.
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0, // first set
                &[
                    self.frame_descriptors.per_frame_sets[self.current_frame],
                    self.texture_manager.descriptor_set,
                ],
                &[],
            );

            // Draw all visible chunks (depth only).
            for coord in &drawable {
                let chunk = &self.scene.chunks[coord];
                let vk_vb = match chunk.vertex_vk_buffer { Some(b) => b, None => continue };
                let vk_ib = match chunk.index_vk_buffer { Some(b) => b, None => continue };

                self.device.cmd_bind_vertex_buffers(cmd, 0, &[vk_vb], &[0]);
                self.device.cmd_bind_index_buffer(cmd, vk_ib, 0, vk::IndexType::UINT32);

                let mut vertex_offset: i32 = 0;
                let mut index_offset: u32 = 0;

                for mesh in &chunk.meshes {
                    let ubo = UniformBufferObject::new(
                        mesh.transform,
                        view_mat,
                        proj_mat,
                        mesh.material_id,
                    );

                    let ring_slice = self
                        .memory_ctx
                        .ring
                        .push_data(&ubo)
                        .expect("Ring buffer overflow – increase RING_BUFFER_SIZE");

                    // Bind set 3 (per-draw dynamic UBO) with dynamic offset.
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.layout,
                        3, // set index
                        &[self.frame_descriptors.per_draw_sets[self.current_frame]],
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
                }
            }

            self.device.cmd_end_render_pass(cmd);

            // ============================================================
            //  PASS 2: Lighting Pass
            // ============================================================

            let light_clear = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.05, 0.05, 0.08, 1.0],
                    },
                },
                // Depth not cleared — loaded from pre-pass.
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let light_rp = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_passes.lighting)
                .framebuffer(self.framebuffers.lighting[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: device_ctx.swapchain_extent,
                })
                .clear_values(&light_clear);

            self.device
                .cmd_begin_render_pass(cmd, &light_rp, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.lighting,
            );
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Bind set 0 (per-frame globals).
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[self.frame_descriptors.per_frame_sets[self.current_frame]],
                &[],
            );

            // Bind set 1 (bindless textures).
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                1,
                &[self.texture_manager.descriptor_set],
                &[],
            );

            // Draw all visible chunks (full shading).
            let mut meshes_drawn = 0u32;

            for coord in &drawable {
                let chunk = &self.scene.chunks[coord];
                let vk_vb = match chunk.vertex_vk_buffer { Some(b) => b, None => continue };
                let vk_ib = match chunk.index_vk_buffer { Some(b) => b, None => continue };

                // Touch LRU.
                if let Some(vh) = chunk.vertex_handle {
                    self.memory_ctx.budget.touch(vh, self.global_frame);
                }
                if let Some(ih) = chunk.index_handle {
                    self.memory_ctx.budget.touch(ih, self.global_frame);
                }

                self.device.cmd_bind_vertex_buffers(cmd, 0, &[vk_vb], &[0]);
                self.device.cmd_bind_index_buffer(cmd, vk_ib, 0, vk::IndexType::UINT32);

                let mut vertex_offset: i32 = 0;
                let mut index_offset: u32 = 0;

                for mesh in &chunk.meshes {
                    let ubo = UniformBufferObject::new(
                        mesh.transform,
                        view_mat,
                        proj_mat,
                        mesh.material_id,
                    );

                    let ring_slice = self
                        .memory_ctx
                        .ring
                        .push_data(&ubo)
                        .expect("Ring buffer overflow – increase RING_BUFFER_SIZE");

                    // Bind set 3 (per-draw).
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.layout,
                        3,
                        &[self.frame_descriptors.per_draw_sets[self.current_frame]],
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
            }

            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            // ---- Submit ----

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

            // ---- Present ----

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
        unsafe { self.device.device_wait_idle()? };

        self.framebuffers.destroy(&self.device);
        self.framebuffers = PassFramebuffers::new(
            &self.device,
            &self.render_passes,
            &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view,
            device_ctx.swapchain_extent,
        )?;

        self.scene.camera.update_aspect(
            device_ctx.swapchain_extent.width,
            device_ctx.swapchain_extent.height,
        );

        Ok(())
    }

    // ================================================================
    //  Public accessors for material / texture systems
    // ================================================================

    /// Get a mutable reference to the material library.
    pub fn materials_mut(&mut self) -> &mut MaterialLibrary {
        &mut self.material_library
    }

    pub fn materials(&self) -> &MaterialLibrary {
        &self.material_library
    }

    pub fn texture_manager_mut(&mut self) -> &mut TextureManager {
        &mut self.texture_manager
    }

    // ================================================================
    //  Helpers
    // ================================================================

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

            self.frame_descriptors.destroy(&self.device);
            self.framebuffers.destroy(&self.device);
            self.pipelines.destroy(&self.device);
            self.render_passes.destroy(&self.device);
            self.descriptor_layouts.destroy(&self.device);

            // TextureManager::drop destroys its descriptor pool, layout, sampler.
            // MaterialSsbo is cleaned up via the allocator.
            // MemoryContext::drop handles ring buffer, staging, transfer queue.
            // GpuAllocator::drop handles all pool blocks and live buffers/images.
        }
    }
}