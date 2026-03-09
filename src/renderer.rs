use ash::{vk, Device};
use std::collections::HashMap;

use crate::device::DeviceContext;
use crate::light::{
    self, cube_face_matrices, ClusterParamsUbo, Light, LightManager, LightType,
    ShadowAtlas, ShadowBudgetManager, ShadowPushConstants, CLUSTER_X, CLUSTER_Y,
    CLUSTER_Z, MAX_SHADOW_SLOTS, SHADOW_MAP_SIZE, TOTAL_CLUSTERS,
};
use crate::material::{MaterialData, MaterialLibrary, MaterialSsbo};
use crate::memory::{MemoryContext, MAX_FRAMES_IN_FLIGHT};
use crate::pipeline::{
    DescriptorLayouts, FrameDescriptors, FrameLightingBuffers, PassFramebuffers,
    Pipelines, RenderPasses,
};
use crate::scene::{
    ChunkCoord, ChunkLoadState, Scene, UniformBufferObject, Vertex,
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

    // Lighting system (Phase 2)
    light_manager: LightManager,
    shadow_budget: ShadowBudgetManager,
    shadow_atlas: ShadowAtlas,
    lighting_buffers: FrameLightingBuffers,
    /// Per-frame shadow assignments: global light index → shadow slot.
    shadow_assignments: HashMap<usize, u32>,

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
        //
        // Phase 2 PBR showcase palette — 12 materials (IDs 0–11).
        // Must match the IDs assigned in worldgen.rs.
        //
        //   0  default (white PBR, auto-created by MaterialLibrary::new)
        //   1  ground          earthy, rough dielectric
        //   2  polished_metal  silver, low roughness, full metallic
        //   3  rough_stone     gray, high roughness dielectric
        //   4  copper          orange-brown, metallic
        //   5  ceramic_red     red, glossy dielectric
        //   6  ceramic_blue    blue, glossy dielectric
        //   7  gold            gold, metallic
        //   8  rubber          dark, very rough dielectric
        //   9  marble          whitish, medium roughness
        //  10  emissive_warm   warm glow
        //  11  emissive_cool   cool glow

        let mut material_library = MaterialLibrary::new();

        // ID 1: ground
        material_library.add(
            "ground",
            MaterialData {
                base_color: [0.35, 0.28, 0.18, 1.0],
                roughness: 0.92,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 2: polished_metal (silver)
        material_library.add(
            "polished_metal",
            MaterialData {
                base_color: [0.95, 0.93, 0.88, 1.0],
                roughness: 0.08,
                metallic: 1.0,
                ..Default::default()
            },
        );

        // ID 3: rough_stone
        material_library.add(
            "rough_stone",
            MaterialData {
                base_color: [0.55, 0.52, 0.50, 1.0],
                roughness: 0.95,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 4: copper
        material_library.add(
            "copper",
            MaterialData {
                base_color: [0.95, 0.64, 0.54, 1.0],
                roughness: 0.25,
                metallic: 1.0,
                ..Default::default()
            },
        );

        // ID 5: ceramic_red
        material_library.add(
            "ceramic_red",
            MaterialData {
                base_color: [0.85, 0.15, 0.12, 1.0],
                roughness: 0.15,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 6: ceramic_blue
        material_library.add(
            "ceramic_blue",
            MaterialData {
                base_color: [0.12, 0.35, 0.85, 1.0],
                roughness: 0.15,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 7: gold
        material_library.add(
            "gold",
            MaterialData {
                base_color: [1.0, 0.76, 0.33, 1.0],
                roughness: 0.18,
                metallic: 1.0,
                ..Default::default()
            },
        );

        // ID 8: rubber (dark)
        material_library.add(
            "rubber",
            MaterialData {
                base_color: [0.12, 0.12, 0.14, 1.0],
                roughness: 0.98,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 9: marble
        material_library.add(
            "marble",
            MaterialData {
                base_color: [0.92, 0.90, 0.85, 1.0],
                roughness: 0.35,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 10: emissive_warm
        material_library.add(
            "emissive_warm",
            MaterialData {
                base_color: [1.0, 0.85, 0.4, 1.0],
                emissive: [1.0, 0.7, 0.2, 8.0], // rgb + intensity
                roughness: 0.5,
                metallic: 0.0,
                ..Default::default()
            },
        );

        // ID 11: emissive_cool
        material_library.add(
            "emissive_cool",
            MaterialData {
                base_color: [0.4, 0.7, 1.0, 1.0],
                emissive: [0.3, 0.5, 1.0, 8.0],
                roughness: 0.5,
                metallic: 0.0,
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

        // ---- Lighting system ----

        let mut light_manager = LightManager::new();

        // Register a directional sun light (not tied to any chunk).
        // Intensity 10.0 compensates for the Cook-Torrance 1/pi diffuse divisor.
        // Direction pitched steeply downward for good ground coverage.
        // light_manager.register(
        //     (i32::MAX, i32::MAX), // sentinel coord for global lights
        //     vec![Light::directional(
        //         [0.3, -0.9, 0.2],   // steep downward angle
        //         [1.0, 0.95, 0.9],   // warm sunlight
        //         10.0,                // Phase 2: bright enough for PBR
        //     )],
        // );

        let shadow_budget = ShadowBudgetManager::new();

        // ---- Descriptor layouts ----

        let descriptor_layouts = DescriptorLayouts::new(
            &device,
            texture_manager.descriptor_set_layout,
        )?;

        // ---- Render passes ----

        let render_passes = RenderPasses::new(
            &device,
            device_ctx.surface_format.format,
        )?;

        // ---- Shadow atlas ----

        let shadow_atlas = ShadowAtlas::new(
            &device,
            &mut memory_ctx.allocator,
            render_passes.shadow,
            device_ctx.command_pool,
            device_ctx.queue,
        )?;

        // ---- Lighting buffers ----

        let lighting_buffers = FrameLightingBuffers::new(&mut memory_ctx.allocator)?;

        // ---- Shaders & Pipelines ----

        let vert_spv = include_bytes!("../shaders/compiled/basic.vert.spv");
        let frag_spv = include_bytes!("../shaders/compiled/basic.frag.spv");
        let depth_vert_spv: &[u8] = include_bytes!("../shaders/compiled/depth.vert.spv");
        let depth_frag_spv: &[u8] = include_bytes!("../shaders/compiled/depth.frag.spv");
        let shadow_vert_spv: &[u8] = include_bytes!("../shaders/compiled/shadow.vert.spv");
        let shadow_frag_spv: &[u8] = include_bytes!("../shaders/compiled/shadow.frag.spv");
        let cluster_comp_spv: &[u8] =
            include_bytes!("../shaders/compiled/cluster_assign.comp.spv");

        let pipelines = Pipelines::new(
            &device,
            &descriptor_layouts,
            &render_passes,
            vert_spv,
            frag_spv,
            depth_vert_spv,
            depth_frag_spv,
            shadow_vert_spv,
            shadow_frag_spv,
            cluster_comp_spv,
        )?;

        // ---- Framebuffers ----

        let framebuffers = PassFramebuffers::new(
            &device,
            &render_passes,
            &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view,
            device_ctx.swapchain_extent,
        )?;

        // ---- Frame descriptors ----

        let global_ubo_size = std::mem::size_of::<GlobalUbo>() as u64;
        let cluster_params_size = std::mem::size_of::<ClusterParamsUbo>() as u64;
        let per_draw_ubo_size = std::mem::size_of::<UniformBufferObject>() as u64;
        let material_ssbo_size =
            (material_library.count() * std::mem::size_of::<MaterialData>()) as u64;

        let frame_descriptors = FrameDescriptors::new(
            &device,
            &descriptor_layouts,
            memory_ctx.ring.buffer,
            global_ubo_size,
            cluster_params_size,
            per_draw_ubo_size,
            material_ssbo.buffer,
            material_ssbo_size.max(128),
            &lighting_buffers,
            shadow_atlas.sampling_view,
            shadow_atlas.shadow_sampler,
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
            "[Renderer] Phase 2 initialized.  {} chunks pending, {} materials, shadow atlas ready.",
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
            light_manager,
            shadow_budget,
            shadow_atlas,
            lighting_buffers,
            shadow_assignments: HashMap::new(),
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

                // Phase 2: Register lights for this chunk.
                self.register_chunk_lights(coord);

                println!(
                    "[Renderer] Chunk ({},{}) → Ready  (frame {})",
                    coord.0, coord.1, self.global_frame,
                );
            }
        }
    }

    /// Register procedural lights for a newly-ready chunk.
    fn register_chunk_lights(&mut self, coord: ChunkCoord) {
        use crate::scene::CHUNK_SIZE;

        let cx = coord.0 as f32 * CHUNK_SIZE;
        let cz = coord.1 as f32 * CHUNK_SIZE;

        let center_x = cx + CHUNK_SIZE * 0.5; // 32.0 for chunk (0,0)
        let center_z = cz + CHUNK_SIZE * 0.5;

        let mut light = Light::point(
            [center_x, 12.0, center_z],   // elevated at Y=12 for long shadows
            [1.0, 0.95, 0.85],            // warm white — easy to read shadows
            120.0,                         // very bright — single light must illuminate everything
            50.0,                          // covers entire chunk with margin
        );
        light.shadow_capable = true;

        self.light_manager.register(coord, vec![light]);
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
                    let coord = chunk.coord;

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

                    // Phase 2: Unregister lights for evicted chunk.
                    self.light_manager.unregister(coord);

                    println!(
                        "[Renderer] Evicted chunk ({},{})",
                        coord.0, coord.1,
                    );
                    break;
                }
            }
        }
    }

    // ================================================================
    //  Frame rendering — multi-pass (Phase 2)
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

            // ---- Light cull + shadow assignment ----

            let frustum = self.scene.camera.extract_frustum_planes();
            let view_mat = self.scene.camera.get_view_matrix();
            let proj_mat = self.scene.camera.get_projection_matrix();
            let camera_pos = self.scene.camera.position;

            self.shadow_assignments = self.shadow_budget.assign(
                &self.light_manager,
                camera_pos,
            );

            let active_light_count = self.light_manager.cull_and_sort(
                camera_pos,
                &frustum,
                &self.shadow_assignments,
            );

            // Upload light SSBO for this frame.
            let light_data = self.light_manager.ssbo_bytes();
            self.lighting_buffers.upload_lights(self.current_frame, &light_data);

            // Push GlobalUbo to ring buffer.
            let global_ubo = GlobalUbo {
                view: view_mat,
                proj: proj_mat,
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], self.global_frame as f32],
            };
            let global_slice = self
                .memory_ctx
                .ring
                .push_data(&global_ubo)
                .expect("Ring overflow: GlobalUbo");
            let global_ubo_offset = global_slice.offset as u32;

            // Push ClusterParamsUbo to ring buffer.
            let cluster_params = ClusterParamsUbo::new(
                view_mat,
                proj_mat,
                self.scene.camera.near,
                self.scene.camera.far,
                device_ctx.swapchain_extent.width,
                device_ctx.swapchain_extent.height,
                active_light_count,
            );
            let cluster_slice = self
                .memory_ctx
                .ring
                .push_data(&cluster_params)
                .expect("Ring overflow: ClusterParamsUbo");
            let cluster_params_offset = cluster_slice.offset as u32;

            // Dynamic offsets for set 0: [binding 0, binding 4].
            let set0_dynamic_offsets = [global_ubo_offset, cluster_params_offset];

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

            // ---- Frustum cull (for drawable chunks) ----

            let drawable: Vec<ChunkCoord> = self
                .scene
                .chunks
                .iter()
                .filter(|(_, c)| c.is_ready() && c.is_visible(&frustum))
                .map(|(&coord, _)| coord)
                .collect();

            // ---- Stats ----

            if self.global_frame % 60 == 0 {
                let total = self.scene.chunks.len();
                let ready = self.scene.chunks.values().filter(|c| c.is_ready()).count();
                let streaming = self.scene.chunks.values()
                    .filter(|c| matches!(c.load_state, ChunkLoadState::Streaming { .. })).count();
                let visible = drawable.len();
                let culled = ready.saturating_sub(visible);

                println!(
                    "[Frame {:>5}] vis: {:>2}/{}  cull: {:>2}  stream: {}  lights: {}/{}  shadows: {}  tex: {}/{}",
                    self.global_frame,
                    visible, total, culled, streaming,
                    active_light_count, self.light_manager.total_count(),
                    self.shadow_budget.active_shadow_count(),
                    self.texture_manager.active_count(),
                    self.texture_manager.pending_count(),
                );
            }

            // ============================================================
            //  PASS 0: Shadow Pass
            // ============================================================

            let shadow_viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: SHADOW_MAP_SIZE as f32,
                height: SHADOW_MAP_SIZE as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let shadow_scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: SHADOW_MAP_SIZE,
                    height: SHADOW_MAP_SIZE,
                },
            };

            for (slot, light_idx) in self.shadow_budget.assigned_slots().collect::<Vec<_>>() {
                let Some(light) = self.light_manager.get(light_idx) else {
                    continue;
                };
                if light.light_type == LightType::Directional {
                    continue; // CSM for directional is Phase 3+
                }

                let push = ShadowPushConstants {
                    light_pos: light.position,
                    light_radius: light.radius,
                };

                // Render 6 cube faces.
                for face in 0..6u32 {
                    let (face_view, face_proj) =
                        cube_face_matrices(light.position, light.radius, face);

                    let clear = [vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    }];

                    let rp_info = vk::RenderPassBeginInfo::default()
                        .render_pass(self.render_passes.shadow)
                        .framebuffer(self.shadow_atlas.framebuffer(slot, face))
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: SHADOW_MAP_SIZE,
                                height: SHADOW_MAP_SIZE,
                            },
                        })
                        .clear_values(&clear);

                    self.device
                        .cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.shadow,
                    );
                    self.device.cmd_set_viewport(cmd, 0, &[shadow_viewport]);
                    self.device.cmd_set_scissor(cmd, 0, &[shadow_scissor]);

                    // Push constants.
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipelines.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const ShadowPushConstants as *const u8,
                            std::mem::size_of::<ShadowPushConstants>(),
                        ),
                    );

                    // Draw chunks within light radius.
                    for coord in &drawable {
                        let chunk = &self.scene.chunks[coord];
                        let vk_vb = match chunk.vertex_vk_buffer { Some(b) => b, None => continue };
                        let vk_ib = match chunk.index_vk_buffer { Some(b) => b, None => continue };

                        // Rough sphere-AABB cull against light.
                        if !chunk_in_light_range(chunk, light) {
                            continue;
                        }

                        self.device.cmd_bind_vertex_buffers(cmd, 0, &[vk_vb], &[0]);
                        self.device.cmd_bind_index_buffer(cmd, vk_ib, 0, vk::IndexType::UINT32);

                        let mut vertex_offset: i32 = 0;
                        let mut index_offset: u32 = 0;

                        for mesh in &chunk.meshes {
                            let ubo = UniformBufferObject::new(
                                mesh.transform,
                                face_view,
                                face_proj,
                                0, // material_id unused in shadow pass
                            );
                            let ring_slice = self
                                .memory_ctx
                                .ring
                                .push_data(&ubo)
                                .expect("Ring overflow: shadow UBO");

                            // Bind set 3 for the shadow pass.
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
                        }
                    }

                    self.device.cmd_end_render_pass(cmd);
                }
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

            // Bind set 0 + set 1 for depth prepass.
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[
                    self.frame_descriptors.per_frame_sets[self.current_frame],
                    self.texture_manager.descriptor_set,
                ],
                &set0_dynamic_offsets,
            );

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
                        .expect("Ring overflow: depth UBO");

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
                }
            }

            self.device.cmd_end_render_pass(cmd);

            // ============================================================
            //  PASS 2: Cluster Assignment Compute
            // ============================================================

            // Zero the global light index counter from the CPU side.
            // This eliminates the cross-workgroup race condition where
            // the GPU-side reset (workgroup 0 only) may not be visible
            // to other workgroups before their atomicAdd executes.
            self.device.cmd_fill_buffer(
                cmd,
                self.lighting_buffers.index_ssbo_buffers[self.current_frame],
                0,   // offset: global_count is at byte 0
                16,  // size: header is [global_count, _pad, _pad, _pad] = 16 bytes
                0,   // data: zero
            );

            // Barrier: fill → compute (ensures zero is visible before atomicAdd).
            let fill_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                );

            // Also need the light SSBO host writes visible to compute.
            let host_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[fill_barrier, host_barrier],
                &[],
                &[],
            );

            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.cluster_compute,
            );

            // Bind set 0 for compute (uses compute_layout which has only set 0).
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.compute_layout,
                0,
                &[self.frame_descriptors.per_frame_sets[self.current_frame]],
                &set0_dynamic_offsets,
            );

            self.device.cmd_dispatch(cmd, TOTAL_CLUSTERS, 1, 1);

            // Memory barrier: compute writes → fragment reads.
            let post_compute_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&post_compute_barrier),
                &[],
                &[],
            );

            // ============================================================
            //  PASS 3: Lighting Pass
            // ============================================================

            let light_clear = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.01, 0.01, 0.015, 1.0],
                    },
                },
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

            // Bind set 0 (per-frame globals + lighting data).
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[self.frame_descriptors.per_frame_sets[self.current_frame]],
                &set0_dynamic_offsets,
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

            // Bind set 2 (shadow maps).
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                2,
                &[self.frame_descriptors.shadow_map_sets[self.current_frame]],
                &[],
            );

            let mut meshes_drawn = 0u32;

            for coord in &drawable {
                let chunk = &self.scene.chunks[coord];
                let vk_vb = match chunk.vertex_vk_buffer { Some(b) => b, None => continue };
                let vk_ib = match chunk.index_vk_buffer { Some(b) => b, None => continue };

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
                        .expect("Ring overflow: lighting UBO");

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
    //  Public accessors
    // ================================================================

    pub fn materials_mut(&mut self) -> &mut MaterialLibrary {
        &mut self.material_library
    }

    pub fn materials(&self) -> &MaterialLibrary {
        &self.material_library
    }

    pub fn texture_manager_mut(&mut self) -> &mut TextureManager {
        &mut self.texture_manager
    }

    pub fn light_manager_mut(&mut self) -> &mut LightManager {
        &mut self.light_manager
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
//  Chunk-vs-light range helper
// ====================================================================

/// Rough sphere-AABB test: is any part of the chunk within the light's radius?
fn chunk_in_light_range(
    chunk: &crate::scene::Chunk,
    light: &Light,
) -> bool {
    let aabb_min = chunk.aabb_min;
    let aabb_max = chunk.aabb_max;
    let pos = light.position;
    let r = light.radius;

    // Closest point on AABB to light.
    let cx = pos[0].clamp(aabb_min[0], aabb_max[0]);
    let cy = pos[1].clamp(aabb_min[1], aabb_max[1]);
    let cz = pos[2].clamp(aabb_min[2], aabb_max[2]);

    let dx = pos[0] - cx;
    let dy = pos[1] - cy;
    let dz = pos[2] - cz;

    dx * dx + dy * dy + dz * dz <= r * r
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
            self.shadow_atlas.destroy(&self.device);
            self.lighting_buffers.destroy(&mut self.memory_ctx.allocator);
        }
    }
}