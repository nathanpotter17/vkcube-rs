use ash::{vk, Device};
use std::collections::HashMap;

use crate::device::DeviceContext;
use crate::light::{
    self, cube_face_matrices, ClusterParamsUbo, Light, LightManager, LightType,
    ShadowAtlas, ShadowBudgetManager, ShadowPushConstants, CLUSTER_X, CLUSTER_Y,
    CLUSTER_Z, MAX_SHADOW_SLOTS, SHADOW_MAP_SIZE, TOTAL_CLUSTERS,
};
use crate::material::{MaterialData, MaterialLibrary, MaterialSsbo};
use crate::memory::{BufferHandle, MemoryContext, MAX_FRAMES_IN_FLIGHT};
use crate::pipeline::{
    DescriptorLayouts, FrameDescriptors, FrameLightingBuffers, PassFramebuffers,
    Pipelines, RenderPasses,
};
use crate::scene::{Scene, UniformBufferObject, Vertex};
use crate::texture::TextureManager;
use crate::world::{
    Aabb, DrawCommand, LodChain, MeshRange, RenderFlags,
    RenderObjectId, SectorCoord, SectorState, World,
    EVICTION_RADIUS, GROUND_TILE_SIZE, MAX_SECTOR_STARTS_PER_FRAME,
    SECTOR_SIZE, STREAMING_RADIUS, generate_sector_objects,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct GlobalUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
}

// ====================================================================
//  Per-sector pending upload
// ====================================================================

/// Per-object metadata stored while the sector's batch upload is in-flight.
struct PendingObject {
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
    transform: [[f32; 4]; 4],
    material_id: u32,
    flags: RenderFlags,
    bounds: Aabb,
}

/// One in-flight upload per sector — a single vertex buffer + index buffer
/// containing ALL objects concatenated.
struct PendingSectorUpload {
    sector: SectorCoord,
    vertex_handle: BufferHandle,
    index_handle: BufferHandle,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    vertex_ticket: crate::memory::TransferTicket,
    index_ticket: crate::memory::TransferTicket,
    vertex_size: u64,
    index_size: u64,
    /// Per-object metadata for creating RenderObjects on completion.
    objects: Vec<PendingObject>,
}

// ====================================================================
//  Renderer
// ====================================================================

pub struct Renderer {
    device: Device,
    memory_ctx: MemoryContext,

    scene: Scene,
    world: World,
    pending_sectors: Vec<PendingSectorUpload>,

    material_library: MaterialLibrary,
    material_ssbo: MaterialSsbo,
    texture_manager: TextureManager,

    light_manager: LightManager,
    shadow_budget: ShadowBudgetManager,
    shadow_atlas: ShadowAtlas,
    lighting_buffers: FrameLightingBuffers,
    shadow_assignments: HashMap<usize, u32>,

    descriptor_layouts: DescriptorLayouts,
    render_passes: RenderPasses,
    pipelines: Pipelines,
    framebuffers: PassFramebuffers,
    frame_descriptors: FrameDescriptors,

    command_buffers: Vec<vk::CommandBuffer>,

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
        let world = World::new();

        // ---- Materials ----

        let mut material_library = MaterialLibrary::new();
        material_library.add("ground", MaterialData {
            base_color:[0.35,0.28,0.18,1.0], roughness:0.92, metallic:0.0, ..Default::default() });
        material_library.add("polished_metal", MaterialData {
            base_color:[0.95,0.93,0.88,1.0], roughness:0.08, metallic:1.0, ..Default::default() });
        material_library.add("rough_stone", MaterialData {
            base_color:[0.55,0.52,0.50,1.0], roughness:0.95, metallic:0.0, ..Default::default() });
        material_library.add("copper", MaterialData {
            base_color:[0.95,0.64,0.54,1.0], roughness:0.25, metallic:1.0, ..Default::default() });
        material_library.add("ceramic_red", MaterialData {
            base_color:[0.85,0.15,0.12,1.0], roughness:0.15, metallic:0.0, ..Default::default() });
        material_library.add("ceramic_blue", MaterialData {
            base_color:[0.12,0.35,0.85,1.0], roughness:0.15, metallic:0.0, ..Default::default() });
        material_library.add("gold", MaterialData {
            base_color:[1.0,0.76,0.33,1.0], roughness:0.18, metallic:1.0, ..Default::default() });
        material_library.add("rubber", MaterialData {
            base_color:[0.12,0.12,0.14,1.0], roughness:0.98, metallic:0.0, ..Default::default() });
        material_library.add("marble", MaterialData {
            base_color:[0.92,0.90,0.85,1.0], roughness:0.35, metallic:0.0, ..Default::default() });
        material_library.add("emissive_warm", MaterialData {
            base_color:[1.0,0.85,0.4,1.0], emissive:[1.0,0.7,0.2,8.0], roughness:0.5, metallic:0.0, ..Default::default() });
        material_library.add("emissive_cool", MaterialData {
            base_color:[0.4,0.7,1.0,1.0], emissive:[0.3,0.5,1.0,8.0], roughness:0.5, metallic:0.0, ..Default::default() });

        let material_ssbo = MaterialSsbo::new(&mut memory_ctx)?;
        material_ssbo.upload(&material_library);
        material_library.clear_dirty();

        let texture_manager = TextureManager::new(&device, &mut memory_ctx, device_ctx.command_pool, device_ctx.queue)?;
        let light_manager = LightManager::new();
        let shadow_budget = ShadowBudgetManager::new();
        let descriptor_layouts = DescriptorLayouts::new(&device, texture_manager.descriptor_set_layout)?;
        let render_passes = RenderPasses::new(&device, device_ctx.surface_format.format)?;
        let shadow_atlas = ShadowAtlas::new(&device, &mut memory_ctx.allocator, render_passes.shadow, device_ctx.command_pool, device_ctx.queue)?;
        let lighting_buffers = FrameLightingBuffers::new(&mut memory_ctx.allocator)?;

        let vert_spv = include_bytes!("../shaders/compiled/basic.vert.spv");
        let frag_spv = include_bytes!("../shaders/compiled/basic.frag.spv");
        let depth_vert_spv: &[u8] = include_bytes!("../shaders/compiled/depth.vert.spv");
        let depth_frag_spv: &[u8] = include_bytes!("../shaders/compiled/depth.frag.spv");
        let shadow_vert_spv: &[u8] = include_bytes!("../shaders/compiled/shadow.vert.spv");
        let shadow_frag_spv: &[u8] = include_bytes!("../shaders/compiled/shadow.frag.spv");
        let cluster_comp_spv: &[u8] = include_bytes!("../shaders/compiled/cluster_assign.comp.spv");

        let pipelines = Pipelines::new(&device, &descriptor_layouts, &render_passes,
            vert_spv, frag_spv, depth_vert_spv, depth_frag_spv, shadow_vert_spv, shadow_frag_spv, cluster_comp_spv)?;
        let framebuffers = PassFramebuffers::new(&device, &render_passes, &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view, device_ctx.swapchain_extent)?;

        let global_ubo_size = std::mem::size_of::<GlobalUbo>() as u64;
        let cluster_params_size = std::mem::size_of::<ClusterParamsUbo>() as u64;
        let per_draw_ubo_size = std::mem::size_of::<UniformBufferObject>() as u64;
        let material_ssbo_size = (material_library.count() * std::mem::size_of::<MaterialData>()) as u64;

        let frame_descriptors = FrameDescriptors::new(&device, &descriptor_layouts, memory_ctx.ring.buffer,
            global_ubo_size, cluster_params_size, per_draw_ubo_size,
            material_ssbo.buffer, material_ssbo_size.max(128),
            &lighting_buffers, shadow_atlas.sampling_view, shadow_atlas.shadow_sampler)?;

        let command_buffers = Self::allocate_command_buffers(&device, device_ctx.command_pool)?;

        let mut image_available = Vec::new();
        let mut render_finished = Vec::new();
        let mut in_flight_fences = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT { unsafe {
            image_available.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            render_finished.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            in_flight_fences.push(device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?);
        }}

        let cam_pos = scene.camera.position;
        println!("[Renderer] Phase 2 initialized. {} materials. Streaming radius: {}m", material_library.count(), STREAMING_RADIUS);

        Ok(Self {
            device, memory_ctx, scene, world,
            pending_sectors: Vec::new(),
            material_library, material_ssbo, texture_manager,
            light_manager, shadow_budget, shadow_atlas, lighting_buffers,
            shadow_assignments: HashMap::new(),
            descriptor_layouts, render_passes, pipelines, framebuffers,
            frame_descriptors, command_buffers,
            image_available, render_finished, in_flight_fences,
            current_frame: 0, global_frame: 0,
        })
    }

    // ================================================================
    //  Streaming: batched per-sector upload
    // ================================================================

    fn update_streaming(&mut self) {
        let camera_pos = self.scene.camera.position;
        let camera_xz = [camera_pos[0], camera_pos[2]];
        let velocity_xz = self.scene.camera_velocity_xz();
        let frustum = self.scene.camera.extract_frustum_planes();

        self.world.update_sector_grid(camera_xz, STREAMING_RADIUS);
        self.evict_distant_sectors(camera_xz);

        let to_stream = self.world.prioritized_unloaded_sectors(camera_xz, velocity_xz, &frustum);
        let mut started = 0;

        for coord in to_stream {
            if started >= MAX_SECTOR_STARTS_PER_FRAME { break; }

            let descriptors = generate_sector_objects(coord);
            if descriptors.is_empty() {
                if let Some(sec) = self.world.sectors.get_mut(&coord) {
                    sec.state = SectorState::Ready;
                }
                continue;
            }

            match self.upload_sector_batch(coord, descriptors) {
                Ok(()) => started += 1,
                Err(e) => {
                    eprintln!("[Renderer] Sector ({},{}) upload failed: {}", coord.0, coord.1, e);
                    if let Some(sec) = self.world.sectors.get_mut(&coord) {
                        sec.state = SectorState::Failed;
                    }
                }
            }
        }
    }

    /// Concatenate all object vertices/indices into ONE vertex buffer +
    /// ONE index buffer, upload via async transfer, track per-object offsets.
    fn upload_sector_batch(
        &mut self, sector: SectorCoord, descriptors: Vec<crate::world::ObjectDescriptor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Concatenate all vertices and indices.
        let mut all_verts: Vec<Vertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut objects: Vec<PendingObject> = Vec::new();

        for desc in &descriptors {
            let vertex_offset = all_verts.len() as i32;
            let first_index = all_indices.len() as u32;
            let index_count = desc.indices.len() as u32;

            all_verts.extend_from_slice(&desc.vertices);
            all_indices.extend_from_slice(&desc.indices);

            objects.push(PendingObject {
                first_index, index_count, vertex_offset,
                transform: desc.transform, material_id: desc.material_id,
                flags: desc.flags, bounds: desc.bounds,
            });
        }

        // Upload ONE vertex buffer.
        let (valloc, vticket) = self.memory_ctx.upload_async_typed(
            &all_verts, vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        // Upload ONE index buffer.
        let (ialloc, iticket) = match self.memory_ctx.upload_async_typed(
            &all_indices, vk::BufferUsageFlags::INDEX_BUFFER,
        ) {
            Ok(v) => v,
            Err(e) => {
                self.memory_ctx.allocator.free_buffer(valloc.handle);
                return Err(e);
            }
        };

        self.memory_ctx.budget.track(valloc.handle, valloc.size, self.global_frame);
        self.memory_ctx.budget.track(ialloc.handle, ialloc.size, self.global_frame);

        if let Some(sec) = self.world.sectors.get_mut(&sector) {
            sec.state = SectorState::Streaming;
            sec.vertex_handle = Some(valloc.handle);
            sec.index_handle = Some(ialloc.handle);
        }

        self.pending_sectors.push(PendingSectorUpload {
            sector,
            vertex_handle: valloc.handle, index_handle: ialloc.handle,
            vertex_buffer: valloc.buffer, index_buffer: ialloc.buffer,
            vertex_ticket: vticket, index_ticket: iticket,
            vertex_size: valloc.size, index_size: ialloc.size,
            objects,
        });

        Ok(())
    }

    fn poll_uploads(&mut self) {
        let mut completed: Vec<PendingSectorUpload> = Vec::new();

        // Extract ref to transfer queue — disjoint from pending_sectors.
        let transfer = &self.memory_ctx.transfer;

        self.pending_sectors.retain_mut(|upload| {
            let done = transfer.is_complete(&upload.vertex_ticket)
                && transfer.is_complete(&upload.index_ticket);
            if done {
                completed.push(PendingSectorUpload {
                    sector: upload.sector,
                    vertex_handle: upload.vertex_handle, index_handle: upload.index_handle,
                    vertex_buffer: upload.vertex_buffer, index_buffer: upload.index_buffer,
                    vertex_ticket: upload.vertex_ticket.clone(),
                    index_ticket: upload.index_ticket.clone(),
                    vertex_size: upload.vertex_size, index_size: upload.index_size,
                    objects: std::mem::take(&mut upload.objects),
                });
                false
            } else { true }
        });

        for upload in completed {
            let vb = upload.vertex_buffer;
            let ib = upload.index_buffer;

            for pobj in &upload.objects {
                let mesh_range = MeshRange {
                    vertex_buffer: vb, index_buffer: ib,
                    first_index: pobj.first_index,
                    index_count: pobj.index_count,
                    vertex_offset: pobj.vertex_offset,
                };
                self.world.add_object(
                    upload.sector, pobj.bounds,
                    LodChain::single(mesh_range),
                    pobj.transform, pobj.material_id, pobj.flags,
                );
            }

            if let Some(sec) = self.world.sectors.get_mut(&upload.sector) {
                sec.state = SectorState::Ready;
                let obj_count = sec.objects.len();
                println!(
                    "[Renderer] Sector ({},{}) → Ready ({} objects, frame {})",
                    upload.sector.0, upload.sector.1, obj_count, self.global_frame,
                );
            }
            self.register_sector_lights(upload.sector);
        }
    }

    fn register_sector_lights(&mut self, sector: SectorCoord) {
        let tiles = (SECTOR_SIZE / GROUND_TILE_SIZE) as i32;
        let bx = sector.0 * tiles;
        let bz = sector.1 * tiles;
        let mut lights = Vec::new();
        for dx in 0..tiles { for dz in 0..tiles {
            let cx = (bx+dx) as f32 * GROUND_TILE_SIZE + GROUND_TILE_SIZE * 0.5;
            let cz = (bz+dz) as f32 * GROUND_TILE_SIZE + GROUND_TILE_SIZE * 0.5;
            let mut l = Light::point([cx,12.0,cz], [1.0,0.95,0.85], 120.0, 50.0);
            l.shadow_capable = true;
            lights.push(l);
        }}
        self.light_manager.register(sector, lights);
    }

    fn evict_distant_sectors(&mut self, camera_xz: [f32; 2]) {
        let r2 = EVICTION_RADIUS * EVICTION_RADIUS;
        let to_evict: Vec<SectorCoord> = self.world.sectors.iter()
            .filter(|(_, s)| s.state == SectorState::Ready)
            .filter(|(_, s)| {
                let c = s.world_center();
                (c[0]-camera_xz[0]).powi(2) + (c[1]-camera_xz[1]).powi(2) > r2
            })
            .map(|(&c,_)| c).collect();

        for coord in to_evict {
            // Free the 2 sector-level buffer handles.
            if let Some(sec) = self.world.sectors.get(&coord) {
                if let Some(vh) = sec.vertex_handle {
                    self.memory_ctx.allocator.free_buffer(vh);
                    self.memory_ctx.budget.untrack(vh);
                }
                if let Some(ih) = sec.index_handle {
                    self.memory_ctx.allocator.free_buffer(ih);
                    self.memory_ctx.budget.untrack(ih);
                }
            }

            let ids = self.world.evict_sector(coord);
            self.light_manager.unregister(coord);

            if !ids.is_empty() {
                println!("[Renderer] Evicted sector ({},{}) — {} objects", coord.0, coord.1, ids.len());
            }
        }
    }

    // ================================================================
    //  Frame rendering
    // ================================================================

    pub fn render(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        self.global_frame += 1;

        self.update_streaming();
        self.poll_uploads();
        self.texture_manager.poll_pending(&self.memory_ctx);

        if self.material_library.is_dirty() {
            self.material_ssbo.upload(&self.material_library);
            self.material_library.clear_dirty();
        }

        unsafe {
            self.device.wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)?;
            self.scene.update(0.016);
            self.memory_ctx.ring.begin_frame(self.current_frame);

            let frustum = self.scene.camera.extract_frustum_planes();
            let view_mat = self.scene.camera.get_view_matrix();
            let proj_mat = self.scene.camera.get_projection_matrix();
            let camera_pos = self.scene.camera.position;

            let xz = crate::world::frustum_aabb_xz(camera_pos, self.scene.camera.far);
            self.world.cull_and_select_lod(camera_pos, &frustum, &xz);

            self.shadow_assignments = self.shadow_budget.assign(&self.light_manager, camera_pos);
            let active_light_count = self.light_manager.cull_and_sort(camera_pos, &frustum, &self.shadow_assignments);

            self.lighting_buffers.upload_lights(self.current_frame, &self.light_manager.ssbo_bytes());

            let global_ubo = GlobalUbo {
                view: view_mat, proj: proj_mat,
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], self.global_frame as f32],
            };
            let g_off = self.memory_ctx.ring.push_data(&global_ubo).expect("Ring: GlobalUbo").offset as u32;

            let cluster_params = ClusterParamsUbo::new(view_mat, proj_mat,
                self.scene.camera.near, self.scene.camera.far,
                device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height, active_light_count);
            let c_off = self.memory_ctx.ring.push_data(&cluster_params).expect("Ring: ClusterParams").offset as u32;

            let dyn_off = [g_off, c_off];

            let (image_index, _) = device_ctx.swapchain_loader.acquire_next_image(
                device_ctx.swapchain, u64::MAX, self.image_available[self.current_frame], vk::Fence::null())?;
            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            let cmd = self.command_buffers[self.current_frame];
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            let viewport = vk::Viewport { x:0.0, y:0.0,
                width: device_ctx.swapchain_extent.width as f32,
                height: device_ctx.swapchain_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0 };
            let scissor = vk::Rect2D { offset: vk::Offset2D{x:0,y:0}, extent: device_ctx.swapchain_extent };

            // ---- Stats ----
            if self.global_frame % 60 == 0 {
                println!(
                    "[Frame {:>5}] sectors: {}/{} (stream:{} fail:{})  objects: {}  draws: {} opq + {} shd  lights: {}/{}  shadows: {}  tex: {}/{}",
                    self.global_frame,
                    self.world.ready_sector_count(), self.world.sectors.len(),
                    self.world.streaming_sector_count(),
                    self.world.sectors.values().filter(|s| s.state == SectorState::Failed).count(),
                    self.world.total_objects(),
                    self.world.opaque_draws.len(), self.world.shadow_draws.len(),
                    active_light_count, self.light_manager.total_count(),
                    self.shadow_budget.active_shadow_count(),
                    self.texture_manager.active_count(), self.texture_manager.pending_count(),
                );
            }

            // ============================================================
            //  PASS 0: Shadow Pass
            // ============================================================
            let sv = vk::Viewport { x:0.0,y:0.0, width:SHADOW_MAP_SIZE as f32, height:SHADOW_MAP_SIZE as f32, min_depth:0.0, max_depth:1.0 };
            let ss = vk::Rect2D { offset:vk::Offset2D{x:0,y:0}, extent:vk::Extent2D{width:SHADOW_MAP_SIZE,height:SHADOW_MAP_SIZE} };

            for (slot, light_idx) in self.shadow_budget.assigned_slots().collect::<Vec<_>>() {
                let Some(light) = self.light_manager.get(light_idx) else { continue };
                if light.light_type == LightType::Directional { continue; }
                let push = ShadowPushConstants { light_pos: light.position, light_radius: light.radius };
                let lp = light.position; let lr2 = light.radius * light.radius;

                for face in 0..6u32 {
                    let (fv, fp) = cube_face_matrices(lp, light.radius, face);
                    let clear = [vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue{depth:1.0,stencil:0} }];
                    let rp = vk::RenderPassBeginInfo::default()
                        .render_pass(self.render_passes.shadow)
                        .framebuffer(self.shadow_atlas.framebuffer(slot, face))
                        .render_area(ss).clear_values(&clear);
                    self.device.cmd_begin_render_pass(cmd, &rp, vk::SubpassContents::INLINE);
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.shadow);
                    self.device.cmd_set_viewport(cmd, 0, &[sv]);
                    self.device.cmd_set_scissor(cmd, 0, &[ss]);
                    self.device.cmd_push_constants(cmd, self.pipelines.layout,
                        vk::ShaderStageFlags::VERTEX|vk::ShaderStageFlags::FRAGMENT, 0,
                        std::slice::from_raw_parts(&push as *const _ as *const u8, std::mem::size_of_val(&push)));

                    for draw in &self.world.shadow_draws {
                        let oi = draw.object_id.0 as usize;
                        if oi >= self.world.objects.len() { continue; }
                        let obj = &self.world.objects[oi];
                        if !obj.alive || obj.bounds.distance_sq_to_point(lp) > lr2 { continue; }

                        self.device.cmd_bind_vertex_buffers(cmd, 0, &[draw.vertex_buffer], &[0]);
                        self.device.cmd_bind_index_buffer(cmd, draw.index_buffer, 0, vk::IndexType::UINT32);
                        let ubo = UniformBufferObject::new(draw.transform, fv, fp, 0);
                        let rs = self.memory_ctx.ring.push_data(&ubo).expect("Ring: shadow UBO");
                        self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                            self.pipelines.layout, 3,
                            &[self.frame_descriptors.per_draw_sets[self.current_frame]],
                            &[rs.offset as u32]);
                        self.device.cmd_draw_indexed(cmd, draw.index_count, 1, draw.first_index, draw.vertex_offset, 0);
                    }
                    self.device.cmd_end_render_pass(cmd);
                }
            }

            // ============================================================
            //  PASS 1: Depth Pre-Pass
            // ============================================================
            let dc = [vk::ClearValue{depth_stencil:vk::ClearDepthStencilValue{depth:1.0,stencil:0}}];
            let drp = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_passes.depth_prepass)
                .framebuffer(self.framebuffers.depth_prepass[image_index as usize])
                .render_area(vk::Rect2D{offset:vk::Offset2D{x:0,y:0},extent:device_ctx.swapchain_extent})
                .clear_values(&dc);
            self.device.cmd_begin_render_pass(cmd, &drp, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.depth_prepass);
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.layout, 0,
                &[self.frame_descriptors.per_frame_sets[self.current_frame], self.texture_manager.descriptor_set], &dyn_off);

            let opaque_draws = std::mem::take(&mut self.world.opaque_draws);
            self.record_draw_list(cmd, &opaque_draws, view_mat, proj_mat);
            self.device.cmd_end_render_pass(cmd);

            // ============================================================
            //  PASS 2: Cluster Assignment Compute
            // ============================================================
            self.device.cmd_fill_buffer(cmd, self.lighting_buffers.index_ssbo_buffers[self.current_frame], 0, 16, 0);
            let fb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER|vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[fb,hb], &[], &[]);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.cluster_compute);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.compute_layout, 0,
                &[self.frame_descriptors.per_frame_sets[self.current_frame]], &dyn_off);
            self.device.cmd_dispatch(cmd, TOTAL_CLUSTERS, 1, 1);
            let pc = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(),
                std::slice::from_ref(&pc), &[], &[]);

            // ============================================================
            //  PASS 3: Lighting Pass
            // ============================================================
            let lc = [
                vk::ClearValue{color:vk::ClearColorValue{float32:[0.01,0.01,0.015,1.0]}},
                vk::ClearValue{depth_stencil:vk::ClearDepthStencilValue{depth:1.0,stencil:0}},
            ];
            let lrp = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_passes.lighting)
                .framebuffer(self.framebuffers.lighting[image_index as usize])
                .render_area(vk::Rect2D{offset:vk::Offset2D{x:0,y:0},extent:device_ctx.swapchain_extent})
                .clear_values(&lc);
            self.device.cmd_begin_render_pass(cmd, &lrp, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.lighting);
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.layout, 0,
                &[self.frame_descriptors.per_frame_sets[self.current_frame]], &dyn_off);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.layout, 1,
                &[self.texture_manager.descriptor_set], &[]);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.layout, 2,
                &[self.frame_descriptors.shadow_map_sets[self.current_frame]], &[]);

            self.record_draw_list(cmd, &opaque_draws, view_mat, proj_mat);
            self.world.opaque_draws = opaque_draws;

            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            // ---- Submit + Present ----
            let ws = [self.image_available[self.current_frame]];
            let wst = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let ss2 = [self.render_finished[self.current_frame]];
            let sub = vk::SubmitInfo::default().wait_semaphores(&ws).wait_dst_stage_mask(&wst)
                .command_buffers(std::slice::from_ref(&cmd)).signal_semaphores(&ss2);
            self.device.queue_submit(device_ctx.queue, &[sub], self.in_flight_fences[self.current_frame])?;

            let pres = vk::PresentInfoKHR::default().wait_semaphores(&ss2)
                .swapchains(std::slice::from_ref(&device_ctx.swapchain))
                .image_indices(std::slice::from_ref(&image_index));
            device_ctx.swapchain_loader.queue_present(device_ctx.queue, &pres)?;

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        Ok(())
    }

    /// Record draw calls using per-object offsets (first_index, vertex_offset).
    fn record_draw_list(&mut self, cmd: vk::CommandBuffer, draws: &[DrawCommand],
        view_mat: [[f32;4];4], proj_mat: [[f32;4];4]) -> u32 {
        let mut count = 0u32;
        let mut bound_vb: Option<vk::Buffer> = None;
        let mut bound_ib: Option<vk::Buffer> = None;

        for draw in draws { unsafe {
            // Skip redundant binds — objects in the same sector share buffers.
            if bound_vb != Some(draw.vertex_buffer) {
                self.device.cmd_bind_vertex_buffers(cmd, 0, &[draw.vertex_buffer], &[0]);
                bound_vb = Some(draw.vertex_buffer);
            }
            if bound_ib != Some(draw.index_buffer) {
                self.device.cmd_bind_index_buffer(cmd, draw.index_buffer, 0, vk::IndexType::UINT32);
                bound_ib = Some(draw.index_buffer);
            }

            let ubo = UniformBufferObject::new(draw.transform, view_mat, proj_mat, draw.material_id);
            let rs = self.memory_ctx.ring.push_data(&ubo).expect("Ring: per-draw UBO");
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout, 3,
                &[self.frame_descriptors.per_draw_sets[self.current_frame]], &[rs.offset as u32]);

            self.device.cmd_draw_indexed(cmd, draw.index_count, 1, draw.first_index, draw.vertex_offset, 0);
        } count += 1; }
        count
    }

    pub fn recreate_framebuffers(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.device.device_wait_idle()? };
        self.framebuffers.destroy(&self.device);
        self.framebuffers = PassFramebuffers::new(&self.device, &self.render_passes,
            &device_ctx.swapchain_image_views, device_ctx.depth_image_view, device_ctx.swapchain_extent)?;
        self.scene.camera.update_aspect(device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height);
        Ok(())
    }

    pub fn materials_mut(&mut self) -> &mut MaterialLibrary { &mut self.material_library }
    pub fn materials(&self) -> &MaterialLibrary { &self.material_library }
    pub fn texture_manager_mut(&mut self) -> &mut TextureManager { &mut self.texture_manager }
    pub fn light_manager_mut(&mut self) -> &mut LightManager { &mut self.light_manager }

    fn allocate_command_buffers(device: &Device, pool: vk::CommandPool) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
        let info = vk::CommandBufferAllocateInfo::default().command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
        unsafe { Ok(device.allocate_command_buffers(&info)?) }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) { unsafe {
        let _ = self.device.device_wait_idle();
        for &f in &self.in_flight_fences { self.device.destroy_fence(f, None); }
        for &s in &self.render_finished { self.device.destroy_semaphore(s, None); }
        for &s in &self.image_available { self.device.destroy_semaphore(s, None); }
        self.frame_descriptors.destroy(&self.device);
        self.framebuffers.destroy(&self.device);
        self.pipelines.destroy(&self.device);
        self.render_passes.destroy(&self.device);
        self.descriptor_layouts.destroy(&self.device);
        self.shadow_atlas.destroy(&self.device);
        self.lighting_buffers.destroy(&mut self.memory_ctx.allocator);
    }}
}