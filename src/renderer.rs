//! Phase 8A: GPU-Driven Rendering
//!
//! This module replaces CPU frustum culling (~8000 per-frame ring buffer pushes)
//! with GPU compute-based culling and indirect draw commands.
//!
//! Key changes from pre-8A:
//! - PerDrawUbo removed (set 3 now binds object SSBO)
//! - DrawCommand CPU lists removed (cull shader writes VkDrawIndexedIndirectCommand)
//! - record_draw_list() removed (replaced by cmd_draw_indexed_indirect per buffer group)
//! - GpuCullResources manages object SSBO, indirect buffers, cull compute pipeline

use ash::{vk, Device};
use std::collections::HashMap;
use std::path::Path;
use crate::device::DeviceContext;
use crate::gi::{GIResources, GpuProbeGridParams, ProbeBakeTarget, ProbeGrid,
                 SHProjectPushConstants, MAX_PROBE_BAKES_PER_FRAME,
                 PROBE_CAPTURE_NEAR, PROBE_CAPTURE_FAR, PROBE_CAPTURE_SIZE};
use crate::gpu_cull::{GpuCullResources, GpuObjectData, CullPushConstants,
                       MegaAlloc, INDIRECT_COMMAND_STRIDE, MAX_INDIRECT_DRAWS, MAX_BUFFER_GROUPS};
use crate::light::{
    self, cube_face_matrices, ClusterParamsUbo, Light, LightCategory, LightManager, LightType,
    ShadowAtlas, ShadowBudgetManager, ShadowPushConstants, SunShadow,
    compute_sun_shadow_matrices,
    CLUSTER_X, CLUSTER_Y, CLUSTER_Z, MAX_SHADOW_SLOTS, SHADOW_MAP_SIZE,
    SUN_SHADOW_SIZE, TOTAL_CLUSTERS, SUN_COLOR, SUN_INTENSITY, SUN_DIR,
};
use crate::material::{MaterialData, MaterialLibrary, MaterialSsbo};
use crate::memory::{BufferHandle, GpuAllocator, MemoryContext, MAX_FRAMES_IN_FLIGHT};
use crate::pipeline::{
    DescriptorLayouts, FrameDescriptors, FrameLightingBuffers, PassFramebuffers,
    Pipelines, RenderPasses,
};
// Phase 8A: PerDrawUbo removed — SSBO replaces per-draw UBO
// Phase 8A: DrawCommand removed — GPU cull writes indirect commands
use crate::scene::{InputAction, InputState, Scene, Vertex};
use crate::texture::TextureManager;
use crate::world::{
    Aabb, LodChain, MeshRange, ObjectDescriptor, RenderFlags,
    RenderObjectId, Sector, SectorCoord, SectorState, World,
    EVICTION_RADIUS, GROUND_TILE_SIZE, MAX_SECTOR_STARTS_PER_FRAME,
    SECTOR_SIZE, STREAMING_RADIUS, generate_sector_objects,
    demo_transform, make_cube, make_pyramid, make_column,
};
use crate::loader::GltfLoader;
use crate::postprocess::HbaoPass;
use crate::profiler::{GpuProfiler, PassId};
use crate::overlay::DebugOverlay;

#[repr(C)]
#[derive(Clone, Copy)]
struct GlobalUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    sun_light_vp: [[f32; 4]; 4],
    sun_direction: [f32; 4],
}

// ====================================================================
//  Dynamic spawn limits and PRNG
// ====================================================================

const MAX_SPAWNED_LIGHTS: usize = 32;
/// Phase 8C.1: After dynamic limit, user spawns become baked (no shadow cost).
const MAX_SPAWNED_BAKED_LIGHTS: usize = 256;
const MAX_SPAWNED_GEOMETRY: usize = 50;
const SPAWN_RADIUS: f32 = 40.0;
const SPAWN_LIGHT_HEIGHT_MIN: f32 = 3.0;
const SPAWN_LIGHT_HEIGHT_MAX: f32 = 15.0;

const DYNAMIC_SECTOR_BASE: i32 = i32::MAX - 5000;
const GLTF_SECTOR_BASE: i32 = i32::MAX - 4000;

struct DeferredFree {
    handle: BufferHandle,
    frame: u64,
}

struct SimpleRng { s: u64 }

impl SimpleRng {
    fn new(seed: u64) -> Self { Self { s: seed.max(1) } }
    fn next_u64(&mut self) -> u64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        self.s
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() & 0xFF_FFFF) as f32 / 16_777_216.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[(self.next_u64() as usize) % items.len()]
    }
}

// ====================================================================
//  Per-sector pending upload
// ====================================================================

struct PendingObject {
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
    transform: [[f32; 4]; 4],
    material_id: u32,
    flags: RenderFlags,
    bounds: Aabb,
}

struct PendingSectorUpload {
    sector: SectorCoord,
    /// Phase 8B: mega buffer sub-allocation for this sector's geometry
    mega_alloc: MegaAlloc,
    /// Transfer tickets for vertex and index data uploads to mega buffers
    vertex_ticket: crate::memory::TransferTicket,
    index_ticket: crate::memory::TransferTicket,
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
    sun_shadow: SunShadow,
    sun_direction: [f32; 3],

    dynamic_light_indices: Vec<usize>,
    dynamic_light_time: f32,

    // Phase 3: Global Illumination
    probe_grid: ProbeGrid,
    gi_resources: GIResources,
    probe_bake_target: ProbeBakeTarget,

    // Phase 8A: GPU-driven culling resources
    gpu_cull: GpuCullResources,

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

    rng: SimpleRng,
    spawned_light_count: usize,
    spawned_baked_light_count: usize,
    spawned_light_sector: SectorCoord,
    spawned_geometry_count: usize,
    next_dynamic_sector: i32,

    normal_image: vk::Image,
    normal_memory: vk::DeviceMemory,
    normal_view: vk::ImageView,

    hbao_pass: HbaoPass,
    has_hdr_skybox: bool,
    profiler: GpuProfiler,
    overlay: DebugOverlay,
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
        let mut world = World::new();

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

        let mut texture_manager = TextureManager::new(&device, &mut memory_ctx, device_ctx.command_pool, device_ctx.queue)?;
        let mut light_manager = LightManager::new();
        let shadow_budget = ShadowBudgetManager::new();
        let descriptor_layouts = DescriptorLayouts::new(&device, texture_manager.descriptor_set_layout)?;
        let render_passes = RenderPasses::new(&device, device_ctx.surface_format.format)?;
        let shadow_atlas = ShadowAtlas::new(&device, &mut memory_ctx.allocator, render_passes.shadow, device_ctx.command_pool, device_ctx.queue)?;
        let sun_shadow = SunShadow::new(&device, &mut memory_ctx.allocator, render_passes.shadow, device_ctx.command_pool, device_ctx.queue)?;
        let lighting_buffers = FrameLightingBuffers::new(&mut memory_ctx.allocator)?;

        let probe_grid = ProbeGrid::new(&mut memory_ctx.allocator, [0.0, 0.0])?;
        let hdr_path: Option<&Path> = Some(Path::new("assets/park1k.hdr"));
        let gi_resources = GIResources::new(&device, &mut memory_ctx, device_ctx.command_pool, device_ctx.queue, hdr_path)?;

        // Phase 8A: Initialize GPU culling resources
        let mut gpu_cull = GpuCullResources::new(&device, &mut memory_ctx.allocator)?;

        // ================================================================
        //  glTF asset loading
        // ================================================================
        // Phase 8B
        {
            let gltf_path = std::path::Path::new("assets/models/ubg/utility_box_02_1k.gltf");
            if gltf_path.exists() {
                match GltfLoader::load(
                    gltf_path,
                    &mut material_library,
                    &mut texture_manager,
                    &mut memory_ctx,
                    device_ctx.command_pool,
                    device_ctx.queue,
                ) {
                    Ok(asset) => {
                        let gltf_sector: SectorCoord = (GLTF_SECTOR_BASE, 0);
                        world.sectors.entry(gltf_sector)
                            .or_insert_with(|| Sector::new(gltf_sector));
                        if let Some(sec) = world.sectors.get_mut(&gltf_sector) {
                            sec.state = SectorState::Ready;
                        }
 
                        let model_transform = crate::scene::identity_matrix();
 
                        for mesh in &asset.meshes {
                            // Phase 8B: Read raw vertex/index data from the temporary
                            // per-mesh buffers, upload into mega buffer, then free temps.
                            let vert_bytes = mesh.vertex_count as u64 * 60; // sizeof(Vertex)
                            let idx_bytes = mesh.index_count as u64 * 4;    // sizeof(u32)
 
                            let mega_alloc = gpu_cull.mega.alloc(vert_bytes, idx_bytes)
                                .expect("Mega buffer full during glTF load");
 
                            // Upload vertex data to mega buffer region (synchronous for init)
                            memory_ctx.transfer.upload_region(
                                &mesh.vertices_raw,
                                gpu_cull.mega.vertex_buffer,
                                mega_alloc.vertex_offset_bytes,
                            ).expect("Failed to upload glTF vertices to mega buffer");
 
                            // Upload index data to mega buffer region
                            memory_ctx.transfer.upload_region(
                                &mesh.indices_raw,
                                gpu_cull.mega.index_buffer,
                                mega_alloc.index_offset_bytes,
                            ).expect("Failed to upload glTF indices to mega buffer");
 
                            // Free temporary per-mesh GPU allocations
                            memory_ctx.allocator.free_buffer(mesh.vertex_alloc.handle);
                            memory_ctx.allocator.free_buffer(mesh.index_alloc.handle);
 
                            let bounds = Aabb::new(
                                [
                                    mesh.aabb_min[0] + model_transform[3][0],
                                    mesh.aabb_min[1] + model_transform[3][1],
                                    mesh.aabb_min[2] + model_transform[3][2],
                                ],
                                [
                                    mesh.aabb_max[0] + model_transform[3][0],
                                    mesh.aabb_max[1] + model_transform[3][1],
                                    mesh.aabb_max[2] + model_transform[3][2],
                                ],
                            );
                            let flags = RenderFlags::STATIC | RenderFlags::SHADOW_CASTER;
 
                            // Phase 8B: MeshRange offsets are mega-buffer-relative
                            let mesh_range = MeshRange {
                                first_index: mega_alloc.base_index(),
                                index_count: mesh.index_count,
                                vertex_offset: mega_alloc.base_vertex(),
                            };
 
                            let obj_id = world.add_object(
                                gltf_sector, bounds,
                                LodChain::single(mesh_range),
                                model_transform, mesh.material_id, flags,
                            );
 
                            let gpu_data = GpuObjectData::new(
                                model_transform,
                                bounds.min, bounds.max,
                                mega_alloc.base_index(),
                                mesh.index_count,
                                mega_alloc.base_vertex(),
                                mesh.material_id,
                                flags,
                            );
                            gpu_cull.queue_dirty(obj_id.0, gpu_data);
                            gpu_cull.total_alive = gpu_cull.total_alive.max(obj_id.0 + 1);
                        }
 
                        // Store mega alloc in sector
                        // (For glTF with multiple meshes, store the first mesh's alloc;
                        //  each mesh gets its own MegaAlloc tracked separately)
                        // Note: for eviction, glTF sectors are special-cased (never evicted)
 
                        println!(
                            "[Renderer] glTF '{}' loaded to mega buffers: {} meshes",
                            gltf_path.display(), asset.meshes.len(),
                        );
                    }
                    Err(e) => {
                        eprintln!("[Renderer] glTF load failed: {}", e);
                    }
                }
            }
        }

        if material_library.is_dirty() {
            material_ssbo.upload(&material_library);
            material_library.clear_dirty();
        }

        let vert_spv = include_bytes!("../shaders/compiled/basic.vert.spv");
        let frag_spv = include_bytes!("../shaders/compiled/basic.frag.spv");
        let depth_vert_spv: &[u8] = include_bytes!("../shaders/compiled/depth.vert.spv");
        let depth_frag_spv: &[u8] = include_bytes!("../shaders/compiled/depth.frag.spv");
        let shadow_vert_spv: &[u8] = include_bytes!("../shaders/compiled/shadow.vert.spv");
        let shadow_frag_spv: &[u8] = include_bytes!("../shaders/compiled/shadow.frag.spv");
        let cluster_comp_spv: &[u8] = include_bytes!("../shaders/compiled/cluster_assign.comp.spv");
        let probe_capture_frag_spv: &[u8] = include_bytes!("../shaders/compiled/probe_capture.frag.spv");
        let sh_project_comp_spv: &[u8] = include_bytes!("../shaders/compiled/sh_project.comp.spv");
        let hbao_comp_spv: &[u8] = include_bytes!("../shaders/compiled/hbao.comp.spv");
        let hbao_blur_comp_spv: &[u8] = include_bytes!("../shaders/compiled/hbao_blur.comp.spv");
        let skybox_vert_spv: &[u8] = include_bytes!("../shaders/compiled/skybox.vert.spv");
        let skybox_frag_spv: &[u8] = include_bytes!("../shaders/compiled/skybox.frag.spv");
        let overlay_vert_spv: &[u8] = include_bytes!("../shaders/compiled/overlay.vert.spv");
        let overlay_frag_spv: &[u8] = include_bytes!("../shaders/compiled/overlay.frag.spv");
        let sun_shadow_vert_spv: &[u8] = include_bytes!("../shaders/compiled/sun_shadow.vert.spv");
        let sun_shadow_frag_spv: &[u8] = include_bytes!("../shaders/compiled/sun_shadow.frag.spv");

        let probe_bake_target = ProbeBakeTarget::new(
            &device, &memory_ctx.allocator, render_passes.probe_capture,
            probe_grid.ssbo_buffer(), probe_grid.ssbo_size(),
            sh_project_comp_spv,
            device_ctx.command_pool, device_ctx.queue,
        )?;

        let pipelines = Pipelines::new(&device, &descriptor_layouts, &render_passes,
            vert_spv, frag_spv, depth_vert_spv, depth_frag_spv, shadow_vert_spv, shadow_frag_spv,
            cluster_comp_spv, probe_capture_frag_spv, skybox_vert_spv, skybox_frag_spv,
            sun_shadow_vert_spv, sun_shadow_frag_spv)?;

        let (normal_image, normal_memory, normal_view) = Self::create_normal_image(
            &device, &memory_ctx.allocator,
            device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height,
        )?;

        let framebuffers = PassFramebuffers::new(&device, &render_passes, &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view, normal_view, device_ctx.swapchain_extent)?;

        let inv_proj = crate::scene::invert_projection(scene.camera.get_projection_matrix());
        let mut hbao_pass = HbaoPass::new(
            &device,
            &mut memory_ctx.allocator,
            device_ctx.depth_image_view,
            normal_view,
            device_ctx.swapchain_extent,
            scene.camera.get_projection_matrix(),
            inv_proj,
            hbao_comp_spv,
            hbao_blur_comp_spv,
        )?;

        let profiler = GpuProfiler::new(
            &device,
            device_ctx.timestamp_period,
            device_ctx.timestamp_valid_bits,
        )?;

        let overlay = DebugOverlay::new(
            &device,
            &memory_ctx.allocator,
            render_passes.lighting,
            device_ctx.command_pool,
            device_ctx.queue,
            overlay_vert_spv,
            overlay_frag_spv,
        )?;

        let global_ubo_size = std::mem::size_of::<GlobalUbo>() as u64;
        let cluster_params_size = std::mem::size_of::<ClusterParamsUbo>() as u64;
        // Phase 8A: per_draw_ubo_size removed — SSBO replaces per-draw UBO
        let material_ssbo_size = (material_library.count() * std::mem::size_of::<MaterialData>()) as u64;
        let probe_grid_params_size = std::mem::size_of::<GpuProbeGridParams>() as u64;

        // Phase 8A: per_draw_ubo_range parameter removed from FrameDescriptors::new()
        let frame_descriptors = FrameDescriptors::new(&device, &descriptor_layouts, memory_ctx.ring.buffer,
            global_ubo_size, cluster_params_size,
            material_ssbo.buffer, material_ssbo_size.max(128),
            &lighting_buffers, shadow_atlas.sampling_view, shadow_atlas.shadow_sampler,
            sun_shadow.sampling_view, sun_shadow.sampler,
            probe_grid.ssbo_buffer(), probe_grid.ssbo_size(),
            probe_grid_params_size,
            gi_resources.brdf_lut_view, gi_resources.brdf_lut_sampler,
            gi_resources.irradiance_view, gi_resources.irradiance_sampler,
            gi_resources.prefiltered_view, gi_resources.prefiltered_sampler,
            hbao_pass.ao_sampled_view, hbao_pass.ao_sampler,
        )?;

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

        let sun_dir = crate::scene::normalize(SUN_DIR);
        let sun_light = Light::directional(sun_dir, SUN_COLOR, SUN_INTENSITY);
        light_manager.register((i32::MAX, i32::MAX), vec![sun_light]);

        let dynamic_sector: (i32, i32) = (i32::MAX - 2, i32::MAX - 2);
        let dynamic_light_indices;
        {
            // Phase 8C perf: tighter radii reduce cluster overlap.
            let mut dl1 = Light::point([25.0, 6.0, 0.0], [1.0, 0.7, 0.3], 280.0, 25.0);
            dl1.shadow_capable = true;
            let mut dl2 = Light::point([0.0, 8.0, 35.0], [0.3, 0.5, 1.0], 250.0, 28.0);
            dl2.shadow_capable = true;
            let mut dl3 = Light::point([18.0, 4.0, 0.0], [1.0, 0.3, 0.8], 220.0, 22.0);
            dl3.shadow_capable = true;
            dynamic_light_indices = light_manager.register(dynamic_sector, vec![dl1, dl2, dl3]);
        }

        let spawned_light_sector: SectorCoord = (i32::MAX - 3, i32::MAX - 3);

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        println!(
            "[Renderer] GPU-driven culling initialized. {} materials, {} probes.",
            material_library.count(), probe_grid.probe_count(),
        );

        Ok(Self {
            device, memory_ctx, scene, world,
            pending_sectors: Vec::new(),
            material_library, material_ssbo, texture_manager,
            light_manager, shadow_budget, shadow_atlas, lighting_buffers,
            shadow_assignments: HashMap::new(),
            sun_shadow,
            sun_direction: sun_dir,
            dynamic_light_indices,
            dynamic_light_time: 0.0,
            probe_grid, gi_resources, probe_bake_target,
            gpu_cull,
            descriptor_layouts, render_passes, pipelines, framebuffers,
            frame_descriptors, command_buffers,
            image_available, render_finished, in_flight_fences,
            current_frame: 0, global_frame: 0,
            rng: SimpleRng::new(seed),
            spawned_light_count: 0,
            spawned_baked_light_count: 0,
            spawned_light_sector,
            spawned_geometry_count: 0,
            next_dynamic_sector: 0,
            normal_image, normal_memory, normal_view,
            hbao_pass,
            has_hdr_skybox: hdr_path.is_some(),
            profiler,
            overlay,
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

    fn upload_sector_batch(
        &mut self, sector: SectorCoord, descriptors: Vec<crate::world::ObjectDescriptor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
 
        let vert_bytes = (all_verts.len() * std::mem::size_of::<Vertex>()) as u64;
        let idx_bytes = (all_indices.len() * std::mem::size_of::<u32>()) as u64;
 
        // Phase 8B: sub-allocate from mega buffers
        let mega_alloc = self.gpu_cull.mega.alloc(vert_bytes, idx_bytes)
            .ok_or_else(|| format!(
                "Mega buffer full: need {} VB + {} IB bytes",
                vert_bytes, idx_bytes,
            ))?;
 
        // Async upload vertex data to mega VB sub-region
        let vert_raw = unsafe {
            std::slice::from_raw_parts(
                all_verts.as_ptr() as *const u8,
                vert_bytes as usize,
            )
        };
        let vticket = self.memory_ctx.transfer.upload_buffer_region_async(
            vert_raw,
            self.gpu_cull.mega.vertex_buffer,
            mega_alloc.vertex_offset_bytes,
        )?;
 
        // Async upload index data to mega IB sub-region
        let idx_raw = unsafe {
            std::slice::from_raw_parts(
                all_indices.as_ptr() as *const u8,
                idx_bytes as usize,
            )
        };
        let iticket = self.memory_ctx.transfer.upload_buffer_region_async(
            idx_raw,
            self.gpu_cull.mega.index_buffer,
            mega_alloc.index_offset_bytes,
        )?;
 
        if let Some(sec) = self.world.sectors.get_mut(&sector) {
            sec.state = SectorState::Streaming;
            sec.mega_alloc = Some(mega_alloc);
        }
 
        self.pending_sectors.push(PendingSectorUpload {
            sector,
            mega_alloc,
            vertex_ticket: vticket,
            index_ticket: iticket,
            objects,
        });
 
        Ok(())
    }

    /// Phase 8B
    fn poll_uploads(&mut self) {
        let mut completed: Vec<PendingSectorUpload> = Vec::new();
 
        let transfer = &self.memory_ctx.transfer;
 
        self.pending_sectors.retain_mut(|upload| {
            let done = transfer.is_complete(&upload.vertex_ticket)
                && transfer.is_complete(&upload.index_ticket);
            if done {
                completed.push(PendingSectorUpload {
                    sector: upload.sector,
                    mega_alloc: upload.mega_alloc,
                    vertex_ticket: upload.vertex_ticket.clone(),
                    index_ticket: upload.index_ticket.clone(),
                    objects: std::mem::take(&mut upload.objects),
                });
                false
            } else { true }
        });
 
        for upload in completed {
            let ma = upload.mega_alloc;
 
            for pobj in &upload.objects {
                // Phase 8B: Offsets are mega-buffer-relative.
                // Each object's within-sector offsets are added to the sector's
                // mega buffer base offsets.
                let mega_first_index = ma.base_index() + pobj.first_index;
                let mega_vertex_offset = ma.base_vertex() + pobj.vertex_offset;
 
                let mesh_range = MeshRange {
                    first_index: mega_first_index,
                    index_count: pobj.index_count,
                    vertex_offset: mega_vertex_offset,
                };
 
                let obj_id = self.world.add_object(
                    upload.sector, pobj.bounds,
                    LodChain::single(mesh_range),
                    pobj.transform, pobj.material_id, pobj.flags,
                );
 
                let gpu_data = GpuObjectData::new(
                    pobj.transform,
                    pobj.bounds.min,
                    pobj.bounds.max,
                    mega_first_index,
                    pobj.index_count,
                    mega_vertex_offset,
                    pobj.material_id,
                    pobj.flags,
                );
                self.gpu_cull.queue_dirty(obj_id.0, gpu_data);
                self.gpu_cull.total_alive = self.gpu_cull.total_alive.max(obj_id.0 + 1);
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
            // Phase 8C.2: Invalidate shadow cache — new geometry/lights may affect shadows.
            self.shadow_budget.invalidate_all();
            self.probe_grid.bake_sector_probes(upload.sector, &self.light_manager);
        }
    }

    fn register_sector_lights(&mut self, sector: SectorCoord) {
        if sector.0 >= DYNAMIC_SECTOR_BASE || sector.0 >= i32::MAX - 10 {
            return;
        }
        let is_demo_origin = sector.0.abs() <= 1 && sector.1.abs() <= 1;
        let tiles = (SECTOR_SIZE / GROUND_TILE_SIZE) as i32;
        let bx = sector.0 * tiles;
        let bz = sector.1 * tiles;
        let mut lights = Vec::new();
        for dx in 0..tiles { for dz in 0..tiles {
            let cx = (bx+dx) as f32 * GROUND_TILE_SIZE + GROUND_TILE_SIZE * 0.5;
            let cz = (bz+dz) as f32 * GROUND_TILE_SIZE + GROUND_TILE_SIZE * 0.5;

            // Dynamic shadow-casters only outside demo origin (avoids competing
            // with the 3 orbiting showcase lights for shadow slots).
            if !is_demo_origin {
                // Phase 8C perf: radius 50→30 cuts cluster overlap ~4.6×.
                // Intensity 120→180 compensates for tighter falloff window.
                let mut l = Light::point([cx,12.0,cz], [1.0,0.95,0.85], 180.0, 30.0);
                l.shadow_capable = true;
                lights.push(l);
            }

            // Phase 8C.1: Baked ambient fill lights — always emitted, even at origin.
            // Zero shadow cost: no cubemap samples, no slot competition.
            // Phase 8C perf: tighter radii reduce cluster overlap.
            let baked1 = Light::baked_point(
                [cx - 10.0, 3.0, cz + 10.0],
                [0.8, 0.75, 0.65], // warm ambient fill
                1.2,               // was 0.8, boosted for tighter radius
                10.0,              // was 15.0
            );
            lights.push(baked1);

            let baked2 = Light::baked_point(
                [cx + 10.0, 5.0, cz - 10.0],
                [0.65, 0.7, 0.85], // cool ambient fill
                1.0,               // was 0.6, boosted for tighter radius
                8.0,               // was 12.0
            );
            lights.push(baked2);
        }}
        if !lights.is_empty() {
            self.light_manager.register(sector, lights);
        }
    }

    fn update_dynamic_lights(&mut self, dt: f32) {
        self.dynamic_light_time += dt;
        let t = self.dynamic_light_time;

        struct Orbit { radius: f32, height: f32, speed: f32 }
        const ORBITS: [Orbit; 3] = [
            Orbit { radius: 25.0, height: 6.0, speed:  0.37 },
            Orbit { radius: 35.0, height: 8.0, speed: -0.53 },
            Orbit { radius: 18.0, height: 4.0, speed:  0.71 },
        ];

        for (i, orb) in ORBITS.iter().enumerate() {
            if i >= self.dynamic_light_indices.len() { break; }
            let idx = self.dynamic_light_indices[i];
            let angle = t * orb.speed;
            let new_pos = [
                orb.radius * angle.cos(),
                orb.height,
                orb.radius * angle.sin(),
            ];
            if let Some(light) = self.light_manager.get_mut(idx) {
                light.position = new_pos;
            }
        }
    }

    /// Phase 8B
    fn evict_distant_sectors(&mut self, camera_xz: [f32; 2]) {
        let r2 = EVICTION_RADIUS * EVICTION_RADIUS;
        let to_evict: Vec<SectorCoord> = self.world.sectors.iter()
            .filter(|(_, s)| s.state == SectorState::Ready)
            .filter(|&(&c, _)| c.0 < DYNAMIC_SECTOR_BASE && c.0 < i32::MAX - 10)
            .filter(|(_, s)| s.distance_sq_to_point_xz(camera_xz) > r2)
            .map(|(&c,_)| c).collect();
 
        for coord in to_evict {
            // Phase 8B: Free mega buffer sub-allocation
            if let Some(sec) = self.world.sectors.get(&coord) {
                if let Some(ref ma) = sec.mega_alloc {
                    self.gpu_cull.mega.free(ma);
                }
            }
 
            // Zero out GpuObjectData for all objects in this sector
            let ids = self.world.evict_sector(coord);
            for &id in &ids {
                self.gpu_cull.queue_dirty(id.0, GpuObjectData::dead());
            }
 
            self.light_manager.unregister(coord);
            // Phase 8C.2: Invalidate shadow cache — evicted geometry may leave stale shadows.
            self.shadow_budget.invalidate_all();
            self.probe_grid.invalidate_sector(coord);
            self.world.sectors.remove(&coord);
 
            if !ids.is_empty() {
                println!("[Renderer] Evicted sector ({},{}) — {} objects", coord.0, coord.1, ids.len());
            }
        }
    }

    // ================================================================
    //  Frame rendering — Phase 8A GPU-driven culling
    // ================================================================

    pub fn render(&mut self, device_ctx: &DeviceContext, input: &InputState, dt: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.global_frame += 1;
        self.process_actions(input);

        self.update_streaming();
        self.poll_uploads();
        self.texture_manager.poll_pending(&self.memory_ctx);
        self.probe_grid.upload_if_dirty();

        if self.material_library.is_dirty() {
            self.material_ssbo.upload(&self.material_library);
            self.material_library.clear_dirty();
        }

        unsafe {
            // ════════════════════════════════════════════════════════════
            // STEP 1: Fence wait (MUST precede flush_dirty — §3.3)
            // ════════════════════════════════════════════════════════════
            self.device.wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)?;

            #[cfg(debug_assertions)]
            self.gpu_cull.assert_fence_waited();

            self.scene.update(dt, input);
            self.update_dynamic_lights(dt);

            // ════════════════════════════════════════════════════════════
            // STEP 3: Flush dirty objects to SSBO (AFTER fence wait — §3.3)
            // ════════════════════════════════════════════════════════════
            self.gpu_cull.flush_dirty(&mut self.memory_ctx);

            self.profiler.read_results(&self.device, self.current_frame);
            self.memory_ctx.ring.begin_frame(self.current_frame);

            let view_mat = self.scene.camera.get_view_matrix();
            let proj_mat = self.scene.camera.get_projection_matrix();
            let camera_pos = self.scene.camera.position;

            let camera_vp = crate::scene::multiply_matrices(view_mat, proj_mat);
            let frustum = crate::scene::extract_frustum_planes_from_vp(&camera_vp);

            // Phase 8A: CPU cull_and_select_lod removed — GPU does this now

            // Phase 8C.5: Adaptive budget — adjust effective_max_slots before assignment.
            self.shadow_budget.adapt_budget();

            self.shadow_assignments = self.shadow_budget.assign(&self.light_manager, camera_pos);

            // Phase 8C.2: Update shadow cache state after slot assignment.
            self.shadow_budget.update_cache(&self.light_manager, self.global_frame);

            let active_light_count = self.light_manager.cull_and_sort(camera_pos, &frustum, &self.shadow_assignments);

            self.lighting_buffers.upload_lights_direct(
                self.current_frame,
                &self.light_manager.ssbo_header(),
                self.light_manager.gpu_lights(),
            );

            let (sun_view, sun_proj) = compute_sun_shadow_matrices(self.sun_direction, camera_pos);
            let sun_vp = crate::scene::multiply_matrices(sun_view, sun_proj);

            let global_ubo = GlobalUbo {
                view: view_mat, proj: proj_mat,
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], self.global_frame as f32],
                sun_light_vp: sun_vp,
                sun_direction: [self.sun_direction[0], self.sun_direction[1], self.sun_direction[2], 1.0],
            };
            let g_off = self.memory_ctx.ring.push_data(&global_ubo).expect("Ring: GlobalUbo").offset as u32;

            let cluster_params = ClusterParamsUbo::new(view_mat, proj_mat,
                self.scene.camera.near, self.scene.camera.far,
                device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height, active_light_count);
            let c_off = self.memory_ctx.ring.push_data(&cluster_params).expect("Ring: ClusterParams").offset as u32;

            let probe_params = self.probe_grid.gpu_params();
            let p_off = self.memory_ctx.ring.push_data(&probe_params).expect("Ring: ProbeGridParams").offset as u32;

            let dyn_off = [g_off, c_off, p_off];

            let (image_index, _) = device_ctx.swapchain_loader.acquire_next_image(
                device_ctx.swapchain, u64::MAX, self.image_available[self.current_frame], vk::Fence::null())?;
            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            let cmd = self.command_buffers[self.current_frame];
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            self.profiler.reset_queries(&self.device, cmd, self.current_frame);

            let viewport = vk::Viewport { x:0.0, y:0.0,
                width: device_ctx.swapchain_extent.width as f32,
                height: device_ctx.swapchain_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0 };
            let scissor = vk::Rect2D { offset: vk::Offset2D{x:0,y:0}, extent: device_ctx.swapchain_extent };

            let f = self.current_frame;

            // ════════════════════════════════════════════════════════════
            // STEP 4: Zero count buffers (vkCmdFillBuffer)
            // ════════════════════════════════════════════════════════════
            let count_size = (MAX_BUFFER_GROUPS as u64) * 4;
            self.device.cmd_fill_buffer(cmd, self.gpu_cull.opaque_counts[f], 0, count_size, 0);
            self.device.cmd_fill_buffer(cmd, self.gpu_cull.shadow_counts[f], 0, count_size, 0);

            // ════════════════════════════════════════════════════════════
            // STEP 5: Pre-cull barrier (§3.4)
            // ════════════════════════════════════════════════════════════
            let transfer_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

            let ssbo_barrier = if self.gpu_cull.needs_host_barrier {
                vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            } else {
                vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            };

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[transfer_barrier, ssbo_barrier],
                &[],
                &[],
            );
            self.gpu_cull.needs_host_barrier = false;

            // ════════════════════════════════════════════════════════════
            // STEP 6: GPU Cull Dispatch
            // ════════════════════════════════════════════════════════════
            self.gpu_cull.update_group_base_offsets_inline(&device_ctx.device, cmd);

            let cull_push = CullPushConstants::new(&frustum, camera_pos, self.gpu_cull.total_alive);

            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.gpu_cull.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, self.gpu_cull.pipeline_layout, 0,
                &[self.gpu_cull.descriptor_sets[f]], &[],
            );
            self.device.cmd_push_constants(
                cmd, self.gpu_cull.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0,
                std::slice::from_raw_parts(&cull_push as *const _ as *const u8, std::mem::size_of_val(&cull_push)),
            );

            let workgroups = (self.gpu_cull.total_alive + 255) / 256;
            self.device.cmd_dispatch(cmd, workgroups, 1, 1);

            // ════════════════════════════════════════════════════════════
            // STEP 7: Post-cull barrier
            // ════════════════════════════════════════════════════════════
            let post_cull_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::VERTEX_SHADER,
                vk::DependencyFlags::empty(),
                &[post_cull_barrier],
                &[],
                &[],
            );

            // ---- Stats ----
            let total_alive = self.gpu_cull.total_alive;
            let active_groups = 1u32; // Phase 8B: single mega buffer group
            if self.global_frame % 60 == 0 {
                println!(
                    "[Frame {:>5}] cam:[{:.1},{:.1},{:.1}] spd:{:.0}  objects: {}  groups: {}  lights: {}/{}  shadows: {}",
                    self.global_frame,
                    camera_pos[0], camera_pos[1], camera_pos[2],
                    self.scene.camera.move_speed,
                    total_alive,
                    active_groups,
                    active_light_count, self.light_manager.total_count(),
                    self.shadow_budget.active_shadow_count(),
                );
            }

            // Phase 7: Collect overlay stats
            if self.overlay.visible {
                self.overlay.collect_profiler_stats(&self.profiler);
                self.overlay.stats.sectors_ready = self.world.ready_sector_count();
                self.overlay.stats.sectors_streaming = self.world.streaming_sector_count();
                self.overlay.stats.sectors_unloaded = self.world.sectors.values()
                    .filter(|s| s.state == SectorState::Unloaded).count();
                self.overlay.stats.in_flight_uploads = self.pending_sectors.len();
                self.overlay.stats.pool_used_mb = self.memory_ctx.allocator.total_used() as f32 / (1024.0 * 1024.0);
                self.overlay.stats.pool_allocated_mb = self.memory_ctx.allocator.total_allocated() as f32 / (1024.0 * 1024.0);
                self.overlay.stats.budget_mb = self.memory_ctx.budget.budget_bytes() as f32 / (1024.0 * 1024.0);
                self.overlay.stats.tracked_mb = self.memory_ctx.budget.tracked_bytes() as f32 / (1024.0 * 1024.0);
                self.overlay.stats.lights_total = self.light_manager.total_count();
                self.overlay.stats.lights_active = active_light_count;
                self.overlay.stats.lights_shadow = self.shadow_budget.active_shadow_count();
                // Phase 8C: Count baked vs dynamic among active lights.
                {
                    let mut baked = 0u32;
                    let mut dynamic = 0u32;
                    for &idx in self.light_manager.active_indices() {
                        if let Some(light) = self.light_manager.get(idx) {
                            if light.category == LightCategory::Baked {
                                baked += 1;
                            } else {
                                dynamic += 1;
                            }
                        }
                    }
                    self.overlay.stats.lights_baked = baked;
                    self.overlay.stats.lights_dynamic = dynamic;
                }
                self.overlay.stats.effective_shadow_slots = self.shadow_budget.effective_max_slots();
                self.overlay.stats.draw_calls_opaque = 1; // Phase 8B: single indirect_count
                self.overlay.stats.draw_calls_shadow = 1;
                self.overlay.stats.staging_fill_pct = self.memory_ctx.transfer.staging_fill_ratio() * 100.0;
                self.overlay.stats.ring_fill_pct = self.memory_ctx.ring.frame_fill_ratio() * 100.0;
                self.overlay.stats.frame_dt_ms = dt * 1000.0;
            }

            // ============================================================
            //  PASS 0: Shadow Pass — Phase 8A indirect draws
            // ============================================================
            self.profiler.begin_pass(&self.device, cmd, PassId::Shadow, self.current_frame);

            // ---- Sun directional shadow ----
            {
                let sun_sv = vk::Viewport { x: 0.0, y: 0.0,
                    width: SUN_SHADOW_SIZE as f32, height: SUN_SHADOW_SIZE as f32,
                    min_depth: 0.0, max_depth: 1.0 };
                let sun_ss = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: SUN_SHADOW_SIZE, height: SUN_SHADOW_SIZE },
                };
                let clear = [vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }];
                let rp = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_passes.shadow)
                    .framebuffer(self.sun_shadow.framebuffer)
                    .render_area(sun_ss)
                    .clear_values(&clear);
                self.device.cmd_begin_render_pass(cmd, &rp, vk::SubpassContents::INLINE);
                self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.sun_shadow);
                self.device.cmd_set_viewport(cmd, 0, &[sun_sv]);
                self.device.cmd_set_scissor(cmd, 0, &[sun_ss]);

                let sun_global = GlobalUbo {
                    view: sun_view, proj: sun_proj,
                    camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], self.global_frame as f32],
                    sun_light_vp: sun_vp,
                    sun_direction: [self.sun_direction[0], self.sun_direction[1], self.sun_direction[2], 1.0],
                };
                let sun_g_off = self.memory_ctx.ring.push_data(&sun_global).expect("Ring: sun shadow GlobalUbo").offset as u32;
                let sun_dyn_off = [sun_g_off, c_off, p_off];
                self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.layout, 0,
                    &[self.frame_descriptors.per_frame_sets[f]], &sun_dyn_off);

                // Phase 8A: Bind object SSBO (set 3) once for all draws
                self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.layout, 3,
                    &[self.gpu_cull.object_ssbo_set], &[]);

                // Phase 8B: single mega VB/IB bind
                self.device.cmd_bind_vertex_buffers(cmd, 0,
                    &[self.gpu_cull.mega.vertex_buffer], &[0]);
                self.device.cmd_bind_index_buffer(cmd,
                    self.gpu_cull.mega.index_buffer, 0, vk::IndexType::UINT32);
 
                // Phase 8B: single indirect_count call (group 0, offset 0)
                self.device.cmd_draw_indexed_indirect_count(
                    cmd,
                    self.gpu_cull.shadow_cmds[f],
                    0, // cmd buffer offset: start of group 0
                    self.gpu_cull.shadow_counts[f],
                    0, // count buffer offset: index 0
                    MAX_INDIRECT_DRAWS,
                    INDIRECT_COMMAND_STRIDE,
                );
                self.device.cmd_end_render_pass(cmd);
            }

            // ---- Point light cube map shadows (Phase 8C.2: with caching) ----
            let sv = vk::Viewport { x:0.0,y:0.0, width:SHADOW_MAP_SIZE as f32, height:SHADOW_MAP_SIZE as f32, min_depth:0.0, max_depth:1.0 };
            let ss = vk::Rect2D { offset:vk::Offset2D{x:0,y:0}, extent:vk::Extent2D{width:SHADOW_MAP_SIZE,height:SHADOW_MAP_SIZE} };

            // Collect assigned slots into a Vec to avoid borrowing self.shadow_budget
            // while also needing &mut self for mark_slot_clean.
            let assigned: Vec<(u32, usize)> = self.shadow_budget.assigned_slots().collect();

            // Phase 8E stability: Cap dirty slot re-renders per frame.
            // Prevents spikes when invalidate_all() fires on sector load/unload.
            // 8 slots × 6 faces × ~0.02ms/face ≈ 1ms max spike.
            const MAX_DIRTY_RENDERS_PER_FRAME: u32 = 8;
            let mut dirty_rendered = 0u32;

            for (slot, light_idx) in assigned {
                // Phase 8C.2: Shadow caching — skip clean (unchanged) slots.
                if !self.shadow_budget.is_slot_dirty(slot) {
                    continue;
                }

                // Stagger: defer remaining dirty slots to subsequent frames.
                if dirty_rendered >= MAX_DIRTY_RENDERS_PER_FRAME {
                    break;
                }

                let Some(light) = self.light_manager.get(light_idx) else { continue };
                if light.light_type == LightType::Directional { continue; }
                let push = ShadowPushConstants { light_pos: light.position, light_radius: light.radius };
                let lp = light.position;

                for face in 0..6u32 {
                    let (fv, fp) = cube_face_matrices(lp, light.radius, face);

                    let face_global = GlobalUbo {
                        view: fv, proj: fp,
                        camera_pos: [lp[0], lp[1], lp[2], self.global_frame as f32],
                        sun_light_vp: sun_vp,
                        sun_direction: [self.sun_direction[0], self.sun_direction[1], self.sun_direction[2], 1.0],
                    };
                    let face_g_off = self.memory_ctx.ring.push_data(&face_global).expect("Ring: shadow face GlobalUbo").offset as u32;
                    let face_dyn_off = [face_g_off, c_off, p_off];

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

                    self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.layout, 0,
                        &[self.frame_descriptors.per_frame_sets[f]], &face_dyn_off);

                    // Phase 8A: Bind object SSBO (set 3)
                    self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.layout, 3,
                        &[self.gpu_cull.object_ssbo_set], &[]);

                    // Phase 8B: single mega VB/IB bind
                    self.device.cmd_bind_vertex_buffers(cmd, 0,
                        &[self.gpu_cull.mega.vertex_buffer], &[0]);
                    self.device.cmd_bind_index_buffer(cmd,
                        self.gpu_cull.mega.index_buffer, 0, vk::IndexType::UINT32);

                    // Phase 8C §2 bug fix: use shadow caster commands, not opaque
                    self.device.cmd_draw_indexed_indirect_count(
                        cmd,
                        self.gpu_cull.shadow_cmds[f],
                        0,
                        self.gpu_cull.shadow_counts[f],
                        0,
                        MAX_INDIRECT_DRAWS,
                        INDIRECT_COMMAND_STRIDE,
                    );
                    self.device.cmd_end_render_pass(cmd);
                }

                // Phase 8C.2: Mark slot as clean after all 6 faces rendered.
                self.shadow_budget.mark_slot_clean(slot, self.global_frame);
                dirty_rendered += 1;
            }

            self.profiler.end_pass(&self.device, cmd, PassId::Shadow, self.current_frame);

            // Phase 8C.5: Feed measured shadow pass time into the budget manager.
            {
                let shadow_ms = self.profiler.latest_ms(PassId::Shadow);
                self.shadow_budget.set_last_shadow_time(shadow_ms);
            }

            // ============================================================
            //  PASS 0.5: GPU Probe Cubemap Capture + SH Projection
            // ============================================================
            {
                let pv = vk::Viewport { x: 0.0, y: 0.0,
                    width: PROBE_CAPTURE_SIZE as f32, height: PROBE_CAPTURE_SIZE as f32,
                    min_depth: 0.0, max_depth: 1.0 };
                let ps = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: PROBE_CAPTURE_SIZE, height: PROBE_CAPTURE_SIZE },
                };

                let mut probes_baked = 0u32;
                while probes_baked < MAX_PROBE_BAKES_PER_FRAME as u32 {
                    let Some((probe_idx, probe_pos)) = self.probe_grid.next_bake_probe() else { break };

                    let face_matrices = ProbeGrid::probe_capture_matrices(probe_pos);

                    for face in 0..6u32 {
                        let (face_view, face_proj) = face_matrices[face as usize];

                        let probe_global = GlobalUbo {
                            view: face_view, proj: face_proj,
                            camera_pos: [probe_pos[0], probe_pos[1], probe_pos[2], self.global_frame as f32],
                            sun_light_vp: sun_vp,
                            sun_direction: [self.sun_direction[0], self.sun_direction[1], self.sun_direction[2], 1.0],
                        };
                        let probe_g_off = self.memory_ctx.ring.push_data(&probe_global).expect("Ring: probe GlobalUbo").offset as u32;
                        let probe_dyn_off = [probe_g_off, c_off, p_off];

                        let clear = [
                            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
                            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
                        ];
                        let rp = vk::RenderPassBeginInfo::default()
                            .render_pass(self.render_passes.probe_capture)
                            .framebuffer(self.probe_bake_target.framebuffers[face as usize])
                            .render_area(ps)
                            .clear_values(&clear);

                        self.device.cmd_begin_render_pass(cmd, &rp, vk::SubpassContents::INLINE);
                        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.probe_capture);
                        self.device.cmd_set_viewport(cmd, 0, &[pv]);
                        self.device.cmd_set_scissor(cmd, 0, &[ps]);

                        self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                            self.pipelines.layout, 0,
                            &[self.frame_descriptors.per_frame_sets[f]], &probe_dyn_off);
                        self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                            self.pipelines.layout, 1, &[self.texture_manager.descriptor_set], &[]);
                        self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                            self.pipelines.layout, 2,
                            &[self.frame_descriptors.shadow_map_sets[f]], &[]);

                        // Phase 8A: Bind object SSBO (set 3)
                        self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                            self.pipelines.layout, 3,
                            &[self.gpu_cull.object_ssbo_set], &[]);

                        // Phase 8B: single mega VB/IB bind
                        self.device.cmd_bind_vertex_buffers(cmd, 0,
                            &[self.gpu_cull.mega.vertex_buffer], &[0]);
                        self.device.cmd_bind_index_buffer(cmd,
                            self.gpu_cull.mega.index_buffer, 0, vk::IndexType::UINT32);
 
                        self.device.cmd_draw_indexed_indirect_count(
                            cmd,
                            self.gpu_cull.opaque_cmds[f],
                            0,
                            self.gpu_cull.opaque_counts[f],
                            0,
                            MAX_INDIRECT_DRAWS,
                            INDIRECT_COMMAND_STRIDE,
                        );
                        self.device.cmd_end_render_pass(cmd);
                    }

                    let barrier = vk::ImageMemoryBarrier::default()
                        .image(self.probe_bake_target.color_image)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0, level_count: 1,
                            base_array_layer: 0, layer_count: 6,
                        });
                    self.device.cmd_pipeline_barrier(cmd,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(), &[], &[],
                        std::slice::from_ref(&barrier));

                    let sh_push = SHProjectPushConstants {
                        probe_index: probe_idx,
                        face_size: PROBE_CAPTURE_SIZE,
                    };
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.probe_bake_target.sh_compute_pipeline);
                    self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                        self.probe_bake_target.sh_compute_pipeline_layout, 0,
                        &[self.probe_bake_target.sh_compute_set], &[]);
                    self.device.cmd_push_constants(cmd, self.probe_bake_target.sh_compute_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE, 0,
                        std::slice::from_raw_parts(&sh_push as *const _ as *const u8, std::mem::size_of_val(&sh_push)));
                    self.device.cmd_dispatch(cmd, 1, 1, 1);

                    let ssbo_barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                    self.device.cmd_pipeline_barrier(cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        std::slice::from_ref(&ssbo_barrier), &[], &[]);

                    probes_baked += 1;
                }
            }

            // ============================================================
            //  PASS 1: Depth Pre-Pass + G-Buffer Normal
            // ============================================================
            self.profiler.begin_pass(&self.device, cmd, PassId::Depth, self.current_frame);

            let dc = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.5, 0.5, 0.0, 0.0] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];
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
                &[self.frame_descriptors.per_frame_sets[f], self.texture_manager.descriptor_set], &dyn_off);

            // Phase 8A: Bind object SSBO (set 3)
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout, 3,
                &[self.gpu_cull.object_ssbo_set], &[]);

            // Phase 8B: single mega VB/IB bind
            self.device.cmd_bind_vertex_buffers(cmd, 0,
                &[self.gpu_cull.mega.vertex_buffer], &[0]);
            self.device.cmd_bind_index_buffer(cmd,
                self.gpu_cull.mega.index_buffer, 0, vk::IndexType::UINT32);
 
            self.device.cmd_draw_indexed_indirect_count(
                cmd,
                self.gpu_cull.opaque_cmds[f],
                0,
                self.gpu_cull.opaque_counts[f],
                0,
                MAX_INDIRECT_DRAWS,
                INDIRECT_COMMAND_STRIDE,
            );
            self.device.cmd_end_render_pass(cmd);

            self.profiler.end_pass(&self.device, cmd, PassId::Depth, self.current_frame);

            // ============================================================
            //  PASS 1.5: HBAO
            // ============================================================
            self.profiler.begin_pass(&self.device, cmd, PassId::Hbao, self.current_frame);

            let normal_barrier = vk::ImageMemoryBarrier::default()
                .image(self.normal_image)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], std::slice::from_ref(&normal_barrier),
            );

            HbaoPass::barrier_depth_to_read(&self.device, cmd, device_ctx.depth_image);
            self.hbao_pass.dispatch(&self.device, cmd);
            HbaoPass::barrier_depth_to_attachment(&self.device, cmd, device_ctx.depth_image);

            self.profiler.end_pass(&self.device, cmd, PassId::Hbao, self.current_frame);

            // ============================================================
            //  PASS 2: Cluster Assignment Compute
            // ============================================================
            self.profiler.begin_pass(&self.device, cmd, PassId::Cluster, self.current_frame);

            self.device.cmd_fill_buffer(cmd, self.lighting_buffers.index_ssbo_buffers[f], 0, 16, 0);
            let fb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER|vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[fb,hb], &[], &[]);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.cluster_compute);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.compute_layout, 0,
                &[self.frame_descriptors.per_frame_sets[f]], &dyn_off);
            self.device.cmd_dispatch(cmd, TOTAL_CLUSTERS, 1, 1);
            let pc = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(),
                std::slice::from_ref(&pc), &[], &[]);

            self.profiler.end_pass(&self.device, cmd, PassId::Cluster, self.current_frame);

            // ============================================================
            //  PASS 3: Lighting Pass — Phase 8A indirect draws
            // ============================================================
            self.profiler.begin_pass(&self.device, cmd, PassId::Lighting, self.current_frame);

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
                &[self.frame_descriptors.per_frame_sets[f]], &dyn_off);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.layout, 1,
                &[self.texture_manager.descriptor_set], &[]);
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.layout, 2,
                &[self.frame_descriptors.shadow_map_sets[f]], &[]);

            // Phase 8A: Bind object SSBO (set 3)
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout, 3,
                &[self.gpu_cull.object_ssbo_set], &[]);

            // Phase 8B: single mega VB/IB bind
            self.device.cmd_bind_vertex_buffers(cmd, 0,
                &[self.gpu_cull.mega.vertex_buffer], &[0]);
            self.device.cmd_bind_index_buffer(cmd,
                self.gpu_cull.mega.index_buffer, 0, vk::IndexType::UINT32);
 
            self.device.cmd_draw_indexed_indirect_count(
                cmd,
                self.gpu_cull.opaque_cmds[f],
                0,
                self.gpu_cull.opaque_counts[f],
                0,
                MAX_INDIRECT_DRAWS,
                INDIRECT_COMMAND_STRIDE,
            );

            // Skybox
            if self.has_hdr_skybox {
                self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipelines.skybox);
                self.device.cmd_set_viewport(cmd, 0, &[viewport]);
                self.device.cmd_set_scissor(cmd, 0, &[scissor]);
                self.device.cmd_draw(cmd, 36, 1, 0, 0);
            }

            // Debug overlay
            self.overlay.record_commands(
                &self.device, cmd, &mut self.memory_ctx.ring,
                device_ctx.swapchain_extent.width as f32,
                device_ctx.swapchain_extent.height as f32,
            );

            self.device.cmd_end_render_pass(cmd);

            self.profiler.end_pass(&self.device, cmd, PassId::Lighting, self.current_frame);

            // Post pass placeholder
            self.profiler.begin_pass(&self.device, cmd, PassId::Post, self.current_frame);
            self.profiler.end_pass(&self.device, cmd, PassId::Post, self.current_frame);

            self.device.end_command_buffer(cmd)?;

            // ---- Submit + Present ----
            let ws = [self.image_available[f]];
            let wst = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let ss2 = [self.render_finished[f]];
            let sub = vk::SubmitInfo::default().wait_semaphores(&ws).wait_dst_stage_mask(&wst)
                .command_buffers(std::slice::from_ref(&cmd)).signal_semaphores(&ss2);
            self.device.queue_submit(device_ctx.queue, &[sub], self.in_flight_fences[f])?;

            let pres = vk::PresentInfoKHR::default().wait_semaphores(&ss2)
                .swapchains(std::slice::from_ref(&device_ctx.swapchain))
                .image_indices(std::slice::from_ref(&image_index));
            device_ctx.swapchain_loader.queue_present(device_ctx.queue, &pres)?;

            #[cfg(debug_assertions)]
            self.gpu_cull.reset_fence_flag();

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        Ok(())
    }

    // Phase 8A: record_draw_list removed — replaced by indirect draws

    // ================================================================
    //  Keybind action processing
    // ================================================================

    fn process_actions(&mut self, input: &InputState) {
        for action in &input.actions {
            match action {
                InputAction::SpawnLight => self.spawn_random_light(),
                InputAction::SpawnGeometry => self.spawn_random_geometry(),
                InputAction::ToggleOverlay => {
                    self.overlay.visible = !self.overlay.visible;
                    println!("[Overlay] {}", if self.overlay.visible { "shown" } else { "hidden" });
                }
            }
        }
    }

    fn spawn_random_light(&mut self) {
        // Phase 8C.1: After dynamic limit, spawn baked (no-shadow) fill lights
        // that still contribute to scene lighting.
        let is_baked = self.spawned_light_count >= MAX_SPAWNED_LIGHTS;

        if is_baked && self.spawned_baked_light_count >= MAX_SPAWNED_BAKED_LIGHTS {
            println!(
                "[Spawn] Baked light limit reached ({}/{})",
                self.spawned_baked_light_count, MAX_SPAWNED_BAKED_LIGHTS,
            );
            return;
        }

        let cam = self.scene.camera.position;
        let x = cam[0] + self.rng.range(-SPAWN_RADIUS, SPAWN_RADIUS);
        let y = self.rng.range(SPAWN_LIGHT_HEIGHT_MIN, SPAWN_LIGHT_HEIGHT_MAX);
        let z = cam[2] + self.rng.range(-SPAWN_RADIUS, SPAWN_RADIUS);

        let r = self.rng.range(0.3, 1.0);
        let g = self.rng.range(0.3, 1.0);
        let b = self.rng.range(0.3, 1.0);

        if is_baked {
            // Phase 8C perf: baked spawns use tighter radii (no shadow cost
            // but still occupies cluster slots → fragment shader cost).
            let intensity = self.rng.range(100.0, 250.0);
            let radius = self.rng.range(10.0, 25.0);
            let light = Light::baked_point([x, y, z], [r, g, b], intensity, radius);
            self.light_manager.register(self.spawned_light_sector, vec![light]);
            self.spawned_baked_light_count += 1;
            println!(
                "[Spawn] Baked light #{} at [{:.1}, {:.1}, {:.1}] color=[{:.2},{:.2},{:.2}] I={:.0} R={:.0} (no shadow)",
                self.spawned_baked_light_count, x, y, z, r, g, b, intensity, radius,
            );
        } else {
            // Phase 8C perf: tighter radii reduce cluster overlap.
            let intensity = self.rng.range(120.0, 300.0);
            let radius = self.rng.range(15.0, 35.0);
            let mut light = Light::point([x, y, z], [r, g, b], intensity, radius);
            light.shadow_capable = true;
            self.light_manager.register(self.spawned_light_sector, vec![light]);
            self.spawned_light_count += 1;
            println!(
                "[Spawn] Dynamic light #{} at [{:.1}, {:.1}, {:.1}] color=[{:.2},{:.2},{:.2}] I={:.0} R={:.0}",
                self.spawned_light_count, x, y, z, r, g, b, intensity, radius,
            );
        }
    }

    fn spawn_random_geometry(&mut self) {
        if self.spawned_geometry_count >= MAX_SPAWNED_GEOMETRY {
            println!("[Spawn] Geometry limit reached ({}/{})", self.spawned_geometry_count, MAX_SPAWNED_GEOMETRY);
            return;
        }

        let cam = self.scene.camera.position;
        let x = cam[0] + self.rng.range(-SPAWN_RADIUS, SPAWN_RADIUS);
        let z = cam[2] + self.rng.range(-SPAWN_RADIUS, SPAWN_RADIUS);
        let pos = [x, 0.0, z];

        let shape: u32 = (self.rng.next_u64() % 3) as u32;
        let mat = self.rng.pick(&[2u32, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

        let r = self.rng.range(0.3, 1.0);
        let g = self.rng.range(0.3, 1.0);
        let b = self.rng.range(0.3, 1.0);
        let color = [r, g, b];

        let scale = self.rng.range(1.5, 6.0);
        let y_rot = self.rng.range(0.0, std::f32::consts::TAU);

        let raw = match shape {
            0 => make_cube(color, mat),
            1 => {
                let height = self.rng.range(1.5, 3.0);
                make_pyramid(color, height, mat)
            }
            _ => {
                let height = self.rng.range(3.0, 12.0);
                let radius = self.rng.range(0.3, 0.8);
                make_column(color, height, radius, mat)
            }
        };

        let transform = demo_transform(pos, scale, y_rot);
        let bounds = Aabb::from_vertices(&raw.vertices, &transform);
        let flags = RenderFlags::STATIC | RenderFlags::SHADOW_CASTER;

        let descriptor = ObjectDescriptor {
            vertices: raw.vertices,
            indices: raw.indices,
            transform,
            material_id: mat,
            flags,
            bounds,
        };

        let sector_coord: SectorCoord = (DYNAMIC_SECTOR_BASE + self.next_dynamic_sector, 0);
        self.next_dynamic_sector += 1;

        self.world.sectors.entry(sector_coord)
            .or_insert_with(|| Sector::new(sector_coord));
        if let Some(sec) = self.world.sectors.get_mut(&sector_coord) {
            sec.state = SectorState::Unloaded;
        }

        let shape_name = match shape { 0 => "cube", 1 => "pyramid", _ => "column" };

        match self.upload_sector_batch(sector_coord, vec![descriptor]) {
            Ok(()) => {
                self.spawned_geometry_count += 1;
                println!(
                    "[Spawn] Geometry #{} ({}) at [{:.1}, 0, {:.1}] scale={:.1} mat={}",
                    self.spawned_geometry_count, shape_name, x, z, scale, mat,
                );
            }
            Err(e) => {
                eprintln!("[Spawn] Geometry upload failed: {}", e);
                if let Some(sec) = self.world.sectors.get_mut(&sector_coord) {
                    sec.state = SectorState::Failed;
                }
            }
        }
    }

    pub fn recreate_framebuffers(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { self.device.device_wait_idle()? };
        self.framebuffers.destroy(&self.device);

        unsafe {
            self.device.destroy_image_view(self.normal_view, None);
            self.device.destroy_image(self.normal_image, None);
            self.device.free_memory(self.normal_memory, None);
        }
        let (ni, nm, nv) = Self::create_normal_image(
            &self.device, &self.memory_ctx.allocator,
            device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height,
        )?;
        self.normal_image = ni;
        self.normal_memory = nm;
        self.normal_view = nv;

        self.framebuffers = PassFramebuffers::new(&self.device, &self.render_passes,
            &device_ctx.swapchain_image_views, device_ctx.depth_image_view,
            self.normal_view, device_ctx.swapchain_extent)?;
        self.scene.camera.update_aspect(device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height);
        let inv_proj = crate::scene::invert_projection(self.scene.camera.get_projection_matrix());
        self.hbao_pass.resize(
            &self.device,
            &mut self.memory_ctx.allocator,
            device_ctx.depth_image_view,
            self.normal_view,
            device_ctx.swapchain_extent,
            self.scene.camera.get_projection_matrix(),
            inv_proj,
        )?;
        Ok(())
    }

    pub fn materials_mut(&mut self) -> &mut MaterialLibrary { &mut self.material_library }
    pub fn materials(&self) -> &MaterialLibrary { &self.material_library }
    pub fn texture_manager_mut(&mut self) -> &mut TextureManager { &mut self.texture_manager }
    pub fn light_manager_mut(&mut self) -> &mut LightManager { &mut self.light_manager }

    fn create_normal_image(
        device: &Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16_SFLOAT)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
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

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R16G16_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    }),
                None,
            )?
        };

        println!("[Renderer] G-buffer normal image created: {}×{} R16G16_SFLOAT", width, height);
        Ok((image, memory, view))
    }

    fn allocate_command_buffers(device: &Device, pool: vk::CommandPool) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
        let info = vk::CommandBufferAllocateInfo::default().command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
        unsafe { Ok(device.allocate_command_buffers(&info)?) }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) { unsafe {
        let _ = self.device.device_wait_idle();
        // Phase 8A: Destroy GPU cull resources
        self.gpu_cull.destroy(&mut self.memory_ctx.allocator);
        for &f in &self.in_flight_fences { self.device.destroy_fence(f, None); }
        for &s in &self.render_finished { self.device.destroy_semaphore(s, None); }
        for &s in &self.image_available { self.device.destroy_semaphore(s, None); }
        self.frame_descriptors.destroy(&self.device);
        self.framebuffers.destroy(&self.device);
        self.pipelines.destroy(&self.device);
        self.render_passes.destroy(&self.device);
        self.descriptor_layouts.destroy(&self.device);
        self.shadow_atlas.destroy(&self.device);
        self.sun_shadow.destroy(&self.device);
        self.gi_resources.destroy(&self.device);
        self.device.destroy_image_view(self.normal_view, None);
        self.device.destroy_image(self.normal_image, None);
        self.device.free_memory(self.normal_memory, None);
        self.hbao_pass.destroy(&self.device, &mut self.memory_ctx.allocator);
        self.probe_bake_target.destroy(&self.device);
        self.profiler.destroy(&self.device);
        self.overlay.destroy(&self.device);
        self.lighting_buffers.destroy(&mut self.memory_ctx.allocator);
    }}
}