use ash::{vk, Device};
use std::sync::Arc;
use crate::memory::{MemoryContext, AllocationPurpose};
use crate::device::DeviceContext;
use crate::scene::{Scene, Vertex, UniformBufferObject};

pub struct Renderer {
    device: Arc<Device>,
    memory_ctx: MemoryContext,
    
    // Scene data
    scene: Scene,
    
    // Buffers
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffer: vk::Buffer,
    uniform_buffer_memory: vk::DeviceMemory,
    uniform_buffer_mapped: *mut std::ffi::c_void,
    
    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    
    // Render pass
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    
    // Command buffers
    command_buffers: Vec<vk::CommandBuffer>,
    
    // Synchronization
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    
    current_frame: usize,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

impl Renderer {
    pub fn new(device_ctx: &DeviceContext, memory_ctx: MemoryContext) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Arc::new(device_ctx.device.clone());
        
        // Initialize scene
        let aspect = device_ctx.swapchain_extent.width as f32 / device_ctx.swapchain_extent.height as f32;
        let scene = Scene::new(aspect);
        
        // Create descriptor set layout for UBO
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        
        // Create render pass
        let render_pass = Self::create_render_pass(&device, device_ctx.surface_format.format)?;
        
        // Create pipeline
        let (pipeline, pipeline_layout) = Self::create_3d_pipeline(&device, render_pass, descriptor_set_layout)?;
        
        // Create framebuffers
        let framebuffers = Self::create_framebuffers(
            &device,
            render_pass,
            &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view,
            device_ctx.swapchain_extent,
        )?;
        
        // Create vertex and index buffers
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(&device, device_ctx, &scene)?;
        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(&device, device_ctx, &scene)?;
        
        // Create uniform buffer
        let (uniform_buffer, uniform_buffer_memory, uniform_buffer_mapped) = 
            Self::create_uniform_buffer(&device, device_ctx)?;
        
        // Create descriptor pool and sets
        let (descriptor_pool, descriptor_sets) = 
            Self::create_descriptor_sets(&device, descriptor_set_layout, uniform_buffer)?;
        
        // Allocate command buffers
        let command_buffers = Self::allocate_command_buffers(&device, device_ctx.command_pool)?;
        
        // Create synchronization objects
        let mut image_available = Vec::new();
        let mut render_finished = Vec::new();
        let mut in_flight_fences = Vec::new();
        
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
                render_finished.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
                in_flight_fences.push(device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?);
            }
        }
        
        Ok(Self {
            device,
            memory_ctx,
            scene,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffer,
            uniform_buffer_memory,
            uniform_buffer_mapped,
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
    
    fn create_vertex_buffer(
        device: &Device, 
        device_ctx: &DeviceContext,
        scene: &Scene
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let mut all_vertices = Vec::new();
        for mesh in &scene.meshes {
            all_vertices.extend_from_slice(&mesh.vertices);
        }
        
        let buffer_size = (std::mem::size_of::<Vertex>() * all_vertices.len()) as u64;
        
        // Create staging buffer
        let (staging_buffer, staging_memory) = Self::create_buffer(
            device,
            device_ctx,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        // Copy data to staging buffer
        unsafe {
            let data = device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(
                all_vertices.as_ptr() as *const u8,
                data as *mut u8,
                buffer_size as usize,
            );
            device.unmap_memory(staging_memory);
        }
        
        // Create device local buffer
        let (buffer, memory) = Self::create_buffer(
            device,
            device_ctx,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        // Copy from staging to device
        Self::copy_buffer(device, device_ctx, staging_buffer, buffer, buffer_size)?;
        
        // Cleanup staging
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }
        
        Ok((buffer, memory))
    }
    
    fn create_index_buffer(
        device: &Device,
        device_ctx: &DeviceContext,
        scene: &Scene
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let mut all_indices = Vec::new();
        for mesh in &scene.meshes {
            all_indices.extend_from_slice(&mesh.indices);
        }
        
        let buffer_size = (std::mem::size_of::<u32>() * all_indices.len()) as u64;
        
        // Create staging buffer
        let (staging_buffer, staging_memory) = Self::create_buffer(
            device,
            device_ctx,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        // Copy data
        unsafe {
            let data = device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(
                all_indices.as_ptr() as *const u8,
                data as *mut u8,
                buffer_size as usize,
            );
            device.unmap_memory(staging_memory);
        }
        
        // Create device local buffer
        let (buffer, memory) = Self::create_buffer(
            device,
            device_ctx,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        Self::copy_buffer(device, device_ctx, staging_buffer, buffer, buffer_size)?;
        
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }
        
        Ok((buffer, memory))
    }
    
    fn create_uniform_buffer(
        device: &Device,
        device_ctx: &DeviceContext,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut std::ffi::c_void), Box<dyn std::error::Error>> {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;
        
        let (buffer, memory) = Self::create_buffer(
            device,
            device_ctx,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        let mapped = unsafe { device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())? };
        
        Ok((buffer, memory, mapped))
    }
    
    fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, Box<dyn std::error::Error>> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&binding));
        
        unsafe { Ok(device.create_descriptor_set_layout(&layout_info, None)?) }
    }
    
    fn create_descriptor_sets(
        device: &Device,
        layout: vk::DescriptorSetLayout,
        uniform_buffer: vk::Buffer,
    ) -> Result<(vk::DescriptorPool, Vec<vk::DescriptorSet>), Box<dyn std::error::Error>> {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
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
        
        // Update descriptor sets
        for set in &sets {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer)
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as u64);
            
            let write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]); }
        }
        
        Ok((pool, sets))
    }
    
    fn create_3d_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn std::error::Error>> {
        unsafe {
            let vert_shader_code = include_bytes!("../shaders/compiled/basic.vert.spv");
            let frag_shader_code = include_bytes!("../shaders/compiled/basic.frag.spv");
            
            let vert_code_aligned = align_shader_code(vert_shader_code);
            let frag_code_aligned = align_shader_code(frag_shader_code);
            
            let vert_shader_module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vert_code_aligned), None)?;
            let frag_shader_module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code_aligned), None)?;
            
            let vert_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(c"main");
            
            let frag_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(c"main");
            
            let stages = [vert_stage, frag_stage];
            
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
            
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false);
            
            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&color_blend_attachment));
            
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);
            
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;
            
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
            
            let pipelines = device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None
            ).map_err(|(_, e)| e)?;
            
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
            
            Ok((pipelines[0], pipeline_layout))
        }
    }
    
    pub fn render(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            // Wait for previous frame
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX
            )?;
            
            // Update scene
            self.scene.update(0.016); // 60 FPS frame time
            
            // Acquire next image
            let (image_index, _) = device_ctx.swapchain_loader.acquire_next_image(
                device_ctx.swapchain,
                u64::MAX,
                self.image_available[self.current_frame],
                vk::Fence::null()
            )?;
            
            // Reset fence
            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]])?;
            
            // Record command buffer
            let command_buffer = self.command_buffers[self.current_frame];
            self.device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device.begin_command_buffer(command_buffer, &begin_info)?;
            
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
            
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: device_ctx.swapchain_extent,
                })
                .clear_values(&clear_values);
            
            self.device.cmd_begin_render_pass(command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            
            // Set viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: device_ctx.swapchain_extent.width as f32,
                height: device_ctx.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: device_ctx.swapchain_extent,
            };
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            
            // Bind vertex and index buffers once
            self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_bind_index_buffer(command_buffer, self.index_buffer, 0, vk::IndexType::UINT32);
            
            // Draw each mesh with its transform
            let mut vertex_offset = 0;
            let mut index_offset = 0;
            
            for mesh in &self.scene.meshes {
                // Update uniform buffer with this mesh's transform
                let ubo = UniformBufferObject {
                    model: mesh.transform,
                    view: self.scene.camera.get_view_matrix(),
                    proj: self.scene.camera.get_projection_matrix(),
                };
                
                std::ptr::copy_nonoverlapping(
                    &ubo as *const UniformBufferObject,
                    self.uniform_buffer_mapped as *mut UniformBufferObject,
                    1
                );
                
                // Bind descriptor set
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[self.current_frame]],
                    &[]
                );
                
                // Draw this mesh
                self.device.cmd_draw_indexed(
                    command_buffer, 
                    mesh.indices.len() as u32, 
                    1, 
                    index_offset, 
                    vertex_offset as i32, 
                    0
                );
                
                vertex_offset += mesh.vertices.len() as u32;
                index_offset += mesh.indices.len() as u32;
            }
            
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
            
            // Submit
            let wait_semaphores = [self.image_available[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished[self.current_frame]];
            let command_buffers = [command_buffer];
            
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            
            self.device.queue_submit(
                device_ctx.queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame]
            )?;
            
            // Present
            let swapchains = [device_ctx.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            
            device_ctx.swapchain_loader.queue_present(device_ctx.queue, &present_info)?;
            
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        
        Ok(())
    }
    
    // Add remaining helper methods...
    
    fn create_buffer(
        device: &Device,
        device_ctx: &DeviceContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        
        let memory_type = Self::find_memory_type(
            device_ctx.memory_properties,
            mem_requirements.memory_type_bits,
            properties
        )?;
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);
        
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
        
        Ok((buffer, memory))
    }
    
    fn find_memory_type(
        properties: vk::PhysicalDeviceMemoryProperties,
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        for i in 0..properties.memory_type_count {
            if (type_filter & (1 << i)) != 0 &&
               properties.memory_types[i as usize].property_flags.contains(required_properties) {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".into())
    }
    
    fn copy_buffer(
        device: &Device,
        device_ctx: &DeviceContext,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(device_ctx.command_pool)
                .command_buffer_count(1);
            
            let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];
            
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            
            device.begin_command_buffer(command_buffer, &begin_info)?;
            
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            
            device.cmd_copy_buffer(command_buffer, src, dst, std::slice::from_ref(&copy_region));
            device.end_command_buffer(command_buffer)?;
            
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&command_buffer));
            
            device.queue_submit(device_ctx.queue, std::slice::from_ref(&submit_info), vk::Fence::null())?;
            device.queue_wait_idle(device_ctx.queue)?;
            
            device.free_command_buffers(device_ctx.command_pool, std::slice::from_ref(&command_buffer));
        }
        
        Ok(())
    }
    
    fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
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
            
            let color_attachment_ref = vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            
            let depth_attachment_ref = vk::AttachmentReference::default()
                .attachment(1)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            
            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::slice::from_ref(&color_attachment_ref))
                .depth_stencil_attachment(&depth_attachment_ref);
            
            let dependency = vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
            
            let render_pass_info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(std::slice::from_ref(&subpass))
                .dependencies(std::slice::from_ref(&dependency));
            
            Ok(device.create_render_pass(&render_pass_info, None)?)
        }
    }
    
    fn create_framebuffers(
        device: &Device,
        render_pass: vk::RenderPass,
        image_views: &[vk::ImageView],
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>, Box<dyn std::error::Error>> {
        let framebuffers = image_views
            .iter()
            .map(|&image_view| {
                let attachments = [image_view, depth_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                
                unsafe { device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(framebuffers)
    }
    
    fn allocate_command_buffers(device: &Device, command_pool: vk::CommandPool) 
        -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
        
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
        
        unsafe { Ok(device.allocate_command_buffers(&alloc_info)?) }
    }
    
    pub fn recreate_framebuffers(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.device_wait_idle()?;
            
            for &framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            
            self.framebuffers = Self::create_framebuffers(
                &self.device,
                self.render_pass,
                &device_ctx.swapchain_image_views,
                device_ctx.depth_image_view,
                device_ctx.swapchain_extent,
            )?;
            
            // Update camera aspect ratio
            self.scene.camera.update_aspect(device_ctx.swapchain_extent.width, device_ctx.swapchain_extent.height);
        }
        Ok(())
    }
}

fn align_shader_code(code: &[u8]) -> Vec<u32> {
    let mut aligned = Vec::with_capacity(code.len() / 4);
    for chunk in code.chunks_exact(4) {
        aligned.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    aligned
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.image_available {
                self.device.destroy_semaphore(semaphore, None);
            }
            
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            
            for &framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            
            self.device.unmap_memory(self.uniform_buffer_memory);
            self.device.destroy_buffer(self.uniform_buffer, None);
            self.device.free_memory(self.uniform_buffer_memory, None);
            
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
        }
    }
}