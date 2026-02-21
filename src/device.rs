use ash::{vk, vk::Handle, ext::debug_utils, khr::{surface, swapchain}, Device, Entry, Instance};
use std::ffi::{CStr, CString};
use sdl2::video::Window;

const APP_NAME: &CStr = c"VKE001";
pub const WINDOW_NAME: &str = "VKE001";

pub struct DeviceContext {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: surface::Instance,
    pub surface_format: vk::SurfaceFormatKHR,
    pub swapchain_loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_extent: vk::Extent2D,
    pub command_pool: vk::CommandPool,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub debug_utils: Option<debug_utils::Instance>,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT, 
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>, 
    _: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    println!("[{:?}] [{:?}]: {}", severity, ty, CStr::from_ptr((*data).p_message).to_string_lossy());
    vk::FALSE
}

impl DeviceContext {
    pub fn new(
        window: &Window,
        enable_validation: bool
    ) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let entry = Entry::linked();
            let app_info = vk::ApplicationInfo::default()
                .application_name(APP_NAME)
                .engine_name(APP_NAME)
                .api_version(vk::API_VERSION_1_3);
            
            // Get required extensions from SDL2
            let sdl_extensions = window.vulkan_instance_extensions()
                .map_err(|e| format!("Failed to get SDL2 Vulkan extensions: {}", e))?;
            
            // Convert SDL2 extension names to CStrings and then to pointers
            let extension_names: Vec<CString> = sdl_extensions
                .iter()
                .map(|name| CString::new(*name).unwrap())
                .collect();
            
            let mut extension_ptrs: Vec<*const i8> = extension_names
                .iter()
                .map(|name| name.as_ptr())
                .collect();;
            
            // Add debug extension if validation is enabled
            let debug_ext_name = CString::new(debug_utils::NAME.to_bytes()).unwrap();
            if enable_validation { 
                extension_ptrs.push(debug_ext_name.as_ptr());
            }
            
            // Validation layers
            let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
            let layer_names = if enable_validation { 
                vec![validation_layer.as_ptr()]
            } else { 
                vec![] 
            };
            
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extension_ptrs)
                .enabled_layer_names(&layer_names), None)?;
            
            let (debug_utils, debug_messenger) = if enable_validation {
                let loader = debug_utils::Instance::new(&entry, &instance);
                let messenger = loader.create_debug_utils_messenger(&vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                    .pfn_user_callback(Some(debug_callback)), None)?;
                (Some(loader), Some(messenger))
            } else { (None, None) };
            
            // Create surface using SDL2's built-in method
            let surface_raw = window.vulkan_create_surface(instance.handle().as_raw() as usize)
                .map_err(|e| format!("Failed to create Vulkan surface: {}", e))?;
            let surface = vk::SurfaceKHR::from_raw(surface_raw as u64);
            let surface_loader = surface::Instance::new(&entry, &instance);
            
            let (physical_device, queue_family_index) = instance.enumerate_physical_devices()?
                .iter().find_map(|&pd| {
                    instance.get_physical_device_queue_family_properties(pd).iter().enumerate()
                        .find_map(|(i, info)| {
                            let idx = i as u32;
                            (info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER) &&
                             surface_loader.get_physical_device_surface_support(pd, idx, surface).unwrap_or(false))
                            .then_some((pd, idx))
                        })
                }).ok_or("No suitable device")?;
            
            let device = instance.create_device(physical_device, &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index).queue_priorities(&[1.0])])
                .enabled_extension_names(&[swapchain::NAME.as_ptr()])
                .enabled_features(&vk::PhysicalDeviceFeatures::default()), None)?;
            
            let queue = device.get_device_queue(queue_family_index, 0);
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            
            let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
            let surface_format = formats.first().copied().ok_or("No formats")?;
            
            let caps = surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;
            let swapchain_loader = swapchain::Device::new(&instance, &device);
            let swapchain = swapchain_loader.create_swapchain(&vk::SwapchainCreateInfoKHR::default()
                .surface(surface).min_image_count(caps.min_image_count + 1)
                .image_color_space(surface_format.color_space).image_format(surface_format.format)
                .image_extent(caps.current_extent).image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE).pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE).present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true).image_array_layers(1), None)?;
            
            let images = swapchain_loader.get_swapchain_images(swapchain)?;
            let views = images.iter().map(|&img| device.create_image_view(&vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D).format(surface_format.format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0,
                    level_count: 1, base_array_layer: 0, layer_count: 1,
                }).image(img), None).unwrap()).collect();
            
            let command_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index), None)?;
            
            // Create depth buffer
            let depth_image = device.create_image(&vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(vk::Extent3D { width: caps.current_extent.width, height: caps.current_extent.height, depth: 1 })
                .mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
            
            let mem_requirements = device.get_image_memory_requirements(depth_image);
            let memory_type_index = (0..memory_properties.memory_type_count)
                .find(|&i| {
                    (mem_requirements.memory_type_bits & (1 << i)) != 0 &&
                    memory_properties.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                }).ok_or("No suitable memory")?;
            
            let depth_image_memory = device.allocate_memory(&vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index), None)?;
            
            device.bind_image_memory(depth_image, depth_image_memory, 0)?;
            
            let depth_image_view = device.create_image_view(&vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH, base_mip_level: 0,
                    level_count: 1, base_array_layer: 0, layer_count: 1,
                }).image(depth_image), None)?;
            
            Ok(Self {
                entry, instance, device, physical_device, queue, queue_family_index, surface, surface_loader,
                surface_format, swapchain_loader, swapchain, swapchain_images: images,
                swapchain_image_views: views, swapchain_extent: caps.current_extent,
                command_pool, depth_image, depth_image_view, depth_image_memory, 
                memory_properties, debug_utils, debug_messenger,
            })
        }
    }
    
    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.swapchain_image_views.iter().for_each(|&v| self.device.destroy_image_view(v, None));
            
            let caps = self.surface_loader.get_physical_device_surface_capabilities(self.physical_device, self.surface).unwrap();
            let extent = vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            };
            
            let old = self.swapchain;
            self.swapchain = self.swapchain_loader.create_swapchain(&vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface).min_image_count(caps.min_image_count + 1)
                .image_color_space(self.surface_format.color_space).image_format(self.surface_format.format)
                .image_extent(extent).image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE).pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE).present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true).image_array_layers(1).old_swapchain(old), None).unwrap();
            
            self.swapchain_loader.destroy_swapchain(old, None);
            
            self.swapchain_images = self.swapchain_loader.get_swapchain_images(self.swapchain).unwrap();
            self.swapchain_image_views = self.swapchain_images.iter().map(|&img| 
                self.device.create_image_view(&vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D).format(self.surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0,
                        level_count: 1, base_array_layer: 0, layer_count: 1,
                    }).image(img), None).unwrap()).collect();
            self.swapchain_extent = extent;
        }
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_image_view(self.depth_image_view, None);
            self.swapchain_image_views.iter().for_each(|&v| self.device.destroy_image_view(v, None));
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let (Some(debug), Some(msg)) = (&self.debug_utils, self.debug_messenger) {
                debug.destroy_debug_utils_messenger(msg, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}