//! Bindless Texture Manager (Phase 1)
//!
//! Manages a fixed-size descriptor array of `sampler2D` textures using
//! `VK_EXT_descriptor_indexing` with `PARTIALLY_BOUND_BIT`.  Materials
//! reference textures by slot index.
//!
//! Slot 0 is always a 1×1 white fallback texture used when a material's
//! textures have not loaded yet.

use ash::{vk, Device};
use std::collections::HashMap;
use crate::memory::{ImageAllocation, ImageHandle, MemoryContext, TransferTicket};

/// Maximum number of texture slots in the bindless array.
pub const MAX_TEXTURES: u32 = 4096;

/// Fallback texture slot (1×1 white, always valid).
pub const FALLBACK_TEXTURE_SLOT: u32 = 0;

// ====================================================================
//  Texture Slot Tracking
// ====================================================================

#[derive(Debug)]
struct TextureSlot {
    image_handle: ImageHandle,
    image_view: vk::ImageView,
    name: String,
    width: u32,
    height: u32,
}

/// In-flight texture upload being tracked by the manager.
#[derive(Debug)]
pub struct PendingTexture {
    pub slot: u32,
    pub name: String,
    pub image_handle: ImageHandle,
    pub image_view: vk::ImageView,
    pub ticket: TransferTicket,
}

// ====================================================================
//  TextureManager
// ====================================================================

/// Owns the bindless texture descriptor set, manages texture slot
/// allocation, and tracks in-flight async texture uploads.
pub struct TextureManager {
    device: Device,

    /// Descriptor set layout for bindless textures (set 1).
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool for the bindless set.
    descriptor_pool: vk::DescriptorPool,
    /// The single descriptor set (updated via vkUpdateDescriptorSets).
    pub descriptor_set: vk::DescriptorSet,

    /// Default sampler: linear filtering, repeat wrap, aniso 16x.
    pub default_sampler: vk::Sampler,

    /// Occupied slots.
    slots: HashMap<u32, TextureSlot>,
    /// Free slot indices (excluding slot 0 which is the fallback).
    free_slots: Vec<u32>,
    /// Name → slot lookup.
    name_to_slot: HashMap<String, u32>,

    /// In-flight async uploads.
    pending: Vec<PendingTexture>,

    /// Fallback texture resources.
    fallback_image_handle: ImageHandle,
}

impl TextureManager {
    pub fn new(
        device: &Device,
        memory_ctx: &mut MemoryContext,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = device.clone();

        // ---- Create sampler ----
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(true)
                    .max_anisotropy(16.0)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?
        };

        // ---- Descriptor set layout with partially-bound flag ----
        //
        // One binding: an unbounded array of combined image samplers.
        // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT allows slots to be
        // unwritten (empty) without validation errors.

        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);

        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_TEXTURES)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&binding))
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags_info);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&layout_info, None)?
        };

        // ---- Descriptor pool ----

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_TEXTURES);

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&pool_info, None)?
        };

        // ---- Allocate descriptor set ----

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(&alloc_info)?[0]
        };

        // ---- Create 1×1 white fallback texture ----

        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        let fallback_alloc = memory_ctx.create_image_with_data(
            &white_pixel,
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            command_pool,
            queue,
        )?;

        let fallback_image_handle = fallback_alloc.handle;

        // Write fallback to slot 0.
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(fallback_alloc.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

        // ---- Free slot pool (1 .. MAX_TEXTURES-1) ----

        let mut free_slots: Vec<u32> = (1..MAX_TEXTURES).rev().collect();

        // Slot 0 is occupied by fallback.
        let mut slots = HashMap::new();
        slots.insert(0, TextureSlot {
            image_handle: fallback_image_handle,
            image_view: fallback_alloc.view,
            name: "fallback_white".into(),
            width: 1,
            height: 1,
        });

        let mut name_to_slot = HashMap::new();
        name_to_slot.insert("fallback_white".into(), 0u32);

        println!(
            "[TextureManager] Initialized.  {} slots available, fallback at slot 0",
            free_slots.len(),
        );

        Ok(Self {
            device,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            default_sampler: sampler,
            slots,
            free_slots,
            name_to_slot,
            pending: Vec::new(),
            fallback_image_handle,
        })
    }

    /// Allocate a texture slot and begin an async upload.
    ///
    /// Returns the slot index (usable immediately in material definitions
    /// — the shader will sample the fallback until the upload completes
    /// and the descriptor is written).
    pub fn load_async(
        &mut self,
        name: impl Into<String>,
        data: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        memory_ctx: &mut MemoryContext,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let name = name.into();

        // Already loaded?
        if let Some(&slot) = self.name_to_slot.get(&name) {
            return Ok(slot);
        }

        let slot = self
            .free_slots
            .pop()
            .ok_or("TextureManager: no free slots")?;

        let (alloc, ticket) =
            memory_ctx.upload_image_async(data, width, height, format)?;

        self.pending.push(PendingTexture {
            slot,
            name: name.clone(),
            image_handle: alloc.handle,
            image_view: alloc.view,
            ticket,
        });

        self.name_to_slot.insert(name, slot);

        Ok(slot)
    }

    /// Poll pending uploads.  For each completed upload, write the
    /// image view into the descriptor set at the assigned slot.
    pub fn poll_pending(&mut self, memory_ctx: &MemoryContext) {
        let mut completed = Vec::new();
        let mut still_pending = Vec::new();

        for p in self.pending.drain(..) {
            if memory_ctx.transfer.is_complete(&p.ticket) {
                completed.push(p);
            } else {
                still_pending.push(p);
            }
        }

        for p in completed {
            // Write descriptor.
            let image_info = vk::DescriptorImageInfo::default()
                .sampler(self.default_sampler)
                .image_view(p.image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .dst_array_element(p.slot)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            unsafe {
                self.device.update_descriptor_sets(
                    std::slice::from_ref(&write),
                    &[],
                );
            }

            // Track in slots map.
            self.slots.insert(p.slot, TextureSlot {
                image_handle: p.image_handle,
                image_view: p.image_view,
                name: p.name.clone(),
                width: 0, // TODO: store dimensions from alloc
                height: 0,
            });

            println!(
                "[TextureManager] Texture '{}' ready at slot {}",
                p.name, p.slot,
            );
        }

        self.pending = still_pending;
    }

    /// Synchronous texture load (blocks until upload completes).
    /// Returns the slot index.
    pub fn load_sync(
        &mut self,
        name: impl Into<String>,
        data: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        memory_ctx: &mut MemoryContext,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let name = name.into();

        if let Some(&slot) = self.name_to_slot.get(&name) {
            return Ok(slot);
        }

        let slot = self
            .free_slots
            .pop()
            .ok_or("TextureManager: no free slots")?;

        let alloc = memory_ctx.create_image_with_data(
            data,
            width,
            height,
            format,
            command_pool,
            queue,
        )?;

        // Write descriptor.
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(self.default_sampler)
            .image_view(alloc.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            self.device.update_descriptor_sets(
                std::slice::from_ref(&write),
                &[],
            );
        }

        self.slots.insert(slot, TextureSlot {
            image_handle: alloc.handle,
            image_view: alloc.view,
            name: name.clone(),
            width,
            height,
        });
        self.name_to_slot.insert(name, slot);

        Ok(slot)
    }

    /// Release a texture slot and free its GPU resources.
    pub fn unload(&mut self, slot: u32, memory_ctx: &mut MemoryContext) {
        if slot == FALLBACK_TEXTURE_SLOT {
            return; // Never unload fallback.
        }

        if let Some(tex) = self.slots.remove(&slot) {
            self.name_to_slot.remove(&tex.name);
            memory_ctx.allocator.free_image(tex.image_handle);
            self.free_slots.push(slot);

            println!(
                "[TextureManager] Unloaded '{}' from slot {}",
                tex.name, slot,
            );
        }
    }

    /// Look up a texture's slot by name.
    pub fn find_slot(&self, name: &str) -> Option<u32> {
        self.name_to_slot.get(name).copied()
    }

    /// Number of occupied slots.
    pub fn active_count(&self) -> usize {
        self.slots.len()
    }

    /// Number of pending uploads.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        unsafe {
            // Note: image views and images are cleaned up by GpuAllocator::drop
            // via free_image.  We only need to destroy Vulkan objects we own.
            self.device.destroy_sampler(self.default_sampler, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}