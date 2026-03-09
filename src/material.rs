//! PBR Material System (Phase 1)
//!
//! Provides a `MaterialData` GPU struct matching the PBR fragment shader
//! layout, a `MaterialLibrary` that maps string names to material IDs,
//! and SSBO management for uploading the material array to the GPU.
//!
//! Materials reference textures by index into the bindless texture array
//! (descriptor set 1).  A material_id of 0 is always the default white
//! PBR material.

use ash::vk;
use std::collections::HashMap;
use std::ptr::NonNull;
use crate::memory::{BufferHandle, MemoryContext, MemoryLocation};

/// Maximum number of materials supported.  The SSBO is pre-allocated
/// to this capacity × sizeof(MaterialData).
pub const MAX_MATERIALS: usize = 1024;

// ====================================================================
//  GPU-side material struct (128 bytes, std430 layout)
// ====================================================================

/// Matches the GLSL MaterialData struct in the PBR fragment shader.
///
/// ```glsl
/// struct MaterialData {
///     vec4  base_color;           // rgba
///     vec4  emissive;             // rgb + intensity
///     float metallic;
///     float roughness;
///     float ao;
///     float normal_scale;
///     uint  albedo_tex;           // bindless index (0 = no texture)
///     uint  normal_tex;
///     uint  metallic_roughness_tex;
///     uint  emissive_tex;
///     uint  ao_tex;
///     uint  flags;                // bitfield: alpha_cutoff, double_sided, etc.
///     float alpha_cutoff;
///     float _pad;
///     vec4  _reserved0;
///     vec4  _reserved1;
/// };  // total: 128 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MaterialData {
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub normal_scale: f32,
    pub albedo_tex: u32,
    pub normal_tex: u32,
    pub metallic_roughness_tex: u32,
    pub emissive_tex: u32,
    pub ao_tex: u32,
    pub flags: u32,
    pub alpha_cutoff: f32,
    pub _pad: f32,
    pub _reserved0: [f32; 4],
    pub _reserved1: [f32; 4],
    pub _reserved2: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<MaterialData>() == 128);

/// Material flags bitfield.
pub mod material_flags {
    pub const DOUBLE_SIDED: u32 = 1 << 0;
    pub const ALPHA_BLEND: u32 = 1 << 1;
    pub const ALPHA_CUTOFF: u32 = 1 << 2;
    pub const UNLIT: u32 = 1 << 3;
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0, 0.0, 0.0, 0.0],
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            normal_scale: 1.0,
            albedo_tex: 0,     // slot 0 = fallback white 1x1
            normal_tex: 0,
            metallic_roughness_tex: 0,
            emissive_tex: 0,
            ao_tex: 0,
            flags: 0,
            alpha_cutoff: 0.5,
            _pad: 0.0,
            _reserved0: [0.0; 4],
            _reserved1: [0.0; 4],
            _reserved2: [0.0; 4],
        }
    }
}

impl MaterialData {
    /// Create a simple colored material with no textures.
    pub fn colored(r: f32, g: f32, b: f32) -> Self {
        Self {
            base_color: [r, g, b, 1.0],
            ..Default::default()
        }
    }

    /// Create a metallic material.
    pub fn metallic(r: f32, g: f32, b: f32, metallic: f32, roughness: f32) -> Self {
        Self {
            base_color: [r, g, b, 1.0],
            metallic,
            roughness,
            ..Default::default()
        }
    }

    /// Is this material transparent (needs alpha blend pass)?
    pub fn is_transparent(&self) -> bool {
        (self.flags & material_flags::ALPHA_BLEND) != 0
            || self.base_color[3] < 1.0
    }
}

// ====================================================================
//  MaterialLibrary — CPU-side material management
// ====================================================================

/// Owns all material definitions.  Maps string names to material IDs
/// (indices into the GPU SSBO).  Material ID 0 is always the default.
pub struct MaterialLibrary {
    materials: Vec<MaterialData>,
    name_to_id: HashMap<String, u32>,
    dirty: bool,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        let mut lib = Self {
            materials: Vec::with_capacity(MAX_MATERIALS),
            name_to_id: HashMap::new(),
            dirty: true,
        };

        // ID 0: default white PBR material.
        lib.materials.push(MaterialData::default());
        lib.name_to_id.insert("default".into(), 0);

        lib
    }

    /// Register a new material.  Returns its ID.
    /// If a material with the same name exists, it is overwritten.
    pub fn add(&mut self, name: impl Into<String>, data: MaterialData) -> u32 {
        let name = name.into();

        if let Some(&existing_id) = self.name_to_id.get(&name) {
            self.materials[existing_id as usize] = data;
            self.dirty = true;
            return existing_id;
        }

        assert!(
            self.materials.len() < MAX_MATERIALS,
            "Material limit ({}) exceeded",
            MAX_MATERIALS,
        );

        let id = self.materials.len() as u32;
        self.materials.push(data);
        self.name_to_id.insert(name, id);
        self.dirty = true;
        id
    }

    /// Get a material by ID.
    pub fn get(&self, id: u32) -> Option<&MaterialData> {
        self.materials.get(id as usize)
    }

    /// Get a mutable material by ID.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut MaterialData> {
        self.dirty = true;
        self.materials.get_mut(id as usize)
    }

    /// Look up material ID by name.
    pub fn find(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    /// Number of materials registered.
    pub fn count(&self) -> usize {
        self.materials.len()
    }

    /// Has the library been modified since the last GPU upload?
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark clean after a successful GPU upload.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Get the raw byte slice for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.materials.as_ptr() as *const u8,
                self.materials.len() * std::mem::size_of::<MaterialData>(),
            )
        }
    }
}

// ====================================================================
//  MaterialSsbo — GPU buffer management
// ====================================================================

/// Manages the GPU-side material SSBO.
///
/// Allocated once at startup with capacity for `MAX_MATERIALS` entries.
/// Re-uploaded (sub-region) whenever the `MaterialLibrary` is dirty.
pub struct MaterialSsbo {
    pub buffer_handle: BufferHandle,
    pub buffer: vk::Buffer,
    /// Persistently mapped pointer (HOST_VISIBLE | HOST_COHERENT).
    mapped_ptr: NonNull<u8>,
    capacity_bytes: u64,
}

unsafe impl Send for MaterialSsbo {}

impl MaterialSsbo {
    /// Allocate the SSBO via the pool allocator as CPU-to-GPU memory.
    pub fn new(
        memory_ctx: &mut MemoryContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let capacity_bytes =
            (MAX_MATERIALS * std::mem::size_of::<MaterialData>()) as u64;

        let alloc = memory_ctx.allocator.create_buffer(
            capacity_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;

        let mapped_ptr = alloc
            .mapped_ptr
            .ok_or("MaterialSsbo: CpuToGpu buffer was not mapped")?;

        println!(
            "[MaterialSsbo] Allocated {} for {} materials",
            format_bytes(capacity_bytes),
            MAX_MATERIALS,
        );

        Ok(Self {
            buffer_handle: alloc.handle,
            buffer: alloc.buffer,
            mapped_ptr,
            capacity_bytes,
        })
    }

    /// Upload the full material library to the GPU.
    ///
    /// Since the buffer is HOST_COHERENT, a simple memcpy is sufficient
    /// — no staging buffer, no command buffer, no flush.
    pub fn upload(&self, library: &MaterialLibrary) {
        let data = library.as_bytes();
        assert!(data.len() as u64 <= self.capacity_bytes);

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.mapped_ptr.as_ptr(),
                data.len(),
            );
        }
    }

    /// Destroy the SSBO.
    pub fn destroy(&self, memory_ctx: &mut MemoryContext) {
        memory_ctx.allocator.free_buffer(self.buffer_handle);
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}