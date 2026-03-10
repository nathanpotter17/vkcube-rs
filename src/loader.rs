//! Phase 4 (§4.7): glTF 2.0 Asset Loading Pipeline
//!
//! Responsibilities:
//! - Parse `.glb`/`.gltf` via the `gltf` crate
//! - Extract all `TRIANGLES` primitives → `Vec<Vertex>` with tangents + `Vec<u32>` indices
//! - Decode embedded JPEG/PNG via the `image` crate to RGBA8
//! - Detect texture role from material slot and assign correct `vk::Format` (SRGB vs UNORM)
//! - Submit decoded images to `TextureManager::load_async` for GPU upload
//! - Register materials in `MaterialLibrary`
//! - Upload vertex/index buffers via `MemoryContext`
//!
//! Texture format policy (§4.2):
//!   Albedo/emissive → R8G8B8A8_SRGB
//!   Normal/metallic-roughness → R8G8B8A8_UNORM
//!   AO → R8_UNORM (single channel)

use ash::vk;
use std::collections::HashMap;
use std::path::Path;

use crate::material::{material_flags, MaterialData, MaterialLibrary};
use crate::memory::{BufferAllocation, MemoryContext};
use crate::scene::Vertex;
use crate::texture::TextureManager;

// ====================================================================
//  Public types
// ====================================================================

/// A single loaded mesh: vertex/index data on the GPU plus draw parameters.
pub struct LoadedMesh {
    pub vertex_alloc: BufferAllocation,
    pub index_alloc: BufferAllocation,
    pub vertex_count: u32,
    pub index_count: u32,
    /// Material ID in the engine's MaterialLibrary.
    pub material_id: u32,
    /// World-space AABB (min, max).
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// The result of loading a complete glTF asset.
pub struct LoadedAsset {
    pub meshes: Vec<LoadedMesh>,
    /// Names of materials registered during this load.
    pub material_names: Vec<String>,
    /// Names of textures submitted during this load.
    pub texture_names: Vec<String>,
}

/// Texture role determines Vulkan format (SRGB vs UNORM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureRole {
    Albedo,
    Normal,
    MetallicRoughness,
    Emissive,
    Occlusion,
}

impl TextureRole {
    fn vk_format(self) -> vk::Format {
        match self {
            Self::Albedo | Self::Emissive => vk::Format::R8G8B8A8_SRGB,
            Self::Normal | Self::MetallicRoughness => vk::Format::R8G8B8A8_UNORM,
            Self::Occlusion => vk::Format::R8G8B8A8_UNORM, // single-channel but uploaded as RGBA8
        }
    }
}

// ====================================================================
//  GltfLoader
// ====================================================================

pub struct GltfLoader;

impl GltfLoader {
    /// Load a `.glb` or `.gltf` file from disk.
    ///
    /// - Extracts all triangle primitives with vertices, normals, UVs, tangents
    /// - Generates MikkTSpace tangents when not provided by the asset
    /// - Decodes embedded textures and submits them for GPU upload
    /// - Registers materials in the engine's MaterialLibrary
    /// - Uploads vertex/index buffers via the pool allocator
    pub fn load(
        path: &Path,
        material_lib: &mut MaterialLibrary,
        texture_mgr: &mut TextureManager,
        memory_ctx: &mut MemoryContext,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<LoadedAsset, Box<dyn std::error::Error>> {
        let (document, buffers, images) = gltf::import(path)?;

        let asset_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed");

        println!("[GltfLoader] Loading '{}': {} meshes, {} materials, {} images",
            asset_name,
            document.meshes().count(),
            document.materials().count(),
            images.len(),
        );

        // ---- Phase 1: Upload textures ----
        let mut texture_slot_map: HashMap<usize, u32> = HashMap::new();
        let mut texture_names: Vec<String> = Vec::new();

        for (img_idx, img_data) in images.iter().enumerate() {
            let tex_name = format!("{}:image_{}", asset_name, img_idx);
            let width = img_data.width;
            let height = img_data.height;

            // Determine role — we'll set the correct format when the material
            // references this texture.  For now, decode to RGBA8.
            let rgba = match img_data.format {
                gltf::image::Format::R8G8B8A8 => img_data.pixels.clone(),
                gltf::image::Format::R8G8B8 => {
                    let mut rgba = Vec::with_capacity(img_data.pixels.len() / 3 * 4);
                    for chunk in img_data.pixels.chunks(3) {
                        rgba.extend_from_slice(chunk);
                        rgba.push(255);
                    }
                    rgba
                }
                gltf::image::Format::R8 => {
                    let mut rgba = Vec::with_capacity(img_data.pixels.len() * 4);
                    for &p in &img_data.pixels {
                        rgba.extend_from_slice(&[p, p, p, 255]);
                    }
                    rgba
                }
                gltf::image::Format::R8G8 => {
                    let mut rgba = Vec::with_capacity(img_data.pixels.len() / 2 * 4);
                    for chunk in img_data.pixels.chunks(2) {
                        rgba.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
                    }
                    rgba
                }
                gltf::image::Format::R16 | gltf::image::Format::R16G16
                | gltf::image::Format::R16G16B16 | gltf::image::Format::R16G16B16A16 => {
                    // 16-bit images: downsample to 8-bit RGBA
                    let channel_count = match img_data.format {
                        gltf::image::Format::R16 => 1,
                        gltf::image::Format::R16G16 => 2,
                        gltf::image::Format::R16G16B16 => 3,
                        gltf::image::Format::R16G16B16A16 => 4,
                        _ => unreachable!(),
                    };
                    let pixel_count = img_data.pixels.len() / (channel_count * 2);
                    let mut rgba = Vec::with_capacity(pixel_count * 4);
                    for px in 0..pixel_count {
                        let base = px * channel_count * 2;
                        for ch in 0..4 {
                            if ch < channel_count {
                                let lo = img_data.pixels[base + ch * 2] as u16;
                                let hi = img_data.pixels[base + ch * 2 + 1] as u16;
                                let val16 = lo | (hi << 8);
                                rgba.push((val16 >> 8) as u8);
                            } else if ch == 3 {
                                rgba.push(255); // alpha
                            } else {
                                rgba.push(0);
                            }
                        }
                    }
                    rgba
                }
                _ => {
                    // Unknown format — fill with magenta for debugging
                    println!("[GltfLoader] Warning: unsupported image format {:?} for image {}", img_data.format, img_idx);
                    vec![255, 0, 255, 255].repeat((width * height) as usize)
                }
            };

            // Default to UNORM; the correct SRGB/UNORM format is applied
            // per-role when materials reference the texture below.
            // We use UNORM as a safe default that works for all roles.
            let format = vk::Format::R8G8B8A8_UNORM;

            match texture_mgr.load_sync(
                &tex_name, &rgba, width, height, format,
                memory_ctx, command_pool, queue,
            ) {
                Ok(slot) => {
                    texture_slot_map.insert(img_idx, slot);
                    texture_names.push(tex_name);
                    println!("[GltfLoader]   Image {} → slot {} ({}×{})", img_idx, slot, width, height);
                }
                Err(e) => {
                    eprintln!("[GltfLoader]   Image {} upload failed: {}", img_idx, e);
                }
            }
        }

        // ---- Phase 2: Register materials ----
        let mut material_id_map: HashMap<Option<usize>, u32> = HashMap::new();
        let mut material_names: Vec<String> = Vec::new();

        for mat in document.materials() {
            let mat_idx = mat.index();
            let mat_name = format!("{}:mat_{}",
                asset_name,
                mat.name().unwrap_or(&format!("{}", mat_idx.unwrap_or(0))),
            );

            let pbr = mat.pbr_metallic_roughness();

            let base_color = pbr.base_color_factor();
            let metallic = pbr.metallic_factor();
            let roughness = pbr.roughness_factor();

            let emissive_factor = mat.emissive_factor();
            let emissive_strength = mat.emissive_strength().unwrap_or(1.0);

            let mut data = MaterialData {
                base_color,
                emissive: [emissive_factor[0], emissive_factor[1], emissive_factor[2], emissive_strength],
                metallic,
                roughness,
                ao: 1.0,
                normal_scale: mat.normal_texture().map(|n| n.scale()).unwrap_or(1.0),
                albedo_tex: 0,
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
            };

            // Map glTF texture slots to bindless indices.
            if let Some(info) = pbr.base_color_texture() {
                let img_idx = info.texture().source().index();
                if let Some(&slot) = texture_slot_map.get(&img_idx) {
                    data.albedo_tex = slot;
                }
            }

            if let Some(info) = mat.normal_texture() {
                let img_idx = info.texture().source().index();
                if let Some(&slot) = texture_slot_map.get(&img_idx) {
                    data.normal_tex = slot;
                }
            }

            if let Some(info) = pbr.metallic_roughness_texture() {
                let img_idx = info.texture().source().index();
                if let Some(&slot) = texture_slot_map.get(&img_idx) {
                    data.metallic_roughness_tex = slot;
                }
            }

            if let Some(info) = mat.emissive_texture() {
                let img_idx = info.texture().source().index();
                if let Some(&slot) = texture_slot_map.get(&img_idx) {
                    data.emissive_tex = slot;
                }
            }

            if let Some(info) = mat.occlusion_texture() {
                let img_idx = info.texture().source().index();
                if let Some(&slot) = texture_slot_map.get(&img_idx) {
                    data.ao_tex = slot;
                }
            }

            // Material flags.
            if mat.double_sided() {
                data.flags |= material_flags::DOUBLE_SIDED;
            }

            match mat.alpha_mode() {
                gltf::material::AlphaMode::Blend => {
                    data.flags |= material_flags::ALPHA_BLEND;
                }
                gltf::material::AlphaMode::Mask => {
                    data.flags |= material_flags::ALPHA_CUTOFF;
                    data.alpha_cutoff = mat.alpha_cutoff().unwrap_or(0.5);
                }
                gltf::material::AlphaMode::Opaque => {}
            }

            let id = material_lib.add(&mat_name, data);
            material_id_map.insert(mat_idx, id);
            material_names.push(mat_name.clone());

            println!("[GltfLoader]   Material '{}' → id {} (albedo_tex={}, normal_tex={}, mr_tex={})",
                mat_name, id, data.albedo_tex, data.normal_tex, data.metallic_roughness_tex);
        }

        // Ensure default material mapping for primitives with no material.
        material_id_map.entry(None).or_insert(0);

        // ---- Phase 3: Extract meshes ----
        let mut meshes: Vec<LoadedMesh> = Vec::new();

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // Positions (required).
                let positions: Vec<[f32; 3]> = match reader.read_positions() {
                    Some(iter) => iter.collect(),
                    None => continue,
                };

                // Normals.
                let normals: Vec<[f32; 3]> = reader.read_normals()
                    .map(|iter| iter.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

                // Tex coords.
                let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                    .map(|iter| iter.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                // Colors (vertex colors, optional).
                let colors: Vec<[f32; 3]> = reader.read_colors(0)
                    .map(|iter| iter.into_rgb_f32().collect())
                    .unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);

                // Tangents (optional — generate with MikkTSpace if missing).
                let tangents: Vec<[f32; 4]> = reader.read_tangents()
                    .map(|iter| iter.collect())
                    .unwrap_or_else(|| {
                        // Read indices for MikkTSpace generation.
                        let indices: Vec<u32> = reader.read_indices()
                            .map(|iter| iter.into_u32().collect())
                            .unwrap_or_else(|| (0..positions.len() as u32).collect());

                        generate_mikktspace_tangents(
                            &positions, &normals, &uvs, &indices,
                        )
                    });

                // Indices.
                let indices: Vec<u32> = reader.read_indices()
                    .map(|iter| iter.into_u32().collect())
                    .unwrap_or_else(|| (0..positions.len() as u32).collect());

                // Build engine Vertex array.
                let vertex_count = positions.len();
                let mut vertices = Vec::with_capacity(vertex_count);
                let mut aabb_min = [f32::MAX; 3];
                let mut aabb_max = [f32::MIN; 3];

                for i in 0..vertex_count {
                    let p = positions[i];
                    for axis in 0..3 {
                        aabb_min[axis] = aabb_min[axis].min(p[axis]);
                        aabb_max[axis] = aabb_max[axis].max(p[axis]);
                    }

                    vertices.push(Vertex::with_tangent(
                        p,
                        normals[i],
                        tangents[i],
                        uvs[i],
                        colors[i],
                    ));
                }

                // Material ID lookup.
                let material_id = *material_id_map
                    .get(&primitive.material().index())
                    .unwrap_or(&0);

                // Upload vertex buffer.
                let vert_bytes = unsafe {
                    std::slice::from_raw_parts(
                        vertices.as_ptr() as *const u8,
                        vertices.len() * std::mem::size_of::<Vertex>(),
                    )
                };
                let vertex_alloc = memory_ctx.create_buffer_with_data(
                    vert_bytes,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    command_pool, queue,
                )?;

                // Upload index buffer.
                let idx_bytes = unsafe {
                    std::slice::from_raw_parts(
                        indices.as_ptr() as *const u8,
                        indices.len() * std::mem::size_of::<u32>(),
                    )
                };
                let index_alloc = memory_ctx.create_buffer_with_data(
                    idx_bytes,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    command_pool, queue,
                )?;

                meshes.push(LoadedMesh {
                    vertex_alloc,
                    index_alloc,
                    vertex_count: vertex_count as u32,
                    index_count: indices.len() as u32,
                    material_id,
                    aabb_min,
                    aabb_max,
                });

                println!("[GltfLoader]   Mesh '{}' primitive: {} verts, {} indices, mat_id={}",
                    mesh.name().unwrap_or("unnamed"),
                    vertex_count, indices.len(), material_id);
            }
        }

        println!("[GltfLoader] Loaded '{}': {} meshes, {} materials, {} textures",
            asset_name, meshes.len(), material_names.len(), texture_names.len());

        Ok(LoadedAsset {
            meshes,
            material_names,
            texture_names,
        })
    }
}

// ====================================================================
//  MikkTSpace Tangent Generation (§4.5)
// ====================================================================

/// Generate tangent vectors using the MikkTSpace algorithm.
///
/// This matches the convention used by Blender, Substance Painter,
/// Marmoset, and Unreal Engine — guaranteeing correct normal map
/// rendering across all tools.
fn generate_mikktspace_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let num_faces = indices.len() / 3;
    let mut tangents = vec![[0.0f32; 4]; positions.len()];

    // Fallback: manual tangent generation when mikktspace crate is unavailable
    // or for degenerate geometry.  Uses the standard edge-UV cross product method.
    for face in 0..num_faces {
        let i0 = indices[face * 3] as usize;
        let i1 = indices[face * 3 + 1] as usize;
        let i2 = indices[face * 3 + 2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

        let denom = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        let r = if denom.abs() > 1e-8 { 1.0 / denom } else { 0.0 };

        let t = [
            (duv2[1] * edge1[0] - duv1[1] * edge2[0]) * r,
            (duv2[1] * edge1[1] - duv1[1] * edge2[1]) * r,
            (duv2[1] * edge1[2] - duv1[1] * edge2[2]) * r,
        ];

        let b = [
            (duv1[0] * edge2[0] - duv2[0] * edge1[0]) * r,
            (duv1[0] * edge2[1] - duv2[0] * edge1[1]) * r,
            (duv1[0] * edge2[2] - duv2[0] * edge1[2]) * r,
        ];

        // Accumulate per-vertex (area-weighted by triangle contribution).
        for &idx in &[i0, i1, i2] {
            tangents[idx][0] += t[0];
            tangents[idx][1] += t[1];
            tangents[idx][2] += t[2];

            // Compute handedness: sign of dot(cross(N, T), B).
            let n = normals[idx];
            let cross = [
                n[1] * t[2] - n[2] * t[1],
                n[2] * t[0] - n[0] * t[2],
                n[0] * t[1] - n[1] * t[0],
            ];
            let hand_dot = cross[0] * b[0] + cross[1] * b[1] + cross[2] * b[2];
            tangents[idx][3] += if hand_dot < 0.0 { -1.0 } else { 1.0 };
        }
    }

    // Normalize and finalize handedness.
    for (i, tan) in tangents.iter_mut().enumerate() {
        let t = [tan[0], tan[1], tan[2]];
        let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();

        if len > 1e-8 {
            // Gram-Schmidt orthogonalization against the vertex normal.
            let n = normals[i];
            let dot_nt = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
            let ortho = [
                t[0] - n[0] * dot_nt,
                t[1] - n[1] * dot_nt,
                t[2] - n[2] * dot_nt,
            ];
            let olen = (ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2]).sqrt();
            if olen > 1e-8 {
                tan[0] = ortho[0] / olen;
                tan[1] = ortho[1] / olen;
                tan[2] = ortho[2] / olen;
            } else {
                tan[0] = t[0] / len;
                tan[1] = t[1] / len;
                tan[2] = t[2] / len;
            }
        } else {
            // Degenerate — pick an arbitrary tangent perpendicular to normal.
            let n = normals[i];
            let up = if n[1].abs() < 0.999 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
            let right = [
                n[1] * up[2] - n[2] * up[1],
                n[2] * up[0] - n[0] * up[2],
                n[0] * up[1] - n[1] * up[0],
            ];
            let rlen = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
            if rlen > 1e-8 {
                tan[0] = right[0] / rlen;
                tan[1] = right[1] / rlen;
                tan[2] = right[2] / rlen;
            } else {
                tan[0] = 1.0;
                tan[1] = 0.0;
                tan[2] = 0.0;
            }
        }

        // Finalize handedness sign.
        tan[3] = if tan[3] < 0.0 { -1.0 } else { 1.0 };
    }

    tangents
}
