//! Phase 3: Global Illumination
//!
//! Provides spatially-varying indirect lighting via:
//! - **SH Probe Grid**: L2 spherical harmonics probes baked analytically from
//!   scene lights.  Fragment shader bilinearly interpolates 4 nearest probes
//!   in XZ and evaluates the SH polynomial with the surface normal.
//! - **BRDF LUT**: Pre-integrated split-sum BRDF lookup texture (512×512 RG16F).
//! - **Environment Map**: Procedural gradient sky cube map (placeholder for
//!   Phase 6 atmospheric scattering) convolved into diffuse irradiance and
//!   roughness-varying specular pre-filtered maps.
//!
//! Descriptor Set 0 new bindings (Phase 3):
//!   binding  6: Probe SSBO           (STORAGE_BUFFER)
//!   binding  7: ProbeGridParams      (UNIFORM_BUFFER_DYNAMIC)
//!   binding  8: BRDF LUT             (COMBINED_IMAGE_SAMPLER)
//!   binding  9: Irradiance cube map  (COMBINED_IMAGE_SAMPLER)
//!   binding 10: Pre-filtered env map (COMBINED_IMAGE_SAMPLER)

use ash::{vk, Device};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::time::Instant;
use crate::light::{cube_face_matrices, LightManager};
use crate::memory::{BufferHandle, GpuAllocator, ImageHandle, MemoryContext, MemoryLocation};
use crate::world::SectorCoord;

// ====================================================================
//  Constants
// ====================================================================

/// World-space distance between adjacent SH probes (meters).
pub const PROBE_SPACING: f32 = 32.0;

/// Default height at which probes are placed.
pub const PROBE_HEIGHT: f32 = 4.0;

/// Maximum probes along each horizontal axis.  32×32 = 1024 probes max
/// covers 1024m × 1024m — sufficient for the current 640m streaming radius.
pub const MAX_PROBES_X: u32 = 32;
pub const MAX_PROBES_Z: u32 = 32;
pub const MAX_PROBES: u32 = MAX_PROBES_X * MAX_PROBES_Z;

/// BRDF LUT resolution (square).
pub const BRDF_LUT_SIZE: u32 = 512;

/// Environment cube map face resolution.
pub const ENV_MAP_SIZE: u32 = 128;

/// Number of mip levels for the pre-filtered specular map.
/// log2(128) + 1 = 8 mips, but we only need ~6 for roughness mapping.
pub const ENV_MAP_MIP_LEVELS: u32 = 7;

/// Resolution of the probe capture cubemap (per face, in pixels).
pub const PROBE_CAPTURE_SIZE: u32 = 32;

/// Maximum probes baked per frame (6 render passes + 1 compute dispatch each).
pub const MAX_PROBE_BAKES_PER_FRAME: usize = 2;

/// Far plane distance for probe capture (meters).
pub const PROBE_CAPTURE_FAR: f32 = 500.0;

/// Near plane distance for probe capture.
pub const PROBE_CAPTURE_NEAR: f32 = 0.5;

/// Number of SH L2 basis functions.
const SH_COUNT: usize = 9;

// ====================================================================
//  SH Basis Constants
// ====================================================================

// Real spherical harmonics basis evaluation constants.
// Band 0:  Y00  = 0.282095  (1/(2√π))
// Band 1:  Y1-1 = 0.488603  (√3/(2√π))     * y
//          Y10  = 0.488603                   * z
//          Y11  = 0.488603                   * x
// Band 2:  Y2-2 = 1.092548  (√15/(2√π))    * xy
//          Y2-1 = 1.092548                   * yz
//          Y20  = 0.315392  (√5/(4√π))      * (3z²-1)
//          Y21  = 1.092548                   * xz
//          Y22  = 0.546274  (√15/(4√π))     * (x²-y²)

const SH_Y00:  f32 = 0.282_094_8;
const SH_Y1N1: f32 = 0.488_602_5;
const SH_Y10:  f32 = 0.488_602_5;
const SH_Y11:  f32 = 0.488_602_5;
const SH_Y2N2: f32 = 1.092_548_4;
const SH_Y2N1: f32 = 1.092_548_4;
const SH_Y20:  f32 = 0.315_391_6;
const SH_Y21:  f32 = 1.092_548_4;
const SH_Y22:  f32 = 0.546_274_2;

// Cosine-lobe convolution factors (Ramamoorthi & Hanrahan 2001).
// These pre-multiply the SH coefficients so that dot(SH, normal_SH) gives
// the irradiance integral under a clamped-cosine kernel.
const A_HAT: [f32; 3] = [
    std::f32::consts::PI,                                // l=0: π
    2.0 * std::f32::consts::PI / 3.0,                   // l=1: 2π/3
    std::f32::consts::PI / 4.0,                          // l=2: π/4
];

// Per-coefficient cosine convolution weights (A_hat[band] for each coeff).
const COSINE_WEIGHT: [f32; SH_COUNT] = [
    A_HAT[0],                    // L00
    A_HAT[1], A_HAT[1], A_HAT[1], // L1-1, L10, L11
    A_HAT[2], A_HAT[2], A_HAT[2], A_HAT[2], A_HAT[2], // L2-2..L22
];

// ====================================================================
//  SH Coefficient Storage
// ====================================================================

/// L2 spherical harmonics coefficients for a single color channel.
#[derive(Clone, Copy, Debug)]
pub struct SHBand {
    pub coeffs: [f32; SH_COUNT],
}

impl Default for SHBand {
    fn default() -> Self {
        Self { coeffs: [0.0; SH_COUNT] }
    }
}

impl SHBand {
    /// Add a scaled copy of another SHBand.
    #[inline]
    pub fn add_scaled(&mut self, other: &SHBand, scale: f32) {
        for i in 0..SH_COUNT {
            self.coeffs[i] += other.coeffs[i] * scale;
        }
    }

    /// Multiply all coefficients by a scalar.
    #[inline]
    pub fn scale(&mut self, s: f32) {
        for c in &mut self.coeffs { *c *= s; }
    }

    /// Apply cosine-lobe convolution (convert radiance SH → irradiance SH).
    pub fn convolve_cosine(&mut self) {
        for i in 0..SH_COUNT {
            self.coeffs[i] *= COSINE_WEIGHT[i];
        }
    }
}

/// Full RGB L2 spherical harmonics (3 channels × 9 coefficients = 27 floats).
#[derive(Clone, Copy, Debug, Default)]
pub struct SHCoeffsRGB {
    pub r: SHBand,
    pub g: SHBand,
    pub b: SHBand,
}

impl SHCoeffsRGB {
    /// Add a scaled copy of another set of coefficients.
    pub fn add_scaled(&mut self, other: &SHCoeffsRGB, scale: f32) {
        self.r.add_scaled(&other.r, scale);
        self.g.add_scaled(&other.g, scale);
        self.b.add_scaled(&other.b, scale);
    }

    /// Lerp between two SH coefficient sets.
    pub fn lerp(a: &SHCoeffsRGB, b: &SHCoeffsRGB, t: f32) -> SHCoeffsRGB {
        let mut result = *a;
        let t1 = 1.0 - t;
        for i in 0..SH_COUNT {
            result.r.coeffs[i] = a.r.coeffs[i] * t1 + b.r.coeffs[i] * t;
            result.g.coeffs[i] = a.g.coeffs[i] * t1 + b.g.coeffs[i] * t;
            result.b.coeffs[i] = a.b.coeffs[i] * t1 + b.b.coeffs[i] * t;
        }
        result
    }

    /// Apply cosine-lobe convolution to all channels.
    pub fn convolve_cosine(&mut self) {
        self.r.convolve_cosine();
        self.g.convolve_cosine();
        self.b.convolve_cosine();
    }
}

// ====================================================================
//  GPU Probe struct (must match GLSL layout)
// ====================================================================

/// GPU SH probe — 144 bytes, std430 layout.
///
/// Each of the 9 vec4s stores (R, G, B, 0) for one SH basis function.
/// This layout allows simple shader evaluation without channel-swizzling.
///
/// ```glsl
/// struct GpuSHProbe {
///     vec4 coeffs[9]; // [i].rgb = SH basis i for (R,G,B), .w = 0
/// };
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuSHProbe {
    pub coeffs: [[f32; 4]; SH_COUNT],
}

const _: () = assert!(std::mem::size_of::<GpuSHProbe>() == 144);

impl Default for GpuSHProbe {
    fn default() -> Self {
        Self { coeffs: [[0.0; 4]; SH_COUNT] }
    }
}

impl GpuSHProbe {
    /// Convert from CPU-side RGB SH coefficients to GPU layout.
    #[allow(dead_code)]
    pub fn from_sh(sh: &SHCoeffsRGB) -> Self {
        let mut gpu = Self::default();
        for i in 0..SH_COUNT {
            gpu.coeffs[i] = [sh.r.coeffs[i], sh.g.coeffs[i], sh.b.coeffs[i], 0.0];
        }
        gpu
    }
}

/// SSBO header for the probe array.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProbeSsboHeader {
    pub probe_count: u32,
    pub _pad: [u32; 3],
}

/// Probe grid parameters UBO — passed as dynamic UBO through the ring buffer.
///
/// ```glsl
/// layout(set = 0, binding = 7, std140) uniform ProbeGridParams {
///     vec4  grid_origin;   // xyz = world origin, w = spacing
///     uvec4 grid_dims;     // x, z, total_probes, _pad
///     vec4  probe_config;  // probe_height, blend_weight, time_of_day, _pad
/// };
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuProbeGridParams {
    pub grid_origin: [f32; 4],  // xyz = world-space origin, w = spacing
    pub grid_dims: [u32; 4],    // x, z, total_probes, _pad
    pub probe_config: [f32; 4], // probe_height, blend_weight, time_of_day, _pad
}

const _: () = assert!(std::mem::size_of::<GpuProbeGridParams>() == 48);

impl GpuProbeGridParams {
    pub fn new(origin: [f32; 3], spacing: f32, dims_x: u32, dims_z: u32,
               probe_count: u32, probe_height: f32, time_of_day: f32) -> Self {
        Self {
            grid_origin: [origin[0], origin[1], origin[2], spacing],
            grid_dims: [dims_x, dims_z, probe_count, 0],
            probe_config: [probe_height, 1.0, time_of_day, 0.0],
        }
    }
}

// ====================================================================
//  SH Projection Functions
// ====================================================================

/// Evaluate all 9 SH basis functions for a direction vector (must be normalized).
#[allow(dead_code)]
fn sh_basis(dir: [f32; 3]) -> [f32; SH_COUNT] {
    let x = dir[0];
    let y = dir[1];
    let z = dir[2];
    [
        SH_Y00,
        SH_Y1N1 * y,
        SH_Y10  * z,
        SH_Y11  * x,
        SH_Y2N2 * x * y,
        SH_Y2N1 * y * z,
        SH_Y20  * (3.0 * z * z - 1.0),
        SH_Y21  * x * z,
        SH_Y22  * (x * x - y * y),
    ]
}

/// Project a point light contribution into SH coefficients for a probe.
///
/// Uses the analytic SH projection of a clamped-cosine-weighted point source.
/// The light is treated as a directional source from the probe's perspective,
/// weighted by inverse-square attenuation.
#[allow(dead_code)]
fn project_point_light_sh(
    probe_pos: [f32; 3],
    light_pos: [f32; 3],
    light_color: [f32; 3],
    light_intensity: f32,
    light_radius: f32,
) -> SHCoeffsRGB {
    let dx = light_pos[0] - probe_pos[0];
    let dy = light_pos[1] - probe_pos[1];
    let dz = light_pos[2] - probe_pos[2];
    let dist_sq = dx * dx + dy * dy + dz * dz;
    let dist = dist_sq.sqrt();

    if dist < 0.001 || dist > light_radius {
        return SHCoeffsRGB::default();
    }

    // Normalized direction from probe to light.
    let inv_dist = 1.0 / dist;
    let dir = [dx * inv_dist, dy * inv_dist, dz * inv_dist];

    // Smooth distance attenuation matching pbr.frag.
    let dist_ratio = dist / light_radius;
    let falloff = {
        let f = 1.0 - dist_ratio * dist_ratio;
        let f = f.max(0.0);
        f * f
    };
    let attenuation = falloff / dist_sq.max(0.001);

    // Radiance from the light as seen at the probe.
    let radiance = [
        light_color[0] * light_intensity * attenuation,
        light_color[1] * light_intensity * attenuation,
        light_color[2] * light_intensity * attenuation,
    ];

    // Project into SH: weight = 4π (full sphere solid angle normalization
    // for a point source) — the cosine convolution is applied later.
    let basis = sh_basis(dir);
    let weight = 4.0 * std::f32::consts::PI;

    let mut sh = SHCoeffsRGB::default();
    for i in 0..SH_COUNT {
        sh.r.coeffs[i] = radiance[0] * basis[i] * weight;
        sh.g.coeffs[i] = radiance[1] * basis[i] * weight;
        sh.b.coeffs[i] = radiance[2] * basis[i] * weight;
    }
    sh
}

/// Project a directional light into SH.
#[allow(dead_code)]
fn project_directional_light_sh(
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
) -> SHCoeffsRGB {
    // Incoming light direction (flip from light.direction which is "where light points").
    let dir = [-direction[0], -direction[1], -direction[2]];
    let basis = sh_basis(dir);
    let weight = 4.0 * std::f32::consts::PI;

    let mut sh = SHCoeffsRGB::default();
    for i in 0..SH_COUNT {
        sh.r.coeffs[i] = color[0] * intensity * basis[i] * weight;
        sh.g.coeffs[i] = color[1] * intensity * basis[i] * weight;
        sh.b.coeffs[i] = color[2] * intensity * basis[i] * weight;
    }
    sh
}

/// Add a procedural sky contribution to SH coefficients.
///
/// Models a simple gradient sky: blue at zenith, warm horizon, ground bounce.
/// This is a placeholder for Phase 6's atmospheric scattering.
#[allow(dead_code)]
fn project_sky_sh(sun_elevation: f32) -> SHCoeffsRGB {
    // Approximate sky as a few directional samples.
    let sun_factor = sun_elevation.max(0.0).min(1.0);

    // Zenith color (deep blue, intensity varies with sun elevation).
    let zenith_r = 0.05 + 0.10 * sun_factor;
    let zenith_g = 0.08 + 0.15 * sun_factor;
    let zenith_b = 0.15 + 0.35 * sun_factor;

    // Horizon color (warm, brighter with sun).
    let horiz_r = 0.12 + 0.20 * sun_factor;
    let horiz_g = 0.10 + 0.15 * sun_factor;
    let horiz_b = 0.08 + 0.10 * sun_factor;

    // Ground bounce (dark, muted).
    let ground_r = 0.02;
    let ground_g = 0.02;
    let ground_b = 0.01;

    let mut sh = SHCoeffsRGB::default();

    // L00: uniform ambient (average of all directions).
    let avg_r = (zenith_r + horiz_r * 4.0 + ground_r) / 6.0;
    let avg_g = (zenith_g + horiz_g * 4.0 + ground_g) / 6.0;
    let avg_b = (zenith_b + horiz_b * 4.0 + ground_b) / 6.0;
    sh.r.coeffs[0] = avg_r * SH_Y00 * 4.0 * std::f32::consts::PI;
    sh.g.coeffs[0] = avg_g * SH_Y00 * 4.0 * std::f32::consts::PI;
    sh.b.coeffs[0] = avg_b * SH_Y00 * 4.0 * std::f32::consts::PI;

    // L1 (Y direction = up): zenith - ground gradient.
    let vert_r = (zenith_r - ground_r) * 0.5;
    let vert_g = (zenith_g - ground_g) * 0.5;
    let vert_b = (zenith_b - ground_b) * 0.5;
    sh.r.coeffs[1] = vert_r * SH_Y1N1 * 4.0 * std::f32::consts::PI;
    sh.g.coeffs[1] = vert_g * SH_Y1N1 * 4.0 * std::f32::consts::PI;
    sh.b.coeffs[1] = vert_b * SH_Y1N1 * 4.0 * std::f32::consts::PI;

    sh
}

// ====================================================================
//  BRDF LUT Generation (Split-Sum Approximation)
// ====================================================================

/// Generate a BRDF integration LUT for the split-sum IBL approximation.
///
/// Each texel stores (scale, bias) such that:
///   specular_ibl = F0 * scale + bias
///
/// Parameterized by: u = NdotV, v = roughness.
/// Output: `BRDF_LUT_SIZE × BRDF_LUT_SIZE` pixels, 2 channels (RG), f16.
pub fn generate_brdf_lut() -> Vec<u8> {
    let size = BRDF_LUT_SIZE as usize;
    // Output as R16G16 (4 bytes per pixel).
    let mut data = vec![0u8; size * size * 4];

    const NUM_SAMPLES: u32 = 1024;

    for y in 0..size {
        let roughness = (y as f32 + 0.5) / size as f32;
        let roughness = roughness.max(0.04);
        let a = roughness * roughness;

        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let n_dot_v = n_dot_v.max(0.001);

            let v = [
                (1.0 - n_dot_v * n_dot_v).sqrt(), // sin
                0.0,
                n_dot_v, // cos
            ];
            let n = [0.0f32, 0.0, 1.0];

            let mut scale = 0.0f32;
            let mut bias = 0.0f32;

            for i in 0..NUM_SAMPLES {
                // Hammersley sequence.
                let xi = hammersley(i, NUM_SAMPLES);

                // Importance-sample GGX.
                let h = importance_sample_ggx(xi, a);
                let v_dot_h = (v[0] * h[0] + v[1] * h[1] + v[2] * h[2]).max(0.0);
                let l = [
                    2.0 * v_dot_h * h[0] - v[0],
                    2.0 * v_dot_h * h[1] - v[1],
                    2.0 * v_dot_h * h[2] - v[2],
                ];
                let n_dot_l = l[2].max(0.0); // n = (0,0,1)
                let n_dot_h = h[2].max(0.0);

                if n_dot_l > 0.0 {
                    let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
                    let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(0.001);
                    let fc = (1.0 - v_dot_h).powi(5);

                    scale += g_vis * (1.0 - fc);
                    bias += g_vis * fc;
                }
            }

            scale /= NUM_SAMPLES as f32;
            bias /= NUM_SAMPLES as f32;

            // Write as f16 (half-float).
            let offset = (y * size + x) * 4;
            let s_half = f32_to_f16(scale);
            let b_half = f32_to_f16(bias);
            data[offset..offset + 2].copy_from_slice(&s_half.to_le_bytes());
            data[offset + 2..offset + 4].copy_from_slice(&b_half.to_le_bytes());
        }
    }

    data
}

fn hammersley(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    [
        i as f32 / n as f32,
        bits as f32 * 2.328_306_4e-10, // 1.0 / 0x100000000
    ]
}

fn importance_sample_ggx(xi: [f32; 2], a2: f32) -> [f32; 3] {
    let phi = 2.0 * std::f32::consts::PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a2 * a2 - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(0.0);
    [
        sin_theta * phi.cos(),
        sin_theta * phi.sin(),
        cos_theta,
    ]
}

fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    g1_v * g1_l
}

/// Convert f32 to IEEE 754 half-float (f16).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 255 {
        // Inf / NaN.
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }

    let exp16 = exponent - 127 + 15;
    if exp16 >= 31 {
        return (sign | 0x7C00) as u16; // overflow → inf
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign as u16; // underflow → zero
        }
        let m = (mantissa | 0x800000) >> (1 - exp16);
        return (sign | (m >> 13)) as u16;
    }
    (sign | ((exp16 as u32) << 10) | (mantissa >> 13)) as u16
}

// ====================================================================
//  Environment Cube Map Generation (Procedural Sky)
// ====================================================================

/// Generate a procedural gradient sky cube map.
///
/// Returns 6 face images (RGBA8) in order: +X, -X, +Y, -Y, +Z, -Z.
/// Each face is `ENV_MAP_SIZE × ENV_MAP_SIZE` pixels.
pub fn generate_sky_cubemap(sun_dir: [f32; 3]) -> Vec<Vec<u8>> {
    let size = ENV_MAP_SIZE as usize;
    let mut faces = Vec::with_capacity(6);

    for face in 0..6u32 {
        let mut pixels = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                // Map pixel to cube map direction.
                let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let dir = cube_face_dir(face, u, v);
                let dir = normalize3(dir);

                // Sky color based on elevation.
                let elevation = dir[1]; // -1 (nadir) to 1 (zenith)

                let (r, g, b) = if elevation > 0.0 {
                    // Above horizon: gradient from warm horizon to blue zenith.
                    let t = elevation;
                    let sun_dot = dot3(dir, sun_dir).max(0.0);
                    let sun_glow = sun_dot.powf(64.0) * 2.0;
                    let sun_haze = sun_dot.powf(8.0) * 0.3;
                    (
                        lerp(0.7, 0.15, t) + sun_glow + sun_haze * 0.8,
                        lerp(0.55, 0.25, t) + sun_glow * 0.8 + sun_haze * 0.5,
                        lerp(0.45, 0.6, t) + sun_glow * 0.3 + sun_haze * 0.3,
                    )
                } else {
                    // Below horizon: dark ground.
                    let t = (-elevation).min(1.0);
                    (
                        lerp(0.15, 0.02, t),
                        lerp(0.12, 0.02, t),
                        lerp(0.10, 0.01, t),
                    )
                };

                let offset = (y * size + x) * 4;
                pixels[offset]     = float_to_u8(r);
                pixels[offset + 1] = float_to_u8(g);
                pixels[offset + 2] = float_to_u8(b);
                pixels[offset + 3] = 255;
            }
        }
        faces.push(pixels);
    }
    faces
}

/// Generate a diffuse irradiance cube map by convolving the sky cubemap.
///
/// Uses Monte Carlo integration with cosine-weighted hemisphere sampling.
/// Output face resolution is typically smaller (e.g. 32×32).
pub fn convolve_irradiance_cubemap(sky_faces: &[Vec<u8>], out_size: u32) -> Vec<Vec<u8>> {
    let size = out_size as usize;
    let sky_size = ENV_MAP_SIZE as usize;
    let mut faces = Vec::with_capacity(6);

    const NUM_SAMPLES: u32 = 256;

    for face in 0..6u32 {
        let mut pixels = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let normal = normalize3(cube_face_dir(face, u, v));

                // Build TBN from normal.
                let up = if normal[1].abs() < 0.999 { [0.0, 1.0, 0.0] }
                         else { [0.0, 0.0, 1.0] };
                let tangent = normalize3(cross3(up, normal));
                let bitangent = cross3(normal, tangent);

                let mut irradiance = [0.0f32; 3];
                let mut total_weight = 0.0f32;

                for i in 0..NUM_SAMPLES {
                    let xi = hammersley(i, NUM_SAMPLES);
                    // Cosine-weighted hemisphere.
                    let phi = 2.0 * std::f32::consts::PI * xi[0];
                    let cos_theta = (1.0 - xi[1]).sqrt();
                    let sin_theta = xi[1].sqrt();

                    let local = [
                        sin_theta * phi.cos(),
                        sin_theta * phi.sin(),
                        cos_theta,
                    ];
                    let world = [
                        tangent[0] * local[0] + bitangent[0] * local[1] + normal[0] * local[2],
                        tangent[1] * local[0] + bitangent[1] * local[1] + normal[1] * local[2],
                        tangent[2] * local[0] + bitangent[2] * local[1] + normal[2] * local[2],
                    ];

                    let color = sample_cubemap(sky_faces, sky_size, world);
                    irradiance[0] += color[0];
                    irradiance[1] += color[1];
                    irradiance[2] += color[2];
                    total_weight += 1.0;
                }

                if total_weight > 0.0 {
                    let inv = std::f32::consts::PI / total_weight;
                    irradiance[0] *= inv;
                    irradiance[1] *= inv;
                    irradiance[2] *= inv;
                }

                let offset = (y * size + x) * 4;
                pixels[offset]     = float_to_u8(irradiance[0]);
                pixels[offset + 1] = float_to_u8(irradiance[1]);
                pixels[offset + 2] = float_to_u8(irradiance[2]);
                pixels[offset + 3] = 255;
            }
        }
        faces.push(pixels);
    }
    faces
}

/// Generate pre-filtered specular environment map at a given roughness level.
pub fn prefilter_env_cubemap(
    sky_faces: &[Vec<u8>],
    out_size: u32,
    roughness: f32,
) -> Vec<Vec<u8>> {
    let size = out_size as usize;
    let sky_size = ENV_MAP_SIZE as usize;
    let a = roughness * roughness;
    let mut faces = Vec::with_capacity(6);

    let num_samples: u32 = if roughness < 0.05 { 64 } else { 256 };

    for face in 0..6u32 {
        let mut pixels = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let n = normalize3(cube_face_dir(face, u, v));
                let r_dir = n; // Approximation: reflection = normal
                let v_dir = r_dir;

                let up = if n[1].abs() < 0.999 { [0.0, 1.0, 0.0] }
                         else { [0.0, 0.0, 1.0] };
                let tangent = normalize3(cross3(up, n));
                let bitangent = cross3(n, tangent);

                let mut color = [0.0f32; 3];
                let mut total_weight = 0.0f32;

                for i in 0..num_samples {
                    let xi = hammersley(i, num_samples);
                    let h_local = importance_sample_ggx(xi, a);

                    let h = [
                        tangent[0] * h_local[0] + bitangent[0] * h_local[1] + n[0] * h_local[2],
                        tangent[1] * h_local[0] + bitangent[1] * h_local[1] + n[1] * h_local[2],
                        tangent[2] * h_local[0] + bitangent[2] * h_local[1] + n[2] * h_local[2],
                    ];

                    let v_dot_h = dot3(v_dir, h).max(0.0);
                    let l = [
                        2.0 * v_dot_h * h[0] - v_dir[0],
                        2.0 * v_dot_h * h[1] - v_dir[1],
                        2.0 * v_dot_h * h[2] - v_dir[2],
                    ];
                    let n_dot_l = dot3(n, l);

                    if n_dot_l > 0.0 {
                        let sample = sample_cubemap(sky_faces, sky_size, l);
                        color[0] += sample[0] * n_dot_l;
                        color[1] += sample[1] * n_dot_l;
                        color[2] += sample[2] * n_dot_l;
                        total_weight += n_dot_l;
                    }
                }

                if total_weight > 0.0 {
                    color[0] /= total_weight;
                    color[1] /= total_weight;
                    color[2] /= total_weight;
                }

                let offset = (y * size + x) * 4;
                pixels[offset]     = float_to_u8(color[0]);
                pixels[offset + 1] = float_to_u8(color[1]);
                pixels[offset + 2] = float_to_u8(color[2]);
                pixels[offset + 3] = 255;
            }
        }
        faces.push(pixels);
    }
    faces
}

// ====================================================================
//  Cube Map Helpers
// ====================================================================

/// Map (face, u, v) in [-1,1] to a world-space direction.
fn cube_face_dir(face: u32, u: f32, v: f32) -> [f32; 3] {
    match face {
        0 => [ 1.0, -v, -u],  // +X
        1 => [-1.0, -v,  u],  // -X
        2 => [ u,  1.0,  v],  // +Y
        3 => [ u, -1.0, -v],  // -Y
        4 => [ u, -v,  1.0],  // +Z
        5 => [-u, -v, -1.0],  // -Z
        _ => [0.0, 0.0, 1.0],
    }
}

/// Sample a cubemap (6 RGBA8 face images) at a given direction.
fn sample_cubemap(faces: &[Vec<u8>], size: usize, dir: [f32; 3]) -> [f32; 3] {
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();

    let (face, u, v) = if ax >= ay && ax >= az {
        if dir[0] > 0.0 {
            (0, -dir[2] / ax, -dir[1] / ax)
        } else {
            (1, dir[2] / ax, -dir[1] / ax)
        }
    } else if ay >= ax && ay >= az {
        if dir[1] > 0.0 {
            (2, dir[0] / ay, dir[2] / ay)
        } else {
            (3, dir[0] / ay, -dir[2] / ay)
        }
    } else if dir[2] > 0.0 {
        (4, dir[0] / az, -dir[1] / az)
    } else {
        (5, -dir[0] / az, -dir[1] / az)
    };

    let px = ((u * 0.5 + 0.5) * size as f32).clamp(0.0, (size - 1) as f32) as usize;
    let py = ((v * 0.5 + 0.5) * size as f32).clamp(0.0, (size - 1) as f32) as usize;
    let offset = (py * size + px) * 4;

    if offset + 2 < faces[face].len() {
        [
            faces[face][offset] as f32 / 255.0,
            faces[face][offset + 1] as f32 / 255.0,
            faces[face][offset + 2] as f32 / 255.0,
        ]
    } else {
        [0.0; 3]
    }
}

// ====================================================================
//  ProbeGrid — CPU-side probe management
// ====================================================================

/// A single probe's state.
#[derive(Clone)]
pub struct Probe {
    pub position: [f32; 3],
    /// Current irradiance SH (cosine-convolved).
    pub irradiance: SHCoeffsRGB,
    /// Grid-local index (row-major: iz * dims_x + ix).
    pub grid_index: u32,
    /// Has this probe been baked at least once?
    pub baked: bool,
}

/// Manages the SH probe grid and its GPU SSBO.
pub struct ProbeGrid {
    /// Grid origin in world-space (min-corner XZ, at probe_height Y).
    pub origin: [f32; 3],
    pub spacing: f32,
    pub dims_x: u32,
    pub dims_z: u32,

    /// All probes in row-major order (Z-major: z * dims_x + x).
    pub probes: Vec<Probe>,

    /// Sector → set of probe indices affected by that sector's lights.
    sector_probe_map: HashMap<SectorCoord, Vec<usize>>,

    /// Queue of probe indices that need GPU cubemap re-baking.
    bake_queue: VecDeque<usize>,

    /// GPU SSBO: header + GpuSHProbe array.
    ssbo_handle: BufferHandle,
    ssbo_buffer: vk::Buffer,
    ssbo_mapped: NonNull<u8>,
    ssbo_capacity: u64,

    /// Dirty flag — set when any probe's SH changes (CPU-side analytical fallback).
    dirty: bool,

    /// Time-of-day parameter (0.0 = dawn, 0.5 = noon, 1.0 = dusk).
    pub time_of_day: f32,
}

unsafe impl Send for ProbeGrid {}

impl ProbeGrid {
    /// Create a new probe grid centered on `center_xz`.
    pub fn new(
        allocator: &mut GpuAllocator,
        center_xz: [f32; 2],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dims_x = MAX_PROBES_X;
        let dims_z = MAX_PROBES_Z;
        let spacing = PROBE_SPACING;

        let origin = [
            center_xz[0] - (dims_x as f32 * spacing) * 0.5,
            PROBE_HEIGHT,
            center_xz[1] - (dims_z as f32 * spacing) * 0.5,
        ];

        let total = (dims_x * dims_z) as usize;
        let mut probes = Vec::with_capacity(total);

        for iz in 0..dims_z {
            for ix in 0..dims_x {
                let pos = [
                    origin[0] + ix as f32 * spacing + spacing * 0.5,
                    PROBE_HEIGHT,
                    origin[2] + iz as f32 * spacing + spacing * 0.5,
                ];
                probes.push(Probe {
                    position: pos,
                    irradiance: SHCoeffsRGB::default(),
                    grid_index: iz * dims_x + ix,
                    baked: false,
                });
            }
        }

        // Allocate SSBO: header + max_probes × GpuSHProbe.
        let header_size = std::mem::size_of::<ProbeSsboHeader>() as u64;
        let probe_data_size = (total * std::mem::size_of::<GpuSHProbe>()) as u64;
        let ssbo_capacity = header_size + probe_data_size;

        let alloc = allocator.create_buffer(
            ssbo_capacity,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;

        let ssbo_mapped = alloc.mapped_ptr
            .ok_or("ProbeGrid SSBO: CpuToGpu buffer was not mapped")?;

        // Write initial zero data.
        unsafe {
            std::ptr::write_bytes(ssbo_mapped.as_ptr(), 0, ssbo_capacity as usize);
        }

        println!(
            "[ProbeGrid] {}×{} probes ({} total), spacing {}m, SSBO {:.1} KB",
            dims_x, dims_z, total, spacing, ssbo_capacity as f64 / 1024.0,
        );

        Ok(Self {
            origin,
            spacing,
            dims_x,
            dims_z,
            probes,
            sector_probe_map: HashMap::new(),
            bake_queue: VecDeque::new(),
            ssbo_handle: alloc.handle,
            ssbo_buffer: alloc.buffer,
            ssbo_mapped,
            ssbo_capacity,
            dirty: true,
            time_of_day: 0.5, // noon default
        })
    }

    /// The VkBuffer for descriptor binding.
    pub fn ssbo_buffer(&self) -> vk::Buffer { self.ssbo_buffer }

    /// SSBO byte size for descriptor range.
    pub fn ssbo_size(&self) -> u64 { self.ssbo_capacity }

    /// BufferHandle for cleanup.
    pub fn ssbo_handle(&self) -> BufferHandle { self.ssbo_handle }

    /// Probe count.
    pub fn probe_count(&self) -> u32 { self.probes.len() as u32 }

    /// Build the GpuProbeGridParams for the current frame.
    pub fn gpu_params(&self) -> GpuProbeGridParams {
        GpuProbeGridParams::new(
            self.origin, self.spacing,
            self.dims_x, self.dims_z,
            self.probes.len() as u32,
            PROBE_HEIGHT, self.time_of_day,
        )
    }

    /// Queue probes affected by a newly-loaded sector for GPU cubemap baking.
    ///
    /// Finds all probes within influence range of the sector and enqueues
    /// them for incremental GPU capture (MAX_PROBE_BAKES_PER_FRAME per frame).
    pub fn bake_sector_probes(
        &mut self,
        sector: SectorCoord,
        _light_manager: &LightManager,
    ) {
        let sector_center_x = sector.0 as f32 * crate::world::SECTOR_SIZE
            + crate::world::SECTOR_SIZE * 0.5;
        let sector_center_z = sector.1 as f32 * crate::world::SECTOR_SIZE
            + crate::world::SECTOR_SIZE * 0.5;

        // Influence radius: sector diagonal + max light radius.
        let half_diag = crate::world::SECTOR_SIZE * 0.707;
        let max_light_radius: f32 = 100.0;
        let influence_radius = half_diag + max_light_radius;
        let influence_sq = influence_radius * influence_radius;

        let mut affected = Vec::new();

        for (i, probe) in self.probes.iter().enumerate() {
            let dx = probe.position[0] - sector_center_x;
            let dz = probe.position[2] - sector_center_z;
            if dx * dx + dz * dz < influence_sq {
                affected.push(i);
            }
        }

        if affected.is_empty() { return; }

        // Enqueue for GPU baking (deduplicate: don't re-add if already queued).
        for &pi in &affected {
            if !self.bake_queue.contains(&pi) {
                self.bake_queue.push_back(pi);
            }
        }

        self.sector_probe_map.insert(sector, affected);

        // Upload header so the shader knows probe count (even before GPU bake).
        self.dirty = true;
    }

    /// Number of probes waiting for GPU cubemap bake.
    pub fn pending_bake_count(&self) -> usize {
        self.bake_queue.len()
    }

    /// Pop the next probe index that needs GPU baking.
    /// Returns `(probe_index, probe_world_position)`.
    pub fn next_bake_probe(&mut self) -> Option<(u32, [f32; 3])> {
        let pi = self.bake_queue.pop_front()?;
        if pi >= self.probes.len() { return None; }
        let pos = self.probes[pi].position;
        self.probes[pi].baked = true;
        Some((pi as u32, pos))
    }

    /// Get the 6 (view, projection) matrix pairs for cubemap capture from a probe position.
    /// Reuses the same face convention as the shadow atlas cube maps.
    pub fn probe_capture_matrices(probe_pos: [f32; 3]) -> [([[f32; 4]; 4], [[f32; 4]; 4]); 6] {
        let mut result = [([[0.0f32; 4]; 4], [[0.0f32; 4]; 4]); 6];
        for face in 0..6u32 {
            result[face as usize] = cube_face_matrices(probe_pos, PROBE_CAPTURE_FAR, face);
        }
        result
    }

    /// SSBO offset (in bytes) for a specific probe's SH data.
    /// Used by the SH projection compute shader push constants.
    pub fn probe_ssbo_offset(probe_index: u32) -> u64 {
        let header_size = std::mem::size_of::<ProbeSsboHeader>() as u64;
        let probe_stride = std::mem::size_of::<GpuSHProbe>() as u64;
        header_size + (probe_index as u64) * probe_stride
    }

    /// Invalidate probes when a sector is evicted.
    pub fn invalidate_sector(&mut self, sector: SectorCoord) {
        if let Some(indices) = self.sector_probe_map.remove(&sector) {
            // Don't zero the probes — they retain their last-baked values.
            // They'll be re-baked when nearby sectors load again.
            // Just mark dirty so the GPU data is refreshed.
            self.dirty = true;
            let _ = indices; // suppress unused warning
        }
    }

    /// Upload SSBO header (probe count) to GPU.
    /// The actual SH coefficient data is written by the SH projection compute shader.
    pub fn upload_if_dirty(&mut self) {
        if !self.dirty { return; }
        self.dirty = false;

        let header = ProbeSsboHeader {
            probe_count: self.probes.len() as u32,
            _pad: [0; 3],
        };
        let header_size = std::mem::size_of::<ProbeSsboHeader>();

        unsafe {
            std::ptr::copy_nonoverlapping(
                &header as *const _ as *const u8,
                self.ssbo_mapped.as_ptr(),
                header_size,
            );
        }
    }
}

// ====================================================================
//  GIResources — Vulkan resources for environment IBL
// ====================================================================

/// Holds the BRDF LUT, irradiance cube map, and pre-filtered env map
/// Vulkan resources for IBL sampling in the PBR shader.
pub struct GIResources {
    pub brdf_lut_handle: ImageHandle,
    pub brdf_lut_view: vk::ImageView,
    pub brdf_lut_sampler: vk::Sampler,

    pub irradiance_image: vk::Image,
    pub irradiance_memory: vk::DeviceMemory,
    pub irradiance_view: vk::ImageView,
    pub irradiance_sampler: vk::Sampler,

    pub prefiltered_image: vk::Image,
    pub prefiltered_memory: vk::DeviceMemory,
    pub prefiltered_view: vk::ImageView,
    pub prefiltered_sampler: vk::Sampler,
}

impl GIResources {
    /// Create all GI textures: BRDF LUT, irradiance cube map, pre-filtered env map.
    pub fn new(
        device: &Device,
        memory_ctx: &mut MemoryContext,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let start = Instant::now();
        // ---- BRDF LUT ----
        println!("[GI] Generating BRDF LUT ({}×{})...", BRDF_LUT_SIZE, BRDF_LUT_SIZE);
        let brdf_data = generate_brdf_lut();
        let brdf_alloc = memory_ctx.create_image_with_data(
            &brdf_data,
            BRDF_LUT_SIZE, BRDF_LUT_SIZE,
            vk::Format::R16G16_SFLOAT,
            command_pool, queue,
        )?;

        let brdf_lut_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(1.0),
                None,
            )?
        };
        let start2 = Instant::now();
        // ---- Sky cube map generation ----
        let sun_dir = normalize3([0.5, 0.7, 0.3]);
        println!("[GI] Generating sky cube map ({}×{} per face)...", ENV_MAP_SIZE, ENV_MAP_SIZE);
        let sky_faces = generate_sky_cubemap(sun_dir);
        let start3 = Instant::now();
        // ---- Irradiance cube map (diffuse convolution) ----
        let irr_size = 32u32;
        println!("[GI] Convolving irradiance cube map ({}×{})...", irr_size, irr_size);
        let irr_faces = convolve_irradiance_cubemap(&sky_faces, irr_size);

        let (irradiance_image, irradiance_memory, irradiance_view) =
            create_cubemap_from_faces(
                device, memory_ctx,
                &irr_faces, irr_size, 1,
                command_pool, queue,
            )?;

        let irradiance_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(1.0),
                None,
            )?
        };
        let start4 = Instant::now();
        // ---- Pre-filtered specular env map (mip chain for roughness) ----
        println!("[GI] Pre-filtering specular env map ({} mip levels)...", ENV_MAP_MIP_LEVELS);
        let mut prefilter_mips: Vec<Vec<Vec<u8>>> = Vec::with_capacity(ENV_MAP_MIP_LEVELS as usize);
        for mip in 0..ENV_MAP_MIP_LEVELS {
            let mip_size = (ENV_MAP_SIZE >> mip).max(1);
            let roughness = mip as f32 / (ENV_MAP_MIP_LEVELS - 1).max(1) as f32;
            let faces = prefilter_env_cubemap(&sky_faces, mip_size, roughness);
            prefilter_mips.push(faces);
        }

        let (prefiltered_image, prefiltered_memory, prefiltered_view) =
            create_cubemap_from_mips(
                device, memory_ctx,
                &prefilter_mips, ENV_MAP_SIZE, ENV_MAP_MIP_LEVELS,
                command_pool, queue,
            )?;

        let prefiltered_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(ENV_MAP_MIP_LEVELS as f32),
                None,
            )?
        };

        println!("[GI] All resources initialized.");
        println!("        [GI-BRDF LUT] took {}sec
        [GI-SkyMap LUT] took {}sec
        [GI-IrrMap] took {}sec
        [GI-IBL] took {}sec", start.elapsed().as_secs(), 
                  start2.elapsed().as_secs(), start3.elapsed().as_secs(), 
                  start4.elapsed().as_secs());

        Ok(Self {
            brdf_lut_handle: brdf_alloc.handle,
            brdf_lut_view: brdf_alloc.view,
            brdf_lut_sampler,
            irradiance_image, irradiance_memory, irradiance_view, irradiance_sampler,
            prefiltered_image, prefiltered_memory, prefiltered_view, prefiltered_sampler,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.prefiltered_sampler, None);
            device.destroy_image_view(self.prefiltered_view, None);
            device.destroy_image(self.prefiltered_image, None);
            device.free_memory(self.prefiltered_memory, None);

            device.destroy_sampler(self.irradiance_sampler, None);
            device.destroy_image_view(self.irradiance_view, None);
            device.destroy_image(self.irradiance_image, None);
            device.free_memory(self.irradiance_memory, None);

            device.destroy_sampler(self.brdf_lut_sampler, None);
            // brdf_lut image/view owned by GpuAllocator via handle
        }
    }
}

// ====================================================================
//  ProbeBakeTarget — GPU cubemap capture resources
// ====================================================================

/// Push constants for the SH projection compute shader.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SHProjectPushConstants {
    pub probe_index: u32,
    pub face_size: u32,
}

/// Vulkan resources for rendering a probe cubemap and projecting it to SH.
///
/// Holds one reusable 32×32×6 HDR cubemap target (color + depth), per-face
/// framebuffers, and the SH projection compute shader's descriptor set.
/// One instance is shared across all probe bakes (baked serially).
pub struct ProbeBakeTarget {
    /// HDR color cubemap: PROBE_CAPTURE_SIZE², 6 layers, R16G16B16A16_SFLOAT.
    pub color_image: vk::Image,
    pub color_memory: vk::DeviceMemory,
    /// Per-face image views (for framebuffer attachment).
    pub color_face_views: Vec<vk::ImageView>,
    /// Cube view (for SH compute shader sampling).
    pub color_cube_view: vk::ImageView,

    /// Depth image: PROBE_CAPTURE_SIZE², single layer, D32_SFLOAT (reused per face).
    pub depth_image: vk::Image,
    pub depth_memory: vk::DeviceMemory,
    pub depth_view: vk::ImageView,

    /// 6 framebuffers (one per cube face).
    pub framebuffers: Vec<vk::Framebuffer>,

    /// Sampler for the SH compute shader's cubemap read.
    pub sampler: vk::Sampler,

    /// SH compute shader descriptor pool, layout, and set.
    pub sh_compute_layout: vk::DescriptorSetLayout,
    pub sh_compute_pool: vk::DescriptorPool,
    pub sh_compute_set: vk::DescriptorSet,
    pub sh_compute_pipeline_layout: vk::PipelineLayout,
    pub sh_compute_pipeline: vk::Pipeline,
}

impl ProbeBakeTarget {
    /// Create the probe bake target resources.
    ///
    /// `probe_capture_pass` — the render pass for probe cubemap capture.
    /// `probe_ssbo` / `probe_ssbo_size` — the probe grid SSBO (SH compute writes here).
    /// `sh_compute_spv` — compiled sh_project.comp SPIR-V.
    pub fn new(
        device: &Device,
        allocator: &GpuAllocator,
        probe_capture_pass: vk::RenderPass,
        probe_ssbo: vk::Buffer,
        probe_ssbo_size: u64,
        sh_compute_spv: &[u8],
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = PROBE_CAPTURE_SIZE;

        // ---- Color cubemap (HDR) ----
        let color_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D { width: size, height: size, depth: 1 })
            .mip_levels(1)
            .array_layers(6)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

        let color_image = unsafe { device.create_image(&color_info, None)? };
        let color_req = unsafe { device.get_image_memory_requirements(color_image) };
        let color_mem_type = allocator.find_memory_type(
            color_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let color_memory = unsafe { device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(color_req.size)
                .memory_type_index(color_mem_type), None)? };
        unsafe { device.bind_image_memory(color_image, color_memory, 0)? };

        // Per-face views for framebuffer.
        let mut color_face_views = Vec::with_capacity(6);
        for layer in 0..6u32 {
            let view = unsafe { device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(color_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: layer, layer_count: 1,
                    }), None)? };
            color_face_views.push(view);
        }

        // Cube view for compute shader sampling.
        let color_cube_view = unsafe { device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(color_image)
                .view_type(vk::ImageViewType::CUBE)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 6,
                }), None)? };

        // ---- Depth image (single layer, reused per face) ----
        let depth_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D { width: size, height: size, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let depth_image = unsafe { device.create_image(&depth_info, None)? };
        let depth_req = unsafe { device.get_image_memory_requirements(depth_image) };
        let depth_mem_type = allocator.find_memory_type(
            depth_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let depth_memory = unsafe { device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(depth_req.size)
                .memory_type_index(depth_mem_type), None)? };
        unsafe { device.bind_image_memory(depth_image, depth_memory, 0)? };

        let depth_view = unsafe { device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(depth_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }), None)? };

        // ---- Framebuffers (one per face) ----
        let mut framebuffers = Vec::with_capacity(6);
        for face in 0..6 {
            let attachments = [color_face_views[face], depth_view];
            let fb = unsafe { device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(probe_capture_pass)
                    .attachments(&attachments)
                    .width(size)
                    .height(size)
                    .layers(1), None)? };
            framebuffers.push(fb);
        }

        // ---- Sampler for SH compute ----
        let sampler = unsafe { device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0.0)
                .max_lod(1.0), None)? };

        // ---- SH compute descriptor set layout ----
        // Binding 0: samplerCube (captured cubemap)
        // Binding 1: SSBO (probe data, read-write)
        let sh_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let sh_compute_layout = unsafe { device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&sh_bindings), None)? };

        // ---- Descriptor pool + set ----
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        ];
        let sh_compute_pool = unsafe { device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(1), None)? };

        let sh_compute_set = unsafe { device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(sh_compute_pool)
                .set_layouts(std::slice::from_ref(&sh_compute_layout)))? [0] };

        // Write descriptors.
        let cubemap_info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(color_cube_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let ssbo_info = vk::DescriptorBufferInfo::default()
            .buffer(probe_ssbo)
            .offset(0)
            .range(probe_ssbo_size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(sh_compute_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&cubemap_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(sh_compute_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&ssbo_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // ---- SH compute pipeline layout (push constants: probe_index + face_size) ----
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SHProjectPushConstants>() as u32);

        let sh_compute_pipeline_layout = unsafe { device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&sh_compute_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range)), None)? };

        // ---- SH compute pipeline ----
        let sh_compute_pipeline = {
            let code = crate::pipeline::align_shader_code(sh_compute_spv);
            let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
            let module = unsafe { device.create_shader_module(&module_info, None)? };
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(sh_compute_pipeline_layout);
            let pipeline = unsafe { device.create_compute_pipelines(
                vk::PipelineCache::null(), std::slice::from_ref(&info), None)
                .map_err(|(_, e)| e)? [0] };
            unsafe { device.destroy_shader_module(module, None) };
            pipeline
        };

        let total_kb = (color_req.size + depth_req.size) / 1024;
        println!(
            "[ProbeBakeTarget] {}×{} HDR cubemap + depth, {} KB VRAM, SH compute ready",
            size, size, total_kb,
        );

        Ok(Self {
            color_image, color_memory, color_face_views, color_cube_view,
            depth_image, depth_memory, depth_view,
            framebuffers, sampler,
            sh_compute_layout, sh_compute_pool, sh_compute_set,
            sh_compute_pipeline_layout, sh_compute_pipeline,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.sh_compute_pipeline, None);
            device.destroy_pipeline_layout(self.sh_compute_pipeline_layout, None);
            device.destroy_descriptor_pool(self.sh_compute_pool, None);
            device.destroy_descriptor_set_layout(self.sh_compute_layout, None);
            device.destroy_sampler(self.sampler, None);
            for &fb in &self.framebuffers { device.destroy_framebuffer(fb, None); }
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_image_view(self.color_cube_view, None);
            for &v in &self.color_face_views { device.destroy_image_view(v, None); }
            device.destroy_image(self.color_image, None);
            device.free_memory(self.color_memory, None);
        }
    }
}

// ====================================================================
//  Vulkan Cube Map Creation Helpers
// ====================================================================

/// Create a VkImage cube map from 6 face images (single mip level).
fn create_cubemap_from_faces(
    device: &Device,
    memory_ctx: &mut MemoryContext,
    faces: &[Vec<u8>],
    size: u32,
    mip_levels: u32,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn std::error::Error>> {
    assert_eq!(faces.len(), 6);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D { width: size, height: size, depth: 1 })
        .mip_levels(mip_levels)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let image = unsafe { device.create_image(&image_info, None)? };
    let mem_req = unsafe { device.get_image_memory_requirements(image) };

    let mem_type = memory_ctx.allocator.find_memory_type(
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    // Upload faces via staging.
    upload_cubemap_faces(device, memory_ctx, image, faces, size, 0, command_pool, queue)?;

    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::CUBE)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 6,
                }),
            None,
        )?
    };

    Ok((image, memory, view))
}

/// Create a VkImage cube map with multiple mip levels.
fn create_cubemap_from_mips(
    device: &Device,
    memory_ctx: &mut MemoryContext,
    mips: &[Vec<Vec<u8>>], // [mip_level][face] -> pixel data
    base_size: u32,
    mip_levels: u32,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn std::error::Error>> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D { width: base_size, height: base_size, depth: 1 })
        .mip_levels(mip_levels)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let image = unsafe { device.create_image(&image_info, None)? };
    let mem_req = unsafe { device.get_image_memory_requirements(image) };

    let mem_type = memory_ctx.allocator.find_memory_type(
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    // Upload each mip level.
    for (mip, faces) in mips.iter().enumerate() {
        let mip_size = (base_size >> mip).max(1);
        upload_cubemap_faces(device, memory_ctx, image, faces, mip_size, mip as u32, command_pool, queue)?;
    }

    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::CUBE)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 6,
                }),
            None,
        )?
    };

    Ok((image, memory, view))
}

/// Upload 6 face images to a cube map at a specific mip level.
fn upload_cubemap_faces(
    device: &Device,
    memory_ctx: &mut MemoryContext,
    image: vk::Image,
    faces: &[Vec<u8>],
    size: u32,
    mip_level: u32,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    let face_bytes = (size * size * 4) as usize;

    // Concatenate all 6 faces into one staging upload.
    let total_bytes = face_bytes * 6;
    let staging_size = memory_ctx.staging_size();
    assert!(total_bytes as u64 <= staging_size,
        "Cube map mip {} ({} bytes) exceeds staging buffer", mip_level, total_bytes);

    let staging_ptr = memory_ctx.staging_ptr();

    unsafe {
        let mut offset = 0usize;
        for face in faces {
            let len = face.len().min(face_bytes);
            std::ptr::copy_nonoverlapping(
                face.as_ptr(),
                staging_ptr.add(offset),
                len,
            );
            offset += face_bytes;
        }

        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = device.allocate_command_buffers(&cmd_info)?[0];
        device.begin_command_buffer(cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

        // Transition to TRANSFER_DST_OPTIMAL.
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip_level,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });
        device.cmd_pipeline_barrier(cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[], &[],
            std::slice::from_ref(&barrier));

        // Copy each face from staging.
        let mut regions = Vec::with_capacity(6);
        for layer in 0..6u32 {
            regions.push(vk::BufferImageCopy::default()
                .buffer_offset((layer as u64) * face_bytes as u64)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level,
                    base_array_layer: layer,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: size, height: size, depth: 1 }));
        }
        device.cmd_copy_buffer_to_image(cmd,
            memory_ctx.staging_buffer(), image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);

        // Transition to SHADER_READ_ONLY_OPTIMAL.
        let barrier2 = vk::ImageMemoryBarrier::default()
            .image(image)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip_level,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });
        device.cmd_pipeline_barrier(cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(), &[], &[],
            std::slice::from_ref(&barrier2));

        device.end_command_buffer(cmd)?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device.queue_submit(queue, std::slice::from_ref(&submit), vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));
    }

    Ok(())
}

// ====================================================================
//  Math Helpers (module-local)
// ====================================================================

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { v }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn float_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}