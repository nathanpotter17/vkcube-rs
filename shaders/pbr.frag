// Phase 4 PBR Fragment Shader — Clustered Shading + Shadows + GI + TBN Normal Mapping
// Compile: glslangValidator -V pbr.frag -o compiled/basic.frag.spv
//
// Phase 4 additions:
//   - TBN normal mapping via fragTangent/fragBitangent from vertex shader
//   - getShadingNormal() constructs TBN matrix, samples normal_tex when slot > 0
//   - ORM unpacking: metallic_roughness_tex R=occlusion, G=roughness, B=metallic
//   - Emissive texture multiplication when emissive_tex > 0
//   - AO texture sampling when ao_tex > 0
//   - ACES filmic tone mapping (replaces Reinhard from Phase 3)
//
// Phase 8A modification:
//   - Sun shadow now affects ambient lighting (not just direct)
//   - Shadowed areas receive reduced IBL/probe contribution for visible shadows
//
// Phase 8B modification:
//   - Front-face culling in shadow pass eliminates acne without large bias
//   - Hardware slope-only depth bias as sub-texel safety net
//   - Decoupled normal offset: offset UV prevents acne, original Z preserves contact
//   - 3×3 hardware-assisted PCF for soft shadow edges
//
// Phase 9A modification:
//   - ACES tonemapping + gamma correction removed from main()
//   - Fragment shader now outputs raw linear HDR to R16G16B16A16_SFLOAT target
//   - Tonemapping handled by tonemap.comp fullscreen compute pass
//
// Phase 10A modification (Fixes A+C):
//   - evaluateLight() accepts precomputed sun shadow — eliminates redundant
//     calculateCascadeShadow() call that duplicated 9-tap PCF per directional light
//   - Adaptive PCF: cascades 0-1 use 3×3 (9-tap), cascades 2-3 use single tap
//   - Cross-cascade blending at split boundaries eliminates visible seam lines
//   - Smooth blend zone (20% of cascade depth range) transitions both shadow
//     value and PCF quality seamlessly between adjacent cascades
//
// Phase 10A modification (Fix D):
//   - evaluateLight() replaced with 3 specialized branchless functions:
//     evaluateDirectionalLight, evaluateShadowedLocalLight, evaluateUnshadowedLocalLight
//   - GpuCluster.count is bit-packed: [7:0]=total, [15:8]=dir, [23:16]=shadowed_local
//   - cluster_assign.comp sorts lights into 3 contiguous sections
//   - Fragment shader runs 3 sub-loops — zero warp divergence on light type or
//     shadow/no-shadow branching. All threads in a warp execute identical code
//     within each sub-loop.

#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec3 fragColor;
layout(location = 4) flat in uint fragMaterialId;
layout(location = 5) in vec3 fragTangent;      // Phase 4
layout(location = 6) in vec3 fragBitangent;    // Phase 4

layout(location = 0) out vec4 outColor;

// ---- Material data (128 bytes, matches material.rs) ----

struct MaterialData {
    vec4  base_color;
    vec4  emissive;
    float metallic;
    float roughness;
    float ao;
    float normal_scale;
    uint  albedo_tex;
    uint  normal_tex;
    uint  metallic_roughness_tex;
    uint  emissive_tex;
    uint  ao_tex;
    uint  flags;
    float alpha_cutoff;
    float _pad;
    vec4  _reserved0;
    vec4  _reserved1;
    vec4  _reserved2;
};

// ---- Light data (64 bytes, matches light.rs GpuLight) ----

struct GpuLight {
    vec4  position_radius;
    vec4  direction_cos_outer;
    vec4  color_intensity;
    uint  type_flags;           // bits 0-1: type, bit 2: shadow_capable, bit 3: baked (8C.1)
    uint  shadow_index;
    float falloff;
    float cos_inner_angle;
};

// ---- Cluster data ----
// Phase 10A Fix D: GpuCluster.count is bit-packed by cluster_assign.comp:
//   bits  0-7:  total light count for this cluster
//   bits  8-15: directional count (at start of index list)
//   bits 16-23: shadowed local count (after directional section)
// This enables 3 branch-free sub-loops in the fragment shader.

struct GpuCluster {
    uint offset;
    uint count;     // packed: [7:0]=total, [15:8]=dir_count, [23:16]=shadow_local_count
};

// ---- SH Probe data (144 bytes, matches gi.rs GpuSHProbe) ----

struct GpuSHProbe {
    vec4 coeffs[9]; // [i].rgb = SH basis i for (R,G,B), .w = 0
};

// ---- Descriptor Set 0: Per-frame globals ----

layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;    // xyz = position, w = time
    mat4 sun_light_vp;  // sun shadow light-space view-projection
    vec4 sun_direction;  // xyz = direction, w = shadow_enabled
} frame;

layout(set = 0, binding = 1, std430) readonly buffer LightSSBO {
    uint     light_count;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
    GpuLight lights[];
} lightBuf;

layout(set = 0, binding = 2, std430) readonly buffer ClusterSSBO {
    GpuCluster clusters[];
} clusterBuf;

layout(set = 0, binding = 3, std430) readonly buffer LightIndexSSBO {
    uint global_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint indices[];
} indexBuf;

layout(set = 0, binding = 4, std140) uniform ClusterParams {
    mat4  view_mat;
    mat4  proj_mat;
    mat4  inv_proj;
    uvec4 grid_size;
    vec4  z_params;     // near, far, log_ratio, _
    uvec2 screen_size;
    uint  light_count;
    uint  _pad;
} cluster;

layout(set = 0, binding = 5, std430) readonly buffer MaterialSSBO {
    MaterialData materials[];
} materialBuf;

// ---- Phase 3: GI Bindings ----

layout(set = 0, binding = 6, std430) readonly buffer ProbeSSBO {
    uint        probe_count;
    uint        _pad0;
    uint        _pad1;
    uint        _pad2;
    GpuSHProbe  probes[];
} probeBuf;

layout(set = 0, binding = 7, std140) uniform ProbeGridParams {
    vec4  grid_origin;   // xyz = world origin, w = spacing
    uvec4 grid_dims;     // x, z, total_probes, _pad
    vec4  probe_config;  // probe_height, blend_weight, time_of_day, _pad
} probeGrid;

layout(set = 0, binding = 8)  uniform sampler2D   brdfLUT;
layout(set = 0, binding = 9)  uniform samplerCube irradianceMap;
layout(set = 0, binding = 10) uniform samplerCube prefilteredEnvMap;

layout(set = 0, binding = 11) uniform sampler2D aoScreen;

// ---- Descriptor Set 1: Bindless textures ----

layout(set = 1, binding = 0) uniform sampler2D textures[];

// ---- Descriptor Set 2: Shadow maps ----

layout(set = 2, binding = 0) uniform samplerCubeArray shadowCubeMaps;
// CSM: 2D array + comparison sampler for hardware PCF
layout(set = 2, binding = 1) uniform sampler2DArrayShadow cascadeShadowMaps;

// ---- CSM: Cascade shadow data (set 0, binding 12) ----
// IMPORTANT: CASCADE_COUNT must match light.rs CASCADE_COUNT.

const uint CASCADE_COUNT = 4;

layout(set = 0, binding = 12, std140) uniform CascadeShadowUBO {
    mat4  cascade_matrices[CASCADE_COUNT];
    vec4  split_distances;      // .x/.y/.z/.w = far boundary of cascade 0/1/2/3
    vec4  csm_light_direction;  // xyz = direction, w = shadow_enabled
    vec4  shadow_params;        // x = bias, y = strength, z = fade_start, w = fade_range
} csm;

// ---- Constants ----

const float PI = 3.14159265358979;
const float SHADOW_BIAS = 0.005;
const vec3  AMBIENT_MIN = vec3(0.015, 0.015, 0.02); // absolute minimum floor

// SH basis constants (must match gi.rs).
const float SH_Y00  = 0.282095;
const float SH_Y1N1 = 0.488603;
const float SH_Y10  = 0.488603;
const float SH_Y11  = 0.488603;
const float SH_Y2N2 = 1.092548;
const float SH_Y2N1 = 1.092548;
const float SH_Y20  = 0.315392;
const float SH_Y21  = 1.092548;
const float SH_Y22  = 0.546274;

// Maximum number of pre-filtered env map mip levels.
const float MAX_REFLECTION_LOD = 6.0;

// Material flags (must match material.rs).
const uint FLAG_DOUBLE_SIDED = 1u;
const uint FLAG_ALPHA_BLEND  = 2u;
const uint FLAG_ALPHA_CUTOFF = 4u;

// Phase 8A: Sun shadow ambient attenuation.
// In full shadow, ambient is reduced to this fraction (0.0 = pure black, 1.0 = no effect).
// 0.25-0.4 gives realistic outdoor shadow darkness while preserving fill light.
const float SUN_SHADOW_AMBIENT_MIN = 0.3;

// Phase 10A: Cross-cascade blend zone fraction.
// 20% of each cascade's depth range is used for blending into the next cascade.
// Higher = smoother transitions but more fragments pay the double-sample cost.
// Lower = sharper transitions, less double-sampling overhead.
const float CASCADE_BLEND_FRACTION = 0.2;

// ====================================================================
//  ACES Filmic Tone Mapping (Phase 4: replaces Reinhard)
// ====================================================================

// sRGB → ACEScg (approximate AP1 working space).
// Fitted RRT+ODT from Stephen Hill's ACES reference implementation.
vec3 acesFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ====================================================================
//  Phase 4: TBN Normal Mapping
// ====================================================================

/// Construct the shading normal from the TBN basis and normal map.
/// When normal_tex == 0 (no normal map), returns the interpolated
/// geometric normal.  Uses gl_FrontFacing for double-sided materials.
vec3 getShadingNormal(MaterialData mat) {
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    vec3 B = normalize(fragBitangent);

    // Flip normal for back faces (double-sided materials).
    if (!gl_FrontFacing && (mat.flags & FLAG_DOUBLE_SIDED) != 0u) {
        N = -N;
        T = -T;
        B = -B;
    }

    if (mat.normal_tex > 0u) {
        // Sample tangent-space normal from the normal map.
        vec3 tangentNormal = texture(textures[nonuniformEXT(mat.normal_tex)], fragUV).rgb;
        tangentNormal = tangentNormal * 2.0 - 1.0;
        tangentNormal.xy *= mat.normal_scale;
        tangentNormal = normalize(tangentNormal);

        // Transform from tangent space to world space via TBN matrix.
        mat3 TBN = mat3(T, B, N);
        N = normalize(TBN * tangentNormal);
    }

    return N;
}

// ====================================================================
//  PBR BRDF functions
// ====================================================================

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ====================================================================
//  Cluster lookup
// ====================================================================

uint getClusterIndex() {
    vec4 viewPos = cluster.view_mat * vec4(fragWorldPos, 1.0);
    float z = -viewPos.z;
    float near = cluster.z_params.x;
    float far  = cluster.z_params.y;
    float logRatio = log(z / near) / log(far / near);
    uint cz = uint(max(logRatio * float(cluster.grid_size.z), 0.0));
    cz = min(cz, cluster.grid_size.z - 1u);
    uint cx = uint(gl_FragCoord.x / (float(cluster.screen_size.x) / float(cluster.grid_size.x)));
    uint cy = uint(gl_FragCoord.y / (float(cluster.screen_size.y) / float(cluster.grid_size.y)));
    cx = min(cx, cluster.grid_size.x - 1u);
    cy = min(cy, cluster.grid_size.y - 1u);
    return cz * cluster.grid_size.x * cluster.grid_size.y + cy * cluster.grid_size.x + cx;
}

// ====================================================================
//  Shadow sampling
// ====================================================================

float samplePointShadow(vec3 fragPos, vec3 lightPos, float lightRadius, uint shadowIndex) {
    vec3 fragToLight = fragPos - lightPos;
    float currentDist = length(fragToLight) / lightRadius;
    float closestDist = texture(shadowCubeMaps, vec4(fragToLight, float(shadowIndex))).r;
    return currentDist - SHADOW_BIAS > closestDist ? 0.0 : 1.0;
}

// ====================================================================
//  Sun Shadow Sampling (Phase 10A: adaptive PCF + cross-cascade blending)
// ====================================================================
//
// Phase 8B foundation retained:
//   - Front-face culling in shadow pass eliminates acne without large bias
//   - Decoupled normal offset: UV from offset position, Z from original position
//
// Phase 10A changes:
//   - sampleCascadeAtIndex(): samples a single cascade with quality-adaptive PCF
//     Cascades 0-1: 3×3 hardware PCF (9 taps → 36 effective samples)
//     Cascades 2-3: single hardware PCF tap (1 tap → 4 effective bilinear samples)
//   - calculateCascadeShadow(): selects cascade, applies cross-cascade blending
//     at split boundaries to eliminate visible seam lines, applies far fade
//
// Cross-cascade blending:
//   At each cascade boundary, a blend zone covers the last CASCADE_BLEND_FRACTION
//   of the cascade's depth range. Within this zone, both the current and next
//   cascade are sampled and smoothstep-blended. This eliminates the hard seam
//   caused by resolution and PCF quality differences between cascades.

// Poisson disk samples for soft shadow PCF (16 samples, well-distributed).
const vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

// Interleaved gradient noise for sample rotation (stable, no banding).
float interleavedGradientNoise(vec2 screenPos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screenPos, magic.xy)));
}

// Sample a single cascade with quality-adaptive PCF.
//
// cascadeIndex: which cascade to sample (0..CASCADE_COUNT-1)
// worldPos:     fragment world position (used for depth comparison projection)
// N:            surface normal (used for normal offset UV projection)
//
// Returns 0.0 in shadow, 1.0 in light, or -1.0 if out-of-bounds.
// Out-of-bounds sentinel lets the caller decide how to handle it
// (e.g. fall through to next cascade during blending).
float sampleCascadeAtIndex(vec3 worldPos, vec3 N, uint cascadeIndex) {
    // Normal offset: shift position along surface normal for UV lookup.
    // Scales with sin(angle-to-light) — zero when facing the light, max at grazing.
    // Increases with cascade index to match growing texel footprint.
    vec3 L = -normalize(csm.csm_light_direction.xyz);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float sinAngle = sqrt(1.0 - NdotL * NdotL);
    float normalOffsetScale = 0.015 * (1.0 + float(cascadeIndex) * 0.75);
    vec3 offsetPos = worldPos + N * sinAngle * normalOffsetScale;

    mat4 cascadeMat = csm.cascade_matrices[cascadeIndex];

    // Project offset position → UV lookup (prevents acne via texel shift)
    vec4 shadowPosUV = cascadeMat * vec4(offsetPos, 1.0);
    vec3 coordUV = shadowPosUV.xyz / shadowPosUV.w;
    vec2 shadowUV = coordUV.xy * 0.5 + 0.5;

    // Project original position → depth comparison (preserves contact shadows)
    vec4 shadowPosZ = cascadeMat * vec4(worldPos, 1.0);
    float compareRef = (shadowPosZ.xyz / shadowPosZ.w).z;

    // Out-of-bounds → sentinel
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        compareRef < 0.0 || compareRef > 1.0) {
        return -1.0;
    }

    // Phase 10A: Adaptive PCF quality by cascade distance.
    // Cascades 0-1 (near): 3×3 hardware PCF (9 taps × bilinear = 36 effective samples)
    // Cascades 2-3 (far):  single hardware PCF tap (1 tap × bilinear = 4 effective samples)
    //
    // Rationale: far cascades cover large world-space areas per texel. The bilinear
    // filtering from a single comparison-sampler tap already covers ~4 shadow texels,
    // providing adequate softness. Near cascades need the full 3×3 kernel for
    // smooth penumbra edges where texel density is high enough to reveal aliasing.
    float shadow = 0.0;
    if (cascadeIndex <= 1u) {
        // Near cascades: full 3×3 PCF
        vec2 texelSize = 1.0 / vec2(textureSize(cascadeShadowMaps, 0).xy);
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                vec2 offset = vec2(x, y) * texelSize;
                shadow += texture(cascadeShadowMaps,
                    vec4(shadowUV + offset, float(cascadeIndex), compareRef));
            }
        }
        shadow /= 9.0;
    } else {
        // Far cascades: single hardware PCF tap (bilinear gives 2×2 = 4 samples)
        shadow = texture(cascadeShadowMaps,
            vec4(shadowUV, float(cascadeIndex), compareRef));
    }

    return shadow;
}

// Main cascade shadow entry point with cross-cascade blending.
//
// Eliminates visible seam lines at cascade boundaries by sampling both the
// current and next cascade within a blend zone and smoothstep-interpolating.
//
// Returns 0.0 in shadow, 1.0 in light.
float calculateCascadeShadow(vec3 worldPos, vec3 N) {
    if (csm.csm_light_direction.w < 0.5) return 1.0; // shadow disabled

    // Determine cascade from view-space depth (same view matrix as cluster assignment)
    float viewZ = -(cluster.view_mat * vec4(worldPos, 1.0)).z;

    uint cascadeIndex = CASCADE_COUNT - 1u;
    for (uint i = 0u; i < CASCADE_COUNT - 1u; i++) {
        if (viewZ < csm.split_distances[i]) {
            cascadeIndex = i;
            break;
        }
    }

    // Sample the primary cascade.
    float shadow = sampleCascadeAtIndex(worldPos, N, cascadeIndex);

    // Handle OOB from primary cascade — fall through to lit.
    if (shadow < 0.0) return 1.0;

    // ---- Cross-cascade blending at split boundaries ----
    // When the fragment is near the far edge of a non-final cascade,
    // blend with the next cascade to eliminate the visible seam line.
    //
    // The blend zone covers the last CASCADE_BLEND_FRACTION of the current
    // cascade's depth range. Within this zone, both cascades are sampled
    // and smoothstep-blended. This smooths both resolution differences
    // and PCF quality transitions (9-tap → 1-tap at the cascade 1→2 boundary).
    if (cascadeIndex < CASCADE_COUNT - 1u) {
        float cascadeFar  = csm.split_distances[cascadeIndex];
        float cascadeNear = (cascadeIndex > 0u)
            ? csm.split_distances[cascadeIndex - 1u]
            : cluster.z_params.x;  // camera near plane

        float cascadeRange = cascadeFar - cascadeNear;
        float blendStart   = cascadeFar - cascadeRange * CASCADE_BLEND_FRACTION;

        if (viewZ > blendStart) {
            // We're in the blend zone — sample the next cascade too.
            float nextShadow = sampleCascadeAtIndex(worldPos, N, cascadeIndex + 1u);

            // If next cascade returns OOB, just use the current cascade's value.
            if (nextShadow >= 0.0) {
                // smoothstep blend: 0 at blendStart, 1 at cascadeFar
                float blendFactor = smoothstep(blendStart, cascadeFar, viewZ);
                shadow = mix(shadow, nextShadow, blendFactor);
            }
        }
    }

    // Fade out at the last cascade's far boundary.
    // Beyond the last cascade, shadow gracefully fades to 1.0 (lit) instead
    // of abruptly cutting off.
    if (cascadeIndex == CASCADE_COUNT - 1u) {
        float maxZ = csm.split_distances[CASCADE_COUNT - 1u];
        float fadeStart = maxZ * csm.shadow_params.z;
        float fadeRange = maxZ * csm.shadow_params.w;
        float fadeFactor = 1.0 - smoothstep(fadeStart, fadeStart + fadeRange, viewZ);
        shadow = mix(1.0, shadow, fadeFactor);
    }

    // Strength modulation
    shadow = mix(1.0, shadow, csm.shadow_params.y);

    return shadow;
}

// ====================================================================
//  SH Probe Evaluation (Phase 3)
// ====================================================================

// Evaluate L2 SH irradiance from a single probe.
vec3 evaluateSH(GpuSHProbe probe, vec3 N) {
    vec3 irr = probe.coeffs[0].rgb * SH_Y00;

    irr += probe.coeffs[1].rgb * (SH_Y1N1 * N.y);
    irr += probe.coeffs[2].rgb * (SH_Y10  * N.z);
    irr += probe.coeffs[3].rgb * (SH_Y11  * N.x);

    irr += probe.coeffs[4].rgb * (SH_Y2N2 * N.x * N.y);
    irr += probe.coeffs[5].rgb * (SH_Y2N1 * N.y * N.z);
    irr += probe.coeffs[6].rgb * (SH_Y20  * (3.0 * N.z * N.z - 1.0));
    irr += probe.coeffs[7].rgb * (SH_Y21  * N.x * N.z);
    irr += probe.coeffs[8].rgb * (SH_Y22  * (N.x * N.x - N.y * N.y));

    return max(irr, vec3(0.0));
}

// Sample the SH probe grid with bilinear interpolation in XZ.
vec3 sampleProbeGrid(vec3 worldPos, vec3 N) {
    if (probeBuf.probe_count == 0u || probeGrid.grid_dims.x == 0u) {
        return AMBIENT_MIN;
    }

    float spacing = probeGrid.grid_origin.w;
    vec3  origin  = probeGrid.grid_origin.xyz;
    uint  dimX    = probeGrid.grid_dims.x;
    uint  dimZ    = probeGrid.grid_dims.y;

    float gx = (worldPos.x - origin.x - spacing * 0.5) / spacing;
    float gz = (worldPos.z - origin.z - spacing * 0.5) / spacing;

    float fx = fract(gx);
    float fz = fract(gz);
    int ix0 = int(floor(gx));
    int iz0 = int(floor(gz));
    int ix1 = ix0 + 1;
    int iz1 = iz0 + 1;

    ix0 = clamp(ix0, 0, int(dimX) - 1);
    ix1 = clamp(ix1, 0, int(dimX) - 1);
    iz0 = clamp(iz0, 0, int(dimZ) - 1);
    iz1 = clamp(iz1, 0, int(dimZ) - 1);

    uint i00 = uint(iz0) * dimX + uint(ix0);
    uint i10 = uint(iz0) * dimX + uint(ix1);
    uint i01 = uint(iz1) * dimX + uint(ix0);
    uint i11 = uint(iz1) * dimX + uint(ix1);

    uint maxIdx = probeBuf.probe_count - 1u;
    i00 = min(i00, maxIdx);
    i10 = min(i10, maxIdx);
    i01 = min(i01, maxIdx);
    i11 = min(i11, maxIdx);

    vec3 v00 = evaluateSH(probeBuf.probes[i00], N);
    vec3 v10 = evaluateSH(probeBuf.probes[i10], N);
    vec3 v01 = evaluateSH(probeBuf.probes[i01], N);
    vec3 v11 = evaluateSH(probeBuf.probes[i11], N);

    vec3 top    = mix(v00, v10, fx);
    vec3 bottom = mix(v01, v11, fx);
    return max(mix(top, bottom, fz), AMBIENT_MIN);
}

// ====================================================================
//  IBL (Image-Based Lighting — Phase 3)
// ====================================================================

vec3 sampleSpecularIBL(vec3 R, float roughness, vec3 F, vec2 brdf) {
    float lod = roughness * MAX_REFLECTION_LOD;
    vec3 prefilteredColor = textureLod(prefilteredEnvMap, R, lod).rgb;
    return prefilteredColor * (F * brdf.x + brdf.y);
}

// ====================================================================
//  Light evaluation — Phase 10A Fix D: 3 specialized branchless functions
// ====================================================================
//
// Previously a single evaluateLight() function handled all light types
// with runtime branches on lightType, shadow_index, and baked status.
// When a GPU warp contains fragments lit by mixed light types, ALL threads
// must execute ALL branches — the most expensive being the samplePointShadow()
// cube map fetch (~100+ cycles) that only shadowed-light threads actually need.
//
// Fix D eliminates this by splitting into 3 specialized functions with
// zero conditional branching on light type or shadow state. The cluster
// assignment shader sorts lights into 3 contiguous sections (directional,
// shadowed local, unshadowed local), and the fragment shader runs a
// separate sub-loop for each section — every thread in a warp executes
// identical code within each sub-loop.

// ---- Sub-loop 1: Directional lights (sun, moon) ----
// No distance/attenuation calc. Uses precomputed cascade shadow.
// Warp-uniform: all threads in a warp take the exact same path.
vec3 evaluateDirectionalLight(
    GpuLight light,
    vec3 N, vec3 V, vec3 albedo,
    float metallic, float roughness, vec3 F0,
    float precomputedSunShadow
) {
    vec3 L = -light.direction_cos_outer.xyz;

    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (D * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;
    return (diffuse + specular) * lightColor * NdotL * precomputedSunShadow;
}

// ---- Sub-loop 2: Shadowed local lights (point/spot with active shadow map) ----
// Always samples point shadow — no branch. All threads in the warp execute
// the cube map texture fetch simultaneously, maximizing texture unit utilization.
vec3 evaluateShadowedLocalLight(
    GpuLight light,
    vec3 N, vec3 V, vec3 albedo,
    float metallic, float roughness, vec3 F0
) {
    vec3 toLight = light.position_radius.xyz - fragWorldPos;
    float dist = length(toLight);
    vec3 L = toLight / max(dist, 0.001);
    float radius = light.position_radius.w;
    if (dist > radius) return vec3(0.0);

    float distRatio = dist / radius;
    float falloffFactor = 1.0 - distRatio * distRatio;
    falloffFactor = max(falloffFactor, 0.0);
    falloffFactor = falloffFactor * falloffFactor;
    float attenuation = falloffFactor / max(dist * dist, 0.001);

    // Spot light cone (point lights have cos_outer=-1 → spotFactor always 1.0)
    uint lightType = light.type_flags & 3u;
    if (lightType == 1u) {
        float cosAngle = dot(-L, light.direction_cos_outer.xyz);
        float cosOuter = light.direction_cos_outer.w;
        float cosInner = light.cos_inner_angle;
        float spotFactor = clamp(
            (cosAngle - cosOuter) / max(cosInner - cosOuter, 0.001), 0.0, 1.0);
        attenuation *= spotFactor * spotFactor;
    }

    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (D * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // Unconditional shadow sample — every thread in the warp does this.
    // No branch on shadow_index (cluster_assign.comp guarantees it's valid).
    float shadow = samplePointShadow(fragWorldPos, light.position_radius.xyz,
        light.position_radius.w, light.shadow_index);

    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;
    return (diffuse + specular) * lightColor * attenuation * NdotL * shadow;
}

// ---- Sub-loop 3: Unshadowed local lights (no shadow slot, or baked) ----
// No shadow sampling at all — no cube map fetch, no branch.
// Covers: point/spot without shadow slots, baked lights.
vec3 evaluateUnshadowedLocalLight(
    GpuLight light,
    vec3 N, vec3 V, vec3 albedo,
    float metallic, float roughness, vec3 F0
) {
    vec3 toLight = light.position_radius.xyz - fragWorldPos;
    float dist = length(toLight);
    vec3 L = toLight / max(dist, 0.001);
    float radius = light.position_radius.w;
    if (dist > radius) return vec3(0.0);

    float distRatio = dist / radius;
    float falloffFactor = 1.0 - distRatio * distRatio;
    falloffFactor = max(falloffFactor, 0.0);
    falloffFactor = falloffFactor * falloffFactor;
    float attenuation = falloffFactor / max(dist * dist, 0.001);

    // Spot light cone
    uint lightType = light.type_flags & 3u;
    if (lightType == 1u) {
        float cosAngle = dot(-L, light.direction_cos_outer.xyz);
        float cosOuter = light.direction_cos_outer.w;
        float cosInner = light.cos_inner_angle;
        float spotFactor = clamp(
            (cosAngle - cosOuter) / max(cosInner - cosOuter, 0.001), 0.0, 1.0);
        attenuation *= spotFactor * spotFactor;
    }

    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (D * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // No shadow sampling — shadow = 1.0 implicit.
    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;
    return (diffuse + specular) * lightColor * attenuation * NdotL;
}

// ====================================================================
//  Main
// ====================================================================

void main() {
    MaterialData mat = materialBuf.materials[fragMaterialId];

    // ---- Base color (albedo texture × vertex color × material factor) ----
    vec4 baseColor = mat.base_color * vec4(fragColor, 1.0);
    if (mat.albedo_tex > 0u) {
        baseColor *= texture(textures[nonuniformEXT(mat.albedo_tex)], fragUV);
    }
    if ((mat.flags & FLAG_ALPHA_CUTOFF) != 0u && baseColor.a < mat.alpha_cutoff) discard;

    // ---- Phase 4: Shading normal via TBN ----
    vec3 N = getShadingNormal(mat);
    vec3 V = normalize(frame.camera_pos.xyz - fragWorldPos);

    // ---- Phase 4: ORM unpacking ----
    // When metallic_roughness_tex is bound, sample it.
    // glTF convention: R = occlusion, G = roughness, B = metallic.
    float metallic  = mat.metallic;
    float roughness = mat.roughness;
    float ao        = mat.ao;

    if (mat.metallic_roughness_tex > 0u) {
        vec4 orm = texture(textures[nonuniformEXT(mat.metallic_roughness_tex)], fragUV);
        ao        *= orm.r;    // Occlusion channel
        roughness *= orm.g;    // Roughness channel
        metallic  *= orm.b;    // Metallic channel
    }

    // Phase 4: Separate AO texture (when present, overrides ORM occlusion).
    if (mat.ao_tex > 0u) {
        ao = texture(textures[nonuniformEXT(mat.ao_tex)], fragUV).r;
    }

    roughness = max(roughness, 0.04);
    vec3  albedo = baseColor.rgb;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NdotV = max(dot(N, V), 0.0);

    // ════════════════════════════════════════════════════════════════════════
    // CSM: Sample cascade shadow ONCE, use for both direct and ambient
    // Phase 10A: This is now the ONLY cascade shadow computation per fragment.
    // evaluateLight() receives this value instead of recomputing it.
    // ════════════════════════════════════════════════════════════════════════
    float sunShadow = calculateCascadeShadow(fragWorldPos, N);

    // Gate ambient shadow by sun-facing angle.
    // Back-facing surfaces (NdotL ≈ 0) produce unreliable shadow map lookups,
    // so we smoothly blend out the ambient darkening as surfaces turn away.
    // Direct lighting handles itself (NdotL=0 → contribution=0, shadow irrelevant).
    vec3 sunL = -normalize(csm.csm_light_direction.xyz);
    float sunFacing = smoothstep(0.0, 0.1, max(dot(N, sunL), 0.0));
    float gatedShadow = mix(1.0, sunShadow, sunFacing);
    float ambientShadowFactor = mix(SUN_SHADOW_AMBIENT_MIN, 1.0, gatedShadow);

    // ---- Clustered direct lighting — Phase 10A Fix D: 3 branch-free sub-loops ----
    // cluster_assign.comp sorts lights into 3 contiguous sections and packs
    // the counts into GpuCluster.count. We unpack and run a specialized
    // evaluation function for each section — zero type/shadow branching.
    uint clusterIdx = getClusterIndex();
    GpuCluster c = clusterBuf.clusters[clusterIdx];

    // Unpack the 3-way count from the packed uint.
    uint totalCount       = c.count & 0xFFu;
    uint dirCount         = (c.count >> 8u) & 0xFFu;
    uint shadowLocalCount = (c.count >> 16u) & 0xFFu;

    // Index ranges within the cluster's index list:
    //   [0 .. dirCount):                                directional lights
    //   [dirCount .. dirCount + shadowLocalCount):      shadowed point/spot
    //   [dirCount + shadowLocalCount .. totalCount):    unshadowed/baked point/spot
    uint shadowStart    = dirCount;
    uint unshadowStart  = dirCount + shadowLocalCount;

    vec3 Lo = vec3(0.0);

    // ---- Sub-loop 1: Directional lights (warp-uniform, no distance calc) ----
    for (uint i = 0; i < dirCount; i++) {
        uint lightIdx = indexBuf.indices[c.offset + i];
        Lo += evaluateDirectionalLight(lightBuf.lights[lightIdx], N, V, albedo,
                                        metallic, roughness, F0, sunShadow);
    }

    // ---- Sub-loop 2: Shadowed local lights (all threads sample cube map) ----
    for (uint i = shadowStart; i < unshadowStart; i++) {
        uint lightIdx = indexBuf.indices[c.offset + i];
        Lo += evaluateShadowedLocalLight(lightBuf.lights[lightIdx], N, V, albedo,
                                          metallic, roughness, F0);

        // Phase 8C.4: Luminance early exit — check every 4th shadowed light.
        // Shadowed lights are the most expensive, so early exit here saves the most.
        // Only check after processing at least 4 shadowed lights to avoid premature bail.
        uint processed = i - shadowStart;
        if ((processed & 3u) == 3u && processed >= 4u) {
            float lum = dot(Lo, vec3(0.2126, 0.7152, 0.0722));
            if (lum > 1.5) {
                // Skip remaining shadowed AND all unshadowed lights.
                unshadowStart = totalCount;
                break;
            }
        }
    }

    // ---- Sub-loop 3: Unshadowed local lights (no shadow fetch at all) ----
    for (uint i = unshadowStart; i < totalCount; i++) {
        uint lightIdx = indexBuf.indices[c.offset + i];
        Lo += evaluateUnshadowedLocalLight(lightBuf.lights[lightIdx], N, V, albedo,
                                            metallic, roughness, F0);

        // Luminance early exit — unshadowed lights are cheap, check less often.
        if (((i - unshadowStart) & 7u) == 7u) {
            float lum = dot(Lo, vec3(0.2126, 0.7152, 0.0722));
            if (lum > 1.5) break;
        }
    }

    // ---- Phase 3: Indirect diffuse (SH probes + irradiance map) ----
    vec3 F_env = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD_env = (vec3(1.0) - F_env) * (1.0 - metallic);

    vec3 probeIrradiance = sampleProbeGrid(fragWorldPos, N);
    vec3 cubeIrradiance  = texture(irradianceMap, N).rgb;
    float probeWeight = probeGrid.probe_config.y;
    vec3 diffuseGI = mix(cubeIrradiance * albedo * kD_env,
                         kD_env * albedo * probeIrradiance,
                         probeWeight);

    // ---- Phase 3: Indirect specular (IBL) ----
    vec3 R = reflect(-V, N);
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 indirectSpecular = sampleSpecularIBL(R, roughness, F_env, brdf);

    // ---- Compose ----
    // Screen-space AO modulates the ambient term.
    vec2 screenUV = gl_FragCoord.xy / vec2(cluster.screen_size);
    float ao_screen = texture(aoScreen, screenUV).r;

    // Multi-bounce AO approximation: apply partial occlusion to direct lighting.
    // Full AO on ambient, sqrt-attenuated on direct. Prevents AO from being
    // invisible when direct lighting dominates.
    float ao_direct = mix(1.0, ao_screen, 0.35);  // 35% direct influence

    vec3 ambient = (diffuseGI + indirectSpecular) * ao * ao_screen * ambientShadowFactor;

    vec3 emissive = mat.emissive.rgb * mat.emissive.a;
    if (mat.emissive_tex > 0u) {
        emissive *= texture(textures[nonuniformEXT(mat.emissive_tex)], fragUV).rgb;
    }

    vec3 color = ambient + Lo * ao_direct + emissive;

    // ---- DEBUG: Uncomment to visualize sun shadow ----
    // outColor = vec4(vec3(sunShadow), 1.0); return;

    // ---- DEBUG: Uncomment to visualize cascade index (R=0, G=1, B=2, W=3) ----
    // float vz = -(cluster.view_mat * vec4(fragWorldPos, 1.0)).z;
    // uint ci = 3u; for (uint ii=0u;ii<3u;ii++){if(vz<csm.split_distances[ii]){ci=ii;break;}}
    // vec3 cc = vec3(ci==0u?1.0:0.0, ci==1u?1.0:0.0, ci==2u?1.0:(ci==3u?1.0:0.0));
    // outColor = vec4(cc, 1.0); return;

    // Phase 9A: Output raw linear HDR to R16G16B16A16_SFLOAT target.
    // Tonemapping + gamma correction moved to tonemap.comp compute pass.
    outColor = vec4(color, baseColor.a);
}
