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
//   - Normal offset bias eliminates peter-panning (shadow detachment)
//   - Slope-scaled depth bias adapts to surface angle
//   - 16-sample rotated Poisson disk PCF for soft shadow edges

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

struct GpuCluster {
    uint offset;
    uint count;
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
//  Sun Shadow Sampling (Phase 8B: improved bias + PCF)
// ====================================================================

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

// ====================================================================
//  Cascaded Sun Shadow Sampling (replaces sampleSunShadow / sampleSunShadowPCF)
// ====================================================================

// Cascade shadow sampling with 3×3 hardware-assisted PCF.
//
// Uses sampler2DArrayShadow for hardware depth comparison.
// LINEAR filter + compareOp gives 2×2 bilinear PCF per tap;
// 3×3 grid = 36 effective samples for smooth penumbra.
//
// N = surface normal (world space).
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

    // Project into cascade's light space
    vec4 shadowPos = csm.cascade_matrices[cascadeIndex] * vec4(worldPos, 1.0);
    vec3 shadowCoord = shadowPos.xyz / shadowPos.w;
    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;

    // Out-of-bounds → lit
    if (shadowCoord.x < 0.0 || shadowCoord.x > 1.0 ||
        shadowCoord.y < 0.0 || shadowCoord.y > 1.0 ||
        shadowCoord.z < 0.0 || shadowCoord.z > 1.0) {
        return 1.0;
    }

    // Slope-scaled bias, increasing with cascade index (farther cascades need more)
    vec3 L = -normalize(csm.csm_light_direction.xyz);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float baseBias = 0.001;
    float maxBias = 0.005;
    float bias = mix(maxBias, baseBias, NdotL);
    bias *= (1.0 + float(cascadeIndex) * 0.5);

    // Derivative-based slope bias for sub-texel accuracy
    float slopeBias = length(dFdx(worldPos)) + length(dFdy(worldPos));
    bias += slopeBias * 0.01;

    float compareRef = shadowCoord.z - bias;

    // 3×3 PCF via hardware comparison sampler
    // sampler2DArrayShadow: texture(sampler, vec4(uv, layer, compareRef)) → [0,1]
    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(cascadeShadowMaps, 0).xy);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            shadow += texture(cascadeShadowMaps,
                vec4(shadowCoord.xy + offset, float(cascadeIndex), compareRef));
        }
    }
    shadow /= 9.0;

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
//  Light evaluation (unchanged from Phase 2)
// ====================================================================

vec3 evaluateLight(
    GpuLight light,
    vec3 N, vec3 V, vec3 albedo,
    float metallic, float roughness, vec3 F0
) {
    uint lightType = light.type_flags & 3u;
    vec3 L;
    float attenuation = 1.0;

    if (lightType == 2u) {
        L = -light.direction_cos_outer.xyz;
    } else {
        vec3 toLight = light.position_radius.xyz - fragWorldPos;
        float dist = length(toLight);
        L = toLight / max(dist, 0.001);
        float radius = light.position_radius.w;
        if (dist > radius) return vec3(0.0);
        float distRatio = dist / radius;
        float falloffFactor = 1.0 - distRatio * distRatio;
        falloffFactor = max(falloffFactor, 0.0);
        falloffFactor = falloffFactor * falloffFactor;
        attenuation = falloffFactor / max(dist * dist, 0.001);
        if (lightType == 1u) {
            float cosAngle = dot(-L, light.direction_cos_outer.xyz);
            float cosOuter = light.direction_cos_outer.w;
            float cosInner = light.cos_inner_angle;
            float spotFactor = clamp(
                (cosAngle - cosOuter) / max(cosInner - cosOuter, 0.001), 0.0, 1.0);
            attenuation *= spotFactor * spotFactor;
        }
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

    // Phase 8C.1: Skip shadow sampling for baked lights (bit 3 = baked).
    float shadow = 1.0;
    bool is_baked = (light.type_flags & 8u) != 0u;

    if (!is_baked) {
        if (light.shadow_index != 0xFFFFFFFFu && lightType != 2u) {
            shadow = samplePointShadow(fragWorldPos, light.position_radius.xyz,
                light.position_radius.w, light.shadow_index);
        }
        // Directional (sun) shadow — handled separately in main() for ambient effect.
        // Here we still apply it to the sun's direct contribution.
        if (lightType == 2u) {
            shadow = calculateCascadeShadow(fragWorldPos, N);
        }
    }

    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;
    return (diffuse + specular) * lightColor * attenuation * NdotL * shadow;
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
    // ════════════════════════════════════════════════════════════════════════
    float sunShadow = calculateCascadeShadow(fragWorldPos, N);

    // Compute ambient shadow factor: in full shadow, reduce ambient to SUN_SHADOW_AMBIENT_MIN.
    // This simulates reduced sky visibility in shadowed areas.
    float ambientShadowFactor = mix(SUN_SHADOW_AMBIENT_MIN, 1.0, sunShadow);

    // ---- Clustered direct lighting (Phase 8C.4: with luminance early exit) ----
    uint clusterIdx = getClusterIndex();
    GpuCluster c = clusterBuf.clusters[clusterIdx];
    vec3 Lo = vec3(0.0);
    for (uint i = 0; i < c.count; i++) {
        uint lightIdx = indexBuf.indices[c.offset + i];
        Lo += evaluateLight(lightBuf.lights[lightIdx], N, V, albedo, metallic, roughness, F0);

        // Phase 8C.4: Luminance-based early exit — if accumulated contribution is
        // saturated AND we've processed at least half the lights, bail.
        // Check every 4th iteration to minimize branch overhead.
        // Threshold 1.5: ACES maps this to ~0.95 (near-white), imperceptible savings.
        if ((i & 3u) == 3u && i > (c.count >> 1u)) {
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

    // Phase 8A: Apply sun shadow to ambient (IBL + probes).
    // This makes shadows clearly visible even with strong ambient lighting.
    vec3 ambient = (diffuseGI + indirectSpecular) * ao * ao_screen * ambientShadowFactor;

    // ---- Phase 4: Emissive texture multiplication ----
    vec3 emissive = mat.emissive.rgb * mat.emissive.a;
    if (mat.emissive_tex > 0u) {
        emissive *= texture(textures[nonuniformEXT(mat.emissive_tex)], fragUV).rgb;
    }

    vec3 color = ambient + Lo + emissive;

    // ---- DEBUG: Uncomment to visualize sun shadow ----
    // outColor = vec4(vec3(sunShadow), 1.0); return;

    // ---- Phase 4: ACES Filmic Tone Mapping (replaces Reinhard) ----
    color = acesFilm(color);
    // Gamma correction.
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, baseColor.a);
}
