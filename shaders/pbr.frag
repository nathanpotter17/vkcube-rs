// Phase 3 PBR Fragment Shader — Clustered Shading + Shadows + GI
// Compile: glslangValidator -V pbr.frag -o compiled/basic.frag.spv
//
// Phase 2 features retained:
//   - Clustered shading lookup (set 0, bindings 1-4)
//   - Point / spot / directional light evaluation
//   - Shadow cube map sampling (set 2, binding 0)
//   - Cook-Torrance specular BRDF
//
// Phase 3 additions:
//   - SH probe grid for diffuse GI (set 0, bindings 6-7)
//   - BRDF LUT for split-sum IBL (set 0, binding 8)
//   - Irradiance cube map for diffuse IBL (set 0, binding 9)
//   - Pre-filtered environment map for specular IBL (set 0, binding 10)

#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec3 fragColor;
layout(location = 4) flat in uint fragMaterialId;

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
    uint  type_flags;
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

// ---- Descriptor Set 1: Bindless textures ----

layout(set = 1, binding = 0) uniform sampler2D textures[];

// ---- Descriptor Set 2: Shadow maps ----

layout(set = 2, binding = 0) uniform samplerCubeArray shadowCubeMaps;
layout(set = 2, binding = 1) uniform sampler2D sunShadowMap;

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

// Sample the sun's 2D shadow map (orthographic projection).
float sampleSunShadow(vec3 worldPos) {
    if (frame.sun_direction.w < 0.5) return 1.0; // shadow disabled

    // Transform to sun light-space clip coordinates.
    vec4 lightClip = frame.sun_light_vp * vec4(worldPos, 1.0);
    // Perspective divide (ortho: w=1, but be safe).
    vec3 ndc = lightClip.xyz / lightClip.w;
    // NDC to UV: x,y [-1,1] → [0,1].
    vec2 shadowUV = ndc.xy * 0.5 + 0.5;

    // Out-of-bounds → lit (border sampler returns 1.0 = max depth).
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0)
        return 1.0;

    float currentDepth = ndc.z;
    float closestDepth = texture(sunShadowMap, shadowUV).r;

    // Simple bias to prevent shadow acne.
    float bias = 0.002;
    return currentDepth - bias > closestDepth ? 0.0 : 1.0;
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

    float shadow = 1.0;
    if (light.shadow_index != 0xFFFFFFFFu && lightType != 2u) {
        shadow = samplePointShadow(fragWorldPos, light.position_radius.xyz,
            light.position_radius.w, light.shadow_index);
    }
    // Directional (sun) shadow.
    if (lightType == 2u) {
        shadow = sampleSunShadow(fragWorldPos);
    }

    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;
    return (diffuse + specular) * lightColor * attenuation * NdotL * shadow;
}

// ====================================================================
//  Main
// ====================================================================

void main() {
    MaterialData mat = materialBuf.materials[fragMaterialId];

    vec4 baseColor = mat.base_color * vec4(fragColor, 1.0);
    if (mat.albedo_tex > 0u) {
        baseColor *= texture(textures[nonuniformEXT(mat.albedo_tex)], fragUV);
    }
    if ((mat.flags & 4u) != 0u && baseColor.a < mat.alpha_cutoff) discard;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(frame.camera_pos.xyz - fragWorldPos);
    float metallic  = mat.metallic;
    float roughness = max(mat.roughness, 0.04);
    vec3  albedo    = baseColor.rgb;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NdotV = max(dot(N, V), 0.0);

    // ---- Clustered direct lighting ----
    uint clusterIdx = getClusterIndex();
    GpuCluster c = clusterBuf.clusters[clusterIdx];
    vec3 Lo = vec3(0.0);
    for (uint i = 0; i < c.count; i++) {
        uint lightIdx = indexBuf.indices[c.offset + i];
        Lo += evaluateLight(lightBuf.lights[lightIdx], N, V, albedo, metallic, roughness, F0);
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
    float ao = mat.ao;
    vec3 ambient = (diffuseGI + indirectSpecular) * ao;
    vec3 emissive = mat.emissive.rgb * mat.emissive.a;
    vec3 color = ambient + Lo + emissive;

    // Tone mapping (Reinhard; ACES in Phase 6).
    color = color / (color + vec3(1.0));
    // Gamma correction.
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, baseColor.a);
}
