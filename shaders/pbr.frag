// Phase 2 PBR Fragment Shader — Clustered Shading + Shadows
// Compile: glslangValidator -V pbr.frag -o compiled/basic.frag.spv
//
// Replaces the Phase 1 hardcoded directional light with:
//   - Clustered shading lookup (set 0, bindings 1-4)
//   - Point / spot / directional light evaluation
//   - Shadow cube map sampling (set 2, binding 0)
//   - Cook-Torrance specular BRDF

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

// ---- Descriptor Set 0: Per-frame globals ----

layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos; // xyz = position, w = time
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

// ---- Descriptor Set 1: Bindless textures ----

layout(set = 1, binding = 0) uniform sampler2D textures[];

// ---- Descriptor Set 2: Shadow maps ----

layout(set = 2, binding = 0) uniform samplerCubeArray shadowCubeMaps;

// ---- Constants ----

const float PI = 3.14159265358979;
const float SHADOW_BIAS = 0.005;
const vec3  AMBIENT_MIN = vec3(0.03, 0.03, 0.04); // minimum ambient while GI is absent

// ====================================================================
//  PBR BRDF functions
// ====================================================================

// GGX/Trowbridge-Reitz normal distribution.
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function.
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method for combined geometry obstruction.
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick approximation.
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ====================================================================
//  Cluster lookup
// ====================================================================

// Determine which cluster this fragment belongs to.
uint getClusterIndex() {
    // Fragment position in view space.
    vec4 viewPos = cluster.view_mat * vec4(fragWorldPos, 1.0);
    float z = -viewPos.z; // positive depth

    float near = cluster.z_params.x;
    float far  = cluster.z_params.y;

    // Logarithmic depth slice.
    float logRatio = log(z / near) / log(far / near);
    uint cz = uint(max(logRatio * float(cluster.grid_size.z), 0.0));
    cz = min(cz, cluster.grid_size.z - 1u);

    // Screen-space tile.
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

    // Sample cube map array: direction = fragToLight, layer = shadowIndex.
    float closestDist = texture(shadowCubeMaps, vec4(fragToLight, float(shadowIndex))).r;

    // PCF-like soft shadow: sample with small bias.
    return currentDist - SHADOW_BIAS > closestDist ? 0.0 : 1.0;
}

// ====================================================================
//  Light evaluation
// ====================================================================

// Evaluate a single light's contribution using Cook-Torrance BRDF.
vec3 evaluateLight(
    GpuLight light,
    vec3 N, vec3 V, vec3 albedo,
    float metallic, float roughness, vec3 F0
) {
    uint lightType = light.type_flags & 3u;

    vec3 L;
    float attenuation = 1.0;

    if (lightType == 2u) {
        // Directional light.
        L = -light.direction_cos_outer.xyz;
    } else {
        // Point or spot.
        vec3 toLight = light.position_radius.xyz - fragWorldPos;
        float dist = length(toLight);
        L = toLight / max(dist, 0.001);

        float radius = light.position_radius.w;
        if (dist > radius) {
            return vec3(0.0);
        }

        // Distance attenuation (smooth falloff).
        float distRatio = dist / radius;
        float falloffFactor = 1.0 - distRatio * distRatio;
        falloffFactor = max(falloffFactor, 0.0);
        falloffFactor = falloffFactor * falloffFactor;
        attenuation = falloffFactor / max(dist * dist, 0.001);

        // Spot cone attenuation.
        if (lightType == 1u) {
            float cosAngle = dot(-L, light.direction_cos_outer.xyz);
            float cosOuter = light.direction_cos_outer.w;
            float cosInner = light.cos_inner_angle;

            float spotFactor = clamp(
                (cosAngle - cosOuter) / max(cosInner - cosOuter, 0.001),
                0.0, 1.0
            );
            attenuation *= spotFactor * spotFactor;
        }
    }

    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return vec3(0.0);
    }

    // Cook-Torrance specular.
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    // Energy conservation.
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // Shadow.
    float shadow = 1.0;
    if (light.shadow_index != 0xFFFFFFFFu && lightType != 2u) {
        shadow = samplePointShadow(
            fragWorldPos,
            light.position_radius.xyz,
            light.position_radius.w,
            light.shadow_index
        );
    }

    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;

    return (diffuse + specular) * lightColor * attenuation * NdotL * shadow;
}

// ====================================================================
//  Main
// ====================================================================

void main() {
    MaterialData mat = materialBuf.materials[fragMaterialId];

    // ---- Base color ----
    vec4 baseColor = mat.base_color * vec4(fragColor, 1.0);
    if (mat.albedo_tex > 0u) {
        baseColor *= texture(textures[nonuniformEXT(mat.albedo_tex)], fragUV);
    }

    // Alpha cutoff.
    if ((mat.flags & 4u) != 0u && baseColor.a < mat.alpha_cutoff) {
        discard;
    }

    // ---- Surface properties ----
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(frame.camera_pos.xyz - fragWorldPos);

    float metallic  = mat.metallic;
    float roughness = max(mat.roughness, 0.04); // avoid zero roughness singularity
    vec3  albedo    = baseColor.rgb;

    // Fresnel F0: dielectrics ≈ 0.04, metals use albedo.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // ---- Cluster lookup ----
    uint clusterIdx = getClusterIndex();
    GpuCluster c = clusterBuf.clusters[clusterIdx];

    // ---- Accumulate light contributions ----
    vec3 Lo = vec3(0.0);

    for (uint i = 0; i < c.count; i++) {
        uint lightIdx = indexBuf.indices[c.offset + i];
        GpuLight light = lightBuf.lights[lightIdx];

        Lo += evaluateLight(light, N, V, albedo, metallic, roughness, F0);
    }

    // ---- Ambient (will be replaced by SH probes in Phase 3) ----
    vec3 ambient = albedo * AMBIENT_MIN * mat.ao;

    // ---- Emissive ----
    vec3 emissive = mat.emissive.rgb * mat.emissive.a;

    // ---- Final color (HDR) ----
    vec3 color = ambient + Lo + emissive;

    // ---- Tone mapping (Reinhard; ACES in Phase 6) ----
    color = color / (color + vec3(1.0));

    // ---- Gamma correction ----
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, baseColor.a);
}
