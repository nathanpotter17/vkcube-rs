// Phase 4: Probe Capture Fragment Shader
// Compile: glslangValidator -V probe_capture.frag -o compiled/probe_capture.frag.spv
//
// Renders the scene from a probe's viewpoint into a low-res HDR cubemap.
// Key differences from the main pbr.frag:
//   - Iterates ALL lights in the SSBO (no cluster lookup — cubemap is 32×32)
//   - No GI/IBL/probe sampling (would be circular dependency)
//   - Output is raw linear HDR (no tone mapping, no gamma correction)
//   - This ensures the SH projection captures accurate radiance
//
// Phase 4 changes:
//   - TBN normal mapping (same as pbr.frag)
//   - ORM unpacking, emissive texture
//   - view/proj from per-frame UBO (set 0) — probe eye extracted from frame.view
//   - Per-draw UBO (set 3) reduced to model + materialId only

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

// ---- Structs (identical to pbr.frag) ----

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

struct GpuLight {
    vec4  position_radius;
    vec4  direction_cos_outer;
    vec4  color_intensity;
    uint  type_flags;
    uint  shadow_index;
    float falloff;
    float cos_inner_angle;
};

// ---- Descriptor Set 0: Per-frame globals ----
// Phase 4: Probe passes rebind set 0 per face with the probe's face view/proj.

layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    mat4 sun_light_vp;
    vec4 sun_direction;
} frame;

layout(set = 0, binding = 1, std430) readonly buffer LightSSBO {
    uint     light_count;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
    GpuLight lights[];
} lightBuf;

// Bindings 2-4: cluster data — not used, but must be declared for layout compatibility.
// Binding 5: material SSBO.
layout(set = 0, binding = 5, std430) readonly buffer MaterialSSBO {
    MaterialData materials[];
} materialBuf;

// Bindings 6-10: GI data — not used during probe capture.

// ---- Descriptor Set 1: Bindless textures ----
layout(set = 1, binding = 0) uniform sampler2D textures[];

// ---- Descriptor Set 2: Shadow maps ----
layout(set = 2, binding = 0) uniform samplerCubeArray shadowCubeMaps;
layout(set = 2, binding = 1) uniform sampler2D sunShadowMap;

// ---- Constants ----
const float PI = 3.14159265358979;
const float SHADOW_BIAS = 0.005;
const uint FLAG_DOUBLE_SIDED = 1u;
const uint FLAG_ALPHA_CUTOFF = 4u;

// ====================================================================
//  Extract eye position from view matrix
// ====================================================================

// For a standard look-at view matrix V, the eye position is:
//   eye = -transpose(mat3(V)) * V[3].xyz
vec3 getEyeFromViewMatrix(mat4 V) {
    mat3 rot = mat3(V);
    vec3 t = vec3(V[3]);
    return -transpose(rot) * t;
}

// ====================================================================
//  Phase 4: TBN Normal Mapping (same as pbr.frag)
// ====================================================================

vec3 getShadingNormal(MaterialData mat) {
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent);
    vec3 B = normalize(fragBitangent);

    if (!gl_FrontFacing && (mat.flags & FLAG_DOUBLE_SIDED) != 0u) {
        N = -N; T = -T; B = -B;
    }

    if (mat.normal_tex > 0u) {
        vec3 tangentNormal = texture(textures[nonuniformEXT(mat.normal_tex)], fragUV).rgb;
        tangentNormal = tangentNormal * 2.0 - 1.0;
        tangentNormal.xy *= mat.normal_scale;
        tangentNormal = normalize(tangentNormal);
        mat3 TBN = mat3(T, B, N);
        N = normalize(TBN * tangentNormal);
    }
    return N;
}

// ====================================================================
//  PBR BRDF (same as pbr.frag)
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

// ====================================================================
//  Shadow sampling (same as pbr.frag)
// ====================================================================

float samplePointShadow(vec3 fragPos, vec3 lightPos, float lightRadius, uint shadowIndex) {
    vec3 fragToLight = fragPos - lightPos;
    float currentDist = length(fragToLight) / lightRadius;
    float closestDist = texture(shadowCubeMaps, vec4(fragToLight, float(shadowIndex))).r;
    return currentDist - SHADOW_BIAS > closestDist ? 0.0 : 1.0;
}

float sampleSunShadow(vec3 worldPos) {
    if (frame.sun_direction.w < 0.5) return 1.0;
    vec4 lightClip = frame.sun_light_vp * vec4(worldPos, 1.0);
    vec3 ndc = lightClip.xyz / lightClip.w;
    vec2 shadowUV = ndc.xy * 0.5 + 0.5;
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0)
        return 1.0;
    float currentDepth = ndc.z;
    float closestDepth = texture(sunShadowMap, shadowUV).r;
    return currentDepth - 0.002 > closestDepth ? 0.0 : 1.0;
}

// ====================================================================
//  Light evaluation (same BRDF, same shadows, same attenuation)
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
    if (lightType == 2u) {
        shadow = sampleSunShadow(fragWorldPos);
    }

    vec3 lightColor = light.color_intensity.rgb * light.color_intensity.w;
    return (diffuse + specular) * lightColor * attenuation * NdotL * shadow;
}

// ====================================================================
//  Main — iterate ALL lights, output raw linear HDR
// ====================================================================

void main() {
    MaterialData mat = materialBuf.materials[fragMaterialId];

    vec4 baseColor = mat.base_color * vec4(fragColor, 1.0);
    if (mat.albedo_tex > 0u) {
        baseColor *= texture(textures[nonuniformEXT(mat.albedo_tex)], fragUV);
    }
    if ((mat.flags & FLAG_ALPHA_CUTOFF) != 0u && baseColor.a < mat.alpha_cutoff) discard;

    // Phase 4: TBN normal mapping.
    vec3 N = getShadingNormal(mat);

    // V is computed from the PROBE position (extracted from per-frame view matrix).
    // Phase 4: was draw.view, now frame.view (set 0 is rebound per probe face).
    vec3 probeEye = getEyeFromViewMatrix(frame.view);
    vec3 V = normalize(probeEye - fragWorldPos);

    // Phase 4: ORM unpacking.
    float metallic  = mat.metallic;
    float roughness = mat.roughness;
    float ao        = mat.ao;
    if (mat.metallic_roughness_tex > 0u) {
        vec4 orm = texture(textures[nonuniformEXT(mat.metallic_roughness_tex)], fragUV);
        ao        *= orm.r;
        roughness *= orm.g;
        metallic  *= orm.b;
    }
    if (mat.ao_tex > 0u) {
        ao = texture(textures[nonuniformEXT(mat.ao_tex)], fragUV).r;
    }

    roughness = max(roughness, 0.04);
    vec3  albedo = baseColor.rgb;
    vec3  F0 = mix(vec3(0.04), albedo, metallic);

    // ---- Iterate ALL active lights (no cluster lookup) ----
    vec3 Lo = vec3(0.0);
    uint count = lightBuf.light_count;
    for (uint i = 0; i < count; i++) {
        Lo += evaluateLight(lightBuf.lights[i], N, V, albedo, metallic, roughness, F0);
    }

    // Minimal ambient floor (no GI — we're PRODUCING the GI data).
    vec3 ambient = albedo * vec3(0.015, 0.015, 0.02) * ao;

    // Phase 4: Emissive texture multiplication.
    vec3 emissive = mat.emissive.rgb * mat.emissive.a;
    if (mat.emissive_tex > 0u) {
        emissive *= texture(textures[nonuniformEXT(mat.emissive_tex)], fragUV).rgb;
    }

    // Raw linear HDR — NO tone mapping, NO gamma.
    // The SH projection needs the actual radiance values.
    vec3 color = ambient + Lo + emissive;

    outColor = vec4(color, 1.0);
}
