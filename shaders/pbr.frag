// Phase 1 PBR Fragment Shader
// Compile: glslangValidator -V pbr.frag -o compiled/basic.frag.spv
//
// Reads material data from the SSBO (set 0, binding 5).
// Supports bindless texture sampling (set 1, binding 0).
// Phase 1 uses a single hardcoded directional light.
// Clustered shading and shadow maps are Phase 2.

#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec3 fragColor;
layout(location = 4) flat in uint fragMaterialId;

layout(location = 0) out vec4 outColor;

// Material data struct (128 bytes, matches material.rs).
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

// Set 0: Per-frame globals.
layout(set = 0, binding = 5) readonly buffer MaterialSSBO {
    MaterialData materials[];
} materialBuf;

// Set 1: Bindless texture array.
layout(set = 1, binding = 0) uniform sampler2D textures[];

// Hardcoded directional light for Phase 1.
const vec3 LIGHT_DIR   = normalize(vec3(0.4, 0.8, 0.3));
const vec3 LIGHT_COLOR = vec3(1.0, 0.95, 0.9);
const vec3 AMBIENT     = vec3(0.08, 0.08, 0.10);

void main() {
    MaterialData mat = materialBuf.materials[fragMaterialId];

    // Base color: material × vertex color × albedo texture.
    vec4 baseColor = mat.base_color * vec4(fragColor, 1.0);
    if (mat.albedo_tex > 0) {
        baseColor *= texture(textures[nonuniformEXT(mat.albedo_tex)], fragUV);
    }

    // Alpha cutoff test.
    if ((mat.flags & 4u) != 0u && baseColor.a < mat.alpha_cutoff) {
        discard;
    }

    vec3 N = normalize(fragNormal);

    // Simple Lambertian diffuse.
    float NdotL = max(dot(N, LIGHT_DIR), 0.0);
    vec3 diffuse = baseColor.rgb * LIGHT_COLOR * NdotL;

    // Ambient term (will be replaced by SH probes in Phase 3).
    vec3 ambient = baseColor.rgb * AMBIENT * mat.ao;

    // Emissive.
    vec3 emissive = mat.emissive.rgb * mat.emissive.a;

    vec3 color = ambient + diffuse + emissive;

    // Simple Reinhard tone mapping (ACES in Phase 6).
    color = color / (color + vec3(1.0));

    // Gamma correction.
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, baseColor.a);
}
