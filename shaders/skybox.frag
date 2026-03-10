// Skybox Fragment Shader — HDR environment background
// Compile: glslc shaders/skybox.frag -o shaders/compiled/skybox.frag.spv --target-env=vulkan1.1 --target-spv=spv1.3
//
// Samples the pre-filtered environment cubemap at LOD 0 (roughness=0, sharpest mip).
// For a 256×256 base with a 1024-wide equirect source this is effectively the
// original HDR data.  ACES tone map + gamma matches pbr.frag output.

#version 450

layout(location = 0) in vec3 localPos;
layout(location = 0) out vec4 fragColor;

// Reuses the pre-filtered env map already bound at set 0 binding 10.
// LOD 0 = roughness 0.0 = sharpest mip ≈ source cubemap.
layout(set = 0, binding = 10) uniform samplerCube prefilteredEnvMap;

// ACES filmic tone mapping (matches pbr.frag).
vec3 acesFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 dir = normalize(localPos);
    vec3 envColor = textureLod(prefilteredEnvMap, dir, 0.0).rgb;

    // Exposure-adjusted tone map.
    const float exposure = 1.0;
    vec3 mapped = acesFilm(envColor * exposure);

    // Gamma correction.
    mapped = pow(mapped, vec3(1.0 / 2.2));

    fragColor = vec4(mapped, 1.0);
}