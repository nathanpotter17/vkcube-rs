// Phase 2: Shadow Pass Vertex Shader
// Compile: glslangValidator -V shadow.vert -o compiled/shadow.vert.spv
//
// Renders depth from a point light's perspective into a cube map face.
// Uses the same per-draw UBO (set 3) with the light's face view/proj.
// Push constants provide the light position and radius.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // unused, declared for layout compat
layout(location = 2) in vec2 inUV;       // unused
layout(location = 3) in vec3 inColor;    // unused

layout(location = 0) out vec3 fragWorldPos;

// Per-draw UBO (same layout as PBR vertex shader).
layout(set = 3, binding = 0) uniform PerDrawUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    uint materialId;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} draw;

// Push constant: light position + radius.
layout(push_constant) uniform ShadowPush {
    vec3  lightPos;
    float lightRadius;
} shadow;

void main() {
    vec4 worldPos = draw.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;

    gl_Position = draw.proj * draw.view * worldPos;
}
