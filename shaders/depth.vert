// Phase 4: Depth Pre-Pass Vertex Shader
// Compile: glslangValidator -V depth.vert -o compiled/depth.vert.spv

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;     // unused, must be declared
layout(location = 2) in vec4 inTangent;    // unused  [Phase 4: new]
layout(location = 3) in vec2 inUV;         // unused
layout(location = 4) in vec3 inColor;      // unused

// Per-frame UBO (set 0, binding 0) — provides view and proj matrices.
// Phase 4: view/proj sourced from here instead of per-draw UBO.
layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    mat4 sun_light_vp;
    vec4 sun_direction;
} frame;

// Per-draw UBO (Phase 4: model + materialId only, 80 bytes).
layout(set = 3, binding = 0) uniform PerDrawUBO {
    mat4 model;
    uint materialId;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} draw;

invariant gl_Position;

void main() {
    // Must match PBR vertex shader computation order exactly to
    // produce bit-identical depth values (prevents z-fighting).
    vec4 worldPos = draw.model * vec4(inPosition, 1.0);
    gl_Position = frame.proj * frame.view * worldPos;
}
