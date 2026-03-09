// Depth Pre-Pass Vertex Shader
// Compile: glslangValidator -V depth.vert -o compiled/depth.vert.spv

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // unused, must be declared
layout(location = 2) in vec2 inUV;       // unused
layout(location = 3) in vec3 inColor;    // unused

// Per-draw UBO (same as PBR vertex shader).
layout(set = 3, binding = 0) uniform PerDrawUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
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
    gl_Position = draw.proj * draw.view * worldPos;
}
