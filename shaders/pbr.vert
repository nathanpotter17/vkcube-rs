// Phase 1 PBR Vertex Shader
// Compile: glslangValidator -V pbr.vert -o compiled/basic.vert.spv
//
// Vertex layout (44 bytes):
//   location 0: vec3 position  (offset 0)
//   location 1: vec3 normal    (offset 12)
//   location 2: vec2 uv        (offset 24)
//   location 3: vec3 color     (offset 32)
//
// Descriptor Set 3, Binding 0: Per-draw dynamic UBO.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec3 fragColor;
layout(location = 4) flat out uint fragMaterialId;

// Per-draw UBO (set 3, binding 0, dynamic offset).
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
    vec4 worldPos = draw.model * vec4(inPosition, 1.0);
    gl_Position = draw.proj * draw.view * worldPos;

    fragWorldPos = worldPos.xyz;
    // Transform normal by the upper-left 3x3 of the model matrix.
    // For uniform scaling this is correct; non-uniform scaling would
    // need the inverse-transpose.
    fragNormal = mat3(draw.model) * inNormal;
    fragUV = inUV;
    fragColor = inColor;
    fragMaterialId = draw.materialId;
}
