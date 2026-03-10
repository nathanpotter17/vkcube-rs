// Phase 6: Depth Pre-Pass Vertex Shader + G-Buffer Normal Output
// Compile: glslangValidator -V depth.vert -o compiled/depth.vert.spv
//
// Writes depth (automatic) and view-space normal (color attachment 0).
// The normal is passed to the fragment shader for octahedral encoding.
//
// NOTE: This shader is also used by the sun shadow pipeline (which has
// no color attachment).  The fragment output is simply discarded by
// Vulkan when no color attachment exists — no validation error.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;    // unused here but must match vertex layout
layout(location = 3) in vec2 inUV;         // unused
layout(location = 4) in vec3 inColor;      // unused

// Per-frame UBO (set 0, binding 0) — provides view and proj matrices.
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

layout(location = 0) out vec3 outViewNormal;

invariant gl_Position;

void main() {
    // Must match PBR vertex shader computation order exactly to
    // produce bit-identical depth values (prevents z-fighting).
    vec4 worldPos = draw.model * vec4(inPosition, 1.0);
    gl_Position = frame.proj * frame.view * worldPos;

    // Transform normal to view space for HBAO consumption.
    // The normal matrix is transpose(inverse(mat3(view * model))).
    mat3 normalMat = transpose(inverse(mat3(frame.view * draw.model)));
    outViewNormal = normalize(normalMat * inNormal);
}
