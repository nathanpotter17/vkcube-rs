// Phase 4: Shadow Pass Vertex Shader
// Compile: glslangValidator -V shadow.vert -o compiled/shadow.vert.spv
//
// Renders depth from a point light's perspective into a cube map face.
// Phase 4: view/proj sourced from per-frame UBO (set 0, binding 0) which
// is rebound per face with the light's face view/proj matrices.
// Push constants provide the light position and radius.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;     // unused, declared for layout compat
layout(location = 2) in vec4 inTangent;    // unused  [Phase 4: new]
layout(location = 3) in vec2 inUV;         // unused
layout(location = 4) in vec3 inColor;      // unused

layout(location = 0) out vec3 fragWorldPos;

// Per-frame UBO (set 0, binding 0).
// Phase 4: Shadow pass now reads view/proj from here.
// The renderer rebinds set 0 per cube face with a face-specific GlobalUbo.
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

// Push constant: light position + radius.
layout(push_constant) uniform ShadowPush {
    vec3  lightPos;
    float lightRadius;
} shadow;

void main() {
    vec4 worldPos = draw.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;

    gl_Position = frame.proj * frame.view * worldPos;
}
