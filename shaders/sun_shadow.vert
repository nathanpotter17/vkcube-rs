// Phase 8A: Sun Shadow (Directional Light) Vertex Shader
// Compile: glslangValidator -V sun_shadow.vert -o compiled/sun_shadow.vert.spv
//
// Minimal vertex shader for orthographic directional shadow mapping.
// Uses hardware depth (no gl_FragDepth override) - orthographic projection
// naturally produces linear depth which is ideal for shadow comparison.
//
// Phase 8A: Object data read from persistent SSBO via gl_InstanceIndex.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;     // unused, declared for vertex layout compatibility
layout(location = 2) in vec4 inTangent;    // unused
layout(location = 3) in vec2 inUV;         // unused
layout(location = 4) in vec3 inColor;      // unused

// Per-frame UBO (set 0, binding 0).
// For sun shadows, view/proj contain the sun's orthographic matrices.
layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    mat4 sun_light_vp;
    vec4 sun_direction;
} frame;

// Phase 8A: Object SSBO (set 3, binding 0) — persistent storage for all objects.
struct ObjectData {
    mat4  model;        // 64 bytes
    vec4  aabb_min;     // 16 bytes (w unused)
    vec4  aabb_max;     // 16 bytes (w unused)
    uint  first_index;  //  4 bytes
    uint  index_count;  //  4 bytes
    int   vertex_offset;//  4 bytes
    uint  material_id;  //  4 bytes
    uint  buffer_group; //  4 bytes
    uint  flags;        //  4 bytes
    float lod_bias;     //  4 bytes
    uint  _pad;         //  4 bytes
};

layout(set = 3, binding = 0) readonly buffer ObjectSSBO {
    ObjectData objects[];
};

invariant gl_Position;

void main() {
    // Index into object SSBO using gl_InstanceIndex (== firstInstance from indirect cmd)
    ObjectData obj = objects[gl_InstanceIndex];

    vec4 worldPos = obj.model * vec4(inPosition, 1.0);
    gl_Position = frame.proj * frame.view * worldPos;
}
