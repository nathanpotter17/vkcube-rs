// Phase 8A: Depth Pre-Pass Vertex Shader with GPU-Driven Object SSBO
// Compile: glslangValidator -V depth.vert -o compiled/depth.vert.spv
//
// Writes depth (automatic) and view-space normal (color attachment 0).
// The normal is passed to the fragment shader for octahedral encoding.
//
// NOTE: This shader is also used by the sun shadow pipeline (which has
// no color attachment).  The fragment output is simply discarded by
// Vulkan when no color attachment exists — no validation error.
//
// Phase 8A: Object data read from persistent SSBO via gl_InstanceIndex.

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

layout(location = 0) out vec3 outViewNormal;

invariant gl_Position;

void main() {
    // Index into object SSBO using gl_InstanceIndex
    ObjectData obj = objects[gl_InstanceIndex];

    // Must match PBR vertex shader computation order exactly to
    // produce bit-identical depth values (prevents z-fighting).
    vec4 worldPos = obj.model * vec4(inPosition, 1.0);
    gl_Position = frame.proj * frame.view * worldPos;

    // Full cofactor normal matrix — equivalent to transpose(inverse(M))
    // but avoids the matrix inversion. 9 cross products vs 1 inverse.
    mat3 M = mat3(frame.view * obj.model);
    mat3 cofactor = mat3(
        cross(M[1], M[2]),
        cross(M[2], M[0]),
        cross(M[0], M[1])
    );
    outViewNormal = normalize(cofactor * inNormal);
}
