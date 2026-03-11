// Phase 8A: Shadow Pass Vertex Shader with GPU-Driven Object SSBO
// Compile: glslangValidator -V shadow.vert -o compiled/shadow.vert.spv
//
// Renders depth from a point light's perspective into a cube map face.
// Phase 8A: Object data read from persistent SSBO via gl_InstanceIndex.
// Push constants provide the light position and radius.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;     // unused, declared for layout compat
layout(location = 2) in vec4 inTangent;    // unused
layout(location = 3) in vec2 inUV;         // unused
layout(location = 4) in vec3 inColor;      // unused

layout(location = 0) out vec3 fragWorldPos;

// Per-frame UBO (set 0, binding 0).
// Shadow pass reads view/proj from here.
// The renderer rebinds set 0 per cube face with a face-specific GlobalUbo.
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

// Push constant: light position + radius.
layout(push_constant) uniform ShadowPush {
    vec3  lightPos;
    float lightRadius;
} shadow;

void main() {
    // Index into object SSBO using gl_InstanceIndex
    ObjectData obj = objects[gl_InstanceIndex];

    vec4 worldPos = obj.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;

    gl_Position = frame.proj * frame.view * worldPos;
}
