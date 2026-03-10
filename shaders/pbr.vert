// Phase 4 PBR Vertex Shader
// Compile: glslangValidator -V pbr.vert -o compiled/basic.vert.spv
//
// Vertex layout (60 bytes):
//   location 0: vec3 position  (offset 0)
//   location 1: vec3 normal    (offset 12)
//   location 2: vec4 tangent   (offset 24)  [Phase 4: xyz=tangent dir, w=handedness]
//   location 3: vec2 uv        (offset 40)
//   location 4: vec3 color     (offset 48)
//
// Phase 4 changes:
//   - Added tangent input (location 2) for TBN basis
//   - view/proj now sourced from per-frame GlobalUbo (set 0, binding 0)
//   - Per-draw UBO (set 3) reduced to model + materialId only (80 bytes)
//   - Outputs fragTangent and fragBitangent for normal mapping

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;    // Phase 4: xyz=tangent, w=handedness (±1)
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec3 inColor;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec3 fragColor;
layout(location = 4) flat out uint fragMaterialId;
layout(location = 5) out vec3 fragTangent;      // Phase 4
layout(location = 6) out vec3 fragBitangent;    // Phase 4

// Per-frame UBO (set 0, binding 0) — provides view and proj matrices.
layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    mat4 sun_light_vp;
    vec4 sun_direction;
} frame;

// Per-draw UBO (set 3, binding 0, dynamic offset).
// Phase 4: slimmed to model + materialId only (80 bytes).
layout(set = 3, binding = 0) uniform PerDrawUBO {
    mat4 model;
    uint materialId;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} draw;

invariant gl_Position;

void main() {
    vec4 worldPos = draw.model * vec4(inPosition, 1.0);
    gl_Position = frame.proj * frame.view * worldPos;

    fragWorldPos = worldPos.xyz;

    // Transform normal and tangent by upper-left 3×3 of model matrix.
    // For uniform scaling this is correct; non-uniform scaling would
    // need the inverse-transpose.
    mat3 normalMatrix = mat3(draw.model);
    vec3 N = normalize(normalMatrix * inNormal);
    vec3 T = normalize(normalMatrix * inTangent.xyz);

    // Gram-Schmidt re-orthogonalization: ensure T is perpendicular to N.
    T = normalize(T - dot(T, N) * N);

    // Bitangent: cross(N, T) * handedness.  The w component of the
    // tangent vector carries the handedness sign (±1) to handle mirrored UVs.
    vec3 B = cross(N, T) * inTangent.w;

    fragNormal    = N;
    fragTangent   = T;
    fragBitangent = B;
    fragUV        = inUV;
    fragColor     = inColor;
    fragMaterialId = draw.materialId;
}
