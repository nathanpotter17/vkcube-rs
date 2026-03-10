// Skybox Vertex Shader — HDR environment background
// Compile: glslc shaders/skybox.vert -o shaders/compiled/skybox.vert.spv --target-env=vulkan1.1 --target-spv=spv1.3
//
// Procedural unit cube (36 verts, no vertex buffer).
// Strips translation from view matrix so the skybox is always centered on camera.
// xyww trick places all fragments at depth=1.0 (behind everything).

#version 450

layout(location = 0) out vec3 localPos;

layout(set = 0, binding = 0) uniform PerFrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    mat4 sun_light_vp;
    vec4 sun_direction;
} frame;

const vec3 cubeVertices[36] = vec3[](
    // Front face (+Z)
    vec3(-1, -1,  1), vec3( 1, -1,  1), vec3( 1,  1,  1),
    vec3( 1,  1,  1), vec3(-1,  1,  1), vec3(-1, -1,  1),
    // Back face (-Z)
    vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
    vec3(-1,  1, -1), vec3( 1,  1, -1), vec3( 1, -1, -1),
    // Top face (+Y)
    vec3(-1,  1,  1), vec3( 1,  1,  1), vec3( 1,  1, -1),
    vec3( 1,  1, -1), vec3(-1,  1, -1), vec3(-1,  1,  1),
    // Bottom face (-Y)
    vec3(-1, -1, -1), vec3( 1, -1, -1), vec3( 1, -1,  1),
    vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, -1, -1),
    // Right face (+X)
    vec3( 1, -1,  1), vec3( 1, -1, -1), vec3( 1,  1, -1),
    vec3( 1,  1, -1), vec3( 1,  1,  1), vec3( 1, -1,  1),
    // Left face (-X)
    vec3(-1, -1, -1), vec3(-1, -1,  1), vec3(-1,  1,  1),
    vec3(-1,  1,  1), vec3(-1,  1, -1), vec3(-1, -1, -1)
);

void main() {
    localPos = cubeVertices[gl_VertexIndex];

    // Strip translation — skybox rotates with camera but never translates.
    mat4 rotView = mat4(mat3(frame.view));
    vec4 clipPos = frame.proj * rotView * vec4(localPos, 1.0);

    // xyww: depth = w/w = 1.0 in NDC → renders behind all geometry.
    gl_Position = clipPos.xyww;
}