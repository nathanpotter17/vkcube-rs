// Phase 2: Shadow Pass Fragment Shader
// Compile: glslangValidator -V shadow.frag -o compiled/shadow.frag.spv
//
// Writes normalized linear distance (fragWorldPos → lightPos) into
// gl_FragDepth for point light cube map shadows.
//
// This replaces the hardware perspective depth with a linear metric
// so the PBR shader can sample and compare consistently across all
// 6 cube faces using a single direction vector.
//
// Value range: [0, 1] where 0 = at light, 1 = at light radius.

#version 450

layout(location = 0) in vec3 fragWorldPos;

// Push constant: light position + radius (same as vertex shader).
layout(push_constant) uniform ShadowPush {
    vec3  lightPos;
    float lightRadius;
} shadow;

void main() {
    float dist = length(fragWorldPos - shadow.lightPos);
    // Apply bias at write time (hardware depth_bias is ineffective because
    // gl_FragDepth overwrites the rasterizer depth).  Push depth slightly
    // further from the light to prevent self-shadowing.
    gl_FragDepth = dist / shadow.lightRadius + 0.002;
}
