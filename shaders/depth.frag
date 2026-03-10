// Phase 6: Depth Pre-Pass Fragment Shader + G-Buffer Normal Output
// Compile: glslangValidator -V depth.frag -o compiled/depth.frag.spv
//
// Writes octahedral-encoded view-space normal to color attachment 0
// (R16G16_SFLOAT).  Depth is written automatically by fixed-function.
//
// When used by the sun shadow pipeline (no color attachment), the
// color output is silently discarded by Vulkan — no validation error.

#version 450

layout(location = 0) in vec3 inViewNormal;

layout(location = 0) out vec2 outNormal;

// Octahedral encoding: maps unit sphere → [-1,1]² losslessly.
// Compact, GPU-friendly, no singularities.
// Output remapped to [0,1] for storage in SFLOAT (works for UNORM too).
vec2 octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(
            n.x >= 0.0 ? 1.0 : -1.0,
            n.y >= 0.0 ? 1.0 : -1.0
        );
    }
    return n.xy * 0.5 + 0.5;
}

void main() {
    outNormal = octEncode(normalize(inViewNormal));
}
