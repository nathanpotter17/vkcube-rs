// Phase 8A: Sun Shadow (Directional Light) Fragment Shader
// Compile: glslangValidator -V sun_shadow.frag -o compiled/sun_shadow.frag.spv
//
// Empty fragment shader for depth-only directional shadow mapping.
// Hardware automatically writes depth from the rasterizer - no gl_FragDepth needed.
//
// Unlike point light shadows (shadow.frag) which write linear distance,
// directional shadows use standard orthographic depth which is already linear.

#version 450

void main() {
    // Depth is written automatically by the rasterizer.
    // No color output - shadow pass has no color attachment.
}
