// Depth Pre-Pass Fragment Shader
// Compile: glslangValidator -V depth.frag -o compiled/depth.frag.spv
//
// Empty fragment shader for depth-only rendering.
// The pipeline has no color attachments, so no output is needed.
// Depth is written automatically by the fixed-function depth test.

#version 450

void main() {
    // Nothing to do — depth write is automatic.
}
