#version 450

// Phase 7 — Debug overlay vertex shader.
// Converts pixel-space positions to NDC for screen-space text rendering.

layout(location = 0) in vec2 inPos;   // Screen pixels (origin = top-left)
layout(location = 1) in vec2 inUV;    // Font atlas UV

layout(location = 0) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    vec2 screenSize;   // Viewport width, height
    vec2 _pad0;
    vec4 textColor;    // RGBA text color
};

void main() {
    // Pixel coords → NDC: x ∈ [0, W] → [-1, 1], y ∈ [0, H] → [-1, 1]
    // Note: Vulkan NDC has Y pointing down, so top-left = (-1, -1).
    vec2 ndc = (inPos / screenSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    fragUV = inUV;
}
