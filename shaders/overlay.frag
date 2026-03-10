#version 450

// Phase 7 — Debug overlay fragment shader.
// Samples a single-channel (R8) font atlas and multiplies by the text color.
// The font atlas stores glyph coverage in the red channel.

layout(location = 0) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D fontAtlas;

layout(push_constant) uniform PushConstants {
    vec2 screenSize;
    vec2 _pad0;
    vec4 textColor;
};

void main() {
    float alpha = texture(fontAtlas, fragUV).r;
    // Discard fully transparent fragments to avoid depth/blend artefacts.
    if (alpha < 0.01) discard;
    outColor = vec4(textColor.rgb, textColor.a * alpha);
}
