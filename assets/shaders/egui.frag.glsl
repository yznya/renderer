#version 450

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 out_color;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

void main() {
    out_color = color * texture(font_texture, uv);
}
