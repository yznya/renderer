#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
}
pushConstants;

void main() {
    gl_Position =
        vec4(2.0 * pos.x / pushConstants.screen_size.x - 1.0,
            2.0 * pos.y / pushConstants.screen_size.y - 1.0, 0.0, 1.0);
    out_color = vec4(color.rgb, color.a);
    out_uv = uv;
}
