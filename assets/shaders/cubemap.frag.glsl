#version 460 core

layout(location = 0) in vec3 dir;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform samplerCube cubemap_texture;

void main()
{
    out_color = texture(cubemap_texture, dir);
};
