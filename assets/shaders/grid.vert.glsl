#version 460 core
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require

#include "mesh.inc.glsl"
#include "grid.inc.glsl"

layout(binding = 0) uniform UniformBufferObject {
    UniformBuffer ubo;
};

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec2 out_cam_pos;

void main()
{
    int idx = indices[gl_VertexIndex];
    vec3 position = pos[idx] * grid_size;

    position.x += ubo.camera_position.x;
    position.z += ubo.camera_position.z;

    out_cam_pos = ubo.camera_position.xz;

    gl_Position = ubo.view_proj * vec4(position, 1.0);
    out_uv = position.xz;
}
