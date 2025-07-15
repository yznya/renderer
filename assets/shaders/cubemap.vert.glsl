#version 460 core
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require

#include "mesh.inc.glsl"

layout(location = 0) out vec3 dir;

const vec3 pos[8] = vec3[8](
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(-1.0, 1.0, 1.0),

        vec3(-1.0, -1.0, -1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(-1.0, 1.0, -1.0)
    );

const int indices[36] = int[36](
        // front
        0, 1, 2, 2, 3, 0,
        // right
        1, 5, 6, 6, 2, 1,
        // back
        7, 6, 5, 5, 4, 7,
        // left
        4, 0, 3, 3, 7, 4,
        // bottom
        4, 5, 1, 1, 0, 4,
        // top
        3, 2, 6, 6, 7, 3
    );

layout(binding = 0) uniform UniformBufferObject {
    UniformBuffer ubo;
};

void main()
{
    int idx = indices[gl_VertexIndex];
    gl_Position = ubo.view_proj * vec4(10000.0 * pos[idx] + ubo.camera_position, 1.0);
    dir = pos[idx].xyz;
}
