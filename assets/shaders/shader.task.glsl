#version 450
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_mesh_shader: require
#extension GL_GOOGLE_include_directive: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_EXT_debug_printf: require
#extension GL_EXT_spirv_intrinsics: require

#include "mesh.inc.glsl"
#include "math.inc.glsl"

layout(constant_id = 0) const int LATE = 0;
layout(constant_id = 1) const int PASS = 0;

layout(local_size_x = TASK_WGSIZE, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform UniformBufferObject {
    UniformBuffer ubo;
};

layout(binding = 1) readonly buffer Vertices
{
    Vertex vertices[];
};

layout(binding = 2) readonly buffer Meshlets
{
    Meshlet meshlets[];
};

layout(binding = 5) readonly buffer MeshDraws
{
    MeshDraw mesh_draws[];
};

layout(binding = 6) readonly buffer MeshTaskCommands
{
    MeshTaskCommand mesh_task_commands[];
};

layout(binding = 7) buffer MeshletVisibilty
{
    uint meshlet_visibility[];
};

layout(binding = 8) uniform sampler2D depth_pyramid;

layout(binding = 12) buffer Models
{
    mat4 models[];
};

taskPayloadSharedEXT MeshTaskPayload payload;

bool cone_cull(vec3 center, float radius, vec3 cone_axis, float cone_cutoff) {
    return dot(center, cone_axis) >= cone_cutoff * length(center) + radius;
}

shared int shared_count;

void main()
{
    uint draw_id = mesh_task_commands[gl_WorkGroupID.x].draw_id;
    uint late_draw_visibility = mesh_task_commands[gl_WorkGroupID.x].late_draw_visibility;
    uint task_count = mesh_task_commands[gl_WorkGroupID.x].task_count;

    MeshDraw mesh_draw = mesh_draws[draw_id];

    uint mgi = gl_LocalInvocationID.x;
    uint mi = mgi + mesh_task_commands[gl_WorkGroupID.x].task_offset;
    uint meshlet_visibility_index = mgi + mesh_task_commands[gl_WorkGroupID.x].meshlet_visibility_offset;

    uint meshlet_visibility_bit = meshlet_visibility[meshlet_visibility_index >> 5] & (1u << (meshlet_visibility_index & 31));

    shared_count = 0;
    barrier();

    mat4 model = models[mesh_draw.model_index];

    vec3 cone_center = meshlets[mi].center;
    cone_center = (ubo.view * (model * vec4(cone_center, 1.0))).xyz;
    vec3 cone_axis = vec3(int(meshlets[mi].cone_axis[0]) / 127.0, int(meshlets[mi].cone_axis[1]) / 127.0, int(meshlets[mi].cone_axis[2]) / 127.0);
    cone_axis = normalize((ubo.view * (model * vec4(cone_axis, 0.0))).xyz);

    vec3 scale = extract_scale(model);
    float max_scale = max(scale.x, max(scale.y, scale.z));

    float radius = max_scale * meshlets[mi].radius;

    bool valid = mgi < task_count;
    bool accept = valid;

    if (LATE == 0 && meshlet_visibility_bit == 0) {
        accept = false;
    }
    bool skip = LATE == 1 && late_draw_visibility == 1 && meshlet_visibility_bit != 0 && PASS != MASK_PASS;

    accept = accept && !cone_cull(cone_center, radius, cone_axis, int(meshlets[mi].cone_cutoff) / 127.0);
    accept = accept && cone_center.z * -ubo.frustum[1] + cone_center.x * ubo.frustum[0] < radius;
    accept = accept && cone_center.z * -ubo.frustum[3] + cone_center.y * ubo.frustum[2] < radius;
    accept = accept && cone_center.z + radius > -ubo.near && cone_center.z + radius < ubo.far;

    if (LATE == 1 && accept && PASS != MASK_PASS) {
        vec4 aabb;
        if (projectSphere(cone_center, radius, ubo.near, ubo.proj[0][0], ubo.proj[1][1], aabb)) {
            float width = (aabb.z - aabb.x) * ubo.depth_pyramid_width;
            float height = (aabb.w - aabb.y) * ubo.depth_pyramid_height;

            float level = ceil(log2(max(width, height)));
            float depth = textureLod(depth_pyramid, (aabb.xy + aabb.zw) * 0.5, level).x;
            float depth_sphere = ubo.near / (cone_center.z - radius);

            accept = accept && depth_sphere > depth;
        }
    }

    if (LATE == 1 && valid) {
        if (accept) {
            atomicOr(meshlet_visibility[meshlet_visibility_index >> 5], 1 << (meshlet_visibility_index & 31));
        } else {
            atomicAnd(meshlet_visibility[meshlet_visibility_index >> 5], ~(1 << (meshlet_visibility_index & 31)));
        }
    }

    if (accept && !skip) {
        uint index = atomicAdd(shared_count, 1);
        payload.meshlet_indices[index] = mi;
    }
    payload.draw_index = draw_id;

    barrier();
    EmitMeshTasksEXT(shared_count, 1, 1);
}
