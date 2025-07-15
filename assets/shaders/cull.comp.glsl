#version 450
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_KHR_shader_subgroup_ballot: require

#include "mesh.inc.glsl"
#include "math.inc.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const int LATE = 0;
layout(constant_id = 1) const int PASS = 0;

layout(binding = 0) uniform UniformBufferObject {
    UniformBuffer ubo;
};

layout(binding = 1) readonly buffer Meshes
{
    Mesh meshes[];
};

layout(binding = 2) readonly buffer MeshDraws
{
    MeshDraw mesh_draws[];
};

layout(binding = 3) writeonly buffer MeshTaskCommands
{
    MeshTaskCommand mesh_task_commands[];
};

layout(binding = 4) buffer MeshDrawCommandsCount
{
    uint draw_count;
};

layout(binding = 5) buffer DrawVisibility
{
    uint draw_visibility[];
};

layout(binding = 6) buffer Models
{
    mat4 models[];
};

layout(binding = 7) uniform sampler2D depth_pyramid;

void main()
{
    uint mi = gl_GlobalInvocationID.x;

    if (mi >= ubo.draw_count) {
        return;
    }

    if (LATE == 0 && draw_visibility[mi] == 0) {
        return;
    }

    MeshDraw mesh_draw = mesh_draws[mi];

    if (mesh_draw.pass_type != PASS) {
        return;
    }

    mat4 model = models[mesh_draw.model_index];

    vec3 center = (ubo.view * (model * vec4(meshes[mesh_draw.mesh_index].center, 1.0))).xyz;
    vec3 scale = extract_scale(model);
    float max_scale = max(scale.x, max(scale.y, scale.z));
    float radius = max_scale * meshes[mesh_draw.mesh_index].radius;

    bool visible = true;

    visible = visible && center.z * -ubo.frustum[1] + abs(center.x) * ubo.frustum[0] < radius;
    visible = visible && center.z * -ubo.frustum[3] + abs(center.y) * ubo.frustum[2] < radius;
    visible = visible && center.z + radius > -ubo.near && center.z + radius < ubo.far;

    if (LATE == 1 && visible && ubo.cull_enabled == 1) {
        vec4 aabb;
        if (projectSphere(center, abs(radius), ubo.near, ubo.proj[0][0], ubo.proj[1][1], aabb)) {
            float width = (aabb.z - aabb.x) * ubo.depth_pyramid_width;
            float height = (aabb.w - aabb.y) * ubo.depth_pyramid_height;

            float level = ceil(log2(max(width, height)));
            float depth = textureLod(depth_pyramid, (aabb.xy + aabb.zw) * 0.5, level).x;
            float depth_sphere = ubo.near / (center.z - radius);

            visible = visible && depth_sphere > depth;
        }
    }

    if (visible || PASS == MASK_PASS) {
        uint lod_index = 0;
        float threshold = max(length(center) - radius, 0) * ubo.lod_target / max_scale;

        for (uint i = 1; i < meshes[mesh_draw.mesh_index].lod_count; ++i) {
            if (meshes[mesh_draw.mesh_index].lods[i].error < threshold) {
                lod_index = i;
            }
        }

        MeshLod lod = meshes[mesh_draw.mesh_index].lods[ubo.lod_enabled == 1 ? lod_index : 0];

        uint task_groups = (lod.meshlet_count + TASK_WGSIZE - 1) / TASK_WGSIZE;
        uint draw_call_index = atomicAdd(draw_count, task_groups);
        uint late_draw_visibility = draw_visibility[mi];
        uint meshlet_visibility_offset = mesh_draw.meshlet_visibility_offset;

        if (draw_call_index + task_groups <= TASK_WGLIMIT) {
            for (uint i = 0; i < task_groups; ++i) {
                mesh_task_commands[draw_call_index + i].draw_id = mi;
                mesh_task_commands[draw_call_index + i].task_offset = lod.meshlet_offset + i * TASK_WGSIZE;
                mesh_task_commands[draw_call_index + i].task_count = min(lod.meshlet_count, lod.meshlet_count - i * TASK_WGSIZE);
                mesh_task_commands[draw_call_index + i].late_draw_visibility = late_draw_visibility;
                mesh_task_commands[draw_call_index + i].meshlet_visibility_offset = meshlet_visibility_offset + i * TASK_WGSIZE;
            }
        }
    }

    if (LATE == 1) {
        draw_visibility[mi] = visible ? 1 : 0;
    }
}
