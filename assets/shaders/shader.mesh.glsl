#version 450
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_mesh_shader: require
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require

#include "mesh.inc.glsl"
#include "utils.inc.glsl"

layout(local_size_x = MESH_WGSIZE, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

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

layout(binding = 3) readonly buffer MeshletVertices
{
    uint meshlet_vertices[];
};

layout(binding = 4) readonly buffer MeshletTriangles
{
    uint8_t meshlet_triangles[];
};

layout(binding = 5) readonly buffer MeshDraws
{
    MeshDraw mesh_draws[];
};

layout(binding = 11) readonly buffer Meshes
{
    Mesh meshes[];
};

layout(binding = 12) buffer Models
{
    mat4 models[];
};

taskPayloadSharedEXT MeshTaskPayload payload;

perprimitiveEXT out gl_MeshPerPrimitiveEXT {
    int gl_PrimitiveID;
    int gl_Layer;
    int gl_ViewportIndex;
    bool gl_CullPrimitiveEXT;
    int gl_PrimitiveShadingRateEXT;
} gl_MeshPrimitivesEXT[];

layout(location = 0) out perprimitiveEXT uint out_draw_id[];
layout(location = 1) out perprimitiveEXT uint out_triangle_id[];
layout(location = 2) out perprimitiveEXT uint out_vertex_offset[];
layout(location = 3) out vec4 out_uv[];

shared vec3 vertex_clip[64];

void main()
{
    uint mi = payload.meshlet_indices[gl_WorkGroupID.x];
    uint ti = gl_LocalInvocationIndex;
    MeshDraw mesh_draw = mesh_draws[payload.draw_index];
    Mesh mesh = meshes[mesh_draw.mesh_index];
    Meshlet meshlet = meshlets[mi];

    SetMeshOutputsEXT(uint(meshlet.vertex_count), uint(meshlet.triangle_count));

    vec2 screen = vec2(ubo.screen_width, ubo.screen_height);
    mat4 model = models[mesh_draw.model_index];

    for (uint i = ti; i < uint(meshlet.vertex_count); i += MESH_WGSIZE) {
        uint vi = meshlet_vertices[meshlet.vertex_offset + i] + mesh.vertex_offset;
        vec3 position = vec3(vertices[vi].p.xyz);
        vec4 clip = ubo.view_proj * (model * vec4(position, 1.0));
        gl_MeshVerticesEXT[i].gl_Position = clip;

        out_uv[i] = vertices[vi].uv;

        vertex_clip[i] = vec3((clip.xy / clip.w * 0.5 + vec2(0.5)) * screen, clip.w);
    }

    barrier();

    uint index_offset = meshlet.triangle_offset;
    for (uint i = ti; i < uint(meshlet.triangle_count); i += MESH_WGSIZE) {
        uint a = uint(meshlet_triangles[index_offset + i * 3]);
        uint b = uint(meshlet_triangles[index_offset + i * 3 + 1]);
        uint c = uint(meshlet_triangles[index_offset + i * 3 + 2]);

        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(a, b, c);
        out_triangle_id[i] = i;
        out_draw_id[i] = payload.draw_index;
        out_vertex_offset[i] = mi;

        bool culled = false;

        vec2 pa = vertex_clip[a].xy;
        vec2 pb = vertex_clip[b].xy;
        vec2 pc = vertex_clip[c].xy;

        vec2 eba = pb - pa;
        vec2 eca = pc - pa;

        // TODO: leaves should be rendereed without backface culling
        culled = culled || (eba.x * eca.y <= eba.y * eca.x);

        vec2 bmin = min(pa, min(pb, pc));
        vec2 bmax = max(pa, max(pb, pc));
        float sbprec = 1.0 / 256.0;

        culled = culled || (round(bmin.x - sbprec) == round(bmax.x) || round(bmin.y) == round(bmax.y + sbprec));

        culled = culled && (vertex_clip[a].z > 0 && vertex_clip[b].z > 0 && vertex_clip[c].z > 0);

        gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT = culled;
    }
}
