#version 460
#extension GL_EXT_shader_atomic_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_ray_query: require

#include "mesh.inc.glsl"
#include "math.inc.glsl"
#include "utils.inc.glsl"

layout(constant_id = 0) const int LATE = 0;
layout(constant_id = 1) const int PASS = 0;

layout(binding = 0, set = 1) uniform sampler2D textures[];

layout(binding = 0) uniform UniformBufferObject {
    UniformBuffer ubo;
};

layout(binding = 5) readonly buffer MeshDraws
{
    MeshDraw mesh_draws[];
};

layout(binding = 10) readonly buffer Materials
{
    Material materials[];
};

layout(location = 0) in flat uint draw_id;
layout(location = 1) in flat uint triangle_id;
layout(location = 2) in flat uint vertex_offset;
// TODO: we only need this in the alpha pass
layout(location = 3) in vec4 uv;

layout(location = 0) out uvec2 out_color;

vec4 sample_or_default(uint texture_index, vec2 uv, vec4 def) {
    if (texture_index != ~0) {
        return texture(textures[nonuniformEXT(texture_index)], uv);
    } else {
        return def;
    }
}

vec2 get_uv(uint uv_index, vec4 uv) {
    return uv_index == 0 ? uv.xy : uv.zw;
}

void main() {
    if (PASS == MASK_PASS) {
        MeshDraw mesh_draw = mesh_draws[draw_id];
        Material mat = materials[mesh_draw.material_index];

        vec4 albedo = sample_or_default(
                mat.albedo_texture,
                get_uv(mat.albedo_texture_uv, uv) * mat.albedo_texture_transform.xy + mat.albedo_texture_transform.zw,
                vec4(1.0, 1.0, 1.0, 1.0)
            ) * mat.base_color_factor;

        if (albedo.a < mat.emissive_factor_alpha_cutoff.w) {
            discard;
        }
    }

    out_color = uvec2(
            vertex_offset << 8 | triangle_id & 0xff,
            draw_id
        );
}
