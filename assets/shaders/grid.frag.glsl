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

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 cam_pos;
layout(location = 0) out vec4 out_color;

vec4 grid_color(vec2 uv, vec2 cam_pos)
{
    vec2 dudv = vec2(
            length(vec2(dFdx(uv.x), dFdy(uv.x))),
            length(vec2(dFdx(uv.y), dFdy(uv.y)))
        );

    float lod_level = max(0.0, log10((length(dudv) * grid_min_pixels_between_cells) / grid_cell_size) + 1.0);
    float lod_fade = fract(lod_level);

    // cell sizes for lod0, lod1 and lod2
    float lod0 = grid_cell_size * pow(10.0, floor(lod_level));
    float lod1 = lod0 * 10.0;
    float lod2 = lod1 * 10.0;

    dudv *= 2.0;

    // Update grid coordinates for subsequent alpha calculations (centers each anti-aliased line)
    uv += dudv / 2.0F;

    // calculate absolute distances to cell line centers for each lod and pick max X/Y to get coverage alpha value
    float lod0a = max2(vec2(1.0) - abs(satv(mod(uv, lod0) / dudv) * 2.0 - vec2(1.0)));
    float lod1a = max2(vec2(1.0) - abs(satv(mod(uv, lod1) / dudv) * 2.0 - vec2(1.0)));
    float lod2a = max2(vec2(1.0) - abs(satv(mod(uv, lod2) / dudv) * 2.0 - vec2(1.0)));

    uv -= cam_pos;

    // blend between falloff colors to handle LOD transition
    vec4 c = lod2a > 0.0 ? grid_color_thick : lod1a > 0.0 ? mix(grid_color_thick, grid_color_thin, lod_fade) : grid_color_thin;

    // calculate opacity falloff based on distance to grid extents
    float opacity_falloff = (1.0 - satf(length(uv) / grid_size));

    // blend between LOD level alphas and scale with opacity falloff
    c.a *= (lod2a > 0.0 ? lod2a : lod1a > 0.0 ? lod1a : (lod0a * (1.0 - lod_fade))) * opacity_falloff;
    c.a *= 0.5;

    return c;
}

void main()
{
    out_color = grid_color(uv, cam_pos);
};
