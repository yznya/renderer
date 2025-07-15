#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_KHR_shader_subgroup_ballot: require

#include "math.inc.glsl"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

#define NUM_SAMPLES 10000

layout(binding = 0) writeonly buffer Data
{
    float floats[];
};

// based on http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
float random(vec2 co) {
    float a = 12.9898;
    float b = 78.233;
    float c = 43758.5453;
    float dt = dot(co.xy, vec2(a, b));
    float sn = mod(dt, 3.14);
    return fract(sin(sn) * c);
}

// based on http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_slides.pdf
vec3 importance_sample_ggx(vec2 xi, float roughness, vec3 normal) {
    // Maps a 2D point to a hemisphere with spread based on roughness
    float alpha = roughness * roughness;
    float phi = 2.0 * PI * xi.x + random(normal.xz) * 0.1;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    vec3 h = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // Tangent space
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent_x = normalize(cross(up, normal));
    vec3 tangent_y = normalize(cross(normal, tangent_x));

    // Convert to world Space
    return normalize(tangent_x * h.x + tangent_y * h.y + normal * h.z);
}

// Geometric Shadowing function
float g_schlicksmith_ggx(float n_dot_l, float n_dot_v, float roughness) {
    float k = (roughness * roughness) / 2.0;
    float GL = n_dot_l / (n_dot_l * (1.0 - k) + k);
    float GV = n_dot_v / (n_dot_v * (1.0 - k) + k);
    return GL * GV;
}

// https://github.com/google/filament/blob/master/shaders/src/brdf.fs#L136
float v_ashikhmin(float n_dot_l, float n_dot_v) {
    return clamp(1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)), 0.0, 1.0);
}

float d_charlie(float sheen_roughness, float n_dot_h) {
    sheen_roughness = max(sheen_roughness, 0.000001); //clamp (0,1]
    float inv_r = 1.0 / sheen_roughness;
    float cos2h = n_dot_h * n_dot_h;
    float sin2h = 1.0 - cos2h;
    return (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) / (2.0 * PI);
}

vec3 importance_sample_charlie(vec2 xi, float roughness, vec3 normal) {
    // Maps a 2D point to a hemisphere with spread based on roughness
    float alpha = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float sin_theta = pow(xi.y, alpha / (2.0 * alpha + 1.0));
    float cos_theta = sqrt(1.0 - sin_theta * sin_theta);

    vec3 H = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // Tangent space
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent_x = normalize(cross(up, normal));
    vec3 tangent_y = normalize(cross(normal, tangent_x));

    // Convert to world Space
    return normalize(tangent_x * H.x + tangent_y * H.y + normal * H.z);
}

vec3 brdf(float n_dot_v, float roughness) {
    // Normal always points along z-axis for the 2D lookup
    const vec3 n = vec3(0.0, 0.0, 1.0);
    vec3 v = vec3(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);

    vec3 lut = vec3(0.0);
    for (uint i = 0u; i < NUM_SAMPLES; i++) {
        vec2 xi = hammersley2d(i, NUM_SAMPLES);
        vec3 h = importance_sample_ggx(xi, roughness, n);
        vec3 l = 2.0 * dot(v, h) * h - v;

        float n_dot_l = max(dot(n, l), 0.0);
        float n_dot_v = max(dot(n, v), 0.0);
        float v_dot_h = max(dot(v, h), 0.0);
        float h_dot_n = max(dot(h, n), 0.0);

        if (n_dot_l > 0.0) {
            float g = g_schlicksmith_ggx(n_dot_l, n_dot_v, roughness);
            float g_vis = (g * v_dot_h) / (h_dot_n * n_dot_v);
            float fc = pow(1.0 - v_dot_h, 5.0);
            lut.rg += vec2((1.0 - fc) * g_vis, fc * g_vis);
        }
    }
    for (uint i = 0u; i < NUM_SAMPLES; i++) {
        vec2 xi = hammersley2d(i, NUM_SAMPLES);
        vec3 h = importance_sample_charlie(xi, roughness, n);
        vec3 l = 2.0 * dot(v, h) * h - v;

        float n_dot_l = max(dot(n, l), 0.0);
        float n_dot_v = max(dot(n, v), 0.0);
        float v_dot_h = max(dot(v, h), 0.0);
        float h_dot_n = max(dot(h, n), 0.0);

        if (n_dot_l > 0.0) {
            float sheen_distribution = d_charlie(roughness, h_dot_n);
            float sheen_visibility = v_ashikhmin(n_dot_l, n_dot_v);
            lut.b += sheen_visibility * sheen_distribution * n_dot_l * v_dot_h;
        }
    }

    return lut / float(NUM_SAMPLES);
}

void main()
{
    vec2 uv;
    uv.x = (float(gl_GlobalInvocationID.x) + 0.5) / float(256);
    uv.y = (float(gl_GlobalInvocationID.y) + 0.5) / float(256);

    vec3 v = brdf(uv.x, 1.0 - uv.y);

    uint offset = gl_GlobalInvocationID.y * 256 + gl_GlobalInvocationID.x;

    floats[offset * 4 + 0] = v.x;
    floats[offset * 4 + 1] = v.y;

    floats[offset * 4 + 2] = v.z;
}
