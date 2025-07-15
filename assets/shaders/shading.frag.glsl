#version 460
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

#define DEBUG_MESHLET_VISUALIZATION 0
#define DEBUG_TRIANGLE_VISUALIZATION 0
#define DEBUG_DRAW_VISUALIZATION 0

layout(binding = 0) uniform UniformBufferObject {
    UniformBuffer ubo;
};

layout(binding = 1, rg32ui) uniform readonly uimage2D source;

layout(binding = 2) readonly buffer Vertices
{
    Vertex vertices[];
};

layout(binding = 3) readonly buffer Meshlets
{
    Meshlet meshlets[];
};

layout(binding = 4) readonly buffer MeshletVertices
{
    uint meshlet_vertices[];
};

layout(binding = 5) readonly buffer MeshletTriangles
{
    uint8_t meshlet_triangles[];
};

layout(binding = 6) readonly buffer MeshDraws
{
    MeshDraw mesh_draws[];
};

layout(binding = 7) readonly buffer Meshes
{
    Mesh meshes[];
};

layout(binding = 8) buffer Models
{
    mat4 models[];
};

layout(binding = 9, set = 0) uniform sampler2D brdf_lut;
layout(binding = 10, set = 0) uniform samplerCube lambertian;
layout(binding = 11, set = 0) uniform samplerCube ggx;
layout(binding = 12, set = 0) uniform samplerCube charlie;
layout(binding = 13, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 14) readonly buffer Materials
{
    Material materials[];
};

layout(binding = 0, set = 1) uniform sampler2D textures[];

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

struct BarycentricDeriv
{
    vec3 lambda;
    vec3 ddx;
    vec3 ddy;
};

BarycentricDeriv calc_full_bary(vec4 pt0, vec4 pt1, vec4 pt2, vec2 pixel_ndc, vec2 win_size)
{
    BarycentricDeriv ret;

    vec3 inv_w = 1.0 / vec3(pt0.w, pt1.w, pt2.w);

    vec2 ndc0 = pt0.xy * inv_w.x;
    vec2 ndc1 = pt1.xy * inv_w.y;
    vec2 ndc2 = pt2.xy * inv_w.z;

    float inv_det = 1.0 / (determinant(mat2(ndc2 - ndc1, ndc0 - ndc1)));
    ret.ddx = vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * inv_det * inv_w;
    ret.ddy = vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * inv_det * inv_w;
    float ddx_sum = dot(ret.ddx, vec3(1, 1, 1));
    float ddy_sum = dot(ret.ddy, vec3(1, 1, 1));

    vec2 delta_vec = pixel_ndc - ndc0;
    float interp_inv_w = inv_w.x + delta_vec.x * ddx_sum + delta_vec.y * ddy_sum;
    float interp_w = 1.0 / (interp_inv_w);

    ret.lambda.x = interp_w * (inv_w[0] + delta_vec.x * ret.ddx.x + delta_vec.y * ret.ddy.x);
    ret.lambda.y = interp_w * (0.0f + delta_vec.x * ret.ddx.y + delta_vec.y * ret.ddy.y);
    ret.lambda.z = interp_w * (0.0f + delta_vec.x * ret.ddx.z + delta_vec.y * ret.ddy.z);

    ret.ddx *= (2.0f / win_size.x);
    ret.ddy *= (2.0f / win_size.y);
    ddx_sum *= (2.0f / win_size.x);
    ddy_sum *= (2.0f / win_size.y);

    ret.ddy *= -1.0f;
    ddy_sum *= -1.0f;

    float interp_w_ddx = 1.0f / (interp_inv_w + ddx_sum);
    float interp_w_ddy = 1.0f / (interp_inv_w + ddy_sum);

    ret.ddx = interp_w_ddx * (ret.lambda * interp_inv_w + ret.ddx) - ret.lambda;
    ret.ddy = interp_w_ddy * (ret.lambda * interp_inv_w + ret.ddy) - ret.lambda;

    return ret;
}

vec3 interpolate_with_deriv(BarycentricDeriv deriv, float v0, float v1, float v2)
{
    vec3 merged_v = vec3(v0, v1, v2);
    vec3 ret;
    ret.x = dot(merged_v, deriv.lambda);
    ret.y = dot(merged_v, deriv.ddx);
    ret.z = dot(merged_v, deriv.ddy);
    return ret;
}

struct InterpolatedVec3 {
    vec3 value;
    vec3 ddx;
    vec3 ddy;
};

struct InterpolatedVec2 {
    vec2 value;
    vec2 ddx;
    vec2 ddy;
};

InterpolatedVec3 interpolate(BarycentricDeriv deriv, vec3 v0, vec3 v1, vec3 v2) {
    vec3 vx = interpolate_with_deriv(deriv, v0.x, v1.x, v2.x);
    vec3 vy = interpolate_with_deriv(deriv, v0.y, v1.y, v2.y);
    vec3 vz = interpolate_with_deriv(deriv, v0.z, v1.z, v2.z);
    InterpolatedVec3 ret;
    ret.value = vec3(vx.x, vy.x, vz.x);
    ret.ddx = vec3(vx.y, vy.y, vz.y);
    ret.ddy = vec3(vx.z, vy.z, vz.z);
    return ret;
}

InterpolatedVec2 interpolate(BarycentricDeriv deriv, vec2 v0, vec2 v1, vec2 v2) {
    vec3 vx = interpolate_with_deriv(deriv, v0.x, v1.x, v2.x);
    vec3 vy = interpolate_with_deriv(deriv, v0.y, v1.y, v2.y);
    InterpolatedVec2 ret;
    ret.value = vec2(vx.x, vy.x);
    ret.ddx = vec2(vx.y, vy.y);
    ret.ddy = vec2(vx.z, vy.z);
    return ret;
}

InterpolatedVec2 get_uv(uint uv_index, InterpolatedVec2 uv0, InterpolatedVec2 uv1, vec2 scale, vec2 offset) {
    InterpolatedVec2 uv = uv_index == 0 ? uv0 : uv1;
    uv.ddx = uv.ddx * scale;
    uv.ddy = uv.ddy * scale;
    uv.value = uv.value * scale + offset;
    return uv;
}

vec4 sample_or_default(uint texture_index, InterpolatedVec2 uv, vec4 def) {
    if (texture_index != ~0) {
        return textureGrad(textures[nonuniformEXT(texture_index)], uv.value, uv.ddx, uv.ddy);
    } else {
        return def;
    }
}

const float MIN_ROUGHNESS = 0.04;

float geometric_occlusion(float n_dot_l, float n_dot_v, float alpha_roughness) {
    float r_sqr = alpha_roughness * alpha_roughness;

    float attenuation_l = 2.0 * n_dot_l / (n_dot_l + sqrt(r_sqr + (1.0 - r_sqr) * (n_dot_l * n_dot_l)));
    float attenuation_v = 2.0 * n_dot_v / (n_dot_v + sqrt(r_sqr + (1.0 - r_sqr) * (n_dot_v * n_dot_v)));
    return attenuation_l * attenuation_v;
}

float microfacet_distribution(float alpha_roughness, float n_dot_h) {
    float roughness_sq = alpha_roughness * alpha_roughness;
    float f = (n_dot_h * roughness_sq - n_dot_h) * n_dot_h + 1.0;
    return roughness_sq / (PI * f * f);
}

vec3 f_schlick(vec3 f0, vec3 f90, float v_dot_h) {
    return f0 + (f90 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

vec3 diffuse_burley(vec3 diffuse_color, float l_dot_h, float n_dot_l, float n_dot_v, float alpha_roughness) {
    float f90 = 2.0 * l_dot_h * l_dot_h * alpha_roughness - 0.5;

    return (diffuse_color / PI) * (1.0 + f90 * pow((1.0 - n_dot_l), 5.0)) * (1.0 + f90 * pow((1.0 - n_dot_v), 5.0));
}

vec3 get_ibl_radiance_contribution_ggx(vec3 n, vec3 v, float n_dot_v, float perceptual_roughness, vec3 reflectance0, float specular_weight) {
    vec3 reflection = normalize(reflect(-v, n));
    float mip_count = textureQueryLevels(ggx);
    float lod = perceptual_roughness * (mip_count - 1);

    // retrieve a scale and bias to F0. See [1], Figure 3
    vec2 brdf_sample_point = clamp(vec2(n_dot_v, perceptual_roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec3 brdf = texture(brdf_lut, brdf_sample_point).rgb;
    // HDR envmaps are already linear
    vec3 specular_light = textureLod(ggx, reflection.xyz, lod).rgb;

    // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
    // Roughness dependent fresnel, from Fdez-Aguera
    vec3 fr = max(vec3(1.0 - perceptual_roughness), reflectance0) - reflectance0;
    vec3 k_s = reflectance0 + fr * pow(1.0 - n_dot_v, 5.0);
    vec3 fss_ess = k_s * brdf.x + brdf.y;

    return specular_weight * specular_light * fss_ess;
}

vec3 get_ibl_radiance_lambertian(float n_dot_v, vec3 n, float roughness, vec3 diffuse_color, vec3 f0, float specular_weight) {
    vec2 brdf_sample_point = clamp(vec2(n_dot_v, roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 f_ab = texture(brdf_lut, brdf_sample_point).rg;

    vec3 irradiance = texture(lambertian, n.xyz).rgb;

    // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
    // Roughness dependent fresnel, from Fdez-Aguera
    vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
    vec3 k_s = f0 + fr * pow(1.0 - n_dot_v, 5.0);
    vec3 fss_ess = specular_weight * k_s * f_ab.x + f_ab.y; // <--- GGX / specular light contribution (scale it down if the specularWeight is low)

    // Multiple scattering, from Fdez-Aguera
    float ems = (1.0 - (f_ab.x + f_ab.y));
    vec3 f_avg = specular_weight * (f0 + (1.0 - f0) / 21.0);
    vec3 fms_ems = ems * fss_ess * f_avg / (1.0 - f_avg * ems);
    vec3 k_d = diffuse_color * (1.0 - fss_ess + fms_ems); // we use +FmsEms as indicated by the formula in the blog post (might be a typo in the implementation)

    return (fms_ems + k_d) * irradiance;
}

vec3 get_ibl_radiance_charlie(vec3 n, vec3 v, float n_dot_v, float sheen_roughness, vec3 sheen_color) {
    float mip_count = textureQueryLevels(charlie);
    float lod = sheen_roughness * float(mip_count - 1);
    vec3 reflection = normalize(reflect(-v, n));

    vec2 brdf_sample_point = clamp(vec2(n_dot_v, sheen_roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    float brdf = texture(brdf_lut, brdf_sample_point).b;
    vec3 sheen_sample = textureLod(charlie, reflection.xyz, lod).rgb;

    return sheen_sample * sheen_color * brdf;
}

float d_charlie(float sheen_roughness, float n_dot_h) {
    sheen_roughness = max(sheen_roughness, 0.000001);
    float alpha_g = sheen_roughness * sheen_roughness;
    float inv_r = 1.0 / alpha_g;
    float cos2h = n_dot_h * n_dot_h;
    float sin2h = 1.0 - cos2h;
    return (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) / (2.0 * PI);
}

float lambda_sheen_numeric_helper(float x, float alpha_g) {
    float one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
    float a = mix(21.5473, 25.3245, one_minus_alpha_sq);
    float b = mix(3.82987, 3.32435, one_minus_alpha_sq);
    float c = mix(0.19823, 0.16801, one_minus_alpha_sq);
    float d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
    float e = mix(-4.32054, -4.85967, one_minus_alpha_sq);
    return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float lambda_sheen(float cos_theta, float alpha_g) {
    if (abs(cos_theta) < 0.5) {
        return exp(lambda_sheen_numeric_helper(cos_theta, alpha_g));
    }

    return exp(2.0 * lambda_sheen_numeric_helper(0.5, alpha_g) - lambda_sheen_numeric_helper(1.0 - cos_theta, alpha_g));
}

float v_sheen(float n_dot_l, float n_dot_v, float sheen_roughness) {
    sheen_roughness = max(sheen_roughness, 0.000001);
    float alpha_g = sheen_roughness * sheen_roughness;

    return clamp(1.0 / ((1.0 + lambda_sheen(n_dot_l, alpha_g) + lambda_sheen(n_dot_l, alpha_g)) * (4.0 * n_dot_v * n_dot_l)), 0.0, 1.0);
}

vec3 get_brdf_specular_sheen(vec3 sheen_color, float sheen_roughness, float n_dot_l, float n_dot_v, float n_dot_h) {
    float sheen_distribution = d_charlie(sheen_roughness, n_dot_h);
    float sheen_visibility = v_sheen(n_dot_l, n_dot_v, sheen_roughness);
    return sheen_color * sheen_distribution * sheen_visibility;
}

vec3 get_punctual_radiance_sheen(vec3 sheen_color, float sheen_roughness, float n_dot_l, float n_dot_v, float n_dot_h) {
    return n_dot_l * get_brdf_specular_sheen(sheen_color, sheen_roughness, n_dot_l, n_dot_v, n_dot_h);
}

vec3 get_punctual_radiance_clearcoat(vec3 clearcoat_normal, vec3 v, vec3 l, vec3 h, float v_dot_h, vec3 f0, vec3 f90, float clearcoat_roughness) {
    float n_dot_l = clamp(dot(clearcoat_normal, l), 0.001, 1.0);
    float n_dot_v = clamp(dot(clearcoat_normal, v), 0.001, 1.0);
    float n_dot_h = clamp(dot(clearcoat_normal, h), 0.001, 1.0);

    vec3 f = f_schlick(f0, f90, v_dot_h);
    float g = geometric_occlusion(n_dot_l, n_dot_v, clearcoat_roughness * clearcoat_roughness);
    float d = microfacet_distribution(clearcoat_roughness * clearcoat_roughness, n_dot_h);
    vec3 spec_contrib = f * g * d / (4.0 * n_dot_l * n_dot_v);

    return n_dot_l * (spec_contrib);
}

void main() {
    vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
    uint index_x = uint(flipped_uv.x * ubo.screen_width);
    uint index_y = uint(flipped_uv.y * ubo.screen_height);

    uvec2 pixel_data = imageLoad(source, ivec2(int(index_x), int(index_y))).xy;

    if (pixel_data.y == ~0) {
        return;
    }

    uint meshlet_index = pixel_data.x >> 8;
    uint triangle_index = pixel_data.x & 0xff;
    uint draw_id = pixel_data.y;

    MeshDraw mesh_draw = mesh_draws[draw_id];

    Mesh mesh = meshes[mesh_draw.mesh_index];
    mat4 model = models[mesh_draw.model_index];
    Material mat = materials[mesh_draw.material_index];
    Meshlet meshlet = meshlets[meshlet_index];

    uint i0 = meshlet_triangles[meshlet.triangle_offset + triangle_index * 3];
    uint i1 = meshlet_triangles[meshlet.triangle_offset + triangle_index * 3 + 1];
    uint i2 = meshlet_triangles[meshlet.triangle_offset + triangle_index * 3 + 2];

    uint t0 = meshlet_vertices[meshlet.vertex_offset + i0];
    uint t1 = meshlet_vertices[meshlet.vertex_offset + i1];
    uint t2 = meshlet_vertices[meshlet.vertex_offset + i2];

    Vertex v1 = vertices[mesh.vertex_offset + t0];
    Vertex v2 = vertices[mesh.vertex_offset + t1];
    Vertex v3 = vertices[mesh.vertex_offset + t2];

    vec4 w_pos1 = model * vec4(v1.p.xyz, 1.0);
    vec4 w_pos2 = model * vec4(v2.p.xyz, 1.0);
    vec4 w_pos3 = model * vec4(v3.p.xyz, 1.0);

    vec4 pos1 = ubo.view_proj * w_pos1;
    vec4 pos2 = ubo.view_proj * w_pos2;
    vec4 pos3 = ubo.view_proj * w_pos3;

    BarycentricDeriv bary = calc_full_bary(pos1, pos2, pos3, uv * 2.0 - 1.0, vec2(ubo.screen_width, ubo.screen_height));

    vec3 w_pos = interpolate(bary, w_pos1.xyz, w_pos2.xyz, w_pos3.xyz).value;

    vec3 norm1 = mat3(model) * v1.n.xyz;
    vec3 norm2 = mat3(model) * v2.n.xyz;
    vec3 norm3 = mat3(model) * v3.n.xyz;
    vec3 norm = interpolate(bary, norm1, norm2, norm3).value;

    vec3 tangent1 = mat3(model) * v1.t.xyz;
    vec3 tangent2 = mat3(model) * v2.t.xyz;
    vec3 tangent3 = mat3(model) * v3.t.xyz;
    vec3 tangent = interpolate(bary, tangent1, tangent2, tangent3).value;

    InterpolatedVec2 uv0 = interpolate(bary, v1.uv.xy, v2.uv.xy, v3.uv.xy);
    InterpolatedVec2 uv1 = interpolate(bary, v1.uv.xy, v2.uv.xy, v3.uv.xy);

    vec4 albedo = sample_or_default(
            mat.albedo_texture,
            get_uv(mat.albedo_texture_uv, uv0, uv1, mat.albedo_texture_transform.xy, mat.albedo_texture_transform.zw),
            vec4(1.0, 1.0, 1.0, 1.0)
        ) * mat.base_color_factor;

    vec3 nmap = sample_or_default(
            mat.normal_texture,
            get_uv(mat.normal_texture_uv, uv0, uv1, mat.normal_texture_transform.xy, mat.normal_texture_transform.zw),
            vec4(0.5, 0.5, 1.0, 0.0)
        ).rgb * 2.0 - 1.0;

    vec4 metallic_roughness = sample_or_default(
            mat.metallic_roughness_texture,
            get_uv(mat.metallic_roughness_texture_uv, uv0, uv1, mat.metallic_roughness_texture_transform.xy, mat.metallic_roughness_texture_transform.zw),
            vec4(1.0)
        );

    vec3 occlusion = sample_or_default(
            mat.occlusion_texture,
            get_uv(mat.occlusion_texture_uv, uv0, uv1, mat.occlusion_texture_transform.xy, mat.occlusion_texture_transform.zw),
            vec4(1.0)).xyz;

    vec3 emissive = mat.emissive_factor_alpha_cutoff.rgb;
    emissive *= sample_or_default(
            mat.emissive_texture,
            get_uv(mat.emissive_texture_uv, uv0, uv1, vec2(1.0), vec2(0.0)),
            vec4(1.0)).rgb;

    vec4 specular_color_factor = mat.specular_factor;
    specular_color_factor.rgb *= sample_or_default(
            mat.specular_color_texture,
            get_uv(mat.specular_color_texture_uv, uv0, uv1, vec2(1.0), vec2(0.0)),
            vec4(1.0)).rgb;

    specular_color_factor.w *= sample_or_default(
            mat.specular_texture,
            get_uv(mat.specular_texture_uv, uv0, uv1, vec2(1.0), vec2(0.0)),
            vec4(1.0)).r;

    vec3 nnormal = normalize(norm);
    vec3 ntangent = normalize(tangent.xyz);
    vec3 bitangent = cross(nnormal, ntangent) * (v1.t.w * v2.t.w * v3.t.w);
    vec3 nrm = normalize(nmap.x * ntangent.xyz + nmap.y * bitangent + nmap.z * nnormal);

    vec3 light_direction = normalize(ubo.sun_direction.xyz);

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, 0xff, w_pos, 1e-2f, light_direction, 1000);
    rayQueryProceedEXT(rq);

    float metallic = mat.metallic_roughness_normal_occlusion.x * metallic_roughness.b;
    metallic = clamp(metallic, 0.0, 1.0);

    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
    float perceptual_roughness = metallic_roughness.g * mat.metallic_roughness_normal_occlusion.y;

    // spec gloss model
    if ((mat.material_type & SPEC_GLOSS) != 0) {
        perceptual_roughness = 1.0 - metallic_roughness.a * mat.specular_glossiness_factor.w;
    } else {
        perceptual_roughness = clamp(perceptual_roughness, MIN_ROUGHNESS, 1.0);
    }

    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness [2].
    float alpha_roughness = perceptual_roughness * perceptual_roughness;

    // The albedo may be defined from a base texture or a flat color
    vec4 base_color = albedo;

    vec3 f0 = vec3(pow((mat.ior - 1) / (mat.ior + 1), 2));
    vec3 diffuse_color = mix(base_color.rgb, vec3(0.0), metallic);
    vec3 specular_color = mix(f0, base_color.rgb, metallic);
    float specular_weight = 1.0;

    if ((mat.material_type & SPEC) != 0) {
        vec3 dielectric_spec_f0 = min(f0 * specular_color_factor.rgb, vec3(1.0));
        f0 = mix(dielectric_spec_f0, base_color.rgb, metallic);
        specular_weight = specular_color_factor.w;
    }

    // spec gloss model
    if ((mat.material_type & SPEC_GLOSS) != 0) {
        f0 = mat.specular_glossiness_factor.xyz * metallic_roughness.xyz;
        diffuse_color = base_color.rgb * (1.0 - max(max(f0.r, f0.g), f0.b));
        specular_color = f0;
    }

    // Compute reflectance.
    float reflectance = max(max(specular_color.r, specular_color.g), specular_color.b);

    // For typical incident reflectance range (between 4% to 100%) set the grazing reflectance to 100% for typical fresnel effect.
    // For very low reflectance range on highly diffuse objects (below 4%), incrementally reduce grazing reflecance to 0%.
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specular_environment_r0 = specular_color.rgb;
    vec3 specular_environment_r90 = vec3(reflectance90);

    vec3 world_pos = w_pos;
    vec3 v = -normalize(world_pos - ubo.camera_position);
    vec3 h = normalize(light_direction + v); // Half vector between both l and v

    float n_dot_v = clamp(abs(dot(nrm, v)), 0.001, 1.0);
    float n_dot_l = clamp(dot(nrm, light_direction), 0.001, 1.0);
    float n_dot_h = clamp(dot(nrm, h), 0.0, 1.0);
    float l_dot_h = clamp(dot(light_direction, h), 0.0, 1.0);
    float v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    vec3 color = vec3(0);
    vec3 light_intensity = vec3(5.0);

    // Calculate the shading terms for the microfacet specular shading model
    vec3 f = f_schlick(specular_environment_r0, specular_environment_r90, v_dot_h);
    float g = geometric_occlusion(n_dot_l, n_dot_v, alpha_roughness);
    float d = microfacet_distribution(alpha_roughness, n_dot_h);
    // Calculation of analytical lighting contribution
    vec3 diffuse_contrib = (1.0 - f) * diffuse_burley(diffuse_color, l_dot_h, n_dot_l, n_dot_v, alpha_roughness);
    vec3 spec_contrib = f * g * d / (4.0 * n_dot_l * n_dot_v);
    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
    color = n_dot_l * light_intensity * (diffuse_contrib + spec_contrib);

    vec3 ibl_specular_color = get_ibl_radiance_contribution_ggx(nrm, v, n_dot_v, perceptual_roughness, specular_environment_r0, specular_weight);
    vec3 ibl_diffuse_color = get_ibl_radiance_lambertian(n_dot_v, nrm, perceptual_roughness, diffuse_color, specular_environment_r0, specular_weight);

    color = clamp(ibl_specular_color + ibl_diffuse_color + color + emissive, 0.0, 1.0) * occlusion.r;

    if ((mat.material_type & CLEARCOAT) != 0) {
        vec3 nmap = sample_or_default(
                mat.clearcoat_texture,
                get_uv(mat.clearcoat_texture_uv, uv0, uv1, vec2(1.0), vec2(0.0)),
                vec4(0.5, 0.5, 1.0, 0.0)).rgb * 2.0 - 1.0;

        vec3 nrm = normalize(nmap.x * ntangent.xyz + nmap.y * bitangent + nmap.z * nnormal);
        float n_dot_v = clamp(abs(dot(nrm, v)), 0.001, 1.0);

        float clearcoat_roughness = mat.clearcoat_factor.y;
        vec3 clearcoat_f0 = vec3(pow((mat.ior - 1) / (mat.ior + 1), 2));
        vec3 clearcoat_contrib = get_ibl_radiance_contribution_ggx(nrm, v, n_dot_v, clearcoat_roughness, clearcoat_f0, 1.0);
        vec3 clearcoat_fresnel = f_schlick(specular_environment_r0, specular_environment_r90, n_dot_v);

        color = color * (1.0 - mat.clearcoat_factor.x * clearcoat_fresnel) + clearcoat_contrib;
        color += light_intensity * get_punctual_radiance_clearcoat(nrm, v, light_direction, h, v_dot_h, vec3(MIN_ROUGHNESS), vec3(1.0), clearcoat_roughness);
    }

    if ((mat.material_type & SHEEN) != 0) {
        vec3 sheen_color = mat.sheen_color_roughness_factor.xyz;
        float sheen_roughness = mat.sheen_color_roughness_factor.w;

        color += get_ibl_radiance_charlie(nrm, v, n_dot_v, sheen_roughness, sheen_color);
        color += light_intensity * get_punctual_radiance_sheen(sheen_color, sheen_roughness, n_dot_l, n_dot_v, n_dot_h);
    }

    color *= (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.35;

    out_color = vec4(color, 1.0);

    #if DEBUG_MESHLET_VISUALIZATION
    uint mh = hash(meshlet_index);
    vec3 meshlet_color = vec3((mh >> 24) & 0xff, (mh >> 16) & 0xff, (mh >> 8) & 0xff) / 127.0 - 1.0;
    out_color = vec4(meshlet_color, 1.0);
    #endif

    #if DEBUG_TRIANGLE_VISUALIZATION
    uint th = hash(triangle_index);
    vec3 triangle_color = vec3((th >> 24) & 0xff, (th >> 16) & 0xff, (th >> 8) & 0xff) / 127.0 - 1.0;
    out_color = vec4(triangle_color, 1.0);
    #endif

    #if DEBUG_DRAW_VISUALIZATION
    uint dh = hash(draw_id);
    vec3 draw_color = vec3((dh >> 24) & 0xff, (dh >> 16) & 0xff, (dh >> 8) & 0xff) / 127.0 - 1.0;
    out_color = vec4(draw_color, 1.0);
    #endif

    // out_color = pow(vec4((nrm + 1.0) * 0.5, 1.0), vec4(2.2 / 1.0));
    // out_color = vec4((nnormal + 1.0) * 0.5, 1.0);
    // out_color = vec4((ntangent + 1.0) * 0.5, 1.0);
    // out_color = vec4((perceptual_roughness.xxx), 1.0);
    // out_color = pow(vec4((nmap + 1.0) * 0.5, 1.0), vec4(2.2 / 1.0));
    // out_color = vec4(uv.xy, 0.0, 1.0);
    // out_color = vec4(uv.zw, 0.0, 1.0);
    // out_color = base_color;
    // out_color = vec4(ibl_specular_color, 1.0);
    // out_color = vec4(diffuse_color, 1.0);
}
