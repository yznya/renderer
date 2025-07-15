#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_KHR_shader_subgroup_ballot: require
#include "math.inc.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform samplerCube cubemap_texture;

layout(push_constant) uniform PerFrameData {
    uint face;
    float roughness;
    uint sample_count;
    uint width;
    uint height;
    uint distribution;
} per_frame_data;

const int LAMBERTIAN = 0;
const int GGX = 1;
const int CHARLIE = 2;

vec3 uv_to_xyz(uint face, vec2 uv)
{
    if (face == 0) return vec3(1., uv.y, -uv.x);
    if (face == 1) return vec3(-1., uv.y, uv.x);
    if (face == 2) return vec3(+uv.x, 1., -uv.y);
    if (face == 3) return vec3(+uv.x, -1., +uv.y);
    if (face == 4) return vec3(+uv.x, uv.y, 1.);
    if (face == 5) return vec3(-uv.x, +uv.y, -1.);
}

struct MicrofacetDistributionSample {
    float pdf;
    float cos_theta;
    float sin_theta;
    float phi;
};

float d_ggx(float n_dot_h, float roughness) {
    float a = n_dot_h * roughness;
    float k = roughness / (1.0 - n_dot_h * n_dot_h + a * a);
    return k * k * (1.0 / PI);
}

// ggx microfacet distribution
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html
// This implementation is based on https://bruop.github.io/ibl/,
//  https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html
// and https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
MicrofacetDistributionSample ggx(vec2 xi, float roughness) {
    MicrofacetDistributionSample ggx;

    // evaluate sampling equations
    float alpha = roughness * roughness;
    ggx.cos_theta = clamp(sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y)), 0.0, 1.0);
    ggx.sin_theta = sqrt(1.0 - ggx.cos_theta * ggx.cos_theta);
    ggx.phi = 2.0 * PI * xi.x;

    // evaluate ggx pdf (for half vector)
    ggx.pdf = d_ggx(ggx.cos_theta, alpha);

    // Apply the Jacobian to obtain a pdf that is parameterized by l see https://bruop.github.io/ibl/
    // Typically you'd have the following:
    //   float pdf = d_ggx(NoH, roughness) * NoH / (4.0 * VoH);
    // but since V = N => VoH == NoH
    ggx.pdf /= 4.0;

    return ggx;
}

// NDF
float d_charlie(float sheen_roughness, float NdotH) {
    sheen_roughness = max(sheen_roughness, 0.000001); //clamp (0,1]
    float invR = 1.0 / sheen_roughness;
    float cos2h = NdotH * NdotH;
    float sin2h = 1.0 - cos2h;
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

MicrofacetDistributionSample charlie(vec2 xi, float roughness) {
    MicrofacetDistributionSample charlie;

    float alpha = roughness * roughness;
    charlie.sin_theta = pow(xi.y, alpha / (2.0 * alpha + 1.0));
    charlie.cos_theta = sqrt(1.0 - charlie.sin_theta * charlie.sin_theta);
    charlie.phi = 2.0 * PI * xi.x;

    // evaluate charlie pdf (for half vector)
    charlie.pdf = d_charlie(alpha, charlie.cos_theta);

    // Apply the Jacobian to obtain a pdf that is parameterized by l
    charlie.pdf /= 4.0;

    return charlie;
}

MicrofacetDistributionSample lambertian(vec2 xi, float roughness) {
    MicrofacetDistributionSample lambertian;

    // Cosine weighted hemisphere sampling
    // http://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations.html#Cosine-WeightedHemisphereSampling
    lambertian.cos_theta = sqrt(1.0 - xi.y);
    lambertian.sin_theta = sqrt(xi.y); // equivalent to `sqrt(1.0 - cos_theta*cos_theta)`;
    lambertian.phi = 2.0 * PI * xi.x;

    lambertian.pdf = lambertian.cos_theta / PI; // evaluation for solid angle, therefore drop the sin_theta

    return lambertian;
}

// TBN generates a tangent bitangent normal coordinate frame from the normal (the normal must be normalized)
mat3 generate_tbn(vec3 normal) {
    vec3 bitangent = vec3(0.0, 1.0, 0.0);

    float n_dot_up = dot(normal, vec3(0.0, 1.0, 0.0));
    float epsilon = 0.0000001;
    if (1.0 - abs(n_dot_up) <= epsilon) {
        // Sampling +Y or -Y, so we need a more robust bitangent.
        bitangent = (n_dot_up > 0.0) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
    }

    vec3 tangent = normalize(cross(bitangent, normal));
    bitangent = cross(normal, tangent);

    return mat3(tangent, bitangent, normal);
}

// get_importance_sample returns an importance sample direction with pdf in the .w component
vec4 get_importance_sample(uint sample_index, vec3 n, float roughness) {
    // generate a quasi monte carlo point in the unit square [0.1)^2
    vec2 xi = hammersley2d(sample_index, per_frame_data.sample_count);

    MicrofacetDistributionSample importance_sample;

    // generate the points on the hemisphere with a fitting mapping for
    // the distribution (e.g. lambertian uses a cosine importance)
    if (per_frame_data.distribution == LAMBERTIAN) {
        importance_sample = lambertian(xi, roughness);
    }
    else if (per_frame_data.distribution == GGX) {
        // Trowbridge-Reitz / ggx microfacet model (Walter et al)
        // https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html
        importance_sample = ggx(xi, roughness);
    }
    else if (per_frame_data.distribution == CHARLIE) {
        importance_sample = charlie(xi, roughness);
    }

    // transform the hemisphere sample to the normal coordinate frame
    // i.e. rotate the hemisphere to the normal direction
    vec3 local_space_direction = normalize(vec3(
                importance_sample.sin_theta * cos(importance_sample.phi),
                importance_sample.sin_theta * sin(importance_sample.phi),
                importance_sample.cos_theta));
    mat3 tbn = generate_tbn(n);
    vec3 direction = tbn * local_space_direction;

    return vec4(direction, importance_sample.pdf);
}

// Mipmap Filtered Samples (GPU Gems 3, 20.4)
// https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
// https://cgg.mff.cuni.cz/~jaroslav/papers/2007-sketch-fis/Final_sap_0073.pdf
float compute_lod(float pdf) {
    // // Solid angle of current sample -- bigger for less likely samples
    // float omegaS = 1.0 / (float(per_frame_data.sample_count) * pdf);
    // // Solid angle of texel
    // // note: the factor of 4.0 * PI
    // float omegaP = 4.0 * PI / (6.0 * float(per_frame_data.width) * float(per_frame_data.width));
    // // Mip level is determined by the ratio of our sample's solid angle to a texel's solid angle
    // // note that 0.5 * log2 is equivalent to log4
    // float lod = 0.5 * log2(omegaS / omegaP);

    // babylon introduces a factor of K (=4) to the solid angle ratio
    // this helps to avoid undersampling the environment map
    // this does not appear in the original formulation by Jaroslav Krivanek and Mark Colbert
    // log4(4) == 1
    // lod += 1.0;

    // We achieved good results by using the original formulation from Krivanek & Colbert adapted to cubemaps

    // https://cgg.mff.cuni.cz/~jaroslav/papers/2007-sketch-fis/Final_sap_0073.pdf
    float width = float(per_frame_data.width);
    float height = float(per_frame_data.height);
    float sample_count = float(per_frame_data.sample_count);
    float lod = 0.5 * log2(6.0 * width * height / (sample_count * pdf));

    return lod;
}

vec3 filter_color(vec3 n) {
    vec3 color = vec3(0.f);
    float weights = 0.0f;

    for (uint i = 0; i < per_frame_data.sample_count; i++) {
        vec4 importance_sample = get_importance_sample(i, n, per_frame_data.roughness);

        vec3 h = vec3(importance_sample.xyz);
        float pdf = importance_sample.w;

        // mipmap filtered samples (GPU Gems 3, 20.4)
        float lod = compute_lod(pdf);

        if (per_frame_data.distribution == LAMBERTIAN) {
            // sample lambertian at a lower resolution to avoid fireflies
            vec3 lambertian = textureLod(cubemap_texture, h, lod).xyz;

            color += lambertian;
        }
        else if (per_frame_data.distribution == GGX || per_frame_data.distribution == CHARLIE) {
            // Note: reflect takes incident vector.
            vec3 v = n;
            vec3 l = normalize(reflect(-v, h));
            float n_dot_l = dot(n, l);

            if (n_dot_l > 0.0) {
                if (per_frame_data.roughness == 0.0) {
                    // without this the roughness=0 lod is too high
                    lod = 0.0;
                }
                vec3 sample_color = textureLod(cubemap_texture, l, lod).xyz;
                color += sample_color * n_dot_l;
                weights += n_dot_l;
            }
        }
    }

    color /= (weights != 0.0f) ? weights : float(per_frame_data.sample_count);

    return color.rgb;
}

void main() {
    vec2 new_uv = uv * 2.0 - 1.0;

    vec3 scan = uv_to_xyz(per_frame_data.face, new_uv);
    vec3 direction = normalize(scan);

    out_color = vec4(filter_color(direction), 1.0);
}
