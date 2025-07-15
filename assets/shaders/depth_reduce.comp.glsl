#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D in_image;
layout(binding = 1, r32f) uniform writeonly image2D out_image;

layout(push_constant) uniform block {
    vec2 image_size;
};

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;

    float depth = texture(in_image, (vec2(pos) + vec2(0.5)) / image_size).x;

    imageStore(out_image, ivec2(pos), vec4(depth));
}
