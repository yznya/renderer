#define TASK_WGSIZE 32
#define MESH_WGSIZE 32
#define TASK_WGLIMIT (1 << 22)

#define OPAQUE_PASS 0
#define MASK_PASS 1

struct Vertex
{
    vec4 p;
    vec4 n;
    vec4 uv;
    vec4 t;
};

struct Meshlet
{
    vec3 center;
    float radius;
    int8_t cone_axis[3];
    int8_t cone_cutoff;

    uint vertex_offset;
    uint triangle_offset;

    uint8_t triangle_count;
    uint8_t vertex_count;
};

struct MeshLod {
    uint meshlet_offset;
    uint meshlet_count;
    float error;
    uint _padding;
};

struct Mesh {
    vec3 center;
    float radius;
    uint vertex_offset;
    uint vertex_count;
    uint lod_count;
    uint _padding;
    MeshLod lods[8];
};

struct MeshDraw {
    uint pass_type;
    uint meshlet_visibility_offset;
    uint model_index;
    uint16_t mesh_index;
    uint16_t material_index;
};

struct Material {
    vec4 base_color_factor;
    vec4 metallic_roughness_normal_occlusion;
    vec4 emissive_factor_alpha_cutoff;
    vec4 specular_glossiness_factor;
    vec4 clearcoat_factor;
    vec4 sheen_color_roughness_factor;
    vec4 albedo_texture_transform;
    vec4 normal_texture_transform;
    vec4 metallic_roughness_texture_transform;
    vec4 emissive_texture_transform;
    vec4 occlusion_texture_transform;
    vec4 specular_factor;

    uint albedo_texture;
    uint albedo_texture_uv;
    uint normal_texture;
    uint normal_texture_uv;

    uint metallic_roughness_texture;
    uint metallic_roughness_texture_uv;
    uint emissive_texture;
    uint emissive_texture_uv;

    uint occlusion_texture;
    uint occlusion_texture_uv;
    uint clearcoat_texture;
    uint clearcoat_texture_uv;

    uint specular_color_texture;
    uint specular_color_texture_uv;
    uint specular_texture;
    uint specular_texture_uv;

    float ior;
    uint pass_type;
    uint material_type;
};

struct MeshTaskCommand
{
    uint draw_id;
    uint task_offset;
    uint task_count;
    uint late_draw_visibility;
    uint meshlet_visibility_offset;
    uint _padding[3];
};

struct UniformBuffer {
    mat4 view_proj;
    mat4 view;
    mat4 proj;
    vec3 camera_position;
    uint _padding;
    float near;
    float far;
    float depth_pyramid_width;
    float depth_pyramid_height;
    vec4 frustum;
    uint draw_count;
    uint lod_enabled;
    uint cull_enabled;
    float lod_target;
    float screen_width;
    float screen_height;
    vec4 sun_direction;
};

struct MeshTaskPayload {
    uint draw_index;
    uint meshlet_indices[TASK_WGSIZE];
};

const uint METAL_ROUGHNESS = 1;
const uint SPEC_GLOSS = (1 << 1);
const uint CLEARCOAT = (1 << 2);
const uint SHEEN = (1 << 3);
const uint SPEC = (1 << 4);
