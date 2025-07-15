
float grid_size = 1000.0;

float grid_cell_size = 0.125;
vec4 grid_color_thin = vec4(0.5, 0.5, 0.5, 1.0);
vec4 grid_color_thick = vec4(0.3, 0.3, 0.3, 1.0);
const float grid_min_pixels_between_cells = 2.0;

const vec3 pos[4] = vec3[4](
        vec3(-1.0, 0.0, -1.0),
        vec3(1.0, 0.0, -1.0),
        vec3(1.0, 0.0, 1.0),
        vec3(-1.0, 0.0, 1.0)
    );

const int indices[6] = int[6](
        0, 1, 2, 2, 3, 0
    );

float log10(float x)
{
    return log(x) / log(10.0);
}

float satf(float x)
{
    return clamp(x, 0.0, 1.0);
}

vec2 satv(vec2 x)
{
    return clamp(x, vec2(0.0), vec2(1.0));
}

float max2(vec2 v)
{
    return max(v.x, v.y);
}
