#version 460

layout(location = 0) out vec4 f_color;
layout(location = 1) in vec2 position;

void main() {
    vec2 position_uniform = position*0.5 + 0.5;
    float radius = length(position);
    vec4 fragColor = vec4( position_uniform, position_uniform.x, radius);
    f_color = fragColor;
}
