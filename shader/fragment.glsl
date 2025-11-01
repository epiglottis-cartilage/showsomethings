#version 460

layout(location = 0) out vec4 f_color;
layout(location = 1) in vec3 position;

void main() {
    vec3 position_uniform = position*0.5 + 0.5;
    float radius = length(position.xy);
    vec4 fragColor = vec4(position_uniform, radius);
    f_color = fragColor;
}
