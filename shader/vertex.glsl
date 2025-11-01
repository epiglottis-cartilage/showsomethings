#version 460
layout(location = 0) in vec3 position;
layout(location = 1) out vec3 position_out;

void main() {
    gl_Position = vec4(position, 1.0);
    position_out = position;
}
