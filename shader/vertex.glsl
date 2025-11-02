#version 460
layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 normal;

//layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(location = 2) out vec3 position_out;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    // v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    position_out = gl_Position.xyz;
    // gl_Position = vec4(position_raw, 1.0);
}