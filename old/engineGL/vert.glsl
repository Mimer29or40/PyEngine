#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCords;

out vec3 FragPos;
out vec2 TexCords;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    mat3 normalMatrix = transpose(inverse(mat3(model)));

    FragPos = vec3(model * vec4(aPos, 1.0));
    // FragPos = aPos;
    TexCords = aTexCords;

    gl_Position = projection * view * model * vec4(aPos, 1.0);
    // gl_Position = model * vec4(aPos, 1.0);
}
