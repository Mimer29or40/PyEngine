#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCords;

void main()
{
   // FragColor = vec4(FragPos, 1.0);
   FragColor = vec4(vec3(TexCords, 1.0), 1.0);
}
