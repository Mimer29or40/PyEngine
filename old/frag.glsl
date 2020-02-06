#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCords;

struct Material
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    vec3 Tf;
    int illum;
    float d;
    float Ns;
    float sharpness;
    float Ni;

    bool use_map_Ke;
    bool use_bump;

    sampler2D map_Ka;
    sampler2D map_Kd;
    sampler2D map_Ks;
    sampler2D map_Ke;
    sampler2D map_d;
    sampler2D map_Ns;

    sampler2D decal;
    sampler2D disp;
    sampler2D bump;
    sampler2D refl;
};

uniform Material material;

void main()
{
   FragColor = vec4(FragPos, 1.0);
   FragColor = vec4(material.Ka, 1.0);
   // FragColor = vec4(vec3(TexCords, 1.0), 1.0);
}
