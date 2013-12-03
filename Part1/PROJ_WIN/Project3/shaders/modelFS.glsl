#version 410

in vec2 TexCoord0;
in vec3 Normal0;
in vec3 WorldPos0;
flat in int InstanceID;

out vec4 FragColor;

main
{
    FragColor = vec4(1.0);
}