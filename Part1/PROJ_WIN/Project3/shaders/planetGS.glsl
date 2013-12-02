#version 330

uniform mat4 u_projMatrix;

layout (points) in;
layout (points) out;
layout (max_vertices = 1) out;

in Data {
    vec4 vColor;
} DataIn[];
 
 
out Data {
    vec4 vColor;
} DataOut;

void main()
{
    vec3 Position = gl_in[0].gl_Position.xyz;
    gl_Position = u_projMatrix * vec4(Position, 1.0);
	DataOut.vColor = DataIn[0].vColor;
    EmitVertex();
    EndPrimitive();
}
