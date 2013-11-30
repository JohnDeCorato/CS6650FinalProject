#version 330

in vec4 Position;

out vec4 vColor;

void main(void)
{
	gl_FrontColor = vec4(1,1,1,1);
	vColor = vec4(1.0);

	gl_Position = Position;
}
