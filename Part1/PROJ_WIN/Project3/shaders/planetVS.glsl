uniform int sideLen;

in vec4 Position;
in int Index;


out Data{
 vec4 vColor;
} DataOut;

void main(void)
{
	if (Position.w == 0)
	{
		DataOut.vColor = vec4(1,0,0,1);
	}
	else
	{
		DataOut.vColor = vec4(0,1,0,1);
	}

	gl_Position = vec4(Position.xyz,1);
}
