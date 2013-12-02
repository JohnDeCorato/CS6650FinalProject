in vec4 Position;
in int Index;

out Data{
 vec4 vColor;
} DataOut;

void main(void)
{
	float sideLen = 10.0;
	DataOut.vColor = vec4(Index/10.0, Index%10, 0, 1);

	gl_Position = Position;
}
