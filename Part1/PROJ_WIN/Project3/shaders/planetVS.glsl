uniform int sideLen;

in vec4 Position;
in int Index;


out Data{
 vec4 vColor;
} DataOut;

void main(void)
{
	DataOut.vColor = vec4((float)(Index/(int)sideLen)/(float)sideLen, (float)(Index%(int)sideLen)/(float)sideLen, 0, 1);//Index % (int)sideLen, 0, 1);

	gl_Position = Position;
}
