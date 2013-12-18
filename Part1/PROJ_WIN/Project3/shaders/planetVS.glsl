uniform int sideLen;

in vec4 Position;
in int Index;


out Data{
 vec4 vColor;
} DataOut;

void main(void)
{

	DataOut.vColor = vec4((float)(Index/(int)sideLen)/(float)sideLen, (float)(Index%(int)sideLen)/(float)sideLen, 0, 1);//Index % (int)sideLen, 0, 1);
	/*if (Position.z > 0) {
		DataOut.vColor = vec4(1,0,0,1);
	} else {
		DataOut.vColor = vec4(0,1,0,1);
	}*/
	
	gl_Position = Position;
}
