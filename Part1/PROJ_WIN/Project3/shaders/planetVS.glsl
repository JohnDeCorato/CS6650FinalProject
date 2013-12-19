uniform int sideLen;
uniform bool matrixColoring;

in vec4 Position;
in int Index;


out Data{
 vec4 vColor;
} DataOut;

void main(void)
{
	
	if (matrixColoring) {
		DataOut.vColor = vec4((float)(Index/(int)sideLen)/(float)sideLen, (float)(Index%(int)sideLen)/(float)sideLen, 0, 1);//Index % (int)sideLen, 0, 1);
	} else {
		if (Index == 0)
			DataOut.vColor = vec4(1,0,0,1);
		else
			DataOut.vColor = vec4(0,1,0,1);
	}

	gl_Position = Position;
}
