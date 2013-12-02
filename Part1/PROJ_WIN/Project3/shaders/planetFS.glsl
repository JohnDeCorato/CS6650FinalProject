in Data{
 vec4 vColor;
} DataIn;

void main(void)
{
	gl_FragColor = DataIn.vColor;
}
