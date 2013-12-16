#include "crowdGenerator.h"

void generatorTwoLines(int numBodiesPerRow, int numRows, float positions[], float targets[], int colorId[])
{
	for (int i = 0; i < numRows; i++) 
	{
		for (int j = 0; j < numBodiesPerRow; j++) 
		{
			int location = (i+1)*j;
			positions[8*location] = -0.2 - i * 0.2;
			positions[8*location] = 0.1 * (j - numBodiesPerRow/2);
			positions[8*location] = 0;
			positions[8*location] = 1;
			positions[8*location] = 0.2 + i * 0.2;
			positions[8*location] = 0.1 * (j - numBodiesPerRow/2);
			positions[8*location] = 0;
			positions[8*location] = 1;
			colorId[2*location] = 0;
			colorId[2*location+1] = 1;

			targets[8*location] = 0.2 + (numRows - i) * 0.2;
			targets[8*location] = 0.1 * (j - numBodiesPerRow/2);
			targets[8*location] = 0;
			targets[8*location] = 1;
			targets[8*location] = -0.2 - (numRows - i) * 0.2;
			targets[8*location] = 0.1 * (j - numBodiesPerRow/2);
			targets[8*location] = 0;
			targets[8*location] = 1;
		}
	}
}

void generateCross(float positions[], float targets[], int colorId[])
{

}