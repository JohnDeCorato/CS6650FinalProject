#ifndef CAMERA_H
#define CAMERA_H

#include <gl\glew.h>
#include <GL/glut.h>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

using namespace glm;

#define PI 3.1415926535897932384626433832795
#define PIdiv180 (PI/180.0)

class Camera
{
private:
	
	vec3 ViewDir;
	vec3 RightVector;	
	vec3 UpVector;
	

	double RotatedX, RotatedY, RotatedZ;	
	
public:
	vec3 Position;
	Camera();				//inits the values (Position: (0|0|0) Target: (0|0|-1) )
	mat4 getViewMatrix ( void );	//executes some glRotates and a glTranslate command
							//Note: You should call glLoadIdentity before using Render

	void Move ( vec3 Direction );
	void RotateX ( double Angle );
	void RotateY ( double Angle );
	void RotateZ ( double Angle );

	void MoveForward ( float Distance );
	void MoveUpward ( float Distance );
	void StrafeRight ( float Distance );


};

#endif