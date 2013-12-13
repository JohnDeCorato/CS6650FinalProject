#include "camera.h"

#include <math.h>

#define SQR(x) (x*x)

#define NULL_VECTOR vec3(0.0f,0.0f,0.0f)

Camera::Camera()
{
	//Init with standard OGL values:
	Position = vec3 (0.0, 0.0,	10.0);
	ViewDir = vec3( 0.0, 0.0, -1.0);
	RightVector = vec3 (1.0, 0.0, 0.0);
	UpVector = vec3 (0.0, 1.0, 0.0);

	//Only to be sure:
	RotatedX = RotatedY = RotatedZ = 0.0;
}

void Camera::Move (vec3 Direction)
{
	Position = Position + Direction;
}

void Camera::RotateX (double Angle)
{
	RotatedX += Angle;
	
	//Rotate viewdir around the right vector:
	ViewDir = ViewDir*cosf(Angle*PIdiv180) + UpVector*sinf(Angle*PIdiv180);

	//now compute the new UpVector (by cross product)
	UpVector = cross(ViewDir, RightVector)*-1.0f;

	
}

void Camera::RotateY (double Angle)
{
	RotatedY += Angle;
	
	//Rotate viewdir around the up vector:
	ViewDir = normalize(ViewDir*cosf(Angle*PIdiv180)
								- RightVector*sinf(Angle*PIdiv180));

	//now compute the new RightVector (by cross product)
	RightVector = cross(ViewDir, UpVector);
}

void Camera::RotateZ (double Angle)
{
	RotatedZ += Angle;
	
	//Rotate viewdir around the right vector:
	RightVector = normalize(RightVector*cosf(Angle*PIdiv180)
								+ UpVector*sinf(Angle*PIdiv180));

	//now compute the new UpVector (by cross product)
	UpVector = cross(ViewDir, RightVector)*-1.0f;
}

mat4 Camera::getViewMatrix( void )
{

	//The point at which the camera looks:
	vec3 ViewPoint = Position+ViewDir;

	//as we know the up vector, we can easily use gluLookAt:
	return lookAt(	Position,ViewPoint,UpVector);

}

void Camera::MoveForward( float Distance )
{
	Position = Position + (ViewDir*-Distance);
}

void Camera::StrafeRight ( float Distance )
{
	Position = Position + (RightVector*Distance);
}

void Camera::MoveUpward( float Distance )
{
	Position = Position + (UpVector*Distance);
}