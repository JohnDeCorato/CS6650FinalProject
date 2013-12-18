#ifndef MAIN_H
#define MAIN_H

#include <gl\glew.h>
#include <GL/glut.h>

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "kernel.h"
#include "utilities.h"
#include "camera.h"

#if CUDA_VERSION >= 5000
    #include <helper_cuda.h>
    #include <helper_cuda_gl.h>
    #define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
    #include <cutil_inline.h>
    #include <cutil_gl_inline.h>
    #define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint indexLocation = 2;
const char *attributeLocations[] = { "Position", "Texcoords", "Index"};
GLuint planeVBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint planetVBO = (GLuint)NULL;
GLuint planetInBO = (GLuint)NULL;
GLuint planetIBO = (GLuint)NULL;
GLuint targetVBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[2];

const unsigned int HEIGHT_FIELD = 0;
const unsigned int PASS_THROUGH = 1;

const int field_width  = 2000;
const int field_height = 2000;

float fovy = 60.0f;
float zNear = 0.10;
float zFar = 50.0;

glm::mat4 projection;
glm::mat4 view;
Camera camera = Camera();

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1000; int height=1000;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

void display();
void keyboard(unsigned char key, int x, int y);

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init(int argc, char* argv[]);


void initCuda();
void initTextures();
void initVAO();
void initShaders(GLuint * program);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif
