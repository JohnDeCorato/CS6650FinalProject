#ifndef MODEL_H
#define MODEL_H

#include <stdlib.h>
#include <string>
#include <vector>
#include <gl\glew.h>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "util.h"
using namespace glm;

class Model {
public:
	Model();
	~Model();

	bool LoadMesh(const char* path);
	 
    void Render(unsigned int NumInstances, const mat4* WVPMats, const mat4* WorldMats);

private:
    void Clear();

#define INDEX_BUFFER 0    
#define POS_VB       1
#define NORMAL_VB    2
#define TEXCOORD_VB  3    
#define WVP_MAT_VB   4
#define WORLD_MAT_VB 5

	int numVertices;
	GLuint m_VAO;
    GLuint m_Buffers[6];
};

#endif
