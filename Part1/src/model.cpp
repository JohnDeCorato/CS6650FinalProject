#include "model.h"
#include "objloader.hpp"

using namespace std;

#define POSITION_LOCATION 0
#define TEX_COORD_LOCATION 1
#define NORMAL_LOCATION 3
#define WVP_LOCATION 4
#define WORLD_LOCATION 8

Model::Model()
{
	m_VAO = 0;
	ZERO_MEM(m_Buffers);
}

Model::~Model()
{
	Clear();
}

void Model::Clear()
{
    if (m_Buffers[0] != 0) {
        glDeleteBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);
    }
       
    if (m_VAO != 0) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
}

bool Model::LoadMesh(const char* path)
{
	Clear();

	glGenVertexArrays(1, &m_VAO);
	glBindVertexArray(m_VAO);

	glGenBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);

	bool Ret = false;

	vector<vec3> Positions;
	vector<vec3> Normals;
	vector<vec2> TexCoords;

	if (!loadOBJ(path, Positions, TexCoords, Normals)) {
		return false;
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Positions[0]) * Positions.size(), &Positions[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(TexCoords[0]) * TexCoords.size(), &TexCoords[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(TEX_COORD_LOCATION);
    glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

   	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Normals[0]) * Normals.size(), &Normals[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(NORMAL_LOCATION);
    glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[WVP_MAT_VB]);

	numVertices = Positions.size()/3;

	GLuint *Indices   = new GLuint[numVertices];

	for (int i = 0; i < numVertices; i++) {
		Indices[i] = i;
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices[0]) * numVertices, &Indices[0], GL_STATIC_DRAW);
    
    for (unsigned int i = 0; i < 4 ; i++) {
        glEnableVertexAttribArray(WVP_LOCATION + i);
        glVertexAttribPointer(WVP_LOCATION + i, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (const GLvoid*)(sizeof(GLfloat) * i * 4));
        glVertexAttribDivisorARB(WVP_LOCATION + i, 1);
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[WORLD_MAT_VB]);

    for (unsigned int i = 0; i < 4 ; i++) {
        glEnableVertexAttribArray(WORLD_LOCATION + i);
        glVertexAttribPointer(WORLD_LOCATION + i, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (const GLvoid*)(sizeof(GLfloat) * i * 4));
        glVertexAttribDivisorARB(WORLD_LOCATION + i, 1);
    }
    
    return GLCheckError();
}

void Model::Render(unsigned int NumInstances, const mat4* WVPMats, const mat4* WorldMats)
{
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[WVP_MAT_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * NumInstances, WVPMats, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[WORLD_MAT_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * NumInstances, WorldMats, GL_DYNAMIC_DRAW);

    glBindVertexArray(m_VAO);

	glDrawElementsInstancedBaseVertex(GL_TRIANGLES, numVertices, GL_UNSIGNED_INT, 0, NumInstances, 0);

	glBindVertexArray(0);
}
