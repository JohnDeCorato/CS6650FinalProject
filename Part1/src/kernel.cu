#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"
#include "const.h"
#include "device_functions.h"

//GLOBALS
dim3 threadsPerBlock(blockSize);

// A device memory boolean that we use to tell if a bubble pass swaps any elements
__device__ bool d_swappedAny = false;

#if SHARED == 1
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif

int numObjects;
const __constant__ float planetMass = PLANET_MASS;
const __device__ float starMass = STAR_MASS;
const float scene_scale = SCENE_SCALE; //size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;
glm::vec4 *dev_pos_buffer; // used to double buffer when sorting

void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 

__host__ __device__
unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__device__ __host__
int getMatOffset(int sideLen, int col, int row) {
	return col * sideLen + row;
}

/**
 *  x = gridWidth
 *  y = gridHeight
 *  z = 1
 */
dim3 calcGridThreadDimensions(int w, int h) {
	dim3 dims;
	dims.x = (int)ceil((float)w / (float)BLOCK_SIDE_SIZE);
	dims.y = (int)ceil((float)h / (float)BLOCK_SIDE_SIZE);
	return dims;
}
//Function that generates static.
__host__ __device__ 
glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Generate randomized starting positions for the planets in the XY plane
//Also initialized the masses
__global__
void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale, float mass)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.0f;//rand.z;
        arr[index].w = mass;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__
void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 R = glm::vec3(pos[index].x, pos[index].y, pos[index].z);
        float r = glm::length(R) + EPSILON;
        float s = sqrt(G*starMass/r);
        glm::vec3 D = glm::normalize(glm::cross(R/r,glm::vec3(0,0,1)));
        arr[index].x = s*D.x;
        arr[index].y = s*D.y;
        arr[index].z = s*D.z;
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index) - 0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.0;//rand.z;
    }
}

//TODO: Determine force between two bodies
__device__
glm::vec3 calculateAcceleration(glm::vec4 us, glm::vec4 them)
{
    //    G*m_us*m_them
    //F = -------------
    //         r^2
    //
    //        G*m_us*m_them   G*m_them
    //a     = ------------- = --------
    //         m_us*r^2        r^2

	/// What we're actually doing is :
	///   (G*m_them)/(r^2 + eps)  *  (them - us)/r    which ==>
	///   (G*m_them)/(r^3 + eps)  *  (them - us)
	///   where eps is some softening factor to avoid the blow up when particles collide

	
	float dist = glm::length(us - them);
	float force_mag = (G * planetMass) / (dist * dist + SOFTENING_FACTOR);
	glm::vec3 result_force(force_mag*(them-us));

    return result_force;
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 *their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, *their_pos);
    return acc;
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    return acc;
}

//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec4 my_pos;
    glm::vec3 accel;

    if(index < N) my_pos = pos[index];

    accel = ACC(N, my_pos, pos);

    if(index < N) acc[index] = accel;
}

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        vel[index]   += acc[index]   * dt;
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;
    }
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = 0;
        vbo[4*index+3] = 1;
    }
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_pos_buffer, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");

	// copy the new positions into the position buffer as well
	cudaMemcpy(dev_pos_buffer, dev_pos, N*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
	checkCUDAErrorWithLine("initCuda: memcpy failed!");
    cudaThreadSynchronize();
}


/**
 *  Each thread takes care of a single body on body calculation.
 *  We make a grid of n x n of blocks/threads. Each .x is a new body
 *  each .y is a new calculation.
 */
__global__ 
void updateForces(int num_agents, float dt, glm::vec4 *d_pos, glm::vec3 *d_acc) 
{
	 int this_index = threadIdx.x + (blockIdx.x * blockDim.x);
     int other_index = threadIdx.y + (blockIdx.y * blockDim.y);

	 if (this_index >= num_agents || other_index >= num_agents) { return; }

	 glm::vec3 acc = naiveAcc(num_agents, d_pos[this_index], &d_pos[other_index]);

	 atomicAdd(&d_acc[this_index].x, acc.x * dt);
	 atomicAdd(&d_acc[this_index].y, acc.y * dt);
	 atomicAdd(&d_acc[this_index].z, acc.z * dt);
      
}


/**
 *  This kernel takes two agent cells and flips them horizontally if a2.x < a1.x
 *  Returns:   dirty:  true if any 
 *			   
 */
__global__ 
	void iterateBubble(bool horizontal, bool odd, int sideLen, glm::vec4 *d_pos_mat, glm::vec3 *d_vel_mat, glm::vec4 *out_d_pos_mat)
{
	// Get the column for the agent
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int col = tidx * 2;
	if (odd) col += 1;

	// Get the row for the agent
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	//  sideLen - 1 because we look ahead +1 for the swap
	if ((horizontal && col < sideLen-1 && row < sideLen) ||
		(!horizontal && col < sideLen && row < sideLen-1)) {

		int agent1Off = getMatOffset(sideLen, col, row);
		int agent2Off = horizontal ? getMatOffset(sideLen, col+1, row) : getMatOffset(sideLen, col, row+1);

		float agent1Value = horizontal ? d_pos_mat[agent1Off].x : d_pos_mat[agent1Off].y;
		float agent2Value = horizontal ? d_pos_mat[agent2Off].x : d_pos_mat[agent2Off].y;

		// Swap them if needed
		if (agent2Value < agent1Value) {
			out_d_pos_mat[agent1Off] = d_pos_mat[agent2Off];
			out_d_pos_mat[agent2Off] = d_pos_mat[agent1Off];
			
			// swap velocities too
			glm::vec3 tempVel = d_vel_mat[agent1Off];
			d_vel_mat[agent1Off] = d_vel_mat[agent2Off];
			d_vel_mat[agent2Off] = tempVel;

			d_swappedAny = true;
		}
	}
}


void runBubbleKernel(bool horizontal, bool odd, int numAgents, glm::vec4 *d_pos_mat, glm::vec3 *d_vel_mat, glm::vec4 *d_pos_buffer)
{	
	// Run the sort kernel for one bubble pass
	int sideLen = (int)sqrt((float)numAgents);
	dim3 gridDims = calcGridThreadDimensions(sideLen, sideLen);
	dim3 blockDims = dim3(BLOCK_SIDE_SIZE, BLOCK_SIDE_SIZE);

	// Writes all new positions into d_pos_buffer
	iterateBubble<<<gridDims, blockDims>>>(horizontal, odd, sideLen, d_pos_mat, d_vel_mat, d_pos_buffer);
	cudaThreadSynchronize();

	cudaMemcpy(d_pos_mat, d_pos_buffer, numAgents*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
	// Swap dev_pos with d_pos_buffer
	//glm::vec4 *temp = d_pos_mat;
	//d_pos_mat = d_pos_buffer;
	//d_pos_buffer = temp;
	
}

void spacialSort(int N, glm::vec4 *d_pos, glm::vec3 *d_vel, glm::vec4 *d_pos_buffer, int max_iterations) {
	bool swappedAny;
	int count = 0;
	do {
		swappedAny = false;
		cudaMemcpyToSymbol(d_swappedAny, &swappedAny, sizeof(bool), 0, cudaMemcpyHostToDevice);
		
		// iterate bubble sort on odd and even columns and odd and even rows
		runBubbleKernel(true, false, N, d_pos, d_vel, d_pos_buffer);
		//runBubbleKernel(true, true, N, d_pos, d_vel, d_pos_buffer);
		//runBubbleKernel(false, false, N, d_pos, d_vel, d_pos_buffer);
		//runBubbleKernel(false, true, N, d_pos, d_vel, d_pos_buffer);

		// iterate the bubbling again if we made any swaps (otherwise, it's correct)
		cudaMemcpyFromSymbol(&swappedAny, d_swappedAny, sizeof(bool), 0, cudaMemcpyDeviceToHost);
		
		checkCUDAError("Kernel failed");
		count++;
	} while(swappedAny == true && (count < max_iterations || max_iterations == -1));

}
__global__
void clearAccs(int num_agents, glm::vec3 *d_acc)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	d_acc[index] = glm::vec3(0.0f);
}

void cudaNBodyUpdateWrapper(float dt)
{
	dim3 gridDims = calcGridThreadDimensions(numObjects, numObjects);
	dim3 blockDims = dim3(BLOCK_SIDE_SIZE, BLOCK_SIDE_SIZE);

	// Clear the 2Darray of accelerations
	clearAccs<<<gridDims, blockDims>>>(numObjects, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();

	// Sort the 2Darray of positions spatially
	// Need to pass dev_pos and dev_vel because it swaps array elements around
	spacialSort(numObjects, dev_pos, dev_vel, dev_pos_buffer, -1);

	// Average forces from the 16 objects spacially around you
	//updateForces<<<gridDims, blockDims>>>(numObjects, dt, dev_pos, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();

	// Update positions of all particles!
	updateS<<<gridDims, blockDims>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");

	// Update the position buffer with the new positions
	cudaMemcpy(dev_pos_buffer, dev_pos, numObjects*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
    
	cudaThreadSynchronize();
	return;
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}