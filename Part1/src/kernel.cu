#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"
#include "const.h"
#include "device_functions.h"

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


__device__
glm::vec3 naiveSeparation(glm::vec4 us, glm::vec4 them)
{
	
	float dist = glm::length(us - them);
	float force_mag = (G * planetMass) / (dist * dist * dist + SOFTENING_FACTOR);
	glm::vec3 result_force(force_mag*(us-them));

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
		// bounds
		/*if (pos[index].x < 0) vel[index].x = abs(vel[index].x);
		if (pos[index].x > 1000) vel[index].x = -1*abs(vel[index].x);
		if (pos[index].y < 0) vel[index].y = abs(vel[index].y);
		if (pos[index].y > 1000) vel[index].y = -1*abs(vel[index].y);*/
		
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

/**
 *  Each thread takes care of a single body on body calculation.
 *  We make a grid of n x n of blocks/threads. Each .x is a new body
 *  each .y is a new calculation.
 */
__global__ 
void updateForces(int num_agents, int sideLen, float dt, glm::vec4 *d_pos, glm::vec3 *d_vel, glm::vec3 *d_acc) 
{
	 int this_index = threadIdx.x + (blockIdx.x * blockDim.x);
	 if (this_index >= num_agents) { return; }

	 int col = this_index / sideLen;
	 int row = this_index % sideLen;

	 // Sum up the average vel and pos of the ones around it
	 float2 avgVel;	 avgVel.x = 0; avgVel.y = 0;
	 int totalInChunk;

	 // Loop through the SHELL_NUM shells of agents around this body
	 for (int i=-SHELL_NUM; i <= SHELL_NUM; i++) {
		 for (int j=-SHELL_NUM; j <= SHELL_NUM; j++) {
			 int other_index = getMatOffset(sideLen, i+col, j+row);
			 if (other_index < 0 || other_index >= num_agents) continue;
			 
			 totalInChunk++;

			 avgVel.x += d_vel[other_index].x;
			 avgVel.y += d_vel[other_index].y;
			 glm::vec3 acc = naiveAcc(num_agents, d_pos[this_index], &d_pos[other_index]);
			 acc += 2.0f*naiveSeparation(d_pos[this_index], d_pos[other_index]);

			 d_acc[this_index].x += acc.x * dt;
			 d_acc[this_index].y += acc.y * dt;
			 d_acc[this_index].z += acc.z * dt;
		 }
	 }

	 avgVel.x /= totalInChunk;
	 avgVel.y /= totalInChunk;
	 
	 glm::vec3 aVelV = glm::vec3();
	 aVelV.x = avgVel.x;
	 aVelV.y = avgVel.y;

	 glm::vec3 targetVel = aVelV * .5f + d_acc[this_index] * 0.5f;
	 glm::vec3 finalVel = glm::normalize(d_vel[this_index]) * .9f + glm::normalize(targetVel) * 0.1f;

	 d_vel[this_index] = finalVel;
	 

      
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
	int col = tidx;
	if (horizontal) {
		col = tidx * 2;
		if (odd) col += 1;
	}

	// Get the row for the agent
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	int row = tidy;
	if (!horizontal) {
		row = tidy * 2;
		if (odd) row += 1;
	}

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

			//printf("\n(%f, %f) ==> (%f, %f)\n  (%f, %f) ==> (%f, %f)\n", d_pos_mat[agent1Off].x, d_pos_mat[agent1Off].y, out_d_pos_mat[agent1Off].x, out_d_pos_mat[agent1Off].y, 
			//	d_pos_mat[agent2Off].x, d_pos_mat[agent2Off].y, out_d_pos_mat[agent2Off].x, out_d_pos_mat[agent2Off].y);
			
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
	
	cudaMemcpy(d_pos_buffer, d_pos_mat, numAgents*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);

	// Writes all new positions into d_pos_buffer
	iterateBubble<<<gridDims, blockDims>>>(horizontal, odd, sideLen, d_pos_mat, d_vel_mat, d_pos_buffer);
	cudaThreadSynchronize();

	cudaMemcpy(d_pos_mat, d_pos_buffer, numAgents*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
	
}

void spacialSort(int N, glm::vec4 *d_pos, glm::vec3 *d_vel, glm::vec4 *d_pos_buffer, int max_iterations) {
	bool swappedAny;
	int count = 0;
	do {
		swappedAny = false;
		cudaMemcpyToSymbol(d_swappedAny, &swappedAny, sizeof(bool), 0, cudaMemcpyHostToDevice);
		
		// iterate bubble sort on odd and even columns and odd and even rows
		runBubbleKernel(true, false, N, d_pos, d_vel, d_pos_buffer);
		runBubbleKernel(true, true, N, d_pos, d_vel, d_pos_buffer);
		runBubbleKernel(false, false, N, d_pos, d_vel, d_pos_buffer);
		runBubbleKernel(false, true, N, d_pos, d_vel, d_pos_buffer);

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

//Initialize memory, update some globals
void initCuda(int N)
{
	if ((int)sqrt((float)N) != sqrt((float)N)) {
		printf("Number of agents must be a perfect square!!");
		exit(0);
	}

    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(BLOCK_SIZE)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&dev_pos_buffer, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, BLOCK_SIZE>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    generateCircularVelArray<<<fullBlocksPerGrid, BLOCK_SIZE>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");

	// copy the new positions into the position buffer as well
	cudaMemcpy(dev_pos_buffer, dev_pos, N*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
	checkCUDAErrorWithLine("initCuda: memcpy failed!");
    cudaThreadSynchronize();

	// Initialize the matrix as sorted
	spacialSort(N, dev_pos, dev_vel, dev_pos_buffer, -1);
}

void cudaNBodyUpdateWrapper(float dt)
{
	// Grid and block dimensions for a 1D kernel of all objects
	int gridLength = (int)ceil((float)numObjects / BLOCK_SIZE);
	int blockLength = BLOCK_SIZE;
	
	int sideLen = (int)sqrt((float)numObjects);

	// Clear the 2Darray of accelerations
	clearAccs<<<gridLength, blockLength>>>(numObjects, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaThreadSynchronize();
	cudaMemcpy(dev_pos_buffer, dev_pos, numObjects*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
	// Sort the 2Darray of positions spatially
	// Need to pass dev_pos and dev_vel because it swaps array elements around
	spacialSort(numObjects, dev_pos, dev_vel, dev_pos_buffer, 1);

	// Average forces from the 16 objects spacially around you
	updateForces<<<gridLength, blockLength>>>(numObjects, sideLen, dt, dev_pos, dev_vel, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();

	// Update positions of all particles!
	updateS<<<gridLength, blockLength>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
    checkCUDAErrorWithLine("Kernel failed!");

	// Update the position buffer with the new positions
	
    
	cudaThreadSynchronize();
	return;
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(BLOCK_SIZE)));
    sendToVBO<<<fullBlocksPerGrid, BLOCK_SIZE>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}