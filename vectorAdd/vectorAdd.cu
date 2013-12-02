#include <iostream>
#include <cuda_runtime.h>
#include "common\book.h"
#include "common\cpu_bitmap.h"
#include <time.h>
#include <stdlib.h>

#define DIM 400
#define BLOCK_SIDE_SIZE 16  // the x and y dimension of a single block in threads

void printMat(float2 *pos_mat, int w, int h);
void checkCUDAError(const char *msg);
dim3 calcGridThreadDimensions(int w, int h);

// A device memory boolean that we use to tell if a bubble pass swaps any elements
__device__ bool d_swappedAny = false;

__global__ void add( int *a, int *b, int *c ) {
	int tid = blockIdx.x;

	if (tid < DIM)
		 c[tid] = a[tid] + b[tid];
}

__global__ void setRed(unsigned char *img ) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < DIM*DIM) {
		img[tid*4] = (char)threadIdx.x;
	}
}


__device__ __host__
int getMatOffset(int numAgents, int col, int row) {
	return col * numAgents + row;
}


__device__ bool d_temp_swapped = false;
/**
 *  This kernel takes two agent cells and flips them horizontally if a2.x < a1.x
 *  Returns:   dirty:  true if any 
 *			   
 */
__global__ 
	void iterateBubble(bool horizontal, bool odd, int numAgents, float2 *d_pos_mat, float2 *out_d_pos_mat)
{
	// Get the column for the agent
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int col = tidx * 2;
	if (odd && horizontal) col += 1;

	// Get the row for the agent
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (odd && !horizontal) row += 1;

	//  numAgents - 1 because we look ahead +1 for the swap
	if ((horizontal && col < numAgents-1 && row < numAgents) ||
		(!horizontal && col < numAgents && row < numAgents-1)) {

		int agent1Off = getMatOffset(numAgents, col, row);
		int agent2Off = horizontal ? getMatOffset(numAgents, col+1, row) : getMatOffset(numAgents, col, row+1);

		float agent1Value = horizontal ? d_pos_mat[agent1Off].x : d_pos_mat[agent1Off].y;
		float agent2Value = horizontal ? d_pos_mat[agent2Off].x : d_pos_mat[agent2Off].y;

		// Swap them if needed
		if (agent2Value < agent1Value) {
			out_d_pos_mat[agent1Off] = d_pos_mat[agent2Off];
			out_d_pos_mat[agent2Off] = d_pos_mat[agent1Off];

			d_swappedAny = true;
			d_temp_swapped = true;
		}
	}
}


void runBubbleKernel(bool horizontal, bool odd, int numAgents, float2 *d_pos_mat, float2 *out_d_pos_mat)
{	
	bool swapped = false;
	
	cudaMemcpyToSymbol(d_temp_swapped, &swapped, sizeof(bool), 0, cudaMemcpyHostToDevice);

	// Run the sort kernel for one bubble pass
	dim3 gridDims = calcGridThreadDimensions(numAgents, numAgents);
	dim3 blockDims = dim3(BLOCK_SIDE_SIZE, BLOCK_SIDE_SIZE);

	iterateBubble<<<gridDims, blockDims>>>(horizontal, odd, numAgents, d_pos_mat, out_d_pos_mat);
	cudaThreadSynchronize();
	cudaMemcpy(d_pos_mat, out_d_pos_mat, numAgents*numAgents*sizeof(float2), cudaMemcpyDeviceToDevice);
	
	cudaMemcpyFromSymbol(&swapped, d_temp_swapped, sizeof(bool), 0, cudaMemcpyDeviceToHost);
	/// Pull the results back to the host and print
	if (swapped && false) {
		float2 *result_matrix = (float2 *)malloc(sizeof(float2) * (numAgents*numAgents));
		cudaMemcpy(result_matrix, d_pos_mat, numAgents*numAgents*sizeof(float2), cudaMemcpyDeviceToHost);
		printMat(result_matrix, numAgents, numAgents);
		printf("\n");
	}
}

int main( void ) {
	
	int N = 100;

	// Create some matrix
	float2 *matrix = (float2 *)malloc(sizeof(float2) * (N*N));

	// Initialize the matrix
	srand(time(NULL));
	for (int i = 0; i < N*N; i++) {
		float2 data;
		data.x = (float)(rand() % N*N);
		data.y = (float)(rand() % N*N);
		matrix[i] = data;
	}

	printf("Starting matrix\n");
	printMat(matrix, N, N);

	// Allocate input matrix on device
	float2 *d_matrix;
	cudaMalloc((void**)&d_matrix, sizeof(float2) * (N*N));
	checkCUDAError("Malloc failed");

	// Allocate result matrix on device
	float2 *d_result_matrix;
	cudaMalloc((void**)&d_result_matrix, sizeof(float2) * (N*N));
	checkCUDAError("Malloc failed");

	// copy the matrix over to device memory
	cudaMemcpy(d_matrix, matrix, N*N*sizeof(float2), cudaMemcpyHostToDevice);
	checkCUDAError("initial memcpy failed");

	// copy the matrix into the result because we only write on swap
	cudaMemcpy(d_result_matrix, matrix, N*N*sizeof(float2), cudaMemcpyHostToDevice);
	checkCUDAError("initial memcpy failed");

	
	bool swappedAny;
	do {
		swappedAny = false;
		cudaMemcpyToSymbol(d_swappedAny, &swappedAny, sizeof(bool), 0, cudaMemcpyHostToDevice);
		
		// iterate bubble sort on odd and even columns and odd and even rows
		runBubbleKernel(true, false, N, d_matrix, d_result_matrix);
		runBubbleKernel(true, true, N, d_matrix, d_result_matrix);
		runBubbleKernel(false, false, N, d_matrix, d_result_matrix);
		runBubbleKernel(false, true, N, d_matrix, d_result_matrix);

		// iterate the bubbling again if we made any swaps (otherwise, it's correct)
		cudaMemcpyFromSymbol(&swappedAny, d_swappedAny, sizeof(bool), 0, cudaMemcpyDeviceToHost);
		
		checkCUDAError("Kernel failed");
	} while(swappedAny == true);



	/// Pull the results back to the host and print
	float2 *result_matrix = (float2 *)malloc(sizeof(float2) * (N*N));
	cudaMemcpy(result_matrix, d_result_matrix, N*N*sizeof(float2), cudaMemcpyDeviceToHost);
	checkCUDAError("Memcpy failed");

	printMat(result_matrix, N, N);

	cudaDeviceReset();
	
	return 0;


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


void printMat(float2 *pos_mat, int w, int h) {
	for (int j = 0; j < h; j++) {  // for each row
		for (int i = 0; i < w; i++) {  // for each column
			float2 item = pos_mat[getMatOffset(h, i, j)];
			printf("(%d, %d) ", (int)item.x, (int)item.y);
		}
		printf("\n");
	}
}

		
/*
int main( void ) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();

	unsigned char *d_im_ptr;
	
	// Allocate some memory for the device pixels
	if (cudaMalloc((void**)&d_im_ptr, 4*DIM*DIM*sizeof(char)) != cudaSuccess) {
		printf("Could not allocate device memory\n");
	}

	cudaMemset(ptr, 0, DIM*DIM*4);
	cudaMemcpy(d_im_ptr, ptr, DIM*DIM*4, cudaMemcpyDefault);
	
	

	setRed<<<(DIM * DIM) / 256 + 1, 256>>>(d_im_ptr);

	cudaMemcpy(ptr, d_im_ptr, DIM*DIM*4, cudaMemcpyDeviceToHost);

	//kernel(ptr);

	bitmap.display_and_exit();

	//TODO:  delete pixel memory
	
	return 0;


}
*/
/*

	
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c; 

	HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(int)));
	HANDLE_ERROR( cudaMalloc((void**)&dev_b, N*sizeof(int)));
	HANDLE_ERROR( cudaMalloc((void**)&dev_c, N*sizeof(int)));

	for(int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i*2;
	}

	HANDLE_ERROR( cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR( cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

	add<<<N, 1>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR( cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) {
		printf("%i\n", c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

*/


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 