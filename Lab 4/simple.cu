// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const int N = 16;
const int blocksize = 16;

__global__
void simple(float *c)
{
	c[threadIdx.x] = threadIdx.x;
}

__global__
void simple_sqrt(float *c)
{
	c[threadIdx.x] = sqrtf(threadIdx.x);	// Calculate the square root of the thread index as floats
}

int main()
{
	float *c = new float[N];
	float *cd;
	const int size = N*sizeof(float);	// Allocate memory for an array of floats of size N

	cudaMalloc( (void**)&cd, size );
	dim3 dimBlock( blocksize, 1 );		// QUESTION 1: Here we create a block with 16 threads --> We use 16 threads/cores on 1 SM
	dim3 dimGrid( 1, 1 );							// QUESTION 1: Here we see that the grid is 1 x 1 --> We have one block
	simple_sqrt<<<dimGrid, dimBlock>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );
	cudaFree( cd );

	printf("\n\n*--------SQUARE ROOT USING GPU: --------*\n");
	for (int i = 0; i < N; i++)
		printf("%.10f ", c[i]);
	printf("\n*---------------------------------------*");

	printf("\n\n*--------SQUARE ROOT USING CPU: --------*\n");
	for(int i = 0; i < N; i++) {
		c[i] = sqrtf(i);
		printf("%.10f ", c[i]);
	}
	printf("\n*---------------------------------------*");

	delete[] c;
	printf("\ndone\n");
	return EXIT_SUCCESS;
}
