// Matrix addition, GPU version
// nvcc matrix_gpu.cu -o matrix_gpu

#include <stdio.h>

const int N = 256;
const int numberOfThreads = 16;

__global__
void add_matrix(float *pA, float *pB, float *pC, int N)
{
  // Get thread indices
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  // Map the "2D thread strucutre" to the "1D matrix structure"
  int index = x + y * N;

  pC[index] = pA[index] + pB[index];
}

int main()
{

  // Declare and initialize event stuff
  float elapsedTime = 0;
  cudaEvent_t event1, event2;
  cudaEventCreate(&event1);
  cudaEventCreate(&event2);

  // Setup matrices
  float *a;
	float *b;
	float *c;
	a = (float *)malloc((N*N)*sizeof(float));
	b = (float *)malloc((N*N)*sizeof(float));
	c = (float *)malloc((N*N)*sizeof(float));

  float *pA, *pB, *pC;

  for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

  // CUDA stuff here
  cudaMalloc((void**)&pA, (N*N)*sizeof(float));
  cudaMalloc((void**)&pB, (N*N)*sizeof(float));
  cudaMalloc((void**)&pC, (N*N)*sizeof(float));

  cudaMemcpy(pA, a, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pB, b, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

  int blocksize = max(N / numberOfThreads, 1);
  dim3 blockDim(numberOfThreads, numberOfThreads);
  dim3 gridDim(blocksize, blocksize);

  // "Warm up the GPU kernel when measuring"
  add_matrix<<<gridDim, blockDim>>>(pA,pB,pC, N);
  add_matrix<<<gridDim, blockDim>>>(pA,pB,pC, N);
  add_matrix<<<gridDim, blockDim>>>(pA,pB,pC, N);
  add_matrix<<<gridDim, blockDim>>>(pA,pB,pC, N);

  cudaEventRecord(event1, 0);  // Insert event into CUDA stream (0 == default stream)
  add_matrix<<<gridDim, blockDim>>>(pA,pB,pC, N);
  cudaEventRecord(event2, 0);  // Insert event into CUDA stream (0 == default stream)

  cudaDeviceSynchronize();


  cudaThreadSynchronize();
  cudaMemcpy(c, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(pA);
  cudaFree(pB);
  cudaFree(pC);

  cudaEventSynchronize(event2);  // Make sure event has finished
  cudaEventElapsedTime(&elapsedTime, event1, event2);
  // Print stuff here
/*
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      printf("%0.2f ", c[i+j*N]);
    }
    printf("\n");
  }
*/
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
    printf("\n Error: %s\n",cudaGetErrorString(err));

  printf("\n*----------- BENCHMARKING -----------*");
  printf("\n\nCalculations on the GPU with block dimension = %ix%i and grid dimension = %i x %i ran in %f miliseconds \n\n", numberOfThreads, numberOfThreads, blocksize, blocksize, elapsedTime );
}
