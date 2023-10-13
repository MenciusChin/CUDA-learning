/*
SAXPY example By Mark Harris
https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
*/

#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}


int main(void)
{
    // Initialize variables
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;

    // Allocate host memories
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    // Allocate device memories
    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Initialize device arrays
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    // Retrive result back to host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    // Cleaning up memories
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}