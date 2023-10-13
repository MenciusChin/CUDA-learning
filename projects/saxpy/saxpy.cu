// SAXPY example By Mark Harris

#include <stdio.h>
#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <vector>
#include <utility>   // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream>   // std::stringstream
#include <ctime>
using namespace std;

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

__host__ void copyFromHostToDevice(float *x, float *y, float *d_x, float *d_y, int N)
{
    size_t size = N * sizeof(float);

    cudaError_t err = cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy x from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpyToSymbol(d_y, y, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy y from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(float *d_x, float *d_y, int N)
{
    // Launch the search CUDA Kernel
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Report max error
__host__ void maxError(int N, float *y)
{
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);
}

// Retrieve results
__host__ void copyFromDeviceToHost(float *y, float *d_y, int N)
{
    std::cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = N * sizeof(int);

    cudaError_t err = cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_i from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free memory
__host__ void deallocateMemory(float *x, float *y, float *d_x, float *d_y)
{
    cudaError_t err = cudaFree(d_x);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device float variable d_x (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_y);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device float variable d_y (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(x);
    free(y);
}

__host__ void cleanUpDevice()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
    copyFromHostToDevice(x, y, d_x, d_y, N);

    // Perform SAXPY on 1M elements
    executeKernel(d_x, d_y, N);
    copyFromDeviceToHost(y, d_y, N);

    maxError(N, y);

    deallocateMemory(x, y, d_x, d_y);
    cleanUpDevice();
}
