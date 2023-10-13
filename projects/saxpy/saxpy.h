#include <stdio.h>
#include <tuple>
#include <bits/stdc++.h>
#include <string>
#include <fstream>
#include <vector>
#include <utility>   // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream>   // std::stringstream
#include <ctime>
using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")
__global__ void saxpy(int n, float a, float *x, float *y);

__host__ void copyFromHostToDevice(float *x, float *y, float *d_x, float *d_y, int N);
__host__ void executeKernel(float *d_x, float *d_y, int N);
__host__ void copyFromDeviceToHost(float *y, float *d_y, int N);
__host__ void maxError(int N, float *y);
__host__ void deallocateMemory(float *x, float *y, float *d_x, float *d_y);
__host__ void cleanUpDevice();