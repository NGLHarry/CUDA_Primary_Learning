#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ float factor = 3.2;

__global__ void globalMemory(float *out)
{
    printf("Get constant memory:%.2f\n", factor);
    *out = factor;
}

int main(int argc, char*argv[])
{
    int nDeviceNumber = 0;
    int error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("NO Cuda campatable GPU found\n");
        return -1;
    }

    // set up device
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("Fail to set GPU 0 for computing\n");
        return -1;
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }
    dim3 block(1,1);
    dim3 grid(1,1);

    float *d_A;
    float h_A;

    cudaMalloc((void **)&d_A, sizeof(float));

    globalMemory<<<grid, block>>>(d_A);
    cudaMemcpy(&h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    printf("host memory:%.2f\n", h_A);
    cudaDeviceSynchronize();

    // reset device
    cudaFree(d_A);
    cudaDeviceReset();

    return 0;
}