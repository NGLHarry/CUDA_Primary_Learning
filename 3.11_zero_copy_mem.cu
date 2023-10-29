#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

__global__ void zeroCopyMemory(float *A)
{
    printf("GPU page-locked memory:%.2f\n", *A);
}


int main(int argc, char* argv[])
{
    int nDeviceNumber = 0;
    cudaError error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("No CUDA Campatable GPU found\n");
        return -1;
    }
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {   
        printf("fail to set GPU 0 for computing\n");
        return -1;
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }


    // calculate on GPU
    dim3 block(1);
    dim3 grid(1);

    // 分配零拷贝内存
    float *h_zerorcpyMem = NULL;
    ErrorCheck(cudaHostAlloc((void **)&h_zerorcpyMem, sizeof(float), cudaHostAllocDefault), __FILE__, __LINE__);

    *h_zerorcpyMem = 4.8;
    printf("CPU page-locked memory:%.2f\n", *h_zerorcpyMem);

    zeroCopyMemory<<<grid, block>>>(h_zerorcpyMem);
    cudaDeviceSynchronize();

    cudaFreeHost(h_zerorcpyMem);

    cudaDeviceReset();


    return 0;
}