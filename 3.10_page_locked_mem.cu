#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

__global__ void pageLockedMemory(float *A)
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

    // 分配页锁定内存
    float *h_PinnedMem = NULL;
    ErrorCheck(cudaMallocHost((void **)&h_PinnedMem, sizeof(float)), __FILE__, __LINE__);

    *h_PinnedMem = 4.8;
    printf("CPU page-locked memory:%.2f\n", *h_PinnedMem);

    pageLockedMemory<<<grid, block>>>(h_PinnedMem);
    cudaDeviceSynchronize();

    cudaFreeHost(h_PinnedMem);

    cudaDeviceReset();


    return 0;
}