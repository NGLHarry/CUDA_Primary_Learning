#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

__managed__ float y = 9.0;
__global__ void unifiedMemory(float *A)
{
    *A += y;
    printf("GPU unifed memory:%.2f\n", *A);
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

    // check whether to support unified memory
    int supportManagedMemory = 0;
    // cudaDevAttrManagedMemory: 1 if device supports allocating managed memory, 0 if not 
    ErrorCheck(cudaDeviceGetAttribute(&supportManagedMemory, cudaDevAttrManagedMemory, dev), __FILE__, __LINE__);

    if(0 == supportManagedMemory)
    {
        printf("allocate managed memory is not support\n");
        return -1;
    }

    printf("unified memory model is supported\n");

    // calculate on GPU
    dim3 block(1);
    dim3 grid(1);
    float *unified_mem = NULL;
    ErrorCheck(cudaMallocManaged((void **)&unified_mem, sizeof(float), cudaMemAttachGlobal), __FILE__, __LINE__);

    *unified_mem = 6.4;
    unifiedMemory<<<grid, block>>>(unified_mem);
    cudaDeviceSynchronize();
    printf("CPU unified memory:%.2f\n", *unified_mem);
    cudaDeviceReset();


    return 0;
}