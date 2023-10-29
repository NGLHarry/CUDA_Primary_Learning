#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"


__shared__ float g_shared;

__global__ void kernel_1()
{
    __shared__ float k1_shared;
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(blockIdx.x == 0 && id == 0)
    {
        k1_shared = 15.0;
    }

    if(blockIdx.x == 1 && id == 16)
    {
        k1_shared = 6.0;
        
    }
    __syncthreads();
    printf("access local shred in kernel_1, k1_shared=%.2f, blockIdx = %d, threadIdx = %d, threadID = %d\n",
        k1_shared, blockIdx.x, threadIdx.x, id);
}


__global__ void kernel_2()
{
    g_shared = 1.0;
    printf("access global shared in kernel_2, g_shared=%.2f\n", g_shared);
    // printf("access local shared in kernel_2, k_shared=%.2f\n", k1_shared);

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
    dim3 block(16);
    dim3 grid(2);

    kernel_1<<<grid, block>>>();
    kernel_2<<<grid, block>>>();


    

    cudaDeviceReset();


    return 0;
}