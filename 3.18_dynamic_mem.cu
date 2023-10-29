#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"


extern __shared__ int dynamic_array[];

__global__ void dynamic_shared_mem()
{
   dynamic_array[threadIdx.x] = threadIdx.x;
   printf("access dynamic_array in kernel, dynamic_array[%d]=%d\n",threadIdx.x, dynamic_array[threadIdx.x]);
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
    dim3 grid(1);

    dynamic_shared_mem<<<grid, block, sizeof(int) * block.x>>>();

    cudaDeviceReset();


    return 0;
}