#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

__global__ void thread_block_fence()
{
   __shared__ float shared;
   shared = 0.0;

   int id = threadIdx.x + blockIdx.x * blockDim.x;
   if((id / 32) == 0 && id == 0)
   {
    shared = 5.0;
   }
   else if((id /32) != 0 && id == 32)
   {
    shared = 7.0;
   }
   __threadfence_block();
   printf("access local shred in thread_barrier,shared=%.2f, blockIdx = %d, threadIdx = %d, threadId = %d\n",
    shared, blockIdx.x, threadIdx.x, id);
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

    dim3 block(32);
    dim3 grid(2);
    thread_block_fence<<<grid, block>>>();
    cudaDeviceSynchronize();
    

    cudaDeviceReset();


    return 0;
}