#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

__global__ void thread_barrier()
{
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   __shared__ float shared;
   shared = 0.0;
   if((id / 32) == 0)
   {
    __syncthreads();// __syncthreads避免在分支中使用
    shared = 5.0;
   }
   else
   {
    // while(shared == 0)
    // {

    // }
   }
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

    dim3 block(64);
    dim3 grid(1);
    thread_barrier<<<grid, block>>>();
    cudaDeviceSynchronize();
    

    cudaDeviceReset();


    return 0;
}