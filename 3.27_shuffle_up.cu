#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"


__global__ void shfl_up(int *in, int *out, int const srcLane)
{
   int value = in[threadIdx.x];
   value = __shfl_up(value, srcLane, 16);
   out[threadIdx.x] = value;
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

    // set up data size of vectors
    int nElem = 32;
    // malloc host memory
    int *in = NULL;
    int *out = NULL;
    ErrorCheck(cudaHostAlloc((void **)&in, sizeof(int) * nElem, cudaHostAllocDefault), __FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((void **)&out, sizeof(int) * nElem, cudaHostAllocDefault), __FILE__, __LINE__);
    for(int i = 0;i < nElem; ++i)
    {
        in[i] = i;
    }

    dim3 block(nElem);
    dim3 grid(1);
    shfl_up<<<grid, block>>>(in, out, 3);

    cudaDeviceSynchronize();

    for(int i =0;i < nElem; ++i)
    {
        printf("out element is ,id=%d, value=%d\n",i, out[i]);
    }

    cudaFreeHost(in);
    cudaFreeHost(out);

    cudaDeviceReset();


    return 0;
}