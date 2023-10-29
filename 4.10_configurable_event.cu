#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>
#include <ctime>


__global__ void infiniteKernel()
{
    while(true)
    {
        
    }
}

int main(int argc, char** argv)
{
    // get GPU device Count
    int nDeviceNumber = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("No CUDA capatable GPU found\n");
        return -1;
    }

    // set up device
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

    // calculate on GPU
    dim3 block(nElem);
    dim3 grid(1);  

    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream);
    infiniteKernel<<<grid, block, 0, kernel_stream>>>();

    cudaEvent_t kernel_event;
    ErrorCheck(cudaEventCreateWithFlags(&kernel_event, cudaEventDefault), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(kernel_event, kernel_stream), __FILE__, __LINE__);



    // wait for data copy to complete
    cudaEventSynchronize(kernel_event);
    printf("Event cp_evt is finished\n");
    cudaDeviceSynchronize();


    cudaStreamDestroy(kernel_stream);
    cudaEventDestroy(kernel_event);


    cudaDeviceReset();

    return 0;
}