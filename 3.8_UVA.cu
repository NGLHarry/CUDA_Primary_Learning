#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


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
    
    float *d_mem = NULL;
    ErrorCheck(cudaMalloc((void **)&d_mem, sizeof(d_mem)), __FILE__, __LINE__);

    cudaPointerAttributes pt_Attribute;
    ErrorCheck(cudaPointerGetAttributes(&pt_Attribute, d_mem), __FILE__, __LINE__);
    printf("Pionter Attribute:device = %d, devicePointer=%p, type=%d\n",
        pt_Attribute.device, pt_Attribute.devicePointer, pt_Attribute.type
    );
    cudaFree(d_mem);
    cudaDeviceReset();


    return 0;
}