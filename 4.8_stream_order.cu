#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>
#include <ctime>

// * 设置GPU设备
// * 初始化矩阵
// * 定义CUDA内核
// * 分配GPU内存
// * 将数据传入GPU内存并计算
// * 在主机中获取计算结果

void initialData(float* ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() % 0xFF) / 10.0f;
    } 
    printf("\n");
    return;
}

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

    // malloc host memory
    float *pinned_A;
    size_t nBytes = nElem * sizeof(float);

    ErrorCheck(cudaHostAlloc((void **)&pinned_A, nBytes, cudaHostAllocDefault),__FILE__, __LINE__);

    // initialize data at host side
    initialData(pinned_A, nElem);

    // allocate GPU memory
    float *d_A;
    cudaMalloc((float**)&d_A, nBytes);

    // calculate on GPU
    dim3 block(nElem);
    dim3 grid(2);  
    infiniteKernel<<<grid, block>>>();

    // transfer data from host to device
    cudaStream_t data_stream;
    // cudaStreamCreate(&data_stream); //blocking stream
    cudaStreamCreateWithFlags(&data_stream, cudaStreamNonBlocking); //non-blocking stream


    cudaMemcpyAsync(d_A, pinned_A, nBytes, cudaMemcpyHostToDevice, data_stream);

    cudaEvent_t cp_evt;
    ErrorCheck(cudaEventCreate(&cp_evt), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(cp_evt, data_stream), __FILE__, __LINE__);

    // wait for data copy to complete
    cudaEventSynchronize(cp_evt);
    printf("Event cp_evt is finished\n");
    cudaDeviceSynchronize();

    cudaFreeHost(pinned_A);

    cudaFree(d_A);

    cudaStreamDestroy(data_stream);
    cudaEventDestroy(cp_evt);


    cudaDeviceReset();

    return 0;
}