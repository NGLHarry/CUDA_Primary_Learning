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

#define NSTREAM 4
#define BDIM 128

void initialData(float* ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() % 0xFF) / 10.0f;
    } 
    return;
}

void data_cp_callback(cudaStream_t stream, cudaError_t status, void *userData)
{
    printf("data copy callback is invoked, datasize:%d\n", *(int*)userData);
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
    int nElem = 1 << 12;

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A;
    h_A = (float *)malloc(nBytes);
    
    initialData(h_A, nElem);
    // allocate GPU memory
    float *d_A;
    cudaMalloc((float **)&d_A, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    // register call back
    cudaStreamAddCallback(0, data_cp_callback, &nBytes, 0);


   
    free(h_A);
    cudaFree(d_A);
    cudaDeviceReset();

    return 0;
}
