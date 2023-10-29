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

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N)
    {
        C[i] = A[i] + B[i];
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

    // get the supported priority on this device
    int lowPriority = 0;
    int highPriority = 0;
    cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
    printf("Priority Range is from %d to %d\n", lowPriority, highPriority);

    // set up data size of vectors
    int nElem = 1 << 24;

    // malloc host memory
    float *pinned_A, *pinned_B, *h_C;
    size_t nBytes = nElem * sizeof(float);

    ErrorCheck(cudaHostAlloc((void **)&pinned_A, nBytes, cudaHostAllocDefault),__FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((void **)&pinned_B, nBytes, cudaHostAllocDefault),__FILE__, __LINE__);
    h_C = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(pinned_A, nElem);
    initialData(pinned_B, nElem);
    memset(h_C, 0, nBytes);

    // allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nBytes);
    cudaMalloc(&d_B,nBytes);
    cudaMalloc(&d_C,nBytes);
    if(NULL == d_A && NULL == d_B && NULL == d_C)
    {
        printf("fail to allocate memroy for GPU\n");
        free(pinned_A);
        free(pinned_B);
        free(h_C);
        return -1;
    }
    else
    {
        printf("successfully allocate memory for GPU\n");
    }

    // transfer data from host to device
    cudaStream_t data_stream;
    cudaStreamCreate(&data_stream);

    cudaMemcpyAsync(d_A, pinned_A, nBytes, cudaMemcpyHostToDevice, data_stream);
    cudaMemcpyAsync(d_B, pinned_B, nBytes, cudaMemcpyHostToDevice, data_stream);
    cudaStreamSynchronize(data_stream);

    // calculate on GPU

    dim3 block(512);
    dim3 grid((nElem + block.x - 1)/block.x, 1);

    printf("Execution configure <<<%d, %d>>>, total element:%d\n",grid.x, block.x, nElem);
    double dTime_Begin = GetCPUSecond();
    cudaStreamCreateWithPriority(&data_stream, cudaStreamDefault, highPriority);
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);

    double dTime_End = GetCPUSecond();

    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i = nElem-1; i > nElem - 50; i--)
    {
        printf("matrix_A:%.2f, matrix_B:%.2f, result = %.2f\n",pinned_A[i], pinned_B[i], h_C[i]);
    }

    printf("Element Size:%d, Matrix add time Elapse is:%.5f\n",nElem, dTime_End-dTime_Begin);
    cudaFreeHost(pinned_A);
    cudaFreeHost(pinned_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(data_stream);
    cudaDeviceReset();

    return 0;
}