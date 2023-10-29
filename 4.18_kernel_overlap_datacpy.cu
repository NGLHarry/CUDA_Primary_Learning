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
    printf("\n");
    return;
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        for(int i = 0; i < 9999; ++i)
            C[idx] = A[idx] + B[idx];
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
    int nElem = 1 << 18;

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *gpuRef;
    cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&gpuRef, nBytes, cudaHostAllocDefault);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(gpuRef, 0, nBytes);

    // allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nBytes);
    cudaMalloc(&d_B,nBytes);
    cudaMalloc(&d_C,nBytes);
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // calculate on GPU
    dim3 block(BDIM);
    dim3 grid((nElem + block.x - 1)/ block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];
    for(int i = 0; i < NSTREAM; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }
    cudaEventRecord(start, 0);

    for(int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_B[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
        sumArraysOnGPU<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
        cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float execution_time;
    cudaEventElapsedTime(&execution_time, start, stop);

    printf("\n");
    printf("Actual result from overlapped data transfers:\n");
    printf("overlap with %d streams:%f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6)/ execution_time);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(gpuRef);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for(int i = 0; i < NSTREAM; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaDeviceReset();

    return 0;
}
