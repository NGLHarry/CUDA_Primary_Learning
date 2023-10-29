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

__global__ void float_point_test()
{
    float a = 3.1415927f;
    float b = 3.1415928f;

    // double a = 3.1415927;
    // double b = 3.1415928;
    if(a == b)
    {
        printf("a is equal to b\n");
    }
    else
    {
        printf("a is not equal to b\n");
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
    
    dim3 block(1);
    dim3 grid(1);
    float_point_test<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}
