#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


void printMatirx(int *C, int nx, int ny)
{
    for(int iy = 0; iy < ny; ++iy)
    {
        for(int ix = 0; ix < nx; ++ix)
        {
            printf("%3d\t", C[iy*nx + ix]);
        }
        printf("\n");
    }
}

__global__ void printThreadIndex(int *A, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx +ix;
    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %2d ival %2d blockDim (%d,%d)\n",
            threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx], blockDim.x, blockDim.y
        );
}

int main()
{
    int nDeviceNum = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNum),__FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNum == 0)
    {
        printf("fail to set GPU 0 for computing\n");
        return -1;
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // initialize host matrix with integer
    for(int i = 0; i < nxy; ++i)
    {
        h_A[i] = i;
    }
    printMatirx(h_A, nx, ny);

    // malloc device memory
    int *d_MatA;
    error = ErrorCheck(cudaMalloc((void **)&d_MatA, nBytes), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to allocate memory for GPU\n");
        free(h_A);
        return -1;
    }

    error = ErrorCheck(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to copy data from host to device\n");
        free(h_A);
        return -1;
    }
    dim3 block(4,2);
    // gridDim.x = (nx + block.x - 1) / block.x
    // gridDim.y = (ny + block.y - 1) / block.y
    dim3 grid((nx + block.x -1)/block.x, (ny + block.y - 1)/block.y);
    
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);

    // free host and device memory
    cudaFree(d_MatA);
    free(h_A);
    cudaDeviceReset();

    return 0;
}