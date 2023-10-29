#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


__global__ 
void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // convert global data pointer to  the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n)
        return ;

    // in-palace reduction in global memory 
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char *argv[])
{
    int nDeviceNum = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNum), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNum == 0)
    {
        printf("No Cuda campatable GPU found!\n");
        return -1;
    }

    // set up dev
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to set GPU 0 for campatable\n");
        return -1;
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }

    // set up data size
    int size = 1 << 24; //total number of elements
    printf("Data size:%d\n", size);

    int blockSize = 512;

    // set up execution configuration
    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1)/ block.x, 1);
    printf("Thread config: grid:<%d, %d>, block:<%d, %d>\n",
            grid.x, grid.y, block.x, block.y);

    // allocate host memory
    size_t nBytes = size * sizeof(int);
    int *h_idata = (int *)malloc(nBytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *temp    = (int *)malloc(nBytes);

    // initialize the array
    for(int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(temp, h_idata, nBytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate gpu memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, nBytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    iStart = GetCPUSecond();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = GetCPUSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    
    gpu_sum = 0;
    for(int i =0; i < grid.x; ++i)
    {
        gpu_sum += h_odata[i];
    }
       
    
    printf("GPU Neighbored elapsed %f sec gpu_sum:%d <<<grid:%d, block:%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    free(h_idata);
    free(h_odata);
    free(temp);
    // free gpu memory and reset device
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();
    return 0;
}