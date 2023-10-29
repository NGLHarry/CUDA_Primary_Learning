#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;

    // convert global data pointer to  the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8; 

    // unrolling 8 data blocks
    if(idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];

        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    // in-palace reduction in global memory 
    if(iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();
    if(iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();
    if(iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();
    if(iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();

    // unrolling warp
    if(tid < 32)
    {
        volatile int *vsmem  = idata;
        vsmem[tid] += vsmem[tid+32];
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8];
        vsmem[tid] += vsmem[tid+4];
        vsmem[tid] += vsmem[tid+2];
        vsmem[tid] += vsmem[tid+1];
    }
    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char *argv[])
{
    int blockSize = atoi(argv[1]);

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

    switch (blockSize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 512:
        reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 128:
        reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    case 64:
        reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    }

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