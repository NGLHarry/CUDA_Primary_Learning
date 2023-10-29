#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion = %d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);
    if(iSize == 1)
    {
        return;
    }
    // reduce block size to half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("------>nested execution depth :%d\n",iDepth);
    }
}


int main(int argc, char *argv[])
{
    int size = 8;
    int blockSize =8;
    int igrid = 1;

    if(argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * blockSize;
    }

    dim3 block(blockSize, 1);
    dim3 grid((size + blockSize -1)/blockSize, 1);
    printf("%s Execution Configuation:grid %d block %d\n", argv[0], grid.x, block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);
    cudaDeviceReset();

    return 0;
}