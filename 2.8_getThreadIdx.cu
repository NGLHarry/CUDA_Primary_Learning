#include <stdio.h>

__global__ void getThreadIdx()
{
    printf("blockDim:x=%d, y=%d, z=%d gridDim:x=%d, y=%d, z=%d Current threadIdx:x=%d, y=%d, z=%d\n",
        blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, threadIdx.x, threadIdx.y, threadIdx.z
    );
}

int main()
{
    printf("Hello from CPU\n");
    dim3 grid;
    grid.x = 2;
    grid.y = 2;

    dim3 block;
    block.x = 2;
    block.y = 2;
    getThreadIdx<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}