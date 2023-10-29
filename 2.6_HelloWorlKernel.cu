#include <stdio.h>

__global__ void helloFromGpu()
{
    printf("Hello World from GPU\n");
}

int main()
{
    printf("hello from CPU\n");
    helloFromGpu<<<1,4>>>();
    cudaDeviceReset();
    return 0;
}