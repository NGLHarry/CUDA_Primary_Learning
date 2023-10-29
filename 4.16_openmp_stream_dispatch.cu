#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>
#include <ctime>
#include <omp.h>

#define NSTREAM 4

__device__ void kernel_func()
{
    double sum =0.0;
    long i = 99999;
    while(i > 0)
    {
        for(long j = 0; j < 99999; ++j)
        {
            sum = sum + tan(0.1) * tan(0.1);
        }
        i--;
    }
}

__global__ void kernel_1(int stream)
{
    if(0 == threadIdx.x)
    {
        printf("kernel_1 is excuted in stream %d\n", stream);
    }
    kernel_func();
}


__global__ void kernel_2(int stream)
{
    if(0 == threadIdx.x)
    {
        printf("kernel_2 is excuted in stream %d\n", stream);
    }
    kernel_func();
}

__global__ void kernel_3(int stream)
{
    if(0 == threadIdx.x)
    {
        printf("kernel_3 is excuted in stream %d\n", stream);
    }
    kernel_func();
}
__global__ void kernel_4(int stream)
{
    if(0 == threadIdx.x)
    {
        printf("kernel_4 is excuted in stream %d\n", stream);
    }
    kernel_func();
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

    float elapsed_time;
    // Allocate and initialize an array of stream handles
    int n_stream = NSTREAM;
    cudaStream_t *streams = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));

    for(int i = 0; i < n_stream; ++i)
    {
        ErrorCheck(cudaStreamCreate(&streams[i]),__FILE__, __LINE__);
    }

    // set up execution configuration
    dim3 block(1);
    dim3 grid(1);

    // create event
    cudaEvent_t start, stop;
    ErrorCheck(cudaEventCreate(&start, cudaEventDefault), __FILE__, __LINE__);
    ErrorCheck(cudaEventCreate(&stop, cudaEventDefault), __FILE__, __LINE__);

    // record start event
    ErrorCheck(cudaEventRecord(start, 0), __FILE__, __LINE__);

    // execute kernels
    // dispatch with OpenMP
    omp_set_num_threads(NSTREAM);
    #pragma omp parallel
    {
        int threadid = omp_get_thread_num();
        kernel_1<<<grid, block, 0, streams[threadid]>>>(threadid);
        kernel_2<<<grid, block, 0, streams[threadid]>>>(threadid);
        kernel_3<<<grid, block, 0, streams[threadid]>>>(threadid);
        kernel_4<<<grid, block, 0, streams[threadid]>>>(threadid);
    }

    // recored stop event
    ErrorCheck(cudaEventRecord(stop, 0), __FILE__, __LINE__);
    printf("begin to Synchronize\n");
    ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);

    // calculate elapsed time
    ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
    printf("Mesured time for parallel execution = %.3f ms\n", elapsed_time);

    // realse all stream
    for(int i = 0; i < n_stream; ++i)
    {
        ErrorCheck(cudaStreamDestroy(streams[i]), __FILE__, __LINE__);
    }
    free(streams);

    ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
    ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);


    cudaDeviceReset();

    return 0;
}