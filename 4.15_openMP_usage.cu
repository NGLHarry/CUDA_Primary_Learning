#include <omp.h>
#include <stdio.h>

int main()
{
    omp_set_num_threads(5);
    #pragma omp parallel
    {
        printf("thread is running\n");
    }
    return 0;
}