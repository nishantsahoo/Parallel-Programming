#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void mykernel()
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	printf("Hello world from Kernel\tBlock id: %d\tThread id: %d\n", i, j);
}

int main()
{
	printf("Hello world from CPU\n");
	mykernel<<<4,5>>>();
	cudaDeviceReset();
	while(1);
    return 0;
}
