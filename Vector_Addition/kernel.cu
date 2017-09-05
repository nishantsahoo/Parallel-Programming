#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <cstdio>
#include <math.h>

// this kernel computes the vector sum c = a + b
// each thread performs one pair-wise addition
__global__ void vector_add(const float *a, const float *b, float *c, const size_t n) // vector_add
{
 unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  // avoid accessing out of bounds elements
  if(i < n)
  {
    // sum elements
    c[i] = a[i] + b[i];
  }
} // end of the function vector_add

int main(void) // main function
{
  // create arrays of 1M elements
  int num_elements = 0 ;
  printf("Enter number of elements to add: ");
  scanf("%d", &num_elements);
  // compute the size of the arrays in bytes
  const int num_bytes = num_elements * sizeof(float);

  // points to host & device arrays
  float *device_array_a = 0;
  float *device_array_b = 0;
  float *device_array_c = 0;

  float *host_array_a   = 0;
  float *host_array_b   = 0;
  float *host_array_c   = 0;

  // malloc the host arrays
  host_array_a = (float*)malloc(num_bytes);
  host_array_b = (float*)malloc(num_bytes);
  host_array_c = (float*)malloc(num_bytes);

  // cudaMalloc the device arrays
 cudaMalloc((void**)&device_array_a, num_bytes);
 cudaMalloc((void**)&device_array_b, num_bytes);
 cudaMalloc((void**)&device_array_c, num_bytes);


  // initialize host_array_a & host_array_b
  for(int i = 0; i < num_elements; ++i)
  {
    // make array a a linear ramp
    host_array_a[i] = (float)i;

    // make array b random
    host_array_b[i] = (float)rand() / RAND_MAX;
  }
    
  // copy arrays a & b to the device memory space
  cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_array_b, host_array_b, num_bytes, cudaMemcpyHostToDevice);

  vector_add <<< ceil(num_elements/32.0), 32>>>(device_array_a, device_array_b, device_array_c, num_elements);
  cudaMemcpy(host_array_c, device_array_c, num_bytes, cudaMemcpyDeviceToHost);
  for(int i = 0; i < num_elements; ++i)
  {
    printf("result %d: %1.1f + %7.1f = %7.1f\n", i, host_array_a[i], host_array_b[i], host_array_c[i]);
  }

  // deallocate memory
  free(host_array_a);
  free(host_array_b);
  free(host_array_c);

  cudaFree(device_array_a);
  cudaFree(device_array_b);
  cudaFree(device_array_c);
  while(1);
} // end of the main function