#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n){
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n){
        c[tid] = a[tid] + b[tid];
    }

}

void matrix_init(int* a, int n){
    for (int i = 0; i < n; i++){
        a[i] = rand()%100;
    }
}

void check_error(int* a, int* b, int* c, int n){
    for (int i = 0; i < n; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){
    int id = cudaGetDevice(&id);

    // 2^16
    int n = 1 << 16;

    size_t bytes = sizeof(int) * n;

    // unified memory vector pointers
    int *a, *b, *c;

    // Allocation for unifed memory vectors
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize vectors with random values
    matrix_init(a, n);
    matrix_init(b, n);

    int NUM_THREADS = 256;

    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // Prefetch memory sync to device
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
    
    check_error(a, b, c, n);

    printf("Completed.\n");

    return 0;


}