#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define SHMEM_SIZE 16*16*4


__global__ void matrixMul(int *a, int *b, int *c, int n, int tile_size){
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;  

    int temp_sum = 0;
    
    for (int i = 0; i < (n / tile_size); i++){
        A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
        B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

        __syncthreads();

        for(int j = 0; j < tile_size; j++){
            temp_sum += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
        }
        
        __syncthreads();
    }

    c[(row * n) + col] = temp_sum;
}

void matrix_init(int *a, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            a[i * n + j] = rand() % 100;
        }
    }
}

void check_answer(int *a, int *b, int *c, int n){
    int *verify_c;
    verify_c = (int*)malloc(n*n*sizeof(int));
    int temp_sum;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            temp_sum = 0;
            for (int k = 0; k < n; k++){
                temp_sum += a[i * n + k] * b[k * n + j];
            }
            verify_c[i * n + j] = temp_sum;
        }

    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            assert(c[i * n + j] == verify_c[i * n + j]);
        }
    }
}

int main(){
    int n = 1 << 10;

    size_t bytes = n*n*sizeof(int);

    int *h_a, *h_b, *h_c;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrix_init(h_a, n);
    matrix_init(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;

    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrixMul<<<grid, threads>>>(d_a, d_b, d_c, n, BLOCK_SIZE);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    check_answer(h_a, h_b, h_c, n);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Completed.\n");

    return 0;

}