#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>


void check_answer(float *a, float *b, float *c, int n){
    float temp;
    float epsilon = 0.001;

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            temp = 0;
            for (int k = 0; k < n; k++){
                temp += a[k * n + i] * b[j * n + k];
            }
            assert(fabs(c[j * n + i] - temp) < epsilon);
        }
    }
}

int main(){
    int n = 1 << 10;

    size_t bytes = n * n * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)123);

    curandGenerateUniform(prng, d_a, n*n);
    curandGenerateUniform(prng, d_b, n*n);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;

    // c = (alpha*a) * b + (beta*c)
    // (m X n) * (n * k) = (m X k)
    //          handle, operation,   operation,   m, n, k, alpha,   A, lda, B, ldb, beta,  C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
   
	check_answer(h_a, h_b, h_c, n);

	cublasDestroy(handle);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;

}