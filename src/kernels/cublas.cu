#include <cstdio>
#include <cublas_v2.h>

#include "kernels.cuh"
#include "reference.cuh"
#include "cuda_check.cuh"


cublasHandle_t create_cublas_handle(cudaStream_t stream) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    return handle;
}

void destroy_cublas_handle(cublasHandle_t handle) {
    CUBLAS_CHECK(cublasDestroy(handle));
}

void launch_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle) {
    // cuBLAS uses column-major. We have row-major A(M,K), B(K,N), C(M,N).
    // C = A * B
    // C^T = B^T * A^T
    // So we call cublasSgemm with B, A swapped, it will return C^T, and anyway, it saved in memory at C's row-major form!!! Don't need any processing more.

    // cuBLAS: C = alpha * op(A) * op(B) + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,  // B is K x N row-major -> col-major view is N x K
        A, K,  // A is M x K row-major -> col-major view is K x M
        &beta,
        C, N   // C is M x N row-major -> col-major view is N x M
    ));
}