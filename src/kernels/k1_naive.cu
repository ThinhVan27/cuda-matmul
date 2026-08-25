#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>

__global__ void k1_naive_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    // Intentional poor mapping: threadIdx.x -> row (strided access)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_k1_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle) {
    // 32x32 = 1024 threads/block
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    k1_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
}