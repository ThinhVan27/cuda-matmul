#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>


__global__ void k2_coalesced_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {

    // block is transposed
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] =  acc;
    }
}

void launch_k2_coalesced(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle) {
    // 1D block of 256 threads (TILE_M * TILE_N = 8 * 32 = 256)
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    k2_coalesced_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
}