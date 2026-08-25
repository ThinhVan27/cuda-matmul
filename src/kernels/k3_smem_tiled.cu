#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>

// Tile size - tunable parameter
#define TILE 32

__global__ void k3_smem_tiled_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        // Cooperative load of A tile
        int a_col = tile_k + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Cooperative load of B tile
        int b_row = tile_k + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_k3_smem_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    k3_smem_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
}