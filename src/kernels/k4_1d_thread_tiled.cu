#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>

// K4 Configuration: 1D thread tiling
// Each thread computes TM elements along M dimension
#define BM 64
#define BN 64
#define BK 8
#define TM 8

// Threads per block = (BM * BN) / TM = (64 * 64) / 8 = 512
#define THREADS_PER_BLOCK ((BM * BN) / TM)

__global__ void k4_1d_thread_tiled_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int thread_row = tid / (BN / TM);  // 0..BM-1 in steps of TM
    int thread_col_start = (tid % (BN / TM)) * TM;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Each thread accumulates TM results
    float thread_results[TM] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // Cooperative load of A tile (BM x BK)
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int load_row = i / BK;
            int load_col = i % BK;
            int global_row = block_row + load_row;
            int global_col = tile_k + load_col;
            if (global_row < M && global_col < K) {
                As[load_row][load_col] = A[global_row * K + global_col];
            } else {
                As[load_row][load_col] = 0.0f;
            }
        }

        // Cooperative load of B tile (BK x BN)
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int load_row = i / BN;
            int load_col = i % BN;
            int global_row = tile_k + load_row;
            int global_col = block_col + load_col;
            if (global_row < K && global_col < N) {
                Bs[load_row][load_col] = B[global_row * N + global_col];
            } else {
                Bs[load_row][load_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute TM x BN outer product per thread
        for (int k = 0; k < BK; ++k) {
            // Load one B value, reuse for TM A values
            float b_val = Bs[k][thread_col_start];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                float a_val = As[thread_row + tm][k];
                thread_results[tm] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // Store results
    for (int tm = 0; tm < TM; ++tm) {
        int global_row = block_row + thread_row + tm;
        int global_col = block_col + thread_col_start;
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = thread_results[tm];
        }
    }
}

void launch_k4_1d_thread_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    k4_1d_thread_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
}