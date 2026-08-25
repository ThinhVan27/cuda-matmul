#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>

// K5 Configuration: 2D register tiling
// Template parameters for compile-time constants
// Config 1: BM=64, BN=64, BK=8, TM=4, TN=4 -> 256 threads
// Config 2: BM=128, BN=128, BK=8, TM=8, TN=8 -> 256 threads

// Use Config 2 for better performance on larger matrices
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

#define THREADS_X (BN / TN)
#define THREADS_Y (BM / TM)
#define THREADS_PER_BLOCK (THREADS_X * THREADS_Y)

static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(BN % TN == 0, "BN must be divisible by TN");
static_assert(THREADS_PER_BLOCK <= 1024, "Too many threads per block");
static_assert(THREADS_PER_BLOCK % 32 == 0, "Threads per block should be multiple of warp size");

__global__ void k5_2d_register_tiled_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * THREADS_X + tx;

    int thread_row = ty * TM;
    int thread_col = tx * TN;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Register accumulation: TM x TN per thread
    float thread_results[TM][TN] = {{0.0f}};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // Cooperative load of A tile (BM x BK)
        for (int i = tid; i < BM * BK; i += THREADS_PER_BLOCK) {
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
        for (int i = tid; i < BK * BN; i += THREADS_PER_BLOCK) {
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

        // Compute TM x TN outer product per thread
        for (int k = 0; k < BK; ++k) {
            // Load TM elements from A (row)
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = As[thread_row + tm][k];
            }

            // Load TN elements from B (column)
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = Bs[k][thread_col + tn];
            }

            // Outer product accumulation
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    thread_results[tm][tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }

        __syncthreads();
    }

    // Store TM x TN results
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            int global_row = block_row + thread_row + tm;
            int global_col = block_col + thread_col + tn;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = thread_results[tm][tn];
            }
        }
    }
}

void launch_k5_2d_register_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle) {
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    k5_2d_register_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
}