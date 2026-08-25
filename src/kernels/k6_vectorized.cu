#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>

// Fast-path SGEMM kernel adapted to the row-major C = A * B convention used by
// this project. It intentionally assumes M, N, and K are exact tile multiples.
// Use a scalar/tail kernel for arbitrary dimensions.
template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM / TM) * (BN / TN))
void k6_vectorized_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M,
                          int N,
                          int K) {
    constexpr int THREADS = (BM / TM) * (BN / TN);

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(BK % 4 == 0, "BK must be divisible by float4 width");
    static_assert(BN % 4 == 0, "BN must be divisible by float4 width");
    static_assert(TN % 4 == 0, "TN must be divisible by float4 width");
    static_assert((BM * BK) / 4 == THREADS,
                  "This loader expects one float4 of A per thread");
    static_assert((BK * BN) / 4 == THREADS,
                  "This loader expects one float4 of B per thread");

    // As is stored transposed: logically As[BK][BM]. This makes the TM values
    // consumed by one thread contiguous in shared memory.
    __shared__ __align__(16) float As[BK * BM];
    __shared__ __align__(16) float Bs[BK * BN];

    const int tid = static_cast<int>(threadIdx.x);

    const int blockRow = static_cast<int>(blockIdx.y) * BM;
    const int blockCol = static_cast<int>(blockIdx.x) * BN;

    // Each thread computes one TM x TN output micro-tile.
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const int localCRow = threadRow * TM;
    const int localCCol = threadCol * TN;

    // One float4 (four consecutive FP32 values) is loaded by each thread.
    const int innerRowA = tid / (BK / 4);
    const int innerVecA = tid % (BK / 4);

    const int innerRowB = tid / (BN / 4);
    const int innerVecB = tid % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int blockK = 0; blockK < K; blockK += BK) {
        // Vectorized 128-bit global-memory load from A.
        const int aGlobalIndex =
            (blockRow + innerRowA) * K + blockK + innerVecA * 4;
        const float4 aVec =
            *reinterpret_cast<const float4*>(&A[aGlobalIndex]);

        // Transpose A while moving it from GMEM to SMEM:
        // A tile [BM][BK] -> As [BK][BM].
        const int aSharedCol = innerRowA;
        const int aSharedRow = innerVecA * 4;
        As[(aSharedRow + 0) * BM + aSharedCol] = aVec.x;
        As[(aSharedRow + 1) * BM + aSharedCol] = aVec.y;
        As[(aSharedRow + 2) * BM + aSharedCol] = aVec.z;
        As[(aSharedRow + 3) * BM + aSharedCol] = aVec.w;

        // Vectorized 128-bit GMEM load and 128-bit SMEM store for B.
        const int bGlobalIndex =
            (blockK + innerRowB) * N + blockCol + innerVecB * 4;
        const float4 bVec =
            *reinterpret_cast<const float4*>(&B[bGlobalIndex]);

        const int bSharedIndex = innerRowB * BN + innerVecB * 4;
        *reinterpret_cast<float4*>(&Bs[bSharedIndex]) = bVec;

        __syncthreads();

        // Register-tiled outer product. For every k inside the BK tile, each
        // thread loads TM values of A and TN values of B, then performs TM*TN
        // independent FMAs into its register-resident output tile.
#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[dot * BM + localCRow + i];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[dot * BN + localCCol + j];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }

        // No thread may overwrite As/Bs with the next K tile until every
        // thread has finished consuming the current tile.
        __syncthreads();
    }

    // Vectorized 128-bit stores. This project computes C = A*B (alpha=1,
    // beta=0), so loading the old C is unnecessary.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const float4 out = make_float4(
                threadResults[i * TN + j + 0],
                threadResults[i * TN + j + 1],
                threadResults[i * TN + j + 2],
                threadResults[i * TN + j + 3]);

            const int cGlobalIndex =
                (blockRow + localCRow + i) * N +
                blockCol + localCCol + j;
            *reinterpret_cast<float4*>(&C[cGlobalIndex]) = out;
        }
    }
}

void launch_k6_vectorized(const float* A,
                          const float* B,
                          float* C,
                          int M,
                          int N,
                          int K,
                          cudaStream_t stream,
                          cublasHandle_t handle) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int THREADS = (BM / TM) * (BN / TN);  // 256

    // float4 requires 16-byte alignment. cudaMalloc aligns the base pointers;
    // these divisibility constraints keep every vectorized row/tile offset
    // aligned and ensure no vector crosses a matrix boundary.
    if (M % BM != 0 || N % BN != 0 || K % BK != 0 ||
        N % 4 != 0 || K % 4 != 0) {
        std::fprintf(stderr,
                     "k6_vectorized requires M%%128=0, N%%128=0, "
                     "K%%8=0, N%%4=0, and K%%4=0\n");
        std::abort();
    }

    const dim3 block(THREADS, 1, 1);
    const dim3 grid(N / BN, M / BM, 1);

    k6_vectorized_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
