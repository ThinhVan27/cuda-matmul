# Evaluate and Optimize Matrix Multiplication FP32 CUDA kernels

The report will identify kernel bottleneck and propose optimization. Some configuration below (hardware information in `docs/`):
- Matrix dimensions (with $M = N = K = 4096$): 
    - Matrix $A \in \R^{M \times K}$
    - Matrĩx $B \in \R^{K \times N}$
    - Matrix $C \in \R^{M \times N}$     

Baseline - cuBLAS:
- GFLOP/s: 54246 ms
## 1. Naive version
Kernel is implement as the most intuitive way:
- Each thread handles one element in **C**. Value at $\mathbf{C}[row][col]$ is the dot-product of $\mathbf{A}[row][:]$ and $\mathbf{B}[:][col]$, through a for-loop.
- Dimension `x` of grid/block is mapped to row, dimension `y`is mapped to col.

```c++
__global__ void k1_naive_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    // Intentional poor mapping: threadIdx.x -> row (strided access)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // for fair benchmark with CUBLAS
    float alpha = 1.0f;
    float beta = 0.0f;

    // ensuring this thread should cover an element
    if (row < M && col < N) {
        float acc = 0.0f;
        // dot-product A[row][:] vs B[:][col]
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}
```
For this matrix multiplication, the computational workload is:

* **Floating-point operations:**
  $
  2MNK + MN
  = 2 \times 4096^3  \approx 137.43\ \text{GFLOPs}.
  $
  The (2MNK) term comes from the matrix multiplication ($A \times B$), where each multiplication and addition counts as one FLOP. In practice, these operations are typically fused into an FMA instruction, which performs two FLOPs.

* **Execution time:** ($500.378\ \text{ms}$), corresponding to approximately ($274.7\ \text{GFLOP/s}$), which is significantly lower than the cuBLAS baseline.

In this block configuration, `threadIdx.x` is mapped to the row dimension. Consequently, when a warp loads elements from (A), adjacent threads access memory locations separated by a stride of (K) elements rather than consecutive addresses. These accesses cannot be efficiently coalesced, so the GPU must generate substantially more memory transactions.

The increased number of transactions places pressure on the load/store pipeline and memory queues, increases the number of cycles between issued instructions, and ultimately raises kernel latency. Moreover, accesses to global memory inherently have much higher latency than accesses to shared memory or registers.


## 2. Global Memory coalesing
Consecutive lanes access consecutive elements of B and C, allowing the warp requests to be served using fewer global-memory transactions and improving the ratio of useful bytes to transferred bytes. Checking the profile, we can see the load sectors down from 14M to 2.5M!

```c++
    // block is transposed
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
```

With only adjust the block dimension, the latency down to 64.204 ms, so the GFLOPs/s = 2207.9.


## 3. Shared-memory tiled
Each block cooperatively loads one tile of A and one tile of B from global memory into shared memory. The loaded values are then reused by multiple threads when computing the corresponding tile of C. This reduces repeated global-memory transactions at the cost of shared-memory operations and block-wide synchronization. Global loads decresed frome 4T to 134M, accelerating with shared-memory loads.
- Latency: 51.063 ms
- GFLOPs/s = 2829.687 

## 4. 1D thread-tiled
In the 1D thread-tiled kernel, each thread computes multiple output elements along one matrix dimension. A value loaded from shared memory can therefore be reused across multiple accumulators, reducing the number of shared-memory instructions per FMA and increasing instruction-level parallelism. We can see in the profile, shared-mem and global mem loads scaled down significantly.
- Latency: 24.6 ms
- GFLOPs/s: 5464.9 ms

## 5. 2D thread-tiled

Each thread computes a two-dimensional $TM \times TN$ output tile and keeps its accumulators in registers. However, it can increase register usage -> might reduce warps if exceed reg resource per block.
- Latency: 8.89 ms
- GFLOPs/s: 14595.3

## 6. Vectorized

With the same design as 2D threadtiling, some optimization added:
- Use aligned `float4` operations so that each thread transfers four FP32 values using a 16-byte vector memory operation instead of four separate scalar operations, when the compiler can generate the corresponding vector instruction.
- Tile A is transposed while being stored in shared memory. This permits vectorized, contiguous loads from global memory while providing a shared-memory layout suitable for the later register-loading pattern.

- Latency: 6.73 ms
- GFLOPs/s: 20360.7