#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include "cublas_v2.h"

// Kernel launcher function type
using KernelLauncher = void (*)(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    cublasHandle_t handle
);

// Kernel declarations
void launch_k1_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);
void launch_k2_coalesced(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);
void launch_k3_smem_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);
void launch_k4_1d_thread_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);
void launch_k5_2d_register_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);
void launch_k6_vectorized(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);
void launch_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream, cublasHandle_t handle);

// Get kernel launcher by name
KernelLauncher get_kernel_launcher(const char* kernel_name);
const char* get_kernel_name(KernelLauncher launcher);

#endif // KERNELS_CUH