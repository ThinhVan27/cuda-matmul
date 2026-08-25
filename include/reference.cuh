#ifndef REFERENCE_CUH
#define REFERENCE_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

// CPU reference: C = A * B (row-major), using double precision for accuracy
void cpu_sgemm_ref(const float* A, const float* B, float* C, int M, int N, int K);

// Initialize cuBLAS handle
cublasHandle_t create_cublas_handle(cudaStream_t stream);
void destroy_cublas_handle(cublasHandle_t handle);

void fill_random(float* data, int size, int seed, float min_val = -1.0f, float max_val = 1.0f);

#endif // REFERENCE_CUH