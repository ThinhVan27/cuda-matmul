#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "cuda_check.cuh"

struct BenchmarkConfig {
    const char* kernel_name;
    int M, N, K;
    int warmup;
    int iterations;
    int seed;
    bool check_correctness;
    const char* csv_path;
    int device_id;
    bool debug;
};

struct BenchmarkResult {
    std::string timestamp;
    std::string gpu_name;
    std::string compute_capability;
    std::string cuda_version;
    std::string kernel;
    int M, N, K;
    int warmup;
    int iterations;
    double median_ms;
    double mean_ms;
    double min_ms;
    double p95_ms;
    double gflops;
    // double cublas_gflops;
    // double percent_cublas;
    double max_abs_error;
    double relative_l2_error;
    int BM, BN, BK, TM, TN;
    int block_threads;
    int registers_per_thread;
    size_t smem_bytes;
};

struct Timer {
    cudaEvent_t start, stop;
    Timer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void start_timing(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    float stop_timing(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

void run_benchmark(const BenchmarkConfig& config, BenchmarkResult* result);
void write_csv_header(const char* csv_path);
void write_csv_row(const char* csv_path, const BenchmarkResult& result);
void verify_correctness(const float* C_test, const float* C_ref, int M, int N,
                        double* max_abs_error, double* rel_l2_error);
bool check_correctness(const float* C_test, const float* C_ref, int M, int N,
                       double atol = 1e-2, double rtol = 1e-3, double rel_l2_tol = 1e-4);

std::string get_timestamp();

#endif // BENCHMARK_CUH