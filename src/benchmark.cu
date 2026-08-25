#include "benchmark.cuh"
#include "common.cuh"
#include "kernels.cuh"
#include "reference.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <sstream>

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string get_cuda_version() {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d", runtime_version / 1000, (runtime_version % 1000) / 10);
    return std::string(buf);
}

void run_benchmark(const BenchmarkConfig& config, BenchmarkResult* result) {
    // Set device
    CUDA_CHECK(cudaSetDevice(config.device_id));

    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, config.device_id));
    result->gpu_name = prop.name;
    char cc_buf[16];
    snprintf(cc_buf, sizeof(cc_buf), "%d.%d", prop.major, prop.minor);
    result->compute_capability = cc_buf;
    result->cuda_version = get_cuda_version();

    result->timestamp = get_timestamp();
    result->kernel = config.kernel_name;
    result->M = config.M;
    result->N = config.N;
    result->K = config.K;
    result->warmup = config.warmup;
    result->iterations = config.iterations;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = config.M * config.K * sizeof(float);
    size_t size_B = config.K * config.N * sizeof(float);
    size_t size_C = config.M * config.N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Allocate host memory for initialization and verification
    std::vector<float> h_A(config.M * config.K);
    std::vector<float> h_B(config.K * config.N);
    std::vector<float> h_C(config.M * config.N);
    std::vector<float> h_C_ref(config.M * config.N);

    // Fill with random data
    fill_random(h_A.data(), config.M * config.K, config.seed);
    fill_random(h_B.data(), config.K * config.N, config.seed + 1);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Get kernel launcher
    KernelLauncher launcher = get_kernel_launcher(config.kernel_name);
    if (!launcher) {
        fprintf(stderr, "Unknown kernel: %s\n", config.kernel_name);
        exit(EXIT_FAILURE);
    }

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create handle
    cublasHandle_t handle = create_cublas_handle(stream);

    // Warmup
    for (int i = 0; i < config.warmup; ++i) {
        // CUDA_CHECK(cudaMemsetAsync(d_C, 0, size_C, stream));
        launcher(d_A, d_B, d_C, config.M, config.N, config.K, stream, handle);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark iterations
    std::vector<float> times;
    times.reserve(config.iterations);

    Timer timer;
    printf("====================\n");
    for (int i = 0; i < config.iterations; ++i) {
        // CUDA_CHECK(cudaMemsetAsync(d_C, 0, size_C, stream));
        timer.start_timing(stream);
        launcher(d_A, d_B, d_C, config.M, config.N, config.K, stream, handle);
        float ms = timer.stop_timing(stream);
        if (config.debug) printf("Iteration %d --- Time: %.4f ms --- GFLOPs/s: %.4f\n", i, ms, 2.0 * config.M * config.N * config.K /( ms * 1e6));
        times.push_back(ms);
    }
    printf("====================\n");

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy result back for verification
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // Compute statistics
    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];
    double min_ms = times.front();
    double sum_ms = std::accumulate(times.begin(), times.end(), 0.0);
    double mean_ms = sum_ms / times.size();
    size_t p95_idx = static_cast<size_t>(times.size() * 0.95);
    double p95_ms = times[p95_idx];

    result->median_ms = median_ms;
    result->mean_ms = mean_ms;
    result->min_ms = min_ms;
    result->p95_ms = p95_ms;

    // Compute GFLOP/s
    double flops = 2.0 * config.M * config.N * config.K;
    result->gflops = flops / (median_ms * 1e-3) / 1e9;

    // Correctness check
    if (config.check_correctness) {
        // CPU reference
        cpu_sgemm_ref(h_A.data(), h_B.data(), h_C_ref.data(), config.M, config.N, config.K);

        double max_abs_error, rel_l2_error;
        verify_correctness(h_C.data(), h_C_ref.data(), config.M, config.N,
                          &max_abs_error, &rel_l2_error);
        result->max_abs_error = max_abs_error;
        result->relative_l2_error = rel_l2_error;

        bool passed = check_correctness(h_C.data(), h_C_ref.data(), config.M, config.N);
        if (!passed) {
            fprintf(stderr, "Correctness check FAILED for kernel %s\n", config.kernel_name);
        }
    }

    // cuBLAS reference for performance comparison
    // cublasHandle_t handle = create_cublas_handle();
    // CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    // // Warmup cuBLAS
    // for (int i = 0; i < config.warmup; ++i) {
    //     cublas_sgemm_ref(handle, d_A, d_B, d_C, config.M, config.N, config.K);
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());

    // // Benchmark cuBLAS
    // std::vector<float> cublas_times;
    // cublas_times.reserve(config.iterations);
    // for (int i = 0; i < config.iterations; ++i) {
    //     CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    //     timer.start_timing(stream);
    //     cublas_sgemm_ref(handle, d_A, d_B, d_C, config.M, config.N, config.K);
    //     float ms = timer.stop_timing(stream);
    //     cublas_times.push_back(ms);
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaStreamDestroy(stream));

    // destroy_cublas_handle(handle);

    // std::sort(cublas_times.begin(), cublas_times.end());
    // double cublas_median_ms = cublas_times[cublas_times.size() / 2];
    // result->cublas_gflops = flops / (cublas_median_ms * 1e-3) / 1e9;
    // result->percent_cublas = (result->gflops / result->cublas_gflops) * 100.0;

    // Default tile params (will be overridden by specific kernels if needed)
    result->BM = result->BN = result->BK = 0;
    result->TM = result->TN = 0;
    result->block_threads = 0;
    result->registers_per_thread = 0;
    result->smem_bytes = 0;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    destroy_cublas_handle(handle);
    cudaStreamDestroy(stream);
}

void write_csv_header(const char* csv_path) {
    FILE* f = fopen(csv_path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", csv_path);
        return;
    }
    fprintf(f, "timestamp,gpu_name,compute_capability,cuda_version,kernel,"
             "M,N,K,warmup,iterations,median_ms,mean_ms,min_ms,p95_ms,"
             "gflops,max_abs_error,relative_l2_error,"
             "BM,BN,BK,TM,TN,block_threads,registers_per_thread,smem_bytes\n");
    fclose(f);
}

void write_csv_row(const char* csv_path, const BenchmarkResult& result) {
    FILE* f = fopen(csv_path, "a");
    if (!f) {
        fprintf(stderr, "Failed to open CSV file: %s\n", csv_path);
        return;
    }
    fprintf(f, "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\","
             "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,"
             "%.6f,%.6e,%.6e,"
             "%d,%d,%d,%d,%d,%d,%d,%zu\n",
        result.timestamp.c_str(),
        result.gpu_name.c_str(),
        result.compute_capability.c_str(),
        result.cuda_version.c_str(),
        result.kernel.c_str(),
        result.M, result.N, result.K,
        result.warmup, result.iterations,
        result.median_ms, result.mean_ms, result.min_ms, result.p95_ms,
        result.gflops,
        result.max_abs_error, result.relative_l2_error,
        result.BM, result.BN, result.BK, result.TM, result.TN,
        result.block_threads, result.registers_per_thread, result.smem_bytes);
    fclose(f);
}