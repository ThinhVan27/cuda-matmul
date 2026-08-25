#include "benchmark.cuh"
#include "common.cuh"
#include "kernels.cuh"
#include "cuda_check.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --kernel <name>       Kernel to run: k1, k2, k3, k4, k5, k6, cublas, all\n");
    printf("  --m <int>             Matrix M dimension (default: 4096)\n");
    printf("  --n <int>             Matrix N dimension (default: 4096)\n");
    printf("  --k <int>             Matrix K dimension (default: 4096)\n");
    printf("  --warmup <int>        Warmup iterations (default: 20)\n");
    printf("  --iterations <int>    Benchmark iterations (default: 30)\n");
    printf("  --seed <int>          Random seed (default: 42)\n");
    printf("  --check               Run correctness check (default: false)\n");
    printf("  --csv <path>          Output CSV file path\n");
    printf("  --device <int>        GPU device ID (default: 0)\n");
    printf("  --debug               Print each iteration logs\n");
    printf("  --help                Show this help\n");
}

int main(int argc, char** argv) {
    BenchmarkConfig config = {};
    config.kernel_name = "k1";
    config.M = 4096;
    config.N = 4096;
    config.K = 4096;
    config.warmup = 20;
    config.iterations = 30;
    config.seed = 42;
    config.check_correctness = false;
    config.csv_path = nullptr;
    config.device_id = 0;
    config.debug = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            config.kernel_name = argv[++i];
        } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            config.M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            config.N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            config.K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--check") == 0) {
            config.check_correctness = true;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            config.csv_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            config.device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug") == 0) {
            config.debug = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Print GPU info
    print_gpu_info(config.device_id);
    check_cuda_driver_version();
    check_cuda_runtime_version();

    // Print configuration
    printf("=== Benchmark Configuration ===\n");
    printf("Kernel: %s\n", config.kernel_name);
    printf("M=%d, N=%d, K=%d\n", config.M, config.N, config.K);
    printf("Warmup: %d, Iterations: %d\n", config.warmup, config.iterations);
    printf("Seed: %d\n", config.seed);
    printf("Check correctness: %s\n", config.check_correctness ? "yes" : "no");
    printf("CSV output: %s\n", config.csv_path ? config.csv_path : "none");
    printf("Device: %d\n", config.device_id);
    printf("================================\n\n");

    // Initialize CSV if requested
    if (config.csv_path) {
        write_csv_header(config.csv_path);
    }

    // Run benchmark
    BenchmarkResult result;
    run_benchmark(config, &result);

    // Print results
    printf("\n=== Results ===\n");
    printf("Kernel: %s\n", result.kernel.c_str());
    printf("Median latency: %.3f ms\n", result.median_ms);
    printf("Mean latency:   %.3f ms\n", result.mean_ms);
    printf("Min latency:    %.3f ms\n", result.min_ms);
    printf("P95 latency:    %.3f ms\n", result.p95_ms);
    printf("GFLOP/s:        %.3f\n", result.gflops);
    // printf("cuBLAS GFLOP/s: %.3f\n", result.cublas_gflops);
    // printf("%% cuBLAS:       %.2f%%\n", result.percent_cublas);
    if (config.check_correctness) {
        printf("Max abs error:  %.6e\n", result.max_abs_error);
        printf("Rel L2 error:   %.6e\n", result.relative_l2_error);
    }
    printf("================\n");

    // Write CSV
    if (config.csv_path) {
        write_csv_row(config.csv_path, result);
        printf("Results written to %s\n", config.csv_path);
    }

    return 0;
}