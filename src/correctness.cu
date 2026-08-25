#include "benchmark.cuh"
#include "reference.cuh"
#include "cuda_check.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

void cpu_sgemm_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<float>(acc);
        }
    }
}

void verify_correctness(const float* C_test, const float* C_ref, int M, int N,
                        double* max_abs_error, double* rel_l2_error) {
    double max_abs = 0.0;
    double l2_diff = 0.0;
    double l2_ref = 0.0;
    const double eps = 1e-8;

    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(static_cast<double>(C_test[i]) - static_cast<double>(C_ref[i]));
        max_abs = std::max(max_abs, diff);
        l2_diff += diff * diff;
        double ref_val = static_cast<double>(C_ref[i]);
        l2_ref += ref_val * ref_val;
    }

    *max_abs_error = max_abs;
    *rel_l2_error = std::sqrt(l2_diff) / (std::sqrt(l2_ref) + eps);
}

bool check_correctness(const float* C_test, const float* C_ref, int M, int N,
                       double atol, double rtol, double rel_l2_tol) {
    double max_abs_error, rel_l2_error;
    verify_correctness(C_test, C_ref, M, N, &max_abs_error, &rel_l2_error);

    bool passed = true;
    if (max_abs_error > atol) {
        printf("FAIL: max_abs_error = %.6e > %.6e\n", max_abs_error, atol);
        passed = false;
    }
    if (rel_l2_error > rel_l2_tol) {
        printf("FAIL: rel_l2_error = %.6e > %.6e\n", rel_l2_error, rel_l2_tol);
        passed = false;
    }

    if (passed) {
        printf("PASS: max_abs_error = %.6e, rel_l2_error = %.6e\n", max_abs_error, rel_l2_error);
    }

    return passed;
}

void print_matrix(const float* M, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fill_random(float* data, int size, int seed, float min_val, float max_val) {
    std::srand(seed);
    for (int i = 0; i < size; ++i) {
        data[i] = min_val + (max_val - min_val) * (static_cast<float>(std::rand()) / RAND_MAX);
    }
}