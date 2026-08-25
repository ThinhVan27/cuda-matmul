#include "kernels.cuh"
#include "cuda_check.cuh"

#include <cstring>


struct KernelEntry {
    const char* name;
    KernelLauncher launcher;
};

static const KernelEntry kernels[] = {
    {"k1",      launch_k1_naive},
    {"k2",      launch_k2_coalesced},
    {"k3",      launch_k3_smem_tiled},
    {"k4",      launch_k4_1d_thread_tiled},
    {"k5",      launch_k5_2d_register_tiled},
    {"k6",      launch_k6_vectorized},
    {"cublas",  launch_cublas}
};

KernelLauncher get_kernel_launcher(const char* kernel_name) {
    for (const auto& entry : kernels) {
        if (strcmp(entry.name, kernel_name) == 0) {
            return entry.launcher;
        }
    }
    return nullptr;
}


const char* get_kernel_name(KernelLauncher launcher) {
    for (const auto& entry : kernels) {
        if (entry.launcher == launcher) {
            return entry.name;
        }
    }
    return "unknown";
}