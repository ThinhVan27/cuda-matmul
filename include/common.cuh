#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>

#include "cuda_check.cuh"

inline void print_gpu_info(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("=== GPU Information ===\n");
    printf("Device %d: %s\n", device_id, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("CUDA Cores/SM: %d\n", prop.major >= 7 ? 128 : (prop.major >= 5 ? 128 : 192));
    printf("Total CUDA Cores: %d\n", prop.multiProcessorCount * (prop.major >= 7 ? 128 : (prop.major >= 5 ? 128 : 192)));
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Registers/Block: %d\n", prop.regsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks/SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate * 1e-6);
    printf("Memory Clock: %.2f GHz\n", prop.memoryClockRate * 1e-6);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 1e-9);
    printf("========================\n\n");
}

inline void check_cuda_driver_version() {
    int driver_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    printf("CUDA Driver Version: %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
}

inline void check_cuda_runtime_version() {
    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    printf("CUDA Runtime Version: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
}

inline int get_device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

inline void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

inline void synchronize_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif // COMMON_CUH