cmake_minimum_required(VERSION 3.22)

project(matmul_bench LANGUAGES CXX CUDA)

# ============================================================
# Language standards
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4090: Ada Lovelace, Compute Capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

# ============================================================
# CUDA Toolkit
# ============================================================

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# ============================================================
# Sources
# ============================================================

set(SOURCES
    src/main.cu
    src/benchmark.cu
    src/correctness.cu
    src/kernels/kernel_dispatch.cu

    src/kernels/k1_naive.cu
    src/kernels/k2_coalesced.cu
    src/kernels/k3_smem_tiled.cu
    src/kernels/k4_1d_thread_tiled.cu
    src/kernels/k5_2d_register_tiled.cu
    src/kernels/k6_vectorized.cu
    src/kernels/cublas.cu
)

# ============================================================
# Executable
# ============================================================

add_executable(matmul_bench ${SOURCES})

target_include_directories(
    matmul_bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# Compiler options
# ============================================================

target_compile_options(
    matmul_bench
    PRIVATE

    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >

    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --ptxas-options=-v
        -lineinfo
    >
)

# ============================================================
# CUDA libraries
# ============================================================

target_link_libraries(
    matmul_bench
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# Optional installation
install(
    TARGETS matmul_bench
    RUNTIME DESTINATION bin
)