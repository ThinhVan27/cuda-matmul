# Hardware & Software Environment

## GPU
- **Model**: NVIDIA GeForce RTX 4090
- **VRAM**: 24 GB GDDR6X
- **Compute Capability**: 8.9 (Ada Lovelace)
- **SMs**: 128
- **CUDA Cores**: 16,384 (128 per SM)
- **Tensor Cores**: 512 (4th gen)
- **Base Clock**: 2.23 GHz
- **Boost Clock**: ~2.52 GHz
- **Memory Clock**: 21 Gbps
- **Memory Bus**: 384-bit
- **Peak Memory Bandwidth**: ~1,008 GB/s
- **FP32 TFLOPs**: ~82.6 TFLOPs (boost)
- **FP32 TFLOPs (base)**: ~73.0 TFLOPs

## Software
- **OS**: Ubuntu 22.04.5 LTS
- **CUDA Toolkit**: 12.2
- **cuBLAS**: 12.2
- **NVIDIA Driver**:  550.163.01
- **Compiler**: NVCC (CUDA), GCC (host)
- **CMake**: 3.22.1
- **Python**: 3.10.13 

## Build Configuration
```cmake
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

target_compile_options(matmul_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
```

## Notes
- RTX 4090 supports 128 KB shared memory per SM (configurable up to 164 KB with 8 KB L1)
- 64 KB register file per SM (256 KB total per SM in Ada)
- 128 CUDA cores per SM = 4 warps per cycle throughput
- 4th gen Tensor Cores support FP8, FP16, BF16, TF32, FP64
- For FP32 CUDA core SGEMM, we disable Tensor Core math mode in cuBLAS for fair comparison
- Power limit: 450W default, can affect sustained clocks
- Thermal throttling may occur under sustained load