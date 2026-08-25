#!/usr/bin/env bash

set -euo pipefail

KERNELS=(
  "k1"
  "k2"
  "k3"
  "k4"
  "k5"
  "k6"
  "cublas"
)
SIZES=(
  128
  # 256
  # 512
  # 1024
  # 2048
  # 4096
)

for kernel in "${KERNELS[@]}"; do
    for size in "${SIZES[@]}"; do
        PROFILE_DIR="results/${size}/profiles"
        rm -f "$PROFILE_DIR/${kernel}.ncu-rep"
        /usr/local/cuda-12/bin/ncu \
          --set full \
          --kernel-name "regex:.*${kernel}.*" \
          --launch-skip 20 \
          --launch-count 1 \
          --export "$PROFILE_DIR/${kernel}.ncu-rep" \
          ./build/matmul_bench \
          --kernel "${kernel}"
    done
done