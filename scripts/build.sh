#!/bin/bash
# Build script for CUDA matmul optimization project

set -e

BUILD_DIR="${1:-build}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building in $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_DIR" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "Build complete. Executable: $BUILD_DIR/matmul_bench"