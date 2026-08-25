#!/bin/bash
# Run correctness tests for all kernels on all test shapes

set -e

BUILD_DIR="${1:-build}"
EXECUTABLE="$BUILD_DIR/matmul_bench"
SHAPES_FILE="tests/test_shapes.txt"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found: $EXECUTABLE"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

KERNELS=("k1" "k2" "k3" "k4" "k5" "k6" "cublas")

echo "=== Correctness Testing ==="
echo "Kernels: ${KERNELS[*]}"
echo "Shapes from: $SHAPES_FILE"
echo ""

while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    read -r M N K <<< "$line"
    echo "Testing shape: M=$M, N=$N, K=$K"

    for kernel in "${KERNELS[@]}"; do
        echo "  Kernel: $kernel"
        "$EXECUTABLE" --kernel "$kernel" --m "$M" --n "$N" --k "$K" --check --warmup 5 --iterations 1
        if [ $? -ne 0 ]; then
            echo "  FAILED: $kernel on $M x $N x $K"
            exit 1
        fi
    done
    echo ""
done < "$SHAPES_FILE"

echo "All correctness tests PASSED!"