#!/bin/bash
# Run full benchmark suite for all kernels on all benchmark shapes

set -e

BUILD_DIR="${1:-build}"
EXECUTABLE="$BUILD_DIR/matmul_bench"
RESULTS_DIR="results"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found: $EXECUTABLE"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

KERNELS=("k1" "k2" "k3" "k4" "k5" "k6" "cublas")

# Square shapes
SQUARE_SHAPES=("128 128 128" "256 256 256" "512 512 512" "1024 1024 1024" "2048 2048 2048" "4096 4096 4096")

WARMUP=20
ITERATIONS=100
SEED=42

echo "=== Full Benchmark Suite ==="
echo "Kernels: ${KERNELS[*]}"
echo "Warmup: $WARMUP, Iterations: $ITERATIONS"
echo ""

# Square benchmarks
echo "--- Square Matrices ---"
for shape in "${SQUARE_SHAPES[@]}"; do
    read -r M N K <<< "$shape"
    echo "Shape: ${M}x${N}x${K}"

    for kernel in "${KERNELS[@]}"; do
        CSV_FILE="$RESULTS_DIR/${M}/tables/${kernel}.csv"
        echo "  $kernel -> $CSV_FILE"
        "$EXECUTABLE" --kernel "$kernel" --m "$M" --n "$N" --k "$K" \
            --warmup "$WARMUP" --iterations "$ITERATIONS" --seed "$SEED" \
            --csv "$CSV_FILE"
    done
    echo ""
done

echo "All benchmarks complete! Results in $RESULTS_DIR"