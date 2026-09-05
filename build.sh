#!/usr/bin/env bash
# Builds the CPU targets always, and the CUDA targets when nvcc is available.
#
# The CPU baseline and the correctness gate (test_cpu_correctness) need no
# CUDA at all, so the math in this repo can be built and verified on any
# machine. Only the three GPU implementations require the CUDA toolkit, and
# only running them requires an actual NVIDIA GPU -- nvcc compiles them fine
# without one, which is what CI does.
#
# Overrides: CXX, NVCC, CUDA_ARCH (e.g. CUDA_ARCH=sm_80). Set CPU_ONLY=1 to
# skip the CUDA targets even if nvcc is present.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin

CXX="${CXX:-c++}"
NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-native}"

"$CXX" -O3 -std=c++17 src/linreg_cpu.cpp            -o bin/linreg_cpu
"$CXX" -O3 -std=c++17 src/test_cpu_correctness.cpp  -o bin/test_cpu_correctness
built="bin/linreg_cpu, bin/test_cpu_correctness"

if [ "${CPU_ONLY:-0}" = "1" ]; then
    echo "CPU_ONLY=1 -- skipping CUDA targets."
elif command -v "$NVCC" >/dev/null 2>&1; then
    "$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cuda_naive.cu     -o bin/linreg_cuda_naive
    "$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cuda_optimized.cu -o bin/linreg_cuda_optimized
    "$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cublas.cu -lcublas -o bin/linreg_cublas
    "$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cuda_multigpu.cu -o bin/linreg_cuda_multigpu
    built="$built, bin/linreg_cuda_naive, bin/linreg_cuda_optimized, bin/linreg_cublas, bin/linreg_cuda_multigpu"
else
    echo "note: '$NVCC' not found -- building CPU targets only." >&2
    echo "      Install the CUDA toolkit to build the GPU implementations." >&2
fi

echo "Built $built"
