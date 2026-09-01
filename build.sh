#!/usr/bin/env bash
# Builds all four implementations. Requires CUDA toolkit (nvcc) + a CUDA GPU
# for the last three; run on a CUDA-capable machine.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin

NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-native}"

g++ -O3 -std=c++17 src/linreg_cpu.cpp -o bin/linreg_cpu
"$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cuda_naive.cu -o bin/linreg_cuda_naive
"$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cuda_optimized.cu -o bin/linreg_cuda_optimized
"$NVCC" -O3 -std=c++17 -arch="$ARCH" src/linreg_cublas.cu -lcublas -o bin/linreg_cublas

echo "Built bin/linreg_cpu, bin/linreg_cuda_naive, bin/linreg_cuda_optimized, bin/linreg_cublas"
