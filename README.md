# Linear Regression: CPU vs. CUDA vs. cuBLAS

[![CI](https://github.com/tasadilet-ctrl/linear-regression-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/tasadilet-ctrl/linear-regression-cuda/actions/workflows/ci.yml)

Batch gradient descent for multiple linear regression, implemented five ways — CPU baseline, naive CUDA (global-memory atomics), optimized CUDA (shared-memory reduction), cuBLAS, and data-parallel across two GPUs — with three real crossover points measured, not assumed: where GPU starts beating CPU, where handwritten CUDA stops beating cuBLAS, and where a second GPU stops being a liability and starts being a speedup.

![Time vs dataset size](benchmarks/time_vs_size.png)

Developed and benchmarked on an actual CUDA GPU (NVIDIA RTX PRO 6000 Blackwell, compute capability 12.0) via SSH to a remote box — every number below is a real measurement, not a projection. Hit and fixed a real bug along the way: an early draft dereferenced a device pointer from host code (`*d_b` outside a kernel) — illegal CUDA, caught before it was ever run, not after.

## Correctness first

All four implementations are checked against the *exact same* synthetic dataset (`y = X @ true_w + true_b + noise`, generated from a fixed seed) and compared on two things: final MSE, and max absolute error between the learned weights and the known ground-truth weights (not just "did the loss go down" — a model can have low loss without recovering the right coefficients). At every tested size from n=1,000 to n=8,000,000, **all four methods produce bit-identical MSE and weight-recovery error**. Same optimization trajectory, only the execution strategy differs — so any divergence would be a real bug (a race in the atomics, a wrong transpose flag in the cuBLAS calls, a lost sample in the block split).

"Bit-identical" means *across the four implementations built with one toolchain* — which is the claim that matters, since it isolates scheduling from arithmetic. The absolute values are toolchain-dependent: the same CPU baseline compiled with Apple clang instead of the g++ used on the benchmark box returns mse 0.24381480 rather than 0.24372667 at n=20,000, because the compilers contract and vectorize the reduction differently. If you rebuild elsewhere and see different digits, that's floating-point associativity, not a bug — which is why the correctness gates below compare against tolerances rather than hard-coded constants.

That's enforced by a script rather than left to the reader to eyeball:

```
$ python3 scripts/verify_agreement.py
  n=    1000  AGREE  mse=0.23874602 max_weight_err=0.02721381  [cpu, cublas, cuda_naive, cuda_optimized]
  n=  100000  AGREE  mse=0.24721767 max_weight_err=0.00386745  [cpu, cublas, cuda_naive, cuda_optimized]
  n= 8000000  AGREE  mse=0.25009486 max_weight_err=0.00030971  [cpu, cublas, cuda_naive, cuda_optimized]
  ...
All implementations agree exactly at all 7 dataset sizes.
```

### What the timings do and don't include

The reported times are **training compute only**; the one-time host-to-device copy of `X` and `y` is measured separately and reported alongside it (`transfer_time` in [`results.csv`](benchmarks/results.csv)). That's a deliberate choice — with 500 epochs the transfer is amortized — but it's measured rather than waved away: at the largest size, moving 672 MB costs 0.026s against 1.33s of compute for the optimized kernel (~2%) and 0.49s for cuBLAS (~5%). Including it would not move any crossover in this README.

## Finding 1: naive CUDA never wins

The naive kernel ([`linreg_cuda_naive.cu`](src/linreg_cuda_naive.cu)) assigns one thread per sample; every thread does `d` separate `atomicAdd`s directly into global memory to accumulate the gradient. That's `n × d` atomic operations contending on the same `d+1` memory locations, every single epoch. It loses to the single-threaded CPU baseline at *every* dataset size tested — including n=8,000,000, where it takes **72.6s against the CPU's 21.7s**, a 3.3x loss to a single core. Parallelism doesn't help if the synchronization overhead dominates the work.

## Finding 2: the fix, and where it stops being enough

[`linreg_cuda_optimized.cu`](src/linreg_cuda_optimized.cu) has each block accumulate its threads' contributions into **shared memory** first (still atomics, but low-latency and block-local), then issues just one global `atomicAdd` per block per feature — cutting global atomic traffic from O(n×d) to O((n/256)×d). Result: 31x faster than naive at n=100,000 (1.33s → 0.043s), and it now beats the single-threaded CPU by 6.8x even at this modest scale.

## Finding 3: our fused kernel beats cuBLAS... until it doesn't

[`linreg_cublas.cu`](src/linreg_cublas.cu) delegates the O(n×d) work to `cublasSgemv` (predictions and the weight gradient) and `cublasSdot` (bias gradient) — vendor-tuned BLAS, the same category of library that made `matmul-benchmark`'s naive loops look slow. But cuBLAS makes **5 separate GPU calls per epoch** (2× gemv, 1× custom residual kernel, 1× dot, 1× axpy) against the optimized kernel's **2** (gradient computation, then update). At small-to-medium sizes, that extra launch overhead costs more than BLAS's per-call efficiency saves:

| n | optimized (custom) | cuBLAS | winner |
|---|---|---|---|
| 1,000 | 0.039s | 0.055s | custom |
| 20,000 | 0.043s | 0.059s | custom |
| 100,000 | 0.043s | 0.070s | custom |
| 500,000 | 0.111s | 0.078s | cuBLAS (1.4x) |
| 2,000,000 | 0.338s | 0.135s | cuBLAS (2.5x) |
| 8,000,000 | 1.327s | 0.494s | cuBLAS (2.7x) |

The crossover lands between n=100,000 and n=500,000: below it, our fused two-launch kernel wins on overhead; above it, cuBLAS's actual matrix-vector throughput wins on scale. Neither "always write your own kernel" nor "always call the library" is the right takeaway — it depends on where the workload actually sits.

## Finding 4: the second GPU is a net loss until ~300k samples

The box has two RTX PRO 6000s, so [`linreg_cuda_multigpu.cu`](src/linreg_cuda_multigpu.cu) shards the dataset across them, computes a partial gradient per device, sums the partials, and applies an identical update everywhere — the standard data-parallel recipe, the same shape as PyTorch's DDP.

![Multi-GPU speedup](benchmarks/multigpu_speedup.png)

| n | 1 GPU | 2 GPUs | speedup |
|---|---|---|---|
| 100,000 | 0.050s | 0.059s | **0.84x** (slower) |
| 500,000 | 0.119s | 0.081s | 1.46x |
| 2,000,000 | 0.345s | 0.198s | 1.74x |
| 8,000,000 | 1.318s | 0.675s | 1.95x |
| 20,000,000 | 3.310s | 1.687s | 1.96x |

The gradient here is `d+1 = 21` floats, so the all-reduce moves 84 bytes per device per epoch — the bandwidth is irrelevant. What costs is the *synchronisation*: every epoch has to stop both devices, collect partials, and broadcast the sum. That fixed per-epoch latency is paid 500 times regardless of `n`, while the compute it saves grows with `n`. Below roughly 300k samples the sync costs more than the halved compute saves, and the second GPU actively makes training slower.

At 8M+ it reaches 1.95–1.96x, close to the 2x ceiling. The lesson is the same one as Finding 3, one level up: more hardware is not free, and whether it helps depends on where the workload sits relative to the fixed cost of coordinating it.

Both configurations are timed through the same binary with the same wall-clock timer, so the comparison isn't confounded by a different code path or timing method.

**On exactness:** sharding changes the order the gradient terms are summed, and float addition isn't associative, so divergence from the single-GPU result is permitted in principle. Measured, it doesn't happen here — the 2-GPU run matches bit-for-bit at every size tested. That's an empirical property of this workload (many same-signed terms of similar magnitude), not a guarantee, so the benchmark checks agreement against a tolerance rather than assuming it.


## Structure

```
src/
  linreg_common.h         # synthetic dataset generation, MSE, ground-truth weight check
  linreg_cpu.cpp          # CPU baseline
  linreg_cuda_naive.cu    # one thread/sample, global atomics
  linreg_cuda_optimized.cu# shared-memory block-level reduction
  linreg_cublas.cu        # cublasSgemv/Sdot/Saxpy
  linreg_cuda_multigpu.cu # data-parallel sharding across N GPUs
  test_cpu_correctness.cpp# GPU-free gate: gradient descent vs. closed-form OLS
scripts/
  benchmark.py            # all 4 methods across dataset sizes -> results.csv
  verify_agreement.py     # correctness gate: assert all 4 methods agree at every size
  plot_results.py         # the chart above
  benchmark_multigpu.py   # 1-GPU vs 2-GPU scaling -> multigpu.csv
  plot_multigpu.py        # the multi-GPU chart
build.sh
benchmarks/                # CSV + chart from the runs used in this README
```

## Build & run

`build.sh` builds the CPU targets on any machine and adds the GPU ones when `nvcc` is present, so the mathematics can be verified without CUDA hardware:

```bash
./build.sh              # CPU targets always; CUDA targets too if nvcc is found
CPU_ONLY=1 ./build.sh   # skip the CUDA targets even when nvcc is available

bin/test_cpu_correctness   # gradient descent vs. closed-form OLS -- no GPU needed
```

The three GPU implementations need the CUDA toolkit to build and an actual NVIDIA GPU to run — neither exists on macOS (no CUDA on Apple Silicon or any Mac GPU). Benchmarked with CUDA 13.0 on an RTX PRO 6000 (compute capability 12.0 / Blackwell). CI compiles them with `nvcc` on a GPU-less runner, which catches compile-time regressions but cannot execute them.

```bash
CUDA_ARCH=native ./build.sh   # or CUDA_ARCH=sm_XX for your GPU's compute capability

bin/linreg_cpu             1000000 20 500 0.05 42
bin/linreg_cuda_naive      1000000 20 500 0.05 42
bin/linreg_cuda_optimized  1000000 20 500 0.05 42
bin/linreg_cublas          1000000 20 500 0.05 42
bin/linreg_cuda_multigpu   1000000 20 500 0.05 42 2   # last arg = number of GPUs

python3 scripts/benchmark.py
python3 scripts/verify_agreement.py   # correctness gate; exits nonzero on any divergence
python3 scripts/plot_results.py

python3 scripts/benchmark_multigpu.py   # 1-GPU vs 2-GPU scaling
python3 scripts/plot_multigpu.py
```

## Possible extensions

- **cuSOLVER for the closed-form normal-equation solve** instead of gradient descent, as a fifth comparison point.
- **Mixed precision (TF32/FP16)** to see how much of the cuBLAS gap is precision-related vs. algorithmic.
