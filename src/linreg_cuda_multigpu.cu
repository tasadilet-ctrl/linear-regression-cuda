// Data-parallel multi-GPU training: the dataset is sharded across G devices,
// each computes a partial gradient over its own shard using the same
// shared-memory reduction kernel as linreg_cuda_optimized.cu, and the
// partial gradients are summed before every device applies an identical
// update.
//
// This is the standard data-parallel recipe (the same shape as DDP in
// PyTorch), at a scale small enough to see its costs clearly: the gradient
// here is only d+1 = 21 floats, so the all-reduce moves 84 bytes per device
// per epoch. The bytes are irrelevant; the per-epoch synchronisation is not.
// Whether a second GPU helps is therefore a question about latency versus
// per-device compute, which is what the benchmark measures.
//
// The reduction goes through the host rather than peer-to-peer copies. For
// 84 bytes that choice is immaterial (both are latency-bound and these GPUs
// are PCIe-attached, not NVLinked), and it keeps the synchronisation
// explicit and easy to reason about.
//
// Note on exactness: partial sums are combined across shards, so the
// summation ORDER differs from the single-device version, and float addition
// is not associative. Divergence from the single-GPU result is therefore
// permitted in principle.
//
// Measured, it does not occur here: this matches linreg_cuda_optimized
// bit-for-bit at n = 1e5, 5e5, 2e6 and 8e6. The gradient components are
// accumulated from many same-signed terms of similar magnitude, so the
// regrouping happens not to change any rounding decision at these sizes.
// That is an empirical observation about this workload, NOT a guarantee --
// a different d, a wider dynamic range, or a different shard count could
// break it, and the correctness gate compares against a tolerance rather
// than assuming exactness.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "linreg_common.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void gradientStepShard(const float* X, const float* y, const float* w, const float* b,
                                   int n_local, int d, float* grad_w, float* grad_b) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    for (int j = tid; j < d + 1; j += blockDim.x) shared[j] = 0.0f;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i < n_local) {
        const float* xi = &X[static_cast<size_t>(i) * d];
        float pred = *b;
        for (int j = 0; j < d; ++j) pred += xi[j] * w[j];
        float err = pred - y[i];
        for (int j = 0; j < d; ++j) atomicAdd(&shared[j], err * xi[j]);
        atomicAdd(&shared[d], err);
    }
    __syncthreads();

    for (int j = tid; j < d + 1; j += blockDim.x) {
        if (j < d) atomicAdd(&grad_w[j], shared[j]);
        else atomicAdd(grad_b, shared[d]);
    }
}

// Applies the globally-summed gradient. Every device runs this with the same
// inputs, so the parameters stay identical across devices without any
// further communication.
__global__ void applyUpdateGlobal(float* w, float* b, const float* grad_all,
                                   int d, int n_total, float lr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) w[j] -= lr * grad_all[j] / n_total;
    if (j == 0) *b -= lr * grad_all[d] / n_total;
}

struct Shard {
    int device;
    int n_local;
    int offset;
    float *X, *y, *w, *b, *grad_w, *grad_b, *grad_all;
};

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 100000;
    int d = argc > 2 ? std::atoi(argv[2]) : 20;
    int epochs = argc > 3 ? std::atoi(argv[3]) : 500;
    float lr = argc > 4 ? std::atof(argv[4]) : 0.05f;
    unsigned seed = argc > 5 ? std::atoi(argv[5]) : 42;
    int requested_gpus = argc > 6 ? std::atoi(argv[6]) : 2;

    int available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available));
    int G = requested_gpus < available ? requested_gpus : available;
    if (G < 1) { std::fprintf(stderr, "no CUDA devices\n"); return 1; }

    Dataset ds = generateDataset(n, d, seed);

    const int threads = 256;
    const int blocks_d = (d + threads - 1) / threads;
    const size_t shared_bytes = sizeof(float) * (d + 1);

    // Contiguous shards; the remainder is spread over the first few devices
    // so no device is more than one sample out of balance.
    std::vector<Shard> shards(G);
    int base = n / G, rem = n % G, offset = 0;
    for (int g = 0; g < G; ++g) {
        Shard& s = shards[g];
        s.device = g;
        s.n_local = base + (g < rem ? 1 : 0);
        s.offset = offset;
        offset += s.n_local;

        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&s.X, sizeof(float) * s.n_local * d));
        CUDA_CHECK(cudaMalloc(&s.y, sizeof(float) * s.n_local));
        CUDA_CHECK(cudaMalloc(&s.w, sizeof(float) * d));
        CUDA_CHECK(cudaMalloc(&s.b, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.grad_w, sizeof(float) * d));
        CUDA_CHECK(cudaMalloc(&s.grad_b, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.grad_all, sizeof(float) * (d + 1)));

        CUDA_CHECK(cudaMemcpy(s.X, &ds.X[static_cast<size_t>(s.offset) * d],
                              sizeof(float) * s.n_local * d, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.y, &ds.y[s.offset], sizeof(float) * s.n_local,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(s.w, 0, sizeof(float) * d));
        CUDA_CHECK(cudaMemset(s.b, 0, sizeof(float)));
    }

    std::vector<float> host_partial(static_cast<size_t>(G) * (d + 1));
    std::vector<float> host_total(d + 1);

    for (int g = 0; g < G; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Wall-clock rather than cudaEvent: the cost being measured spans several
    // devices plus the host-side reduction between them, which no single
    // device's timeline captures.
    auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Launch every shard's gradient kernel before synchronising any of
        // them, so the devices actually overlap instead of running serially.
        for (int g = 0; g < G; ++g) {
            Shard& s = shards[g];
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemsetAsync(s.grad_w, 0, sizeof(float) * d));
            CUDA_CHECK(cudaMemsetAsync(s.grad_b, 0, sizeof(float)));
            int blocks_n = (s.n_local + threads - 1) / threads;
            gradientStepShard<<<blocks_n, threads, shared_bytes>>>(
                s.X, s.y, s.w, s.b, s.n_local, d, s.grad_w, s.grad_b);
        }

        for (int g = 0; g < G; ++g) {
            Shard& s = shards[g];
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
            float* dst = &host_partial[static_cast<size_t>(g) * (d + 1)];
            CUDA_CHECK(cudaMemcpy(dst, s.grad_w, sizeof(float) * d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dst + d, s.grad_b, sizeof(float), cudaMemcpyDeviceToHost));
        }

        for (int j = 0; j < d + 1; ++j) {
            float acc = 0.0f;
            for (int g = 0; g < G; ++g) acc += host_partial[static_cast<size_t>(g) * (d + 1) + j];
            host_total[j] = acc;
        }

        for (int g = 0; g < G; ++g) {
            Shard& s = shards[g];
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(s.grad_all, host_total.data(),
                                  sizeof(float) * (d + 1), cudaMemcpyHostToDevice));
            applyUpdateGlobal<<<blocks_d, threads>>>(s.w, s.b, s.grad_all, d, n, lr);
        }
    }

    for (int g = 0; g < G; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Every device holds the same parameters; read them back from device 0.
    std::vector<float> w(d);
    float b;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(w.data(), shards[0].w, sizeof(float) * d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&b, shards[0].b, sizeof(float), cudaMemcpyDeviceToHost));

    float mse = computeMSE(ds, w, b);
    float max_w_err = maxWeightError(w, b, ds);

    std::fprintf(stderr, "cuda_multigpu: n=%d d=%d epochs=%d gpus=%d -> mse=%.6f, max_weight_err=%.6f, %.4fs\n",
                 n, d, epochs, G, mse, max_w_err, secs);
    std::printf("method=cuda_multigpu,n=%d,d=%d,epochs=%d,gpus=%d,mse=%.8f,max_weight_err=%.8f,time=%.6f\n",
                n, d, epochs, G, mse, max_w_err, secs);

    for (int g = 0; g < G; ++g) {
        Shard& s = shards[g];
        CUDA_CHECK(cudaSetDevice(g));
        cudaFree(s.X); cudaFree(s.y); cudaFree(s.w); cudaFree(s.b);
        cudaFree(s.grad_w); cudaFree(s.grad_b); cudaFree(s.grad_all);
    }
    return 0;
}
