// Naive CUDA: one thread per sample. Each thread computes its own
// prediction and residual, then atomically adds its contribution to every
// weight's gradient accumulator. Simple and correct, but d atomicAdd calls
// per thread means heavy contention on the same d+1 memory locations from
// every one of the n threads, every iteration -- exactly the bottleneck the
// optimized version fixes.
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

__global__ void gradientStepNaive(const float* X, const float* y, const float* w, const float* b,
                                   int n, int d, float* grad_w, float* grad_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* xi = &X[static_cast<size_t>(i) * d];
    float pred = *b;
    for (int j = 0; j < d; ++j) pred += xi[j] * w[j];
    float err = pred - y[i];

    for (int j = 0; j < d; ++j) atomicAdd(&grad_w[j], err * xi[j]);
    atomicAdd(grad_b, err);
}

__global__ void applyUpdate(float* w, float* b, const float* grad_w, const float* grad_b, int d, int n, float lr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) w[j] -= lr * grad_w[j] / n;
    if (j == 0) *b -= lr * (*grad_b) / n;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 100000;
    int d = argc > 2 ? std::atoi(argv[2]) : 20;
    int epochs = argc > 3 ? std::atoi(argv[3]) : 500;
    float lr = argc > 4 ? std::atof(argv[4]) : 0.05f;
    unsigned seed = argc > 5 ? std::atoi(argv[5]) : 42;

    Dataset ds = generateDataset(n, d, seed);

    float *d_X, *d_y, *d_w, *d_b, *d_grad_w, *d_grad_b;
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(float) * n * d));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));

    // Time the host-to-device transfer separately from training. The training
    // timer below measures compute only; both numbers are reported so the
    // reader can judge whether the transfer is actually amortized by the
    // epoch count rather than having to take it on faith.
    cudaEvent_t xfer_start, xfer_stop;
    cudaEventCreate(&xfer_start);
    cudaEventCreate(&xfer_stop);
    cudaEventRecord(xfer_start);
    CUDA_CHECK(cudaMemcpy(d_X, ds.X.data(), sizeof(float) * n * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, ds.y.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    cudaEventRecord(xfer_stop);
    cudaEventSynchronize(xfer_stop);
    float xfer_ms = 0;
    cudaEventElapsedTime(&xfer_ms, xfer_start, xfer_stop);

    CUDA_CHECK(cudaMemset(d_w, 0, sizeof(float) * d));
    CUDA_CHECK(cudaMemset(d_b, 0, sizeof(float)));

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;
    int blocks_d = (d + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CUDA_CHECK(cudaMemset(d_grad_w, 0, sizeof(float) * d));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));
        gradientStepNaive<<<blocks_n, threads>>>(d_X, d_y, d_w, d_b, n, d, d_grad_w, d_grad_b);
        applyUpdate<<<blocks_d, threads>>>(d_w, d_b, d_grad_w, d_grad_b, d, n, lr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<float> w(d);
    float b;
    CUDA_CHECK(cudaMemcpy(w.data(), d_w, sizeof(float) * d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost));

    float mse = computeMSE(ds, w, b);
    float max_w_err = maxWeightError(w, b, ds);
    double secs = ms / 1000.0;

    double xfer_s = xfer_ms / 1000.0;
    std::fprintf(stderr, "cuda_naive: n=%d d=%d epochs=%d -> mse=%.6f, max_weight_err=%.6f, %.4fs compute (+%.4fs H2D transfer)\n",
                 n, d, epochs, mse, max_w_err, secs, xfer_s);
    std::printf("method=cuda_naive,n=%d,d=%d,epochs=%d,mse=%.8f,max_weight_err=%.8f,time=%.6f,transfer_time=%.6f\n",
                n, d, epochs, mse, max_w_err, secs, xfer_s);

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b); cudaFree(d_grad_w); cudaFree(d_grad_b);
    return 0;
}
