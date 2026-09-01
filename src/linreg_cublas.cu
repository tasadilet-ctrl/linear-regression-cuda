// cuBLAS-based version: the O(n*d) work (predictions = X @ w, gradient =
// X^T @ residual) is delegated to cublasSgemv, the vendor-tuned BLAS
// implementation. Only the trivial O(n) elementwise glue (residual = pred +
// b - y, and the final scalar update to b) is a custom kernel -- this is
// the realistic division of labor: hand-write the cheap glue, let BLAS
// handle the actual matrix-vector heavy lifting.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "linreg_common.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = call; \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, (int)st); \
        exit(1); \
    } \
} while (0)

// residual[i] = pred[i] + b - y[i]
__global__ void computeResidual(const float* pred, const float* b, const float* y, int n, float* residual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) residual[i] = pred[i] + *b - y[i];
}

__global__ void updateBias(float* b, const float* grad_b, int n, float lr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *b -= lr * (*grad_b) / n;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 100000;
    int d = argc > 2 ? std::atoi(argv[2]) : 20;
    int epochs = argc > 3 ? std::atoi(argv[3]) : 500;
    float lr = argc > 4 ? std::atof(argv[4]) : 0.05f;
    unsigned seed = argc > 5 ? std::atoi(argv[5]) : 42;

    Dataset ds = generateDataset(n, d, seed);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float *d_X, *d_y, *d_w, *d_b, *d_pred, *d_residual, *d_grad_w, *d_grad_b, *d_ones;
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(float) * n * d));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_residual, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_grad_w, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * n));

    CUDA_CHECK(cudaMemcpy(d_X, ds.X.data(), sizeof(float) * n * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, ds.y.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_w, 0, sizeof(float) * d));
    CUDA_CHECK(cudaMemset(d_b, 0, sizeof(float)));
    std::vector<float> ones(n, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_ones, ones.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;

    // cuBLAS is column-major; our X is row-major n x d. A row-major (n,d)
    // matrix is bit-for-bit the same buffer as a column-major (d,n) matrix,
    // i.e. X_row_major(n,d) == X_col_major(d,n)^T. So "pred = X @ w" (row-major
    // semantics) is computed as a column-major gemv with CUBLAS_OP_T on the
    // (d,n) matrix, and "grad_w = X^T @ residual" is CUBLAS_OP_N on the same
    // (d,n) matrix -- the transpose flags swap relative to what you'd expect
    // from row-major intuition alone.
    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // pred = X @ w  (X is (n,d) row-major == (d,n) col-major, so OP_T gives (n,d)*w)
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, d, n, &alpha, d_X, d, d_w, 1, &beta, d_pred, 1));
        computeResidual<<<blocks_n, threads>>>(d_pred, d_b, d_y, n, d_residual);

        // grad_w = X^T @ residual  (OP_N on the (d,n) col-major view gives X^T @ residual)
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, d, n, &alpha, d_X, d, d_residual, 1, &beta, d_grad_w, 1));
        // grad_b = sum(residual) = dot(residual, ones)
        CUBLAS_CHECK(cublasSdot(handle, n, d_residual, 1, d_ones, 1, d_grad_b));

        // w -= (lr/n) * grad_w
        float neg_lr_over_n = -lr / n;
        CUBLAS_CHECK(cublasSaxpy(handle, d, &neg_lr_over_n, d_grad_w, 1, d_w, 1));
        updateBias<<<1, 1>>>(d_b, d_grad_b, n, lr);
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

    std::fprintf(stderr, "cublas: n=%d d=%d epochs=%d -> mse=%.6f, max_weight_err=%.6f, %.4fs\n",
                 n, d, epochs, mse, max_w_err, secs);
    std::printf("method=cublas,n=%d,d=%d,epochs=%d,mse=%.8f,max_weight_err=%.8f,time=%.6f\n",
                n, d, epochs, mse, max_w_err, secs);

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_pred); cudaFree(d_residual); cudaFree(d_grad_w); cudaFree(d_grad_b); cudaFree(d_ones);
    cublasDestroy(handle);
    return 0;
}
