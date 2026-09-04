// Shared data generation and evaluation for every implementation (CPU,
// naive CUDA, optimized CUDA, cuBLAS). Plain C++, no CUDA dependency, so it
// compiles under both g++ and nvcc.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Synthetic regression dataset: y = X @ true_w + true_b + noise.
// X is row-major, n x d. Returns X, y, and the ground-truth weights so
// training can be checked against a known answer, not just "did the loss
// go down."
struct Dataset {
    std::vector<float> X;  // n x d, row-major
    std::vector<float> y;  // n
    std::vector<float> true_w;  // d
    float true_b;
    int n, d;
};

inline Dataset generateDataset(int n, int d, unsigned seed, float noise_std = 0.5f) {
    Dataset ds;
    ds.n = n;
    ds.d = d;
    ds.X.resize(static_cast<size_t>(n) * d);
    ds.y.resize(n);
    ds.true_w.resize(d);

    std::mt19937 rng(seed);
    std::normal_distribution<float> feat_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> w_dist(-3.0f, 3.0f);
    std::normal_distribution<float> noise_dist(0.0f, noise_std);

    for (int j = 0; j < d; ++j) ds.true_w[j] = w_dist(rng);
    ds.true_b = w_dist(rng);

    for (int i = 0; i < n; ++i) {
        float pred = ds.true_b;
        for (int j = 0; j < d; ++j) {
            float xij = feat_dist(rng);
            ds.X[static_cast<size_t>(i) * d + j] = xij;
            pred += xij * ds.true_w[j];
        }
        ds.y[i] = pred + noise_dist(rng);
    }
    return ds;
}

// The CPU reference training loop: full-batch gradient descent. Lives here
// rather than inside main() so the correctness test exercises the exact same
// code path the benchmark does, instead of a copy that could drift.
inline void trainCPU(const Dataset& ds, int epochs, float lr,
                     std::vector<float>& w, float& b) {
    const int n = ds.n, d = ds.d;
    w.assign(d, 0.0f);
    b = 0.0f;
    std::vector<float> grad_w(d);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::fill(grad_w.begin(), grad_w.end(), 0.0f);
        float grad_b = 0.0f;

        for (int i = 0; i < n; ++i) {
            const float* xi = &ds.X[static_cast<size_t>(i) * d];
            float pred = b;
            for (int j = 0; j < d; ++j) pred += xi[j] * w[j];
            float err = pred - ds.y[i];
            for (int j = 0; j < d; ++j) grad_w[j] += err * xi[j];
            grad_b += err;
        }

        for (int j = 0; j < d; ++j) w[j] -= lr * grad_w[j] / n;
        b -= lr * grad_b / n;
    }
}

inline float computeMSE(const Dataset& ds, const std::vector<float>& w, float b) {
    double sse = 0.0;
    for (int i = 0; i < ds.n; ++i) {
        double pred = b;
        for (int j = 0; j < ds.d; ++j) pred += ds.X[static_cast<size_t>(i) * ds.d + j] * w[j];
        double err = pred - ds.y[i];
        sse += err * err;
    }
    return static_cast<float>(sse / ds.n);
}

// Max absolute error between learned and ground-truth weights -- the real
// correctness check (low loss alone doesn't prove the model recovered the
// right coefficients, e.g. under multicollinearity; here features are iid
// so this is a fair, direct check).
inline float maxWeightError(const std::vector<float>& w, float b, const Dataset& ds) {
    float max_err = std::abs(b - ds.true_b);
    for (int j = 0; j < ds.d; ++j) max_err = std::max(max_err, std::abs(w[j] - ds.true_w[j]));
    return max_err;
}
