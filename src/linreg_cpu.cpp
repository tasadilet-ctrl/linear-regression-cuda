// CPU baseline: batch gradient descent for multiple linear regression.
// Every iteration computes predictions and the gradient over the full
// dataset (not SGD/minibatch) so the CPU and GPU versions are solving the
// exact same optimization trajectory and can be compared step-for-step.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "linreg_common.h"

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 100000;
    int d = argc > 2 ? std::atoi(argv[2]) : 20;
    int epochs = argc > 3 ? std::atoi(argv[3]) : 500;
    float lr = argc > 4 ? std::atof(argv[4]) : 0.05f;
    unsigned seed = argc > 5 ? std::atoi(argv[5]) : 42;

    Dataset ds = generateDataset(n, d, seed);

    std::vector<float> w;
    float b = 0.0f;

    auto t0 = std::chrono::steady_clock::now();
    trainCPU(ds, epochs, lr, w, b);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    float mse = computeMSE(ds, w, b);
    float max_w_err = maxWeightError(w, b, ds);

    std::fprintf(stderr, "cpu: n=%d d=%d epochs=%d -> mse=%.6f, max_weight_err=%.6f, %.4fs\n",
                 n, d, epochs, mse, max_w_err, secs);
    std::printf("method=cpu,n=%d,d=%d,epochs=%d,mse=%.8f,max_weight_err=%.8f,time=%.6f\n",
                n, d, epochs, mse, max_w_err, secs);
    return 0;
}
