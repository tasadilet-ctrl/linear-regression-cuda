// GPU-free correctness gate: checks gradient descent against the CLOSED-FORM
// ordinary-least-squares solution.
//
// Why this exists alongside scripts/verify_agreement.py: that script checks
// that all four implementations agree with each other, which is mutual
// consistency, not ground truth. They all share generateDataset() and the
// same gradient formula, so a mistake in the shared math would make all four
// agree on the same wrong answer and still "pass".
//
// This test is independent of that: it solves the normal equations
// (Z^T Z) theta = Z^T y directly in double precision (Z = [X | 1], so the
// intercept is fitted alongside the weights) via Gaussian elimination with
// partial pivoting, and asserts gradient descent converges to that answer.
// The two routes share no arithmetic beyond the dataset itself.
//
// It also needs no CUDA, so the math can be verified in CI and by anyone
// without an NVIDIA GPU -- everything else in this repo requires one.
//
// Scope, established by mutation testing rather than assumed: deleting the
// intercept update and flipping the sign of the residual are both caught.
// Rescaling the step size (e.g. dividing the gradient by n+1 instead of n)
// is NOT caught, and shouldn't be -- that changes only how fast gradient
// descent converges, not where it converges to, and the fixed point is what
// this test pins down. A learning-rate bug that still converges is invisible
// here; a bug in the gradient itself is not.
#include <cmath>
#include <cstdio>
#include <vector>

#include "linreg_common.h"

static int failures = 0;

// Solves the least-squares problem exactly via the normal equations.
// Returns theta of length d+1: [w_0..w_{d-1}, intercept].
static std::vector<double> solveOLS(const Dataset& ds) {
    const int n = ds.n, d = ds.d;
    const int m = d + 1;  // + intercept column

    // A = Z^T Z  (m x m),  rhs = Z^T y  (m), accumulated in double even though
    // the data is float, so the reference isn't limited by float precision.
    std::vector<double> A(static_cast<size_t>(m) * m, 0.0), rhs(m, 0.0);

    for (int i = 0; i < n; ++i) {
        const float* xi = &ds.X[static_cast<size_t>(i) * d];
        // z = [x_i, 1]
        for (int r = 0; r < m; ++r) {
            double zr = (r < d) ? xi[r] : 1.0;
            for (int c = 0; c < m; ++c) {
                double zc = (c < d) ? xi[c] : 1.0;
                A[static_cast<size_t>(r) * m + c] += zr * zc;
            }
            rhs[r] += zr * ds.y[i];
        }
    }

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < m; ++col) {
        int pivot = col;
        for (int r = col + 1; r < m; ++r)
            if (std::abs(A[static_cast<size_t>(r) * m + col]) >
                std::abs(A[static_cast<size_t>(pivot) * m + col])) pivot = r;

        if (pivot != col) {
            for (int c = 0; c < m; ++c)
                std::swap(A[static_cast<size_t>(col) * m + c], A[static_cast<size_t>(pivot) * m + c]);
            std::swap(rhs[col], rhs[pivot]);
        }

        double diag = A[static_cast<size_t>(col) * m + col];
        for (int r = col + 1; r < m; ++r) {
            double factor = A[static_cast<size_t>(r) * m + col] / diag;
            if (factor == 0.0) continue;
            for (int c = col; c < m; ++c)
                A[static_cast<size_t>(r) * m + c] -= factor * A[static_cast<size_t>(col) * m + c];
            rhs[r] -= factor * rhs[col];
        }
    }

    std::vector<double> theta(m, 0.0);
    for (int r = m - 1; r >= 0; --r) {
        double s = rhs[r];
        for (int c = r + 1; c < m; ++c) s -= A[static_cast<size_t>(r) * m + c] * theta[c];
        theta[r] = s / A[static_cast<size_t>(r) * m + r];
    }
    return theta;
}

static void checkCase(int n, int d, int epochs, float lr, unsigned seed, double tol) {
    Dataset ds = generateDataset(n, d, seed);

    std::vector<float> w;
    float b = 0.0f;
    trainCPU(ds, epochs, lr, w, b);

    std::vector<double> theta = solveOLS(ds);

    double max_diff = std::abs(static_cast<double>(b) - theta[d]);
    for (int j = 0; j < d; ++j)
        max_diff = std::max(max_diff, std::abs(static_cast<double>(w[j]) - theta[j]));

    bool ok = max_diff <= tol;
    if (!ok) ++failures;
    std::printf("  n=%-8d d=%-3d epochs=%-4d lr=%-5.3f | max|GD - OLS| = %.3e  (tol %.0e)  %s\n",
                n, d, epochs, lr, max_diff, tol, ok ? "PASS" : "FAIL");
}

int main() {
    std::printf("Gradient descent vs. closed-form OLS (normal equations, double precision):\n");

    // Features are iid N(0,1), so X^T X / n is close to the identity and the
    // GD update contracts the error by roughly (1 - lr) per epoch. At lr=0.05
    // over 500 epochs that is 0.95^500 ~ 7e-12, i.e. fully converged -- any
    // gap left is float round-off in the training loop, not incomplete
    // optimization. The tolerances below reflect that float noise floor.
    checkCase(20000,  10, 500, 0.05f, 42, 1e-4);
    checkCase(20000,  20, 500, 0.05f,  7, 1e-4);
    checkCase(5000,    5, 500, 0.05f, 99, 1e-4);
    checkCase(50000,  20, 500, 0.05f,  1, 1e-4);
    // Slower learning rate, so convergence relies on more epochs.
    checkCase(20000,  10, 800, 0.02f, 13, 1e-4);

    std::printf("\n%s\n", failures == 0 ? "GRADIENT DESCENT MATCHES THE ANALYTIC SOLUTION"
                                        : "FAILURES DETECTED");
    return failures == 0 ? 0 : 1;
}
