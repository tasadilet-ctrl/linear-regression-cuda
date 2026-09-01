#!/usr/bin/env python3
"""Plots wall-clock time vs. dataset size for all four methods (log-log),
making the two crossovers visible: CPU-vs-GPU around n~20k, and
optimized-custom-kernel-vs-cuBLAS around n~1M."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = list(csv.DictReader(open(ROOT / "benchmarks" / "results.csv")))
    methods = {}
    for r in rows:
        methods.setdefault(r["method"], []).append((int(r["n"]), float(r["time"])))
    for m in methods:
        methods[m].sort()

    order = ["cpu", "cuda_naive", "cuda_optimized", "cublas"]
    labels = {
        "cpu": "CPU (serial)",
        "cuda_naive": "CUDA naive (global atomics)",
        "cuda_optimized": "CUDA optimized (shared-mem reduction)",
        "cublas": "cuBLAS (Sgemv + Sdot + Saxpy)",
    }
    colors = {"cpu": "#999999", "cuda_naive": "#d95f02", "cuda_optimized": "#1b9e77", "cublas": "#7570b3"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in order:
        if m not in methods:
            continue
        xs = [n for n, _ in methods[m]]
        ys = [t for _, t in methods[m]]
        ax.plot(xs, ys, "o-", color=colors[m], label=labels[m])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (n samples, d=20 features, 500 epochs)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Linear regression: CPU vs. CUDA vs. cuBLAS")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    out = ROOT / "benchmarks" / "time_vs_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
