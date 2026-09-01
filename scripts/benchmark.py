#!/usr/bin/env python3
"""Times CPU, naive CUDA, optimized CUDA, and cuBLAS across a range of
dataset sizes (n), fixed d and epochs, to find the crossover points: where
does naive CUDA stop losing to CPU, and where (if anywhere) does cuBLAS's
per-call efficiency start to beat our fused custom kernel's fewer launches?
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def run(binary, n, d, epochs, lr=0.05, seed=42):
    result = subprocess.run([str(BIN / binary), str(n), str(d), str(epochs), str(lr), str(seed)],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} n={n} failed:\n{result.stderr}")
    m = re.search(r"mse=([\d.eE+-]+),max_weight_err=([\d.eE+-]+),time=([\d.eE+-]+)", result.stdout)
    # transfer_time is only reported by the CUDA binaries; the CPU path has
    # no host-to-device copy, so it's absent there rather than zero.
    xfer = re.search(r"transfer_time=([\d.eE+-]+)", result.stdout)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)),
            float(xfer.group(1)) if xfer else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                    default=[1000, 5000, 20000, 100000, 500000, 2000000, 8000000])
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--out-csv", default=str(ROOT / "benchmarks" / "results.csv"))
    args = p.parse_args()

    methods = [
        ("cpu", "linreg_cpu"),
        ("cuda_naive", "linreg_cuda_naive"),
        ("cuda_optimized", "linreg_cuda_optimized"),
        ("cublas", "linreg_cublas"),
    ]

    rows = []
    for n in args.sizes:
        print(f"\n=== n={n} ===")
        for name, binary in methods:
            mse, max_w_err, secs, xfer = run(binary, n, args.d, args.epochs)
            xfer_note = f"  (+{xfer:.4f}s H2D)" if xfer is not None else ""
            print(f"{name:16s}: {secs:.4f}s{xfer_note}  (mse={mse:.4f}, max_weight_err={max_w_err:.5f})")
            rows.append({"n": n, "method": name, "time": secs,
                          "transfer_time": "" if xfer is None else f"{xfer:.6f}",
                          "mse": mse, "max_weight_err": max_w_err})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "method", "time", "transfer_time", "mse", "max_weight_err"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
