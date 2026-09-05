#!/usr/bin/env python3
"""Measures data-parallel scaling: how training time changes with the number
of GPUs, across dataset sizes.

Both configurations run through the SAME binary and the same wall-clock
timer, so the 1-GPU and 2-GPU numbers are directly comparable -- comparing
against linreg_cuda_optimized instead would confound the GPU count with a
different timing method (cudaEvent) and a different code path.
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "bin" / "linreg_cuda_multigpu"


def run(n, d, epochs, gpus, lr=0.05, seed=42):
    r = subprocess.run([str(BINARY), str(n), str(d), str(epochs), str(lr), str(seed), str(gpus)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gpus={gpus} n={n} failed:\n{r.stderr}")
    m = re.search(r"mse=([\d.eE+-]+),max_weight_err=([\d.eE+-]+),time=([\d.eE+-]+)", r.stdout)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[100000, 500000, 2000000, 8000000, 20000000])
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--max-gpus", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-csv", default=str(ROOT / "benchmarks" / "multigpu.csv"))
    args = p.parse_args()

    rows, failures = [], 0
    for n in args.sizes:
        print(f"\n=== n={n} ===")
        baseline = None
        ref_mse = None
        for g in range(1, args.max_gpus + 1):
            best = None
            for _ in range(args.repeats):
                mse, err, t = run(n, args.d, args.epochs, g)
                if best is None or t < best[2]:
                    best = (mse, err, t)
            mse, err, t = best
            if baseline is None:
                baseline, ref_mse = t, mse
                agree = "reference"
            else:
                # Sharding changes the summation order, so require agreement
                # to a tolerance rather than exact equality.
                agree = "exact" if mse == ref_mse else f"differs ({abs(mse-ref_mse):.2e})"
                if abs(mse - ref_mse) > 1e-6:
                    failures += 1
            speedup = baseline / t
            print(f"  gpus={g}: {t:.4f}s  speedup {speedup:.2f}x  mse={mse:.8f}  [{agree}]")
            rows.append({"n": n, "gpus": g, "time": t, "speedup": speedup,
                         "mse": mse, "max_weight_err": err, "agreement": agree})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "gpus", "time", "speedup", "mse",
                                          "max_weight_err", "agreement"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    if failures:
        print(f"WARNING: {failures} configuration(s) disagreed beyond tolerance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
