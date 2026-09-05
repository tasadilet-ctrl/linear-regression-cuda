#!/usr/bin/env python3
"""Plots 2-GPU speedup against dataset size, from benchmarks/multigpu.csv."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = list(csv.DictReader(open(ROOT / "benchmarks" / "multigpu.csv")))
    pts = sorted((int(r["n"]), float(r["speedup"])) for r in rows if int(r["gpus"]) == 2)
    xs = [n for n, _ in pts]
    ys = [s for _, s in pts]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(2.0, color="k", linestyle="--", alpha=0.4, label="ideal (2 GPUs)")
    ax.axhline(1.0, color="#d95f02", linestyle=":", alpha=0.8,
               label="break-even (1 GPU)")
    ax.plot(xs, ys, "o-", color="#1b9e77", label="measured 2-GPU speedup")

    # Mark the region where a second GPU actively costs time.
    ax.axvspan(min(xs), 3e5, color="#d95f02", alpha=0.07)
    ax.annotate("2nd GPU is a\nnet loss here", xy=(1.3e5, 1.35),
                color="#d95f02", fontsize=9, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (n samples, d=20, 500 epochs)")
    ax.set_ylabel("Speedup vs. 1 GPU")
    ax.set_title("Data-parallel scaling: the second GPU only pays off past ~300k samples")
    ax.set_ylim(0, 2.3)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")

    out = ROOT / "benchmarks" / "multigpu_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
