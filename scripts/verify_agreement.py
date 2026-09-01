#!/usr/bin/env python3
"""Correctness gate. Asserts that every implementation converged to exactly
the same answer at every dataset size in benchmarks/results.csv.

All four methods run identical batch gradient descent on an identically
seeded dataset, so they should agree bit-for-bit -- they differ only in how
the work is scheduled across hardware, not in what arithmetic is performed.
Any divergence means a real bug (a race in the atomics, a wrong transpose
flag in the cuBLAS calls, a lost sample in the block split), so this is
checked rather than asserted in prose.

Run after scripts/benchmark.py. Exits nonzero on any disagreement.
"""
import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    csv_path = ROOT / "benchmarks" / "results.csv"
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f"No data in {csv_path} -- run scripts/benchmark.py first.")
        return 1

    by_n = collections.defaultdict(dict)
    for r in rows:
        by_n[int(r["n"])][r["method"]] = (r["mse"], r["max_weight_err"])

    failures = 0
    for n in sorted(by_n):
        results = by_n[n]
        distinct = set(results.values())
        methods = ", ".join(sorted(results))
        if len(distinct) == 1:
            mse, err = next(iter(distinct))
            print(f"  n={n:>8}  AGREE  mse={mse} max_weight_err={err}  [{methods}]")
        else:
            failures += 1
            print(f"  n={n:>8}  DISAGREE across methods:")
            for m, v in sorted(results.items()):
                print(f"      {m:16s} mse={v[0]} max_weight_err={v[1]}")

    print()
    if failures:
        print(f"{failures} dataset size(s) show disagreement between implementations.")
        return 1
    print(f"All implementations agree exactly at all {len(by_n)} dataset sizes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
