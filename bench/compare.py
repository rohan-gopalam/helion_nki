"""
Compare two benchmark result CSVs and report regressions/improvements.

Usage:
    python bench/compare.py bench/baseline.csv bench/experiment.csv
    python bench/compare.py bench/baseline.csv bench/experiment.csv --threshold 5
"""
from __future__ import annotations

import argparse
import csv
import sys


def load_csv(path: str) -> dict[str, dict]:
    """Load a benchmark CSV into a dict keyed by example name."""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                results[row["example"]] = {
                    "compile_ms": float(row.get("compile_plus_first_exec_ms", 0)),
                    "median_ms": float(row.get("exec_median_ms", 0)),
                    "min_ms": float(row.get("exec_min_ms", 0)),
                    "p99_ms": float(row.get("exec_p99_ms", 0)),
                }
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare NKI benchmark results")
    parser.add_argument("baseline", help="Baseline CSV")
    parser.add_argument("experiment", help="Experiment CSV")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Percent change threshold to highlight (default: 5%%)")
    parser.add_argument("--sort", choices=["name", "speedup", "regression"],
                        default="speedup", help="Sort order")
    args = parser.parse_args()

    base = load_csv(args.baseline)
    exp = load_csv(args.experiment)

    common = sorted(set(base.keys()) & set(exp.keys()))
    only_base = sorted(set(base.keys()) - set(exp.keys()))
    only_exp = sorted(set(exp.keys()) - set(base.keys()))

    if not common:
        print("ERROR: No common examples between the two CSVs.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for name in common:
        b = base[name]
        e = exp[name]
        if b["median_ms"] > 0:
            change_pct = ((e["median_ms"] - b["median_ms"]) / b["median_ms"]) * 100
        else:
            change_pct = 0.0
        rows.append({
            "example": name,
            "base_median_ms": b["median_ms"],
            "exp_median_ms": e["median_ms"],
            "change_pct": change_pct,
            "base_compile_ms": b["compile_ms"],
            "exp_compile_ms": e["compile_ms"],
        })

    if args.sort == "speedup":
        rows.sort(key=lambda r: r["change_pct"])
    elif args.sort == "regression":
        rows.sort(key=lambda r: -r["change_pct"])
    else:
        rows.sort(key=lambda r: r["example"])

    # Print table
    hdr = f"{'Example':<30} {'Base (ms)':<12} {'Exp (ms)':<12} {'Change':<10} {'Flag'}"
    print(hdr)
    print("-" * len(hdr))

    improvements = 0
    regressions = 0
    for r in rows:
        flag = ""
        if r["change_pct"] < -args.threshold:
            flag = "FASTER"
            improvements += 1
        elif r["change_pct"] > args.threshold:
            flag = "SLOWER"
            regressions += 1

        print(f"{r['example']:<30} {r['base_median_ms']:<12.3f} {r['exp_median_ms']:<12.3f} "
              f"{r['change_pct']:>+7.1f}%   {flag}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Compared: {len(common)} examples")
    print(f"Faster (>{args.threshold}%): {improvements}")
    print(f"Slower (>{args.threshold}%): {regressions}")
    print(f"Neutral: {len(common) - improvements - regressions}")

    if only_base:
        print(f"\nOnly in baseline: {', '.join(only_base)}")
    if only_exp:
        print(f"\nOnly in experiment: {', '.join(only_exp)}")

    # Geometric mean speedup
    import math
    ratios = [base[n]["median_ms"] / exp[n]["median_ms"]
              for n in common if exp[n]["median_ms"] > 0 and base[n]["median_ms"] > 0]
    if ratios:
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"\nGeometric mean speedup: {geomean:.3f}x")


if __name__ == "__main__":
    main()
