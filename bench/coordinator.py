"""
Benchmark coordinator: dispatches examples across 4 NeuronCores in parallel,
collects timing results into a CSV.

Usage:
    python bench/coordinator.py --output results.csv
    python bench/coordinator.py --examples add.py,softmax.py --output results.csv
    python bench/coordinator.py --reps 20 --cores 4 --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BENCH_DIR = Path(__file__).parent
REPO_ROOT = BENCH_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
PYTHON = "/opt/aws_neuronx_venv_pytorch_2_9/bin/python"

PASSING_EXAMPLES = [
    "add.py", "attention.py", "batch_softmax.py", "bmm.py",
    "broadcast_matmul.py", "concatenate.py", "cross_entropy.py",
    "embedding.py", "exp.py", "fp8_gemm.py", "fused_nki_ops.py",
    "gather_gemv.py", "geglu.py", "jsd.py", "kl_div.py",
    "layer_norm_f32.py", "long_sum.py", "low_mem_dropout.py",
    "matmul.py", "matmul_layernorm.py", "matmul_split_k.py",
    "psum_reuse_minimal.py", "psum_reuse_test.py",
    "rms_norm.py", "softmax.py", "softmax_decomposed.py",
    "squeeze_and_excitation_net.py", "sum.py", "swiglu.py", "welford.py",
]


def run_worker(example: str, core_id: int, reps: int, warmup: int,
               cache_dir: str) -> dict:
    """Run a single benchmark worker on a pinned NeuronCore."""
    example_path = str(EXAMPLES_DIR / example)
    worker_path = str(BENCH_DIR / "worker.py")

    env = os.environ.copy()
    env["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    env["HELION_BACKEND"] = "nki"
    env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PATH"] = "/opt/aws_neuronx_venv_pytorch_2_9/bin:" + env.get("PATH", "")
    # Per-core cache to avoid contention
    env["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_dir}/core_{core_id}"

    cmd = [
        PYTHON, worker_path, example_path,
        "--reps", str(reps),
        "--warmup", str(warmup),
    ]

    t_start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env,
        )
        t_wall = time.time() - t_start

        if proc.returncode != 0:
            return {
                "example": example,
                "core_id": core_id,
                "status": "error",
                "error": proc.stderr[-2000:] if proc.stderr else "unknown",
                "wall_time_s": t_wall,
            }

        # Worker outputs JSON on stdout
        result = json.loads(proc.stdout)
        result["core_id"] = core_id
        result["wall_time_s"] = t_wall
        return result

    except subprocess.TimeoutExpired:
        return {
            "example": example,
            "core_id": core_id,
            "status": "timeout",
            "error": "Worker timed out (300s)",
            "wall_time_s": 300.0,
        }
    except Exception as e:
        return {
            "example": example,
            "core_id": core_id,
            "status": "error",
            "error": str(e),
            "wall_time_s": time.time() - t_start,
        }


def dispatch_parallel(examples: list[str], num_cores: int, reps: int,
                      warmup: int, cache_dir: str) -> list[dict]:
    """Dispatch examples across cores, running num_cores in parallel."""
    results = []
    # Process in batches of num_cores
    for batch_start in range(0, len(examples), num_cores):
        batch = examples[batch_start:batch_start + num_cores]
        batch_label = f"[{batch_start+1}-{batch_start+len(batch)}/{len(examples)}]"
        print(f"\n{batch_label} Running: {', '.join(batch)}", file=sys.stderr)

        futures = {}
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for i, example in enumerate(batch):
                core_id = i % num_cores
                future = executor.submit(
                    run_worker, example, core_id, reps, warmup, cache_dir
                )
                futures[future] = example

            for future in as_completed(futures):
                example = futures[future]
                result = future.result()
                results.append(result)
                status = result.get("status", "unknown")
                if status == "ok":
                    median = result.get("exec_median_ms", 0)
                    compile_t = result.get("compile_plus_first_exec_ms", 0)
                    print(f"  {example}: median={median:.2f}ms, "
                          f"compile+first={compile_t:.0f}ms", file=sys.stderr)
                else:
                    err = result.get("error", "")[:100]
                    print(f"  {example}: {status} - {err}", file=sys.stderr)

    return results


def write_csv(results: list[dict], output_path: str, tag: str | None = None):
    """Write results to CSV."""
    fieldnames = [
        "example", "status", "compile_plus_first_exec_ms",
        "exec_min_ms", "exec_median_ms", "exec_p99_ms",
        "reps", "wall_time_s", "core_id", "error",
    ]
    if tag:
        fieldnames.insert(1, "tag")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("example", "")):
            writer.writerow(r)

    print(f"\nResults written to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="NKI benchmark coordinator")
    parser.add_argument("--output", "-o", default="bench/results.csv",
                        help="Output CSV path")
    parser.add_argument("--examples", type=str, default=None,
                        help="Comma-separated list of examples (default: all passing)")
    parser.add_argument("--reps", type=int, default=10,
                        help="Timed repetitions per kernel")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations per kernel")
    parser.add_argument("--cores", type=int, default=4,
                        help="Number of NeuronCores to use in parallel")
    parser.add_argument("--cache-dir", default="/tmp/helion_nki_bench_cache",
                        help="Base directory for TorchInductor caches")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for this run (added as column in CSV)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    if args.examples:
        examples = [e.strip() for e in args.examples.split(",")]
    else:
        examples = PASSING_EXAMPLES

    # Validate examples exist
    missing = [e for e in examples if not (EXAMPLES_DIR / e).exists()]
    if missing:
        print(f"ERROR: Examples not found: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking {len(examples)} examples on {args.cores} cores, "
          f"{args.reps} reps each", file=sys.stderr)
    print(f"Cache dir: {args.cache_dir}", file=sys.stderr)

    if args.dry_run:
        print("\n[DRY RUN] Would run:", file=sys.stderr)
        for i, ex in enumerate(examples):
            core = i % args.cores
            print(f"  Core {core}: {ex}", file=sys.stderr)
        print(f"\nBatches: {(len(examples) + args.cores - 1) // args.cores}", file=sys.stderr)
        sys.exit(0)

    t_total_start = time.time()
    results = dispatch_parallel(
        examples, args.cores, args.reps, args.warmup, args.cache_dir
    )
    t_total = time.time() - t_total_start

    if args.tag:
        for r in results:
            r["tag"] = args.tag

    output_path = str(REPO_ROOT / args.output) if not os.path.isabs(args.output) else args.output
    write_csv(results, output_path, tag=args.tag)

    # Summary
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    err_count = len(results) - ok_count
    print(f"\nDone: {ok_count} ok, {err_count} failed, "
          f"total wall time: {t_total:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
