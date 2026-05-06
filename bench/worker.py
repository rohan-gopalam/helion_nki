"""
Benchmark worker: runs a single Helion NKI example on a pinned NeuronCore,
measures compile time and execution latency, outputs JSON to stdout.

Usage:
    NEURON_RT_VISIBLE_CORES=0 python bench/worker.py examples/add.py --reps 10

Design: We monkey-patch helion._testing.run_example BEFORE importing the example
module. Since the example's `from helion._testing import run_example` executes
during spec.loader.exec_module(), it picks up our patched version.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("HELION_BACKEND", "nki")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")


def benchmark_kernel(kernel_fn, args, reps: int, warmup: int) -> dict:
    """Benchmark a Helion NKI kernel on XLA/Neuron.

    First call includes XLA trace + neuronx-cc compile + execution.
    Subsequent calls use cached NEFF (pure execution).
    """
    import torch
    from torch_xla.core import xla_model as xm

    xm.mark_step()
    t0 = time.perf_counter()
    _ = kernel_fn(*args)
    xm.mark_step()
    t1 = time.perf_counter()
    compile_plus_first_exec_ms = (t1 - t0) * 1000

    for _ in range(warmup):
        _ = kernel_fn(*args)
        xm.mark_step()

    exec_times = []
    for _ in range(reps):
        t_start = time.perf_counter()
        _ = kernel_fn(*args)
        xm.mark_step()
        t_end = time.perf_counter()
        exec_times.append((t_end - t_start) * 1000)

    exec_times.sort()
    n = len(exec_times)
    median_idx = n // 2
    p99_idx = min(int(n * 0.99), n - 1)

    return {
        "compile_plus_first_exec_ms": round(compile_plus_first_exec_ms, 2),
        "exec_times_ms": [round(t, 3) for t in exec_times],
        "exec_median_ms": round(exec_times[median_idx], 3) if n else 0,
        "exec_p99_ms": round(exec_times[p99_idx], 3) if n else 0,
        "exec_min_ms": round(exec_times[0], 3) if n else 0,
        "reps": reps,
    }


def main():
    parser = argparse.ArgumentParser(description="NKI kernel benchmark worker")
    parser.add_argument("example", help="Path to example .py file")
    parser.add_argument("--reps", type=int, default=10, help="Number of timed repetitions")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--kernel-name", type=str, default=None,
                        help="Specific kernel name to benchmark (for multi-kernel examples)")
    args = parser.parse_args()

    example_path = os.path.abspath(args.example)
    if not os.path.exists(example_path):
        json.dump({"error": f"File not found: {example_path}", "status": "error"}, sys.stdout)
        sys.exit(1)

    result = {"example": os.path.basename(example_path), "path": example_path}

    # Patch run_example BEFORE importing the example module
    import helion._testing as _testing

    captured = {}

    def patched_run_example(kernel_fn, baseline_fn, args_tuple, **kwargs):
        # kernel_fn can be a single callable or a dict of {name: callable}
        if isinstance(kernel_fn, dict):
            captured["kernels"] = kernel_fn
        else:
            name = kwargs.get("kernel_name", getattr(kernel_fn, "__name__", "kernel"))
            captured["kernels"] = {name: kernel_fn}
        captured["baseline"] = baseline_fn
        captured["args"] = args_tuple

    _testing.run_example = patched_run_example

    try:
        spec = importlib.util.spec_from_file_location("_bench_example", example_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if hasattr(mod, "main"):
            mod.main()

        if "kernels" not in captured:
            result["error"] = "Could not capture kernel/args from run_example"
            result["status"] = "error"
            json.dump(result, sys.stdout)
            sys.exit(1)

        kernels = captured["kernels"]
        bench_args = captured["args"]

        # Pick which kernel to benchmark
        if args.kernel_name:
            if args.kernel_name not in kernels:
                result["error"] = f"Kernel '{args.kernel_name}' not found. Available: {list(kernels.keys())}"
                result["status"] = "error"
                json.dump(result, sys.stdout)
                sys.exit(1)
            target_kernels = {args.kernel_name: kernels[args.kernel_name]}
        else:
            target_kernels = kernels

        # Benchmark each kernel variant
        all_timings = {}
        for name, kernel_fn in target_kernels.items():
            timing = benchmark_kernel(kernel_fn, bench_args, reps=args.reps, warmup=args.warmup)
            all_timings[name] = timing

        if len(all_timings) == 1:
            # Single kernel: flatten into result
            only_name = next(iter(all_timings))
            result.update(all_timings[only_name])
            result["kernel_name"] = only_name
        else:
            # Multiple kernels: nest under "kernels" key
            result["kernels"] = all_timings

        result["status"] = "ok"

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-1500:]
        result["status"] = "error"

    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
