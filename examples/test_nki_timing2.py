"""Test timing for different NKI block sizes to see if they're distinguishable."""
from __future__ import annotations

import os
import time
import statistics

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
from torch_xla.core import xla_model as xm

import helion
import helion.language as hl
from helion.runtime.config import Config

DEVICE = xm.xla_device()


def make_add_kernel(config):
    @helion.kernel(backend="nki", autotune_effort="none", config=config)
    def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = torch.broadcast_tensors(x, y)
        out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
        for tile in hl.tile(out.size()):
            out[tile] = x[tile] + y[tile]
        return out
    return add_fn


def bench_config(fn, x, y, warmup=2, repeats=5):
    for _ in range(warmup):
        xm.mark_step()
        fn(x, y)
        xm.mark_step()
    times = []
    for _ in range(repeats):
        xm.mark_step()
        t0 = time.perf_counter()
        fn(x, y)
        xm.mark_step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    x = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16)
    y = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16)

    configs_to_test = [
        Config(block_sizes=[16, 64]),
        Config(block_sizes=[32, 128]),
        Config(block_sizes=[64, 256]),
        Config(block_sizes=[128, 512]),
        Config(block_sizes=[16, 512]),
        Config(block_sizes=[128, 64]),
    ]

    print("Testing NKI timing for 1024x1024 float16 add kernel:")
    print("(using xm.mark_step() synchronization)")
    print()

    results = []
    for cfg in configs_to_test:
        fn = make_add_kernel(cfg)
        # First call compiles
        print(f"  Compiling {cfg}...")
        fn(x, y)
        xm.mark_step()
        ms = bench_config(fn, x, y, warmup=2, repeats=5)
        print(f"  block_sizes={cfg.block_sizes}: {ms:.3f}ms")
        results.append((cfg, ms))

    best_cfg, best_ms = min(results, key=lambda r: r[1])
    print(f"\nBest: {best_cfg.block_sizes} = {best_ms:.3f}ms")


if __name__ == "__main__":
    main()
