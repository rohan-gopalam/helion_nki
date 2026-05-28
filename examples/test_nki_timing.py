"""Test NKI timing methodology."""
from __future__ import annotations

import os
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
from torch_xla.core import xla_model as xm

import helion
import helion.language as hl
from helion.runtime.config import Config

DEVICE = xm.xla_device()


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=Config(block_sizes=[32, 128]),
)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def bench_many(fn, x, y, n=10):
    """Time multiple calls to understand timing variance."""
    times = []
    for i in range(n):
        xm.mark_step()
        t0 = time.perf_counter()
        fn(x, y)
        xm.mark_step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        print(f"  run {i}: {times[-1]:.2f}ms")
    return times


def main() -> None:
    x = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)
    y = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)

    print("Timing methodology test for NKI kernels")
    print("First call triggers compilation:")
    t0 = time.perf_counter()
    add(x, y)
    xm.mark_step()
    t1 = time.perf_counter()
    print(f"  First call+mark_step: {(t1-t0)*1000:.1f}ms")

    print("Subsequent calls (post-compilation):")
    times = bench_many(add, x, y, n=5)
    import statistics
    print(f"  Median: {statistics.median(times):.2f}ms")


if __name__ == "__main__":
    main()
