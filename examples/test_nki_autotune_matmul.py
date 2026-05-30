"""Test NKI autotuning on matmul - a more complex kernel where block sizes matter."""
from __future__ import annotations

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
from torch_xla.core import xla_model as xm

import helion
import helion.language as hl

DEVICE = xm.xla_device()


@helion.kernel(backend="nki")
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def main() -> None:
    m, k, n = 256, 256, 256
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

    print(f"Starting NKI autotune for matmul ({m}x{k}x{n})...")
    config = matmul.autotune((x, y))
    print(f"Best config found: {config}")

    # Verify correctness
    result = matmul(x, y)
    xm.mark_step()
    result_cpu = result.cpu()
    expected = (x.cpu().float() @ y.cpu().float()).half()
    torch.testing.assert_close(result_cpu, expected, atol=1e-1, rtol=1e-1)
    print("Correctness check PASSED!")


if __name__ == "__main__":
    main()
