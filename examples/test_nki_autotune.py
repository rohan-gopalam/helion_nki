"""Test NKI autotuning on Trainium."""
from __future__ import annotations

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
from torch_xla.core import xla_model as xm

import helion
import helion.language as hl

DEVICE = xm.xla_device()


@helion.kernel(backend="nki", autotune_effort="quick")
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


def main() -> None:
    x = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)
    y = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)

    print("Starting NKI autotune for add kernel...")
    config = add.autotune((x, y))
    print(f"Best config found: {config}")

    # Verify correctness with best config
    result = add(x, y)
    xm.mark_step()
    expected = (x + y).cpu()
    result_cpu = result.cpu()
    torch.testing.assert_close(result_cpu, expected, atol=1e-2, rtol=1e-2)
    print("Correctness check PASSED!")


if __name__ == "__main__":
    main()
