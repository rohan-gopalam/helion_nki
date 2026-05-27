"""Test NKI autotuner error recovery - bad configs should be skipped, not crash."""
from __future__ import annotations

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
from torch_xla.core import xla_model as xm

import helion
import helion.language as hl
from helion.runtime.config import Config

DEVICE = xm.xla_device()


# Use configs that include a potentially invalid one mixed with valid ones.
# The autotuner should skip bad configs and return the best valid one.
@helion.kernel(
    backend="nki",
    configs=[
        # Include a config that might overflow SBUF (large free dim with small
        # input - may or may not fail depending on hardware, but should not crash)
        Config(block_sizes=[128, 512]),
        Config(block_sizes=[64, 128]),
        Config(block_sizes=[32, 64]),
        Config(block_sizes=[16, 64]),
    ],
)
def add_with_configs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    x = torch.randn([256, 256], device=DEVICE, dtype=torch.float16)
    y = torch.randn([256, 256], device=DEVICE, dtype=torch.float16)

    print("Testing NKI autotune with explicit configs (including potentially bad ones)...")
    # force=False means "use the configs= list from the decorator"
    config = add_with_configs.autotune((x, y), force=False)
    print(f"Best config: {config}")

    # Verify the selected config works
    result = add_with_configs(x, y)
    xm.mark_step()
    expected = (x + y).cpu()
    result_cpu = result.cpu()
    torch.testing.assert_close(result_cpu, expected, atol=1e-2, rtol=1e-2)
    print("Error recovery test PASSED!")


if __name__ == "__main__":
    main()
