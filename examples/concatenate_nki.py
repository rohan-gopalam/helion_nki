"""
Tensor Concatenation Example (NKI)
===================================

NKI-specific concatenation. The original ``concatenate.py`` uses ``extra_mask``
and ``torch.where`` to tile across the concat boundary, but NKI on trn1 has
no tensor comparison ops, no masked loads/stores, and no ``where`` primitive.

Instead, this version copies each input tensor independently via a tile-copy
kernel and concatenates on the host with ``torch.cat``.  The NKI kernel is a
simple identity copy -- the concatenation is just host-side bookkeeping.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128, 128]),
)
def _copy_tile(src: torch.Tensor) -> torch.Tensor:
    """Copy src to a new output tensor (identity kernel)."""
    m, n = src.size()
    out = torch.empty([m, n], dtype=src.dtype, device=src.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = src[tile_m, tile_n]
    return out


def concat2d_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Concatenates two 2D tensors along dimension 1 (columns).

    NKI on trn1 cannot mask or conditionally select elements, so tiling
    across the concat boundary is not possible.  Instead we copy each
    tensor through an NKI kernel and join them on the host.
    """
    x_copy = _copy_tile(x)
    y_copy = _copy_tile(y)
    return torch.cat([x_copy, y_copy], dim=1)


def main() -> None:
    # Dimensions must be multiples of block_size (128) since NKI has no masking
    x = torch.randn([256, 384], device=DEVICE)
    y = torch.randn([256, 384], device=DEVICE)
    run_example(concat2d_dim1, lambda x, y: torch.cat([x, y], dim=1), (x, y))


if __name__ == "__main__":
    main()
