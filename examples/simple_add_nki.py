"""
Simple addition test for NKI backend - no reductions
"""

import torch
import helion
import helion.language as hl
from helion._testing import DEVICE, run_example


@helion.kernel
def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple element-wise addition."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

    return out


def main() -> None:
    batch_size = 128
    dim = 256

    x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)
    y = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float32)

    run_example(
        simple_add,
        lambda x, y: x + y,
        (x, y),
        rtol=1e-5,
        atol=1e-5,
    )
    print("Simple add test PASSED!")


if __name__ == "__main__":
    main()