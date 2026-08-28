"""
Welford Example
===============

This example demonstrates how to implement a welford layernorm using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Welford Kernel Implementations
# ------------------------------


# %%
@helion.kernel(backend="nki", autotune_effort="none", config=helion.Config(block_sizes=[128, 128, 128]))
def welford(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    """
    Applies LayerNorm using Welford's algorithm for mean/variance.
    Args:
        weight: weight tensor of shape [N]
        bias: bias tensor of shape [N]
        x: input tensor of shape [M, N]
    Returns:
        Output tensor of shape [M, N]
    """
    m, n = x.size()

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        sum_x = torch.zeros_like(x[tile_m, 0], dtype=torch.float32)
        sum_x2 = torch.zeros_like(sum_x)

        # Accumulate sum and sum-of-squares in single-use pattern (no multi-user FX nodes)
        for tile_n in hl.tile(n):
            chunk = x[tile_m, tile_n]
            sum_x = sum_x + torch.sum(chunk, dim=-1)
            sum_x2 = sum_x2 + torch.sum(chunk * chunk, dim=-1)

        # Compute mean and variance from accumulated sums
        mean = sum_x / n
        # Var = E[x^2] - E[x]^2 = sum_x2/n - (sum_x/n)^2
        variance = sum_x2 / n - mean * mean
        rstd_tile = torch.rsqrt(torch.clamp(variance, min=0.0) + eps)
        mean_col = mean[:, None]
        rstd_col = rstd_tile[:, None]

        for tile_n in hl.tile(n):
            xi_chuck = x[tile_m, tile_n]
            w_chuck = weight[tile_n][None, :]
            b_chuck = bias[tile_n][None, :]

            y = (xi_chuck - mean_col) * rstd_col
            y = y * w_chuck + b_chuck

            out[tile_m, tile_n] = y.to(x.dtype)
    return out


# %%
# Baseline Function
# -----------------


# %%
def eager_layer_norm(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


# %%
# Verification Function
# ---------------------


# %%
def check(s: int, d: int) -> None:
    """
    Verify the welford kernel implementation against PyTorch's native layer_norm function.

    Args:
        s: First dimension of the test tensor
        d: Second dimension of the test tensor
    """

    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)

    kernels = {"helion": welford}
    run_example(kernels, eager_layer_norm, (weight, bias, x))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the welford kernel verification with different tensor sizes.
    """
    check(4096, 1024)
    check(4096, 1536)
    check(4096, 2048)


if __name__ == "__main__":
    main()
