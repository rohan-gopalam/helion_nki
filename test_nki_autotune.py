#!/usr/bin/env python3
"""
Test script to understand NKI autotuning workflow.

This demonstrates:
1. How to create a simple Helion kernel with NKI backend
2. How to enable autotuning with different effort levels
3. How to inspect the generated configs
"""

import torch
import helion
import helion.language as hl

# Simple element-wise add kernel for testing
@helion.kernel(
    backend="nki",
    autotune_effort="quick",  # Options: "none", "quick", "full"
)
def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple element-wise addition."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

    return out


def main():
    # Create small test tensors on CPU (NKI launcher will move them to XLA device)
    batch_size = 128
    dim = 256

    x = torch.randn([batch_size, dim], dtype=torch.float32)
    y = torch.randn([batch_size, dim], dtype=torch.float32)

    print(f"Input shapes: x={x.shape}, y={y.shape}")
    print(f"Backend: NKI")
    print(f"Autotune effort: quick")

    # Run the kernel (this will trigger autotuning on first call)
    print("\nRunning kernel...")
    result = simple_add(x, y)

    # Verify correctness
    expected = x + y
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    print(f"\nResult shape: {result.shape}")
    print("Correctness check PASSED!")

    # Run again (should use cached best config)
    print("\nRunning kernel again (should use cached config)...")
    result2 = simple_add(x, y)
    torch.testing.assert_close(result2, expected, rtol=1e-5, atol=1e-5)
    print("Second run PASSED!")


if __name__ == "__main__":
    main()
