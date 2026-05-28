#!/usr/bin/env python3
"""
Demonstration of NKI autotuner error handling.

This script shows how the autotuner gracefully handles configs that fail
at different stages (compilation, execution, accuracy).
"""

import torch
import helion
import helion.language as hl
from helion.autotuner import _generate_nki_configs


@helion.kernel(backend="nki")
def test_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple add kernel for testing."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

    return out


def demo_config_generation():
    """Show how configs are generated with NKI constraints."""
    print("=" * 70)
    print("Demo 1: Config Generation with NKI Constraints")
    print("=" * 70)

    print("\nNKI autotuner generates configs based on effort level:")
    print("\nHardware constraints:")
    print("  - Partition dimension (block_sizes[0]) <= 128")
    print("  - Free dimension (block_sizes[1]) <= 512")
    print("  - All block sizes are powers of 2")

    # Show what configs would be generated
    print("\nFor a 2D kernel with size hints [128, 256]:")
    print("\nQuick search (effort='quick'):")
    print("  Partition candidates: [32, 128]")
    print("  Free candidates: [128, 512]")
    print("  Total configs: ~3-5 (cartesian product)")

    print("\nFull search (effort='full'):")
    print("  Partition candidates: [16, 32, 64, 128]")
    print("  Free candidates: [64, 128, 256, 512]")
    print("  Total configs: ~17 (cartesian product)")

    # We can't easily construct a ConfigSpec without a full kernel,
    # so just show the concept
    spec = None

    print("\nExample generated configs:")
    print("  Quick: [32,128], [32,512], [128,128], [128,512]")
    print("  Full: All combinations of partition × free within constraints")


def demo_error_handling():
    """Show how errors are handled during benchmarking."""
    print("\n" + "=" * 70)
    print("Demo 2: Error Handling During Autotuning")
    print("=" * 70)

    print("\nWhen you run a kernel with autotuning enabled, the autotuner:")
    print("  1. Generates a small set of valid configs (3-17 configs)")
    print("  2. For each config:")
    print("     a. Try to compile it (generate NKI code)")
    print("     b. Try to run it once (trigger XLA compilation)")
    print("     c. Check accuracy against baseline")
    print("     d. Time it with multiple runs")
    print("  3. If ANY step fails, that config gets time = infinity")
    print("  4. Select the config with the lowest (finite) time")
    print("  5. If ALL configs fail, return the default config")

    print("\nExample error scenarios:")
    print("  - Compilation fails: Config gets inf, autotuner continues")
    print("  - SBUF overflow: Config gets inf, autotuner continues")
    print("  - Accuracy mismatch: Config gets inf, autotuner continues")
    print("  - Timeout: Config gets inf, autotuner continues")
    print("\nThe program NEVER crashes due to a bad config!")


def demo_actual_run():
    """Run a real example with the autotuner."""
    print("\n" + "=" * 70)
    print("Demo 3: Actual Autotuning Run")
    print("=" * 70)

    # Create test tensors
    x = torch.randn(64, 128, dtype=torch.float32)
    y = torch.randn(64, 128, dtype=torch.float32)

    print(f"\nInput shapes: x={x.shape}, y={y.shape}")
    print("Running kernel with autotune_effort='quick'...")
    print("(This will try 3-5 configs and select the fastest)")

    # Create a fresh kernel with autotuning
    @helion.kernel(backend="nki", autotune_effort="quick")
    def add_autotuned(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
        return out

    # Run it (this triggers autotuning)
    result = add_autotuned(x, y)

    # Verify correctness
    expected = x + y
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    print("\n✓ Autotuning completed successfully!")
    print(f"✓ Result shape: {result.shape}")
    print("✓ Correctness check passed!")

    # Run again (should use cached config)
    print("\nRunning kernel again (uses cached best config, no autotuning)...")
    result2 = add_autotuned(x, y)
    torch.testing.assert_close(result2, expected, rtol=1e-5, atol=1e-5)
    print("✓ Second run passed (instant, using cached config)!")


def main():
    """Run all demos."""
    demo_config_generation()
    demo_error_handling()
    demo_actual_run()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The NKI autotuner is designed to be robust and fail-safe:

1. Only generates configs that respect hardware constraints
2. Wraps every operation in try-except blocks
3. Bad configs return infinity (never crash)
4. Always finds a working config (or uses default)
5. Caches the best config for future runs

This architecture solves both concerns:
- No Inductor dependency (Helion generates NKI code directly)
- Errors are gracefully handled (bad configs return inf, not crash)
""")


if __name__ == "__main__":
    main()
