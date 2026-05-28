#!/usr/bin/env python3
"""
Run the EXACT probe code from test/test_examples.py
to see what actually works and what fails.
"""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/helion_nki')

from helion._testing import check_example, import_path, EXAMPLES_DIR


def test_jagged_softmax():
    """EXACT code from test/test_examples.py::test_jagged_softmax"""
    print("\n" + "=" * 70)
    print("PROBE: jagged_softmax (EXACT test from test_examples.py)")
    print("=" * 70)

    try:
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,))
        x_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(lengths, dim=0),
        ])
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32)
        args = (x_data, x_offsets)

        print(f"Input: {num_rows} rows, {nnz} total elements, {M} features")
        print(f"Max cols: {max_cols}")

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_softmax.py")
        expected = mod.reference_jagged_softmax_pytorch(x_data, x_offsets)

        print("Running kernel with block_sizes=[16, 8, 16, 16]...")
        code = check_example(
            "jagged_softmax",
            args,
            expected,
            fn_name="jagged_softmax_kernel",
            block_sizes=[16, 8, 16, 16],
        )

        print(f"✓ jagged_softmax PASSED (generated {len(code)} chars of code)")
        return True

    except Exception as e:
        print(f"✗ jagged_softmax FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_int4_gemm():
    """EXACT code from test/test_examples.py::test_int4_gemm"""
    print("\n" + "=" * 70)
    print("PROBE: int4_gemm (EXACT test from test_examples.py)")
    print("=" * 70)

    try:
        # Matrix dimensions
        M, K, N = 256, 512, 256

        # Create bfloat16 matrix A
        A = torch.randn(M, K, dtype=torch.bfloat16)

        # Create packed int4 matrix B
        # Generate random int4 values in range [-8, 7]
        B_unpacked = torch.randint(-8, 8, (K, N), dtype=torch.int8)

        # Pack two int4 values per int8
        B_reshaped = B_unpacked.reshape(K // 2, 2, N).permute(1, 0, 2)
        B_packed = ((B_reshaped[0] & 0xF) | (B_reshaped[1] << 4)).to(torch.int8)

        # Convert unpacked to bfloat16 for expected result
        B_unpacked_bf16 = B_unpacked.to(torch.bfloat16)
        expected = torch.matmul(A, B_unpacked_bf16)

        args = (A, B_packed)

        print(f"Input: A={A.shape} (bf16), B_packed={B_packed.shape} (int8)")
        print(f"Expected output: {expected.shape}")

        print("Running kernel with block_sizes=[64, 64, 32]...")
        code = check_example(
            "int4_gemm",
            args,
            expected,
            fn_name="matmul_bf16_int4",
            block_sizes=[64, 64, 32],
            num_warps=4,
            num_stages=3,
            rtol=2e-1,
        )

        print(f"✓ int4_gemm PASSED (generated {len(code)} chars of code)")
        return True

    except Exception as e:
        print(f"✗ int4_gemm FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jagged_layer_norm():
    """EXACT code from test/test_examples.py::test_jagged_layer_norm"""
    print("\n" + "=" * 70)
    print("PROBE: jagged_layer_norm (EXACT test from test_examples.py)")
    print("=" * 70)

    try:
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,))
        x_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(lengths, dim=0),
        ])
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32)
        eps = 1e-6
        args = (x_data, x_offsets, eps)

        print(f"Input: {num_rows} rows, {nnz} total elements, {M} features")

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_layer_norm.py")
        expected = mod.reference_jagged_layer_norm_pytorch(x_data, x_offsets, eps)

        print("Running kernel with block_sizes=[4, 8, 8, 8, 8, 8, 8]...")
        code = check_example(
            "jagged_layer_norm",
            args,
            expected,
            fn_name="jagged_layer_norm_kernel",
            block_sizes=[4, 8, 8, 8, 8, 8, 8],
        )

        print(f"✓ jagged_layer_norm PASSED (generated {len(code)} chars of code)")
        return True

    except Exception as e:
        print(f"✗ jagged_layer_norm FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the exact probes from test suite."""
    print("\n" + "🔬" * 35)
    print("Running EXACT Probes from test/test_examples.py")
    print("These are the tests that supposedly passed before")
    print("🔬" * 35)

    tests = [
        ("jagged_softmax", test_jagged_softmax),
        ("int4_gemm", test_int4_gemm),
        ("jagged_layer_norm", test_jagged_layer_norm),
    ]

    results = []
    for name, test_fn in tests:
        passed = test_fn()
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS - What Actually Works")
    print("=" * 70)

    for name, passed in results:
        status = "✓ WORKS" if passed else "✗ BLOCKED"
        print(f"{status:10} {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} probes work")

    if passed_count > 0:
        print(f"\n✓ {passed_count} kernel(s) actually compile and run correctly")
    if passed_count < total_count:
        print(f"✗ {total_count - passed_count} kernel(s) are genuinely blocked")


if __name__ == "__main__":
    main()
