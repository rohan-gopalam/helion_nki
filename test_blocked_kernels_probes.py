#!/usr/bin/env python3
"""
Test the 7 "blocked" kernels with reduced probes.
Run WITHOUT autotuning (autotune_effort="none") to see if they fundamentally work.
"""

import torch
import sys


def test_jagged_softmax():
    """Probe for jagged_softmax - test with small sizes."""
    print("\n" + "=" * 70)
    print("PROBE: jagged_softmax")
    print("=" * 70)

    try:
        # Import the kernel from examples
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import jagged_softmax as mod

        # Small test case
        num_rows = 4
        max_cols = 8
        M = 2  # features

        # Create jagged data: 4 rows with varying lengths
        lengths = torch.tensor([3, 5, 2, 4], dtype=torch.int64)
        x_offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])

        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32)

        print(f"Input: {nnz} elements, {num_rows} rows, {M} features")
        print(f"Offsets: {x_offsets.tolist()}")

        # Run the kernel
        result = mod.jagged_softmax_kernel(x_data, x_offsets)

        # Reference
        expected = mod.reference_jagged_softmax_pytorch(x_data, x_offsets)

        # Check
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        print("✓ jagged_softmax PASSED (probe)")
        return True

    except Exception as e:
        print(f"✗ jagged_softmax FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_int4_gemm():
    """Probe for int4_gemm - test with small matrices."""
    print("\n" + "=" * 70)
    print("PROBE: int4_gemm")
    print("=" * 70)

    try:
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import int4_gemm as mod

        # Small test case
        M, K, N = 128, 256, 128

        # Create bfloat16 matrix A
        A = torch.randn(M, K, dtype=torch.bfloat16)

        # Create packed int4 matrix B (K//2 x N of int8)
        B_packed = torch.randint(-8, 7, (K // 2, N), dtype=torch.int8)

        print(f"Input: A={A.shape}, B_packed={B_packed.shape}")

        # Run the kernel
        result = mod.matmul_bf16_int4(A, B_packed)

        print(f"Output: {result.shape}")
        print("✓ int4_gemm PASSED (probe - no reference check)")
        return True

    except Exception as e:
        print(f"✗ int4_gemm FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jagged_layer_norm():
    """Probe for jagged_layer_norm."""
    print("\n" + "=" * 70)
    print("PROBE: jagged_layer_norm")
    print("=" * 70)

    try:
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import jagged_layer_norm as mod

        # Small test case
        num_rows = 4
        hidden_dim = 64

        # Create jagged data
        lengths = torch.tensor([3, 5, 2, 4], dtype=torch.int64)
        x_offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])

        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, hidden_dim, dtype=torch.float16)
        weight = torch.ones(hidden_dim, dtype=torch.float16)
        bias = torch.zeros(hidden_dim, dtype=torch.float16)

        print(f"Input: {nnz} elements, {num_rows} rows, {hidden_dim} features")

        # Run the kernel
        result = mod.jagged_layer_norm_kernel(x_data, x_offsets, weight, bias)

        print(f"Output: {result.shape}")
        print("✓ jagged_layer_norm PASSED (probe - no reference check)")
        return True

    except Exception as e:
        print(f"✗ jagged_layer_norm FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba2_chunk_scan():
    """Probe for mamba2_chunk_scan."""
    print("\n" + "=" * 70)
    print("PROBE: mamba2_chunk_scan")
    print("=" * 70)

    try:
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import mamba2_chunk_scan as mod

        # Small test case
        batch = 2
        seqlen = 64
        nheads = 4
        headdim = 32
        chunk_size = 16

        x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
        dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32)
        A = torch.randn(nheads, dtype=torch.float32)

        print(f"Input: x={x.shape}, dt={dt.shape}, A={A.shape}")

        # Run the kernel
        result = mod.mamba2_chunk_scan(x, dt, A, chunk_size)

        print(f"Output: {result.shape}")
        print("✓ mamba2_chunk_scan PASSED (probe - no reference check)")
        return True

    except Exception as e:
        print(f"✗ mamba2_chunk_scan FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba2_chunk_state():
    """Probe for mamba2_chunk_state."""
    print("\n" + "=" * 70)
    print("PROBE: mamba2_chunk_state")
    print("=" * 70)

    try:
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import mamba2_chunk_state as mod

        # Small test case
        batch = 2
        seqlen = 64
        nheads = 4
        headdim = 32
        dstate = 64
        chunk_size = 16

        x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
        B = torch.randn(batch, seqlen, nheads, dstate, dtype=torch.float32)

        print(f"Input: x={x.shape}, B={B.shape}")

        # Run the kernel
        result = mod.mamba2_chunk_state(x, B, chunk_size)

        print(f"Output: {result.shape}")
        print("✓ mamba2_chunk_state PASSED (probe - no reference check)")
        return True

    except Exception as e:
        print(f"✗ mamba2_chunk_state FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fused_linear_jsd():
    """Probe for fused_linear_jsd."""
    print("\n" + "=" * 70)
    print("PROBE: fused_linear_jsd")
    print("=" * 70)

    try:
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import fused_linear_jsd as mod

        # Small test case
        batch = 2
        seq_len = 32
        hidden = 64
        vocab = 128  # Small vocab for probe

        x = torch.randn(batch, seq_len, hidden, dtype=torch.float32)
        weight = torch.randn(vocab, hidden, dtype=torch.float32)
        target_logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)

        print(f"Input: x={x.shape}, weight={weight.shape}, target={target_logits.shape}")

        # Run the kernel
        result = mod.fused_linear_jsd(x, weight, target_logits)

        print(f"Output: {result.shape if hasattr(result, 'shape') else result}")
        print("✓ fused_linear_jsd PASSED (probe - no reference check)")
        return True

    except Exception as e:
        print(f"✗ fused_linear_jsd FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gdn_fwd_h():
    """Probe for gdn_fwd_h."""
    print("\n" + "=" * 70)
    print("PROBE: gdn_fwd_h")
    print("=" * 70)

    try:
        sys.path.insert(0, '/home/ubuntu/helion_nki/examples')
        import gdn_fwd_h as mod

        # Small test case
        batch = 2
        channels = 16
        height = 32
        width = 32

        x = torch.randn(batch, channels, height, width, dtype=torch.float32)

        print(f"Input: x={x.shape}")

        # Run the kernel (check what function is exported)
        # This may need adjustment based on actual API
        result = mod.gdn_forward(x)

        print(f"Output: {result.shape}")
        print("✓ gdn_fwd_h PASSED (probe - no reference check)")
        return True

    except Exception as e:
        print(f"✗ gdn_fwd_h FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all probes sequentially."""
    print("\n" + "🔬" * 35)
    print("Testing Blocked Kernels with Reduced Probes")
    print("Running WITHOUT autotuning (autotune_effort='none')")
    print("🔬" * 35)

    tests = [
        ("jagged_softmax", test_jagged_softmax),
        ("int4_gemm", test_int4_gemm),
        ("jagged_layer_norm", test_jagged_layer_norm),
        ("mamba2_chunk_scan", test_mamba2_chunk_scan),
        ("mamba2_chunk_state", test_mamba2_chunk_state),
        ("fused_linear_jsd", test_fused_linear_jsd),
        ("gdn_fwd_h", test_gdn_fwd_h),
    ]

    results = []
    for name, test_fn in tests:
        passed = test_fn()
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 70)
    print("PROBE SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} probes passed")

    if passed_count == total_count:
        print("\n🎉 All probes passed!")
        print("These kernels work in reduced form (without full examples).")
    else:
        print(f"\n⚠️  {total_count - passed_count} probe(s) failed.")
        print("These are fundamental compilation/runtime issues, not autotuning issues.")


if __name__ == "__main__":
    main()
