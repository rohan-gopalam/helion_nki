#!/usr/bin/env python3
"""
Stress test the NKI autotuner with increasingly complex kernels.

This will help us see:
1. Which kernels trigger autotuning failures
2. How the error handling works in practice
3. What error messages we actually see
"""

import torch
import helion
import helion.language as hl
import logging
import sys

# Enable detailed logging to see all error messages
logging.basicConfig(level=logging.INFO, stream=sys.stderr)


def test_1_simple_elementwise():
    """Simplest case: element-wise add."""
    print("\n" + "=" * 70)
    print("TEST 1: Simple Element-wise Add")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
        return out

    try:
        x = torch.randn(128, 256, dtype=torch.float32)
        y = torch.randn(128, 256, dtype=torch.float32)
        result = add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        print("✓ TEST PASSED")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        return False


def test_2_matmul_small():
    """Matrix multiplication with small matrices."""
    print("\n" + "=" * 70)
    print("TEST 2: Small Matrix Multiplication")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.size()
        k2, n = b.size()
        out = torch.empty([m, n], dtype=a.dtype, device=a.device)

        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            out[tile_m, tile_n] = acc

        return out

    try:
        a = torch.randn(128, 128, dtype=torch.float32)
        b = torch.randn(128, 128, dtype=torch.float32)
        result = matmul(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        print("✓ TEST PASSED")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        return False


def test_3_matmul_large():
    """Matrix multiplication with larger matrices (may trigger SBUF issues)."""
    print("\n" + "=" * 70)
    print("TEST 3: Large Matrix Multiplication (Stress Test)")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def matmul_large(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.size()
        k2, n = b.size()
        out = torch.empty([m, n], dtype=a.dtype, device=a.device)

        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            out[tile_m, tile_n] = acc

        return out

    try:
        # Larger matrices - some configs might overflow
        a = torch.randn(512, 512, dtype=torch.float32)
        b = torch.randn(512, 512, dtype=torch.float32)
        print("Running with 512x512 matrices...")
        print("Watch for configs that might fail with SBUF overflow...")
        result = matmul_large(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        print("✓ TEST PASSED")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_softmax():
    """Softmax with reductions (more complex)."""
    print("\n" + "=" * 70)
    print("TEST 4: Softmax (Reductions)")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def softmax(x: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)

        for tile_m in hl.tile(m):
            row = x[tile_m, :].to(torch.float32)
            # Subtract max for numerical stability
            row_max = torch.max(row, dim=-1, keepdim=True)[0]
            row = row - row_max
            # Exp and sum
            exp_row = torch.exp(row)
            sum_exp = torch.sum(exp_row, dim=-1, keepdim=True)
            # Normalize
            out[tile_m, :] = (exp_row / sum_exp).to(x.dtype)

        return out

    try:
        x = torch.randn(128, 256, dtype=torch.float32)
        result = softmax(x)
        expected = torch.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
        print("✓ TEST PASSED")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_layer_norm():
    """Layer normalization (complex: multiple reductions)."""
    print("\n" + "=" * 70)
    print("TEST 5: Layer Normalization (Multiple Reductions)")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)

        for tile_m in hl.tile(m):
            row = x[tile_m, :].to(torch.float32)
            # Compute mean
            mean = torch.sum(row, dim=-1, keepdim=True) / n
            # Compute variance
            centered = row - mean
            var = torch.sum(centered * centered, dim=-1, keepdim=True) / n
            # Normalize
            rstd = torch.rsqrt(var + eps)
            normalized = centered * rstd
            # Apply affine
            out[tile_m, :] = (normalized * weight.to(torch.float32) + bias.to(torch.float32)).to(x.dtype)

        return out

    try:
        x = torch.randn(128, 256, dtype=torch.float32)
        weight = torch.ones(256, dtype=torch.float32)
        bias = torch.zeros(256, dtype=torch.float32)
        result = layer_norm(x, weight, bias)
        expected = torch.nn.functional.layer_norm(x, (256,), weight, bias)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        print("✓ TEST PASSED")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_fused_attention_simplified():
    """Simplified attention-like kernel (very complex)."""
    print("\n" + "=" * 70)
    print("TEST 6: Simplified Attention-like Kernel (Very Complex)")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def attention_simplified(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Simplified attention: Q @ K.T -> softmax -> @ V"""
        batch_size, seq_len, d_model = q.size()
        out = torch.empty([batch_size, seq_len, d_model], dtype=q.dtype, device=q.device)

        for tile_b in hl.tile(batch_size):
            for tile_i in hl.tile(seq_len):
                # Q[b, i, :] @ K[b, :, :].T = scores[i, seq_len]
                q_row = q[tile_b, tile_i, :].to(torch.float32)
                k_matrix = k[tile_b, :, :].to(torch.float32)

                # Compute attention scores
                scores = torch.matmul(q_row.unsqueeze(0), k_matrix.transpose(-2, -1))
                scores = scores / (d_model ** 0.5)

                # Softmax
                scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
                scores = scores - scores_max
                exp_scores = torch.exp(scores)
                sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
                attn_weights = exp_scores / sum_exp

                # Weighted sum of values
                v_matrix = v[tile_b, :, :].to(torch.float32)
                out[tile_b, tile_i, :] = torch.matmul(attn_weights, v_matrix).squeeze(0).to(q.dtype)

        return out

    try:
        batch_size = 2
        seq_len = 32
        d_model = 64

        q = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

        print(f"Running simplified attention: batch={batch_size}, seq={seq_len}, d={d_model}")
        result = attention_simplified(q, k, v)

        # Compute reference
        def reference_attention(q, k, v):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, v)

        expected = reference_attention(q, k, v)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
        print("✓ TEST PASSED")
        return True
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results."""
    print("\n" + "🔬" * 35)
    print("NKI Autotuner Stress Test Suite")
    print("🔬" * 35)
    print("\nThis will test progressively harder kernels to see:")
    print("  1. Which configs trigger failures")
    print("  2. How errors are handled")
    print("  3. Whether the autotuner recovers gracefully")

    tests = [
        ("Simple Elementwise", test_1_simple_elementwise),
        ("Small MatMul", test_2_matmul_small),
        ("Large MatMul", test_3_matmul_large),
        ("Softmax", test_4_softmax),
        ("Layer Norm", test_5_layer_norm),
        ("Simplified Attention", test_6_fused_attention_simplified),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Autotuner is robust.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        print("Check the error messages above to see what happened.")


if __name__ == "__main__":
    main()
