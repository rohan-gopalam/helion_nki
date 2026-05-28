#!/usr/bin/env python3
"""
Spot test autotuning with kernels that are known to work.

Testing kernels that pass without autotuning to see if autotuning works correctly.
"""

import torch
import helion
import helion.language as hl


# Define all kernels at module level to avoid closure issues

@helion.kernel(backend="nki", autotune_effort="quick")
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a=x[tile_m, tile_k], b=y[tile_k, tile_n], beta=0.0, alpha=1.0)
        out[tile_m, tile_n] = acc

    return out


@helion.kernel(backend="nki", autotune_effort="quick")
def softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        row_max = torch.max(row)
        row = row - row_max
        exp_row = torch.exp(row)
        sum_exp = torch.sum(exp_row)
        out[tile_m, :] = (exp_row / sum_exp).to(x.dtype)

    return out


@helion.kernel(backend="nki", autotune_effort="quick")
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        sq = row * row
        mean_sq = torch.sum(sq) / n
        rms = torch.rsqrt(mean_sq + eps)
        normalized = row * rms
        out[tile_m, :] = (normalized * weight.to(torch.float32)).to(x.dtype)

    return out


def test_matmul():
    """Test matmul with autotuning."""
    print(f"\n{'=' * 70}")
    print("Testing: MatMul")
    print(f"{'=' * 70}")

    x = torch.randn(256, 256, dtype=torch.float32)
    y = torch.randn(256, 256, dtype=torch.float32)

    print("Running matmul with autotuning...")
    result = matmul(x, y)

    expected = torch.matmul(x, y)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    print("✓ MatMul PASSED")
    return True


def test_softmax():
    """Test softmax with autotuning."""
    print(f"\n{'=' * 70}")
    print("Testing: Softmax")
    print(f"{'=' * 70}")

    x = torch.randn(128, 256, dtype=torch.float32)

    print("Running softmax with autotuning...")
    result = softmax(x)

    expected = torch.nn.functional.softmax(x, dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("✓ Softmax PASSED")
    return True


def test_rms_norm():
    """Test RMS norm with autotuning."""
    print(f"\n{'=' * 70}")
    print("Testing: RMS Norm")
    print(f"{'=' * 70}")

    x = torch.randn(128, 256, dtype=torch.float32)
    weight = torch.ones(256, dtype=torch.float32)

    print("Running RMS norm with autotuning...")
    result = rms_norm(x, weight)

    # Reference implementation
    def ref_rms_norm(x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    expected = ref_rms_norm(x, weight)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
    print("✓ RMS Norm PASSED")
    return True


def main():
    """Run all tests sequentially."""
    print("\n" + "🔬" * 35)
    print("NKI Autotuner Spot Check - Working Kernels")
    print("🔬" * 35)
    print("\nTesting kernels that are known to work")
    print("to see if autotuning introduces any failures\n")

    tests = [
        ("MatMul", test_matmul),
        ("Softmax", test_softmax),
        ("RMS Norm", test_rms_norm),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, True))
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
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
        print("\n🎉 All tests passed! Autotuner works correctly.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        print("Check error messages above for details.")


if __name__ == "__main__":
    main()
