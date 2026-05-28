#!/usr/bin/env python3
"""
Spot test autotuning with kernels that are known to work.

These kernels pass without autotuning (autotune_effort="none").
Let's see if autotuning (autotune_effort="quick") introduces any failures.
"""

import torch
import sys

def test_kernel(name, test_fn):
    """Run a single test and report results."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"{'=' * 70}")
    try:
        test_fn()
        print(f"✓ {name} PASSED")
        return True
    except Exception as e:
        print(f"✗ {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matmul():
    """Test from examples/matmul.py"""
    import helion
    import helion.language as hl

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

    x = torch.randn(256, 256, dtype=torch.float32)
    y = torch.randn(256, 256, dtype=torch.float32)
    result = matmul(x, y)
    expected = torch.matmul(x, y)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


def test_softmax():
    """Test from examples/softmax.py"""
    import helion
    import helion.language as hl

    @helion.kernel(backend="nki", autotune_effort="quick")
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "Only dim=-1 supported"
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

    x = torch.randn(128, 256, dtype=torch.float32)
    result = softmax(x)
    expected = torch.nn.functional.softmax(x, dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_layer_norm_f32():
    """Test from examples/layer_norm_f32.py (this one passes)"""
    import helion
    import helion.language as hl

    @helion.kernel(backend="nki", autotune_effort="quick")
    def layer_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)

        for tile_m in hl.tile(m):
            row = x[tile_m, :].to(torch.float32)
            mean = torch.sum(row) / n
            centered = row - mean
            var = torch.sum(centered * centered) / n
            rstd = torch.rsqrt(var + eps)
            normalized = centered * rstd
            out[tile_m, :] = (normalized * weight.to(torch.float32) + bias.to(torch.float32)).to(x.dtype)

        return out

    x = torch.randn(128, 256, dtype=torch.float32)
    weight = torch.ones(256, dtype=torch.float32)
    bias = torch.zeros(256, dtype=torch.float32)
    result = layer_norm(x, weight, bias)
    expected = torch.nn.functional.layer_norm(x, (256,), weight, bias)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def test_rms_norm():
    """Test from examples/rms_norm.py"""
    import helion
    import helion.language as hl

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

    x = torch.randn(128, 256, dtype=torch.float32)
    weight = torch.ones(256, dtype=torch.float32)
    result = rms_norm(x, weight)

    # Reference implementation
    def ref_rms_norm(x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    expected = ref_rms_norm(x, weight)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def test_cross_entropy():
    """Test from examples/cross_entropy.py"""
    import helion
    import helion.language as hl

    @helion.kernel(backend="nki", autotune_effort="quick")
    def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.size()
        loss = torch.zeros([batch_size], dtype=torch.float32, device=logits.device)

        for tile_b in hl.tile(batch_size):
            row = logits[tile_b, :].to(torch.float32)
            label = labels[tile_b]

            # Log-softmax
            row_max = torch.max(row)
            row = row - row_max
            exp_row = torch.exp(row)
            sum_exp = torch.sum(exp_row)
            log_sum_exp = torch.log(sum_exp)
            log_softmax = row - log_sum_exp

            # Negative log likelihood
            loss[tile_b] = -log_softmax[label.to(torch.int32)]

        return loss

    batch_size = 32
    vocab_size = 128
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int64)

    result = cross_entropy(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def main():
    """Run all tests sequentially."""
    print("\n" + "🔬" * 35)
    print("NKI Autotuner Spot Check")
    print("Testing kernels that work WITHOUT autotuning")
    print("to see if autotuning introduces any failures")
    print("🔬" * 35)

    tests = [
        ("MatMul", test_matmul),
        ("Softmax", test_softmax),
        ("Layer Norm F32", test_layer_norm_f32),
        ("RMS Norm", test_rms_norm),
        ("Cross Entropy", test_cross_entropy),
    ]

    results = []
    for name, test_fn in tests:
        passed = test_kernel(name, test_fn)
        results.append((name, passed))

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
        print("\n🎉 All tests passed! Autotuner works correctly with known-good kernels.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed with autotuning enabled.")
        print("This means autotuning introduced failures in previously working kernels.")


if __name__ == "__main__":
    main()
