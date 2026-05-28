#!/usr/bin/env python3
"""
Test all 11 blocked kernels to find the EXACT compilation error for each.
"""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/helion_nki')

from helion._testing import check_example, import_path, EXAMPLES_DIR


def test_kernel(name, test_fn):
    """Run a single test and capture the exact error."""
    print("\n" + "=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)
    try:
        test_fn()
        print(f"✓ {name} WORKS (unexpected!)")
        return "WORKS", None
    except Exception as e:
        error_msg = str(e)
        # Extract the key error message
        if "Backend 'nki' is missing required implementation" in error_msg:
            # Find the operation name
            if "Aten lowering codegen not registered for" in error_msg:
                start = error_msg.find("for <OpOverload(op='")
                if start != -1:
                    start += len("for <OpOverload(op='")
                    end = error_msg.find("'", start)
                    op = error_msg[start:end]
                    return "MISSING_OP", f"Operation not supported: {op}"

        if "Backend 'nki' does not support" in error_msg:
            # Find what's not supported
            start = error_msg.find("does not support: ")
            if start != -1:
                start += len("does not support: ")
                end = error_msg.find("\n", start)
                if end == -1:
                    end = len(error_msg)
                reason = error_msg[start:end]
                return "NOT_SUPPORTED", reason

        # Just return first line of error
        first_line = error_msg.split('\n')[0]
        return "ERROR", first_line


# Test functions for each kernel

def test_fused_linear_jsd():
    """Test fused_linear_jsd.py"""
    # Read the example to understand what it needs
    mod = import_path(EXAMPLES_DIR / "fused_linear_jsd.py")

    # Small test case
    batch_size, seq_len = 2, 16
    hidden_dim = 64
    vocab_size = 128

    logits = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
    weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float32)
    target_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)

    # Try to run whatever function exists
    # Need to check what the actual function is
    result = mod.fused_linear_jsd_kernel(logits, weight, target_logits)


def test_gdn_fwd_h():
    """Test gdn_fwd_h.py"""
    mod = import_path(EXAMPLES_DIR / "gdn_fwd_h.py")

    batch = 2
    channels = 16
    height, width = 32, 32

    x = torch.randn(batch, channels, height, width, dtype=torch.float32)
    gamma = torch.randn(channels, dtype=torch.float32)
    beta = torch.randn(channels, dtype=torch.float32)

    result = mod.gdn_kernel(x, gamma, beta)


def test_grouped_gemm():
    """Test grouped_gemm.py"""
    mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")

    # Small grouped GEMM
    num_groups = 4
    M, K, N = 64, 64, 64

    x = [torch.randn(M, K, dtype=torch.float32) for _ in range(num_groups)]
    w = [torch.randn(K, N, dtype=torch.float32) for _ in range(num_groups)]

    result = mod.grouped_gemm_kernel(x, w)


def test_int4_gemm():
    """Test int4_gemm.py - we know this one fails with torch.stack"""
    M, K, N = 256, 512, 256
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B_unpacked = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    B_reshaped = B_unpacked.reshape(K // 2, 2, N).permute(1, 0, 2)
    B_packed = ((B_reshaped[0] & 0xF) | (B_reshaped[1] << 4)).to(torch.int8)
    B_unpacked_bf16 = B_unpacked.to(torch.bfloat16)
    expected = torch.matmul(A, B_unpacked_bf16)

    check_example(
        "int4_gemm",
        (A, B_packed),
        expected,
        fn_name="matmul_bf16_int4",
        block_sizes=[64, 64, 32],
    )


def test_jagged_hstu_attn():
    """Test jagged_hstu_attn.py"""
    mod = import_path(EXAMPLES_DIR / "jagged_hstu_attn.py")

    batch_size = 2
    num_heads = 4
    head_dim = 32
    seq_len = 64

    q = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=torch.float32)

    # Create jagged offsets
    lengths = torch.randint(10, seq_len, (batch_size,))
    offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])

    result = mod.jagged_hstu_attn_kernel(q, k, v, offsets)


def test_jagged_layer_norm():
    """Test jagged_layer_norm.py - we know this one"""
    num_rows, max_cols = 128, 64
    M = 8
    lengths = torch.randint(1, max_cols + 1, (num_rows,))
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, dim=0)])
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, dtype=torch.float32)
    eps = 1e-6

    mod = import_path(EXAMPLES_DIR / "jagged_layer_norm.py")
    expected = mod.reference_jagged_layer_norm_pytorch(x_data, x_offsets, eps)

    check_example(
        "jagged_layer_norm",
        (x_data, x_offsets, eps),
        expected,
        fn_name="jagged_layer_norm_kernel",
        block_sizes=[4, 8, 8, 8, 8, 8, 8],
    )


def test_layer_norm():
    """Test layer_norm.py - note: only backward pass is blocked"""
    mod = import_path(EXAMPLES_DIR / "layer_norm.py")

    batch_size = 32
    dim = 256

    x = torch.randn(batch_size, dim, dtype=torch.float16, requires_grad=True)
    weight = torch.ones(dim, dtype=torch.float16)
    bias = torch.zeros(dim, dtype=torch.float16)

    # Try forward
    out, mean, rstd = mod.layer_norm_fwd(x, [dim], weight, bias)

    # Try backward (this is what's blocked)
    grad_out = torch.randn_like(out)
    grad_x, grad_weight, grad_bias = mod.layer_norm_bwd(
        grad_out, x, mean, rstd, weight, compute_bias_grad=True
    )


def test_mamba2_chunk_scan():
    """Test mamba2_chunk_scan.py"""
    mod = import_path(EXAMPLES_DIR / "mamba2_chunk_scan.py")

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    chunk_size = 16

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32)
    A = torch.randn(nheads, dtype=torch.float32)

    result = mod.helion_mamba2_chunk_scan_kernel(x, dt, A, chunk_size)


def test_mamba2_chunk_state():
    """Test mamba2_chunk_state.py"""
    mod = import_path(EXAMPLES_DIR / "mamba2_chunk_state.py")

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    dstate = 64
    chunk_size = 16

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    B = torch.randn(batch, seqlen, nheads, dstate, dtype=torch.float32)

    result = mod.helion_mamba2_chunk_state_kernel(x, B, chunk_size)


def test_moe_matmul_ogs():
    """Test moe_matmul_ogs.py"""
    mod = import_path(EXAMPLES_DIR / "moe_matmul_ogs.py")

    batch_size = 8
    seq_len = 64
    hidden_dim = 128
    num_experts = 4

    x = torch.randn(batch_size * seq_len, hidden_dim, dtype=torch.float32)
    expert_weights = torch.randn(num_experts, hidden_dim, hidden_dim, dtype=torch.float32)
    expert_ids = torch.randint(0, num_experts, (batch_size * seq_len,))

    result = mod.moe_matmul_ogs_kernel(x, expert_weights, expert_ids)


def test_nvfp4_gemm():
    """Test nvfp4_gemm.py - same as int4_gemm"""
    M, K, N = 256, 512, 256
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8)

    check_example(
        "nvfp4_gemm",
        (A, B_packed),
        None,  # Skip accuracy for now
        fn_name="matmul_bf16_nvfp4",
        block_sizes=[64, 64, 32],
        skip_accuracy=True,
    )


def main():
    """Test all 11 blocked kernels."""
    print("\n" + "🔬" * 35)
    print("Testing All 11 Blocked Kernels")
    print("Finding EXACT compilation errors")
    print("🔬" * 35)

    tests = [
        ("fused_linear_jsd.py", test_fused_linear_jsd),
        ("gdn_fwd_h.py", test_gdn_fwd_h),
        ("grouped_gemm.py", test_grouped_gemm),
        ("int4_gemm.py", test_int4_gemm),
        ("jagged_hstu_attn.py", test_jagged_hstu_attn),
        ("jagged_layer_norm.py", test_jagged_layer_norm),
        ("layer_norm.py", test_layer_norm),
        ("mamba2_chunk_scan.py", test_mamba2_chunk_scan),
        ("mamba2_chunk_state.py", test_mamba2_chunk_state),
        ("moe_matmul_ogs.py", test_moe_matmul_ogs),
        ("nvfp4_gemm.py", test_nvfp4_gemm),
    ]

    results = []
    for name, test_fn in tests:
        error_type, error_msg = test_kernel(name, test_fn)
        results.append((name, error_type, error_msg))

    # Summary
    print("\n" + "=" * 70)
    print("EXACT ERRORS FOR ALL 11 BLOCKED KERNELS")
    print("=" * 70)

    for name, error_type, error_msg in results:
        print(f"\n{name}:")
        print(f"  Type: {error_type}")
        if error_msg:
            print(f"  Error: {error_msg}")


if __name__ == "__main__":
    main()
