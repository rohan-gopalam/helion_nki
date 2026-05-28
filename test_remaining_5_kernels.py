#!/usr/bin/env python3
"""
Test the remaining 5 blocked kernels with CORRECT function signatures.
Find the EXACT compilation error for each.
"""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/helion_nki')

from helion._testing import check_example, import_path, EXAMPLES_DIR


def extract_key_error(e):
    """Extract the most important part of the error message."""
    error_str = str(e)

    # Check for missing operations
    if "Aten lowering codegen not registered for" in error_str:
        start = error_str.find("for <OpOverload(op='")
        if start != -1:
            start += len("for <OpOverload(op='")
            end = error_str.find("'", start)
            op = error_str[start:end]
            return f"Missing operation: {op}"

    # Check for unsupported features
    if "Backend 'nki' does not support:" in error_str:
        start = error_str.find("does not support: ")
        if start != -1:
            start += len("does not support: ")
            end = error_str.find("\n", start)
            if "While processing" in error_str[start:start+200]:
                end2 = error_str.find("While processing", start)
                if end2 < end or end == -1:
                    end = end2
            if end == -1:
                end = min(start + 200, len(error_str))
            reason = error_str[start:end].strip()
            return f"Not supported: {reason}"

    # MLIR verification errors
    if "Generated MLIR failed verification" in error_str:
        # Look for the specific verification error
        lines = error_str.split('\n')
        for line in lines:
            if "'nisa." in line or "partition" in line or "stationary" in line:
                return f"MLIR verification: {line.strip()}"
        return "MLIR verification failed"

    # Get first meaningful line
    lines = error_str.split('\n')
    for line in lines:
        if line.strip() and not line.startswith(' '):
            return line[:200]

    return error_str[:200]


print("\n" + "=" * 70)
print("TESTING REMAINING 5 BLOCKED KERNELS")
print("=" * 70)

# 1. fused_linear_jsd
print("\n1. fused_linear_jsd.py:")
try:
    mod = import_path(EXAMPLES_DIR / "fused_linear_jsd.py")

    # Test jsd_kernel (the simpler one)
    beta, ignore_index, temp = 0.5, -100, 1.0
    chunk_size, vocab_size = 16, 128
    student = torch.randn(chunk_size, vocab_size, dtype=torch.float32)
    teacher = torch.randn(chunk_size, vocab_size, dtype=torch.float32)

    result = mod.jsd_kernel(beta, ignore_index, temp, student, teacher)
    print("   ✓ WORKS! (unexpected)")
except Exception as e:
    error_msg = extract_key_error(e)
    print(f"   ✗ BLOCKED: {error_msg}")

# 2. gdn_fwd_h
print("\n2. gdn_fwd_h.py:")
try:
    mod = import_path(EXAMPLES_DIR / "gdn_fwd_h.py")

    batch, seqlen, nheads, dhead = 2, 64, 4, 32
    expand_v = 2
    dstate = expand_v * dhead
    chunk_size = 16

    k = torch.randn(batch, seqlen, nheads, dhead, dtype=torch.float32)
    w = torch.randn(batch, seqlen, nheads, dhead, dtype=torch.float32)
    u = torch.randn(batch, seqlen, nheads, dstate, dtype=torch.float32)
    g = torch.randn(batch, seqlen, nheads, dtype=torch.float32)

    result = mod.helion_gdn_fwd_h(k, w, u, g, chunk_size)
    print("   ✓ WORKS! (unexpected)")
except Exception as e:
    error_msg = extract_key_error(e)
    print(f"   ✗ BLOCKED: {error_msg}")

# 3. grouped_gemm
print("\n3. grouped_gemm.py:")
try:
    mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")

    # Create jagged grouped GEMM inputs
    total_M, K, N = 256, 128, 64
    num_groups = 4

    A_packed = torch.randn(total_M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    # Create group offsets (equal groups for simplicity)
    group_size = total_M // num_groups
    group_offsets = torch.arange(0, total_M + 1, group_size, dtype=torch.int64)

    result = mod.grouped_gemm_jagged(A_packed, B, group_offsets)
    print("   ✓ WORKS! (unexpected)")
except Exception as e:
    error_msg = extract_key_error(e)
    print(f"   ✗ BLOCKED: {error_msg}")

# 4. jagged_hstu_attn
print("\n4. jagged_hstu_attn.py:")
try:
    mod = import_path(EXAMPLES_DIR / "jagged_hstu_attn.py")

    max_seq_len = 64
    num_batches = 4
    num_heads = 4
    head_dim = 32
    alpha = 0.1

    # Total tokens across all batches
    total_tokens = num_batches * max_seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float32)

    # Create sequence offsets (equal sequences for simplicity)
    seq_offsets = torch.arange(0, total_tokens + 1, max_seq_len, dtype=torch.int64)

    result = mod._helion_jagged_attention_kernel(max_seq_len, alpha, q, k, v, seq_offsets)
    print("   ✓ WORKS! (unexpected)")
except Exception as e:
    error_msg = extract_key_error(e)
    print(f"   ✗ BLOCKED: {error_msg}")

# 5. mamba2_chunk_scan
print("\n5. mamba2_chunk_scan.py:")
try:
    mod = import_path(EXAMPLES_DIR / "mamba2_chunk_scan.py")

    batch, seqlen, nheads, headdim = 2, 64, 4, 32
    chunk_size = 16
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    ngroups = 2  # Typically nheads // 2 or similar
    dstate = 64

    cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, dtype=torch.float32)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    dt = torch.randn(batch, nheads, nchunks, chunk_size, dtype=torch.float32)
    dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, dtype=torch.float32)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32)
    prev_states = torch.randn(batch, nchunks, nheads, headdim, dstate, dtype=torch.float32)
    D = torch.randn(nheads, dtype=torch.float32)

    result = mod.helion_mamba2_chunk_scan_kernel(cb, x, dt, dA_cumsum, C, prev_states, D)
    print("   ✓ WORKS! (unexpected)")
except Exception as e:
    error_msg = extract_key_error(e)
    print(f"   ✗ BLOCKED: {error_msg}")

print("\n" + "=" * 70)
print("DONE - All 5 kernels tested with correct signatures")
print("=" * 70)
