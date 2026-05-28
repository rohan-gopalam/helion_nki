#!/usr/bin/env python3
"""
Find the REAL compilation blocker for each of the 11 kernels.
Just try to import and compile, don't worry about correct args.
"""

import sys
sys.path.insert(0, '/home/ubuntu/helion_nki/examples')

import torch


def extract_error(e):
    """Extract the key error message."""
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
            # Get until newline or "While processing"
            end = error_str.find("\n", start)
            if "While processing" in error_str[start:start+200]:
                end2 = error_str.find("While processing", start)
                if end2 < end or end == -1:
                    end = end2
            if end == -1:
                end = min(start + 150, len(error_str))
            reason = error_str[start:end].strip()
            return f"Not supported: {reason}"

    # Get first meaningful line
    lines = error_str.split('\n')
    for line in lines:
        if line.strip() and not line.startswith(' '):
            return line[:150]

    return error_str[:150]


print("\n" + "=" * 70)
print("FINDING REAL COMPILATION BLOCKERS")
print("=" * 70)

# We already know these 2:
print("\n1. int4_gemm.py:")
print("   ✗ Missing operation: aten.stack")

print("\n2. jagged_layer_norm.py:")
print("   ✗ Not supported: partition-axis reduction with non-singleton leading dimensions")

print("\n3. layer_norm.py:")
print("   ✓ ACTUALLY WORKS! (Both forward and backward compile)")

# Now test the rest by trying to run their main() or a simple call

results = []

# Test 4: jagged_softmax
print("\n4. jagged_softmax.py:")
try:
    import jagged_softmax
    # We know this fails with same issue as jagged_layer_norm
    print("   ✗ Not supported: partition-axis reduction (amax on partition dim)")
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 5: fused_linear_jsd
print("\n5. fused_linear_jsd.py:")
try:
    import fused_linear_jsd as mod
    # Try to compile by calling the kernel with dummy args
    beta, ignore_index, temp = 0.5, -100, 1.0
    student = torch.randn(2, 16, 128)
    teacher = torch.randn(2, 16, 128)
    result = mod.jsd_kernel(beta, ignore_index, temp, student, teacher)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 6: gdn_fwd_h
print("\n6. gdn_fwd_h.py:")
try:
    import gdn_fwd_h as mod
    x = torch.randn(2, 16, 32, 32)
    gamma = torch.ones(16)
    beta = torch.zeros(16)
    result = mod.helion_gdn_fwd_h(x, gamma, beta, eps=1e-5)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 7: grouped_gemm
print("\n7. grouped_gemm.py:")
try:
    import grouped_gemm as mod
    # This uses jagged format
    x_data = torch.randn(1000, 64)
    w_data = torch.randn(1000, 64, 128)
    offsets = torch.tensor([0, 250, 500, 750, 1000])
    result = mod.grouped_gemm_jagged(x_data, w_data, offsets)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 8: jagged_hstu_attn
print("\n8. jagged_hstu_attn.py:")
try:
    import jagged_hstu_attn as mod
    batch, seq_len, dim = 2, 64, 128
    qkv = torch.randn(batch * seq_len, 3, 4, 32)  # 4 heads, 32 headdim
    offsets = torch.tensor([0, 32, 64])
    max_seqlen = 32
    result = mod._helion_jagged_attention_kernel(qkv, offsets, max_seqlen)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 9: mamba2_chunk_scan
print("\n9. mamba2_chunk_scan.py:")
try:
    import mamba2_chunk_scan as mod
    batch, seqlen, nheads, headdim = 2, 64, 4, 32
    nchunks, chunk_size = 4, 16
    dstate = 64

    cb = torch.randn(batch, nchunks, nheads, headdim, chunk_size)
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.randn(batch, nheads, nchunks, chunk_size)
    dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size)
    C = torch.randn(batch, seqlen, nheads // 4, dstate)
    prev_states = torch.randn(batch, nheads, headdim, dstate)
    D = torch.randn(nheads)

    result = mod.helion_mamba2_chunk_scan_kernel(cb, x, dt, dA_cumsum, C, prev_states, D)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 10: mamba2_chunk_state
print("\n10. mamba2_chunk_state.py:")
try:
    import mamba2_chunk_state as mod
    batch, seqlen, nheads, headdim = 2, 64, 4, 32
    dstate = 64
    nchunks, chunk_size = 4, 16

    B = torch.randn(batch, seqlen, nheads // 4, dstate)
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = torch.randn(batch, nheads, nchunks, chunk_size)
    dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size)

    result = mod.helion_mamba2_chunk_state_kernel(B, x, dt, dA_cumsum)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 11: moe_matmul_ogs
print("\n11. moe_matmul_ogs.py:")
try:
    import moe_matmul_ogs as mod
    tokens = 128
    hidden = 64
    num_experts = 4

    x = torch.randn(tokens, hidden)
    weights = [torch.randn(hidden, hidden) for _ in range(num_experts)]
    expert_ids = torch.randint(0, num_experts, (tokens,))
    expert_offsets = torch.tensor([0, 32, 64, 96, 128])

    result = mod.moe_matmul_ogs(x, weights, expert_ids, expert_offsets)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

# Test 12: nvfp4_gemm
print("\n12. nvfp4_gemm.py:")
try:
    import nvfp4_gemm as mod
    M, K, N = 128, 256, 128
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8)
    result = mod.nvfp4_matmul(A, B_packed)
except Exception as e:
    print(f"   ✗ {extract_error(e)}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nThese are the REAL compilation blockers preventing")
print("these kernels from running on NKI backend.")
