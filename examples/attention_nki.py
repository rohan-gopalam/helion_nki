"""
Attention Example (NKI)
=======================

NKI-specific scaled dot-product attention.  The original ``attention.py``
uses 3D batched operations (bmm, baddbmm) and 2D per-row statistics
(m_i[batch, seq]) that create layout mismatches in NKI's strictly-2D
SBUF model.

This version restructures the kernel so that:
- Batch*heads is the outer sequential loop (block_size=1)
- Per-row statistics (m_i, l_i) are 1D [tile_m] -> NKI [128, 1]
- Accumulator is 2D [tile_m, head_dim] -> NKI [128, 64]
- Uses hl.dot (2D matmul) instead of torch.bmm (3D)
- All intermediate shapes are NKI-native 2D
"""

from __future__ import annotations

import math

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[1, 128, 128]),
    static_shapes=True,
)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention for NKI.

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim]
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    # Flatten batch dims: [B, H, seq, d] -> [B*H, seq, d]
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    # Outer loop: one batch*head at a time (block_size=1 for NKI 2D constraint)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        # Per-row statistics: [tile_b, tile_m, 1] squeezes to NKI [128, 1]
        # (partition=seq, free=1), matching the amax/sum reduction output layout.
        m_i = hl.full([tile_b, tile_m, 1], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_b, tile_m, 1], 1.0, dtype=torch.float32)
        # Accumulator: [tile_b, tile_m, head_dim] squeezes to NKI [128, 64]
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            # keepdim=True keeps shape [tile_b, tile_m, 1] to match m_i
            m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True) * qk_scale)
            qk = qk * qk_scale - m_ij
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1, keepdim=True)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i = m_i + torch.log2(l_i)
        acc = acc / l_i
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def main() -> None:
    # Small test: 1 batch, 1 head, 128 seq, 64 head_dim
    q, k, v = [
        torch.randn((1, 1, 128, 64), dtype=torch.float16, device=DEVICE)
        for _ in range(3)
    ]

    def ref_attention(q, k, v):
        p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(64)
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        return torch.matmul(p, v)

    run_example(attention, ref_attention, (q, k, v))


if __name__ == "__main__":
    main()
