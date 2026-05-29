"""
INT4 General Matrix Multiplication (GEMM) with Helion
=====================================================
This example demonstrates an INT4 GEMM kernel implemented in Helion. The kernel performs
matrix multiplication where the second matrix B is packed with two 4-bit values per byte.
The kernel unpacks the int4 values, converts to bfloat16, and performs matmul with
the bfloat16 matrix A.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# INT4 GEMM Kernel
# ----------------


# %%
@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[32, 128, 128]),
    static_shapes=False,
)
def matmul_bf16_int4(A: Tensor, B: Tensor) -> Tensor:
    """
    BFloat16 x INT4 General Matrix Multiplication (GEMM).

    This kernel performs matrix multiplication where:
    - A is a bfloat16 matrix of shape [M, K]
    - B is an int8 matrix of shape [K//2, N] containing packed int4 values
      (two 4-bit values packed into each int8)

    Args:
        A (Tensor): Input tensor of shape [M, K] in bfloat16 format.
        B (Tensor): Packed int4 tensor of shape [K//2, N] in int8 format.

    Returns:
        Tensor: Output tensor of shape [M, N] in bfloat16 format.
    """
    M, K = A.shape
    _, N = B.shape

    C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)
    block_size_k_packed = hl.register_block_size(K // 2)

    # Use Helion to tile the computation
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k_packed in hl.tile(K // 2, block_size=block_size_k_packed):
            # Load packed int8 data from B (lo=first K//2, hi=second K//2)
            b_tile = B[tile_k_packed, tile_n]  # [k_packed, N]
            b_lo = ((b_tile << 4) >> 4).to(torch.float32)  # [k_packed, N] first-half rows
            b_hi = (b_tile >> 4).to(torch.float32)          # [k_packed, N] second-half rows

            # A[:, 0:k_packed] corresponds to b_lo, A[:, k_packed:2*k_packed] to b_hi
            # Packing: b_lo[k] = B_unpacked[k, :], b_hi[k] = B_unpacked[k + K//2, :]
            # A columns that pair with b_lo: A[:, k_begin:k_begin+k_packed]
            # A columns that pair with b_hi: A[:, K//2 + k_begin : K//2 + k_begin + k_packed]
            k_begin = tile_k_packed.begin
            k_packed = tile_k_packed.block_size
            K_half = K // 2
            a_lo = A[tile_m, k_begin : k_begin + k_packed].to(torch.float32)
            a_hi = A[tile_m, K_half + k_begin : K_half + k_begin + k_packed].to(torch.float32)
            acc = hl.dot(a_lo, b_lo, acc=acc)
            acc = hl.dot(a_hi, b_hi, acc=acc)

        C[tile_m, tile_n] = acc.to(torch.bfloat16)

    return C


# %%
# TritonBench Wrapper
# -------------------


# %%
def int4_gemm_tritonbench(tb_op: object, x: torch.Tensor, w: torch.Tensor) -> Callable:
    """
    Wrapper for TritonBench compatibility.

    Args:
        tb_op: TritonBench operator instance
        x (torch.Tensor): Left input tensor in bfloat16 format.
        w (torch.Tensor): Right input tensor of shape [K, N] containing int4 values.
                          Will be packed to int4 format.

    Returns:
        Callable: A function that performs the int4 gemm.
    """

    # Pack w to int4 format (two 4-bit values per int8 byte)
    x_2d = x.reshape(-1, x.size(-1))
    w_int8 = w.to(torch.int8)
    k = w_int8.shape[0]
    w_packed = ((w_int8[:k // 2, :] & 0xF) | (w_int8[k // 2:, :] << 4)).to(torch.int8)

    def run_kernel() -> torch.Tensor:
        return matmul_bf16_int4(x_2d, w_packed)

    return run_kernel


# %%
# Verification Function
# ---------------------


# %%
def _pack_int4_matrix(unpacked: torch.Tensor) -> torch.Tensor:
    """
    Pack int4 matrix into int8 container with two values per byte.
    Packing convention: lo nibble = first half of K, hi nibble = second half of K.
    This enables contiguous A-tile slicing in the NKI kernel.

    Args:
        unpacked (torch.Tensor): Tensor of shape [K, N] with values in [-8, 7].

    Returns:
        torch.Tensor: Packed tensor of shape [K//2, N] in int8 format.
    """
    k, n = unpacked.shape
    assert k % 2 == 0, "K dimension must be even for int4 packing"
    lo = unpacked[:k // 2, :]   # First half = lo nibbles
    hi = unpacked[k // 2:, :]   # Second half = hi nibbles
    return ((lo & 0xF) | (hi << 4)).to(torch.int8)


def _unpack_int4_matrix(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack an int4 matrix stored as two 4-bit values per int8 byte.
    Packing convention: lo=first_half, hi=second_half.

    Args:
        packed (torch.Tensor): Packed tensor of shape [K//2, N] in int8 format.

    Returns:
        torch.Tensor: Unpacked tensor of shape [K, N] in int8 format.
    """
    b_lo = ((packed << 4) >> 4).to(torch.int8)   # first K//2 rows
    b_hi = (packed >> 4).to(torch.int8)            # second K//2 rows
    return torch.cat([b_lo, b_hi], dim=0)          # [K, N]


def reference_matmul_bf16_int4(A: Tensor, B_packed: Tensor) -> Tensor:
    """
    Reference implementation that unpacks the int4 weights and performs matmul.

    Args:
        A (Tensor): Input tensor in bfloat16 format.
        B_packed (Tensor): Packed int4 tensor.

    Returns:
        Tensor: Output tensor in bfloat16 format.
    """
    B_unpacked = _unpack_int4_matrix(B_packed).to(torch.bfloat16)
    return torch.matmul(A, B_unpacked)


def check(m: int, k: int, n: int) -> None:
    """
    Test the INT4 GEMM implementation using the run_example utility.

    Args:
        m (int): Number of rows in the left input matrix.
        k (int): Shared dimension (must be even).
        n (int): Number of columns in the right input matrix.
    """
    A = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
    B_unpacked = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=DEVICE)
    B_packed = _pack_int4_matrix(B_unpacked)
    run_example(
        matmul_bf16_int4,
        reference_matmul_bf16_int4,
        (A, B_packed),
        rtol=2e-1,
        atol=1.0,
    )
    print(f"Test passed for shapes: M={m}, K={k}, N={n}")


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main function to run tests with different matrix sizes.
    """
    check(4, 8192, 7168)
    check(8192, 8192, 8192)


# %%
# Run Example
# -----------

# %%
if __name__ == "__main__":
    main()
