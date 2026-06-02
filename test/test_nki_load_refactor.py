"""
Characterization tests for NKI load codegen paths.

These tests verify that the generated NKI code contains expected patterns
for each dimension-access path. They use `bound.to_triton_code(config)`
which generates the NKI Python source without requiring hardware.

Each test exercises one specific code path in the load codegen and asserts
key markers in the generated code. When refactoring, all tests must continue
to pass with identical or functionally-equivalent generated code.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("HELION_BACKEND", "nki")

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl


def _get_nki_source(kernel_fn, args, config=None):
    """Compile a kernel and return the generated NKI Python source string."""
    bound = kernel_fn.bind(args)
    if config is not None:
        bound.set_config(config)
    code = bound.to_triton_code(bound._config)
    return code


class TestNKILoadCodegen(unittest.TestCase):
    """Test the generated code patterns for various load access patterns."""

    # ─────────────────────────────────────────────────────────────────────
    # Path 1: Plain 2D contiguous load
    # ─────────────────────────────────────────────────────────────────────
    def test_2d_contiguous(self):
        """x[tile_m, tile_k] on [M, K] → plain DMA, no ap()"""

        @helion.kernel(config=helion.Config(block_sizes=[128, 64]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_k in hl.tile([M, K]):
                out[tile_m, tile_k] = x[tile_m, tile_k]
            return out

        x = torch.zeros(256, 128, device=DEVICE)
        src = _get_nki_source(kernel, (x,))
        self.assertIn("nisa.dma_copy", src)
        self.assertNotIn(".ap(", src)
        self.assertNotIn("__AP_ROW_GATHER__", src)
        self.assertNotIn("__DYN_AP__", src)

    # ─────────────────────────────────────────────────────────────────────
    # Path 2: 3D tensor with tile + scalar + slice (strided gather)
    # ─────────────────────────────────────────────────────────────────────
    def test_3d_strided_gather(self):
        """w[tile_s, head_scalar, :] on [S, H, D] → ap() strided gather"""

        @helion.kernel(config=helion.Config(block_sizes=[1, 32]))
        def kernel(w: torch.Tensor, nheads: int) -> torch.Tensor:
            S, H, D = w.shape
            H = hl.specialize(nheads)
            out = torch.empty(H, S, D, device=w.device, dtype=w.dtype)
            for tile_h, tile_s in hl.tile([H, S]):
                i_h = tile_h.id
                out[i_h, tile_s, :] = w[tile_s, i_h, :]
            return out

        w = torch.zeros(32, 4, 64, device=DEVICE)
        src = _get_nki_source(kernel, (w, 4))
        self.assertIn(".ap(", src)
        self.assertIn("_sg_scaled", src)
        self.assertIn("vector_offset", src)

    # ─────────────────────────────────────────────────────────────────────
    # Path 3: 3D contiguous flatten (no stride)
    # ─────────────────────────────────────────────────────────────────────
    def test_3d_flatten_contiguous(self):
        """x[tile_b, tile_m, :] on [B, M, K] → flat DMA, no ap()"""

        @helion.kernel(config=helion.Config(block_sizes=[2, 64, 128]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            B, M, K = x.shape
            out = torch.empty_like(x)
            for tile_b, tile_m, tile_k in hl.tile([B, M, K]):
                out[tile_b, tile_m, tile_k] = x[tile_b, tile_m, tile_k]
            return out

        x = torch.zeros(4, 128, 128, device=DEVICE)
        src = _get_nki_source(kernel, (x,))
        self.assertIn("nisa.dma_copy", src)
        # Should be a flat contiguous DMA, not ap()
        self.assertNotIn(".ap(", src)
        self.assertNotIn("__AP_ROW_GATHER__", src)

    # ─────────────────────────────────────────────────────────────────────
    # Path 4: Shifted tile subscript (concatenate-style)
    # ─────────────────────────────────────────────────────────────────────
    def test_2d_shifted_subscript(self):
        """y[tile0, tile1.index - C] → shifted DMA slice"""

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M = x.size(0)
            N = x.size(1) + y.size(1)
            out = torch.empty(M, N, device=x.device, dtype=x.dtype)
            for tile0, tile1 in hl.tile([M, N]):
                y_part = hl.load(
                    y, [tile0, tile1.index - x.size(1)],
                    extra_mask=(tile1.index >= x.size(1))[None, :]
                )
                out[tile0, tile1] = y_part
            return out

        x = torch.zeros(64, 32, device=DEVICE)
        y = torch.zeros(64, 64, device=DEVICE)
        src = _get_nki_source(kernel, (x, y))
        # Should contain a shifted slice expression (negative offset)
        self.assertIn("nisa.dma_copy", src)
        # The shifted subscript produces offset_1 - 32 in the slice
        self.assertIn("- 32", src)

    # ─────────────────────────────────────────────────────────────────────
    # Path 5: Extra mask → predicated copy
    # ─────────────────────────────────────────────────────────────────────
    def test_extra_mask(self):
        """hl.load with extra_mask → tensor_copy_predicated"""

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            out = torch.zeros_like(x)
            for tile_m, tile_k in hl.tile([M, K]):
                mask = tile_k.index < 16
                val = hl.load(x, [tile_m, tile_k], extra_mask=mask[None, :])
                out[tile_m, tile_k] = val
            return out

        x = torch.zeros(64, 64, device=DEVICE)
        src = _get_nki_source(kernel, (x,))
        self.assertIn("tensor_copy_predicated", src)
        self.assertIn("_nki_masked_load", src)

    # ─────────────────────────────────────────────────────────────────────
    # Path 6: 1D tensor load (reshapes to [1, N])
    # ─────────────────────────────────────────────────────────────────────
    def test_1d_load(self):
        """x[tile_n] on 1D [N] → reshapes to [1, N] partition access"""

        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            N = x.size(0)
            out = torch.empty_like(x)
            for tile_n in hl.tile(N):
                out[tile_n] = x[tile_n]
            return out

        x = torch.zeros(256, device=DEVICE)
        src = _get_nki_source(kernel, (x,))
        self.assertIn("nisa.dma_copy", src)
        # 1D tensors get reshaped to [1, N]
        self.assertIn("reshape", src)

    # ─────────────────────────────────────────────────────────────────────
    # Path 7: Scalar subscript (tile.begin on block_size=1)
    # ─────────────────────────────────────────────────────────────────────
    @unittest.skip("hl.grid/block_size=1 kernels don't expose config block_sizes")
    def test_scalar_subscript_tile_begin(self):
        """seq_offsets[tile_b.begin] with block_size=1 — tested via jagged examples"""
        pass


class TestNKILoadCodegenRuntime(unittest.TestCase):
    """Runtime correctness tests that verify actual values on Trainium."""

    def test_3d_strided_correctness(self):
        """Strided gather produces numerically correct results."""

        @helion.kernel(config=helion.Config(block_sizes=[1, 32]))
        def kernel(w: torch.Tensor, nheads: int) -> torch.Tensor:
            S, H, D = w.shape
            H = hl.specialize(nheads)
            out = torch.empty(H, S, D, device=w.device, dtype=w.dtype)
            for tile_h, tile_s in hl.tile([H, S]):
                i_h = tile_h.id
                out[i_h, tile_s, :] = w[tile_s, i_h, :]
            return out

        S, H, D = 4, 2, 4
        w = torch.arange(S * H * D, dtype=torch.float32, device=DEVICE).reshape(S, H, D)
        result = kernel(w, H)
        expected = w.permute(1, 0, 2)
        self.assertTrue(torch.allclose(result, expected),
                        f"Strided gather wrong:\n{result}\nvs\n{expected}")

    def test_3d_indirect_gather_codegen(self):
        """Jagged-style 3D load via _combine_leading_dims indirect case.

        When sub0 produces __AP_ROW_GATHER__ and sub1 is a scalar,
        _combine_leading_dims Case 1 builds flat_idx = gather*H + head.
        Verifies generated code contains _3d_flat_idx.
        """

        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def kernel(
            x: torch.Tensor,
            seq_offsets: torch.Tensor,
        ) -> torch.Tensor:
            num_batches = seq_offsets.size(0) - 1
            L = x.size(0)
            nheads = x.size(1)
            out = torch.empty_like(x)
            for tile_b, tile_h, tile_q in hl.tile(
                [num_batches, nheads, L], block_size=[1, 1, None]
            ):
                starts = seq_offsets[tile_b.begin]
                i_h = tile_h.id
                out[tile_q.index + starts, i_h, :] = x[tile_q.index + starts, i_h, :]
            return out

        L, H, D = 100, 2, 16
        x = torch.zeros(L, H, D, device=DEVICE)
        offsets = torch.zeros(5, device=DEVICE, dtype=torch.int32)
        src = _get_nki_source(kernel, (x, offsets))
        self.assertIn(".ap(", src)
        self.assertIn("_3d_flat_idx", src)

    def test_2d_contiguous_correctness(self):
        """Plain 2D copy is exact."""

        @helion.kernel(config=helion.Config(block_sizes=[128, 128]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_k in hl.tile([M, K]):
                out[tile_m, tile_k] = x[tile_m, tile_k]
            return out

        x = torch.randn(256, 128, device=DEVICE)
        result = kernel(x)
        self.assertTrue(torch.allclose(result, x))


if __name__ == "__main__":
    unittest.main()
