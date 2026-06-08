"""
Codegen-snapshot tests for the AP Sentinel refactor.

The refactor replaces three sentinel strings (__AP_ROW_GATHER__, __AP_VEC_OFFSET__,
__DYN_AP__) with two typed dataclasses (IndirectAP, DynamicAP) in slice_parts. The
emitted NKI code must be bit-for-bit identical before and after.

Each test:
  1. Compiles a kernel that exercises exactly one sentinel path.
  2. Asserts that the sentinel strings are absent from the generated code
     (they were internal implementation details and must never appear in output).
  3. Asserts that the expected NKI constructs (.ap(), nisa.dma_copy, etc.) are present.
  4. Uses assertExpectedJournal to snapshot the full generated code — any deviation
     from the pre-refactor output will cause the test to fail.

Snapshot tests use _nki_code() (compile only) so they work without hardware and don't
depend on kernel correctness.  Correctness is verified in *_correctness tests via
code_and_output() which also executes the kernel.

Run with EXPECTTEST_ACCEPT=1 to capture initial baselines before the refactor; the
snapshots then serve as regression guards.

Backend guard: all tests require HELION_BACKEND=nki.  The @onlyBackends decorator
skips the entire class on other backends.
"""

from __future__ import annotations

import os

os.environ.setdefault("HELION_BACKEND", "nki")

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nki_code(kernel_fn: helion.Kernel, args: tuple) -> str:
    """Compile a kernel and return the generated NKI source string (no execution)."""
    bound = kernel_fn.bind(args)
    return bound.to_triton_code(bound._config)


def _assert_no_sentinels(test: TestCase, code: str) -> None:
    """Assert that no raw sentinel strings leaked into the generated code."""
    test.assertNotIn("__AP_ROW_GATHER__", code)
    test.assertNotIn("__AP_VEC_OFFSET__", code)
    test.assertNotIn("__DYN_AP__", code)


# ---------------------------------------------------------------------------
# Path 1: __AP_ROW_GATHER__ — embedding-style 1D index gather (load)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestRowGatherLoad(TestCase):
    """
    Covers the __AP_ROW_GATHER__ sentinel emitted by _nki_row_index_gather and
    consumed in _build_hbm_src (row-gather path, pattern=None).

    The kernel does weight[idx[tile_b], tile_e] where idx is a 1-D int tensor —
    the classic embedding lookup pattern.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 128]),
        )
        def embedding_lookup(
            idx: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            B = idx.size(0)
            D = weight.size(1)
            out = torch.empty([B, D], dtype=weight.dtype, device=weight.device)
            for tile_b, tile_e in hl.tile([B, D]):
                out[tile_b, tile_e] = weight[idx[tile_b], tile_e]
            return out

        return embedding_lookup

    def test_row_gather_load_no_sentinels(self) -> None:
        """Sentinel strings must never appear in the generated NKI source."""
        kernel = self._make_kernel()
        weight = torch.zeros(32, 128, device=DEVICE)
        idx = torch.zeros(8, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (idx, weight))
        _assert_no_sentinels(self, code)

    def test_row_gather_load_uses_ap(self) -> None:
        """Row-gather DMA uses .ap(vector_offset=...) in the generated code."""
        kernel = self._make_kernel()
        weight = torch.zeros(32, 128, device=DEVICE)
        idx = torch.zeros(8, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (idx, weight))
        self.assertIn(".ap(", code)
        self.assertIn("vector_offset", code)
        self.assertIn("indirect_dim=0", code)

    def test_row_gather_load_codegen_snapshot(self) -> None:
        """Full codegen snapshot — must be identical before and after refactor."""
        kernel = self._make_kernel()
        weight = torch.zeros(32, 128, device=DEVICE)
        idx = torch.zeros(8, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (idx, weight))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)



# ---------------------------------------------------------------------------
# Path 2: __AP_ROW_GATHER__ — row gather load via tile.index + starts
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestRowGatherWithOffset(TestCase):
    """
    Covers __AP_ROW_GATHER__ created by _nki_row_index_gather in the add.Tensor path
    (tensor + scalar SBUF broadcast), consumed in _build_hbm_src.

    Pattern: x[tile.index + starts, :] where starts is a scalar SBUF tile.
    This is the dominant pattern in jagged kernels.

    Config note: hl.tile([B, L, D], block_size=[1, None, None]) produces 2 tunable
    block sizes (for L and D — B is fixed to 1 and not tunable).
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128, 128]),
        )
        def jagged_copy(
            x: torch.Tensor,
            offsets: torch.Tensor,
        ) -> torch.Tensor:
            B = offsets.size(0) - 1
            L, D = x.shape
            out = torch.empty_like(x)
            for tile_b, tile_q, tile_d in hl.tile(
                [B, L, D], block_size=[1, None, None]
            ):
                starts = offsets[tile_b.begin]
                out[tile_q.index + starts, tile_d] = x[tile_q.index + starts, tile_d]
            return out

        return jagged_copy

    def test_row_gather_offset_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        L, D = 64, 128
        x = torch.zeros(L, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, offsets))
        _assert_no_sentinels(self, code)

    def test_row_gather_offset_uses_ap(self) -> None:
        kernel = self._make_kernel()
        L, D = 64, 128
        x = torch.zeros(L, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, offsets))
        self.assertIn(".ap(", code)
        self.assertIn("vector_offset", code)

    def test_row_gather_offset_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        L, D = 64, 128
        x = torch.zeros(L, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, offsets))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 3: __AP_ROW_GATHER__ — row gather store
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestRowGatherStore(TestCase):
    """
    Covers __AP_ROW_GATHER__ on the store side (out[tile.index + starts, :] = value).
    The store codegen at ~line 6034 detects __AP_ROW_GATHER__ in slice_str and emits
    an .ap() dst expression; this becomes an isinstance(p, IndirectAP) check.

    Config note: block_size=[1, None, None] gives 2 tunable dims (L and D).
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128, 64]),
        )
        def jagged_store(
            src: torch.Tensor,
            out: torch.Tensor,
            offsets: torch.Tensor,
        ) -> torch.Tensor:
            B = offsets.size(0) - 1
            L, D = src.shape
            for tile_b, tile_q, tile_d in hl.tile(
                [B, L, D], block_size=[1, None, None]
            ):
                starts = offsets[tile_b.begin]
                val = src[tile_q.index + starts, tile_d]
                out[tile_q.index + starts, tile_d] = val
            return out

        return jagged_store

    def test_row_gather_store_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        L, D = 64, 64
        src = torch.zeros(L, D, device=DEVICE)
        out = torch.zeros(L, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (src, out, offsets))
        _assert_no_sentinels(self, code)

    def test_row_gather_store_uses_ap(self) -> None:
        kernel = self._make_kernel()
        L, D = 64, 64
        src = torch.zeros(L, D, device=DEVICE)
        out = torch.zeros(L, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (src, out, offsets))
        self.assertIn(".ap(", code)
        self.assertIn("vector_offset", code)

    def test_row_gather_store_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        L, D = 64, 64
        src = torch.zeros(L, D, device=DEVICE)
        out = torch.zeros(L, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (src, out, offsets))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 4: __AP_ROW_GATHER__ — 3D early-exit with flat_idx (load)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestRowGather3DLoad(TestCase):
    """
    Covers the 3D early-exit path in load codegen (~line 1674) where sub0 is a
    gather vector and sub1 is a scalar head index. _combine_leading_dims Case 1
    produces __AP_ROW_GATHER__{flat_var}__ from the combined flat index.

    Pattern: q[tile.index + starts, i_h, :] on [L, H, D].

    Config note: block_size=[1, 1, None] gives 1 tunable dim (for the sequence
    tile — B and H are fixed scalars).
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128]),
        )
        def jagged_3d_load(
            x: torch.Tensor,
            offsets: torch.Tensor,
        ) -> torch.Tensor:
            num_batches = offsets.size(0) - 1
            L = x.size(0)
            nheads = x.size(1)
            out = torch.empty_like(x)
            for tile_b, tile_h, tile_q in hl.tile(
                [num_batches, nheads, L], block_size=[1, 1, None]
            ):
                starts = offsets[tile_b.begin]
                i_h = tile_h.id
                out[tile_q.index + starts, i_h, :] = x[tile_q.index + starts, i_h, :]
            return out

        return jagged_3d_load

    def test_3d_load_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        L, H, D = 100, 2, 16
        x = torch.zeros(L, H, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, offsets))
        _assert_no_sentinels(self, code)

    def test_3d_load_uses_ap_with_flat_idx(self) -> None:
        """_combine_leading_dims emits _3d_flat_idx with .ap()."""
        kernel = self._make_kernel()
        L, H, D = 100, 2, 16
        x = torch.zeros(L, H, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, offsets))
        self.assertIn(".ap(", code)
        self.assertIn("_3d_flat_idx", code)

    def test_3d_load_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        L, H, D = 100, 2, 16
        x = torch.zeros(L, H, D, device=DEVICE)
        offsets = torch.zeros(5, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, offsets))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 5: __AP_ROW_GATHER__ — strided gather via _combine_leading_dims
#         (3D tensor, tile followed by scalar dim)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestStridedGather(TestCase):
    """
    Covers the strided-gather path in _combine_leading_dims (~line 3844).
    w[tile_s, i_h, :] on [S, H, D] — rows are non-contiguous (stride=H).
    Emits __AP_ROW_GATHER__{_vec_ld}__ which becomes IndirectAP after refactor.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[1, 32]),
        )
        def strided_gather(w: torch.Tensor, nheads: int) -> torch.Tensor:
            S, H, D = w.shape
            H = hl.specialize(nheads)
            out = torch.empty(H, S, D, device=w.device, dtype=w.dtype)
            for tile_h, tile_s in hl.tile([H, S]):
                i_h = tile_h.id
                out[i_h, tile_s, :] = w[tile_s, i_h, :]
            return out

        return strided_gather

    def test_strided_gather_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        w = torch.zeros(32, 4, 64, device=DEVICE)
        code = _nki_code(kernel, (w, 4))
        _assert_no_sentinels(self, code)

    def test_strided_gather_uses_ap(self) -> None:
        kernel = self._make_kernel()
        w = torch.zeros(32, 4, 64, device=DEVICE)
        code = _nki_code(kernel, (w, 4))
        self.assertIn(".ap(", code)
        self.assertIn("_sg_scaled", code)
        self.assertIn("vector_offset", code)

    def test_strided_gather_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        w = torch.zeros(32, 4, 64, device=DEVICE)
        code = _nki_code(kernel, (w, 4))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)



# ---------------------------------------------------------------------------
# Path 6: __DYN_AP__ — dynamic loop on partition dim
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestDynamicAPPartition(TestCase):
    """
    Covers __DYN_AP__ emitted in _classify_load_dim (~line 1883) when the subscript
    is inside a nl.dynamic_range loop on the partition dimension.

    Pattern: x[tile_m, tile_n] where tile_m iterates over a tensor-bounded range.
    _build_hbm_src detects DynamicAP (was __DYN_AP__) and emits .ap(scalar_offset=).
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_partition_copy(
            x: torch.Tensor, bound: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros_like(x)
            bs = hl.register_block_size(x.size(0))
            for tile_n in hl.tile(x.size(1)):
                for tile_m in hl.tile(bound[0], block_size=bs):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1
            return out

        return dyn_partition_copy

    def test_dyn_ap_partition_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(16, 4, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        _assert_no_sentinels(self, code)

    def test_dyn_ap_partition_uses_ap(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(16, 4, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        self.assertIn(".ap(", code)
        self.assertIn("nl.dynamic_range(", code)

    def test_dyn_ap_partition_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(16, 4, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)



# ---------------------------------------------------------------------------
# Path 7: __DYN_AP__ — dynamic loop on free dim
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestDynamicAPFreeDim(TestCase):
    """
    Covers __DYN_AP__ on the free (column) dimension.

    Pattern: x[tile_m, tile_n] where tile_n iterates over a tensor-bounded range
    on the inner (free) dimension.  The partition is static; only the column dim
    is dynamic.  _build_hbm_src handles dyn_dim_idx == 1.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_free_copy(
            x: torch.Tensor, bound: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros_like(x)
            bs = hl.register_block_size(x.size(1))
            for tile_m in hl.tile(x.size(0)):
                for tile_n in hl.tile(bound[0], block_size=bs):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1
            return out

        return dyn_free_copy

    def test_dyn_ap_free_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 16, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        _assert_no_sentinels(self, code)

    def test_dyn_ap_free_uses_ap(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 16, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        self.assertIn(".ap(", code)
        self.assertIn("nl.dynamic_range(", code)

    def test_dyn_ap_free_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 16, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)



# ---------------------------------------------------------------------------
# Path 8: __DYN_AP__ — dynamic reduction loop (sum over dynamic range)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestDynamicAPReduction(TestCase):
    """
    Covers __DYN_AP__ in a dynamic reduction loop — the load happens inside
    hl.tile(bound[0]) on the inner dimension and the result is accumulated.
    This exercises the _build_hbm_src path that builds the full AP pattern
    and verifies the store DMA (to the accumulator) is also correct.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_reduction(
            x: torch.Tensor, bound: torch.Tensor
        ) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(x.size(1))
            for tile_m in hl.tile(x.size(0)):
                acc = hl.zeros([tile_m, bs])
                for tile_n in hl.tile(bound[0], block_size=bs):
                    acc = acc + x[tile_m, tile_n]
                out[tile_m] = acc.sum(-1)
            return out

        return dyn_reduction

    def test_dyn_reduction_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 16, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        _assert_no_sentinels(self, code)

    def test_dyn_reduction_uses_ap_and_dynamic_range(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 16, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        self.assertIn(".ap(", code)
        self.assertIn("nl.dynamic_range(", code)
        self.assertIn("nisa.dma_copy", code)

    def test_dyn_reduction_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 16, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, bound))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)



# ---------------------------------------------------------------------------
# Path 9: __DYN_AP__ — store side (dynamic-partition store)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestDynamicAPStore(TestCase):
    """
    Covers __DYN_AP__ on the store path (~line 5004).
    The store codegen at ~line 6092 handles __DYN_AP__ in slice_str;
    after refactor it checks isinstance(p, DynamicAP) directly.

    Pattern: out[tile_m, tile_n] = val  where tile_m is dynamic.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_store(
            src: torch.Tensor, out: torch.Tensor, bound: torch.Tensor
        ) -> torch.Tensor:
            bs = hl.register_block_size(src.size(0))
            for tile_n in hl.tile(src.size(1)):
                for tile_m in hl.tile(bound[0], block_size=bs):
                    out[tile_m, tile_n] = src[tile_m, tile_n]
            return out

        return dyn_store

    def test_dyn_store_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        src = torch.zeros(16, 4, device=DEVICE)
        out = torch.zeros(16, 4, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (src, out, bound))
        _assert_no_sentinels(self, code)

    def test_dyn_store_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        src = torch.zeros(16, 4, device=DEVICE)
        out = torch.zeros(16, 4, device=DEVICE)
        bound = torch.tensor([8], dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (src, out, bound))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 10: __AP_ROW_GATHER__ — atomic_ops.py row-scatter RMW path
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestAtomicRowGather(TestCase):
    """
    Covers atomic_ops.py:585 which inspects the result of _nki_row_index_gather.
    The existing code does prefix+strip string operations on the sentinel;
    after refactor it uses isinstance(row_part, IndirectAP) + row_part.vec_var.

    Pattern: hl.atomic_add(out, [rows, :], value) where rows is a gather index.
    This exercises the row-scatter RMW path in atomic_ops.py.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 64]),
        )
        def atomic_scatter(
            out: torch.Tensor,
            src: torch.Tensor,
            idx: torch.Tensor,
        ) -> torch.Tensor:
            B = idx.size(0)
            D = src.size(1)
            for tile_b, tile_d in hl.tile([B, D]):
                rows = idx[tile_b]
                hl.atomic_add(out, [rows, tile_d], src[tile_b, tile_d])
            return out

        return atomic_scatter

    def test_atomic_row_gather_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        out = torch.zeros(32, 64, device=DEVICE)
        src = torch.zeros(8, 64, device=DEVICE)
        idx = torch.zeros(8, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (out, src, idx))
        _assert_no_sentinels(self, code)

    def test_atomic_row_gather_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        out = torch.zeros(32, 64, device=DEVICE)
        src = torch.zeros(8, 64, device=DEVICE)
        idx = torch.zeros(8, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (out, src, idx))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 11: __AP_VEC_OFFSET__ — _nki_indirect_gather (line 945)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestVecOffsetGather(TestCase):
    """
    Covers __AP_VEC_OFFSET__ created by _nki_indirect_gather (line 945) and
    consumed in _build_hbm_src (VEC_OFFSET path, pattern is pre-built string).

    This path fires when _nki_indirect_gather succeeds and returns the sentinel
    with a pre-built pattern.  Exercises hl.load with a complex index expression
    that forces _nki_indirect_gather rather than _nki_row_index_gather.

    Pattern: x[flat_indices] where flat_indices is a non-trivial mul+mod expression.

    Note: this test only verifies codegen (no runtime execution) because the
    mul+mod index pattern can produce partition counts that NRT rejects.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128, 64]),
        )
        def indirect_gather(
            x: torch.Tensor,
            idx: torch.Tensor,
        ) -> torch.Tensor:
            B = idx.size(0)
            D = x.size(1)
            out = torch.empty([B, D], dtype=x.dtype, device=x.device)
            for tile_b, tile_d in hl.tile([B, D]):
                flat_idx = idx[tile_b] % x.size(0)
                out[tile_b, tile_d] = x[flat_idx, tile_d]
            return out

        return indirect_gather

    def test_vec_offset_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(64, 64, device=DEVICE)
        idx = torch.zeros(128, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, idx))
        _assert_no_sentinels(self, code)

    def test_vec_offset_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(64, 64, device=DEVICE)
        idx = torch.zeros(128, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, idx))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 12: Contiguous load — regression guard (no AP, no sentinels)
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestContiguousRegressionGuard(TestCase):
    """
    Guard: a plain 2D contiguous load must not use .ap() or emit any sentinel.
    Any regression in the isinstance guard logic that makes contiguous paths
    accidentally take the indirect path would be caught here.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128, 128]),
        )
        def plain_copy(x: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_k in hl.tile([M, K]):
                out[tile_m, tile_k] = x[tile_m, tile_k]
            return out

        return plain_copy

    def test_contiguous_no_ap_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(256, 128, device=DEVICE)
        code = _nki_code(kernel, (x,))
        _assert_no_sentinels(self, code)
        self.assertNotIn(".ap(", code)
        self.assertIn("nisa.dma_copy", code)

    def test_contiguous_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(256, 128, device=DEVICE)
        code = _nki_code(kernel, (x,))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)



# ---------------------------------------------------------------------------
# Path 13: 3D contiguous flatten — regression guard for _combine_leading_dims
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestFlattenRegressionGuard(TestCase):
    """
    Guard: 3D contiguous (no strided, no indirect) must flatten to a plain DMA.
    _combine_leading_dims guard checks (~line 3859) must not accidentally trigger
    the AP path after the isinstance refactor.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[2, 64, 128]),
        )
        def flat_3d_copy(x: torch.Tensor) -> torch.Tensor:
            B, M, K = x.shape
            out = torch.empty_like(x)
            for tile_b, tile_m, tile_k in hl.tile([B, M, K]):
                out[tile_b, tile_m, tile_k] = x[tile_b, tile_m, tile_k]
            return out

        return flat_3d_copy

    def test_3d_flatten_no_ap_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 128, 128, device=DEVICE)
        code = _nki_code(kernel, (x,))
        _assert_no_sentinels(self, code)
        self.assertNotIn(".ap(", code)

    def test_3d_flatten_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        x = torch.zeros(4, 128, 128, device=DEVICE)
        code = _nki_code(kernel, (x,))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 14: Jagged dense add — exercises DYN_AP + ROW_GATHER in same kernel
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestJaggedDenseAdd(TestCase):
    """
    Integration test: the jagged_dense_add_2d kernel uses both DYN_AP (for the
    dynamic tile1 loop over max_nnz) and ROW_GATHER (for starts + tile.index).
    Both sentinels appear in the same kernel; after refactor both must be absent
    from the output while the generated code is unchanged.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128, 128, 8]),
        )
        def jagged_dense_add(
            x_data: torch.Tensor,
            x_offsets: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            num_rows = y.size(0)
            out = torch.zeros_like(y)
            for tile0 in hl.tile(num_rows):
                starts = x_offsets[tile0]
                ends = x_offsets[tile0.index + 1]
                nnz = ends - starts
                max_nnz = nnz.amax()
                for tile1 in hl.tile(0, max_nnz):
                    x_slice = hl.load(
                        x_data,
                        [starts[:, None] + tile1.index[None, :]],
                        extra_mask=tile1.index[None, :] < nnz[:, None],
                    )
                    out[tile0, tile1] = y[tile0, tile1] + x_slice
                for tile1 in hl.tile(max_nnz, out.size(1)):
                    out[tile0, tile1] = y[tile0, tile1]
            return out

        return jagged_dense_add

    def test_jagged_dense_add_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        num_rows, N = 4, 128
        x_offsets = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device=DEVICE)
        x_data = torch.zeros(128, device=DEVICE)
        y = torch.zeros(num_rows, N, device=DEVICE)
        code = _nki_code(kernel, (x_data, x_offsets, y))
        _assert_no_sentinels(self, code)

    def test_jagged_dense_add_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        num_rows, N = 4, 128
        x_offsets = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device=DEVICE)
        x_data = torch.zeros(128, device=DEVICE)
        y = torch.zeros(num_rows, N, device=DEVICE)
        code = _nki_code(kernel, (x_data, x_offsets, y))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)


# ---------------------------------------------------------------------------
# Path 15: Embedding kernel from examples/ — canonical ROW_GATHER reference
# ---------------------------------------------------------------------------


@onlyBackends(["nki"])
class TestEmbeddingKernel(TestCase):
    """
    Canonical embedding lookup from examples/embedding.py.
    This is the primary example used in the refactor guide's testing section.
    Exercises __AP_ROW_GATHER__ in the most straightforward form.
    """

    def _make_kernel(self) -> helion.Kernel:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[128, 128]),
        )
        def embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_flat = x.reshape(-1)
            _, embedding_dim = weight.size()
            out = torch.empty(
                [x_flat.size(0), embedding_dim],
                dtype=weight.dtype,
                device=weight.device,
            )
            for tile_b, tile_e in hl.tile([x_flat.size(0), embedding_dim]):
                out[tile_b, tile_e] = weight[x_flat[tile_b], tile_e]
            return out.view(*x.size(), embedding_dim)

        return embedding

    def test_embedding_no_sentinels(self) -> None:
        kernel = self._make_kernel()
        weight = torch.zeros(16, 128, device=DEVICE)
        x = torch.zeros(256, 32, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, weight))
        _assert_no_sentinels(self, code)

    def test_embedding_uses_ap(self) -> None:
        kernel = self._make_kernel()
        weight = torch.zeros(16, 128, device=DEVICE)
        x = torch.zeros(256, 32, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, weight))
        self.assertIn(".ap(", code)
        self.assertIn("vector_offset", code)

    def test_embedding_codegen_snapshot(self) -> None:
        kernel = self._make_kernel()
        weight = torch.zeros(16, 128, device=DEVICE)
        x = torch.zeros(256, 32, dtype=torch.int32, device=DEVICE)
        code = _nki_code(kernel, (x, weight))
        _assert_no_sentinels(self, code)
        self.assertExpectedJournal(code)
