"""EXPERIMENTAL v2 body-codegen: emit nkilib TileStream/dma/blas calls.

When ``HELION_NKI_TILESTREAM=1`` the NKI backend routes structured ops (load,
store, matmul, alloc) through these emitters instead of hand-rolling the
``nisa.*`` instruction sequences. The Python ``for offset_N in affine_range(...)``
loop scaffold and the ``offset_N``/``indices_N`` contract are KEPT (see
NKI_TILESTREAM_FULL_REFACTOR_PLAN.md Findings A/B/C); only the body mechanics
become nkilib-native.

Bucket policy (plan §1b):
  1 CONVERT      — load/store/matmul/transpose/elementwise/broadcast (here).
  2 HARD-CASE    — NEG_START -> dma.Load(oob_mode), partition-bcast -> blas.broadcast.
  3 STAYS nisa   — reduce/scan/cumulative/predicated/iota/memset (no nkilib primitive;
                   nkilib's own kernels emit raw nisa for these — NOT a failure).
  4 LOUD ERROR   — genuinely unclassified op -> BackendUnsupported.

The emitted code assumes the kernel header imported nkilib as ``_nkitile`` and
``_nkitv`` (see NKIBackend.library_imports).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tile_strategy import CodegenState


class V2Unsupported(Exception):
    """Raised by a v2 emitter when a shape/case is not yet covered (bucket 4).

    Callers translate this into exc.BackendUnsupported so the failure is LOUD
    (never a silent fall-through to legacy body-codegen). Bucket-3 ops do NOT
    raise this — they legitimately stay nisa.
    """


def _shape_tuple(dims: list[int] | tuple[int, ...]) -> str:
    parts = [str(int(d)) for d in dims]
    return f"({parts[0]},)" if len(parts) == 1 else "(" + ", ".join(parts) + ")"


def grid_coord(offset_var: str, block_size: int) -> str:
    """Map a loop offset variable to its tile-grid coordinate for ``tile_at``.

    Helion's loop steps by ``block_size`` (``for offset_N in affine_range(0, dim,
    block_size)``), so the grid index of the current tile is ``offset_N // bs``.
    block_size==1 (grid-only / scalar dims) -> coordinate is the offset itself.
    """
    if block_size in (1, "1"):
        return f"({offset_var})"
    return f"(({offset_var}) // {int(block_size)})"


def get_or_make_hbm_stream(
    state: "CodegenState",
    hbm_name: str,
    tile_shape: tuple[int, ...],
    *,
    tile_dims: tuple[int, ...] | None = None,
) -> str:
    """Emit (once per (hbm_name, tile_shape)) an nkilib HBMStream over the whole
    tensor and return its var name. Subsequent identical requests reuse it.

    NOTE (v2 first-cut): the stream is emitted into the CURRENT statement scope,
    not hoisted before the loop. It is loop-invariant pure metadata, so this is
    correct; hoisting to DeviceLoopState.outer_prefix is a G-phase optimization.
    The dedup registry avoids rebuilding it when the same tensor is loaded twice
    in the same scope.
    """
    from ..ast_extension import statement_from_string
    from .tilestream_emit import emit_tile_hbm

    dev = state.device_function
    key = (hbm_name, tuple(int(d) for d in tile_shape))
    cached = dev._nki_hbm_streams.get(key)
    if cached is not None:
        return cached
    stream_var = dev.new_var("_ts_hbm", dce=True)
    state.codegen.add_statement(
        statement_from_string(
            emit_tile_hbm(stream_var, hbm_name, tile_shape, tile_dims=tile_dims)
        )
    )
    dev._nki_hbm_streams[key] = stream_var
    return stream_var
