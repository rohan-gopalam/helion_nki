"""EXPERIMENTAL: string builders for emitting nkilib TileStream/TensorView calls.

These produce the *text* that the NKI backend emits into a generated ``@nki.jit``
kernel when ``HELION_NKI_TILESTREAM=1``. The emitted code calls ``nkilib`` at
neuronx-cc *trace* time (TileStream/TensorView are traced runtime objects), so
this module never imports nkilib itself — it only builds source strings.

The emitted code assumes the kernel header imported nkilib as ``_nkitile`` (see
``NKIBackend.library_imports``). Convention mirrors how nkilib's own primitives
are called: ``_nkitile.tile_stream`` / ``_nkitile.tile_hbm`` / ``_nkitile.RowMajor``
/ ``_nkitile.dma.Load`` / ``_nkitile.blas.Matmul``.

Pure string builders: testable without a kernel. See NKI_TILESTREAM_REFACTOR_PLAN.md.
"""

from __future__ import annotations


def _shape_tuple(shape: list[int] | tuple[int, ...]) -> str:
    """Render a shape as a Python tuple literal, e.g. (128, 64) or (128,)."""
    parts = [str(int(d)) for d in shape]
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ", ".join(parts) + ")"


def emit_alloc_logical(
    var: str,
    logical_shape: list[int] | tuple[int, ...],
    pdim_size: int,
    dtype: str,
    buffer: str = "nl.sbuf",
) -> str:
    """``var = _nkitile.tile_stream.alloc_logical((P,*F), pdim_size, dtype, buffer=...)``.

    Produces the (pdim_size, n_p_tiles, *F) container TensorView. ``dtype`` and
    ``buffer`` are emitted verbatim (e.g. ``nl.float32``, ``nl.sbuf``/``nl.psum``).
    """
    return (
        f"{var} = _nkitile.tile_stream.alloc_logical("
        f"{_shape_tuple(logical_shape)}, pdim_size={int(pdim_size)}, "
        f"dtype={dtype}, buffer={buffer})"
    )


def emit_tile(
    var: str,
    src_var: str,
    tile_shape: list[int] | tuple[int, ...],
    *,
    tile_dims: tuple[int, ...] | None = None,
    iter_order: str = "_nkitile.RowMajor()",
    logical_p: int | None = None,
) -> str:
    """``var = _nkitile.tile_stream.tile(src, tile_shape=..., iter_order=..., [logical_p=...])``.

    ``src_var`` is an SBUF container (from alloc_logical). ``logical_p`` carries the
    true partition extent for partial last-tile handling.
    """
    args = [src_var, f"tile_shape={_shape_tuple(tile_shape)}"]
    if tile_dims is not None:
        args.append(f"tile_dims={_shape_tuple(tile_dims)}")
    args.append(f"iter_order={iter_order}")
    if logical_p is not None:
        args.append(f"logical_p={int(logical_p)}")
    return f"{var} = _nkitile.tile_stream.tile({', '.join(args)})"


def emit_tile_hbm(
    var: str,
    src_var: str,
    tile_shape: list[int] | tuple[int, ...],
    *,
    tile_dims: tuple[int, ...] | None = None,
    iter_order: str = "_nkitile.RowMajor()",
) -> str:
    """``var = _nkitile.tile_stream.tile_hbm(src, tile_shape=..., [tile_dims=...], iter_order=...)``."""
    args = [src_var, f"tile_shape={_shape_tuple(tile_shape)}"]
    if tile_dims is not None:
        args.append(f"tile_dims={_shape_tuple(tile_dims)}")
    args.append(f"iter_order={iter_order}")
    return f"{var} = _nkitile.tile_stream.tile_hbm({', '.join(args)})"


def emit_load(
    dst_ts: str,
    src_ts: str,
    *,
    vector_index: str | None = None,
    index_dim: int | None = None,
    transpose: bool = False,
) -> str:
    """``_nkitile.dma.Load(dst=dst_ts, src=src_ts[, vector_index=..., index_dim=..., transpose=...]).execute()``."""
    args = [f"dst={dst_ts}", f"src={src_ts}"]
    if vector_index is not None:
        args.append(f"vector_index={vector_index}")
    if index_dim is not None:
        args.append(f"index_dim={int(index_dim)}")
    if transpose:
        args.append("transpose=True")
    return f"_nkitile.dma.Load({', '.join(args)}).execute()"


def emit_store(dst_hbm_ts: str, src_ts: str) -> str:
    """``_nkitile.dma.Store(dst=dst_hbm_ts, src=src_ts).execute()`` (SBUF TileStream -> HBM)."""
    return f"_nkitile.dma.Store(dst={dst_hbm_ts}, src={src_ts}).execute()"


def emit_matmul(dst_ts: str, moving_ts: str, stationary_ts: str) -> str:
    """``_nkitile.blas.Matmul(dst=..., moving=..., stationary=...).execute()``.

    Semantics: ``dst = stationary^T @ moving``; K (contraction) is the partition
    dim of both ``moving`` and ``stationary``. For ``C[M,N]=A[M,K]@B[K,N]``:
    stationary is A^T as [K, M], moving is B as [K, N], dst is [M, N].
    """
    return (
        f"_nkitile.blas.Matmul(dst={dst_ts}, moving={moving_ts}, "
        f"stationary={stationary_ts}).execute()"
    )


# --- Compact blas primitives: take raw nl.ndarray dst/src and emit the nisa op
# internally, so they are statement-level drop-ins that DON'T change the type of
# any variable (unlike tile()/alloc_logical which return stream objects). These
# are the safe, high-readability swaps for the manual nisa.* emissions. ---

def emit_blas_transpose(dst: str, src: str) -> str:
    """``_nkitile.blas.transpose(dst=, src=)`` — replaces the manual
    nc_transpose -> PSUM -> tensor_copy 3-statement dance with one call."""
    return f"_nkitile.blas.transpose(dst={dst}, src={src})"


def emit_blas_tensor_scalar(
    dst: str,
    src: str,
    op0: str,
    operand0: str,
    op1: str | None = None,
    operand1: str | None = None,
) -> str:
    """``_nkitile.blas.tensor_scalar(...)``: dst = op1(op0(src, operand0), operand1)."""
    args = [f"dst={dst}", f"src={src}", f"op0={op0}", f"operand0={operand0}"]
    if op1 is not None:
        args.append(f"op1={op1}")
    if operand1 is not None:
        args.append(f"operand1={operand1}")
    return f"_nkitile.blas.tensor_scalar({', '.join(args)})"


def emit_blas_activation(
    dst: str,
    src: str,
    op: str = "nl.copy",
    scale: str | None = None,
    bias: str | None = None,
) -> str:
    """``_nkitile.blas.activation(...)``: dst = op(src * scale + bias)."""
    args = [f"dst={dst}", f"src={src}", f"op={op}"]
    if scale is not None:
        args.append(f"scale={scale}")
    if bias is not None:
        args.append(f"bias={bias}")
    return f"_nkitile.blas.activation({', '.join(args)})"


def emit_tv_broadcast(src: str, dim: int, size: int | str, *, base: str | None = None) -> str:
    """``_nkitv(src).broadcast(dim, size).get_view()`` — stride-0 broadcast view,
    drop-in for ``nl.broadcast_to(src, shape=...)`` when the broadcast dim is size-1."""
    inner = base if base is not None else src
    return f"_nkitv({inner}).broadcast({int(dim)}, {size}).get_view()"
