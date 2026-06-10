"""NKI backend load/store codegen.

Ported verbatim from ``helion/language/memory_ops.py`` as part of the NKI
subpackage refactor (mirroring ``helion/_compiler/pallas/``). The two
``@_decorators.codegen(load/store, "nki")`` registrations in ``memory_ops.py``
are thin shims that delegate to :func:`load_expr` and :func:`store_stmt` here.

The bodies are the most complex piece of the backend; do not simplify. Most
heavy imports are lazy (inside the function bodies); ``IndirectAP`` /
``DynamicAP`` are imported at module top.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language._nki_dim_access import DynamicAP
from ...language._nki_dim_access import IndirectAP
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from .gather import _nki_as_uint32_p1_vector
from .gather import _nki_indirect_gather
from .gather import _nki_row_index_gather
from .gather import _nki_store_3d_row_scatter
from .indexing import _nki_shifted_tile_subscript
from .indexing import _nki_subscript_block_id
from .sbuf import _nki_lookup_sbuf_shape_dtype

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def load_expr(
    state: CodegenState,
    subscript: list[object],
    tensor: torch.Tensor,
) -> ast.AST:
    from ..ast_extension import create
    from ..ast_extension import expr_from_string
    from ..ast_extension import statement_from_string

    NKI_PARTITION_MAX = 128

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    tensor_node = (
        state.fx_node.args[0]
        if state.fx_node is not None and len(state.fx_node.args) >= 1
        else None
    )
    from ...language.tile_ops import tile_index

    if (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        env = CompileEnvironment.current()
        block_id = env.get_block_id(tensor.size(0))
        assert block_id is not None
        base_var = state.codegen.index_var(block_id)
        parts = []
        for idx in subscript:
            if idx is None:
                parts.append("None")
            elif idx == slice(None):
                parts.append(":")
            else:
                raise AssertionError(
                    f"Unexpected index type in tile_index load: {idx}"
                )
        return expr_from_string(f"{base_var}[{', '.join(parts)}]")
    name = state.device_function.tensor_arg(tensor).name
    device_fn = state.device_function
    # If this tensor was previously written via nki_return_buf (it's both an
    # input/output to the kernel), redirect reads to the return buffer so we
    # see the latest writes. Without this, reads go to the uninitialized
    # input parameter and produce NaN/garbage (e.g. SE net's c @ b chain).
    _ret_bufs = device_fn._nki_return_buffers
    _ret_info = _ret_bufs.get(id(tensor))
    if _ret_info is not None and "buf_name" in _ret_info:
        name = _ret_info["buf_name"]
    device_fn.device_load_index += 1
    device_fn.device_memory_op_index += 1
    env = CompileEnvironment.current()
    backend = env.backend
    sbuf_name = f"_nki_sbuf_{device_fn.device_load_index}"
    output_shape = SubscriptIndexing.compute_shape(tensor, [*subscript], state)

    import sympy as _sympy

    _bs_subs: dict[_sympy.Symbol, int] = {}
    for _bid in range(len(env.block_sizes)):
        _bs = env.block_sizes[_bid]
        _bs_subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))

    def _resolve_dim(s: int | torch.SymInt) -> int:
        if isinstance(s, int):
            return s
        try:
            return int(s._sympy_().subs(_bs_subs))
        except (TypeError, ValueError):
            # Fallback: use size_hint when symbol substitution fails
            # (e.g. for computed expressions like 2*block_size_k_packed)
            return int(env.size_hint(s))

    # First pass: resolve block_id per (tensor_dim_idx, subscript_position).
    # We use the FX subscript nodes (which carry live SymInts even when the
    # materialized subscript value has been concretized to a hint integer)
    # and the centralized ``_nki_subscript_block_id`` helper.
    fx_subscript = (
        state.fx_node.args[1]
        if state.fx_node is not None and len(state.fx_node.args) >= 2
        else None
    )
    # Map: tensor_dim_idx -> block_id (int) or None. Length equals tensor.dim().
    subscript_block_ids: list[int | None] = []
    # Map: tensor_dim_idx -> subscript position i that produced it.
    subscript_positions: list[int] = []
    _tdi = 0
    for i, sub_val in enumerate(subscript):
        if sub_val is None:  # newaxis — doesn't consume a tensor dim
            continue
        if _tdi >= tensor.dim():
            break
        fx_node_i = fx_subscript[i] if fx_subscript is not None and i < len(fx_subscript) else None
        bid = _nki_subscript_block_id(sub_val, fx_node_i, env)
        # slice(None) subscripts over reduction blocks: look up by size-match.
        # Only match full-dimension slices (slice(None) or equivalent), not
        # computed partial slices like a_tile_begin:a_tile_begin+a_tile_len.
        if bid is None and isinstance(sub_val, slice) and sub_val == slice(None):
            dim_size = tensor.size(_tdi)
            for _bid in range(len(env.block_sizes)):
                bs_info = env.block_sizes[_bid]
                if not bs_info.reduction:
                    continue
                block_size = bs_info.from_config_assert(state.config)
                if block_size <= 1:
                    continue
                if isinstance(dim_size, int) and block_size > 0 and dim_size % block_size == 0:
                    # Only match reduction blocks that are actually live.
                    if _bid in state.codegen.active_device_loops:
                        bid = _bid
                        break
        subscript_block_ids.append(bid)
        subscript_positions.append(i)
        _tdi += 1

    # Re-infer any dims that compute_shape dropped. A dim gets dropped when its
    # subscript SymInt was concretized to an integer (loses BlockSizeOrigin).
    # Re-insert the block_size var in the right position so the downstream
    # shape math (partition_dim, free_dims) stays correct.
    if len(output_shape) < tensor.dim():
        # Walk tensor_dim_idx left-to-right, matching output_shape consumption.
        # A dim was dropped if its resolved block_id is known AND the next
        # unconsumed output_shape entry doesn't match its block var.
        new_output_shape: list[int | torch.SymInt] = []
        os_idx = 0
        for tdi in range(tensor.dim()):
            bid = subscript_block_ids[tdi] if tdi < len(subscript_block_ids) else None
            sub_pos = subscript_positions[tdi] if tdi < len(subscript_positions) else None
            sub_val = subscript[sub_pos] if sub_pos is not None and sub_pos < len(subscript) else None
            expected_var = env.block_sizes[bid].var if bid is not None else None

            # For slice(None) subscripts, compute_shape always emits a dim —
            # consume the next output_shape entry.
            if isinstance(sub_val, slice):
                if os_idx < len(output_shape):
                    new_output_shape.append(output_shape[os_idx])
                    os_idx += 1
                else:
                    # Shouldn't happen; fall back to tensor size.
                    new_output_shape.append(tensor.size(tdi))
                continue

            # For SymInt-ish subscripts: if still symbolic, compute_shape
            # kept the dim; if concretized, it was dropped.
            if isinstance(sub_val, torch.SymInt):
                if os_idx < len(output_shape) and (
                    expected_var is None
                    or output_shape[os_idx] is expected_var
                    or (
                        isinstance(output_shape[os_idx], torch.SymInt)
                        and output_shape[os_idx]._sympy_() == expected_var._sympy_()
                    )
                ):
                    new_output_shape.append(output_shape[os_idx])
                    os_idx += 1
                elif expected_var is not None:
                    new_output_shape.append(expected_var)
                elif os_idx < len(output_shape):
                    new_output_shape.append(output_shape[os_idx])
                    os_idx += 1
                continue

            # Integer subscript — dim is fully eliminated; skip.
            if isinstance(sub_val, int):
                continue

            # Tensor indexer or other: keep what compute_shape said.
            if os_idx < len(output_shape):
                new_output_shape.append(output_shape[os_idx])
                os_idx += 1

        # Only apply re-inference if it produced at least as many dims as
        # the original (belt-and-suspenders: we never want to drop MORE).
        if len(new_output_shape) >= len(output_shape):
            output_shape = new_output_shape

    partition_dim = _resolve_dim(output_shape[0]) if output_shape else 1
    free_dims = [_resolve_dim(s) for s in output_shape[1:]]
    dtype_str = backend.dtype_str(tensor.dtype)

    def _tensor_dim_size_str(dim_idx: int) -> str:
        dim_size = tensor.size(dim_idx)
        return (
            state.sympy_expr(dim_size._sympy_())
            if isinstance(dim_size, torch.SymInt)
            else str(dim_size)
        )

    hbm_dim_size_strs = [
        _tensor_dim_size_str(dim_idx) for dim_idx in range(tensor.dim())
    ]

    # Initialize slice_parts and related vars here; the 3D early-exit below may
    # populate them and skip the normal per-dim loop.
    slice_parts: list[str] = []
    is_scalar_dim: list[bool] = []
    partition_offset_var: str | None = None

    # Early-exit: 3D tensor with pattern [vec_1d + starts, scalar_head, :].
    # This occurs for q[tile.index + starts, tile_h.begin, :] where q is [L, H, D].
    # The kernel reshapes q to [L*H, D], so the correct flat row index is:
    #   flat_row = (tile.index + starts) * H + head
    # Detect this and emit a 2D row-gather directly, bypassing the per-dim loop.
    import os as _os
    if _os.environ.get("HELION_DEBUG_3D_LOAD"):
        print(f"DEBUG 3D load: tensor.dim={tensor.dim()}, subscript_block_ids={subscript_block_ids}, subscript_positions={subscript_positions}, fx_subscript_len={len(fx_subscript) if fx_subscript else None}")
        if tensor.dim() == 3 and fx_subscript:
            for i, s in enumerate(subscript):
                print(f"  subscript[{i}] = {type(s).__name__} {getattr(s, 'shape', '')}")
    if (
        tensor.dim() == 3
        and len(subscript_block_ids) == 3
        and len(subscript_positions) == 3
        and fx_subscript is not None
        and len(fx_subscript) >= 3
    ):
        sub0 = subscript[subscript_positions[0]]
        sub1 = subscript[subscript_positions[1]]
        sub2 = subscript[subscript_positions[2]]
        fx0 = fx_subscript[subscript_positions[0]] if subscript_positions[0] < len(fx_subscript) else None
        fx1 = fx_subscript[subscript_positions[1]] if subscript_positions[1] < len(fx_subscript) else None
        # sub0 must be a tensor (1D vector of row indices)
        # sub1 must be a scalar (tile_begin or const)
        # sub2 must be slice(None)
        sub1_is_scalar = (
            not isinstance(sub1, (slice, torch.Tensor))
        )
        if (
            isinstance(sub0, torch.Tensor)
            and sub1_is_scalar
            and isinstance(sub2, slice)
            and isinstance(fx0, torch.fx.Node)
        ):
            # Compute H and D
            h_size = tensor.size(1)
            d_size = tensor.size(2)
            h_int = int(h_size._sympy_().subs(_bs_subs)) if isinstance(h_size, torch.SymInt) else int(h_size)
            d_int = int(d_size._sympy_().subs(_bs_subs)) if isinstance(d_size, torch.SymInt) else int(d_size)
            # Get scalar head expression
            head_expr: str | None = None
            if isinstance(fx1, torch.fx.Node):
                fx1_ast = state.codegen.ast_for_fx_node(fx1)
                if isinstance(fx1_ast, ast.AST):
                    head_expr = ast.unparse(fx1_ast)
            if head_expr is None:
                bid1 = subscript_block_ids[1]
                if bid1 is not None:
                    head_expr = state.codegen.offset_var(bid1)
                elif isinstance(sub1, (int, bool)):
                    head_expr = str(int(sub1))
                elif isinstance(sub1, torch.SymInt):
                    head_expr = str(int(sub1._sympy_().subs(_bs_subs)))
            # Get the 2D row-gather sentinel for the vector index (tile.index + starts)
            _row_gather_3d = _nki_row_index_gather(fx0, state, partition_dim)
            if _row_gather_3d is not None and head_expr is not None:
                if isinstance(_row_gather_3d, IndirectAP):
                    vec_var = _row_gather_3d.vec_var
                    # flat_var = vec_var * H + head
                    flat_var = device_fn.new_var("_3d_flat_idx", dce=True)
                    device_fn._nki_sbuf_shapes[flat_var] = [partition_dim, 1]
                    device_fn._nki_sbuf_dtypes[flat_var] = "nl.uint32"
                    from ..ast_extension import statement_from_string as _sfs3d
                    state.codegen.add_statement(_sfs3d(
                        f"{flat_var} = nl.ndarray([{partition_dim}, 1], nl.uint32, buffer=nl.sbuf)"
                    ))
                    state.codegen.add_statement(_sfs3d(
                        f"nisa.tensor_copy(dst={flat_var}, src={vec_var})"
                    ))
                    if h_int != 1:
                        state.codegen.add_statement(_sfs3d(
                            f"nisa.tensor_scalar(dst={flat_var}, data={flat_var}, op0=nl.multiply, operand0={h_int}, op1=None)"
                        ))
                    state.codegen.add_statement(_sfs3d(
                        f"nisa.tensor_scalar(dst={flat_var}, data={flat_var}, op0=nl.add, operand0={head_expr}, op1=None)"
                    ))
                    # Now emit the 2D gather: [IndirectAP(flat_var), 0:D]
                    slice_parts = [IndirectAP(vec_var=flat_var, p_count=partition_dim, pattern=None), f"0:{d_int}"]
                    is_scalar_dim = [False, False]
                    # Override partition_dim, free_dims, and hbm_dim_size_strs to
                    # match the 2D reshaped tensor [L*H, D] — this controls the
                    # SBUF allocation and DMA shape.
                    partition_dim = partition_dim  # keep [tile_q_size] = 128
                    free_dims = [d_int]
                    total_rows = int(tensor.numel()) // d_int
                    hbm_dim_size_strs = [str(total_rows), str(d_int)]
                    partition_offset_var = None
                    # Skip the normal per-dim loop — go directly to DMA emit
                    # by setting subscript_block_ids to indicate no further processing
                    subscript_block_ids = []
                    subscript_positions = []

    # Second pass: emit slice_parts using the resolved block_ids (unless already
    # populated by the 3D early-exit above).
    used_shifted_subscript = False
    for tdi in range(len(subscript_block_ids)):
        block_id = subscript_block_ids[tdi]
        sub_pos_tdi_check = subscript_positions[tdi] if tdi < len(subscript_positions) else None
        fx_node_tdi_check = (
            fx_subscript[sub_pos_tdi_check]
            if fx_subscript is not None
            and sub_pos_tdi_check is not None
            and sub_pos_tdi_check < len(fx_subscript)
            else None
        )

        # Detect tile_id (or other scalar-index) subscripts: FX node target is
        # ``tile_id`` / ``tile_index`` / ``tile_begin`` / a _get_symnode whose
        # expression contains ``offset // block_size``. For these, emit a
        # scalar index (size-1 slice) rather than a range.
        _is_tile_id = False
        _is_tile_begin = False
        if isinstance(fx_node_tdi_check, torch.fx.Node):
            from ...language.tile_ops import tile_id as _tile_id_fn
            try:
                from ...language.tile_ops import tile_begin as _tile_begin_fn
            except ImportError:
                _tile_begin_fn = None
            if fx_node_tdi_check.target is _tile_id_fn:
                _is_tile_id = True
            elif _tile_begin_fn is not None and fx_node_tdi_check.target is _tile_begin_fn:
                _is_tile_begin = True
            elif fx_node_tdi_check.op == "call_function":
                _tname = str(getattr(fx_node_tdi_check.target, "__name__", fx_node_tdi_check.target))
                if "tile_id" in _tname:
                    _is_tile_id = True
                elif "tile_begin" in _tname:
                    _is_tile_begin = True

        # Also detect plain scalar subscripts (int / SymInt resolved to int /
        # arithmetic expression evaluating to a compile-time scalar). These
        # happen with things like ``x[:, chunk_size - 1]``. Only fires when
        # no block_id is resolved (otherwise it's a tile reference).
        _sub_val_load = subscript[subscript_positions[tdi]] if tdi < len(subscript_positions) else None
        _is_scalar_subscript = (
            not _is_tile_id
            and block_id is None
            and not isinstance(_sub_val_load, slice)
            and (
                isinstance(_sub_val_load, (int, bool))
                or (isinstance(_sub_val_load, torch.SymInt))
            )
        )
        if _is_tile_id and block_id is not None:
            # Scalar access: dst[.., offset//block_size, ...] maps to a single
            # element in that dim. Emit `tile_id_expr : tile_id_expr + 1` so the
            # slice has width 1.
            offset_var = state.codegen.offset_var(block_id)
            block_size = env.block_sizes[block_id].from_config_assert(state.config)
            # ``tile.id`` == offset // block_size
            if int(block_size) == 1:
                id_expr = offset_var
            else:
                id_expr = f"{offset_var} // {int(block_size)}"
            if tdi == 0:
                partition_offset_var = f"({id_expr})"
            slice_parts.append(f"({id_expr}):({id_expr})+1")
            is_scalar_dim.append(True)
        elif _is_tile_begin and block_id is not None:
            # tile.begin is a scalar == offset (no divide).
            offset_var = state.codegen.offset_var(block_id)
            if tdi == 0:
                partition_offset_var = f"({offset_var})"
            slice_parts.append(f"({offset_var}):({offset_var})+1")
            is_scalar_dim.append(True)
        elif _is_scalar_subscript:
            shifted = _nki_shifted_tile_subscript(fx_node_tdi_check, state, env)
            if shifted is not None:
                if tdi == 0:
                    partition_offset_var = shifted.split(":", 1)[0].strip()
                slice_parts.append(shifted)
                is_scalar_dim.append(True)
                continue
            scalar_ast = (
                state.codegen.ast_for_fx_node(fx_node_tdi_check)
                if isinstance(fx_node_tdi_check, torch.fx.Node)
                else None
            )
            if isinstance(scalar_ast, ast.AST):
                scalar_expr = ast.unparse(scalar_ast)
                if scalar_expr and not scalar_expr.isdigit():
                    if tdi == 0:
                        partition_offset_var = scalar_expr
                    slice_parts.append(f"{scalar_expr}:{scalar_expr}+1")
                    is_scalar_dim.append(True)
                    continue
            # Plain scalar subscript (literal int / concretized SymInt).
            if isinstance(_sub_val_load, torch.SymInt):
                try:
                    _scalar_val = int(_sub_val_load._sympy_().subs(_bs_subs))
                except (TypeError, ValueError):
                    _scalar_val = int(env.size_hint(_sub_val_load))
            else:
                _scalar_val = int(_sub_val_load)
            if tdi == 0:
                partition_offset_var = f"{_scalar_val}"
            slice_parts.append(f"{_scalar_val}:{_scalar_val}+1")
            is_scalar_dim.append(True)
        elif block_id is not None and block_id in state.codegen.active_device_loops:
            offset_var = state.codegen.offset_var(block_id)
            block_size = env.block_sizes[block_id].from_config_assert(state.config)
            if tdi == 0:
                partition_offset_var = offset_var
            # A contiguous shifted index like ``A[:, K_half + k_begin : ...]``
            # lowers to an iota whose affine start is ``K_half + offset_var``
            # (recorded in _nki_iota_offsets by codegen_iota_nki), NOT the plain
            # block offset. If this subscript is such an iota and its recorded
            # offset differs from offset_var, slice from the iota's true start so
            # the DMA reads the shifted columns (int4_gemm a_hi bug).
            if isinstance(fx_node_tdi_check, torch.fx.Node):
                _iota_ast = state.codegen.ast_for_fx_node(fx_node_tdi_check)
                if isinstance(_iota_ast, ast.AST):
                    _iota_nm = ast.unparse(_iota_ast)
                    _iota_off = state.device_function._nki_iota_offsets.get(_iota_nm)
                    if _iota_off is not None and _iota_off != offset_var:
                        _shifted_iota = f"{_iota_off}:{_iota_off}+{int(block_size)}"
                        if tdi == 0:
                            partition_offset_var = f"({_iota_off})"
                        slice_parts.append(_shifted_iota)
                        used_shifted_subscript = True
                        is_scalar_dim.append(False)
                        continue
            # Check if the subscript FX node is a SHIFTED tile (e.g.
            # tile_k.index + tile_c.begin * chunk_size). If so, emit the
            # shifted slice instead of the plain offset:offset+block range.
            _shifted_s = _nki_shifted_tile_subscript(fx_node_tdi_check, state, env)
            # Check if this block is inside a dynamic_range loop - if so, we
            # can't use offset_var in a slice (it's a register). Mark this
            # subscript with a sentinel token that the DMA emit will recognize
            # and substitute with .ap(scalar_offset=counter).
            _dyn_loops = getattr(state.device_function, "_nki_dyn_loops", {})
            if _shifted_s is not None:
                if tdi == 0:
                    partition_offset_var = _shifted_s.split(":", 1)[0].strip()
                slice_parts.append(_shifted_s)
                used_shifted_subscript = True
                is_scalar_dim.append(False)
                continue
            elif block_id in _dyn_loops:
                # If the subscript is compound (e.g. start + tile.index where
                # start is an SBUF scalar), the simple __DYN_AP__ sentinel is
                # insufficient — the counter doesn't include the SBUF offset.
                # In that case, try _nki_row_index_gather which handles
                # SBUF-scalar + tile.index combinations correctly.
                _dyn_counter_raw = _dyn_loops[block_id]["counter"]
                _needs_sbuf_offset = False
                if (
                    tdi == 0
                    and tensor.dim() == 2
                    and isinstance(_sub_val_load, torch.Tensor)
                    and isinstance(fx_node_tdi_check, torch.fx.Node)
                ):
                    _ck_target_dyn = str(getattr(fx_node_tdi_check, "target", ""))
                    if "add.Tensor" in _ck_target_dyn or "sub.Tensor" in _ck_target_dyn:
                        _sbuf_shapes_dyn = state.device_function._nki_sbuf_shapes
                        for _dyn_ck_arg in fx_node_tdi_check.args[:2]:
                            if isinstance(_dyn_ck_arg, torch.fx.Node):
                                _dyn_ck_ast = state.codegen.ast_for_fx_node(_dyn_ck_arg)
                                if isinstance(_dyn_ck_ast, ast.AST):
                                    _dyn_ck_nm = ast.unparse(_dyn_ck_ast)
                                    if _dyn_ck_nm in _sbuf_shapes_dyn:
                                        _needs_sbuf_offset = True
                                        break
                if _needs_sbuf_offset:
                    # Use row_gather path to handle SBUF-offset + dyn-tile.index
                    _row_gather_dyn = _nki_row_index_gather(
                        fx_node_tdi_check, state, int(block_size)
                    )
                    if _row_gather_dyn is not None:
                        if tdi == 0:
                            if isinstance(_row_gather_dyn, IndirectAP):
                                partition_offset_var = None
                            else:
                                partition_offset_var = (
                                    _row_gather_dyn.split(":", 1)[0].strip()
                                    if ":" in _row_gather_dyn
                                    else _row_gather_dyn
                                )
                        slice_parts.append(_row_gather_dyn)
                        is_scalar_dim.append(False)
                        continue
                _counter = _dyn_counter_raw
                slice_parts.append(DynamicAP(counter=_counter, block_size=int(block_size)))
                is_scalar_dim.append(False)
                continue
            elif (
                tdi == 0
                and tensor.dim() == 2
                and isinstance(_sub_val_load, torch.Tensor)
            ):
                # Check if the subscript is a non-contiguous gather (indirect load).
                # When block_id was found via size match but the subscript is a
                # computed gather (sorted_to_orig[indices], torch.where result, etc.),
                # use the row gather (.ap() with vector_offset) mechanism.
                row_gather = _nki_row_index_gather(fx_node_tdi_check, state, partition_dim)
                if row_gather is not None:
                    if tdi == 0:
                        if isinstance(row_gather, IndirectAP):
                            partition_offset_var = None
                        else:
                            partition_offset_var = row_gather.split(":", 1)[0].strip() if ":" in row_gather else row_gather
                    slice_parts.append(row_gather)
                    is_scalar_dim.append(False)
                    continue
                # row_gather returned None. Check if the shift was blocked because
                # of an SBUF operand. If so, this needs the flat gather path (else
                # branch below) — don't emit a plain slice. We detect this by
                # checking if fx_node_tdi_check is an add/sub involving an SBUF tile.
                _sbuf_shapes_ck = state.device_function._nki_sbuf_shapes
                _needs_flat_gather = False
                if isinstance(fx_node_tdi_check, torch.fx.Node):
                    _ck_target = str(getattr(fx_node_tdi_check, "target", ""))
                    if "add.Tensor" in _ck_target or "sub.Tensor" in _ck_target:
                        for _ck_arg in fx_node_tdi_check.args[:2]:
                            if isinstance(_ck_arg, torch.fx.Node):
                                _ck_ast = state.codegen.ast_for_fx_node(_ck_arg)
                                if isinstance(_ck_ast, ast.AST):
                                    _ck_nm = ast.unparse(_ck_ast)
                                    if (
                                        _ck_nm in _sbuf_shapes_ck
                                        or _ck_nm.startswith(("_nki_", "indices_"))
                                    ):
                                        _needs_flat_gather = True
                                        break
                if not _needs_flat_gather:
                    slice_parts.append(f"{offset_var}:{offset_var}+{int(block_size)}")
                    is_scalar_dim.append(False)
                    continue
                # _needs_flat_gather=True: this subscript has a SBUF-scalar shift
                # (e.g. A[start + tile.index]). Use the shifted subscript path
                # via _nki_row_index_gather which handles the SBUF broadcast.
                row_gather_sbuf = _nki_row_index_gather(
                    fx_node_tdi_check, state, int(block_size)
                )
                if row_gather_sbuf is not None:
                    if tdi == 0:
                        partition_offset_var = row_gather_sbuf.split(":", 1)[0].strip() if ":" in row_gather_sbuf else row_gather_sbuf
                    slice_parts.append(row_gather_sbuf)
                    is_scalar_dim.append(False)
                else:
                    # Last resort: use plain slice (may be wrong but avoids crash)
                    slice_parts.append(f"{offset_var}:{offset_var}+{int(block_size)}")
                    is_scalar_dim.append(False)
                continue
            else:
                slice_parts.append(f"{offset_var}:{offset_var}+{int(block_size)}")
                is_scalar_dim.append(False)
                continue
        else:
            # Tensor-valued row indexers such as ``weight[indices, tile_f]``
            # and ``A[start + tile_m.index, tile_k]`` lower to an HBM
            # vector-offset access pattern.  NKI expects row ids in a
            # ``[P, 1]`` uint32 SBUF tile; the column/K tile remains an
            # ordinary contiguous AP dimension.
            if (
                tdi == 0
                and tensor.dim() == 2
                and isinstance(_sub_val_load, torch.Tensor)
            ):
                row_gather = _nki_row_index_gather(
                    fx_node_tdi_check, state, partition_dim
                )
                if row_gather is not None:
                    slice_parts.append(row_gather)
                    is_scalar_dim.append(False)
                    continue

            # Detect "tile_index ± constant" subscripts (common in
            # concatenate / jagged kernels): if the FX subscript is an
            # aten.add/sub with tile_index as LHS and an int-constant RHS,
            # rewrite as a shifted slice of the underlying block.
            shifted = _nki_shifted_tile_subscript(fx_node_tdi_check, state, env)
            if shifted is not None:
                slice_parts.append(shifted)
                is_scalar_dim.append(False)
                used_shifted_subscript = True
                continue

            # Detect "base_per_partition + tile_index" pattern for indirect
            # gather (e.g. starts[:, None] + tile.index[None, :]). This is
            # the common jagged pattern: for each partition, a contiguous
            # F-element slice starting at base[p].
            # Emit .ap(pattern=..., vector_offset=base_tile, indirect_dim=0).
            _gather = _nki_indirect_gather(
                fx_node_tdi_check, state, env, tensor, slice_parts
            )
            if _gather is not None:
                slice_parts.append(_gather)
                is_scalar_dim.append(False)
                continue

            # Iota/tensor subscript at non-partition dim: represents a contiguous
            # range in the free dimension. Detect iota-based subscripts (common for
            # computed slices like a_tile_begin:a_tile_begin+a_tile_len which trace
            # as prims.iota) and emit the correct offset+length slice.
            if (
                tdi > 0
                and isinstance(_sub_val_load, torch.Tensor)
                and isinstance(fx_node_tdi_check, torch.fx.Node)
            ):
                # Check if this is an iota-based subscript (prims.iota.default).
                # The iota represents a contiguous range in the free dimension.
                # Its kwargs['start'] gives the offset and args[0] gives the length.
                _fx_target_name = str(getattr(fx_node_tdi_check, "target", ""))
                _is_iota = "iota" in _fx_target_name
                if _is_iota:
                    _fx_val = fx_node_tdi_check.meta.get("val")
                    if isinstance(_fx_val, torch.Tensor) and _fx_val.ndim == 1:
                        _iota_len_sym = _fx_val.size(0)
                        _iota_len = _resolve_dim(_iota_len_sym) if isinstance(_iota_len_sym, torch.SymInt) else int(_iota_len_sym)
                        # Get the start offset from kwargs['start']
                        _iota_start_node = fx_node_tdi_check.kwargs.get("start")
                        _start_expr = "0"
                        if isinstance(_iota_start_node, torch.fx.Node):
                            _start_ast = state.codegen.ast_for_fx_node(_iota_start_node)
                            if isinstance(_start_ast, ast.AST):
                                _start_expr = ast.unparse(_start_ast)
                        elif isinstance(_iota_start_node, (int, float)):
                            _start_expr = str(int(_iota_start_node))
                        # Emit slice: start:start+len
                        slice_parts.append(f"{_start_expr}:{_start_expr}+{_iota_len}")
                        is_scalar_dim.append(False)
                        continue

            # Computed partial slice: slice(start, stop) where start is not None.
            # Emit the slice as "start_expr:stop_expr" using FX AST evaluation.
            if isinstance(_sub_val_load, slice) and _sub_val_load.start is not None:
                # Get AST for the slice start and stop from FX
                _slice_start_ast = None
                _slice_stop_ast = None
                if isinstance(fx_node_tdi_check, torch.fx.Node):
                    # FX slice nodes have args[0]=start, args[1]=stop or similar
                    # Actually the FX subscript for a slice is represented differently.
                    # The slice values are in sub_val directly.
                    _start_val = _sub_val_load.start
                    _stop_val = _sub_val_load.stop
                    if isinstance(_start_val, torch.SymInt):
                        _start_expr = state.sympy_expr(_start_val._sympy_())
                    elif isinstance(_start_val, int):
                        _start_expr = str(_start_val)
                    else:
                        _start_expr = str(_start_val)
                    if isinstance(_stop_val, torch.SymInt):
                        _stop_expr = state.sympy_expr(_stop_val._sympy_())
                    elif isinstance(_stop_val, int):
                        _stop_expr = str(_stop_val)
                    else:
                        _stop_expr = str(_stop_val)
                    # Check if the dynamic range loop has a counter that can be used
                    _dyn_loops = getattr(state.device_function, "_nki_dyn_loops", {})
                    _is_in_dyn_loop = any(
                        bid_candidate in _dyn_loops
                        for bid_candidate in state.codegen.active_device_loops
                    )
                    if _is_in_dyn_loop:
                        # Use __DYN_AP__ sentinel with the slice extent
                        try:
                            _slice_len = int(_stop_val) - int(_start_val) if isinstance(_start_val, int) and isinstance(_stop_val, int) else None
                        except (TypeError, ValueError):
                            _slice_len = None
                        if _slice_len is None:
                            # Try computing length from SymInt
                            if isinstance(_start_val, torch.SymInt) and isinstance(_stop_val, torch.SymInt):
                                _len_sympy = _stop_val._sympy_() - _start_val._sympy_()
                                _len_sympy_subbed = _len_sympy.subs(_bs_subs)
                                try:
                                    _slice_len = int(_len_sympy_subbed)
                                except (TypeError, ValueError):
                                    _slice_len = int(env.size_hint(_stop_val)) - int(env.size_hint(_start_val))
                        if _slice_len is not None:
                            # Find the dyn loop counter to express start in terms of
                            for _bid_c, _dyn_info_c in _dyn_loops.items():
                                _counter_c = _dyn_info_c.get("counter_float") or _dyn_info_c.get("counter")
                                if _counter_c:
                                    slice_parts.append(DynamicAP(counter=_counter_c, block_size=_slice_len))
                                    is_scalar_dim.append(False)
                                    break
                            else:
                                slice_parts.append(f"{_start_expr}:{_stop_expr}")
                                is_scalar_dim.append(False)
                            continue
                    slice_parts.append(f"{_start_expr}:{_stop_expr}")
                    is_scalar_dim.append(False)
                    continue

            # Fixed slice for this dimension.
            size_i = tensor.size(tdi)
            size_str = (
                state.sympy_expr(size_i._sympy_())
                if isinstance(size_i, torch.SymInt)
                else str(size_i)
            )
            slice_parts.append(f"0:{size_str}")
            is_scalar_dim.append(False)

    if (
        tensor.dim() == 1
        and len(subscript_block_ids) == 1
        and subscript
        and isinstance(subscript[0], torch.Tensor)
        and not used_shifted_subscript
        and fx_subscript is not None
        and len(fx_subscript) >= 1
        and isinstance(fx_subscript[0], torch.fx.Node)
    ):
        def _emit_flat_masked(load_name: str, sbuf_shape: list[int]) -> str:
            if extra_mask is None:
                return load_name
            mask_name = ast.unparse(extra_mask)
            masked_name = device_fn.new_var("_nki_masked_load", dce=True)
            device_fn._nki_sbuf_shapes[masked_name] = list(sbuf_shape)
            device_fn._nki_sbuf_dtypes[masked_name] = dtype_str
            shape_str = ", ".join(str(d) for d in sbuf_shape)
            state.codegen.add_statement(
                statement_from_string(
                    f"{masked_name} = nl.ndarray([{shape_str}], "
                    f"{dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(f"nisa.memset({masked_name}, value=0)")
            )
            mask_shape = device_fn._nki_sbuf_shapes.get(mask_name, sbuf_shape)
            pred_src_name = mask_name
            if list(mask_shape) != list(sbuf_shape):
                pred_src_name = device_fn.new_var("_nki_mask_bcast", dce=True)
                device_fn._nki_sbuf_shapes[pred_src_name] = list(sbuf_shape)
                mask_dtype = device_fn._nki_sbuf_dtypes.get(
                    mask_name, "nl.int32"
                )
                device_fn._nki_sbuf_dtypes[pred_src_name] = mask_dtype
                shape_tuple = ", ".join(str(d) for d in sbuf_shape)
                state.codegen.add_statement(
                    statement_from_string(
                        f"{pred_src_name} = nl.broadcast_to({mask_name}, "
                        f"shape=({shape_tuple}))"
                    )
                )
                mask_shape = sbuf_shape
            mask_shape_str = ", ".join(str(d) for d in mask_shape)
            pred_name = device_fn.new_var("_nki_mask_pred", dce=True)
            device_fn._nki_sbuf_shapes[pred_name] = list(mask_shape)
            device_fn._nki_sbuf_dtypes[pred_name] = "nl.uint32"
            state.codegen.add_statement(
                statement_from_string(
                    f"{pred_name} = nl.ndarray([{mask_shape_str}], "
                    "nl.uint32, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy(dst={pred_name}, src={pred_src_name})"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy_predicated(dst={masked_name}, "
                    f"src={load_name}, predicate={pred_name})"
                )
            )
            return masked_name

        def _try_emit_flat_gather_2d(index_node: torch.fx.Node) -> ast.AST | None:
            index_val = index_node.meta.get("val")
            if not isinstance(index_val, torch.Tensor) or index_val.ndim not in (2, 3):
                return None
            if (
                index_node.op != "call_function"
                or len(index_node.args) < 2
                or not (
                    "add.Tensor" in str(index_node.target)
                    or index_node.target is torch.ops.aten.add.Tensor
                )
            ):
                return None
            lhs_node, rhs_node = index_node.args[:2]
            if not (
                isinstance(lhs_node, torch.fx.Node)
                and isinstance(rhs_node, torch.fx.Node)
            ):
                return None
            lhs_val = lhs_node.meta.get("val")
            rhs_val = rhs_node.meta.get("val")
            if not (
                isinstance(lhs_val, torch.Tensor)
                and isinstance(rhs_val, torch.Tensor)
                and lhs_val.ndim in (2, 3)
                and rhs_val.ndim in (2, 3)
            ):
                return None

            def _resolve_index_dim(dim: object) -> int:
                if isinstance(dim, int):
                    return dim
                if isinstance(dim, torch.SymInt):
                    return int(dim._sympy_().subs(_bs_subs))
                return int(dim)

            if index_val.ndim == 3:
                index_dim0 = _resolve_index_dim(index_val.shape[0])
                index_dim1 = _resolve_index_dim(index_val.shape[1])
                if index_dim1 == 1:
                    p_count = index_dim0
                elif index_dim0 == 1:
                    p_count = index_dim1
                else:
                    return None
                f_count = _resolve_index_dim(index_val.shape[2])
            else:
                p_count = _resolve_index_dim(index_val.shape[0])
                f_count = _resolve_index_dim(index_val.shape[1])

            def _shape_2d(value: torch.Tensor) -> tuple[int, int] | None:
                if value.ndim == 3:
                    dim0 = _resolve_index_dim(value.shape[0])
                    dim1 = _resolve_index_dim(value.shape[1])
                    if dim1 == 1:
                        return (
                            dim0,
                            _resolve_index_dim(value.shape[2]),
                        )
                    if dim0 == 1:
                        return (
                            dim1,
                            _resolve_index_dim(value.shape[2]),
                        )
                    else:
                        return None
                return (
                    _resolve_index_dim(value.shape[0]),
                    _resolve_index_dim(value.shape[1]),
                )

            lhs_shape = _shape_2d(lhs_val)
            rhs_shape = _shape_2d(rhs_val)
            if lhs_shape is None or rhs_shape is None:
                return None
            if lhs_shape == (p_count, 1) and rhs_shape == (1, f_count):
                base_node, feature_node = lhs_node, rhs_node
            elif rhs_shape == (p_count, 1) and lhs_shape == (1, f_count):
                base_node, feature_node = rhs_node, lhs_node
            else:
                return None

            base_ast = state.codegen.ast_for_fx_node(base_node)
            feature_ast = state.codegen.ast_for_fx_node(feature_node)
            if not isinstance(base_ast, ast.AST) or not isinstance(feature_ast, ast.AST):
                return None
            base_name = ast.unparse(base_ast)
            feature_name = ast.unparse(feature_ast)
            vec_offset = _nki_as_uint32_p1_vector(state, base_name, p_count)
            if vec_offset is None:
                return None

            feature_base_name = feature_name
            while "_copy" in feature_base_name:
                feature_base_name = feature_base_name[: feature_base_name.rfind("_copy")]
            feature_block_id: int | None = None
            for block_id in state.codegen.active_device_loops:
                try:
                    if state.codegen.index_var(block_id) == feature_base_name:
                        feature_block_id = block_id
                        break
                except (KeyError, IndexError):
                    continue
            if feature_block_id is None:
                return None

            dyn_loops = getattr(device_fn, "_nki_dyn_loops", {})
            if feature_block_id in dyn_loops:
                counter = dyn_loops[feature_block_id]["counter"]
                combined = device_fn.new_var("_ig_base_plus_counter", dce=True)
                counter_u32 = device_fn.new_var("_ig_counter_u32", dce=True)
                counter_bcast = device_fn.new_var("_ig_counter_bcast", dce=True)
                device_fn._nki_sbuf_shapes[combined] = [p_count, 1]
                device_fn._nki_sbuf_dtypes[combined] = "nl.uint32"
                device_fn._nki_sbuf_shapes[counter_u32] = [1, 1]
                device_fn._nki_sbuf_dtypes[counter_u32] = "nl.uint32"
                device_fn._nki_sbuf_shapes[counter_bcast] = [p_count, 1]
                device_fn._nki_sbuf_dtypes[counter_bcast] = "nl.uint32"
                state.codegen.add_statement(
                    statement_from_string(
                        f"{combined} = nl.ndarray([{p_count}, 1], "
                        "nl.uint32, buffer=nl.sbuf)"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.tensor_copy(dst={combined}, src={vec_offset})"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"{counter_u32} = nl.ndarray([1, 1], "
                        "nl.uint32, buffer=nl.sbuf)"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.tensor_copy(dst={counter_u32}, src={counter})"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"{counter_bcast} = nl.broadcast_to({counter_u32}, "
                        f"shape=({p_count}, 1))"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={combined}, data1={combined}, "
                        f"data2={counter_bcast}, op=nl.add)"
                    )
                )
                vec_offset = combined
            else:
                feature_offset = state.codegen.offset_var(feature_block_id)
                if feature_offset != "0":
                    combined = device_fn.new_var("_ig_base_plus_offset", dce=True)
                    device_fn._nki_sbuf_shapes[combined] = [p_count, 1]
                    device_fn._nki_sbuf_dtypes[combined] = "nl.uint32"
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{combined} = nl.ndarray([{p_count}, 1], "
                            "nl.uint32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={combined}, src={vec_offset})"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_scalar(dst={combined}, data={combined}, "
                            f"op0=nl.add, operand0={feature_offset})"
                        )
                    )
                    vec_offset = combined

            total_elems = int(tensor.numel())
            device_fn._nki_sbuf_shapes[sbuf_name] = [p_count, f_count]
            device_fn._nki_sbuf_dtypes[sbuf_name] = dtype_str
            state.codegen.add_statement(
                statement_from_string(
                    f"{sbuf_name} = nl.ndarray([{p_count}, {f_count}], "
                    f"{dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={sbuf_name}, "
                    f"src={name}.reshape([{total_elems}, 1]).ap("
                    f"pattern=[[1, {p_count}], [1, {f_count}]], "
                    f"vector_offset={vec_offset}, indirect_dim=0), "
                    "oob_mode=nisa.oob_mode.skip)"
                )
            )

            # When inside a jagged tile loop, oob_mode=skip only skips
            # physically-OOB addresses.  Logically-invalid k-positions
            # (k >= seqlen) still have valid HBM addresses (other rows' data)
            # and get incorrect values from the DMA.  Apply the jagged auto-
            # mask via predicated copy to zero out invalid k-positions.
            # For float buffers we use -inf fill so that:
            #   max(-inf, x) = x  (correct for amax reduction)
            #   exp(-inf)    = 0  (correct for exp-sum reduction)
            _active_lps_fg = state.codegen.active_device_loops
            from ..compile_environment import CompileEnvironment as _CE_fg
            _env_fg = _CE_fg.current()
            _fg_jagged_mask: str | None = None
            _fg_mask_shape: list[int] | None = None
            for _bid_fg in sorted(_active_lps_fg.keys(), reverse=True):
                if not _env_fg.is_jagged_tile(_bid_fg):
                    continue
                _bid_loops_fg = _active_lps_fg.get(_bid_fg, [])
                _strat_fg = _bid_loops_fg[-1].strategy if _bid_loops_fg else None
                if _strat_fg is None or not hasattr(_strat_fg, "mask_var"):
                    continue
                try:
                    _mv_fg = _strat_fg.mask_var(_bid_fg)
                except Exception:
                    continue
                if _mv_fg is None:
                    continue
                _ms_fg = _nki_lookup_sbuf_shape_dtype(state, _mv_fg)[0]
                if _ms_fg is not None and len(_ms_fg) == 2 and _ms_fg[1] == p_count:
                    # mask shape is [tile_b, k_count] where k_count == p_count
                    _fg_jagged_mask = _mv_fg
                    _fg_mask_shape = _ms_fg
                    break

            if _fg_jagged_mask is not None and dtype_str in ("nl.float32", "nl.float16", "nl.bfloat16"):
                _tb_size = _fg_mask_shape[0]  # tile_b dimension of mask
                _k_size = _fg_mask_shape[1]   # k_count = p_count
                # Transpose mask [tb, k] → [k, tb] then broadcast to [k, f_count]=[p_count, f_count]
                _fg_mcast = device_fn.new_var("_fg_mask_cast", dce=True)
                _fg_mtr = device_fn.new_var("_fg_mask_tr", dce=True)
                _fg_mcol = device_fn.new_var("_fg_mask_col", dce=True)
                _fg_mbcast = device_fn.new_var("_fg_mask_bcast", dce=True)
                _fg_mpred = device_fn.new_var("_fg_mask_pred", dce=True)
                _fg_masked = device_fn.new_var("_fg_masked_load", dce=True)
                _fg_mtr_dtype = "nl.float32"
                device_fn._nki_sbuf_shapes[_fg_mcast] = [_tb_size, _k_size]
                device_fn._nki_sbuf_dtypes[_fg_mcast] = _fg_mtr_dtype
                device_fn._nki_sbuf_shapes[_fg_mtr] = [_k_size, _tb_size]
                device_fn._nki_sbuf_dtypes[_fg_mtr] = _fg_mtr_dtype
                device_fn._nki_sbuf_shapes[_fg_mcol] = [_k_size, 1]
                device_fn._nki_sbuf_dtypes[_fg_mcol] = "nl.int32"
                device_fn._nki_sbuf_shapes[_fg_mbcast] = [p_count, f_count]
                device_fn._nki_sbuf_dtypes[_fg_mbcast] = "nl.int32"
                device_fn._nki_sbuf_shapes[_fg_mpred] = [p_count, f_count]
                device_fn._nki_sbuf_dtypes[_fg_mpred] = "nl.uint32"
                device_fn._nki_sbuf_shapes[_fg_masked] = [p_count, f_count]
                device_fn._nki_sbuf_dtypes[_fg_masked] = dtype_str
                # Emit mask broadcast: [tb, k] → cast → transpose psum → [k, 1] → broadcast [k, f]
                state.codegen.add_statement(statement_from_string(
                    f"{_fg_mcast} = nl.ndarray([{_tb_size}, {_k_size}], {_fg_mtr_dtype}, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"nisa.activation(dst={_fg_mcast}, op=nl.copy, data={_fg_jagged_mask})"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"{_fg_mtr} = nl.ndarray([{_k_size}, {_tb_size}], {_fg_mtr_dtype}, buffer=nl.psum)"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"nisa.nc_transpose(dst={_fg_mtr}, data={_fg_mcast})"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"{_fg_mcol} = nl.ndarray([{_k_size}, 1], nl.int32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"nisa.tensor_scalar(dst={_fg_mcol}, data={_fg_mtr}, op0=nl.add, operand0=0.0)"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"{_fg_mbcast} = nl.broadcast_to({_fg_mcol}, shape=({p_count}, {f_count}))"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"{_fg_mpred} = nl.ndarray([{p_count}, {f_count}], nl.uint32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"nisa.tensor_copy(dst={_fg_mpred}, src={_fg_mbcast})"
                ))
                # Apply predicated copy: invalid k-positions get -inf fill
                state.codegen.add_statement(statement_from_string(
                    f"{_fg_masked} = nl.ndarray([{p_count}, {f_count}], {dtype_str}, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"nisa.memset({_fg_masked}, value=float('-inf'))"
                ))
                state.codegen.add_statement(statement_from_string(
                    f"nisa.tensor_copy_predicated(dst={_fg_masked}, src={sbuf_name}, predicate={_fg_mpred})"
                ))
                # Return the masked buffer instead of the raw gather
                return expr_from_string(
                    _emit_flat_masked(_fg_masked, [p_count, f_count])
                )

            return expr_from_string(
                _emit_flat_masked(sbuf_name, [p_count, f_count])
            )

        def _try_emit_flat_gather_sum_dim1(index_node: torch.fx.Node) -> ast.AST | None:
            import os as _os_dbg2
            _debug_jagged = _os_dbg2.environ.get("HELION_DEBUG_JAGGED")
            index_val = index_node.meta.get("val")
            if _debug_jagged:
                print(f"[JAGGED_DBG] sum_dim1 entry: node={index_node.name} target={index_node.target} val_type={type(index_val).__name__} ndim={getattr(index_val,'ndim','?')}")
            if not isinstance(index_val, torch.Tensor) or index_val.ndim != 3:
                if _debug_jagged:
                    print(f"[JAGGED_DBG] sum_dim1 bail1: ndim={getattr(index_val,'ndim','?')}")
                return None
            if (
                index_node.op != "call_function"
                or len(index_node.args) < 2
                or not (
                    "add.Tensor" in str(index_node.target)
                    or index_node.target is torch.ops.aten.add.Tensor
                )
            ):
                if _debug_jagged:
                    print(f"[JAGGED_DBG] sum_dim1 bail2: op={index_node.op} target={index_node.target}")
                return None

            # Walk the FX graph forward from the load node to find a linear chain
            # of elementwise ops that terminates at either:
            #   (a) a dim=1 reduction (sum/amax/etc.) → fuse into gather+accumulate loop
            #   (b) a hl.store call → fuse into gather+transform+scatter loop
            # _chain_ops is the list of intermediate elementwise nodes.
            # _chain_terminal is 'reduce' or 'store' to indicate the terminal type.
            # If no linear chain is found, return None.
            from ...language._tracing_ops import _mask_to as _mask_to_fn
            # _store_fn is the hl.store API function (defined in this same module)
            _store_fn = store

            _dim1_reduce_targets = {
                torch.ops.aten.sum.dim_IntList,
                torch.ops.aten.mean.dim,
                torch.ops.aten.amax.default,
                torch.ops.aten.amin.default,
            }
            _elementwise_targets = {
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.where.self,
                torch.ops.prims.convert_element_type.default,
            }

            def _is_dim1_reduce(node: torch.fx.Node) -> bool:
                if node.op != "call_function" or node.target not in _dim1_reduce_targets:
                    return False
                _dim_arg = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
                if isinstance(_dim_arg, (list, tuple)):
                    return list(_dim_arg) == [1]
                return _dim_arg == 1

            def _is_store_node(node: torch.fx.Node) -> bool:
                return (
                    node.op == "call_function"
                    and node.target is _store_fn
                )

            def _walk_chain(
                load_node: torch.fx.Node,
            ) -> tuple[list[torch.fx.Node], str] | None:
                """Walk forward from load_node following single-user chains.
                Returns (chain, terminal_type) where terminal_type is 'reduce' or 'store',
                or None if no linear fusable chain is found."""
                chain: list[torch.fx.Node] = []
                cur = load_node
                while True:
                    users = list(cur.users)
                    if len(users) != 1:
                        return None
                    user = users[0]
                    if _is_dim1_reduce(user):
                        return chain, "reduce"
                    if _is_store_node(user):
                        return chain, "store"
                    if user.op == "call_function" and (
                        user.target in _elementwise_targets
                        or user.target is _mask_to_fn
                    ):
                        chain.append(user)
                        cur = user
                        continue
                    return None

            _chain_ops: list[torch.fx.Node] | None = None
            _chain_terminal: str = "reduce"
            _chain_store_node: torch.fx.Node | None = None
            if state.fx_node is not None:
                if _debug_jagged:
                    users = list(state.fx_node.users)
                    print(f"[JAGGED_DBG] walk_chain start: load={state.fx_node.name} users={len(users)}")
                    for _u in users[:3]:
                        print(f"  user={_u.name} target={_u.target} in_reduce={_is_dim1_reduce(_u)}")
                _walk_result = _walk_chain(state.fx_node)
                if _debug_jagged:
                    print(f"[JAGGED_DBG] walk_chain result: {_walk_result}")
                if _walk_result is not None:
                    _chain_ops, _chain_terminal = _walk_result
                    if _chain_terminal == "store":
                        # The store node is the user of the last chain node
                        _last_chain = _chain_ops[-1] if _chain_ops else state.fx_node
                        _chain_store_node = next(iter(_last_chain.users))
            else:
                _chain_ops = []
                _chain_terminal = "reduce"

            # If we cannot fuse (no linear chain to a reduction or store), bail out.
            if _chain_ops is None:
                # Debug: print why _walk_chain failed
                import os as _os_dbg
                if _os_dbg.environ.get("HELION_DEBUG_JAGGED"):
                    if state.fx_node is not None:
                        users = list(state.fx_node.users)
                        print(f"[JAGGED_DBG] _walk_chain failed for load node {state.fx_node.name}")
                        print(f"  users ({len(users)}): {[u.name + '/' + str(u.target) for u in users[:5]]}")
                        if users:
                            u0 = users[0]
                            print(f"  user[0] target={u0.target}, in dim1_reduce={_is_dim1_reduce(u0)}")
                            print(f"  user[0] users: {[uu.name for uu in list(u0.users)[:5]]}")
                return None

            lhs_ast = state.codegen.ast_for_fx_node(index_node.args[0])
            rhs_ast = state.codegen.ast_for_fx_node(index_node.args[1])
            if not isinstance(lhs_ast, ast.AST) or not isinstance(rhs_ast, ast.AST):
                return None
            lhs_name = ast.unparse(lhs_ast)
            rhs_name = ast.unparse(rhs_ast)

            def _lookup_shape(name_str: str) -> list[int] | None:
                shape = device_fn._nki_sbuf_shapes.get(name_str)
                if shape is not None:
                    return shape
                lookup = name_str
                while "_copy" in lookup:
                    lookup = lookup[: lookup.rfind("_copy")]
                    shape = device_fn._nki_sbuf_shapes.get(lookup)
                    if shape is not None:
                        return shape
                return None

            def _resolve_index_dim(dim: object) -> int:
                if isinstance(dim, int):
                    return dim
                if isinstance(dim, torch.SymInt):
                    return int(dim._sympy_().subs(_bs_subs))
                return int(dim)

            try:
                logical_shape = [_resolve_index_dim(dim) for dim in index_val.shape]
            except Exception:
                return None
            if len(logical_shape) != 3:
                return None
            p_count, k_count, m_count = logical_shape
            if p_count > NKI_PARTITION_MAX:
                return None

            def _node_name(node: object) -> str | None:
                if not isinstance(node, torch.fx.Node):
                    return None
                node_ast = state.codegen.ast_for_fx_node(node)
                if not isinstance(node_ast, ast.AST):
                    return None
                return ast.unparse(node_ast)

            def _logical_2d_shape(node: object) -> list[int] | None:
                if not isinstance(node, torch.fx.Node):
                    return None
                node_name = _node_name(node)
                if node_name is not None:
                    node_shape = _lookup_shape(node_name)
                    if node_shape in ([p_count, k_count], [1, m_count]):
                        return list(node_shape)
                node_val = node.meta.get("val")
                if not isinstance(node_val, torch.Tensor):
                    return None
                try:
                    node_shape = [_resolve_index_dim(dim) for dim in node_val.shape]
                except Exception:
                    return None
                if node_shape == [p_count, k_count]:
                    return [p_count, k_count]
                if node_shape == [p_count, k_count, 1]:
                    return [p_count, k_count]
                if node_shape == [1, m_count]:
                    return [1, m_count]
                if node_shape == [1, 1, m_count]:
                    return [1, m_count]
                return None

            def _scalar_expr(node: object) -> str | None:
                if isinstance(node, (int, bool)):
                    return str(int(node))
                if isinstance(node, float):
                    return repr(node)
                if isinstance(node, torch.SymInt):
                    return state.sympy_expr(node._sympy_())
                if not isinstance(node, torch.fx.Node):
                    return None
                node_val = node.meta.get("val")
                if isinstance(node_val, (int, bool)):
                    return str(int(node_val))
                if isinstance(node_val, float):
                    return repr(node_val)
                if isinstance(node_val, torch.SymInt):
                    return state.sympy_expr(node_val._sympy_())
                node_ast = state.codegen.ast_for_fx_node(node)
                if isinstance(node_ast, ast.AST):
                    return ast.unparse(node_ast)
                return None

            def _mul_base_and_stride(
                node: object,
            ) -> tuple[torch.fx.Node, str] | None:
                if not isinstance(node, torch.fx.Node):
                    return None
                node_target = str(getattr(node, "target", ""))
                if (
                    node.op != "call_function"
                    or len(node.args) < 2
                    or not (
                        "mul.Tensor" in node_target
                        or "mul.Scalar" in node_target
                        or node.target is torch.ops.aten.mul.Tensor
                        or node.target is torch.ops.aten.mul.Scalar
                    )
                ):
                    return None
                for base_arg, stride_arg in (
                    (node.args[0], node.args[1]),
                    (node.args[1], node.args[0]),
                ):
                    if _logical_2d_shape(base_arg) == [p_count, k_count]:
                        stride_expr = _scalar_expr(stride_arg)
                        if stride_expr is not None:
                            assert isinstance(base_arg, torch.fx.Node)
                            return base_arg, stride_expr
                return None

            base_name: str | None = None
            feature_index_name: str | None = None
            base_stride_expr: str | None = None
            lhs_logical_shape = _logical_2d_shape(index_node.args[0])
            rhs_logical_shape = _logical_2d_shape(index_node.args[1])
            if lhs_logical_shape == [p_count, k_count] and rhs_logical_shape == [
                1,
                m_count,
            ]:
                base_name = lhs_name
                feature_index_name = rhs_name
            elif rhs_logical_shape == [p_count, k_count] and lhs_logical_shape == [
                1,
                m_count,
            ]:
                base_name = rhs_name
                feature_index_name = lhs_name
            else:
                lhs_mul = _mul_base_and_stride(index_node.args[0])
                rhs_mul = _mul_base_and_stride(index_node.args[1])
                if lhs_mul is not None and rhs_logical_shape == [1, m_count]:
                    base_node, base_stride_expr = lhs_mul
                    base_name = _node_name(base_node)
                    feature_index_name = rhs_name
                elif rhs_mul is not None and lhs_logical_shape == [1, m_count]:
                    base_node, base_stride_expr = rhs_mul
                    base_name = _node_name(base_node)
                    feature_index_name = lhs_name
                else:
                    return None
            if base_name is None or feature_index_name is None:
                return None

            feature_offset = "0"
            feature_base_name = feature_index_name
            while "_copy" in feature_base_name:
                feature_base_name = feature_base_name[: feature_base_name.rfind("_copy")]
            for block_id in state.codegen.active_device_loops:
                try:
                    if state.codegen.index_var(block_id) == feature_base_name:
                        feature_offset = state.codegen.offset_var(block_id)
                        break
                except (KeyError, IndexError):
                    continue
            k_index_name: str | None = None
            for block_id in state.codegen.active_device_loops:
                try:
                    candidate = state.codegen.index_var(block_id)
                except (KeyError, IndexError):
                    continue
                if candidate == feature_base_name:
                    continue
                if _lookup_shape(candidate) == [1, k_count]:
                    k_index_name = candidate
                    break

            mask_node = (
                state.fx_node.args[2]
                if state.fx_node is not None and len(state.fx_node.args) >= 3
                else None
            )
            if not isinstance(mask_node, torch.fx.Node):
                # No explicit extra_mask — try to find the jagged tile's auto-mask.
                # The tile_strategy generates mask_k = index[None,:] < parent[:,None]
                # and registers it in mask_vars with shape [p_count, k_count].
                # Look it up using the CURRENT active loop strategy (not all strategies).
                # Using all strategies would find the first loop's mask even when we're
                # inside a second loop with the same block_idx but a different mask var.
                _jagged_row_mask: str | None = None
                _tile_dispatch = getattr(state, "tile_strategy", None)
                _active_loops = state.codegen.active_device_loops
                if _tile_dispatch is not None:
                    # Upstream wraps block_id_to_strategy in a
                    # BlockIDStrategyMapping (no .values(); has .items());
                    # older/dict forms have .values(). Support both.
                    _bid_map = getattr(_tile_dispatch, "block_id_to_strategy", {})
                    if hasattr(_bid_map, "values"):
                        _strategies = list(_bid_map.values())
                    else:
                        _strategies = [_v for _, _v in _bid_map.items()]
                    # Iterate in descending block_id order so we pick the
                    # innermost active jagged tile's mask (highest block_id =
                    # latest allocated = innermost loop).  This prevents an
                    # outer jagged tile's mask from being used as the row
                    # predicate for a gather that lives inside the inner loop.
                    for _bid in sorted(_active_loops.keys(), reverse=True):
                        _bid_loops = _active_loops.get(_bid, [])
                        _cur_strat = _bid_loops[-1].strategy if _bid_loops else None
                        if _cur_strat is not None and hasattr(_cur_strat, "mask_var"):
                            try:
                                _mvar = _cur_strat.mask_var(_bid)
                            except Exception:
                                _mvar = None
                            if _mvar is not None:
                                _mshape = _nki_lookup_sbuf_shape_dtype(state, _mvar)[0]
                                if _mshape == [p_count, k_count]:
                                    _jagged_row_mask = _mvar
                                    break
                if _debug_jagged:
                    print(f"[JAGGED_DBG] jagged_row_mask={_jagged_row_mask}, strategies={[type(s).__name__ for s in _strategies] if _tile_dispatch else 'none'}, p_count={p_count}, k_count={k_count}")
                    print(f"  active_device_loops={list(getattr(state.codegen, 'active_device_loops', {}).keys())}")
                    # Show all registered SBUF shapes that match [p,k] or look like masks
                    _sbuf = device_fn._nki_sbuf_shapes
                    _matching = [(k, v) for k, v in _sbuf.items() if v == [p_count, k_count]]
                    print(f"  SBUF vars with shape [{p_count},{k_count}]: {_matching[:5]}")
                    _mask_like = [(k, v) for k, v in _sbuf.items() if ("cmp" in k or "mask" in k or "pred" in k)]
                    print(f"  Mask-like SBUF vars: {_mask_like[:10]}")
                    from ..compile_environment import CompileEnvironment as _CE_dbg
                    _env_dbg = _CE_dbg.current()
                    print(f"  jagged_tile_parent_ids: {_env_dbg.jagged_tile_parent_ids}")
                    print(f"  jagged_tile_mask_shapes: {_env_dbg.jagged_tile_mask_shapes}")
                    for _strat in (_strategies or []):
                        if hasattr(_strat, "mask_vars"):
                            for _bid2, _mv2 in list(_strat.mask_vars.items()):
                                if _mv2:
                                    _ms2 = _nki_lookup_sbuf_shape_dtype(state, _mv2)[0]
                                    print(f"  mask_var bid={_bid2} var={_mv2} shape={_ms2}")
                # If strategy mask lookup failed (mask not yet in SBUF shapes),
                # try the jagged tile's registered mask from the env.
                # The mask variable name is known even if its shape isn't yet registered.
                if _jagged_row_mask is None:
                    from ..compile_environment import CompileEnvironment as _CE_jg
                    _env_jg = _CE_jg.current()
                    # Find which block_id is the jagged tile (k_count dimension)
                    for _bid_jg, _parent_ids in _env_jg.jagged_tile_parent_ids.items():
                        # Check if this jagged tile has the right shape
                        _mask_shapes_jg = _env_jg.jagged_tile_mask_shapes.get(_bid_jg)
                        if _mask_shapes_jg is not None and len(_mask_shapes_jg) == 2:
                            # Try to resolve: [parent_dim, k_dim]
                            try:
                                _p_jg = int(_mask_shapes_jg[0]._sympy_().subs(_bs_subs)) if isinstance(_mask_shapes_jg[0], torch.SymInt) else int(_mask_shapes_jg[0])
                                _k_jg = int(_mask_shapes_jg[1]._sympy_().subs(_bs_subs)) if isinstance(_mask_shapes_jg[1], torch.SymInt) else int(_mask_shapes_jg[1])
                            except Exception:
                                continue
                            if _p_jg == p_count and _k_jg == k_count:
                                # Found the right jagged tile — get its mask var.
                                # Prefer the current active loop's strategy so that
                                # multiple loops with the same block_id each use
                                # their own freshly-allocated mask variable.
                                _bid_loops_fb = _active_loops.get(_bid_jg, [])
                                _cur_strat_fb = _bid_loops_fb[-1].strategy if _bid_loops_fb else None
                                _fallback_strats = []
                                if _cur_strat_fb is not None:
                                    _fallback_strats.append(_cur_strat_fb)
                                _fallback_strats.extend(s for s in (_strategies or []) if s is not _cur_strat_fb)
                                for _strat in _fallback_strats:
                                    if hasattr(_strat, "mask_vars"):
                                        _mv_jg = _strat.mask_vars.get(_bid_jg)
                                        if _mv_jg is not None:
                                            # Register the shape so downstream code can use it
                                            device_fn._nki_sbuf_shapes[_mv_jg] = [p_count, k_count]
                                            device_fn._nki_sbuf_dtypes[_mv_jg] = "nl.int32"
                                            _jagged_row_mask = _mv_jg
                                            break
                            if _jagged_row_mask is not None:
                                break
                if _jagged_row_mask is None:
                    return None
                # Synthesize: use the tile mask as the row mask, no feature mask
                row_mask_name = _jagged_row_mask
                feature_mask_name = None
                _skip_mask_processing = True
            else:
                _skip_mask_processing = False

            def _node_contains_name(obj: object, target_name: str) -> bool:
                if isinstance(obj, torch.fx.Node):
                    obj_ast = state.codegen.ast_for_fx_node(obj)
                    if isinstance(obj_ast, ast.AST):
                        obj_name = ast.unparse(obj_ast)
                        while "_copy" in obj_name:
                            obj_name = obj_name[: obj_name.rfind("_copy")]
                        if obj_name == target_name:
                            return True
                    return _node_contains_name(obj.args, target_name) or _node_contains_name(
                        obj.kwargs, target_name
                    )
                if isinstance(obj, (list, tuple)):
                    return any(_node_contains_name(item, target_name) for item in obj)
                if isinstance(obj, dict):
                    return any(_node_contains_name(item, target_name) for item in obj.values())
                return False

            if not _skip_mask_processing:
                mask_args = [
                    arg for arg in getattr(mask_node, "args", ()) if isinstance(arg, torch.fx.Node)
                ]
                row_mask_name: str | None = None
                feature_mask_name: str | None = None
                if len(mask_args) < 2:
                    mask_ast = state.codegen.ast_for_fx_node(mask_node)
                    if isinstance(mask_ast, ast.AST):
                        row_mask_name = ast.unparse(mask_ast)
                else:
                    for mask_arg in mask_args[:2]:
                        mask_ast = state.codegen.ast_for_fx_node(mask_arg)
                        if not isinstance(mask_ast, ast.AST):
                            continue
                        mask_name = ast.unparse(mask_ast)
                        if k_index_name is not None and _node_contains_name(
                            mask_arg, k_index_name
                        ):
                            row_mask_name = mask_name
                        elif _node_contains_name(mask_arg, feature_base_name):
                            feature_mask_name = mask_name
                        elif row_mask_name is None:
                            row_mask_name = mask_name
                        else:
                            feature_mask_name = mask_name
                if row_mask_name is None:
                    return None

                row_mask_shape = _lookup_shape(row_mask_name)
                if row_mask_shape != [p_count, k_count]:
                    return None

                # The row_mask comparison (k < seqlens) is correctly emitted by the
                # backend.py _nki_compare fix when needed (re-emits fresh iota before
                # the comparison if k_index was mutated by flat-index arithmetic).

            if feature_mask_name is not None and _lookup_shape(feature_mask_name) != [
                p_count,
                m_count,
            ]:
                feature_mask_name = None

            total_elems = int(tensor.numel())
            out_p = p_count
            out_m = m_count
            if _chain_terminal == "reduce":
                # Reduce path: allocate [p_count, m_count] accumulator, zero-init
                acc_name = device_fn.new_var("_nki_flat_sum", dce=True)
                device_fn._nki_sbuf_shapes[acc_name] = [out_p, out_m]
                device_fn._nki_sbuf_dtypes[acc_name] = dtype_str
                state.codegen.add_statement(
                    statement_from_string(
                        f"{acc_name} = nl.ndarray([{out_p}, {out_m}], "
                        f"{dtype_str}, buffer=nl.sbuf)"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(f"nisa.memset({acc_name}, value=0)")
                )
            else:
                # Store path: no accumulator needed; use a placeholder name for the
                # return expression (the per-k tile will be scattered directly).
                acc_name = device_fn.new_var("_nki_flat_scatter", dce=True)
                device_fn._nki_sbuf_shapes[acc_name] = [out_p, out_m]
                device_fn._nki_sbuf_dtypes[acc_name] = dtype_str

            k_var = device_fn.new_var("_nki_gather_k")
            base_col = device_fn.new_var("_nki_gather_base", dce=True)
            base_u32 = device_fn.new_var("_nki_gather_base_u32", dce=True)
            gathered = device_fn.new_var("_nki_gather_tile", dce=True)
            row_pred_col = device_fn.new_var("_nki_row_pred_col", dce=True)
            row_pred_full = device_fn.new_var("_nki_row_pred", dce=True)
            pred_name = row_pred_full
            masked = device_fn.new_var("_nki_gather_masked", dce=True)
            device_fn._nki_sbuf_shapes[base_col] = [p_count, 1]
            device_fn._nki_sbuf_dtypes[base_col] = "nl.int32"
            device_fn._nki_sbuf_shapes[base_u32] = [p_count, 1]
            device_fn._nki_sbuf_dtypes[base_u32] = "nl.uint32"
            device_fn._nki_sbuf_shapes[gathered] = [p_count, m_count]
            device_fn._nki_sbuf_dtypes[gathered] = dtype_str
            device_fn._nki_sbuf_shapes[row_pred_col] = [p_count, 1]
            device_fn._nki_sbuf_dtypes[row_pred_col] = "nl.uint32"
            device_fn._nki_sbuf_shapes[row_pred_full] = [p_count, m_count]
            device_fn._nki_sbuf_dtypes[row_pred_full] = "nl.uint32"
            device_fn._nki_sbuf_shapes[masked] = [p_count, m_count]
            device_fn._nki_sbuf_dtypes[masked] = dtype_str

            body: list[ast.AST] = [
                statement_from_string(
                    f"{base_col} = nl.ndarray([{p_count}, 1], nl.int32, buffer=nl.sbuf)"
                ),
                statement_from_string(
                    f"nisa.tensor_copy(dst={base_col}, "
                    f"src={base_name}[0:{p_count}, {k_var}:{k_var} + 1])"
                ),
            ]
            if base_stride_expr is not None:
                body.append(
                    statement_from_string(
                        f"nisa.tensor_scalar(dst={base_col}, data={base_col}, "
                        f"op0=nl.multiply, operand0={base_stride_expr}, op1=None)"
                    )
                )
                if feature_offset != "0":
                    body.append(
                        statement_from_string(
                            f"nisa.tensor_scalar(dst={base_col}, data={base_col}, "
                            f"op0=nl.add, operand0={feature_offset}, op1=None)"
                        )
                    )
            elif feature_offset != "0":
                body.append(
                    statement_from_string(
                        f"nisa.tensor_scalar(dst={base_col}, data={base_col}, "
                        f"op0=nl.add, operand0={feature_offset}, op1=None)"
                    )
                )
            body.extend(
                [
                    statement_from_string(
                        f"{base_u32} = nl.ndarray([{p_count}, 1], nl.uint32, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.tensor_copy(dst={base_u32}, src={base_col})"
                    ),
                    statement_from_string(
                        f"{gathered} = nl.ndarray([{p_count}, {m_count}], "
                        f"{dtype_str}, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.dma_copy(dst={gathered}, "
                        f"src={name}.reshape([{total_elems}, 1]).ap("
                        f"pattern=[[1, {p_count}], [1, {m_count}]], "
                        f"vector_offset={base_u32}, indirect_dim=0), "
                        "oob_mode=nisa.oob_mode.skip)"
                    ),
                    statement_from_string(
                        f"{row_pred_col} = nl.ndarray([{p_count}, 1], "
                        "nl.uint32, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.tensor_copy(dst={row_pred_col}, "
                        f"src={row_mask_name}[0:{p_count}, {k_var}:{k_var} + 1])"
                    ),
                    statement_from_string(
                        f"{row_pred_full} = nl.ndarray([{p_count}, {m_count}], "
                        "nl.uint32, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.tensor_copy(dst={row_pred_full}, "
                        f"src=nl.broadcast_to({row_pred_col}, shape=({p_count}, {m_count})))"
                    ),
                ]
            )
            if feature_mask_name is not None:
                feature_pred = device_fn.new_var("_nki_feature_pred", dce=True)
                combined_pred = device_fn.new_var("_nki_combined_pred", dce=True)
                device_fn._nki_sbuf_shapes[feature_pred] = [p_count, m_count]
                device_fn._nki_sbuf_dtypes[feature_pred] = "nl.uint32"
                device_fn._nki_sbuf_shapes[combined_pred] = [p_count, m_count]
                device_fn._nki_sbuf_dtypes[combined_pred] = "nl.uint32"
                body.extend(
                    [
                        statement_from_string(
                            f"{feature_pred} = nl.ndarray([{p_count}, {m_count}], "
                            "nl.uint32, buffer=nl.sbuf)"
                        ),
                        statement_from_string(
                            f"nisa.tensor_copy(dst={feature_pred}, src={feature_mask_name})"
                        ),
                        statement_from_string(
                            f"{combined_pred} = nl.ndarray([{p_count}, {m_count}], "
                            "nl.uint32, buffer=nl.sbuf)"
                        ),
                        statement_from_string(
                            f"nisa.tensor_tensor(dst={combined_pred}, "
                            f"data1={row_pred_full}, data2={feature_pred}, "
                            "op=nl.bitwise_and)"
                        ),
                    ]
                )
                pred_name = combined_pred
            # Allocate and zero-fill the masked tile, then predicated-copy the gathered data.
            body.extend(
                [
                    statement_from_string(
                        f"{masked} = nl.ndarray([{p_count}, {m_count}], "
                        f"{dtype_str}, buffer=nl.sbuf)"
                    ),
                    statement_from_string(f"nisa.memset({masked}, value=0)"),
                    statement_from_string(
                        f"nisa.tensor_copy_predicated(dst={masked}, "
                        f"src={gathered}, predicate={pred_name})"
                    ),
                ]
            )
            # Emit chain ops (elementwise transforms between load and reduction).
            # pre_loop_stmts: ops that must be emitted BEFORE the k-loop (e.g. transpose)
            pre_loop_stmts: list[ast.AST] = []
            _cur_tile = masked
            assert _chain_ops is not None
            for _chain_node in _chain_ops:
                _target = _chain_node.target
                # _mask_to: no-op on NKI (masking already done by predicated copy)
                if _target is _mask_to_fn:
                    continue
                # where(mask, value, 0.0): after chain transforms, some positions may
                # be non-zero for invalid elements (e.g. 0 - mean ≠ 0 after subtract).
                # Re-apply the predicate mask to zero out invalid positions.
                if _target is torch.ops.aten.where.self:
                    # For reduce path: must re-mask so invalid positions don't corrupt
                    # the accumulated sum. For store path: OOB scatter handles it.
                    _where_false2 = _chain_node.args[2] if len(_chain_node.args) > 2 else None
                    _is_zero_fill2 = isinstance(_where_false2, (int, float)) and _where_false2 == 0
                    if not _is_zero_fill2 and isinstance(_where_false2, torch.fx.Node):
                        _wf_val2 = _where_false2.meta.get("val")
                        if isinstance(_wf_val2, (int, float)):
                            _is_zero_fill2 = (_wf_val2 == 0)
                        elif isinstance(_wf_val2, torch.Tensor) and _wf_val2.numel() == 1:
                            # Check FX node args for scalar(0) pattern
                            _wa2 = getattr(_where_false2, 'args', ())
                            if _wa2 and isinstance(_wa2[0], (int, float)) and _wa2[0] == 0:
                                _is_zero_fill2 = True
                            else:
                                # Try to check constant tensor value using try/except to
                                # prevent TorchInductor guard errors on SymInt shapes
                                try:
                                    with torch.no_grad():
                                        _val_cpu = _wf_val2.to('cpu')
                                        _is_zero_fill2 = bool((_val_cpu == 0).all().item())
                                except Exception:
                                    pass
                    _should_remask = _is_zero_fill2 and pred_name is not None and _chain_terminal == "reduce"
                    if _should_remask:
                        _remasked = device_fn.new_var("_nki_chain_masked", dce=True)
                        device_fn._nki_sbuf_shapes[_remasked] = [p_count, m_count]
                        device_fn._nki_sbuf_dtypes[_remasked] = dtype_str
                        body.append(statement_from_string(
                            f"{_remasked} = nl.ndarray([{p_count}, {m_count}], {dtype_str}, buffer=nl.sbuf)"
                        ))
                        body.append(statement_from_string(
                            f"nisa.memset({_remasked}, value=0)"
                        ))
                        body.append(statement_from_string(
                            f"nisa.tensor_copy_predicated(dst={_remasked}, src={_cur_tile}, predicate={pred_name})"
                        ))
                        _cur_tile = _remasked
                    continue
                # convert_element_type: dtype cast
                if _target is torch.ops.prims.convert_element_type.default:
                    _cast_dtype_val = _chain_node.args[1] if len(_chain_node.args) > 1 else None
                    if _cast_dtype_val is not None:
                        _cast_tile = device_fn.new_var("_nki_chain_tile", dce=True)
                        device_fn._nki_sbuf_shapes[_cast_tile] = [p_count, m_count]
                        try:
                            _nki_dtype = {
                                torch.float32: "nl.float32",
                                torch.float16: "nl.float16",
                                torch.bfloat16: "nl.bfloat16",
                                torch.int32: "nl.int32",
                            }.get(_cast_dtype_val, dtype_str)
                        except TypeError:
                            _nki_dtype = dtype_str
                        device_fn._nki_sbuf_dtypes[_cast_tile] = _nki_dtype
                        body.append(
                            statement_from_string(
                                f"{_cast_tile} = nl.ndarray([{p_count}, {m_count}], "
                                f"{_nki_dtype}, buffer=nl.sbuf)"
                            )
                        )
                        body.append(
                            statement_from_string(
                                f"nisa.tensor_copy(dst={_cast_tile}, src={_cur_tile})"
                            )
                        )
                        _cur_tile = _cast_tile
                    continue
                # sub/add/mul: tensor-tensor ops
                if _target in (  # sub/add/mul chain ops
                    torch.ops.aten.sub.Tensor,
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.mul.Tensor,
                ):
                    _op_map = {
                        torch.ops.aten.sub.Tensor: "nl.subtract",
                        torch.ops.aten.add.Tensor: "nl.add",
                        torch.ops.aten.mul.Tensor: "nl.multiply",
                    }
                    _nki_op = _op_map[_target]
                    # Determine which arg is the current tile and which is "other"
                    _arg0 = _chain_node.args[0] if len(_chain_node.args) > 0 else None
                    _arg1 = _chain_node.args[1] if len(_chain_node.args) > 1 else None
                    # Check if this is self-op (e.g. mul tile * tile for squaring).
                    # Also handles Helion trace of x*x as mul(x, None) (single-arg form).
                    if _arg0 is _arg1 or (_arg1 is None and _nki_op == "nl.multiply"):
                        # Before squaring, re-apply the row predicate mask to zero out
                        # OOB positions.  A prior subtract chain op (e.g. x - mean) turns
                        # OOB zeros into -mean; squaring them gives mean^2 which inflates
                        # the accumulated variance.  Re-masking ensures OOB^2 = 0.
                        # Only apply when pred_name and _cur_tile have matching shapes.
                        _cur_sq_shape = device_fn._nki_sbuf_shapes.get(_cur_tile)
                        _pred_sq_shape = device_fn._nki_sbuf_shapes.get(pred_name) if pred_name else None
                        if (
                            pred_name is not None
                            and _cur_sq_shape is not None
                            and _pred_sq_shape is not None
                            and list(_cur_sq_shape) == list(_pred_sq_shape)
                        ):
                            _sq_remasked = device_fn.new_var("_nki_sq_remasked", dce=True)
                            device_fn._nki_sbuf_shapes[_sq_remasked] = list(_cur_sq_shape)
                            device_fn._nki_sbuf_dtypes[_sq_remasked] = dtype_str
                            body.append(statement_from_string(
                                f"{_sq_remasked} = nl.ndarray([{_cur_sq_shape[0]}, {_cur_sq_shape[1]}], "
                                f"{dtype_str}, buffer=nl.sbuf)"
                            ))
                            body.append(statement_from_string(
                                f"nisa.memset({_sq_remasked}, value=0)"
                            ))
                            body.append(statement_from_string(
                                f"nisa.tensor_copy_predicated(dst={_sq_remasked}, "
                                f"src={_cur_tile}, predicate={pred_name})"
                            ))
                            _cur_tile = _sq_remasked
                        body.append(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={_cur_tile}, data1={_cur_tile}, "
                                f"data2={_cur_tile}, op={_nki_op})"
                            )
                        )
                        continue
                    # Find the "other" arg (not the chain predecessor)
                    # The chain predecessor could be _cur_tile's source node; we check by
                    # walking back to find which arg is the load-chain node.
                    _other_arg = None
                    _tile_is_arg0 = True
                    # We need to figure out which arg is the "current tile" (chain input)
                    # and which is the external operand.
                    # The chain input is the node we just processed (or the load node for
                    # first chain node).  Walk the chain to find previous node.
                    _prev_chain_idx = _chain_ops.index(_chain_node)
                    if _prev_chain_idx == 0:
                        _chain_input_node = state.fx_node
                    else:
                        _chain_input_node = _chain_ops[_prev_chain_idx - 1]
                    if _arg0 is _chain_input_node or (
                        isinstance(_arg0, torch.fx.Node)
                        and _arg0 in (_chain_input_node,)
                    ):
                        _other_arg = _arg1
                        _tile_is_arg0 = True
                    elif _arg1 is _chain_input_node or (
                        isinstance(_arg1, torch.fx.Node)
                        and _arg1 in (_chain_input_node,)
                    ):
                        _other_arg = _arg0
                        _tile_is_arg0 = False
                    else:
                        # Fallback: first arg is tile, second is other
                        _other_arg = _arg1
                        _tile_is_arg0 = True
                    # Get the NKI expression for _other_arg
                    if _other_arg is None:
                        continue
                    _other_expr: str | None = None
                    if isinstance(_other_arg, torch.fx.Node):
                        _other_ast = state.codegen.ast_for_fx_node(_other_arg)
                        if isinstance(_other_ast, ast.AST):
                            _other_expr = ast.unparse(_other_ast)
                        # If not found directly, walk up through view/slice/transpose ops
                        if _other_expr is None:
                            _walk_node = _other_arg
                            _transparent_ops = {
                                'view', 'expand', 'reshape', 'unsqueeze', 'squeeze',
                                'permute', 'transpose', 'select', 'narrow',
                                'aten.view.default', 'aten.expand.default',
                                'aten.unsqueeze.default', 'aten.squeeze.default',
                                'aten.permute.default',
                                'subscript',  # helion.language.view_ops.subscript
                            }
                            _visited = set()
                            while _walk_node is not None and id(_walk_node) not in _visited:
                                _visited.add(id(_walk_node))
                                _tname = str(getattr(_walk_node, 'target', ''))
                                _is_transparent = any(t in _tname for t in _transparent_ops)
                                if not _is_transparent:
                                    break
                                _args = getattr(_walk_node, 'args', ())
                                _walk_node = _args[0] if _args and isinstance(_args[0], torch.fx.Node) else None
                                if _walk_node is not None:
                                    _walk_ast = state.codegen.ast_for_fx_node(_walk_node)
                                    if isinstance(_walk_ast, ast.AST):
                                        _other_expr = ast.unparse(_walk_ast)
                                        break
                    elif isinstance(_other_arg, (int, float)):
                        _other_expr = repr(_other_arg)
                    if _other_expr is None:
                        continue
                    # Handle shape mismatch for _other_expr vs [p_count, m_count]
                    _other_bcast = _other_expr
                    _other_shape = device_fn._nki_sbuf_shapes.get(_other_expr)
                    # Walk copy aliases to find shape
                    if _other_shape is None:
                        _oe_lookup = _other_expr
                        while _other_shape is None and "_copy" in _oe_lookup:
                            _oe_lookup = _oe_lookup[:_oe_lookup.rfind("_copy")]
                            _other_shape = device_fn._nki_sbuf_shapes.get(_oe_lookup)
                    if _other_shape is not None and list(_other_shape) != [p_count, m_count]:
                        _oe_dtype = device_fn._nki_sbuf_dtypes.get(_other_expr, dtype_str)
                        # Use tensor_scalar if we can transpose _other to [p_count, 1]
                        # (broadcasts over free dim m_count). Otherwise fallback to broadcast_to.
                        # Broadcast the [1, P] mean/rstd to [P, M] or [P, 1] for tensor_tensor.
                        # Try to broadcast to [p_count, m_count] directly — NKI broadcast_to
                        # supports [1, N] → [M, N] (repeating partition row).
                        _bc_tile = device_fn.new_var("_nki_chain_bcast", dce=True)
                        device_fn._nki_sbuf_shapes[_bc_tile] = [p_count, m_count]
                        device_fn._nki_sbuf_dtypes[_bc_tile] = dtype_str
                        if _other_shape == [1, p_count]:
                            # [1, P] mean/rstd needs to broadcast differently.
                            # Use a [P, M] broadcast via: first broadcast [1, P] → [1, M*P]
                            # then reshape — but that's complex. Instead, create [P, M] by
                            # iterating: for each free pos, copy the mean value via tensor_scalar
                            # approach. Actually, use [1, P] → [P, 1] via transpose in pre_loop,
                            # then broadcast [P, 1] → [P, M] via broadcast_to.
                            _tr_p = device_fn.new_var("_nki_chain_tr_psum", dce=True)
                            _tr_s = device_fn.new_var("_nki_chain_tr_sbuf", dce=True)
                            _tr_dtype = "nl.float32" if _oe_dtype in ("nl.int32", "nl.uint32") else _oe_dtype
                            device_fn._nki_sbuf_shapes[_tr_s] = [p_count, 1]
                            device_fn._nki_sbuf_dtypes[_tr_s] = _oe_dtype
                            if _tr_dtype != _oe_dtype:
                                _cast_oe = device_fn.new_var("_nki_chain_cast", dce=True)
                                device_fn._nki_sbuf_shapes[_cast_oe] = [1, p_count]
                                device_fn._nki_sbuf_dtypes[_cast_oe] = _tr_dtype
                                pre_loop_stmts.append(statement_from_string(
                                    f"{_cast_oe} = nl.ndarray([1, {p_count}], {_tr_dtype}, buffer=nl.sbuf)"
                                ))
                                pre_loop_stmts.append(statement_from_string(
                                    f"nisa.memset({_cast_oe}, value=0)"
                                ))
                                pre_loop_stmts.append(statement_from_string(
                                    f"nisa.tensor_tensor(dst={_cast_oe}, data1={_cast_oe}, data2={_other_expr}, op=nl.add)"
                                ))
                                _oe_for_tr = _cast_oe
                            else:
                                _oe_for_tr = _other_expr
                            pre_loop_stmts.append(statement_from_string(
                                f"{_tr_p} = nl.ndarray([{p_count}, 1], {_tr_dtype}, buffer=nl.psum)"
                            ))
                            pre_loop_stmts.append(statement_from_string(
                                f"nisa.nc_transpose(dst={_tr_p}, data={_oe_for_tr})"
                            ))
                            pre_loop_stmts.append(statement_from_string(
                                f"{_tr_s} = nl.ndarray([{p_count}, 1], {_oe_dtype}, buffer=nl.sbuf)"
                            ))
                            pre_loop_stmts.append(statement_from_string(
                                f"nisa.tensor_copy(dst={_tr_s}, src={_tr_p})"
                            ))
                            # Use tensor_scalar with [P,1] operand — broadcasts over free dim
                            if _tile_is_arg0:
                                body.append(statement_from_string(
                                    f"nisa.tensor_scalar(dst={_cur_tile}, data={_cur_tile}, "
                                    f"op0={_nki_op}, operand0={_tr_s}, op1=None)"
                                ))
                            else:
                                _rev_tile2 = device_fn.new_var("_nki_chain_tile", dce=True)
                                device_fn._nki_sbuf_shapes[_rev_tile2] = [p_count, m_count]
                                device_fn._nki_sbuf_dtypes[_rev_tile2] = dtype_str
                                body.append(statement_from_string(
                                    f"{_rev_tile2} = nl.ndarray([{p_count}, {m_count}], {dtype_str}, buffer=nl.sbuf)"
                                ))
                                body.append(statement_from_string(
                                    f"nisa.tensor_copy(dst={_rev_tile2}, src={_cur_tile})"
                                ))
                                body.append(statement_from_string(
                                    f"nisa.tensor_scalar(dst={_rev_tile2}, data={_rev_tile2}, "
                                    f"op0={_nki_op}, operand0={_tr_s}, op1=None, reverse0=True)"
                                ))
                                _cur_tile = _rev_tile2
                            continue
                        body.append(statement_from_string(
                            f"{_bc_tile} = nl.broadcast_to({_other_expr}, shape=({p_count}, {m_count}))"
                        ))
                        _other_bcast = _bc_tile
                    if _tile_is_arg0:
                        body.append(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={_cur_tile}, data1={_cur_tile}, "
                                f"data2={_other_bcast}, op={_nki_op})"
                            )
                        )
                    else:
                        # tile is arg1, other is arg0 (e.g. other - tile for sub)
                        # Need a new tile for result since we can't do in-place reversed sub
                        _rev_tile = device_fn.new_var("_nki_chain_tile", dce=True)
                        device_fn._nki_sbuf_shapes[_rev_tile] = [p_count, m_count]
                        device_fn._nki_sbuf_dtypes[_rev_tile] = dtype_str
                        body.append(
                            statement_from_string(
                                f"{_rev_tile} = nl.ndarray([{p_count}, {m_count}], "
                                f"{dtype_str}, buffer=nl.sbuf)"
                            )
                        )
                        body.append(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={_rev_tile}, data1={_other_bcast}, "
                                f"data2={_cur_tile}, op={_nki_op})"
                            )
                        )
                        _cur_tile = _rev_tile
                    continue
            if _chain_terminal == "reduce":
                # Accumulate into sum: acc[p, m] += cur_tile[p, m]
                body.append(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={acc_name}, data1={acc_name}, "
                        f"data2={_cur_tile}, op=nl.add)"
                    )
                )
            else:
                # Scatter path: write the processed tile back to output tensor per k-step.
                # Pre-allocate the NKI return buffer for the output tensor so we can
                # reference it in the scatter DMA inside the gather loop.
                assert _chain_store_node is not None
                _store_tensor_arg = _chain_store_node.args[0]
                _store_out_buf: str | None = None
                _store_total_elems_scatter: int = total_elems
                if isinstance(_store_tensor_arg, torch.fx.Node):
                    _store_tensor_val = _store_tensor_arg.meta.get("val")
                    if isinstance(_store_tensor_val, torch.Tensor):
                        _store_total_elems_scatter = int(_store_tensor_val.numel())
                        _store_tensor_id = id(_store_tensor_val)
                        # Check if the return buffer is already allocated
                        _ret_bufs = device_fn._nki_return_buffers
                        if _store_tensor_id in _ret_bufs:
                            _store_out_buf = _ret_bufs[_store_tensor_id]["buf_name"]
                        else:
                            # Pre-allocate the return buffer for this output tensor
                            try:
                                from ..host_function import HostFunction as _HF
                                _hf = _HF.current()
                                if _hf is not None and _store_tensor_val in _hf.tensor_to_origin:
                                    _store_origin = _hf.tensor_to_origin[_store_tensor_val]
                                    _store_host_var = _store_origin.host_str()
                                    # Follow view chain
                                    _store_base = _store_tensor_val
                                    while getattr(_store_base, "_base", None) is not None:
                                        _store_base = _store_base._base
                                    if _store_base is not _store_tensor_val and _store_base in _hf.tensor_to_origin:
                                        _store_host_var = _hf.tensor_to_origin[_store_base].host_str()
                                    _out_buf_name = device_fn.new_var("nki_return_buf")
                                    _out_dtype_str = env.backend.dtype_str(_store_tensor_val.dtype)
                                    # For 1D output, use free-axis layout [1, N]
                                    _out_flat_extent = device_fn.new_var("nki_return_numel")
                                    device_fn.constexpr_arg(
                                        _out_flat_extent, f"{_store_host_var}.numel()"
                                    )
                                    _out_host_reshape = f"[{_store_host_var}.numel()]"
                                    device_fn.preamble.append(
                                        statement_from_string(
                                            f"{_out_buf_name} = nl.ndarray([1, {_out_flat_extent}], "
                                            f"dtype={_out_dtype_str}, buffer=nl.shared_hbm)"
                                        )
                                    )
                                    if not hasattr(device_fn, "_nki_return_buffers"):
                                        device_fn._nki_return_buffers = {}
                                    device_fn._nki_return_buffers[_store_tensor_id] = {
                                        "buf_name": _out_buf_name,
                                        "host_var": _store_host_var,
                                        "host_reshape": _out_host_reshape,
                                        "flat_extent": _out_flat_extent,
                                    }
                                    if len(device_fn._nki_return_buffers) == 1:
                                        device_fn._nki_return_buffer_name = _out_buf_name
                                        device_fn._nki_return_host_var = _store_host_var
                                        device_fn._nki_return_host_reshape = _out_host_reshape
                                    _store_out_buf = _out_buf_name
                            except Exception:
                                pass
                if _store_out_buf is None:
                    # Can't get/create output buffer; fall back
                    return None
                # For invalid positions, set scatter destination to OOB so writes
                # are skipped (preventing invalid k-positions from overwriting valid ones).
                _safe_base_u32 = base_u32
                if row_pred_col is not None:
                    _safe_offsets = device_fn.new_var("_nki_scatter_safe_offsets", dce=True)
                    device_fn._nki_sbuf_shapes[_safe_offsets] = [p_count, 1]
                    device_fn._nki_sbuf_dtypes[_safe_offsets] = "nl.uint32"
                    body.append(statement_from_string(
                        f"{_safe_offsets} = nl.ndarray([{p_count}, 1], nl.uint32, buffer=nl.sbuf)"
                    ))
                    # Use a large static OOB value; actual tensor size < 2^30 in practice
                    body.append(statement_from_string(
                        f"nisa.memset({_safe_offsets}, value=1073741824)"
                    ))
                    body.append(statement_from_string(
                        f"nisa.tensor_copy_predicated(dst={_safe_offsets}, "
                        f"src={base_u32}, predicate={row_pred_col})"
                    ))
                    _safe_base_u32 = _safe_offsets
                # Emit scatter DMA: write _cur_tile back to the output tensor at this k-step.
                body.append(
                    statement_from_string(
                        f"nisa.dma_copy("
                        f"dst={_store_out_buf}.reshape([{_out_flat_extent}, 1]).ap("
                        f"pattern=[[1, {p_count}], [1, {m_count}]], "
                        f"vector_offset={_safe_base_u32}, indirect_dim=0), "
                        f"src={_cur_tile}, "
                        f"oob_mode=nisa.oob_mode.skip)"
                    )
                )

            # Emit pre-loop statements (transposes etc.) at the top of the k-loop body.
            body = pre_loop_stmts + body

            state.codegen.add_statement(
                create(
                    ast.For,
                    target=create(ast.Name, id=k_var, ctx=ast.Store()),
                    iter=expr_from_string(f"nl.affine_range({k_count})"),
                    body=body,
                    orelse=[],
                )
            )

            # Install passthrough lowerings for all chain nodes AND the store node
            # (if any) so that their normal codegen is suppressed.
            from ..aten_lowering import Lowering as _BaseLowering

            class _PassthroughLowering(_BaseLowering):
                def __init__(self, passthrough_name: str) -> None:
                    self._passthrough_name = passthrough_name

                def codegen(self, ctx: object, node: torch.fx.Node) -> ast.AST | None:
                    return expr_from_string(self._passthrough_name)

            assert _chain_ops is not None
            for _cn in _chain_ops:
                _cn.meta["lowering"] = _PassthroughLowering(acc_name)

            if _chain_terminal == "reduce":
                if not hasattr(device_fn, "_nki_pre_reduced_loads"):
                    device_fn._nki_pre_reduced_loads = {}
                device_fn._nki_pre_reduced_loads[acc_name] = {"dim": 1}
            else:
                # Store terminal: mark the store node as already-emitted with a
                # passthrough that returns None (store codegen returns None anyway).
                assert _chain_store_node is not None

                class _NullLowering(_BaseLowering):
                    def codegen(self, ctx: object, node: torch.fx.Node) -> None:
                        return None

                _chain_store_node.meta["lowering"] = _NullLowering()

            return expr_from_string(acc_name)

        import os as _os_dbg3
        _dbg_jg = _os_dbg3.environ.get("HELION_DEBUG_JAGGED")
        flat_gather_2d = _try_emit_flat_gather_2d(fx_subscript[0])
        if flat_gather_2d is not None:
            return flat_gather_2d

        fused_gather = _try_emit_flat_gather_sum_dim1(fx_subscript[0])
        if _dbg_jg:
            print(f"[JAGGED_DBG] after sum_dim1: fused_gather={fused_gather}, subscript[0]={fx_subscript[0] if fx_subscript else None}")
        if fused_gather is not None:
            return fused_gather

        index_ast = state.codegen.ast_for_fx_node(fx_subscript[0])
        if not isinstance(index_ast, ast.AST):
            raise exc.BackendUnsupported("nki", "flat tensor gather without index AST")
        p_count = partition_dim
        index_node = fx_subscript[0]
        index_name: str | None = None
        target_name = str(getattr(index_node.target, "__name__", index_node.target))
        if (
            index_node.op == "call_function"
            and ("add.Tensor" in target_name or index_node.target is torch.ops.aten.add.Tensor)
            and len(index_node.args) >= 2
        ):
            lhs_ast = state.codegen.ast_for_fx_node(index_node.args[0])
            rhs_ast = state.codegen.ast_for_fx_node(index_node.args[1])
            if isinstance(lhs_ast, ast.AST) and isinstance(rhs_ast, ast.AST):
                lhs_name = ast.unparse(lhs_ast)
                rhs_name = ast.unparse(rhs_ast)
                index_name = device_fn.new_var("_nki_flat_index", dce=True)
                device_fn._nki_sbuf_shapes[index_name] = [1, p_count]
                device_fn._nki_sbuf_dtypes[index_name] = "nl.int32"
                state.codegen.add_statement(
                    statement_from_string(
                        f"{index_name} = nl.ndarray([1, {p_count}], nl.int32, buffer=nl.sbuf)"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={index_name}, data1={lhs_name}, "
                        f"data2={rhs_name}, op=nl.add)"
                    )
                )
        if index_name is None:
            if isinstance(index_ast, ast.BinOp) and isinstance(index_ast.op, ast.FloorDiv):
                lhs_name = ast.unparse(index_ast.left)
                lhs_shape, _lhs_dtype = _nki_lookup_sbuf_shape_dtype(state, lhs_name)
                if lhs_shape == [1, p_count]:
                    denom: float | None = None
                    denom_str = ast.unparse(index_ast.right)
                    try:
                        denom = float(denom_str)
                    except (TypeError, ValueError):
                        try:
                            denom = float(env.size_hint(denom_str))
                        except Exception:
                            denom = None
                    if denom is not None and denom != 0.0:
                        tmp_var = device_fn.new_var("_nki_flat_index_div_f32", dce=True)
                        index_name = device_fn.new_var("_nki_flat_index_div_i32", dce=True)
                        device_fn._nki_sbuf_shapes[tmp_var] = [1, p_count]
                        device_fn._nki_sbuf_shapes[index_name] = [1, p_count]
                        device_fn._nki_sbuf_dtypes[tmp_var] = "nl.float32"
                        device_fn._nki_sbuf_dtypes[index_name] = "nl.int32"
                        state.codegen.add_statement(
                            statement_from_string(
                                f"{tmp_var} = nl.ndarray([1, {p_count}], nl.float32, buffer=nl.sbuf)"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"nisa.tensor_scalar(dst={tmp_var}, data={lhs_name}, "
                                f"op0=nl.multiply, operand0={1.0 / denom})"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"{index_name} = nl.ndarray([1, {p_count}], nl.int32, buffer=nl.sbuf)"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"nisa.tensor_copy(dst={index_name}, src={tmp_var})"
                            )
                        )
            if index_name is None:
                index_name_ast = (
                    state.codegen.lift(index_ast, dce=True, prefix="_nki_flat_index")
                    if not isinstance(index_ast, ast.Name)
                    else index_ast
                )
                index_name = index_name_ast.id
        if index_name is None:
            index_name_ast = (
                state.codegen.lift(index_ast, dce=True, prefix="_nki_flat_index")
                if not isinstance(index_ast, ast.Name)
                else index_ast
            )
            index_name = index_name_ast.id
        idx_shape, idx_dtype = _nki_lookup_sbuf_shape_dtype(state, index_name)
        if idx_shape is None:
            idx_val = fx_subscript[0].meta.get("val")
            if isinstance(idx_val, torch.Tensor):
                if idx_val.ndim == 1:
                    idx_shape = [1, p_count]
                else:
                    idx_shape = [
                        (
                            int(dim)
                            if isinstance(dim, int)
                            else int(dim._sympy_().subs(_bs_subs))
                        )
                        for dim in idx_val.shape
                    ]
                idx_dtype = backend.dtype_str(idx_val.dtype)
            else:
                idx_shape = [1, p_count]
                idx_dtype = "nl.int32"
            device_fn._nki_sbuf_shapes[index_name] = idx_shape
            device_fn._nki_sbuf_dtypes[index_name] = idx_dtype
        if idx_shape is not None and len(idx_shape) == 2:
            if idx_shape[0] == 1 and idx_shape[1] != p_count:
                p_count = int(idx_shape[1])
            elif idx_shape[1] == 1 and idx_shape[0] != p_count:
                p_count = int(idx_shape[0])
        total_elems = int(tensor.numel())
        if (
            dtype_str in {"nl.int32", "nl.int16", "nl.int8", "nl.uint16", "nl.uint8"}
            and idx_shape == [1, p_count]
            and index_name.startswith("_nki_floordiv_i32")
            and p_count == 1
        ):
            scalar_index = device_fn.new_var("_nki_flat_scalar_index", dce=True)
            scalar_index_u32 = device_fn.new_var("_nki_flat_scalar_index_u32", dce=True)
            scalar_value = device_fn.new_var("_nki_flat_scalar_value", dce=True)
            device_fn._nki_sbuf_shapes[scalar_index] = [1, 1]
            device_fn._nki_sbuf_shapes[scalar_index_u32] = [1, 1]
            device_fn._nki_sbuf_shapes[scalar_value] = [1, 1]
            device_fn._nki_sbuf_shapes[sbuf_name] = [1, p_count]
            device_fn._nki_sbuf_dtypes[scalar_index] = idx_dtype
            device_fn._nki_sbuf_dtypes[scalar_index_u32] = "nl.uint32"
            device_fn._nki_sbuf_dtypes[scalar_value] = dtype_str
            device_fn._nki_sbuf_dtypes[sbuf_name] = dtype_str
            state.codegen.add_statement(
                statement_from_string(
                    f"{scalar_index} = nl.ndarray([1, 1], {idx_dtype}, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy(dst={scalar_index}, src={index_name}[0:1, 0:1])"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"{scalar_index_u32} = nl.ndarray([1, 1], nl.uint32, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy(dst={scalar_index_u32}, src={scalar_index})"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"{scalar_value} = nl.ndarray([1, 1], {dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={scalar_value}, "
                    f"src={name}.reshape([{total_elems}, 1]).ap("
                    f"pattern=[[1, 1], [1, 1]], "
                    f"scalar_offset={scalar_index_u32}, indirect_dim=0))"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"{sbuf_name} = nl.broadcast_to({scalar_value}, "
                    f"shape=(1, {p_count}))"
                )
            )
            return expr_from_string(sbuf_name)
        vec_offset = _nki_as_uint32_p1_vector(state, index_name, p_count)
        if vec_offset is None:
            raise exc.BackendUnsupported(
                "nki",
                "flat tensor gather index layout "
                f"(index={index_name}, shape={idx_shape}, p_count={p_count}, "
                f"dtype={idx_dtype}, "
                f"ast={ast.dump(index_ast)})",
            )
        gather_tmp = device_fn.new_var("_nki_flat_gather", dce=True)
        tr_psum = device_fn.new_var("_nki_flat_gather_tr_psum", dce=True)
        transpose_dtype = (
            "nl.float32"
            if dtype_str
            in {"nl.int32", "nl.int16", "nl.int8", "nl.uint32", "nl.uint16", "nl.uint8"}
            else dtype_str
        )
        if transpose_dtype == "nl.float32" and dtype_str != "nl.float32":
            device_fn._nki_host_arg_casts[name] = "torch.float32"
        device_fn._nki_sbuf_shapes[gather_tmp] = [p_count, 1]
        device_fn._nki_sbuf_dtypes[gather_tmp] = transpose_dtype
        device_fn._nki_sbuf_shapes[sbuf_name] = [1, p_count]
        device_fn._nki_sbuf_dtypes[sbuf_name] = dtype_str
        state.codegen.add_statement(
            statement_from_string(
                f"{gather_tmp} = nl.ndarray([{p_count}, 1], "
                f"{transpose_dtype}, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.dma_copy(dst={gather_tmp}, "
                f"src={name}.reshape([{total_elems}, 1]).ap("
                f"pattern=[[1, {p_count}], [1, 1]], "
                f"vector_offset={vec_offset}, indirect_dim=0), "
                "oob_mode=nisa.oob_mode.skip)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(
                f"{tr_psum} = nl.ndarray([1, {p_count}], "
                f"{transpose_dtype}, buffer=nl.psum)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.nc_transpose(dst={tr_psum}, data={gather_tmp})")
        )
        state.codegen.add_statement(
            statement_from_string(
                f"{sbuf_name} = nl.ndarray([1, {p_count}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.tensor_copy(dst={sbuf_name}, src={tr_psum})")
        )
        return expr_from_string(sbuf_name)

    # 3D+ tensors: reshaped to 2D at kernel entry ([B,M,K] → [B*M, K]).
    # Combine the leading slice_parts into one flattened partition slice
    # and squeeze output_shape to 2D.
    if tensor.dim() > 2 and len(slice_parts) > 2:
        # Combine all leading slice_parts (except the last) into one flat slice.
        # Each leading slice is "offset:offset+block_size".  For the flattened
        # tensor the row index is: batch_off * dim1_size + dim1_off (+ ...).
        # Extract (offset, block_size) pairs from leading slice_parts.
        leading_offsets: list[str] = []
        leading_block_sizes: list[int] = []
        original_dim_sizes: list[int] = []
        for dim_i in range(tensor.dim() - 1):
            sp = slice_parts[dim_i]
            if ":" in sp:
                off_str, end_str = sp.split(":")
                off_str = off_str.strip()
                leading_offsets.append(off_str)
                # Try to extract numeric block size
                # end_str is like "offset_0+128" or "expr_a + expr_b+128"
                # Use rfind to handle compound expressions: "a + b+128" → last '+' before 128
                plus_idx = end_str.rfind("+")
                if plus_idx >= 0:
                    bs_str = end_str[plus_idx + 1:].strip()
                    try:
                        leading_block_sizes.append(int(bs_str))
                    except ValueError:
                        leading_block_sizes.append(1)
                else:
                    # No "+": format is "0:size" — block_size is the end value
                    end_str = end_str.strip()
                    try:
                        end_val = int(end_str)
                        start_val = int(off_str) if off_str.strip().isdigit() else 0
                        leading_block_sizes.append(end_val - start_val)
                    except (ValueError, TypeError):
                        leading_block_sizes.append(1)
            else:
                # Scalar index (block_size=1)
                leading_offsets.append(sp.strip())
                leading_block_sizes.append(1)
            original_dim_sizes.append(
                int(tensor.size(dim_i)) if isinstance(tensor.size(dim_i), int)
                else _resolve_dim(tensor.size(dim_i))
            )

        # Build flat offset: off0 * size1 * size2 * ... + off1 * size2 * ... + offN
        # and flat block_size: product of all leading block sizes
        flat_offset_parts: list[str] = []
        for j, off in enumerate(leading_offsets):
            multiplier_parts = [str(original_dim_sizes[k]) for k in range(j + 1, len(original_dim_sizes))]
            if multiplier_parts:
                multiplier = " * ".join(multiplier_parts)
                # Parenthesize compound offset expressions to ensure correct precedence.
                # e.g. "offset_2 + mul_chunk" * 2 must become "(offset_2 + mul_chunk) * 2"
                # not "offset_2 + mul_chunk * 2" (wrong due to operator precedence).
                off_expr = f"({off})" if ("+" in off or "-" in off) else off
                flat_offset_parts.append(f"{off_expr} * {multiplier}")
            else:
                flat_offset_parts.append(off)
        flat_offset = " + ".join(flat_offset_parts)
        flat_block_size = 1
        for bs in leading_block_sizes:
            flat_block_size *= bs

        # Strided-gather detection: one tiled dim followed by scalar dim(s) means
        # rows are non-contiguous in the flattened layout.  Use ap(vector_offset=)
        # instead of a consecutive DMA slice.
        # Example: w[seqlen_tile=32, head_scalar=1, :] on [S, H, D]:
        #   flat rows needed: off, off+H, off+2H, ...  NOT off, off+1, off+2, ...
        _tile_dim_idx_ld = None
        for _di, _bs in enumerate(leading_block_sizes):
            if _bs > 1:
                if _tile_dim_idx_ld is None:
                    _tile_dim_idx_ld = _di
                else:
                    _tile_dim_idx_ld = None  # multiple tiled dims — not simple stride
                    break
        _use_strided_gather = (
            _tile_dim_idx_ld is not None
            and _tile_dim_idx_ld < len(leading_block_sizes) - 1
            and all(leading_block_sizes[_di] == 1
                    for _di in range(_tile_dim_idx_ld + 1, len(leading_block_sizes)))
            and not any(isinstance(p, (IndirectAP, DynamicAP))
                        for p in slice_parts)
        )
        if _use_strided_gather:
            _tile_bs_ld = leading_block_sizes[_tile_dim_idx_ld]
            _tile_off_ld = leading_offsets[_tile_dim_idx_ld]
            _stride_ld = 1
            for _di in range(_tile_dim_idx_ld + 1, len(original_dim_sizes)):
                _stride_ld *= original_dim_sizes[_di]
            if _stride_ld > 1:
                _scalar_parts_ld: list[str] = []
                for _di in range(_tile_dim_idx_ld + 1, len(leading_offsets)):
                    _m = [str(original_dim_sizes[_k])
                          for _k in range(_di + 1, len(original_dim_sizes))]
                    _off_e = leading_offsets[_di]
                    if _m:
                        _off_p = f"({_off_e})" if ("+" in _off_e or "-" in _off_e) else _off_e
                        _scalar_parts_ld.append(f"{_off_p} * {' * '.join(_m)}")
                    else:
                        _scalar_parts_ld.append(_off_e)
                _prefix_parts_ld: list[str] = []
                for _di in range(0, _tile_dim_idx_ld):
                    _m = [str(original_dim_sizes[_k])
                          for _k in range(_di + 1, len(original_dim_sizes))]
                    _off_e = leading_offsets[_di]
                    if _m:
                        _off_p = f"({_off_e})" if ("+" in _off_e or "-" in _off_e) else _off_e
                        _prefix_parts_ld.append(f"{_off_p} * {' * '.join(_m)}")
                    else:
                        _prefix_parts_ld.append(_off_e)
                _scalar_off_ld = " + ".join(_scalar_parts_ld) if _scalar_parts_ld else "0"
                _prefix_off_ld = " + ".join(_prefix_parts_ld) if _prefix_parts_ld else None

                from ..ast_extension import statement_from_string as _sfs_sg
                _iota_ld = device_fn.new_var("_sg_iota", dce=True)
                device_fn._nki_sbuf_shapes[_iota_ld] = [1, _tile_bs_ld]
                device_fn._nki_sbuf_dtypes[_iota_ld] = "nl.int32"
                state.codegen.add_statement(_sfs_sg(
                    f"{_iota_ld} = nl.ndarray([1, {_tile_bs_ld}], nl.int32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(_sfs_sg(
                    f"nisa.iota(dst={_iota_ld}, pattern=[[1, {_tile_bs_ld}]], "
                    f"offset={_tile_off_ld}, channel_multiplier=0)"
                ))
                _scaled_ld = device_fn.new_var("_sg_scaled", dce=True)
                device_fn._nki_sbuf_shapes[_scaled_ld] = [1, _tile_bs_ld]
                device_fn._nki_sbuf_dtypes[_scaled_ld] = "nl.int32"
                state.codegen.add_statement(_sfs_sg(
                    f"{_scaled_ld} = nl.ndarray([1, {_tile_bs_ld}], nl.int32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(_sfs_sg(
                    f"nisa.tensor_scalar(dst={_scaled_ld}, data={_iota_ld}, "
                    f"op0=nl.multiply, operand0={_stride_ld}, op1=None)"
                ))
                _row_ld = device_fn.new_var("_sg_row", dce=True)
                device_fn._nki_sbuf_shapes[_row_ld] = [1, _tile_bs_ld]
                device_fn._nki_sbuf_dtypes[_row_ld] = "nl.int32"
                state.codegen.add_statement(_sfs_sg(
                    f"{_row_ld} = nl.ndarray([1, {_tile_bs_ld}], nl.int32, buffer=nl.sbuf)"
                ))
                if _scalar_off_ld == "0":
                    state.codegen.add_statement(_sfs_sg(
                        f"nisa.tensor_copy(dst={_row_ld}, src={_scaled_ld})"
                    ))
                else:
                    state.codegen.add_statement(_sfs_sg(
                        f"nisa.tensor_scalar(dst={_row_ld}, data={_scaled_ld}, "
                        f"op0=nl.add, operand0={_scalar_off_ld}, op1=None)"
                    ))
                if _prefix_off_ld is not None:
                    state.codegen.add_statement(_sfs_sg(
                        f"nisa.tensor_scalar(dst={_row_ld}, data={_row_ld}, "
                        f"op0=nl.add, operand0={_prefix_off_ld}, op1=None)"
                    ))
                _vec_ld = _nki_as_uint32_p1_vector(state, _row_ld, _tile_bs_ld)
                if _vec_ld is not None:
                    _flat_hbm_ld = 1
                    for _ds in original_dim_sizes:
                        _flat_hbm_ld *= _ds
                    _d_int_ld = _resolve_dim(tensor.size(tensor.dim() - 1))
                    slice_parts = [IndirectAP(vec_var=_vec_ld, p_count=_tile_bs_ld, pattern=None), f"0:{_d_int_ld}"]
                    is_scalar_dim = [False, False]
                    partition_dim = _tile_bs_ld
                    free_dims = [_d_int_ld]
                    output_shape = [partition_dim] + free_dims
                    hbm_dim_size_strs = [str(_flat_hbm_ld), str(_d_int_ld)]
                    partition_offset_var = None
                    _use_strided_gather = False  # signal: flat_slice path below skipped

        if _use_strided_gather is False and isinstance(slice_parts[0], IndirectAP):
            # Strided gather succeeded — skip the consecutive flat slice
            pass
        else:
            # Replace leading slice_parts with one combined slice
            flat_slice = f"({flat_offset}):({flat_offset}) + {flat_block_size}"
            slice_parts = [flat_slice] + [slice_parts[-1]]

            # Fix partition_offset_var to point to the flat offset expression
            partition_offset_var = f"({flat_offset})"

            # Squeeze output_shape to 2D.
            partition_dim = flat_block_size
            free_dims = [_resolve_dim(output_shape[-1])]
            output_shape = [partition_dim] + free_dims

        if not isinstance(slice_parts[0], IndirectAP):
            flat_hbm_partition = 1
            for dim_size in original_dim_sizes:
                flat_hbm_partition *= dim_size
            hbm_dim_size_strs = [
                str(flat_hbm_partition),
                _tensor_dim_size_str(tensor.dim() - 1),
            ]

    def _build_hbm_src(name_str: str, parts: list[str]) -> str:
        """Build an HBM src expression, converting __DYN_AP__counter__size
        sentinels to .ap(pattern=..., scalar_offset=counter, indirect_dim=N)
        when any dim is dynamic.
        Also handles __AP_VEC_OFFSET__var__pattern__ for indirect gather.
        """
        for _p in parts:
            if isinstance(_p, IndirectAP) and _p.pattern is None:
                vec_offset = _p.vec_var
                p_count = _p.p_count
                # Use hbm_dim_size_strs rather than tensor.shape so the 3D
                # early-exit path (which sets hbm_dim_size_strs to [L*H, D])
                # works correctly even though the original tensor is 3D.
                tensor_shape = list(hbm_dim_size_strs)
                if len(tensor_shape) != 2:
                    raise NotImplementedError(
                        f"Row gather requires a 2D tensor, got {list(tensor.shape)}"
                    )
                if isinstance(tensor_shape[1], int):
                    f_total: int | str = int(tensor_shape[1])
                elif isinstance(tensor_shape[1], torch.SymInt):
                    try:
                        f_total = _resolve_dim(tensor_shape[1])
                    except (TypeError, ValueError):
                        f_total = state.sympy_expr(tensor_shape[1]._sympy_())
                else:
                    f_total = str(tensor_shape[1])
                free_part = parts[1] if len(parts) > 1 else f"0:{f_total}"
                if ":" in free_part:
                    f_start, f_end = free_part.split(":", 1)
                    f_start = f_start.strip()
                    f_end = f_end.strip()
                    plus_idx = f_end.find("+")
                    if plus_idx >= 0:
                        f_count_str = f_end[plus_idx + 1 :].strip()
                        try:
                            f_count = int(f_count_str)
                        except ValueError:
                            f_count = free_dims[0] if free_dims else f_total
                    else:
                        try:
                            f_count = int(f_end) - int(f_start)
                        except (TypeError, ValueError):
                            f_count = free_dims[0] if free_dims else f_total
                else:
                    f_start = "0"
                    f_count = free_dims[0] if free_dims else f_total
                # Reshape tensor to [total_elements, 1] and use flat indices.
                # vector_offset contains row indices; compute flat offsets:
                # flat_offset[p] = vector_offset[p] * f_total + f_start
                # Then use pattern [[1, P], [1, F_count]].
                _total_elems = "*".join(hbm_dim_size_strs) if hbm_dim_size_strs else str(f_total)
                _f_start_int = 0
                _f_start_is_dynamic = False
                try:
                    _f_start_int = int(f_start)
                except (ValueError, TypeError):
                    _f_start_is_dynamic = (f_start != "0")
                # Reshape tensor to [total_elements, 1] and compute flat indices.
                # vector_offset contains row indices; flat_offset = row * f_total + f_start
                _flat_vec = device_fn.new_var("_ig_flat_vec", dce=True)
                device_fn._nki_sbuf_shapes[_flat_vec] = [p_count, 1]
                device_fn._nki_sbuf_dtypes[_flat_vec] = "nl.uint32"
                from ..ast_extension import statement_from_string as _sfs_ig
                state.codegen.add_statement(_sfs_ig(
                    f"{_flat_vec} = nl.ndarray([{p_count}, 1], nl.uint32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(_sfs_ig(
                    f"nisa.tensor_scalar(dst={_flat_vec}, data={vec_offset}, op0=nl.multiply, operand0={f_total}, op1=None)"
                ))
                if _f_start_int != 0 or _f_start_is_dynamic:
                    state.codegen.add_statement(_sfs_ig(
                        f"nisa.tensor_scalar(dst={_flat_vec}, data={_flat_vec}, op0=nl.add, operand0={f_start}, op1=None)"
                    ))
                pattern = f"[[1, {p_count}], [1, {f_count}]]"
                return (
                    f"{name_str}.reshape([{_total_elems}, 1]).ap(pattern={pattern}, "
                    f"vector_offset={_flat_vec}, indirect_dim=0)"
                )

        # Handle vector_offset gather first
        for _p in parts:
            if isinstance(_p, IndirectAP) and _p.pattern is not None:
                return f"{name_str}.ap(pattern={_p.pattern}, vector_offset={_p.vec_var}, indirect_dim=0)"
        has_dyn = any(isinstance(p, DynamicAP) for p in parts)
        if not has_dyn:
            return f"{name_str}[{', '.join(parts)}]"
        # Compute strides and pattern for .ap(). Use tensor.shape to get
        # full strides.
        # For each dim: if dynamic, use scalar_offset; else use static slice.
        # NKI .ap accepts a pattern [[stride, count], ...] and a single
        # scalar_offset + indirect_dim.
        # We only support one dynamic dim for now.
        dyn_dim_idx = None
        dyn_counter = None
        dyn_size = 0
        for i, p in enumerate(parts):
            if isinstance(p, DynamicAP):
                assert dyn_dim_idx is None, "multiple dynamic dims not yet supported"
                dyn_dim_idx = i
                dyn_counter = p.counter
                dyn_size = p.block_size
        # For .ap() we need the pattern as [[stride_for_each_axis, count_for_each_axis], ...]
        # Tensor shape dims beyond the slice are unused, but we need strides for
        # multi-dim source. Assume 2D source [P, F]:
        # If dyn dim is 0 (partition): pattern=[[F, dyn_size], [1, static_free_count]]
        # If dyn dim is 1 (free): pattern=[[F_total, P_static_count], [1, dyn_size]]
        # Get tensor shape
        tensor_shape = list(tensor.shape)
        # Flatten 3D+ shapes to 2D (Helion does this at kernel entry)
        if len(tensor_shape) > 2:
            while len(tensor_shape) > 2 and tensor_shape[0] == 1:
                tensor_shape = tensor_shape[1:]
            if len(tensor_shape) > 2:
                _flat = 1
                for d in tensor_shape[:-1]:
                    _d_i = int(d) if isinstance(d, int) else _resolve_dim(d)
                    _flat *= _d_i
                tensor_shape = [_flat, tensor_shape[-1]]
        # Resolve symints to ints
        tensor_shape = [int(d) if isinstance(d, int) else _resolve_dim(d) for d in tensor_shape]
        # Build pattern based on where dyn_dim_idx is
        if len(tensor_shape) == 2:
            P_total, F_total = tensor_shape
            # Use padded stride when reading from the padded return buffer.
            # Check both the committed name and any pre-reserved name.
            _ret_buf_name = getattr(device_fn, "_nki_return_buffer_name", None)
            _padded_f = getattr(device_fn, "_nki_return_buf_free_dim", None)
            if _padded_f is None:
                # May not be set yet if the store hasn't fired; check dyn_loops.
                for _dli_r in getattr(device_fn, "_nki_dyn_loops", {}).values():
                    _pb_r = _dli_r.get("pre_reserved_buf") or _dli_r.get("prefill_buf")
                    if _pb_r and _pb_r == name_str:
                        # The padded size was recorded alongside the step.
                        _step_r = _dli_r.get("step", 1)
                        if _step_r > 1:
                            try:
                                _f_r = int(tensor_shape[1])
                                if _f_r % _step_r != 0:
                                    _padded_f = ((_f_r + _step_r - 1) // _step_r) * _step_r
                            except (ValueError, TypeError):
                                pass
                        break
            if _padded_f is not None and name_str in (
                _ret_buf_name,
                *[_d.get("pre_reserved_buf", "") for _d in getattr(device_fn, "_nki_dyn_loops", {}).values()],
            ):
                F_total = _padded_f
            if dyn_dim_idx == 0:
                # partition is dynamic, free is static from parts[1]
                # parts[1] is like "0:F" or "offset_0:offset_0+128"
                free_part = parts[1]
                if ":" in free_part:
                    f_start, f_end = free_part.split(":", 1)
                    f_start = f_start.strip()
                    f_end = f_end.strip()
                    # Try to compute count
                    _plus = f_end.find("+")
                    if _plus >= 0:
                        _f_count_str = f_end[_plus+1:].strip()
                        try:
                            f_count = int(_f_count_str)
                        except ValueError:
                            f_count = 1
                    else:
                        try:
                            f_count = int(f_end) - int(f_start)
                        except (ValueError, TypeError):
                            f_count = 1
                    # Use scalar_offset for partition (indirect_dim=0)
                    # free offset goes into pattern's base offset (or 'offset' param)
                    pattern = f"[[{F_total}, {dyn_size}], [1, {f_count}]]"
                    # free slice start
                    try:
                        f_off_int = int(f_start)
                        return (
                            f"{name_str}.ap(pattern={pattern}, "
                            f"scalar_offset={dyn_counter}, indirect_dim=0, "
                            f"offset={f_off_int})"
                        )
                    except ValueError:
                        return (
                            f"{name_str}.ap(pattern={pattern}, "
                            f"scalar_offset={dyn_counter}, indirect_dim=0)"
                        )
            elif dyn_dim_idx == 1:
                # free is dynamic; partition is static from parts[0]
                part_part = parts[0]
                if ":" in part_part:
                    p_start, p_end = part_part.split(":", 1)
                    p_start = p_start.strip()
                    p_end = p_end.strip()
                    _plus = p_end.find("+")
                    if _plus >= 0:
                        _p_count_str = p_end[_plus+1:].strip()
                        try:
                            p_count = int(_p_count_str)
                        except ValueError:
                            p_count = 1
                    else:
                        try:
                            p_count = int(p_end) - int(p_start)
                        except (ValueError, TypeError):
                            p_count = 1
                    # partition-offset goes into pattern's stride×count
                    # For pattern [[F, P_count], [1, dyn_size]], scalar_offset
                    # indexes the last dim (free).
                    pattern = f"[[{F_total}, {p_count}], [1, {dyn_size}]]"
                    try:
                        p_off_int = int(p_start)
                        # p_off_int is start partition row
                        return (
                            f"{name_str}.ap(pattern={pattern}, "
                            f"scalar_offset={dyn_counter}, indirect_dim=1, "
                            f"offset={p_off_int * F_total})"
                        )
                    except ValueError:
                        # The partition offset can be an affine loop var.
                        # Keep it in AP's flattened base offset.
                        return (
                            f"{name_str}.ap(pattern={pattern}, "
                            f"scalar_offset={dyn_counter}, indirect_dim=1, "
                            f"offset=({p_start}) * {F_total})"
                        )
        # Fallback: error clearly
        raise NotImplementedError(
            f"Dynamic DMA slice not supported for shape {tensor_shape} parts {parts}"
        )

    def _slice_bounds_guard(parts: list[str]) -> str | None:
        """Return a Python guard that makes every static HBM slice in-bounds."""
        import re as _re_bounds
        checks: list[str] = []
        for dim_idx, part in enumerate(parts):
            if dim_idx >= len(hbm_dim_size_strs):
                break
            if isinstance(part, (IndirectAP, DynamicAP)):
                continue
            if ":" not in part:
                continue
            start, end = part.split(":", 1)
            start = start.strip()
            end = end.strip()
            if not start or not end:
                continue
            dim_size_str = hbm_dim_size_strs[dim_idx]
            # Skip >= 0 check for offsets from dynamic-begin loops (the begin
            # register is non-negative by construction, so the check is vacuous
            # and NKI can't evaluate >= on a VirtualRegister).
            _dyn_begin_vars = getattr(
                getattr(state, "device_function", None), "_nki_dyn_begin_offset_vars", set()
            )
            _start_has_dyn_begin = any(v in start for v in _dyn_begin_vars)
            if _start_has_dyn_begin:
                # dynamic_range guarantees start >= 0 and stays within bounds;
                # skip all guards for this dimension (VirtualRegister arithmetic fails).
                continue
            checks.append(f"({start}) >= 0")
            # For unit-block slices "start:start + 1" use "start < dim_size"
            # instead of "(start + 1) <= dim_size".  When 'start' contains a
            # NKI affine loop variable (e.g. "1 + offset_0") NKI's symbolic
            # tracer cannot evaluate "(start + 1) <= dim" and errors with
            # "'add' expected (int, int) got (object, int)".  The two forms
            # are logically identical for integers.
            if end in (f"{start} + 1", f"({start}) + 1"):
                checks.append(f"({start}) < {dim_size_str}")
            else:
                checks.append(f"({end}) <= {dim_size_str}")
            # Additional inner-dimension bound: when start is "A * stride + tile_offset"
            # (scalar * constant + loop_var), add "tile_offset + block <= stride" so
            # the tile doesn't overflow into the next scalar block. This prevents
            # 3D-flattened 2D accesses (e.g. W[e*K + k_tile]) from reading across
            # expert boundaries when K is not a multiple of the block size.
            # Pattern: start = "X * stride + offset_N" or "X + offset_N"
            _inner_m = _re_bounds.match(
                r'^(.+\*\s*(\d+))\s*\+\s*(offset_\d+)$', start
            )
            if not _inner_m:
                # Try stripping outer parentheses
                _start_stripped = start.strip('()')
                _inner_m = _re_bounds.match(
                    r'^(.+\*\s*(\d+))\s*\+\s*(offset_\d+)$', _start_stripped
                )
            if _inner_m:
                _stride_str = _inner_m.group(2)
                _tile_offset = _inner_m.group(3)
                # end contains "start + block_size"; extract block_size
                _end_m = _re_bounds.match(
                    r'^(.+)\s*\+\s*(\d+)$', end
                )
                if _end_m:
                    _block_size = _end_m.group(2)
                    # Only add the inner-bound check if the block_size matches
                    # the actual block_size of _tile_offset in the active loops.
                    # This prevents spurious checks like "offset_head + 128 <= 4"
                    # when the match is a multi-term expression like
                    # "chunk * nheads + offset_head" (stride=nheads, not K).
                    _tile_offset_bs = None
                    for _bid_check in state.codegen.active_device_loops:
                        if state.codegen.offset_var(_bid_check) == _tile_offset:
                            try:
                                _tile_offset_bs = int(
                                    env.block_sizes[_bid_check].from_config_assert(state.config)
                                )
                            except Exception:
                                pass
                            break
                    # Only emit the inner bound if the block_size matches or if
                    # we couldn't verify (conservative: skip to avoid false guards).
                    if _tile_offset_bs is not None and _tile_offset_bs == int(_block_size):
                        checks.append(f"({_tile_offset}) + {_block_size} <= {_stride_str}")
        if not checks:
            return None
        return " and ".join(checks)

    def _slice_info(part: str, dim_idx: int) -> tuple[str, str, int, str] | None:
        import re as _re_si

        if isinstance(part, (IndirectAP, DynamicAP)):
            return None
        if dim_idx >= len(hbm_dim_size_strs) or ":" not in part:
            return None
        start, end = part.split(":", 1)
        start = start.strip()
        end = end.strip()
        if not start or not end:
            return None
        count: int | None = None

        # First, try "X ± C1 : X ± C2" pattern where start and end share the
        # same base expression. The block size is C2 - C1.  This handles both
        # negative shifts ("offset_1 - 400:offset_1 - 144") and folded positive
        # shifts ("offset_0 + 1:offset_0 + 2").  This takes priority over the
        # simpler "+N suffix" extraction to avoid confusion when start is also
        # "expr + C".
        _m_start = _re_si.match(r'^(.+?)\s*([+-])\s*(\d+)$', start)
        _m_end = _re_si.match(r'^(.+?)\s*([+-])\s*(\d+)$', end)
        if _m_start and _m_end and _m_start.group(1).strip() == _m_end.group(1).strip():
            _s_sign = 1 if _m_start.group(2) == '+' else -1
            _e_sign = 1 if _m_end.group(2) == '+' else -1
            try:
                _s_const = _s_sign * int(_m_start.group(3))
                _e_const = _e_sign * int(_m_end.group(3))
                _candidate = _e_const - _s_const
                if _candidate > 0:
                    count = _candidate
            except ValueError:
                pass

        # Fallback: try "...+ N" at the end of `end` (for plain "offset:offset+N" slices).
        if count is None:
            plus_idx = end.rfind("+")
            if plus_idx >= 0:
                try:
                    count = int(end[plus_idx + 1 :].strip())
                except ValueError:
                    count = None

        if count is None:
            try:
                count = int(end) - int(start)
            except (TypeError, ValueError):
                count = None
        if count is None or count <= 0:
            return None
        dim_size_str = hbm_dim_size_strs[dim_idx]
        return start, end, count, dim_size_str

    def _single_tail_load_cases(dst: str, hbm_base: str, parts: list[str]) -> list[ast.If]:
        # Generate DMA cases for tiles that partially overlap HBM tensor boundaries.
        # Enumerates all 3^N combinations of per-dimension states (FULL / NEG_START /
        # TAIL_OVERFLOW), skipping the all-FULL case (handled by the main fast path).
        # This covers corner cases where multiple dimensions are partial simultaneously,
        # e.g. the last row-tile overlapping a shifted-subscript column boundary.
        import itertools

        _dyn_begin_vars_tl = getattr(
            getattr(state, "device_function", None), "_nki_dyn_begin_offset_vars", set()
        )
        cases: list[ast.If] = []
        infos = [_slice_info(part, i) for i, part in enumerate(parts)]
        if any(info is None for info in infos):
            return cases

        FULL, NEG_START, TAIL_OVERFLOW = "full", "neg_start", "tail_overflow"

        for dim_states in itertools.product([FULL, NEG_START, TAIL_OVERFLOW], repeat=len(infos)):
            if all(s == FULL for s in dim_states):
                continue  # handled by main fast path

            checks: list[str] = []
            src_parts: list[str] = []
            dst_parts: list[str] = []
            valid = True

            for dim_idx, (info, dim_state) in enumerate(zip(infos, dim_states)):
                assert info is not None
                dim_start, dim_end, dim_count, dim_size = info
                _has_dyn_begin = any(v in dim_start for v in _dyn_begin_vars_tl)

                if _has_dyn_begin:
                    # dynamic_range guarantees offset in [begin, end); NEG_START and
                    # TAIL_OVERFLOW are impossible. FULL needs no guards.
                    if dim_state != FULL:
                        valid = False
                        break
                    src_parts.append(parts[dim_idx])
                    dst_parts.append(f"0:{dim_count}")
                elif dim_state == FULL:
                    checks.append(f"({dim_start}) >= 0")
                    checks.append(f"({dim_end}) <= {dim_size}")
                    src_parts.append(parts[dim_idx])
                    dst_parts.append(f"0:{dim_count}")
                elif dim_state == NEG_START:
                    # Tile starts before the tensor; load [0:end] -> dst[(count-end):count]
                    checks.append(f"({dim_start}) < 0")
                    checks.append(f"({dim_end}) > 0")
                    src_parts.append(f"0:{dim_end}")
                    dst_parts.append(f"({dim_count}) - ({dim_end}):({dim_count})")
                else:  # TAIL_OVERFLOW
                    # Tile overflows past the tensor end; load [start:size] -> dst[0:(size-start)]
                    checks.append(f"({dim_start}) >= 0")
                    checks.append(f"({dim_start}) < {dim_size}")
                    checks.append(f"({dim_end}) > {dim_size}")
                    src_parts.append(f"{dim_start}:{dim_size}")
                    dst_parts.append(f"0:{dim_size} - ({dim_start})")

            if valid:
                cases.append(
                    create(
                        ast.If,
                        test=expr_from_string(" and ".join(checks)),
                        body=[
                            statement_from_string(
                                f"nisa.dma_copy(dst={dst}[{', '.join(dst_parts)}], "
                                f"src={hbm_base}[{', '.join(src_parts)}])"
                            )
                        ],
                        orelse=[],
                    )
                )
        return cases

    def _emit_dma_copy(
        dst: str,
        src: str,
        parts: list[str] | None = None,
        hbm_base: str | None = None,
    ) -> None:
        # EXPERIMENTAL (HELION_NKI_TILESTREAM): replace the fast-path DMA + the
        # 3^N boundary-tail enumeration with ONE TensorView-clamped DMA. The SBUF
        # dst is already memset(0), so a partial tile fills only its valid
        # sub-rectangle and the remainder stays zero — matching legacy semantics
        # without any guard/branch explosion. Only fires for the clean contiguous
        # 2D case (hbm_base set, every part a plain "start:end" string); anything
        # else (indirect/dynamic/1D-reshape) falls through to the legacy path.
        if (
            backend.use_tilestream
            and hbm_base is not None
            and parts is not None
            and len(parts) >= 1
            and all(isinstance(p, str) and ":" in p for p in parts)
        ):
            _tv_expr = f"_nkitv({hbm_base})"
            for _d, _p in enumerate(parts):
                _s, _e = _p.split(":", 1)
                _tv_expr += f".slice({_d}, {_s.strip()}, {_e.strip()})"
            _srcv = device_fn.new_var("_ts_srcv")
            _dst_idx = ", ".join(f"0:{_srcv}.shape[{_d}]" for _d in range(len(parts)))
            state.codegen.add_statement(statement_from_string(f"{_srcv} = {_tv_expr}"))
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={dst}[{_dst_idx}], src={_srcv}.get_view())"
                )
            )
            return
        oob_arg = (
            ", oob_mode=nisa.oob_mode.skip"
            if (
                "vector_offset=" in src
                or "vector_offset=" in dst
                or "scalar_offset=" in src
                or "scalar_offset=" in dst
            )
            else ""
        )
        stmt = statement_from_string(f"nisa.dma_copy(dst={dst}, src={src}{oob_arg})")
        guard = _slice_bounds_guard(parts) if parts else None
        if guard is not None:
            import re as _re_guard
            tail_cases = (
                _single_tail_load_cases(dst, hbm_base, parts)
                if hbm_base is not None and parts is not None
                else []
            )
            # Also add inner-bound partial DMA cases for 3D-flattened accesses.
            # When the partition start is "scalar * stride + tile_offset" and the
            # tile_offset overflows the inner stride (K per expert), generate a
            # partial load that only reads the valid elements.
            if hbm_base is not None and parts is not None and len(parts) >= 1:
                _part0 = parts[0]
                if isinstance(_part0, str) and ":" in _part0:
                    _p0_start, _p0_end = _part0.split(":", 1)
                    _p0_start = _p0_start.strip()
                    # Strip exactly one outer paren pair if balanced, e.g. "(expr)" → "expr"
                    # Use explicit check rather than .strip("()") which over-strips "((x)..."
                    if _p0_start.startswith("(") and _p0_start.endswith(")"):
                        _inner = _p0_start[1:-1]
                        # Verify the stripped parens were a balanced outer pair
                        depth = 0
                        _balanced = True
                        for _ch in _inner:
                            if _ch == "(":
                                depth += 1
                            elif _ch == ")":
                                if depth == 0:
                                    _balanced = False
                                    break
                                depth -= 1
                        if _balanced and depth == 0:
                            _p0_start = _inner
                    _inner_m2 = _re_guard.match(r'^(.+\*\s*(\d+))\s*\+\s*(offset_\d+)$', _p0_start)
                    if _inner_m2:
                        _stride2 = _inner_m2.group(2)
                        _tile_var2 = _inner_m2.group(3)
                        _p0_end2 = _p0_end.strip()
                        _block_m2 = _re_guard.match(r'^.+\+\s*(\d+)$', _p0_end2)
                        if _block_m2:
                            _blk2 = _block_m2.group(1)
                            # Partial case: tile_var overflows inner stride
                            # Condition: tile_var >= 0 and tile_var < stride and tile_var + block > stride
                            _inner_overflow_cond = (
                                f"({_tile_var2}) >= 0 and ({_tile_var2}) < {_stride2} "
                                f"and ({_tile_var2}) + {_blk2} > {_stride2}"
                            )
                            # Also check outer bounds still valid and all other
                            # dims are fully in-bounds (no multi-tail scenario)
                            if len(hbm_dim_size_strs) > 0:
                                _inner_overflow_cond += f" and ({_p0_start}) >= 0 and ({_p0_start}) + {_stride2} - ({_tile_var2}) <= {hbm_dim_size_strs[0]}"
                            for _other_i, _other_p in enumerate(parts[1:], 1):
                                if ":" in _other_p and _other_i < len(hbm_dim_size_strs):
                                    _os, _oe = _other_p.split(":", 1)
                                    _inner_overflow_cond += f" and ({_oe.strip()}) <= {hbm_dim_size_strs[_other_i]}"
                            # Build partial src/dst slices
                            _p0_base = _inner_m2.group(1)  # "scalar * stride"
                            _inner_partial_len = f"{_stride2} - ({_tile_var2})"
                            _partial_src_parts = [f"{_p0_start}:{_p0_base} + {_stride2}"] + parts[1:]
                            _partial_src = f"{hbm_base}[{', '.join(_partial_src_parts)}]"
                            # For the dst SBUF, use 0-based indexing in the free dim
                            # (the SBUF tile is always indexed from 0, unlike the HBM src)
                            _free_count = _blk2  # default free count from block size
                            if len(parts) > 1:
                                _free_part = parts[1]
                                if ":" in _free_part:
                                    _fp_start, _fp_end = _free_part.split(":", 1)
                                    _fp_end = _fp_end.strip()
                                    _fp_start = _fp_start.strip()
                                    _fpm = _re_guard.match(r'^.+\+\s*(\d+)$', _fp_end)
                                    if _fpm:
                                        _free_count = _fpm.group(1)
                            _partial_dst = f"{dst}[0:{_inner_partial_len}, 0:{_free_count}]"
                            _partial_stmt = statement_from_string(
                                f"nisa.dma_copy(dst={_partial_dst}, src={_partial_src})"
                            )
                            _inner_tail_cases = [
                                create(
                                    ast.If,
                                    test=expr_from_string(_inner_overflow_cond),
                                    body=[_partial_stmt],
                                    orelse=[],
                                )
                            ]
                            # Also add combined K-tail + other-dim-tail cases
                            if len(parts) == 2 and len(hbm_dim_size_strs) >= 2:
                                _free_part2 = parts[1]
                                if ":" in _free_part2:
                                    _fp2_start, _fp2_end = _free_part2.split(":", 1)
                                    _fp2_start = _fp2_start.strip()
                                    _fp2_end = _fp2_end.strip()
                                    _fp2_size = hbm_dim_size_strs[1]
                                    # Combined case: K overflow AND free-dim overflow
                                    _combined_cond = (
                                        f"({_tile_var2}) >= 0 and ({_tile_var2}) < {_stride2} "
                                        f"and ({_tile_var2}) + {_blk2} > {_stride2} "
                                        f"and ({_p0_start}) >= 0 "
                                        f"and ({_fp2_start}) >= 0 "
                                        f"and ({_fp2_start}) < {_fp2_size} "
                                        f"and ({_fp2_end}) > {_fp2_size}"
                                    )
                                    _combined_dst = f"{dst}[0:{_inner_partial_len}, 0:{_fp2_size} - ({_fp2_start})]"
                                    _combined_src_parts = [f"{_p0_start}:{_p0_base} + {_stride2}", f"{_fp2_start}:{_fp2_size}"]
                                    _combined_src = f"{hbm_base}[{', '.join(_combined_src_parts)}]"
                                    _combined_stmt = statement_from_string(
                                        f"nisa.dma_copy(dst={_combined_dst}, src={_combined_src})"
                                    )
                                    _inner_tail_cases.append(
                                        create(
                                            ast.If,
                                            test=expr_from_string(_combined_cond),
                                            body=[_combined_stmt],
                                            orelse=[],
                                        )
                                    )
                            tail_cases = tail_cases + _inner_tail_cases
            state.codegen.add_statement(
                create(
                    ast.If,
                    test=expr_from_string(guard),
                    body=[stmt],
                    orelse=tail_cases,
                )
            )
        else:
            state.codegen.add_statement(stmt)

    if partition_dim > NKI_PARTITION_MAX and free_dims in ([], [1]):
        # 1D / [1, N] path: allocate a single [1, partition_dim] SBUF.
        # The DMA will copy partition_dim elements from HBM into the free
        # axis — total element count matches, so NKI handles the layout.
        device_fn._nki_sbuf_shapes[sbuf_name] = [1, partition_dim]
        state.codegen.add_statement(
            statement_from_string(
                f"{sbuf_name} = nl.ndarray([1, {partition_dim}], "
                f"{dtype_str}, buffer=nl.sbuf)"
            )
        )
        if extra_mask is not None:
            state.codegen.add_statement(
                statement_from_string(f"nisa.memset({sbuf_name}, value=0)")
            )
        if tensor.dim() == 1:
            # Source tensor reshaped to [1, N] at kernel entry.
            orig_slice = slice_parts[0] if slice_parts else f"0:{partition_dim}"
            if isinstance(orig_slice, (IndirectAP, DynamicAP)):
                # fall through to ap builder with a 2-part slice
                hbm_src_1d = _build_hbm_src(name, ["0:1", orig_slice])
            else:
                hbm_src_1d = f"{name}[0:1, {orig_slice}]"
        else:
            # 2D+ source tensor: use the full multi-D slice so partition_dim
            # elements are read from the partition axis of HBM.
            hbm_src_1d = _build_hbm_src(name, slice_parts)
        _emit_dma_copy(sbuf_name, hbm_src_1d, slice_parts)
        # Register HBM source for partition-broadcast codegen
        if not hasattr(device_fn, "_nki_hbm_sources"):
            device_fn._nki_hbm_sources = {}
        device_fn._nki_hbm_sources[sbuf_name] = hbm_src_1d
        if not hasattr(device_fn, "_nki_sbuf_dtypes"):
            device_fn._nki_sbuf_dtypes = {}
        device_fn._nki_sbuf_dtypes[sbuf_name] = dtype_str
    elif partition_dim > NKI_PARTITION_MAX:
        n_partitions = partition_dim // NKI_PARTITION_MAX
        assert partition_dim % NKI_PARTITION_MAX == 0
        free_str = ", ".join(str(d) for d in free_dims)
        # Fully unroll: N individual named variables + N explicit dma_copy statements
        tile_vars: list[str] = []
        for i in range(n_partitions):
            tile_var = f"{sbuf_name}_{i}"
            tile_vars.append(tile_var)
            state.codegen.add_statement(
                statement_from_string(
                    f"{tile_var} = nl.ndarray([{NKI_PARTITION_MAX}, {free_str}], "
                    f"{dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(f"nisa.memset({tile_var}, value=0)")
            )
        for i, tile_var in enumerate(tile_vars):
            part_slice_parts = list(slice_parts)
            if partition_offset_var is not None:
                part_slice_parts[0] = (
                    f"{partition_offset_var}+{i}*{NKI_PARTITION_MAX} : "
                    f"{partition_offset_var}+{i + 1}*{NKI_PARTITION_MAX}"
                )
            else:
                part_slice_parts[0] = (
                    f"{i}*{NKI_PARTITION_MAX} : {i + 1}*{NKI_PARTITION_MAX}"
                )
            part_slice_str = ", ".join(
                p if isinstance(p, str) else "__SENTINEL__"
                for p in part_slice_parts
            )
            part_hbm_src = f"{name}[{part_slice_str}]"
            _emit_dma_copy(
                tile_var,
                part_hbm_src,
                part_slice_parts,
                hbm_base=name,
            )
            if not hasattr(device_fn, "_nki_hbm_sources"):
                device_fn._nki_hbm_sources = {}
            device_fn._nki_hbm_sources[tile_var] = part_hbm_src
            if not hasattr(device_fn, "_nki_sbuf_dtypes"):
                device_fn._nki_sbuf_dtypes = {}
            device_fn._nki_sbuf_dtypes[tile_var] = dtype_str
        device_fn.register_tile_list(sbuf_name, tile_vars)
    else:
        # NKI requires at least 2D SBUF buffers. If free_dims is empty,
        # add a trailing dimension of 1.
        # Special case: 1D tensors are reshaped to [1, N] at kernel entry,
        # so their tile iterates over the free axis — use [1, tile_size].
        if not free_dims:
            if tensor.dim() == 1:
                sbuf_shape = [1, partition_dim]
            else:
                sbuf_shape = [partition_dim, 1]
        else:
            sbuf_shape = [partition_dim] + free_dims
        shape_str = ", ".join(str(d) for d in sbuf_shape)
        device_fn._nki_sbuf_shapes[sbuf_name] = sbuf_shape
        # Track SBUF dtype for correct broadcast buffer allocation
        if not hasattr(device_fn, "_nki_sbuf_dtypes"):
            device_fn._nki_sbuf_dtypes = {}
        device_fn._nki_sbuf_dtypes[sbuf_name] = dtype_str
        state.codegen.add_statement(
            statement_from_string(
                f"{sbuf_name} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.memset({sbuf_name}, value=0)")
        )
        # Handle dynamic loop subscripts via .ap()
        _has_dyn_2d = any(
            isinstance(p, (IndirectAP, DynamicAP))
            for p in slice_parts
        )
        if _has_dyn_2d:
            if tensor.dim() == 1 and len(slice_parts) == 1:
                hbm_src_expr = _build_hbm_src(name, ["0:1", slice_parts[0]])
            else:
                # OOB-skip fix: when loading from y inside a needs_prefill dynamic
                # loop, redirect to nki_return_buf instead.  That buffer is padded to
                # ceil(F/step)*step and prefilled with y, so no DMA is partially OOB.
                _dyn_loops_ld = getattr(state.device_function, "_nki_dyn_loops", {})
                _load_src = name
                for _dl_info in _dyn_loops_ld.values():
                    if not _dl_info.get("needs_prefill"):
                        continue
                    _ctr = _dl_info["counter"]
                    _ctr_f = _dl_info.get("counter_float", "")
                    if not any(
                        isinstance(p, DynamicAP) and (p.counter == _ctr or p.counter == _ctr_f)
                        for p in slice_parts
                    ):
                        continue
                    # Record y's name for the store path to emit the prefill.
                    if "y_src" not in _dl_info:
                        _dl_info["y_src"] = name
                    # Redirect to the return buffer (padded, prefilled with y).
                    _ret_buf = getattr(state.device_function, "_nki_return_buffer_name", None)
                    if _ret_buf:
                        _load_src = _ret_buf
                    else:
                        # Store hasn't fired yet — pre-reserve the name.
                        _pre = _dl_info.get("pre_reserved_buf")
                        if not _pre:
                            _pre = device_fn.new_var("nki_return_buf")
                            _dl_info["pre_reserved_buf"] = _pre
                        _load_src = _pre
                    break
                hbm_src_expr = _build_hbm_src(_load_src, slice_parts)
        else:
            slice_str = ", ".join(slice_parts)
            # For 1D tensors reshaped to [1, N] at kernel entry, the DMA source
            # needs 2D indexing. Prepend "0:1" for the partition dimension.
            if tensor.dim() == 1 and len(slice_parts) == 1:
                slice_str = f"0:1, {slice_str}"
            hbm_src_expr = f"{name}[{slice_str}]"
        _emit_dma_copy(
            sbuf_name,
            hbm_src_expr,
            slice_parts,
            hbm_base=(
                name
                if not _has_dyn_2d and not (tensor.dim() == 1 and len(slice_parts) == 1)
                else None
            ),
        )
        # Track HBM source for partition-broadcast codegen in tensor_tensor
        if not hasattr(device_fn, "_nki_hbm_sources"):
            device_fn._nki_hbm_sources = {}
        device_fn._nki_hbm_sources[sbuf_name] = hbm_src_expr
    if extra_mask is not None:
        mask_name = ast.unparse(extra_mask)
        tile_vars = device_fn.get_tile_list_vars(sbuf_name)
        if tile_vars is not None:
            raise exc.BackendUnsupported("nki", "masked tile-list loads")
        sbuf_shape = device_fn._nki_sbuf_shapes.get(sbuf_name)
        if sbuf_shape is None:
            raise exc.BackendUnsupported("nki", "masked load with unknown SBUF shape")
        masked_name = device_fn.new_var("_nki_masked_load", dce=True)
        device_fn._nki_sbuf_shapes[masked_name] = list(sbuf_shape)
        if not hasattr(device_fn, "_nki_sbuf_dtypes"):
            device_fn._nki_sbuf_dtypes = {}
        device_fn._nki_sbuf_dtypes[masked_name] = dtype_str
        shape_str = ", ".join(str(d) for d in sbuf_shape)
        state.codegen.add_statement(
            statement_from_string(
                f"{masked_name} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.memset({masked_name}, value=0)")
        )
        mask_shape = device_fn._nki_sbuf_shapes.get(mask_name, sbuf_shape)
        pred_src_name = mask_name
        if list(mask_shape) != list(sbuf_shape):
            pred_src_name = device_fn.new_var("_nki_mask_bcast", dce=True)
            device_fn._nki_sbuf_shapes[pred_src_name] = list(sbuf_shape)
            mask_dtype = device_fn._nki_sbuf_dtypes.get(
                mask_name, "nl.int32"
            )
            device_fn._nki_sbuf_dtypes[pred_src_name] = mask_dtype
            shape_tuple = ", ".join(str(d) for d in sbuf_shape)
            state.codegen.add_statement(
                statement_from_string(
                    f"{pred_src_name} = nl.broadcast_to({mask_name}, shape=({shape_tuple}))"
                )
            )
            mask_shape = sbuf_shape
        mask_shape_str = ", ".join(str(d) for d in mask_shape)
        pred_name = device_fn.new_var("_nki_mask_pred", dce=True)
        device_fn._nki_sbuf_shapes[pred_name] = list(mask_shape)
        device_fn._nki_sbuf_dtypes[pred_name] = "nl.uint32"
        state.codegen.add_statement(
            statement_from_string(
                f"{pred_name} = nl.ndarray([{mask_shape_str}], nl.uint32, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.tensor_copy(dst={pred_name}, src={pred_src_name})")
        )
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.tensor_copy_predicated(dst={masked_name}, src={sbuf_name}, predicate={pred_name})"
            )
        )
        sbuf_name = masked_name

    return expr_from_string(sbuf_name)


def store_stmt(state: CodegenState) -> None:
    from ..ast_extension import create
    from ..ast_extension import expr_from_string
    from ..ast_extension import statement_from_string
    from ..host_function import HostFunction

    NKI_PARTITION_MAX = 128

    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    env = CompileEnvironment.current()
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    fx_subscript = (
        state.fx_node.args[1]
        if state.fx_node is not None and len(state.fx_node.args) >= 2
        else None
    )
    slice_parts: list[str] = []
    is_scalar_dim_s: list[bool] = []
    partition_offset_var: str | None = None
    hbm_dim_size_strs_s: list[str] | None = None  # set by 3D early-exit or after loop
    tensor_dim_idx = 0
    for i, sub_val in enumerate(subscript):
        if sub_val is None:
            continue
        if tensor_dim_idx >= tensor.dim():
            break
        fx_node_i = fx_subscript[i] if fx_subscript is not None and i < len(fx_subscript) else None
        block_id = _nki_subscript_block_id(sub_val, fx_node_i, env)

        # Detect tile_id / tile_begin scalar-index subscripts (same as load).
        _is_tile_id_s = False
        _is_tile_begin_s = False
        if isinstance(fx_node_i, torch.fx.Node):
            from ...language.tile_ops import tile_id as _tile_id_fn
            try:
                from ...language.tile_ops import tile_begin as _tile_begin_fn
            except ImportError:
                _tile_begin_fn = None
            if fx_node_i.target is _tile_id_fn:
                _is_tile_id_s = True
            elif _tile_begin_fn is not None and fx_node_i.target is _tile_begin_fn:
                _is_tile_begin_s = True
            elif fx_node_i.op == "call_function":
                _tname = str(getattr(fx_node_i.target, "__name__", fx_node_i.target))
                if "tile_id" in _tname:
                    _is_tile_id_s = True
                elif "tile_begin" in _tname:
                    _is_tile_begin_s = True

        # slice(None) over a reduction dim: match by size as a last resort.
        if block_id is None and isinstance(sub_val, slice):
            dim_size = tensor.size(tensor_dim_idx) if tensor_dim_idx < tensor.dim() else None
            if dim_size is not None:
                for _bid in range(len(env.block_sizes)):
                    bs_info = env.block_sizes[_bid]
                    if not bs_info.reduction:
                        continue
                    block_size = bs_info.from_config_assert(state.config)
                    if block_size <= 1:
                        continue
                    if isinstance(dim_size, int) and block_size > 0 and dim_size % block_size == 0:
                        if _bid in state.codegen.active_device_loops:
                            block_id = _bid
                            break

        # tile_id(tile) produces an unbacked symbol that _nki_subscript_block_id
        # often can't map back to a block_id. Recover it from the tile_id FX
        # node's first arg (the tile whose .id this is), whose meta["val"] is
        # the block's symint. Without this, the store falls through to the
        # size-hint folding path below and emits a constant row index (e.g.
        # offset_0 // 128 -> 8192) so every block writes the same row.
        if _is_tile_id_s and block_id is None and isinstance(fx_node_i, torch.fx.Node):
            _tid_args = getattr(fx_node_i, "args", ())
            if _tid_args and isinstance(_tid_args[0], torch.fx.Node):
                _tid_val = _tid_args[0].meta.get("val")
                if isinstance(_tid_val, torch.SymInt):
                    _rec_bid = env.get_block_id(_tid_val)
                    if _rec_bid is not None:
                        block_id = _rec_bid
        if _is_tile_id_s and block_id is not None:
            offset_var = state.codegen.offset_var(block_id)
            block_size = env.block_sizes[block_id].from_config_assert(state.config)
            if int(block_size) == 1:
                id_expr = offset_var
            else:
                id_expr = f"{offset_var} // {int(block_size)}"
            if tensor_dim_idx == 0 and tensor.dim() != 1:
                partition_offset_var = f"({id_expr})"
            slice_parts.append(f"({id_expr}) : ({id_expr})+1")
            is_scalar_dim_s.append(True)
        elif _is_tile_begin_s and block_id is not None:
            offset_var = state.codegen.offset_var(block_id)
            if tensor_dim_idx == 0 and tensor.dim() != 1:
                partition_offset_var = f"({offset_var})"
            slice_parts.append(f"({offset_var}) : ({offset_var})+1")
            is_scalar_dim_s.append(True)
        elif (
            block_id is None
            and not isinstance(sub_val, slice)
            and isinstance(sub_val, (int, bool, torch.SymInt))
        ):
            # Plain scalar subscript.
            if isinstance(sub_val, torch.SymInt):
                import sympy as _sp_s
                _bs_subs_s: dict[_sp_s.Symbol, int] = {}
                for _bid in range(len(env.block_sizes)):
                    _bs = env.block_sizes[_bid]
                    _bs_subs_s[_bs.symbol()] = int(_bs.from_config_assert(state.config))
                try:
                    _scalar_val = int(sub_val._sympy_().subs(_bs_subs_s))
                except (TypeError, ValueError):
                    _scalar_val = int(env.size_hint(sub_val))
            else:
                _scalar_val = int(sub_val)
            if tensor_dim_idx == 0 and tensor.dim() != 1:
                partition_offset_var = f"{_scalar_val}"
            slice_parts.append(f"{_scalar_val} : {_scalar_val}+1")
            is_scalar_dim_s.append(True)
        elif block_id is not None and block_id in state.codegen.active_device_loops:
            offset_var = state.codegen.offset_var(block_id)
            block_size = env.block_sizes[block_id].from_config_assert(state.config)
            # 3D store ``out[tile.index + starts, head, :]``: the leading
            # subscript carries the tile block_id, but ``+ starts`` is a runtime
            # SBUF scalar that cannot be folded into a contiguous slice offset
            # (the strided-scatter path below would emit ``nisa.iota(offset=
            # <tensor>)``, which only accepts a compile-time int). Route to the
            # row-gather scatter, exactly as the load does for the mirror case.
            if tensor_dim_idx == 0:
                _scatter_3d = _nki_store_3d_row_scatter(
                    state, tensor, subscript, fx_subscript, i, fx_node_i, value
                )
                if _scatter_3d is not None:
                    slice_parts, hbm_dim_size_strs_s = _scatter_3d
                    is_scalar_dim_s = [False, False]
                    partition_offset_var = None
                    break
            # For 1D tensors, don't set partition_offset_var here.
            # The partition vs free layout decision is deferred to HBM
            # allocation based on the value's SBUF shape.
            if tensor_dim_idx == 0 and tensor.dim() != 1:
                partition_offset_var = offset_var
            # Check if the subscript is a SHIFTED tile: e.g. tile_c * chunk_size + tile_m
            # In that case, use the shifted offset instead of just offset_var.
            # First try _nki_shifted_tile_subscript; if that fails, try direct AST lookup
            # for compound patterns like "mul_expr + tile.index".
            _shifted_s = _nki_shifted_tile_subscript(fx_node_i, state, env)
            if _shifted_s is None and isinstance(fx_node_i, torch.fx.Node):
                _t_name = str(getattr(fx_node_i, "target", ""))
                if "add.Tensor" in _t_name or fx_node_i.target is torch.ops.aten.add.Tensor:
                    _add_args = fx_node_i.args
                    if len(_add_args) == 2:
                        _a0, _a1 = _add_args
                        # Try: "scalar_expr + tile.index" → find the scalar ast
                        for _tile_arg, _scalar_arg in [(_a0, _a1), (_a1, _a0)]:
                            if isinstance(_tile_arg, torch.fx.Node):
                                _tile_val = _tile_arg.meta.get("val")
                                if isinstance(_tile_val, torch.Tensor) and _tile_val.ndim == 1:
                                    _tile_size = _tile_val.size(0)
                                    if isinstance(_tile_size, torch.SymInt):
                                        _tile_bid = env.get_block_id(_tile_size)
                                        if _tile_bid is None:
                                            _sym_expr0 = _tile_size._sympy_()
                                            if hasattr(_sym_expr0, "free_symbols"):
                                                for _s in _sym_expr0.free_symbols:
                                                    _tile_bid = env.get_block_id(_s)
                                                    if _tile_bid is not None:
                                                        break
                                        if _tile_bid == block_id and _tile_bid in state.codegen.active_device_loops:
                                            # Found the tile arg; get AST for scalar arg
                                            if isinstance(_scalar_arg, torch.fx.Node):
                                                _scalar_ast = state.codegen.ast_for_fx_node(_scalar_arg)
                                                if isinstance(_scalar_ast, ast.AST):
                                                    _scalar_expr = ast.unparse(_scalar_ast)
                                                    _start = f"{_scalar_expr} + {offset_var}"
                                                    _shifted_s = f"{_start}:{_start}+{int(block_size)}"
                                                    break
                                            elif isinstance(_scalar_arg, (int, float)):
                                                _start = f"{_scalar_arg} + {offset_var}"
                                                _shifted_s = f"{_start}:{_start}+{int(block_size)}"
                                                break
            # Check dynamic loop
            _dyn_loops_st = getattr(device_fn, "_nki_dyn_loops", {})
            if block_id in _dyn_loops_st:
                _counter_st = _dyn_loops_st[block_id]["counter"]
                # If subscript is start + tile.index where start is an SBUF scalar,
                # __DYN_AP__ would miss the start offset; use row_gather instead.
                _needs_sbuf_offset_st = False
                if (
                    tensor_dim_idx == 0
                    and tensor.dim() == 2
                    and isinstance(sub_val, torch.Tensor)
                    and isinstance(fx_node_i, torch.fx.Node)
                ):
                    _ck_target_st = str(getattr(fx_node_i, "target", ""))
                    if "add.Tensor" in _ck_target_st or "sub.Tensor" in _ck_target_st:
                        _sbuf_shapes_st = device_fn._nki_sbuf_shapes
                        for _st_ck_arg in fx_node_i.args[:2]:
                            if isinstance(_st_ck_arg, torch.fx.Node):
                                _st_ck_ast = state.codegen.ast_for_fx_node(_st_ck_arg)
                                if isinstance(_st_ck_ast, ast.AST):
                                    _st_ck_nm = ast.unparse(_st_ck_ast)
                                    if _st_ck_nm in _sbuf_shapes_st:
                                        _needs_sbuf_offset_st = True
                                        break
                if _needs_sbuf_offset_st:
                    _value_name_sst = ast.unparse(value) if isinstance(value, ast.AST) else str(value)
                    _val_sbuf_shape_sst = device_fn._nki_sbuf_shapes.get(_value_name_sst)
                    _p_count_sst = _val_sbuf_shape_sst[0] if _val_sbuf_shape_sst and len(_val_sbuf_shape_sst) >= 1 else int(block_size)
                    _row_gather_sst = _nki_row_index_gather(fx_node_i, state, _p_count_sst)
                    if _row_gather_sst is not None:
                        slice_parts.append(_row_gather_sst)
                        is_scalar_dim_s.append(False)
                        tensor_dim_idx += 1
                        continue
                slice_parts.append(DynamicAP(counter=_counter_st, block_size=int(block_size)))
            elif _shifted_s is not None:
                # Use the shifted slice (includes both the static offset and the tile offset)
                slice_parts.append(_shifted_s)
                if tensor_dim_idx == 0 and tensor.dim() != 1:
                    partition_offset_var = _shifted_s.split(":", 1)[0].strip()
            elif (
                tensor_dim_idx == 0
                and tensor.dim() == 2
                and isinstance(sub_val, torch.Tensor)
            ):
                # Check if the subscript is a non-contiguous gather (indirect scatter).
                # When block_id was found via size match but the subscript is a gather
                # (e.g. sorted_to_orig_token_idx[indices] or torch.where result),
                # use the row scatter (.ap() with vector_offset) mechanism.
                _value_name_s2 = ast.unparse(value) if isinstance(value, ast.AST) else str(value)
                _val_sbuf_shape = device_fn._nki_sbuf_shapes.get(_value_name_s2)
                _p_count_s2 = _val_sbuf_shape[0] if _val_sbuf_shape and len(_val_sbuf_shape) >= 1 else None
                _row_scatter_s2 = _nki_row_index_gather(fx_node_i, state, _p_count_s2)
                if _row_scatter_s2 is not None:
                    slice_parts.append(_row_scatter_s2)
                    is_scalar_dim_s.append(False)
                    tensor_dim_idx += 1
                    continue
                slice_parts.append(f"{offset_var} : {offset_var}+{int(block_size)}")
            else:
                slice_parts.append(f"{offset_var} : {offset_var}+{int(block_size)}")
            is_scalar_dim_s.append(False)
        else:
            if (
                tensor_dim_idx == 0
                and tensor.dim() == 2
                and isinstance(sub_val, torch.Tensor)
            ):
                # Get partition size from the value being stored
                _store_val_name = ast.unparse(value) if isinstance(value, ast.AST) else str(value)
                _store_val_shape = device_fn._nki_sbuf_shapes.get(_store_val_name)
                _store_p_count = _store_val_shape[0] if _store_val_shape and len(_store_val_shape) >= 1 else None
                row_scatter = _nki_row_index_gather(fx_node_i, state, _store_p_count)
                if row_scatter is not None:
                    slice_parts.append(row_scatter)
                    is_scalar_dim_s.append(False)
                    tensor_dim_idx += 1
                    continue

            # 3D tensor with pattern [vec + starts, scalar_head, :] — same as load.
            if tensor_dim_idx == 0:
                _scatter_3d = _nki_store_3d_row_scatter(
                    state, tensor, subscript, fx_subscript, i, fx_node_i, value
                )
                if _scatter_3d is not None:
                    slice_parts, hbm_dim_size_strs_s = _scatter_3d
                    is_scalar_dim_s = [False, False]
                    partition_offset_var = None
                    # Skip remaining dims — the flat scatter covers them all.
                    break

            size_i = tensor.size(tensor_dim_idx) if tensor_dim_idx < tensor.dim() else sub_val
            size_str = (
                state.sympy_expr(size_i._sympy_())
                if isinstance(size_i, torch.SymInt)
                else str(size_i)
            )
            slice_parts.append(f"0 : {size_str}")
            is_scalar_dim_s.append(False)
        tensor_dim_idx += 1

    def _store_tensor_dim_size_str(dim_idx: int) -> str:
        dim_size = tensor.size(dim_idx)
        return (
            state.sympy_expr(dim_size._sympy_())
            if isinstance(dim_size, torch.SymInt)
            else str(dim_size)
        )

    if hbm_dim_size_strs_s is None:
        hbm_dim_size_strs_s = [
            _store_tensor_dim_size_str(dim_idx) for dim_idx in range(tensor.dim())
        ]
    flattened_high_rank_store = False

    # 3D+ tensors: reshaped to 2D at kernel entry.
    # Combine leading slice_parts into one flat partition slice.
    if tensor.dim() > 2 and len(slice_parts) > 2:
        import sympy as _sympy_store

        flattened_high_rank_store = True
        _bs_subs_store: dict[_sympy_store.Symbol, int] = {}
        for _bid in range(len(env.block_sizes)):
            _bs = env.block_sizes[_bid]
            _bs_subs_store[_bs.symbol()] = int(_bs.from_config_assert(state.config))

        def _resolve_dim_store(s: int | torch.SymInt) -> int:
            if isinstance(s, int):
                return s
            return int(s._sympy_().subs(_bs_subs_store))

        leading_offsets_s: list[str] = []
        original_dim_sizes_s: list[int] = []
        leading_block_sizes_s: list[int] = []
        for dim_i in range(tensor.dim() - 1):
            sp = slice_parts[dim_i]
            if isinstance(sp, str) and ":" in sp:
                off_str, end_str = sp.split(":", 1)
                off_str = off_str.strip()
                leading_offsets_s.append(off_str)
                # Use rfind to find the LAST '+' — avoids compound expressions like
                # "offset_0 + mul_1+128" being split at the first '+' in "offset_0 + mul_1"
                plus_idx = end_str.rfind("+")
                if plus_idx >= 0:
                    bs_str = end_str[plus_idx + 1:].strip()
                    try:
                        leading_block_sizes_s.append(int(bs_str))
                    except ValueError:
                        leading_block_sizes_s.append(1)
                else:
                    end_str = end_str.strip()
                    try:
                        end_val = int(end_str)
                        start_val = int(off_str) if off_str.strip().isdigit() else 0
                        leading_block_sizes_s.append(end_val - start_val)
                    except (ValueError, TypeError):
                        leading_block_sizes_s.append(1)
            else:
                leading_offsets_s.append(sp.strip())
                leading_block_sizes_s.append(1)
            original_dim_sizes_s.append(
                int(tensor.size(dim_i)) if isinstance(tensor.size(dim_i), int)
                else _resolve_dim_store(tensor.size(dim_i))
            )

        flat_offset_parts_s: list[str] = []
        for j, off in enumerate(leading_offsets_s):
            multiplier_parts = [str(original_dim_sizes_s[k]) for k in range(j + 1, len(original_dim_sizes_s))]
            if multiplier_parts:
                multiplier = " * ".join(multiplier_parts)
                off_expr_s = f"({off})" if ("+" in off or "-" in off) else off
                flat_offset_parts_s.append(f"{off_expr_s} * {multiplier}")
            else:
                flat_offset_parts_s.append(off)
        flat_offset_s = " + ".join(flat_offset_parts_s)
        flat_block_size_s = 1
        for bs in leading_block_sizes_s:
            flat_block_size_s *= bs

        # Strided-scatter detection (mirrors strided-gather in load codegen):
        # one tiled dim followed by scalar dim(s) → rows are non-contiguous.
        _tile_dim_idx_st = None
        for _di, _bs in enumerate(leading_block_sizes_s):
            if _bs > 1:
                if _tile_dim_idx_st is None:
                    _tile_dim_idx_st = _di
                else:
                    _tile_dim_idx_st = None
                    break
        _use_strided_scatter = (
            _tile_dim_idx_st is not None
            and _tile_dim_idx_st < len(leading_block_sizes_s) - 1
            and all(leading_block_sizes_s[_di] == 1
                    for _di in range(_tile_dim_idx_st + 1, len(leading_block_sizes_s)))
            and not any(isinstance(p, (IndirectAP, DynamicAP))
                        for p in slice_parts)
        )
        if _use_strided_scatter:
            _tile_bs_st = leading_block_sizes_s[_tile_dim_idx_st]
            _tile_off_st = leading_offsets_s[_tile_dim_idx_st]
            _stride_st = 1
            for _di in range(_tile_dim_idx_st + 1, len(original_dim_sizes_s)):
                _stride_st *= original_dim_sizes_s[_di]
            if _stride_st > 1:
                _scalar_parts_st: list[str] = []
                for _di in range(_tile_dim_idx_st + 1, len(leading_offsets_s)):
                    _m = [str(original_dim_sizes_s[_k])
                          for _k in range(_di + 1, len(original_dim_sizes_s))]
                    _off_e = leading_offsets_s[_di]
                    if _m:
                        _off_p = f"({_off_e})" if ("+" in _off_e or "-" in _off_e) else _off_e
                        _scalar_parts_st.append(f"{_off_p} * {' * '.join(_m)}")
                    else:
                        _scalar_parts_st.append(_off_e)
                _prefix_parts_st: list[str] = []
                for _di in range(0, _tile_dim_idx_st):
                    _m = [str(original_dim_sizes_s[_k])
                          for _k in range(_di + 1, len(original_dim_sizes_s))]
                    _off_e = leading_offsets_s[_di]
                    if _m:
                        _off_p = f"({_off_e})" if ("+" in _off_e or "-" in _off_e) else _off_e
                        _prefix_parts_st.append(f"{_off_p} * {' * '.join(_m)}")
                    else:
                        _prefix_parts_st.append(_off_e)
                _scalar_off_st = " + ".join(_scalar_parts_st) if _scalar_parts_st else "0"
                _prefix_off_st = " + ".join(_prefix_parts_st) if _prefix_parts_st else None

                from ..ast_extension import statement_from_string as _sfs_ss
                _iota_st = device_fn.new_var("_ss_iota", dce=True)
                device_fn._nki_sbuf_shapes[_iota_st] = [1, _tile_bs_st]
                device_fn._nki_sbuf_dtypes[_iota_st] = "nl.int32"
                state.codegen.add_statement(_sfs_ss(
                    f"{_iota_st} = nl.ndarray([1, {_tile_bs_st}], nl.int32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(_sfs_ss(
                    f"nisa.iota(dst={_iota_st}, pattern=[[1, {_tile_bs_st}]], "
                    f"offset={_tile_off_st}, channel_multiplier=0)"
                ))
                _scaled_st = device_fn.new_var("_ss_scaled", dce=True)
                device_fn._nki_sbuf_shapes[_scaled_st] = [1, _tile_bs_st]
                device_fn._nki_sbuf_dtypes[_scaled_st] = "nl.int32"
                state.codegen.add_statement(_sfs_ss(
                    f"{_scaled_st} = nl.ndarray([1, {_tile_bs_st}], nl.int32, buffer=nl.sbuf)"
                ))
                state.codegen.add_statement(_sfs_ss(
                    f"nisa.tensor_scalar(dst={_scaled_st}, data={_iota_st}, "
                    f"op0=nl.multiply, operand0={_stride_st}, op1=None)"
                ))
                _row_st = device_fn.new_var("_ss_row", dce=True)
                device_fn._nki_sbuf_shapes[_row_st] = [1, _tile_bs_st]
                device_fn._nki_sbuf_dtypes[_row_st] = "nl.int32"
                state.codegen.add_statement(_sfs_ss(
                    f"{_row_st} = nl.ndarray([1, {_tile_bs_st}], nl.int32, buffer=nl.sbuf)"
                ))
                if _scalar_off_st == "0":
                    state.codegen.add_statement(_sfs_ss(
                        f"nisa.tensor_copy(dst={_row_st}, src={_scaled_st})"
                    ))
                else:
                    state.codegen.add_statement(_sfs_ss(
                        f"nisa.tensor_scalar(dst={_row_st}, data={_scaled_st}, "
                        f"op0=nl.add, operand0={_scalar_off_st}, op1=None)"
                    ))
                if _prefix_off_st is not None:
                    state.codegen.add_statement(_sfs_ss(
                        f"nisa.tensor_scalar(dst={_row_st}, data={_row_st}, "
                        f"op0=nl.add, operand0={_prefix_off_st}, op1=None)"
                    ))
                _vec_st = _nki_as_uint32_p1_vector(state, _row_st, _tile_bs_st)
                if _vec_st is not None:
                    _flat_hbm_st = 1
                    for _ds in original_dim_sizes_s:
                        _flat_hbm_st *= _ds
                    _d_int_st = _resolve_dim_store(tensor.size(tensor.dim() - 1))
                    slice_parts = [IndirectAP(vec_var=_vec_st, p_count=_tile_bs_st, pattern=None), f"0:{_d_int_st}"]
                    is_scalar_dim_s = [False, False]
                    flat_block_size_s = _tile_bs_st
                    hbm_dim_size_strs_s = [str(_flat_hbm_st), str(_d_int_st)]
                    partition_offset_var = None
                    _use_strided_scatter = False  # signal: skip flat slice below

        if _use_strided_scatter is False and isinstance(slice_parts[0], IndirectAP):
            pass  # strided scatter succeeded — skip flat slice
        else:
            flat_slice = f"({flat_offset_s}) : ({flat_offset_s}) + {flat_block_size_s}"
            slice_parts = [flat_slice] + [slice_parts[-1]]
            partition_offset_var = f"({flat_offset_s})"

        if not isinstance(slice_parts[0], IndirectAP):
            flat_hbm_partition_s = 1
            for dim_size in original_dim_sizes_s:
                flat_hbm_partition_s *= dim_size
            hbm_dim_size_strs_s = [
                str(flat_hbm_partition_s),
                _store_tensor_dim_size_str(tensor.dim() - 1),
            ]

    value_name = ast.unparse(value) if isinstance(value, ast.AST) else str(value)
    tile_vars = device_fn.get_tile_list_vars(value_name)

    # NKI output tensors: use return buffer (allocate, write, return) instead of
    # passed-in tensor; NKI does not propagate writes to passed tensors.
    # Each distinct output tensor gets its own HBM return buffer.
    use_nki_return = False
    ret_buf_name: str | None = None
    try:
        hf = HostFunction.current()
        if hf is not None and tensor in hf.tensor_to_origin:
            use_nki_return = True
            origin = hf.tensor_to_origin[tensor]
            return_host_var = origin.host_str()

            # Follow view chain to find the base tensor.
            # When `out_flat = out.view(-1)`, the NKI wrapper must assign
            # to `out` (the base), not `out_flat` (a view that gets rebound).
            _base_tensor = tensor
            while getattr(_base_tensor, "_base", None) is not None:
                _base_tensor = _base_tensor._base
            if _base_tensor is not tensor and _base_tensor in hf.tensor_to_origin:
                base_origin = hf.tensor_to_origin[_base_tensor]
                return_host_var = base_origin.host_str()
                # We'll set host_reshape to the base tensor's shape below
                # (stored in _nki_base_tensor_shape for use in reshape logic)
                if not hasattr(device_fn, "_nki_base_tensor_shapes"):
                    device_fn._nki_base_tensor_shapes = {}
                device_fn._nki_base_tensor_shapes[id(tensor)] = list(_base_tensor.shape)

            # Use per-tensor return buffers dict for multi-output support
            if not hasattr(device_fn, "_nki_return_buffers"):
                device_fn._nki_return_buffers = {}


            tensor_id = id(tensor)
            if tensor_id in device_fn._nki_return_buffers:
                ret_buf_name = device_fn._nki_return_buffers[tensor_id]["buf_name"]
            else:
                # If the load path pre-reserved a buffer name, reuse it.
                _pre_reserved = None
                for _dli in getattr(device_fn, "_nki_dyn_loops", {}).values():
                    if _dli.get("pre_reserved_buf") and "prefill_emitted" not in _dli:
                        _pre_reserved = _dli["pre_reserved_buf"]
                        break
                ret_buf_name = _pre_reserved if _pre_reserved else device_fn.new_var("nki_return_buf")
                dtype_str = env.backend.dtype_str(tensor.dtype)
                shape_parts = []
                for dim_i in range(tensor.dim()):
                    size_i = tensor.size(dim_i)
                    size_str = (
                        state.sympy_expr(size_i._sympy_())
                        if isinstance(size_i, torch.SymInt)
                        else str(size_i)
                    )
                    shape_parts.append(size_str)
                flat_extent = shape_parts[0] if shape_parts else str(tensor.numel())

                # NKI requires 2D buffers (partition, free). For 1D output
                # tensors, choose layout based on the value's SBUF shape:
                # - [P, 1] value (from reduction): use [N, 1] HBM (partition-axis)
                # - [1, F] value (element-wise): use [1, N] HBM (free-axis)
                host_reshape = None
                # Check if we're storing to a view of a base tensor
                base_shapes = getattr(device_fn, "_nki_base_tensor_shapes", {})
                base_shape = base_shapes.get(tensor_id)
                if tensor.dim() == 1:
                    if return_host_var is not None:
                        flat_extent = device_fn.new_var("nki_return_numel")
                        device_fn.constexpr_arg(
                            flat_extent, f"{return_host_var}.numel()"
                        )
                    val_sbuf_shape = device_fn._nki_sbuf_shapes.get(value_name)
                    use_partition_layout = (
                        val_sbuf_shape is not None
                        and len(val_sbuf_shape) >= 2
                        and val_sbuf_shape[0] > 1
                        and val_sbuf_shape[1] == 1
                    )
                    if use_partition_layout:
                        shape_str = f"{flat_extent}, 1"
                    else:
                        shape_str = f"1, {flat_extent}"
                    if base_shape is not None:
                        # Reshape NKI result to the base tensor's shape
                        host_reshape = f"[{', '.join(str(d) for d in base_shape)}]"
                    elif return_host_var is not None:
                        host_reshape = f"[{return_host_var}.numel()]"
                    else:
                        host_reshape = f"[{flat_extent}]"
                elif tensor.dim() > 2:
                    # 3D+ output: flatten to 2D for NKI HBM buffer,
                    # reshape back to original shape on host.
                    leading = shape_parts[:-1]
                    flat_leading = " * ".join(f"({p})" for p in leading)
                    shape_str = f"({flat_leading}), {shape_parts[-1]}"
                    host_reshape = f"[{', '.join(shape_parts)}]"
                else:
                    shape_str = ", ".join(shape_parts)
                    if base_shape is not None:
                        host_reshape = f"[{', '.join(str(d) for d in base_shape)}]"
                    # OOB-skip padding: when a dynamic loop's DMA tile size does not
                    # evenly divide the output's free dimension, oob_mode=skip silently
                    # drops the entire last DMA (not just the OOB elements), losing the
                    # final valid data.  Pad the free dim to the next tile-size multiple
                    # so every DMA is fully in-bounds.  The host_reshape clips it back
                    # to the real shape when the result is returned.
                    _dyn_loops_alloc = getattr(device_fn, "_nki_dyn_loops", {})
                    for _dli_alloc in _dyn_loops_alloc.values():
                        _step_alloc = _dli_alloc.get("step", 1)
                        if _step_alloc > 1 and len(shape_parts) == 2:
                            try:
                                _f_alloc = int(shape_parts[1])
                                if _f_alloc % _step_alloc != 0:
                                    _f_padded = ((_f_alloc + _step_alloc - 1) // _step_alloc) * _step_alloc
                                    # Use a slice [:, :F] to trim padding rather than
                                    # reshape (which would fail: P*F_padded != P*F).
                                    host_reshape = None
                                    device_fn._nki_return_host_slice = f"[:, :{_f_alloc}]"
                                    shape_str = f"{shape_parts[0]}, {_f_padded}"
                                    # Record the padded free-dim size so the AP-pattern
                                    # builders use the correct stride for this buffer.
                                    device_fn._nki_return_buf_free_dim = _f_padded
                            except (ValueError, TypeError):
                                pass
                            break
                device_fn.preamble.append(
                    statement_from_string(
                        f"{ret_buf_name} = nl.ndarray([{shape_str}], dtype={dtype_str}, buffer=nl.shared_hbm)"
                    )
                )
                device_fn._nki_return_buffers[tensor_id] = {
                    "buf_name": ret_buf_name,
                    "host_var": return_host_var,
                    "host_reshape": host_reshape,
                    "flat_extent": flat_extent,
                }
                # Keep backward compat: also set single-buffer attrs
                # (used if there's only one output)
                if len(device_fn._nki_return_buffers) == 1:
                    device_fn._nki_return_buffer_name = ret_buf_name
                    device_fn._nki_return_host_var = return_host_var
                    device_fn._nki_return_host_reshape = host_reshape
    except Exception:
        pass

    if tile_vars:
        # Fully unroll: N explicit dma_copy statements, one per partition tile
        name = ret_buf_name if use_nki_return else device_fn.tensor_arg(tensor).name

        # Determine whether tiles split along the partition dim (dim 0) or
        # the free dim (last dim). If partition_offset_var is set, tiles split
        # along dim 0 (partition). Otherwise, if the destination is 2D+ and
        # there's no partition tiling, tiles split along the last dim (free).
        tile_along_free = (
            partition_offset_var is None
            and len(slice_parts) >= 2
        )

        for i, tile_var in enumerate(tile_vars):
            part_slice_parts = list(slice_parts)
            if tile_along_free:
                # Tiles distribute along the free dimension (last dim)
                part_slice_parts[-1] = (
                    f"{i}*{NKI_PARTITION_MAX} : {i + 1}*{NKI_PARTITION_MAX}"
                )
            elif partition_offset_var is not None:
                part_slice_parts[0] = (
                    f"{partition_offset_var}+{i}*{NKI_PARTITION_MAX} : "
                    f"{partition_offset_var}+{i + 1}*{NKI_PARTITION_MAX}"
                )
            else:
                part_slice_parts[0] = (
                    f"{i}*{NKI_PARTITION_MAX} : {i + 1}*{NKI_PARTITION_MAX}"
                )
            part_slice_str = ", ".join(
                p if isinstance(p, str) else "__SENTINEL__"
                for p in part_slice_parts
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={name}[{part_slice_str}], src={tile_var})"
                )
            )
    else:
        # Adjust slice for [1, N] SBUF values stored to 2D destinations:
        # When the stored value is a [1, N] accumulator (partition=1) but the
        # destination's first-dim slice has width > 1 (e.g. offset_0:offset_0+32),
        # shrink the first dim to width 1 using the block-id formula.
        adjusted_slice_parts = list(slice_parts)
        if len(slice_parts) >= 1 and partition_offset_var is not None:
            sbuf_shape = device_fn._nki_sbuf_shapes.get(value_name)
            if sbuf_shape is not None and len(sbuf_shape) >= 1 and sbuf_shape[0] == 1:
                # [1, N] value — store to a single row: offset//block_size
                # find block_size from offset_var
                for bid in state.codegen.active_device_loops.keys():
                    if bid < len(env.block_sizes) and state.codegen.active_device_loops[bid]:
                        bsize = env.block_sizes[bid].from_config_assert(state.config)
                        if bsize > 1:
                            boff = state.codegen.offset_var(bid)
                            if boff == partition_offset_var:
                                adjusted_slice_parts[0] = (
                                    f"{partition_offset_var} // {int(bsize)} : "
                                    f"{partition_offset_var} // {int(bsize)} + 1"
                                )
                                break

        slice_str = ", ".join(
            p if isinstance(p, str) else "__SENTINEL__"
            for p in adjusted_slice_parts
        )
        name = ret_buf_name if use_nki_return else device_fn.tensor_arg(tensor).name

        def _try_emit_flat_index_store() -> bool:
            if (
                tensor.dim() != 1
                or len(subscript) != 1
                or not isinstance(subscript[0], torch.Tensor)
                or fx_subscript is None
                or not isinstance(fx_subscript[0], torch.fx.Node)
            ):
                return False
            index_node = fx_subscript[0]
            index_val = index_node.meta.get("val")
            if not isinstance(index_val, torch.Tensor) or index_val.ndim not in (2, 3):
                return False
            if (
                index_node.op != "call_function"
                or len(index_node.args) < 2
                or not (
                    "add.Tensor" in str(index_node.target)
                    or index_node.target is torch.ops.aten.add.Tensor
                )
            ):
                return False

            import sympy as _sp_store_flat

            bs_subs: dict[_sp_store_flat.Symbol, int] = {}
            if state.config is not None:
                for bs in env.block_sizes:
                    cfg = bs.from_config(state.config)
                    if isinstance(cfg, int):
                        bs_subs[bs.symbol()] = cfg

            def _resolve_index_dim(dim: object) -> int:
                if isinstance(dim, int):
                    return dim
                if isinstance(dim, torch.SymInt):
                    try:
                        return int(dim._sympy_().subs(bs_subs))
                    except (TypeError, ValueError):
                        return int(env.size_hint(dim))
                return int(dim)

            if index_val.ndim == 3:
                index_dim0 = _resolve_index_dim(index_val.shape[0])
                index_dim1 = _resolve_index_dim(index_val.shape[1])
                if index_dim1 == 1:
                    p_count = index_dim0
                elif index_dim0 == 1:
                    p_count = index_dim1
                else:
                    return False
                f_count = _resolve_index_dim(index_val.shape[2])
            else:
                p_count = _resolve_index_dim(index_val.shape[0])
                f_count = _resolve_index_dim(index_val.shape[1])
            if p_count > NKI_PARTITION_MAX:
                return False

            lhs_node, rhs_node = index_node.args[:2]
            if not (
                isinstance(lhs_node, torch.fx.Node)
                and isinstance(rhs_node, torch.fx.Node)
            ):
                return False
            lhs_val = lhs_node.meta.get("val")
            rhs_val = rhs_node.meta.get("val")
            if not (
                isinstance(lhs_val, torch.Tensor)
                and isinstance(rhs_val, torch.Tensor)
                and lhs_val.ndim in (2, 3)
                and rhs_val.ndim in (2, 3)
            ):
                return False

            def _shape_2d(value: torch.Tensor) -> tuple[int, int] | None:
                if value.ndim == 3:
                    dim0 = _resolve_index_dim(value.shape[0])
                    dim1 = _resolve_index_dim(value.shape[1])
                    if dim1 == 1:
                        return (
                            dim0,
                            _resolve_index_dim(value.shape[2]),
                        )
                    if dim0 == 1:
                        return (
                            dim1,
                            _resolve_index_dim(value.shape[2]),
                        )
                    else:
                        return None
                return (
                    _resolve_index_dim(value.shape[0]),
                    _resolve_index_dim(value.shape[1]),
                )

            lhs_shape = _shape_2d(lhs_val)
            rhs_shape = _shape_2d(rhs_val)
            if lhs_shape is None or rhs_shape is None:
                return False
            if lhs_shape == (p_count, 1) and rhs_shape == (1, f_count):
                base_node, feature_node = lhs_node, rhs_node
            elif rhs_shape == (p_count, 1) and lhs_shape == (1, f_count):
                base_node, feature_node = rhs_node, lhs_node
            else:
                return False

            base_ast = state.codegen.ast_for_fx_node(base_node)
            feature_ast = state.codegen.ast_for_fx_node(feature_node)
            if not isinstance(base_ast, ast.AST) or not isinstance(feature_ast, ast.AST):
                return False
            base_name = ast.unparse(base_ast)
            feature_name = ast.unparse(feature_ast)
            vec_offset = _nki_as_uint32_p1_vector(state, base_name, p_count)
            if vec_offset is None:
                return False

            feature_base_name = feature_name
            while "_copy" in feature_base_name:
                feature_base_name = feature_base_name[: feature_base_name.rfind("_copy")]
            feature_block_id: int | None = None
            for block_id in state.codegen.active_device_loops:
                try:
                    if state.codegen.index_var(block_id) == feature_base_name:
                        feature_block_id = block_id
                        break
                except (KeyError, IndexError):
                    continue
            if feature_block_id is not None:
                dyn_loops = getattr(device_fn, "_nki_dyn_loops", {})
                if feature_block_id in dyn_loops:
                    counter = dyn_loops[feature_block_id]["counter"]
                    combined = device_fn.new_var("_ig_base_plus_counter", dce=True)
                    counter_u32 = device_fn.new_var("_ig_counter_u32", dce=True)
                    counter_bcast = device_fn.new_var("_ig_counter_bcast", dce=True)
                    device_fn._nki_sbuf_shapes[combined] = [p_count, 1]
                    device_fn._nki_sbuf_dtypes[combined] = "nl.uint32"
                    device_fn._nki_sbuf_shapes[counter_u32] = [1, 1]
                    device_fn._nki_sbuf_dtypes[counter_u32] = "nl.uint32"
                    device_fn._nki_sbuf_shapes[counter_bcast] = [p_count, 1]
                    device_fn._nki_sbuf_dtypes[counter_bcast] = "nl.uint32"
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{combined} = nl.ndarray([{p_count}, 1], "
                            "nl.uint32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={combined}, src={vec_offset})"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{counter_u32} = nl.ndarray([1, 1], "
                            "nl.uint32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={counter_u32}, src={counter})"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{counter_bcast} = nl.broadcast_to({counter_u32}, "
                            f"shape=({p_count}, 1))"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_tensor(dst={combined}, data1={combined}, "
                            f"data2={counter_bcast}, op=nl.add)"
                        )
                    )
                    vec_offset = combined
                else:
                    feature_offset = state.codegen.offset_var(feature_block_id)
                    if feature_offset != "0":
                        combined = device_fn.new_var("_ig_base_plus_offset", dce=True)
                        device_fn._nki_sbuf_shapes[combined] = [p_count, 1]
                        device_fn._nki_sbuf_dtypes[combined] = "nl.uint32"
                        state.codegen.add_statement(
                            statement_from_string(
                                f"{combined} = nl.ndarray([{p_count}, 1], "
                                "nl.uint32, buffer=nl.sbuf)"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"nisa.tensor_copy(dst={combined}, src={vec_offset})"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"nisa.tensor_scalar(dst={combined}, data={combined}, "
                                f"op0=nl.add, operand0={feature_offset})"
                            )
                        )
                        vec_offset = combined

            flat_extent = str(int(tensor.numel()))
            if use_nki_return:
                tensor_id = id(tensor)
                buf_info = device_fn._nki_return_buffers.get(tensor_id, {})
                flat_extent = str(buf_info.get("flat_extent", flat_extent))

            if extra_mask is not None:
                mask_name = ast.unparse(extra_mask)
                mask_shape = device_fn._nki_sbuf_shapes.get(mask_name)
                if mask_shape is not None and len(mask_shape) == 2:
                    pred_col = device_fn.new_var("_nki_store_pred_col", dce=True)
                    masked_offsets = device_fn.new_var("_nki_store_offsets", dce=True)
                    device_fn._nki_sbuf_shapes[pred_col] = [p_count, 1]
                    device_fn._nki_sbuf_dtypes[pred_col] = "nl.uint32"
                    device_fn._nki_sbuf_shapes[masked_offsets] = [p_count, 1]
                    device_fn._nki_sbuf_dtypes[masked_offsets] = "nl.uint32"
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{pred_col} = nl.ndarray([{p_count}, 1], "
                            "nl.uint32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={pred_col}, "
                            f"src={mask_name}[0:{p_count}, 0:1])"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{masked_offsets} = nl.ndarray([{p_count}, 1], "
                            "nl.uint32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.memset({masked_offsets}, value={flat_extent})"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy_predicated(dst={masked_offsets}, "
                            f"src={vec_offset}, predicate={pred_col})"
                        )
                    )
                    vec_offset = masked_offsets

            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={name}.reshape([{flat_extent}, 1]).ap("
                    f"pattern=[[1, {p_count}], [1, {f_count}]], "
                    f"vector_offset={vec_offset}, indirect_dim=0), "
                    f"src={value_name}, oob_mode=nisa.oob_mode.skip)"
                )
            )
            return True

        if _try_emit_flat_index_store():
            return

        # For 1D output tensors stored into 2D return buffers:
        # Check the HBM buffer layout to decide slicing.
        if use_nki_return and tensor.dim() == 1:
            tensor_id = id(tensor)
            buf_info = device_fn._nki_return_buffers.get(tensor_id, {})
            val_sbuf_shape = device_fn._nki_sbuf_shapes.get(value_name)
            use_partition_layout = (
                val_sbuf_shape is not None
                and len(val_sbuf_shape) >= 2
                and val_sbuf_shape[0] > 1
                and val_sbuf_shape[1] == 1
            )
            if use_partition_layout:
                # [N, 1] HBM: partition-axis slicing, append "0:1" for free dim
                slice_str = f"{slice_str}, 0:1"
            else:
                # [1, N] HBM: free-axis slicing, prepend "0:1" for partition dim
                slice_str = f"0:1, {slice_str}"
        numel = tensor.numel()

        # Layout mismatch guard for 2D outputs: if the SBUF value is stored
        # as [N, 1] (partition-major, from reduction accumulator) but the
        # HBM destination is a [1, N] slice (e.g. out[offset_b:+1, off:+N]),
        # emit a cheap nc_transpose to align shapes before dma_copy.
        value_name_for_transpose = value_name if isinstance(value_name, str) else None
        val_sbuf_shape = (
            device_fn._nki_sbuf_shapes.get(value_name_for_transpose)
            if value_name_for_transpose is not None
            else None
        )
        # Parse slice_str to figure out the HBM slice width per dim
        def _slice_width(part: str) -> int | None:
            # Handles "a:a+N", "N", "a:b"
            if ":" not in part:
                try:
                    return int(part)
                except (TypeError, ValueError):
                    return None
            lo, hi = part.split(":", 1)
            # Look for + constant pattern
            if "+" in hi:
                suffix = hi.split("+")[-1].strip()
                try:
                    return int(suffix)
                except (TypeError, ValueError):
                    return None
            try:
                return int(hi) - int(lo)
            except (TypeError, ValueError):
                return None

        def _store_slice_info(
            part: str, dim_idx: int
        ) -> tuple[str, str, int, str] | None:
            if isinstance(part, (IndirectAP, DynamicAP)):
                return None
            if dim_idx >= len(hbm_dim_size_strs_s) or ":" not in part:
                return None
            start, end = part.split(":", 1)
            start = start.strip()
            end = end.strip()
            width = _slice_width(part)
            if width is None or width <= 0:
                return None
            dim_size_str = hbm_dim_size_strs_s[dim_idx]
            return start, end, width, dim_size_str

        def _store_bounds_guard(parts: list[str]) -> str | None:
            _dyn_begin_vars_s = getattr(
                getattr(state, "device_function", None), "_nki_dyn_begin_offset_vars", set()
            )
            checks: list[str] = []
            for dim_idx, part in enumerate(parts):
                info = _store_slice_info(part, dim_idx)
                if info is None:
                    return None
                start, end, _width, dim_size_str = info
                if any(v in start for v in _dyn_begin_vars_s):
                    # Skip all guards for dyn-begin dims (VirtualRegister arithmetic fails)
                    continue
                checks.append(f"({start}) >= 0")
                checks.append(f"({end}) <= {dim_size_str}")
            return " and ".join(checks) if checks else None

        def _single_tail_store_cases(
            dst_base: str, parts: list[str], src_name: str
        ) -> list[ast.If]:
            import itertools

            _dyn_begin_vars_ts = getattr(
                getattr(state, "device_function", None), "_nki_dyn_begin_offset_vars", set()
            )
            cases: list[ast.If] = []
            infos = [_store_slice_info(part, i) for i, part in enumerate(parts)]
            if any(info is None for info in infos):
                return cases

            FULL, TAIL_OVERFLOW = "full", "tail_overflow"

            for dim_states in itertools.product([FULL, TAIL_OVERFLOW], repeat=len(infos)):
                if all(s == FULL for s in dim_states):
                    continue  # handled by the main fast path

                checks: list[str] = []
                dst_parts: list[str] = []
                src_parts: list[str] = []

                for dim_idx, (info, state_d) in enumerate(zip(infos, dim_states)):
                    assert info is not None
                    dim_start, dim_end, dim_width, dim_size = info
                    _has_dyn_begin_s = any(v in dim_start for v in _dyn_begin_vars_ts)

                    if _has_dyn_begin_s:
                        # dynamic_range guarantees offset in [begin, end); TAIL_OVERFLOW impossible.
                        if state_d == FULL:
                            dst_parts.append(parts[dim_idx])
                            src_parts.append(f"0:{dim_width}")
                        else:  # TAIL_OVERFLOW — impossible, skip this case
                            valid = False
                            break
                    elif state_d == FULL:
                        checks.append(f"({dim_start}) >= 0")
                        checks.append(f"({dim_end}) <= {dim_size}")
                        dst_parts.append(parts[dim_idx])
                        src_parts.append(f"0:{dim_width}")
                    else:  # TAIL_OVERFLOW
                        checks.append(f"({dim_start}) >= 0")
                        checks.append(f"({dim_start}) < {dim_size}")
                        checks.append(f"({dim_end}) > {dim_size}")
                        dst_parts.append(f"{dim_start}:{dim_size}")
                        src_parts.append(f"0:{dim_size} - ({dim_start})")

                cases.append(
                    create(
                        ast.If,
                        test=expr_from_string(" and ".join(checks)),
                        body=[
                            statement_from_string(
                                f"nisa.dma_copy(dst={dst_base}[{', '.join(dst_parts)}], "
                                f"src={src_name}[{', '.join(src_parts)}])"
                            )
                        ],
                        orelse=[],
                    )
                )
            return cases

        def _emit_direct_store(dst_base: str, parts: list[str], src_value: ast.AST) -> None:
            src_name = ast.unparse(src_value)
            # EXPERIMENTAL (HELION_NKI_TILESTREAM): replace the fast-path store +
            # 2^N TAIL_OVERFLOW enumeration with ONE TensorView-clamped store. The
            # HBM dst slice is clamped via TensorView.slice (min() on end); the SBUF
            # src is sliced to the clamped extent. Only the clean contiguous case
            # (every part a plain "start:end" string).
            if (
                env.backend.use_tilestream
                and parts
                and all(isinstance(p, str) and ":" in p for p in parts)
                and all(_store_slice_info(p, i) is not None for i, p in enumerate(parts))
            ):
                _dstv = device_fn.new_var("_ts_dstv")
                _tv_expr = f"_nkitv({dst_base})"
                for _d, _p in enumerate(parts):
                    _s, _e = _p.split(":", 1)
                    _tv_expr += f".slice({_d}, {_s.strip()}, {_e.strip()})"
                _src_idx = ", ".join(f"0:{_dstv}.shape[{_d}]" for _d in range(len(parts)))
                state.codegen.add_statement(statement_from_string(f"{_dstv} = {_tv_expr}"))
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.dma_copy(dst={_dstv}.get_view(), src={{value}}[{_src_idx}])",
                        value=src_value,
                    )
                )
                return
            full_stmt = statement_from_string(
                f"nisa.dma_copy(dst={dst_base}[{', '.join(parts)}], src={{value}})",
                value=src_value,
            )
            guard = _store_bounds_guard(parts)
            if guard is None:
                state.codegen.add_statement(full_stmt)
                return
            state.codegen.add_statement(
                create(
                    ast.If,
                    test=expr_from_string(guard),
                    body=[full_stmt],
                    orelse=_single_tail_store_cases(dst_base, parts, src_name),
                )
            )

        transposed_value = None
        if (
            use_nki_return
            and val_sbuf_shape is not None
            and len(val_sbuf_shape) == 2
        ):
            # Parse adjusted_slice_parts to get HBM slice widths.
            parts = [p.strip() for p in slice_str.split(",")]
            if len(parts) == 2:
                w0 = _slice_width(parts[0])
                w1 = _slice_width(parts[1])
                transpose_shape: list[int] | None = None
                if (
                    tensor.dim() == 2
                    and val_sbuf_shape[0] > 1
                    and val_sbuf_shape[1] == 1
                    and w0 == 1
                    and w1 == val_sbuf_shape[0]
                ):
                    # HBM slice is [1, N], SBUF is [N, 1].
                    transpose_shape = [1, val_sbuf_shape[0]]
                elif (
                    flattened_high_rank_store
                    and val_sbuf_shape[0] == 1
                    and val_sbuf_shape[1] > 1
                    and w0 == val_sbuf_shape[1]
                    and w1 == 1
                ):
                    # Flattened high-rank HBM slice is [N, 1], SBUF is [1, N].
                    transpose_shape = [val_sbuf_shape[1], 1]
                if transpose_shape is not None:
                    from ..ast_extension import expr_from_string as _efrom
                    dtype_str = env.backend.dtype_str(tensor.dtype)
                    transposed_var = device_fn.new_var("_nki_store_tr", dce=True)
                    device_fn._nki_sbuf_shapes[transposed_var] = transpose_shape
                    tr_psum = device_fn.new_var("_nki_store_tr_psum", dce=True)
                    transpose_shape_str = ", ".join(str(d) for d in transpose_shape)
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{tr_psum} = nl.ndarray([{transpose_shape_str}], "
                            f"{dtype_str}, buffer=nl.psum)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.nc_transpose(dst={tr_psum}, data={value_name_for_transpose})"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{transposed_var} = nl.ndarray([{transpose_shape_str}], "
                            f"{dtype_str}, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={transposed_var}, src={tr_psum})"
                        )
                    )
                    transposed_value = _efrom(transposed_var)

        final_value = transposed_value if transposed_value is not None else value

        if numel == 1:
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={name}, src={{value}})", value=final_value
                )
            )
        else:
            # If slice contains dynamic sentinels, rewrite to .ap()
            _has_dyn_store = any(isinstance(p, DynamicAP) for p in slice_parts)
            _has_row_store = any(isinstance(p, IndirectAP) for p in slice_parts)
            if _has_row_store:
                _row_part = next(p for p in slice_parts if isinstance(p, IndirectAP))
                _vec_offset = _row_part.vec_var
                _vec_shape = device_fn._nki_sbuf_shapes.get(_vec_offset)
                if _vec_shape is None or len(_vec_shape) < 1:
                    raise NotImplementedError(
                        f"Unknown row-scatter vector offset shape for {_vec_offset}"
                    )
                _p_count = int(_vec_shape[0])
                _tensor_shape = list(hbm_dim_size_strs_s)
                if len(_tensor_shape) != 2:
                    raise NotImplementedError(
                        f"Row scatter requires a 2D tensor, got {list(tensor.shape)}"
                    )
                if isinstance(_tensor_shape[1], int):
                    _f_total: int | str = int(_tensor_shape[1])
                elif isinstance(_tensor_shape[1], torch.SymInt):
                    try:
                        _f_total = int(env.size_hint(_tensor_shape[1]))
                    except (TypeError, ValueError):
                        _f_total = state.sympy_expr(_tensor_shape[1]._sympy_())
                else:
                    _f_total = str(_tensor_shape[1])
                _free_slice = next((p for p in slice_parts if isinstance(p, str) and ":" in p), None)
                _free_part = _free_slice if _free_slice is not None else f"0:{_f_total}"
                if ":" in _free_part:
                    _fs, _fe = _free_part.split(":", 1)
                    _fs = _fs.strip()
                    _plus = _fe.find("+")
                    if _plus >= 0:
                        try:
                            _fc = int(_fe[_plus + 1 :].strip())
                        except ValueError:
                            _fc = device_fn._nki_sbuf_shapes.get(value_name, [1, 1])[1]
                    else:
                        try:
                            _fc = int(_fe.strip()) - int(_fs)
                        except (TypeError, ValueError):
                            _fc = device_fn._nki_sbuf_shapes.get(value_name, [1, 1])[1]
                else:
                    _fs = "0"
                    _fc = device_fn._nki_sbuf_shapes.get(value_name, [1, 1])[1]
                _pat = f"[[{_f_total}, {_p_count}], [1, {_fc}]]"
                _dst_expr = (
                    f"{name}.ap(pattern={_pat}, offset={_fs}, "
                    f"vector_offset={_vec_offset}, indirect_dim=0)"
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.dma_copy(dst={_dst_expr}, src={{value}}, "
                        "oob_mode=nisa.oob_mode.skip)",
                        value=final_value,
                    )
                )
            elif _has_dyn_store:
                _dst_parts = [
                    p if isinstance(p, str) else "__SENTINEL__"
                    for p in slice_parts
                ]
                # Inline the same builder logic as _build_hbm_src (can't use it
                # here because it closes over `tensor` / `_resolve_dim` in load).
                _dyn_idx = None
                _dyn_counter = None
                _dyn_size = 0
                for _i, _p in enumerate(slice_parts):
                    if isinstance(_p, DynamicAP):
                        assert _dyn_idx is None, "multi-dyn store not supported"
                        _dyn_idx = _i
                        _dyn_counter = _p.counter
                        _dyn_size = _p.block_size
                _t_shape = list(tensor.shape)
                if len(_t_shape) > 2:
                    while len(_t_shape) > 2 and _t_shape[0] == 1:
                        _t_shape = _t_shape[1:]
                    if len(_t_shape) > 2:
                        _flat = 1
                        for _d in _t_shape[:-1]:
                            _flat *= int(_d) if isinstance(_d, int) else int(env.size_hint(_d))
                        _t_shape = [_flat, _t_shape[-1]]
                _t_shape = [int(_d) if isinstance(_d, int) else int(env.size_hint(_d)) for _d in _t_shape]
                if len(_t_shape) == 2:
                    _P, _F = _t_shape
                    # If the return buffer was padded, use the padded free-dim as the
                    # stride in the AP pattern so writes land at the correct addresses.
                    _padded_F = getattr(device_fn, "_nki_return_buf_free_dim", None)
                    _orig_F = _F
                    if _padded_F is not None and name == getattr(device_fn, "_nki_return_buffer_name", None):
                        _F = _padded_F
                    if _dyn_idx == 1:
                        # Emit the static prefill (y → nki_return_buf) once, before
                        # the dynamic loop.  The prefill uses static column offsets so
                        # every tile is fully in-bounds in the padded buffer.
                        _dyn_loops_st = getattr(device_fn, "_nki_dyn_loops", {})
                        for _bid_st, _dl_st in _dyn_loops_st.items():
                            if (
                                _dl_st.get("needs_prefill")
                                and _dl_st.get("counter") == _dyn_counter
                                and "prefill_emitted" not in _dl_st
                                and "y_src" in _dl_st
                            ):
                                _dl_st["prefill_emitted"] = True
                                _y_src = _dl_st["y_src"]
                                _stk = state.codegen.statements_stack
                                _pstmts: list = _stk[-2] if len(_stk) >= 2 else _stk[-1]
                                _part_part_pre = _dst_parts[0]
                                if ":" in _part_part_pre:
                                    _pps, _ppe = _part_part_pre.split(":", 1)
                                    _pps = _pps.strip()
                                    _pplus = _ppe.find("+")
                                    _ppc = int(_ppe[_pplus + 1:].strip()) if _pplus >= 0 else 1
                                else:
                                    _pps, _ppc = "0", _P
                                _pre_dtype = env.backend.dtype_str(tensor.dtype)
                                # Full tiles using the ORIGINAL y width (y is unpadded).
                                _n_full = _orig_F // _dyn_size
                                _rem    = _orig_F % _dyn_size
                                try:
                                    _pps_int = int(_pps)
                                    _base_off = _pps_int * _F   # padded stride for out
                                    _base_off_y = _pps_int * _orig_F  # orig stride for y
                                    _row_guard = f"{_pps} >= 0 and {_pps} + {_ppc} <= {_P}"
                                except ValueError:
                                    _base_off = f"({_pps}) * {_F}"
                                    _base_off_y = f"({_pps}) * {_orig_F}"
                                    _row_guard = None
                                from ..ast_extension import create as _ast_create
                                import ast as _ast_pre
                                if _n_full > 0:
                                    _pv = device_fn.new_var("_pre_c")
                                    _pt = device_fn.new_var("_pre_tile")
                                    _pcs = device_fn.new_var("_pre_c_sbuf")
                                    _ppat_y = f"[[{_orig_F}, {_ppc}], [1, {_dyn_size}]]"
                                    _ppat_o = f"[[{_F}, {_ppc}], [1, {_dyn_size}]]"
                                    _pb: list = [
                                        statement_from_string(f"{_pt} = nl.ndarray([{_ppc}, {_dyn_size}], {_pre_dtype}, buffer=nl.sbuf)"),
                                        statement_from_string(f"nisa.memset({_pt}, value=0)"),
                                        statement_from_string(f"{_pcs} = nl.ndarray([1, 1], nl.int32, buffer=nl.sbuf)"),
                                        statement_from_string(f"nisa.memset({_pcs}, value={_pv})"),
                                    ]
                                    _src_ap = f"{_y_src}.ap(pattern={_ppat_y}, scalar_offset={_pcs}, indirect_dim=1, offset={_base_off_y})"
                                    _dst_ap = f"{name}.ap(pattern={_ppat_o}, scalar_offset={_pcs}, indirect_dim=1, offset={_base_off})"
                                    if _row_guard:
                                        _pb.append(statement_from_string(f"if {_row_guard}: nisa.dma_copy(dst={_pt}, src={_src_ap}, oob_mode=nisa.oob_mode.skip)"))
                                        _pb.append(statement_from_string(f"if {_row_guard}: nisa.dma_copy(dst={_dst_ap}, src={_pt}, oob_mode=nisa.oob_mode.skip)"))
                                    else:
                                        _pb.append(statement_from_string(f"nisa.dma_copy(dst={_pt}, src={_src_ap}, oob_mode=nisa.oob_mode.skip)"))
                                        _pb.append(statement_from_string(f"nisa.dma_copy(dst={_dst_ap}, src={_pt}, oob_mode=nisa.oob_mode.skip)"))
                                    _pstmts.append(_ast_create(
                                        _ast_pre.For,
                                        target=_ast_create(_ast_pre.Name, id=_pv, ctx=_ast_pre.Store()),
                                        iter=expr_from_string(f"nl.affine_range(0, {_n_full * _dyn_size}, {_dyn_size})"),
                                        body=_pb, orelse=[], type_comment=None,
                                    ))
                                if _rem > 0:
                                    _rem_start = _n_full * _dyn_size
                                    _rt = device_fn.new_var("_pre_tail_tile")
                                    _rc = device_fn.new_var("_pre_tail_ctr")
                                    _rpat_y = f"[[{_orig_F}, {_ppc}], [1, {_rem}]]"
                                    _rpat_o = f"[[{_F}, {_ppc}], [1, {_rem}]]"
                                    _rs: list = [
                                        statement_from_string(f"{_rt} = nl.ndarray([{_ppc}, {_rem}], {_pre_dtype}, buffer=nl.sbuf)"),
                                        statement_from_string(f"nisa.memset({_rt}, value=0)"),
                                        statement_from_string(f"{_rc} = nl.ndarray([1, 1], nl.int32, buffer=nl.sbuf)"),
                                        statement_from_string(f"nisa.memset({_rc}, value={_rem_start})"),
                                    ]
                                    _rsrc = f"{_y_src}.ap(pattern={_rpat_y}, scalar_offset={_rc}, indirect_dim=1, offset={_base_off_y})"
                                    _rdst = f"{name}.ap(pattern={_rpat_o}, scalar_offset={_rc}, indirect_dim=1, offset={_base_off})"
                                    if _row_guard:
                                        _rs.append(statement_from_string(f"if {_row_guard}: nisa.dma_copy(dst={_rt}, src={_rsrc}, oob_mode=nisa.oob_mode.skip)"))
                                        _rs.append(statement_from_string(f"if {_row_guard}: nisa.dma_copy(dst={_rdst}, src={_rt}, oob_mode=nisa.oob_mode.skip)"))
                                    else:
                                        _rs.append(statement_from_string(f"nisa.dma_copy(dst={_rt}, src={_rsrc}, oob_mode=nisa.oob_mode.skip)"))
                                        _rs.append(statement_from_string(f"nisa.dma_copy(dst={_rdst}, src={_rt}, oob_mode=nisa.oob_mode.skip)"))
                                    _pstmts.extend(_rs)
                                break
                        _part_part = _dst_parts[0]
                        if ":" in _part_part:
                            _ps, _pe = _part_part.split(":", 1)
                            _ps = _ps.strip()
                            _plus = _pe.find("+")
                            if _plus >= 0:
                                _pc = int(_pe[_plus+1:].strip())
                            else:
                                _pc = 1
                            _pat = f"[[{_F}, {_pc}], [1, {_dyn_size}]]"
                            try:
                                _p_off_int = int(_ps)
                                _dst_expr = f"{name}.ap(pattern={_pat}, scalar_offset={_dyn_counter}, indirect_dim=1, offset={_p_off_int * _F})"
                            except ValueError:
                                _dst_expr = f"{name}.ap(pattern={_pat}, scalar_offset={_dyn_counter}, indirect_dim=1, offset=({_ps}) * {_F})"
                    elif _dyn_idx == 0:
                        _free_part = _dst_parts[1]
                        if ":" in _free_part:
                            _fs, _fe = _free_part.split(":", 1)
                            _fs = _fs.strip()
                            _plus = _fe.find("+")
                            if _plus >= 0:
                                _fc = int(_fe[_plus+1:].strip())
                            else:
                                _fc = 1
                            _pat = f"[[{_F}, {_dyn_size}], [1, {_fc}]]"
                            try:
                                _f_off_int = int(_fs)
                                _dst_expr = f"{name}.ap(pattern={_pat}, scalar_offset={_dyn_counter}, indirect_dim=0, offset={_f_off_int})"
                            except ValueError:
                                _dst_expr = f"{name}.ap(pattern={_pat}, scalar_offset={_dyn_counter}, indirect_dim=0)"
                    else:
                        _dst_expr = f"{name}[{slice_str}]"
                else:
                    _dst_expr = f"{name}[{slice_str}]"
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.dma_copy(dst={_dst_expr}, src={{value}}, "
                        "oob_mode=nisa.oob_mode.skip)",
                        value=final_value,
                    )
                )
            else:
                _emit_direct_store(
                    name, [p.strip() for p in slice_str.split(",")], final_value
                )
