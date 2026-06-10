"""NKI indirect gather/scatter codegen helpers.

Moved verbatim from ``helion/language/memory_ops.py`` as part of the NKI
subpackage refactor. These build the index-vector machinery and the
indirect gather / row-scatter copies.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ...language._nki_dim_access import IndirectAP
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .sbuf import _nki_lookup_sbuf_shape_dtype

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _nki_indirect_gather(
    fx_node: object,
    state: "CodegenState",
    env: "CompileEnvironment",
    tensor: torch.Tensor,
    slice_parts_so_far: list[str],
) -> str | None:
    """Detect ``base_per_partition + tile_index`` subscript patterns and emit
    a `.ap(pattern=..., vector_offset=base_tile, indirect_dim=0)` that gathers
    a contiguous slice per partition.

    Matches patterns like ``starts[:, None] + tile.index[None, :]`` where
    ``starts`` has been loaded into an SBUF tile. We extract ``starts`` via
    the FX graph, find its emitted SBUF var name through state's code map,
    and use it as the .ap() vector_offset.

    Returns the per-dim slice string OR None if the pattern doesn't match.
    Currently only supports 1D target tensor (after kernel-entry reshape
    to [1, N]): the pattern emits a single-dim slice that replaces the
    trailing dim.
    """
    if not isinstance(fx_node, torch.fx.Node):
        return None
    target = getattr(fx_node, "target", None)
    target_name = str(target) if target else ""
    if not ("add.Tensor" in target_name or target is torch.ops.aten.add.Tensor):
        return None
    args = fx_node.args
    if len(args) != 2:
        return None
    # One arg should be tile.index (or tile.index with newaxis), the other
    # should be a tile-like [P] SBUF variable.
    # FX shape check: args[0].meta['val'] is [P, 1], args[1].meta['val'] is [1, F]
    # (or permuted). Result is [P, F].
    lhs, rhs = args
    if not (isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)):
        return None
    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if not (isinstance(lhs_val, torch.Tensor) and isinstance(rhs_val, torch.Tensor)):
        return None
    # Detect broadcast pattern: lhs [P, 1], rhs [1, F]
    if lhs_val.ndim != 2 or rhs_val.ndim != 2:
        return None
    # Resolve symints
    import sympy as _sp_ig
    _bs_subs_ig: dict[_sp_ig.Symbol, int] = {}
    if state.config is not None:
        for _bs in env.block_sizes:
            _cfg = _bs.from_config(state.config)
            if isinstance(_cfg, int):
                _bs_subs_ig[_bs.symbol()] = _cfg

    def _res(d: object) -> int:
        if isinstance(d, int):
            return d
        if isinstance(d, torch.SymInt):
            try:
                return int(d._sympy_().subs(_bs_subs_ig))
            except Exception:
                return int(env.size_hint(d))
        return 1

    lhs_p, lhs_f = _res(lhs_val.shape[0]), _res(lhs_val.shape[1])
    rhs_p, rhs_f = _res(rhs_val.shape[0]), _res(rhs_val.shape[1])
    # Expect lhs [P, 1] and rhs [1, F]
    if lhs_f != 1 or rhs_p != 1:
        # Try swap
        if rhs_f == 1 and lhs_p == 1:
            lhs, rhs = rhs, lhs
            lhs_val, rhs_val = rhs_val, lhs_val
            lhs_p, lhs_f = _res(lhs_val.shape[0]), _res(lhs_val.shape[1])
            rhs_p, rhs_f = _res(rhs_val.shape[0]), _res(rhs_val.shape[1])
        else:
            return None
    P = lhs_p
    F = rhs_f
    if P <= 0 or F <= 0:
        return None
    # Need to find SBUF var names for lhs (base) and rhs (tile_index).
    # GraphInterpreter records the final lowered AST for used FX nodes on the
    # active GenerateAST instance.  That gives us the emitted SBUF variable
    # name, which is the key used by the NKI shape/dtype registries below.
    lhs_ast = state.codegen.ast_for_fx_node(lhs)
    if not isinstance(lhs_ast, ast.AST):
        return None
    lhs_name = ast.unparse(lhs_ast)
    # lhs_name should be the SBUF var containing base [P, 1] ints (or [1, P]).
    # For ``.ap(vector_offset=base_tile, indirect_dim=0)`` the tile must be
    # uint32 [P, 1].
    # Emit a cast to uint32 + transpose if needed.
    lhs_shape_lookup = state.device_function._nki_sbuf_shapes.get(lhs_name)
    if lhs_shape_lookup is None:
        # Try stripping _copy suffixes
        _lk = lhs_name
        while "_copy" in _lk:
            _lk = _lk[:_lk.rfind("_copy")]
            lhs_shape_lookup = state.device_function._nki_sbuf_shapes.get(_lk)
            if lhs_shape_lookup is not None:
                break
    if lhs_shape_lookup is None:
        return None
    # Lay out as [P, 1] (partition axis has P elements)
    # If lhs_shape_lookup is [1, P], need nc_transpose to [P, 1]
    from ..ast_extension import statement_from_string
    device_fn = state.device_function
    if lhs_shape_lookup == [P, 1]:
        vec_offset_var = lhs_name
    elif lhs_shape_lookup == [1, P]:
        # Transpose [1, P] -> [P, 1]
        tr_psum = device_fn.new_var("_ig_tr_psum", dce=True)
        tr_sbuf = device_fn.new_var("_ig_tr_sbuf", dce=True)
        # Use int32 dtype (or whatever lhs has)
        _dt = device_fn._nki_sbuf_dtypes.get(lhs_name, "nl.int32")
        # Transpose [1, P] int → [P, 1] uint32 via float32 round-trip.
        # activation(nl.copy) converts int32 → float32 numerically (not bitwise).
        # nc_transpose then tensor_scalar(add 0.0) converts float32 → uint32.
        _dt = device_fn._nki_sbuf_dtypes.get(lhs_name, "nl.int32")
        if _dt in ("nl.int32", "nl.int16", "nl.uint32", "nl.uint16"):
            cast_in = device_fn.new_var("_ig_cast_in", dce=True)
            state.codegen.add_statement(statement_from_string(
                f"{cast_in} = nl.ndarray([1, {P}], nl.float32, buffer=nl.sbuf)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.activation(dst={cast_in}, op=nl.copy, data={lhs_name})"
            ))
            state.codegen.add_statement(statement_from_string(
                f"{tr_psum} = nl.ndarray([{P}, 1], nl.float32, buffer=nl.psum)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.nc_transpose(dst={tr_psum}, data={cast_in})"
            ))
        else:
            state.codegen.add_statement(statement_from_string(
                f"{tr_psum} = nl.ndarray([{P}, 1], {_dt}, buffer=nl.psum)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.nc_transpose(dst={tr_psum}, data={lhs_name})"
            ))
        state.codegen.add_statement(statement_from_string(
            f"{tr_sbuf} = nl.ndarray([{P}, 1], nl.uint32, buffer=nl.sbuf)"
        ))
        state.codegen.add_statement(statement_from_string(
            f"nisa.tensor_scalar(dst={tr_sbuf}, data={tr_psum}, op0=nl.add, operand0=0.0)"
        ))
        vec_offset_var = tr_sbuf
    else:
        return None
    # Get tile.index block_id from rhs to find the counter for the dynamic loop
    # rhs_val shape [1, F] - find the block that has size F
    _tile_bid = None
    for _bid in range(len(env.block_sizes)):
        _bs = env.block_sizes[_bid]
        try:
            _bs_val = _bs.from_config_assert(state.config)
            if int(_bs_val) == F and _bid in state.codegen.active_device_loops:
                _tile_bid = _bid
                break
        except Exception:
            pass
    if _tile_bid is None:
        return None
    # If this tile is dynamic, include counter in vector_offset
    _dyn_loops = getattr(device_fn, "_nki_dyn_loops", {})
    # Build pattern for .ap()
    # x_data is 1D reshaped to [1, N]. We want per-partition gather of F
    # contiguous elements starting at vec_offset.
    # Pattern: [[N, P], [1, F]] with vector_offset over the partition dim.
    _tensor_shape = list(tensor.shape)
    if len(_tensor_shape) == 1:
        N_total = int(_tensor_shape[0])
        leading_stride = 0
    elif len(_tensor_shape) == 2:
        N_total = int(_tensor_shape[1])
        leading_stride = N_total
    else:
        return None
    pattern = f"[[{leading_stride}, {P}], [1, {F}]]"
    # For dynamic loops, add counter to vector_offset (starts + counter)
    if _tile_bid in _dyn_loops:
        _counter = _dyn_loops[_tile_bid]["counter"]
        # Compute vec_offset_combined = vec_offset_var + counter (per partition)
        combined_var = device_fn.new_var("_ig_base_plus_counter", dce=True)
        state.codegen.add_statement(statement_from_string(
            f"{combined_var} = nl.ndarray([{P}, 1], nl.uint32, buffer=nl.sbuf)"
        ))
        # Copy vec_offset to combined_var
        state.codegen.add_statement(statement_from_string(
            f"nisa.tensor_copy(dst={combined_var}, src={vec_offset_var})"
        ))
        counter_u32 = device_fn.new_var("_ig_counter_u32", dce=True)
        counter_bcast = device_fn.new_var("_ig_counter_bcast", dce=True)
        device_fn._nki_sbuf_shapes[counter_u32] = [1, 1]
        device_fn._nki_sbuf_dtypes[counter_u32] = "nl.uint32"
        device_fn._nki_sbuf_shapes[counter_bcast] = [P, 1]
        device_fn._nki_sbuf_dtypes[counter_bcast] = "nl.uint32"
        state.codegen.add_statement(statement_from_string(
            f"{counter_u32} = nl.ndarray([1, 1], nl.uint32, buffer=nl.sbuf)"
        ))
        state.codegen.add_statement(statement_from_string(
            f"nisa.tensor_copy(dst={counter_u32}, src={_counter})"
        ))
        state.codegen.add_statement(statement_from_string(
            f"{counter_bcast} = nl.broadcast_to({counter_u32}, shape=({P}, 1))"
        ))
        state.codegen.add_statement(statement_from_string(
            f"nisa.tensor_tensor(dst={combined_var}, data1={combined_var}, data2={counter_bcast}, op=nl.add)"
        ))
        vec_offset_var = combined_var
    # Emit an .ap() expression inline as the slice_part
    return IndirectAP(vec_var=vec_offset_var, p_count=P, pattern=pattern)


def _nki_as_uint32_p1_vector(
    state: "CodegenState", name: str, p_count: int
) -> str | None:
    """Return an SBUF ``[P, 1]`` uint32 vector-offset tile for ``name``.

    Helion tile indices usually materialize as row vectors ``[1, P]`` while
    NKI vector indirection expects one row id per partition, ``[P, 1]``.
    """

    from ..ast_extension import statement_from_string

    device_fn = state.device_function
    shape, dtype = _nki_lookup_sbuf_shape_dtype(state, name)
    if shape is None:
        return None

    int_dtypes = {"nl.int32", "nl.int16", "nl.int8", "nl.uint16", "nl.uint8"}
    if shape == [p_count, 1]:
        if dtype == "nl.uint32":
            return name
        cast_var = device_fn.new_var("_ig_u32", dce=True)
        device_fn._nki_sbuf_shapes[cast_var] = [p_count, 1]
        device_fn._nki_sbuf_dtypes[cast_var] = "nl.uint32"
        state.codegen.add_statement(
            statement_from_string(
                f"{cast_var} = nl.ndarray([{p_count}, 1], nl.uint32, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.tensor_copy(dst={cast_var}, src={name})")
        )
        return cast_var

    if shape != [1, p_count]:
        return None

    tr_sbuf = device_fn.new_var("_ig_tr_sbuf", dce=True)
    device_fn._nki_sbuf_shapes[tr_sbuf] = [p_count, 1]
    device_fn._nki_sbuf_dtypes[tr_sbuf] = "nl.uint32"
    state.codegen.add_statement(
        statement_from_string(
            f"{tr_sbuf} = nl.ndarray([{p_count}, 1], nl.uint32, buffer=nl.sbuf)"
        )
    )

    # nc_transpose is capped at 128 partitions, so a [1, p_count] -> [p_count, 1]
    # transpose must be tiled into <=128-row chunks when p_count > 128 (e.g. a
    # gather over a chunk_size=256 tile). Each chunk transposes columns
    # [c0:c0+w] of the row vector into rows [c0:c0+w] of the destination.
    NKI_PARTITION_MAX = 128
    needs_cast = dtype in int_dtypes or dtype == "nl.uint32"
    for _c0 in range(0, p_count, NKI_PARTITION_MAX):
        _w = min(NKI_PARTITION_MAX, p_count - _c0)
        tr_psum = device_fn.new_var("_ig_tr_psum", dce=True)
        if needs_cast:
            cast_in = device_fn.new_var("_ig_cast_in", dce=True)
            device_fn._nki_sbuf_shapes[cast_in] = [1, _w]
            device_fn._nki_sbuf_dtypes[cast_in] = "nl.float32"
            state.codegen.add_statement(
                statement_from_string(
                    f"{cast_in} = nl.ndarray([1, {_w}], nl.float32, buffer=nl.sbuf)"
                )
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.activation(dst={cast_in}, op=nl.copy, "
                    f"data={name}[0:1, {_c0}:{_c0 + _w}])"
                )
            )
            tr_src = cast_in
            tr_dtype = "nl.float32"
        else:
            tr_src = f"{name}[0:1, {_c0}:{_c0 + _w}]"
            tr_dtype = dtype
        state.codegen.add_statement(
            statement_from_string(
                f"{tr_psum} = nl.ndarray([{_w}, 1], {tr_dtype}, buffer=nl.psum)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.nc_transpose(dst={tr_psum}, data={tr_src})")
        )
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.tensor_scalar(dst={tr_sbuf}[{_c0}:{_c0 + _w}, 0:1], "
                f"data={tr_psum}, op0=nl.add, operand0=0.0)"
            )
        )
    return tr_sbuf


def _nki_row_index_gather(
    fx_node: object,
    state: "CodegenState",
    p_count: int | None,
) -> str | None:
    if not isinstance(fx_node, torch.fx.Node):
        return None
    value = fx_node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim not in (1, 2):
        return None

    target = getattr(fx_node, "target", None)
    target_name = str(target) if target else ""
    if (
        p_count is not None
        and fx_node.op == "call_function"
        and (
            "add.Tensor" in target_name
            or "add.Scalar" in target_name
            or target is torch.ops.aten.add.Tensor
            or target is torch.ops.aten.add.Scalar
        )
        and len(fx_node.args) >= 2
    ):
        lhs_node, rhs_node = fx_node.args[:2]

        def _target_contains(node: object, *needles: str) -> bool:
            if not isinstance(node, torch.fx.Node):
                return False
            node_target = getattr(node, "target", None)
            node_target_name = str(node_target) if node_target else ""
            return any(needle in node_target_name for needle in needles)

        def _arg_name(node: object) -> str | None:
            if isinstance(node, torch.fx.Node):
                node_ast = state.codegen.ast_for_fx_node(node)
                if not isinstance(node_ast, ast.AST):
                    return None
                return ast.unparse(node_ast)
            if isinstance(node, (int, float, bool)):
                return repr(node)
            return None

        def _row_index_shape(name: str) -> list[int] | None:
            shape, _ = _nki_lookup_sbuf_shape_dtype(state, name)
            if shape in ([1, p_count], [p_count, 1]):
                return shape
            # Fall back: check if name is an active loop index variable (iota)
            # whose shape hasn't been registered yet (outer loop pre-emission).
            if shape is None:
                from ..compile_environment import CompileEnvironment as _CE
                _env = _CE.current()
                for bid in state.codegen.active_device_loops:
                    try:
                        if state.codegen.index_var(bid) == name:
                            bs = int(_env.block_sizes[bid].from_config_assert(state.config))
                            if bs == p_count:
                                return [1, p_count]
                    except Exception:
                        pass
            return None

        lhs_name_simple = _arg_name(lhs_node)
        rhs_name_simple = _arg_name(rhs_node)
        if lhs_name_simple is not None and rhs_name_simple is not None:
            lhs_shape_simple = _row_index_shape(lhs_name_simple)
            rhs_shape_simple = _row_index_shape(rhs_name_simple)
            if (lhs_shape_simple is None) != (rhs_shape_simple is None):
                index_name = (
                    lhs_name_simple
                    if lhs_shape_simple is not None
                    else rhs_name_simple
                )
                scalar_name = (
                    rhs_name_simple
                    if lhs_shape_simple is not None
                    else lhs_name_simple
                )
                index_shape = lhs_shape_simple or rhs_shape_simple
                assert index_shape is not None
                from ..ast_extension import statement_from_string

                device_fn = state.device_function
                row_index = device_fn.new_var("_row_idx_add", dce=True)
                device_fn._nki_sbuf_shapes[row_index] = list(index_shape)
                device_fn._nki_sbuf_dtypes[row_index] = "nl.int32"
                state.codegen.add_statement(
                    statement_from_string(
                        f"{row_index} = nl.ndarray([{index_shape[0]}, "
                        f"{index_shape[1]}], nl.int32, buffer=nl.sbuf)"
                    )
                )
                # If scalar_name is an SBUF variable (not a Python literal),
                # use tensor_tensor with broadcast rather than tensor_scalar.
                scalar_sbuf_shape = _nki_lookup_sbuf_shape_dtype(state, scalar_name)[0]
                if scalar_sbuf_shape is not None:
                    # scalar_name is an SBUF buffer — broadcast it to match index shape
                    bcast_scalar = device_fn.new_var("_row_idx_scalar_bcast", dce=True)
                    device_fn._nki_sbuf_shapes[bcast_scalar] = list(index_shape)
                    device_fn._nki_sbuf_dtypes[bcast_scalar] = "nl.int32"
                    state.codegen.add_statement(statement_from_string(
                        f"{bcast_scalar} = nl.ndarray([{index_shape[0]}, {index_shape[1]}], nl.int32, buffer=nl.sbuf)"
                    ))
                    state.codegen.add_statement(statement_from_string(
                        f"{bcast_scalar} = nl.broadcast_to({scalar_name}, shape=({index_shape[0]}, {index_shape[1]}))"
                    ))
                    state.codegen.add_statement(statement_from_string(
                        f"nisa.tensor_tensor(dst={row_index}, data1={index_name}, data2={bcast_scalar}, op=nl.add)"
                    ))
                else:
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_scalar(dst={row_index}, data={index_name}, "
                            f"op0=nl.add, operand0={scalar_name})"
                        )
                    )
                vec_offset_var = _nki_as_uint32_p1_vector(
                    state, row_index, p_count
                )
                if vec_offset_var is not None:
                    return IndirectAP(vec_var=vec_offset_var, p_count=p_count, pattern=None)

        lhs_is_mul = _target_contains(lhs_node, "mul.Tensor")
        rhs_is_mul = _target_contains(rhs_node, "mul.Tensor")
        lhs_is_mod = _target_contains(lhs_node, "remainder", "mod")
        rhs_is_mod = _target_contains(rhs_node, "remainder", "mod")
        if (lhs_is_mul and rhs_is_mod) or (rhs_is_mul and lhs_is_mod):
            lhs_ast = state.codegen.ast_for_fx_node(lhs_node)
            rhs_ast = state.codegen.ast_for_fx_node(rhs_node)
            if isinstance(lhs_ast, ast.AST) and isinstance(rhs_ast, ast.AST):
                from ..ast_extension import statement_from_string

                device_fn = state.device_function

                def _as_row_index(name: str, *, scalar_ok: bool) -> str | None:
                    shape, dtype = _nki_lookup_sbuf_shape_dtype(state, name)
                    if shape == [1, p_count]:
                        return name
                    if shape == [p_count, 1]:
                        tr_psum = device_fn.new_var("_row_idx_tr_psum", dce=True)
                        tr_sbuf = device_fn.new_var("_row_idx_tr_sbuf", dce=True)
                        device_fn._nki_sbuf_shapes[tr_sbuf] = [1, p_count]
                        device_fn._nki_sbuf_dtypes[tr_sbuf] = "nl.int32"
                        state.codegen.add_statement(
                            statement_from_string(
                                f"{tr_psum} = nl.ndarray([1, {p_count}], "
                                f"{dtype}, buffer=nl.psum)"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"nisa.nc_transpose(dst={tr_psum}, data={name})"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"{tr_sbuf} = nl.ndarray([1, {p_count}], "
                                "nl.int32, buffer=nl.sbuf)"
                            )
                        )
                        state.codegen.add_statement(
                            statement_from_string(
                                f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
                            )
                        )
                        return tr_sbuf
                    if not scalar_ok or shape is None or len(shape) != 2:
                        return None
                    if shape[0] != 1:
                        return None
                    scalar = device_fn.new_var("_row_idx_scalar", dce=True)
                    bcast = device_fn.new_var("_row_idx_bcast", dce=True)
                    device_fn._nki_sbuf_shapes[scalar] = [1, 1]
                    device_fn._nki_sbuf_shapes[bcast] = [1, p_count]
                    device_fn._nki_sbuf_dtypes[scalar] = dtype
                    device_fn._nki_sbuf_dtypes[bcast] = "nl.int32"
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{scalar} = nl.ndarray([1, 1], {dtype}, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={scalar}, src={name}[0:1, 0:1])"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{bcast} = nl.ndarray([1, {p_count}], "
                            "nl.int32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(f"nisa.memset({bcast}, value=0)")
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_scalar(dst={bcast}, data={bcast}, "
                            f"op0=nl.add, operand0={scalar})"
                        )
                    )
                    return bcast

                lhs_name = ast.unparse(lhs_ast)
                rhs_name = ast.unparse(rhs_ast)
                lhs_row = _as_row_index(lhs_name, scalar_ok=lhs_is_mul)
                rhs_row = _as_row_index(rhs_name, scalar_ok=rhs_is_mul)
                if lhs_row is not None and rhs_row is not None:
                    row_index = device_fn.new_var("_row_idx_add", dce=True)
                    device_fn._nki_sbuf_shapes[row_index] = [1, p_count]
                    device_fn._nki_sbuf_dtypes[row_index] = "nl.int32"
                    state.codegen.add_statement(
                        statement_from_string(
                            f"{row_index} = nl.ndarray([1, {p_count}], "
                            "nl.int32, buffer=nl.sbuf)"
                        )
                    )
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.tensor_tensor(dst={row_index}, "
                            f"data1={lhs_row}, data2={rhs_row}, op=nl.add)"
                        )
                    )
                    vec_offset_var = _nki_as_uint32_p1_vector(
                        state, row_index, p_count
                    )
                    if vec_offset_var is not None:
                        return IndirectAP(vec_var=vec_offset_var, p_count=p_count, pattern=None)

    index_ast = state.codegen.ast_for_fx_node(fx_node)
    if not isinstance(index_ast, ast.AST):
        return None
    index_name = ast.unparse(index_ast)
    if p_count is None:
        index_shape, _ = _nki_lookup_sbuf_shape_dtype(state, index_name)
        if index_shape is None or len(index_shape) != 2:
            return None
        if index_shape[0] == 1:
            p_count = int(index_shape[1])
        elif index_shape[1] == 1:
            p_count = int(index_shape[0])
        else:
            return None
    vec_offset_var = _nki_as_uint32_p1_vector(state, index_name, p_count)
    if vec_offset_var is None:
        return None
    return IndirectAP(vec_var=vec_offset_var, p_count=p_count, pattern=None)


def _nki_store_3d_row_scatter(
    state: "CodegenState",
    tensor: torch.Tensor,
    subscript: "list | tuple",
    fx_subscript: object,
    i: int,
    fx_node_i: object,
    value: object,
) -> tuple[list, list[str]] | None:
    """Build the scatter index for a 3D store ``out[vec + starts, head, :]``.

    Mirrors the load's 3D row-gather early-exit (`q[tile.index + starts,
    tile_h.begin, :]`): the kernel reshapes ``out`` from ``[L, H, D]`` to
    ``[L*H, D]``, so the flat row is ``(tile.index + starts) * H + head``.
    The row vector is built by ``_nki_row_index_gather`` (which handles the
    runtime ``+ starts`` SBUF tensor via broadcast+tensor_tensor), then scaled
    by ``H`` and offset by the scalar head. Returns ``(slice_parts,
    hbm_dim_size_strs)`` for the flat 2D scatter, or None if the pattern
    doesn't apply.

    This is reached for both the block-id-resolved subscript (``tile.index +
    starts`` carries the tile_q block) and the unresolved case, so the runtime
    offset is never folded into a scalar ``nisa.iota`` offset (which only
    accepts a compile-time int).
    """
    if not (
        tensor.dim() == 3
        and isinstance(subscript[i], torch.Tensor)
        and len(subscript) >= 3
        and isinstance(fx_node_i, torch.fx.Node)
    ):
        return None
    # Look ahead: next subscript must be scalar, then slice.
    _remaining = [s for j, s in enumerate(subscript) if j > i and s is not None]
    if len(_remaining) < 2:
        return None
    _next_sub, _next_next_sub = _remaining[0], _remaining[1]
    if isinstance(_next_sub, (slice, torch.Tensor)) or not isinstance(
        _next_next_sub, slice
    ):
        return None

    env = CompileEnvironment.current()
    device_fn = state.device_function
    import sympy as _sp_store3d

    _bs_subs: dict[_sp_store3d.Symbol, int] = {}
    for _bid in range(len(env.block_sizes)):
        _bs = env.block_sizes[_bid]
        _bs_subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))

    def _dim_int(s: object) -> int:
        return int(s._sympy_().subs(_bs_subs)) if isinstance(s, torch.SymInt) else int(s)

    h_int = _dim_int(tensor.size(1))
    d_int = _dim_int(tensor.size(2))

    # Scalar head expression (next subscript).
    _head_expr: str | None = None
    _next_fx_idx = i + 1
    while _next_fx_idx < len(subscript) and subscript[_next_fx_idx] is None:
        _next_fx_idx += 1
    _next_fx = (
        fx_subscript[_next_fx_idx]
        if fx_subscript is not None and _next_fx_idx < len(fx_subscript)
        else None
    )
    if isinstance(_next_fx, torch.fx.Node):
        _next_ast = state.codegen.ast_for_fx_node(_next_fx)
        if isinstance(_next_ast, ast.AST):
            _head_expr = ast.unparse(_next_ast)
    if _head_expr is None and isinstance(_next_sub, (int, bool)):
        _head_expr = str(int(_next_sub))
    elif _head_expr is None and isinstance(_next_sub, torch.SymInt):
        _head_expr = str(_dim_int(_next_sub))
    if _head_expr is None:
        return None

    # Partition count from the value being stored.
    _val_name = ast.unparse(value) if isinstance(value, ast.AST) else str(value)
    _val_shape = device_fn._nki_sbuf_shapes.get(_val_name)
    _p_count = _val_shape[0] if _val_shape and len(_val_shape) >= 1 else None

    _row_scatter = _nki_row_index_gather(fx_node_i, state, _p_count)
    if not isinstance(_row_scatter, IndirectAP):
        return None

    _flat_var = device_fn.new_var("_3d_flat_idx_store", dce=True)
    device_fn._nki_sbuf_shapes[_flat_var] = [_p_count, 1]
    device_fn._nki_sbuf_dtypes[_flat_var] = "nl.uint32"
    from ..ast_extension import statement_from_string

    state.codegen.add_statement(statement_from_string(
        f"{_flat_var} = nl.ndarray([{_p_count}, 1], nl.uint32, buffer=nl.sbuf)"
    ))
    state.codegen.add_statement(statement_from_string(
        f"nisa.tensor_copy(dst={_flat_var}, src={_row_scatter.vec_var})"
    ))
    if h_int != 1:
        state.codegen.add_statement(statement_from_string(
            f"nisa.tensor_scalar(dst={_flat_var}, data={_flat_var}, "
            f"op0=nl.multiply, operand0={h_int}, op1=None)"
        ))
    state.codegen.add_statement(statement_from_string(
        f"nisa.tensor_scalar(dst={_flat_var}, data={_flat_var}, "
        f"op0=nl.add, operand0={_head_expr}, op1=None)"
    ))
    slice_parts = [IndirectAP(vec_var=_flat_var, p_count=_p_count, pattern=None), f"0:{d_int}"]
    total_rows = int(tensor.numel()) // d_int
    return slice_parts, [str(total_rows), str(d_int)]
