from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING
from typing import Callable

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = [
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
]


_VALID_SEMS: set[str] = {"relaxed", "acquire", "release", "acq_rel"}


def _validate_sem(sem: str) -> None:
    if sem not in _VALID_SEMS:
        raise exc.InternalError(
            ValueError(
                f"Invalid memory semantic '{sem}'. Valid options are: relaxed, acquire, release, acq_rel"
            )
        )


def _prepare_mem_args(
    target: torch.Tensor,
    index: list[object],
    *values: object,
    sem: str = "relaxed",
) -> tuple:
    from .tile_proxy import Tile

    _validate_sem(sem)
    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)
    return (target, index, *values, sem)


def _codegen_common(
    tl_func: str, state: CodegenState, value_exprs: list[ast.AST]
) -> ast.AST:
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(tl_func)

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    placeholder_names = [f"v{i}" for i in range(len(value_exprs))]
    values_section = (
        ", " + ", ".join([f"{{{n}}}" for n in placeholder_names]) if value_exprs else ""
    )
    placeholders = dict(zip(placeholder_names, value_exprs, strict=False))
    return expr_from_string(
        f"tl.{tl_func}({name} + {{offset}}{values_section}, mask={{mask}}, sem={{sem}})",
        offset=indices.index_expr,
        mask=indices.mask_expr,
        sem=sem,
        **placeholders,
    )


def _cute_pointer_expr(
    state: CodegenState, target: torch.Tensor, index: list[object]
) -> str:
    from .memory_ops import _cute_index_exprs

    index_exprs = _cute_index_exprs(state, index)
    coord = (
        f"({index_exprs[0]},)"
        if len(index_exprs) == 1
        else f"({', '.join(index_exprs)})"
    )
    name = state.device_function.tensor_arg(target).name
    return f"{name}.iterator + cute.crd2idx({coord}, {name}.layout)"


def _codegen_common_cute(
    cute_func: str,
    state: CodegenState,
    *,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
) -> ast.AST:
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(cute_func)

    pointer = _cute_pointer_expr(state, target, index)
    values_section = ", ".join(f"{k}={{{k}}}" for k in keyword_names)
    placeholders = dict(zip(keyword_names, value_exprs, strict=True))
    return expr_from_string(
        f"cute.arch.{cute_func}({{ptr}}, {values_section}, sem={{sem}})",
        ptr=expr_from_string(pointer),
        sem=sem,
        **placeholders,
    )


def _to_ast_values(values: list[object]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for v in values:
        if isinstance(v, (int, float, bool)):
            out.append(expr_from_string(constant_repr(v)))
        else:
            assert isinstance(v, ast.AST)
            out.append(v)
    return out


def _ref_apply(
    target: torch.Tensor,
    index: list[object],
    apply_fn: Callable[[torch.Tensor, tuple, object], None],
    value: object,
) -> None:
    from .ref_tile import RefTile

    # Convert indices to proper format
    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)

    # Find tensor indices that need element-wise processing
    tensor_indices = [
        (i, idx)
        for i, idx in enumerate(processed_index)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1
    ]

    if tensor_indices:
        # Element-wise processing for tensor indices (handle first tensor index)
        i, tensor_idx = tensor_indices[0]

        if tensor_idx.ndim == 0:
            coords_iter = [()]
        else:
            ranges = [range(dim) for dim in tensor_idx.shape]
            coords_iter = itertools.product(*ranges)

        for coords in coords_iter:
            elem = tensor_idx[coords].item()
            new_index = processed_index.copy()
            new_index[i] = int(elem)
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                next_value = value[coords]
            else:
                next_value = value
            _ref_apply(target, new_index, apply_fn, next_value)
    else:
        apply_fn(target, tuple(processed_index), value)


# -- atomic_add --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_add(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically add a value to a target tensor.

    Performs an atomic read-modify-write that adds ``value`` to
    ``target[index]``. This is safe for concurrent access from multiple
    threads/blocks.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to add (tensor or scalar).
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.

    Example:
        @helion.kernel
        def global_sum(x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_add(result, [0], x[tile].sum())
            return result

    Notes:
        - Use for race-free accumulation across parallel execution.
        - Higher memory semantics may reduce performance.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, torch.Tensor | float | int, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_add)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    # Convert indices for shape computation and fast path detection
    processed_index: list[object] = []
    has_tensor_index = False
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor):
            if idx.numel() == 1:
                processed_index.append(int(idx.item()))
            else:
                processed_index.append(idx)
                has_tensor_index = True
        else:
            processed_index.append(idx)

    def _convert_value_to_target_dtype(val: object) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            vt = val.to(device=target.device)
            if vt.dtype != target.dtype:
                vt = vt.to(dtype=target.dtype)
            return vt
        return torch.as_tensor(val, dtype=target.dtype, device=target.device)

    if has_tensor_index:
        ret_shape = SubscriptIndexing.compute_shape(target, processed_index)
        prev_chunks: list[torch.Tensor] = []

        def apply(t: torch.Tensor, idx_tuple: tuple, v: object) -> None:
            prev_val = t[idx_tuple].clone()
            val_tensor = _convert_value_to_target_dtype(v)
            t[idx_tuple] = t[idx_tuple] + val_tensor
            prev_chunks.append(prev_val.reshape(-1))

        _ref_apply(target, index, apply, value)
        if prev_chunks:
            flat_prev = torch.cat(prev_chunks)
        else:
            flat_prev = target.new_empty(0, dtype=target.dtype, device=target.device)
        return flat_prev.reshape(ret_shape)

    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val_tensor = _convert_value_to_target_dtype(value)
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = target[idx_tuple] + val_tensor
    return prev


@_decorators.codegen(atomic_add, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_add", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_add, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_add",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@_decorators.codegen(atomic_add, "nki")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import create
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.backend import NKIOpOverrides
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.host_function import HostFunction
    from .memory_ops import _nki_row_index_gather
    from .memory_ops import _nki_subscript_block_id
    from .memory_ops import load
    from .memory_ops import store

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    value = state.ast_arg(2)
    target_ast = state.ast_arg(0)
    target_name = ast.unparse(target_ast)
    value_name = ast.unparse(value)
    device_fn = state.device_function
    env = CompileEnvironment.current()

    target_shape = device_fn._nki_sbuf_shapes.get(target_name)
    if target_shape is not None:
        fx_index = (
            state.fx_node.args[1]
            if state.fx_node is not None and len(state.fx_node.args) >= 2
            else None
        )
        value_shape = device_fn._nki_sbuf_shapes.get(value_name)
        parts: list[str] = []
        tensor_dim_idx = 0
        for i, sub_val in enumerate(index):
            if sub_val is None:
                continue
            if tensor_dim_idx >= len(target_shape):
                break
            fx_node_i = fx_index[i] if fx_index is not None and i < len(fx_index) else None
            block_id = _nki_subscript_block_id(sub_val, fx_node_i, env)
            if block_id is not None and block_id in state.codegen.active_device_loops:
                offset = state.codegen.offset_var(block_id)
                if value_shape is not None and tensor_dim_idx < len(value_shape):
                    width = int(value_shape[tensor_dim_idx])
                else:
                    width = int(env.block_sizes[block_id].from_config_assert(state.config))
                parts.append(f"{offset}:{offset}+{width}")
            elif isinstance(sub_val, (int, bool)):
                scalar = int(sub_val)
                parts.append(f"{scalar}:{scalar}+1")
            elif isinstance(sub_val, torch.SymInt):
                scalar = int(env.size_hint(sub_val))
                parts.append(f"{scalar}:{scalar}+1")
            elif isinstance(sub_val, slice):
                dim_size = target_shape[tensor_dim_idx]
                parts.append(f"0:{dim_size}")
            elif isinstance(sub_val, torch.Tensor):
                raise exc.BackendUnsupported(
                    "nki",
                    "atomic_add on SBUF tensors does not support tensor-valued indices",
                )
            else:
                dim_size = target_shape[tensor_dim_idx]
                parts.append(f"0:{dim_size}")
            tensor_dim_idx += 1

        if not parts:
            parts = [f"0:{dim}" for dim in target_shape]
        while len(parts) < len(target_shape):
            parts.append(f"0:{target_shape[len(parts)]}")

        out_shape = value_shape if value_shape is not None else list(target_shape)
        out_shape = [int(dim) for dim in out_shape]
        shape_str = ", ".join(str(dim) for dim in out_shape)
        dtype_str = env.backend.dtype_str(target.dtype)
        prev = device_fn.new_var("_nki_atomic_prev", dce=True)
        updated = device_fn.new_var("_nki_atomic_updated", dce=True)
        device_fn._nki_sbuf_shapes[prev] = list(out_shape)
        device_fn._nki_sbuf_shapes[updated] = list(out_shape)
        device_fn._nki_sbuf_dtypes[prev] = dtype_str
        device_fn._nki_sbuf_dtypes[updated] = dtype_str
        target_slice = f"{target_name}[{', '.join(parts)}]"
        state.codegen.add_statement(
            statement_from_string(
                f"{prev} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.tensor_copy(dst={prev}, src={target_slice})")
        )
        state.codegen.add_statement(
            statement_from_string(
                f"{updated} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.tensor_tensor(dst={updated}, data1={prev}, "
                f"data2={{value}}, op=nl.add)",
                value=value,
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.tensor_copy(dst={target_slice}, src={updated})")
        )
        return expr_from_string(prev)

    try:
        hf = HostFunction.current()
    except Exception:
        hf = None
    if hf is not None and target in hf.tensor_to_origin:
        if not hasattr(device_fn, "_nki_return_buffers"):
            device_fn._nki_return_buffers = {}
        tensor_id = id(target)
        if tensor_id not in device_fn._nki_return_buffers:
            ret_buf_name = device_fn.new_var("nki_return_buf")
            dtype_str = env.backend.dtype_str(target.dtype)
            shape_parts = []
            for dim_i in range(target.dim()):
                size_i = target.size(dim_i)
                shape_parts.append(
                    state.sympy_expr(size_i._sympy_())
                    if isinstance(size_i, torch.SymInt)
                    else str(size_i)
                )
            if target.dim() == 1:
                shape_str = f"1, {shape_parts[0]}"
                host_reshape = f"[{shape_parts[0]}]"
            elif target.dim() > 2:
                leading = shape_parts[:-1]
                flat_leading = " * ".join(f"({p})" for p in leading)
                shape_str = f"({flat_leading}), {shape_parts[-1]}"
                host_reshape = f"[{', '.join(shape_parts)}]"
            else:
                shape_str = ", ".join(shape_parts)
                host_reshape = None
            device_fn.preamble.append(
                statement_from_string(
                    f"{ret_buf_name} = nl.ndarray([{shape_str}], "
                    f"dtype={dtype_str}, buffer=nl.shared_hbm)"
                )
            )
            if target_name.startswith("_host_tensor"):
                zero_tile = device_fn.new_var("_nki_atomic_zero", dce=True)
                init_p = device_fn.new_var("_nki_atomic_init_p", dce=True)
                if target.dim() == 1:
                    device_fn.preamble.append(
                        statement_from_string(
                            f"{zero_tile} = nl.ndarray([1, {shape_parts[0]}], "
                            f"{dtype_str}, buffer=nl.sbuf)"
                        )
                    )
                    device_fn.preamble.append(
                        statement_from_string(f"nisa.memset({zero_tile}, value=0)")
                    )
                    device_fn.preamble.append(
                        statement_from_string(
                            f"nisa.dma_copy(dst={ret_buf_name}[0:1, 0:{shape_parts[0]}], "
                            f"src={zero_tile})"
                        )
                    )
                elif target.dim() == 2:
                    device_fn.preamble.append(
                        statement_from_string(
                            f"{zero_tile} = nl.ndarray([1, {shape_parts[1]}], "
                            f"{dtype_str}, buffer=nl.sbuf)"
                        )
                    )
                    device_fn.preamble.append(
                        statement_from_string(f"nisa.memset({zero_tile}, value=0)")
                    )
                    device_fn.preamble.append(
                        create(
                            ast.For,
                            target=create(ast.Name, id=init_p, ctx=ast.Store()),
                            iter=expr_from_string(f"nl.affine_range(0, {shape_parts[0]}, 1)"),
                            body=[
                                statement_from_string(
                                    f"nisa.dma_copy(dst={ret_buf_name}[{init_p}:{init_p}+1, "
                                    f"0:{shape_parts[1]}], src={zero_tile})"
                                )
                            ],
                            orelse=[],
                        )
                    )
                else:
                    device_fn.preamble.append(
                        statement_from_string(
                            f"{zero_tile} = nl.ndarray([1, {shape_parts[-1]}], "
                            f"{dtype_str}, buffer=nl.sbuf)"
                        )
                    )
                    device_fn.preamble.append(
                        statement_from_string(f"nisa.memset({zero_tile}, value=0)")
                    )
                    device_fn.preamble.append(
                        create(
                            ast.For,
                            target=create(ast.Name, id=init_p, ctx=ast.Store()),
                            iter=expr_from_string(
                                f"nl.affine_range(0, ({' * '.join(shape_parts[:-1])}), 1)"
                            ),
                            body=[
                                statement_from_string(
                                    f"nisa.dma_copy(dst={ret_buf_name}[{init_p}:{init_p}+1, "
                                    f"0:{shape_parts[-1]}], src={zero_tile})"
                                )
                            ],
                            orelse=[],
                        )
                    )
            else:
                device_fn.preamble.append(
                    statement_from_string(
                        f"nisa.dma_copy(dst={ret_buf_name}, src={target_name})"
                    )
                )
            origin = hf.tensor_to_origin[target]
            device_fn._nki_return_buffers[tensor_id] = {
                "buf_name": ret_buf_name,
                "host_var": origin.host_str(),
                "host_reshape": host_reshape,
            }
            if len(device_fn._nki_return_buffers) == 1:
                device_fn._nki_return_buffer_name = ret_buf_name
                device_fn._nki_return_host_var = origin.host_str()
                device_fn._nki_return_host_reshape = host_reshape

    def _try_emit_hbm_row_scatter_rmw() -> ast.AST | None:
        if state.fx_node is not None and len(state.fx_node.users) > 0:
            return None
        if target.dim() != 2:
            return None
        value_shape = device_fn._nki_sbuf_shapes.get(value_name)
        value_dtype = device_fn._nki_sbuf_dtypes.get(value_name)
        if value_shape is None or len(value_shape) != 2:
            return None
        p_count = int(value_shape[0])
        f_count = int(value_shape[1])
        if p_count < 1 or p_count > 128 or f_count < 1:
            return None
        dtype_str = env.backend.dtype_str(target.dtype)
        if value_dtype is not None and value_dtype != dtype_str:
            return None
        fx_index = (
            state.fx_node.args[1]
            if state.fx_node is not None and len(state.fx_node.args) >= 2
            else None
        )
        if not isinstance(fx_index, (list, tuple)) or len(fx_index) < 2:
            return None
        if len(index) < 2 or not isinstance(index[0], torch.Tensor):
            return None
        row_part = _nki_row_index_gather(fx_index[0], state, p_count)
        prefix = "__AP_ROW_GATHER__"
        if row_part is None or not row_part.startswith(prefix) or not row_part.endswith("__"):
            return None
        vec_offset = row_part[len(prefix) : -2]

        feature_start: str | None = None
        block_id = _nki_subscript_block_id(index[1], fx_index[1], env)
        if block_id is not None and block_id in state.codegen.active_device_loops:
            feature_start = state.codegen.offset_var(block_id)
        elif isinstance(index[1], (int, bool)):
            feature_start = str(int(index[1]))
        elif isinstance(index[1], torch.SymInt):
            feature_start = state.sympy_expr(index[1]._sympy_())
        elif isinstance(index[1], slice):
            feature_start = "0"
        if feature_start is None:
            return None

        hbm_name: str | None = None
        return_buffers = getattr(device_fn, "_nki_return_buffers", {})
        buf_info = return_buffers.get(id(target))
        if buf_info is not None:
            hbm_name = buf_info["buf_name"]
        elif hf is None or target not in hf.tensor_to_origin:
            try:
                hbm_name = device_fn.tensor_arg(target).name
            except KeyError:
                return None
        if hbm_name is None:
            return None

        target_f = target.size(1)
        if isinstance(target_f, torch.SymInt):
            f_total = state.sympy_expr(target_f._sympy_())
        else:
            f_total = str(int(target_f))

        row_i = device_fn.new_var("_nki_atomic_row_i", dce=True)
        prev = device_fn.new_var("_nki_atomic_row_prev", dce=True)
        row_value = device_fn.new_var("_nki_atomic_row_value", dce=True)
        updated = device_fn.new_var("_nki_atomic_row_updated", dce=True)
        for name in (prev, row_value, updated):
            device_fn._nki_sbuf_shapes[name] = [1, f_count]
            device_fn._nki_sbuf_dtypes[name] = dtype_str
        row_vec = f"{vec_offset}[{row_i}:{row_i}+1, 0:1]"
        dst_expr = (
            f"{hbm_name}.ap(pattern=[[{f_total}, 1], [1, {f_count}]], "
            f"offset={feature_start}, vector_offset={row_vec}, indirect_dim=0)"
        )
        state.codegen.add_statement(
            create(
                ast.For,
                target=create(ast.Name, id=row_i, ctx=ast.Store()),
                iter=expr_from_string(f"nl.affine_range(0, {p_count}, 1)"),
                body=[
                    statement_from_string(
                        f"{prev} = nl.ndarray([1, {f_count}], {dtype_str}, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.dma_copy(dst={prev}, src={dst_expr})"
                    ),
                    statement_from_string(
                        f"{row_value} = nl.ndarray([1, {f_count}], {dtype_str}, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.tensor_copy(dst={row_value}, "
                        f"src={value_name}[{row_i}:{row_i}+1, 0:{f_count}])"
                    ),
                    statement_from_string(
                        f"{updated} = nl.ndarray([1, {f_count}], {dtype_str}, buffer=nl.sbuf)"
                    ),
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={updated}, data1={prev}, "
                        f"data2={row_value}, op=nl.add)"
                    ),
                    statement_from_string(
                        f"nisa.dma_copy(dst={dst_expr}, src={updated})"
                    ),
                ],
                orelse=[],
            )
        )
        return expr_from_string(value_name)

    row_scatter_rmw = _try_emit_hbm_row_scatter_rmw()
    if row_scatter_rmw is not None:
        return row_scatter_rmw

    load_state = state._replace(
        proxy_args=[target, index, None, None],
        ast_args=[target_ast, state.ast_args[1], None, None],
    )
    prev = load._codegen["nki"](load_state)
    prev_name = ast.unparse(prev) if isinstance(prev, ast.AST) else str(prev)
    updated = NKIOpOverrides().add(prev_name, value_name)
    store_state = state._replace(
        proxy_args=[target, index, None, None],
        ast_args=[
            target_ast,
            state.ast_args[1],
            expr_from_string(str(updated)),
            None,
        ],
    )
    store._codegen["nki"](store_state)
    return prev


# -- atomic_xchg --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_xchg(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically exchange (set) a value at ``target[index]``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: New value(s) to set.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_xchg)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_xchg)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_xchg)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [unsupported-operation]
    target[idx_tuple] = val
    return prev


@_decorators.codegen(atomic_xchg, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xchg", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xchg, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_exch",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


# -- atomic_and/or/xor --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_and(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise AND with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to AND with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_and)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_and)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_and)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = target[idx_tuple] & val
    return prev


@_decorators.codegen(atomic_and, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_and", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_and, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_and",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_or(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise OR with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to OR with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_or)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_or)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_or)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = target[idx_tuple] | val
    return prev


@_decorators.codegen(atomic_or, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_or", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_or, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_or",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_xor(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise XOR with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to XOR with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_xor)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_xor)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_xor)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = target[idx_tuple] ^ val
    return prev


@_decorators.codegen(atomic_xor, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xor", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xor, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_xor",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


# -- atomic_max/min --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_max(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically update ``target[index]`` with the maximum of current value
    and ``value``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to compare with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_max)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_max)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_max)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> None:
    _validate_sem(sem)

    def apply(t: torch.Tensor, idx: tuple, v: object) -> None:
        t[idx] = torch.maximum(
            t[idx], torch.as_tensor(v, dtype=t[idx].dtype, device=t.device)
        )

    _ref_apply(target, index, apply, value)


@_decorators.codegen(atomic_max, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_max", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_max, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_max",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_min(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically update ``target[index]`` with the minimum of current value
    and ``value``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to compare with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
        ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_min)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_min)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_min)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = torch.minimum(target[idx_tuple], val)
    return prev


@_decorators.codegen(atomic_min, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_min", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_min, "cute")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common_cute(
        "atomic_min",
        state,
        value_exprs=_to_ast_values([value_expr]),
        keyword_names=["val"],
    )


# -- atomic_cas --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_cas(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor | float | bool,
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically compare-and-swap a value at ``target[index]``.

    If the current value equals ``expected``, writes ``value``. Otherwise
    leaves memory unchanged.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        expected: Expected current value(s) used for comparison.
        value: New value(s) to write if comparison succeeds.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the compare-and-swap.

    Note:
        Triton CAS doesn’t support a masked form; our generated code uses
        an unmasked CAS and relies on index masking to avoid OOB.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: object,
    value: object,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, object, object, str]:
    return _prepare_mem_args(target, index, expected, value, sem=sem)


@_decorators.register_fake(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor,
    value: torch.Tensor,
    sem: str = "relaxed",
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor | float | bool,
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    exp_t = (
        expected
        if isinstance(expected, torch.Tensor)
        else torch.as_tensor(expected, dtype=target.dtype, device=target.device)
    )
    val_t = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index]
    mask = target[idx_tuple] == exp_t
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = torch.where(mask, val_t, target[idx_tuple])
    return prev


@_decorators.codegen(atomic_cas, "triton")
def _(state: CodegenState) -> ast.AST:
    exp_expr = state.ast_args[2]
    val_expr = state.ast_args[3]
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    return expr_from_string(
        f"tl.atomic_cas({name} + {{offset}}, {{exp}}, {{val}}, sem={{sem}})",
        offset=indices.index_expr,
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )


@_decorators.codegen(atomic_cas, "cute")
def _(state: CodegenState) -> ast.AST:
    exp_expr = state.ast_args[2]
    val_expr = state.ast_args[3]
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("atomic_cas")

    pointer = _cute_pointer_expr(state, target, index)
    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    return expr_from_string(
        "cute.arch.atomic_cas({ptr}, cmp={exp}, val={val}, sem={sem})",
        ptr=expr_from_string(pointer),
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )
