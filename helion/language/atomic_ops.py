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
from .._compiler.compile_environment import _symint_expr
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.variable_origin import GridOrigin
from . import _decorators
from . import _tracing_ops
from ._nki_dim_access import IndirectAP

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
    index = Tile._tiles_to_sizes_for_index(index)
    return (target, index, *values, sem)


def _codegen_common(
    op: str, state: CodegenState, value_exprs: list[ast.AST]
) -> ast.AST:
    """Route any single-value atomic op through the atomic_indexing strategy."""
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(op)

    device_fn = state.device_function
    fx_node = state.fx_node
    epilogue_subtile_group_id = (
        None if fx_node is None else fx_node.meta.get("epilogue_subtile_group_id")
    )
    if epilogue_subtile_group_id is None:
        indexing_idx = device_fn.atomic_op_index
        device_fn.atomic_op_index += 1
    elif fx_node is not None and fx_node.meta.get(
        "epilogue_subtile_primary_output", False
    ):
        indexing_idx = device_fn.atomic_op_index
        device_fn.atomic_op_index += 1
        device_fn.epilogue_subtile_atomic_indices[epilogue_subtile_group_id] = (
            indexing_idx
        )
    else:
        indexing_idx = device_fn.epilogue_subtile_atomic_indices[
            epilogue_subtile_group_id
        ]
    strategy = device_fn.get_atomic_indexing_strategy(indexing_idx)
    return strategy.codegen_atomic(op, state, target, index, value_exprs[0], sem)


def _cute_pointer_expr(
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    ast_index: list[object] | tuple[object, ...] | None = None,
) -> str:
    from .memory_ops import _cute_index_exprs

    index_exprs = _cute_index_exprs(state, index, ast_index)
    name = state.device_function.tensor_arg(target).name
    coord = (
        f"({index_exprs[0]},)"
        if len(index_exprs) == 1
        else f"({', '.join(index_exprs)})"
    )
    return f"({name}.iterator + cute.crd2idx({coord}, {name}.layout)).llvm_ptr"


def _resolve_cute_atomic_kwargs(cute_func: str, requested: list[str]) -> list[str]:
    """Map our intended ``cute.arch.<cute_func>`` kwarg names onto whatever
    the live signature actually exposes.

    Helion's emitted code refers to ``cute.arch.atomic_*`` parameters by
    name (``val``, ``cmp``). Some nvidia-cutlass-dsl wheels have shipped
    with these renamed (e.g. ``val`` -> ``value``); the old emission
    style then trips a ``TypeError`` deep inside CUTLASS at run time.
    Probe the signature at codegen time and rewrite the kwarg names to
    match what the live wrapper accepts. Falls back to the requested
    name when none of the rename candidates appears, so healthy installs
    are unaffected.
    """
    import inspect

    try:
        import cutlass.cute as cute  # type: ignore[import-not-found]
    except ImportError:
        return list(requested)
    func = getattr(getattr(cute, "arch", None), cute_func, None)
    if func is None:
        return list(requested)
    try:
        params = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return list(requested)
    rename_candidates: dict[str, tuple[str, ...]] = {
        "val": ("val", "value", "rhs", "src", "a"),
        "cmp": ("cmp", "compare", "expected", "exp"),
    }
    resolved: list[str] = []
    for name in requested:
        candidates = rename_candidates.get(name, (name,))
        chosen = next((c for c in candidates if c in params), name)
        resolved.append(chosen)
    return resolved


_CUTE_FLOAT_ATOMIC_HELPERS: dict[str, str] = {
    "atomic_max": "_cute_atomic_max_float32",
    "atomic_min": "_cute_atomic_min_float32",
}


def _cute_atomic_callee(cute_func: str, target_dtype_torch: torch.dtype) -> str:
    """Pick the callee for a CuTe atomic op.

    NVVM/PTX has no native ``atom.max``/``atom.min`` for floating point, so
    float ``atomic_max``/``atomic_min`` are routed through runtime helpers that
    emulate them with integer atomics (registered in
    ``CuteBackend.library_imports``). All other ops, and integer max/min, use
    the native ``cute.arch.<func>`` directly.
    """
    helper = _CUTE_FLOAT_ATOMIC_HELPERS.get(cute_func)
    if helper is None or not target_dtype_torch.is_floating_point:
        return f"cute.arch.{cute_func}"
    if target_dtype_torch is not torch.float32:
        raise exc.BackendUnsupported(
            "cute",
            f"{cute_func} on floating-point dtype {target_dtype_torch} "
            "(only float32 is supported)",
        )
    return helper


def _codegen_common_cute(
    cute_func: str,
    state: CodegenState,
    *,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
) -> ast.AST:
    from .._compiler.compile_environment import CompileEnvironment

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(cute_func)

    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(target.dtype)
    callee = _cute_atomic_callee(cute_func, target.dtype)
    cast_value_exprs = [
        expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value_expr,
        )
        for value_expr in value_exprs
    ]
    tensor_index_stmt = _codegen_tensor_index_common_cute(
        cute_func,
        state,
        target,
        index,
        sem,
        cast_value_exprs,
        keyword_names,
        callee,
    )
    if tensor_index_stmt is not None:
        return tensor_index_stmt
    ast_index = state.ast_args[1]
    assert isinstance(ast_index, (list, tuple))
    pointer = _cute_pointer_expr(state, target, index, ast_index)
    resolved_kwargs = _resolve_cute_atomic_kwargs(cute_func, keyword_names)
    values_section = ", ".join(
        f"{actual}={{{intent}}}"
        for intent, actual in zip(keyword_names, resolved_kwargs, strict=True)
    )
    placeholders = dict(zip(keyword_names, cast_value_exprs, strict=True))
    atomic_expr = expr_from_string(
        f"{callee}({{ptr}}, {values_section}, sem={{sem}})",
        ptr=expr_from_string(pointer),
        sem=sem,
        **placeholders,
    )
    return _guard_cute_atomic_expr(state, index, target_dtype, atomic_expr)


def _guard_cute_atomic_expr(
    state: CodegenState,
    index: list[object],
    target_dtype: str,
    atomic_expr: ast.AST,
    *,
    extra_predicates: list[str] | None = None,
) -> ast.AST:
    predicates = [
        predicate
        for predicate in (
            _cute_active_mask_predicate(state),
            _cute_leader_thread_predicate(state, index),
            _cute_unindexed_axis_leader_predicate(state, index),
            *(extra_predicates or []),
        )
        if predicate is not None
    ]
    if not predicates:
        return atomic_expr
    predicate_expr = expr_from_string(" and ".join(predicates))
    assert isinstance(predicate_expr, ast.expr)
    assert isinstance(atomic_expr, ast.expr)
    if state.fx_node is not None and len(state.fx_node.users) == 0:
        state.codegen.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=predicate_expr,
                    body=[ast.Expr(value=atomic_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)

    result_var = state.device_function.new_var("_atomic_prev", dce=True)
    zero_value = expr_from_string(f"{target_dtype}(0)")
    assert isinstance(zero_value, ast.expr)
    state.codegen.add_statement(
        ast.fix_missing_locations(
            ast.Assign(
                targets=[ast.Name(id=result_var, ctx=ast.Store())],
                value=zero_value,
            )
        )
    )
    state.codegen.add_statement(
        ast.fix_missing_locations(
            ast.If(
                test=predicate_expr,
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=result_var, ctx=ast.Store())],
                        value=atomic_expr,
                    )
                ],
                orelse=[],
            )
        )
    )
    return expr_from_string(result_var)


def _cute_tensor_index_leader_predicate(
    state: CodegenState,
    tensor_index: torch.Tensor,
) -> str | None:
    from .._compiler.compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor_index.shape[0])
    if block_id is None:
        return None
    assert state.fx_node is not None
    block_id = env.resolve_codegen_block_id(
        block_id, state.codegen, state.fx_node.graph
    )

    index_axes: set[int] = set()
    other_axes: set[int] = set()

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for candidate_block_id, thread_axis in grid_state.block_thread_axes.items():
            if candidate_block_id == block_id:
                index_axes.add(thread_axis)
            else:
                other_axes.add(thread_axis)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            for candidate_block_id, thread_axis in loop_state.block_thread_axes.items():
                if candidate_block_id == block_id:
                    index_axes.add(thread_axis)
                else:
                    other_axes.add(thread_axis)

    leader_axes = sorted(axis for axis in other_axes if axis not in index_axes)
    if not leader_axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in leader_axes
    )


def _cute_leader_thread_predicate(
    state: CodegenState,
    index: list[object],
) -> str | None:
    scalar_origin_block_ids: set[int] = set()
    for idx in index:
        if not isinstance(idx, torch.SymInt):
            continue
        expr = _symint_expr(idx)
        if expr is None:
            continue
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is None or not isinstance(origin_info.origin, GridOrigin):
            continue
        if type(origin_info.origin) is GridOrigin:
            continue
        scalar_origin_block_ids.add(origin_info.origin.block_id)
    if not scalar_origin_block_ids:
        return None

    axes: set[int] = set()
    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for block_id in scalar_origin_block_ids:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                axes.add(thread_axis)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            for block_id in scalar_origin_block_ids:
                thread_axis = loop_state.block_thread_axes.get(block_id)
                if thread_axis is not None:
                    axes.add(thread_axis)
    if not axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in sorted(axes)
    )


def _cute_unindexed_axis_leader_predicate(
    state: CodegenState,
    index: list[object],
) -> str | None:
    """Return predicate restricting atomics to one thread per unindexed parallel axis.

    Two complementary mechanisms:

    1. **Active-axis leaders (block-size-index form):** When an atomic op
       is invoked inside ``hl.tile([m, n])`` but the index only covers a
       subset of the tile dimensions (e.g. ``hl.atomic_add(dy, [tile_i],
       reduced)`` after a reduction across ``tile_j``), every thread on
       the unindexed axis would otherwise re-issue the atomic with the
       same value. This mechanism uses ``BlockSizeOrigin`` symbols in the
       index to decide which currently-active thread axes are indexed,
       and restricts the rest to ``thread_idx[axis] == 0``.

    2. **Ghost-axis leaders (any index form):** A CTA-resident thread
       axis whose owning device loop has exited cannot be referenced by
       any current expression — including gather (``idxs[tile]``) or
       offset-constant (``[tile.begin]``) index forms whose index does
       not flow through a ``BlockSizeOrigin``. Threads on a ghost axis
       still exist (the CUDA ``blockDim`` is fixed for the kernel) and
       would re-issue the atomic with whatever value the surviving
       expression resolved to, multiplying the result by the ghost
       axis's size. Predicate the ghost axis to leader unconditionally.
       The canonical case is ``examples.matmul_split_k`` under
       autotune-picked configs that pack the inner-K loop onto a CTA
       thread axis.
    """
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.variable_origin import BlockSizeOrigin

    env = CompileEnvironment.current()
    indexed_block_ids: set[int] = set()
    has_block_size_index = False
    for idx in index:
        if isinstance(idx, torch.Tensor):
            # A gather/scatter index tensor (e.g. ``output[idxs, tile_f]``)
            # covers the tile dimension it was loaded along: the per-thread
            # ``idxs`` value differs across that axis, so it is *indexed* and
            # must not be collapsed to a single leader thread.
            tensor_block_id = (
                env.resolve_block_id(idx.shape[0]) if idx.ndim >= 1 else None
            )
            if tensor_block_id is not None and state.fx_node is not None:
                has_block_size_index = True
                indexed_block_ids.add(
                    env.resolve_codegen_block_id(
                        tensor_block_id,
                        state.codegen,
                        state.fx_node.graph,
                    )
                )
            continue
        if not isinstance(idx, torch.SymInt):
            continue
        expr = _symint_expr(idx)
        if expr is None:
            continue
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            continue
        has_block_size_index = True
        assert state.fx_node is not None
        indexed_block_ids.add(
            env.resolve_codegen_block_id(
                origin_info.origin.block_id,
                state.codegen,
                state.fx_node.graph,
            )
        )

    fx_graph = state.fx_node.graph if state.fx_node is not None else None

    leader_axes: set[int] = set()
    active_thread_axes: set[int] = set()

    def collect(thread_axes: dict[int, int]) -> None:
        for candidate_block_id, thread_axis in thread_axes.items():
            active_thread_axes.add(thread_axis)
            if not has_block_size_index or fx_graph is None:
                # Without a block-size index there is no reliable mapping
                # from "block_id indexed by this atomic" to a thread axis,
                # so skip the active-axis mechanism. Ghost-axis predicates
                # below still fire.
                continue
            resolved = env.resolve_codegen_block_id(
                candidate_block_id, state.codegen, fx_graph
            )
            if resolved in indexed_block_ids:
                continue
            leader_axes.add(thread_axis)

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        collect(grid_state.block_thread_axes)
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            collect(loop_state.block_thread_axes)

    # Ghost-axis leaders: any CTA-resident thread axis (size > 1) that
    # no active loop currently owns. ``max_thread_block_dims`` tracks the
    # per-axis CTA size accumulated as device loops are entered and is
    # never decremented on exit, so it correctly captures axes whose
    # owning loop has finished but whose threads remain live.
    for axis, size in enumerate(state.codegen.max_thread_block_dims):
        if size > 1 and axis not in active_thread_axes:
            leader_axes.add(axis)

    if not leader_axes:
        return None
    return " and ".join(
        f"(cute.arch.thread_idx()[{axis}] == 0)" for axis in sorted(leader_axes)
    )


def _cute_active_mask_predicate(state: CodegenState) -> str | None:
    masks: list[str] = []
    seen_blocks: set[int] = set()

    for block_id, loops in state.codegen.active_device_loops.items():
        if block_id in seen_blocks or not loops:
            continue
        seen_blocks.add(block_id)
        mask_var = loops[-1].strategy.mask_var(block_id)
        if mask_var is not None:
            masks.append(f"({mask_var})")

    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        for block_id in grid_state.block_ids:
            if block_id in seen_blocks:
                continue
            seen_blocks.add(block_id)
            mask_var = grid_state.strategy.mask_var(block_id)
            if mask_var is not None:
                masks.append(f"({mask_var})")

    if not masks:
        return None
    return " and ".join(masks)


def _resolve_tensor_index_iota_node(
    state: CodegenState, index_node: torch.fx.Node
) -> torch.fx.Node | None:
    from .._compiler.device_ir import NodeArgsGraphInfo

    current = index_node
    visited: set[torch.fx.Node] = set()
    while True:
        if current in visited:
            return None
        visited.add(current)
        if current.target is torch.ops.prims.iota.default:
            return current
        if current.op == "call_function" and current.target in {
            _tracing_ops._new_var,
            _tracing_ops._phi,
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
            torch.ops.prims.convert_element_type.default,
        }:
            arg = current.args[0] if current.args else None
            if not isinstance(arg, torch.fx.Node):
                return None
            current = arg
            continue
        if current.op != "placeholder":
            return None
        graph_infos = [
            graph_info
            for graph_info in state.codegen.codegen_graphs
            if graph_info.graph is current.graph
        ]
        if len(graph_infos) != 1:
            return None
        graph_info = graph_infos[0]
        if not isinstance(graph_info, NodeArgsGraphInfo):
            return None
        outer_node = graph_info.placeholder_to_outer_arg(current)
        if not isinstance(outer_node, torch.fx.Node):
            return None
        current = outer_node


def _codegen_tensor_index_common_cute(
    cute_func: str,
    state: CodegenState,
    target: torch.Tensor,
    index: list[object],
    sem: ast.AST,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
    callee: str,
) -> ast.AST | None:
    from .._compiler.compile_environment import CompileEnvironment
    from .memory_ops import _cute_active_index_var

    fx_node = state.fx_node
    if fx_node is None or len(index) != 1 or len(fx_node.args) < 2:
        return None
    tensor_index = index[0] if isinstance(index[0], torch.Tensor) else None
    fx_index = fx_node.args[1]
    if not isinstance(fx_index, (list, tuple)) or len(fx_index) != 1:
        return None
    index_node = fx_index[0]
    if not isinstance(index_node, torch.fx.Node):
        return None
    iota_node = _resolve_tensor_index_iota_node(state, index_node)
    if iota_node is None:
        return None
    iota_val = iota_node.meta.get("val")
    if isinstance(iota_val, torch.Tensor) and iota_val.ndim == 1:
        tensor_index = iota_val
    if tensor_index is None or tensor_index.ndim != 1:
        return None
    iota_start = iota_node.kwargs.get("start", 0)
    iota_step = iota_node.kwargs.get("step", 1)
    if iota_step != 1 or not isinstance(iota_start, int):
        return _codegen_tensor_index_loop_common_cute(
            cute_func,
            state,
            target,
            tensor_index,
            index_node,
            sem,
            value_exprs,
            keyword_names,
            callee,
        )

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor_index.shape[0])
    if block_id is None:
        return _codegen_tensor_index_loop_common_cute(
            cute_func,
            state,
            target,
            tensor_index,
            index_node,
            sem,
            value_exprs,
            keyword_names,
            callee,
        )
    block_id = env.resolve_codegen_block_id(block_id, state.codegen, fx_node.graph)
    if (index_var := _cute_active_index_var(state, block_id)) is None:
        return _codegen_tensor_index_loop_common_cute(
            cute_func,
            state,
            target,
            tensor_index,
            index_node,
            sem,
            value_exprs,
            keyword_names,
            callee,
        )

    tensor_name = state.device_function.tensor_arg(target).name
    resolved_kwargs = _resolve_cute_atomic_kwargs(cute_func, keyword_names)
    values_section = ", ".join(
        f"{actual}={{{intent}}}"
        for intent, actual in zip(keyword_names, resolved_kwargs, strict=True)
    )
    placeholders = dict(zip(keyword_names, value_exprs, strict=True))
    atomic_expr = expr_from_string(
        callee
        + "("
        + f"({tensor_name}.iterator + "
        + f"cute.crd2idx((cutlass.Int32({iota_start}) + {index_var},), {tensor_name}.layout)).llvm_ptr, "
        + values_section
        + ", sem={sem})",
        sem=sem,
        **placeholders,
    )
    target_dtype = env.backend.dtype_str(target.dtype)
    extra_predicates = [
        predicate
        for predicate in (_cute_tensor_index_leader_predicate(state, tensor_index),)
        if predicate is not None
    ]
    return _guard_cute_atomic_expr(
        state,
        index,
        target_dtype,
        atomic_expr,
        extra_predicates=extra_predicates,
    )


def _codegen_tensor_index_loop_common_cute(
    cute_func: str,
    state: CodegenState,
    target: torch.Tensor,
    tensor_index: torch.Tensor,
    index_node: torch.fx.Node,
    sem: ast.AST,
    value_exprs: list[ast.AST],
    keyword_names: list[str],
    callee: str,
) -> ast.AST | None:
    from .._compiler.ast_extension import statement_from_string

    fx_node = state.fx_node
    if fx_node is None or len(fx_node.users) > 0:
        return None
    if tensor_index.ndim != 1:
        return None
    extent = tensor_index.shape[0]
    if not isinstance(extent, int):
        return None

    ast_index = state.ast_args[1]
    if not isinstance(ast_index, (list, tuple)) or len(ast_index) != 1:
        return None
    ast_index_expr = ast_index[0]
    if not isinstance(ast_index_expr, ast.AST):
        return None

    iota_node = _resolve_tensor_index_iota_node(state, index_node)
    indexed_values: list[ast.AST] = []
    value_arg_offset = 2
    for value_expr, _keyword_name in zip(value_exprs, keyword_names, strict=True):
        value_proxy = state.proxy_arg(value_arg_offset)
        value_arg_offset += 1
        if isinstance(value_proxy, torch.Tensor) and value_proxy.ndim == 1:
            tensor_arg = state.device_function.tensor_arg(value_proxy)
            indexed_values.append(
                expr_from_string(
                    "{value}[{idx}]",
                    value=expr_from_string(tensor_arg.name),
                    idx=expr_from_string("_tensor_index_i"),
                )
            )
            continue
        if extent != 1:
            return None
        indexed_values.append(value_expr)

    if iota_node is not None:
        start = iota_node.kwargs.get("start", 0)
        step = iota_node.kwargs.get("step", 1)
        if not isinstance(start, int) or not isinstance(step, int):
            return None
        index_expr = expr_from_string(
            f"cutlass.Int32({start}) + cutlass.Int32({step}) * cutlass.Int32(_tensor_index_i)"
        )
    else:
        index_expr = expr_from_string(
            "cutlass.Int32({index}[{idx}])",
            index=ast_index_expr,
            idx=expr_from_string("_tensor_index_i"),
        )

    tensor_name = state.device_function.tensor_arg(target).name
    resolved_kwargs = _resolve_cute_atomic_kwargs(cute_func, keyword_names)
    values_section = ", ".join(
        f"{actual}={{{intent}}}"
        for intent, actual in zip(keyword_names, resolved_kwargs, strict=True)
    )
    placeholders = dict(zip(keyword_names, indexed_values, strict=True))
    atomic_expr = expr_from_string(
        callee
        + "("
        + f"({tensor_name}.iterator + "
        + f"cute.crd2idx(({{index}},), {tensor_name}.layout)).llvm_ptr, "
        + values_section
        + ", sem={sem})",
        index=index_expr,
        sem=sem,
        **placeholders,
    )
    assert isinstance(atomic_expr, ast.expr)
    predicate_terms = [
        predicate
        for predicate in (
            _cute_active_mask_predicate(state),
            _cute_tensor_index_leader_predicate(state, tensor_index),
        )
        if predicate is not None
    ]
    predicate_expr = (
        ast.parse(" and ".join(predicate_terms), mode="eval").body
        if predicate_terms
        else None
    )
    inner = (
        ast.fix_missing_locations(
            ast.If(
                test=predicate_expr,
                body=[ast.Expr(value=atomic_expr)],
                orelse=[],
            )
        )
        if predicate_expr is not None
        else ast.Expr(value=atomic_expr)
    )
    loop = statement_from_string(f"for _tensor_index_i in range({extent}):\n    pass")
    assert isinstance(loop, ast.For)
    loop.body = [inner]
    state.codegen.add_statement(loop)
    return ast.Constant(value=None)


def _pallas_atomic_load_prev(
    state: CodegenState,
) -> tuple[str, str, str]:
    """Load previous value for a Pallas atomic op.

    On TPU, each kernel instance has exclusive access to its tile, so
    atomics are implemented as regular load-compute-store sequences.

    Returns (tensor_name, index_str, prev_var_name).
    """
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.pallas import codegen as pallas_codegen

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("pallas atomic")

    name = state.device_function.tensor_arg(target).name
    index_str, _ = pallas_codegen.index_str(state, index, target)

    prev_var = state.device_function.new_var("_prev", dce=True)
    state.codegen.add_statement(
        statement_from_string(f"{prev_var} = {name}[{index_str}]")
    )
    return name, index_str, prev_var  # pyrefly: ignore[bad-return]


def _to_ast_values(values: list[object]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for v in values:
        if isinstance(v, (int, float, bool)):
            out.append(expr_from_string(constant_repr(v)))
        else:
            assert isinstance(v, ast.AST)
            out.append(v)
    return out


def _ref_atomic_binop(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Shared ref implementation for simple atomic binary ops (xchg/and/or/xor/max/min).

    Processes indices, clones the previous value, applies the op, and returns prev.
    For xchg, pass op=lambda old, val: val.
    """
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
    target[idx_tuple] = op(target[idx_tuple], val)
    return prev


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


@_decorators.codegen(atomic_add, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.compile_environment import CompileEnvironment

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    target = state.proxy_arg(0)
    assert isinstance(target, torch.Tensor)
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(target.dtype)
    # Cast the sum to the target dtype so the store doesn't fail when
    # the value dtype differs (e.g. float32 accumulator into bfloat16 ref).
    cast = backend.cast_expr(f"{prev_var} + {{value}}", target_dtype)
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {cast}", value=value_ast)
    )
    return expr_from_string(prev_var)


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
    return _ref_atomic_binop(target, index, value, lambda old, val: val)


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


@_decorators.codegen(atomic_xchg, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {{value}}", value=value_ast)
    )
    return expr_from_string(prev_var)


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
    return _ref_atomic_binop(target, index, value, torch.bitwise_and)


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


@_decorators.codegen(atomic_and, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} & {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


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
    return _ref_atomic_binop(target, index, value, torch.bitwise_or)


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


@_decorators.codegen(atomic_or, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} | {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


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
    return _ref_atomic_binop(target, index, value, torch.bitwise_xor)


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


@_decorators.codegen(atomic_xor, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} ^ {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


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
) -> torch.Tensor:
    _validate_sem(sem)
    return _ref_atomic_binop(target, index, value, torch.maximum)


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


@_decorators.codegen(atomic_max, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.maximum({prev_var}, {{value}})",
            value=value_ast,
        )
    )
    return expr_from_string(prev_var)


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
    return _ref_atomic_binop(target, index, value, torch.minimum)


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


@_decorators.codegen(atomic_min, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string

    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.minimum({prev_var}, {{value}})",
            value=value_ast,
        )
    )
    return expr_from_string(prev_var)


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

    # CAS always uses pointer (not a TMA reduction op, two values),
    # but increment the counter to keep per-op atomic_indexing aligned.
    device_fn = state.device_function
    device_fn.atomic_op_index += 1

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
    cmp_kw, val_kw = _resolve_cute_atomic_kwargs("atomic_cas", ["cmp", "val"])
    return expr_from_string(
        f"cute.arch.atomic_cas({{ptr}}, {cmp_kw}={{exp}}, {val_kw}={{val}}, sem={{sem}})",
        ptr=expr_from_string(pointer),
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )


@_decorators.codegen(atomic_cas, "pallas")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.pallas import codegen as pallas_codegen

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("pallas atomic_cas")

    name = state.device_function.tensor_arg(target).name
    index_str, _ = pallas_codegen.index_str(state, index, target)

    prev_var = state.device_function.new_var("_prev", dce=True)
    state.codegen.add_statement(
        statement_from_string(f"{prev_var} = {name}[{index_str}]")
    )

    exp_ast, val_ast = _to_ast_values([state.ast_args[2], state.ast_args[3]])
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.where({prev_var} == {{exp}}, {{val}}, {prev_var})",
            exp=exp_ast,
            val=val_ast,
        )
    )
    return expr_from_string(prev_var)


ATOMIC_OPS: frozenset[Callable[..., object]] = frozenset(
    {
        atomic_add,
        atomic_and,
        atomic_cas,
        atomic_max,
        atomic_min,
        atomic_or,
        atomic_xchg,
        atomic_xor,
    }
)


# --- NKI atomic_add codegen (calls load/store._codegen['nki'], from P1.17) ---
@_decorators.codegen(atomic_add, "nki")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import create
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.nki_backend import NKIOpOverrides
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
        if not isinstance(row_part, IndirectAP):
            return None
        vec_offset = row_part.vec_var

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


