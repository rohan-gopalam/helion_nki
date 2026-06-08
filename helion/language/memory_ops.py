from __future__ import annotations

import ast
import contextlib
import dataclasses
import logging
import operator
import textwrap
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect
from torch.fx.node import map_arg

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _symint_expr
from .._compiler.cute.cute_epilogue import Tcgen05UnaryEpilogueChain
from .._compiler.cute.cute_epilogue import _AuxiliaryTensorStep
from .._compiler.cute.cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .._compiler.cute.cute_fx_walk import reach_tcgen05_matmul_anchors
from .._compiler.cute.cutedsl_compat import emit_pipeline_advance
from .._compiler.cute.strategies import tcgen05_default_epilogue_tile_expr
from .._compiler.cute.strategies import tcgen05_explicit_d_store_tile_expr
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R,
)
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStoreBodyCoreParams
from .._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStorePipelineParams
from .._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStoreSubtileLoopParams
from .._compiler.cute.tcgen05_pure_matmul import Tcgen05TmaStoreTailParams
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.indexing_strategy import TileWithOffsetInfo
from .._compiler.indexing_strategy import _get_tile_with_offset_info
from .._compiler.pallas import codegen as pallas_codegen
from .._compiler.variable_origin import GridOrigin
from .._compiler.variable_origin import TileBeginOrigin
from .._compiler.variable_origin import TileCountOrigin
from .._compiler.variable_origin import TileEndOrigin
from .._compiler.variable_origin import TileIdOrigin
from . import _decorators
from ._nki_dim_access import DynamicAP
from ._nki_dim_access import IndirectAP
from .stack_tensor import StackTensor

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.tile_strategy import LoopDimInfo

from .._compiler.host_function import SymbolOrigin

# TileBeginWithOffset removed - using TileBeginWithOffsetPattern instead

__all__ = ["load", "store"]

log = logging.getLogger(__name__)


# Map short config names to full Triton API names for eviction policies
_EVICTION_POLICY_MAP = {
    "": None,
    "first": "evict_first",
    "last": "evict_last",
}


@dataclasses.dataclass(frozen=True)
class _AuxStepRecord:
    """Per-step splice-side AST locals for one auxiliary chain step.

    Holds the underlying aux tensor name, broadcast axis (None for
    exact-shape rank-2 aux), and the AST var names allocated for
    the partition pipeline. ``aux_view2d`` is set only for
    broadcast aux steps; exact-shape steps leave it ``None``. Used
    by ``_codegen_cute_store_tcgen05_tile`` to thread per-aux
    locals through the per-output-tile setup helper and the
    per-subtile load source helper.
    """

    aux_tensor_name: str
    broadcast_axis: int | None
    aux_tile: str
    aux_part_base: str
    aux_xfm: str
    aux_planned: str
    aux_epi: str
    aux_dtype: str
    aux_dtype_bits: int
    aux_extent: int | None
    ttr_aux: str
    ttr_aux_grouped: str
    ttr_aux_subtile: str
    aux_rmem: str
    aux_loaded: str
    aux_view2d: str | None


@dataclasses.dataclass(frozen=True)
class _RowvecAuxStageRecord:
    """Per-tile compact SMEM staging locals for one row-vector aux step."""

    smem_layout: str
    smem_ptr: str
    smem: str
    tiled_copy: str
    thr_copy: str
    gmem_tile: str
    gmem_part: str
    smem_part: str
    coord: str
    limit: str
    pred: str
    copy_bits: int
    copy_elems: int
    aux_extent: int


def _tcgen05_rowvec_aux_stage_copy_elems(
    aux_dtype_bits: int,
    block_n: int,
    aux_extent: int | None,
    *,
    copy_bits: int = 128,
) -> int | None:
    """Return the vector width when a row-vector aux can be staged safely."""

    if aux_extent is None or aux_dtype_bits <= 0:
        return None
    if copy_bits % aux_dtype_bits != 0:
        return None
    copy_elems = copy_bits // aux_dtype_bits
    if copy_elems <= 0:
        return None
    if block_n % copy_elems != 0 or aux_extent % copy_elems != 0:
        return None
    return copy_elems


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def store(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value to a tensor using a list of indices.

    This function is equivalent to `tensor[index] = value` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor / stack tensor to store to
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | tuple,
    list[object],
    torch.Tensor | torch.SymInt | float | int,
    torch.Tensor | None,
]:
    from .tile_proxy import Tile

    if isinstance(value, torch.Tensor) and value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = Tile._tiles_to_sizes_for_index(index)

    if isinstance(tensor, StackTensor):
        return (tuple(tensor), index, value, extra_mask)

    if isinstance(tensor, torch.Tensor):
        return (tensor, index, value, extra_mask)

    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    return None


@_decorators.codegen(store, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        fx_node = state.fx_node
        assert fx_node is not None
        epilogue_subtile_group_id = fx_node.meta.get("epilogue_subtile_group_id")
        if epilogue_subtile_group_id is None:
            indexing_idx = device_fn.allocate_store_index()
        elif fx_node.meta.get("epilogue_subtile_primary_output", False):
            indexing_idx = device_fn.allocate_store_index()
            device_fn.epilogue_subtile_store_indices[epilogue_subtile_group_id] = (
                indexing_idx
            )
        else:
            indexing_idx = device_fn.epilogue_subtile_store_indices[
                epilogue_subtile_group_id
            ]
        strategy = device_fn.get_indexing_strategy(indexing_idx)

        if state.codegen.store_transform is not None:
            return state.codegen.store_transform(
                state,
                tensor,
                [*subscript],
                value,
                extra_mask,
                strategy.codegen_store,
            )

        return strategy.codegen_store(state, tensor, [*subscript], value, extra_mask)
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack stores (multi-tensor device pointers);
        # fall through to the unfused path regardless of store_transform.
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


def _record_pad_info(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    block_id: int,
    extra_pad: int = 0,
) -> None:
    """Record that a tensor dimension uses pl.ds() and may need host-side padding.

    *extra_pad* accounts for non-zero loop begins: 0 when the loop starts
    at offset 0, ``begin % block_size`` for a constant begin, or
    ``block_size - 1`` for a data-dependent begin.

    Note: stores one entry per (tensor, dim).  If two inner loops tile the
    same dim with different block_ids, the last one wins.  This is fine when
    both loops use the same block size (the common case).
    """
    pad_info = state.device_function.pallas_pad_info
    tensor_id = id(tensor)
    if tensor_id not in pad_info:
        pad_info[tensor_id] = {}
    pad_info[tensor_id][tensor_dim] = (block_id, extra_pad)


def _maybe_get_symbol_origin(idx: object) -> SymbolOrigin | None:
    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if expr is None:
        return None
    return HostFunction.current().expr_to_origin.get(expr)


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    name = state.device_function.tensor_arg(tensor).name
    name = pallas_codegen.vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    value = pallas_codegen.sliced_value_for_store(
        state, tensor, subscript, parts, value
    )
    idx_str = ", ".join(parts)
    patterns = state.fx_node.meta.get("indexing_patterns") if state.fx_node else ()
    from .._compiler.pallas.gather import emit_scatter_store
    from .._compiler.pallas.plan_tiling import IndirectScatterPattern

    scatter_patterns = [
        pattern
        for pattern in patterns or ()
        if isinstance(pattern, IndirectScatterPattern)
    ]
    assert len(scatter_patterns) <= 1, (
        "Pallas store expected at most one indirect scatter pattern"
    )
    if scatter_patterns:
        value = emit_scatter_store(
            state, scatter_patterns[0].plan, name, idx_str, value
        )
    state.codegen.add_statement(
        statement_from_string(f"{name}[{idx_str}] = {{value}}", value=value)
    )


def _matching_block_ids(env: CompileEnvironment, size: object) -> list[int]:
    """Find all block_ids that match the given dimension size."""
    candidates: list[int] = []
    if isinstance(size, (int, torch.SymInt)):
        if (direct := env.get_block_id(size)) is not None:
            candidates.append(direct)
    if not isinstance(size, (int, torch.SymInt)):
        return candidates
    for info in env.block_sizes:
        if not isinstance(info.size, (int, torch.SymInt)):
            continue
        if not env.known_equal(info.size, size):
            continue
        if info.block_id not in candidates:
            candidates.append(info.block_id)
    return candidates


def _log_cute_layout(state: CodegenState, op_name: str) -> None:
    """Log the CuTe layout annotation for the current node, if any.

    This is used during CuTe load/store codegen to make layout info
    visible for debugging and future codegen integration.
    """
    layout = state.cute_layout
    if layout is None:
        return
    node_name = state.fx_node.name if state.fx_node else "?"
    log.debug(
        "cute %s %s: layout tag=%s thread=%s value=%s",
        op_name,
        node_name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _cute_remap_block_id(state: CodegenState, block_id: int) -> int:
    """Apply the active matmul-operand block-id remap, if any.

    Used while re-materializing a matmul operand load so its contraction
    dimension is indexed by the active contraction block instead of the
    loop-invariant block it was originally lowered with.  Returns *block_id*
    unchanged when no remap is active.
    """
    remap = state.device_function.cute_state.matmul_operand_block_remap
    if not remap:
        return block_id
    return remap.get(block_id, block_id)


def _cute_index_override(state: CodegenState, block_id: int) -> str | None:
    """Return a raw index-expression override for *block_id*, if active.

    Applied after ``_cute_remap_block_id``.  When set (only while
    re-materializing the rhs of a static-MN-collapse baddbmm), the operand's
    free (N) axis is indexed by this serial-loop variable instead of the shared
    M thread index, and masking for that axis is suppressed.
    """
    override = state.device_function.cute_state.matmul_operand_index_override
    if not override:
        return None
    return override.get(_cute_remap_block_id(state, block_id))


def _cute_active_index_var(state: CodegenState, block_id: int) -> str | None:
    if (override := _cute_index_override(state, block_id)) is not None:
        return override
    block_id = _cute_remap_block_id(state, block_id)
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.index_var(block_id)
    grid_state = state.codegen.current_grid_state
    if grid_state is not None and block_id in grid_state.block_ids:
        return grid_state.strategy.index_var(block_id)
    return None


def _cute_active_mask_var(state: CodegenState, block_id: int) -> str | None:
    if _cute_index_override(state, block_id) is not None:
        return None
    block_id = _cute_remap_block_id(state, block_id)
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.mask_var(block_id)
    return None


def _cute_unique_graph_block_id(state: CodegenState) -> int | None:
    fx_node = state.fx_node
    if fx_node is None:
        return None
    graph_block_ids = [
        graph_info.block_ids
        for graph_info in state.codegen.codegen_graphs
        if graph_info.graph is fx_node.graph and hasattr(graph_info, "block_ids")
    ]
    if len(graph_block_ids) != 1 or len(graph_block_ids[0]) != 1:
        return None
    (block_id,) = graph_block_ids[0]
    return block_id


def _maybe_codegen_cute_packed_affine_lhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> object | None:
    from .._compiler.cute.indexing import CutePackedAffineLoad
    from .._compiler.cute.indexing import match_cute_affine_range_iota
    from .._compiler.cute.indexing import match_cute_stack_reshape_rhs
    from .matmul_ops import dot

    fx_node = state.fx_node
    if (
        fx_node is None
        or len(fx_node.users) != 1
        or len(subscript) not in (2, 3)
        or len(fx_node.args) < 2
    ):
        return None

    fx_subscript = fx_node.args[1]
    if not isinstance(fx_subscript, (list, tuple)) or len(fx_subscript) != len(
        subscript
    ):
        return None
    range_node = fx_subscript[-1]
    if not isinstance(range_node, torch.fx.Node):
        return None
    affine_range = match_cute_affine_range_iota(range_node)
    if affine_range is None:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target not in {
        dot,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }:
        return None

    rhs_index = (
        2
        if user.target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default)
        else 1
    )
    rhs_arg = user.args[rhs_index]
    if not isinstance(rhs_arg, torch.fx.Node):
        return None
    packed_rhs = match_cute_stack_reshape_rhs(rhs_arg)
    if packed_rhs is None:
        return None
    _, factor = packed_rhs
    if factor != affine_range.factor:
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-1]]
    row_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(row_index_exprs) != len(leading_subscript):
        return None

    tensor_name = state.device_function.tensor_arg(tensor).name
    mask_terms: list[str] = []
    row_mask = _cute_combined_mask(state, leading_subscript, extra_mask, tensor=tensor)
    if row_mask is not None:
        mask_terms.append(row_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    mask_expr = " and ".join(mask_terms) if mask_terms else None
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    terms: list[ast.AST] = []
    for offset in range(factor):
        index_expr = ", ".join(
            [
                *row_index_exprs,
                f"cutlass.Int32({factor}) * ({packed_index}) + cutlass.Int32({offset})",
            ]
        )
        term = expr_from_string(f"{tensor_name}[{index_expr}]")
        if mask_expr is not None:
            term = expr_from_string(
                f"({{value}} if {mask_expr} else {zero}(0))",
                value=term,
            )
        terms.append(term)
    return CutePackedAffineLoad(tuple(terms))


def _maybe_codegen_cute_packed_rhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    from .._compiler.cute.indexing import match_cute_duplicate_stack_reshape_rhs

    fx_node = state.fx_node
    if fx_node is None or len(subscript) not in (2, 3) or len(fx_node.users) != 1:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target is not torch.ops.aten.stack.default:
        return None
    stack_users = list(user.users)
    if len(stack_users) != 1 or not isinstance(stack_users[0], torch.fx.Node):
        return None
    rhs_node = stack_users[0]
    packed_rhs = match_cute_duplicate_stack_reshape_rhs(rhs_node)
    if packed_rhs != (
        fx_node,
        len(user.args[0]) if isinstance(user.args[0], (list, tuple)) else 0,
    ):
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-2]]
    col_index_exprs = _cute_index_exprs(
        state,
        [subscript[-1]],
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(col_index_exprs) != 1:
        return None
    (col_index,) = col_index_exprs
    leading_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(leading_index_exprs) != len(leading_subscript):
        return None
    tensor_name = state.device_function.tensor_arg(tensor).name
    load_index_expr = ", ".join([*leading_index_exprs, packed_index, col_index])
    load_expr: ast.AST = expr_from_string(f"{tensor_name}[{load_index_expr}]")
    mask_terms: list[str] = []
    col_mask = _cute_combined_mask(
        state,
        [*leading_subscript, subscript[-1]],
        extra_mask,
        tensor=tensor,
    )
    if col_mask is not None:
        mask_terms.append(col_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    if not mask_terms:
        return load_expr
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(
        f"({{value}} if {' and '.join(mask_terms)} else {zero}(0))",
        value=load_expr,
    )


def _cute_index_exprs(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...] | None = None,
    tensor: torch.Tensor | None = None,
    *,
    inactive_slice_expr: str | None = None,
    inactive_singleton_slice_expr: str | None = None,
) -> list[str]:
    env = CompileEnvironment.current()

    def symint_index_expr(idx: torch.SymInt, used_block_ids: set[int]) -> str:
        expr = _symint_expr(idx)
        if expr is not None:
            origin_info = HostFunction.current().expr_to_origin.get(expr)
            if origin_info is not None and isinstance(origin_info.origin, GridOrigin):
                if type(origin_info.origin) is not GridOrigin:
                    block_id = origin_info.origin.block_id
                    loop_info = active_loop_info(block_id)
                    begin_var = tile_begin_expr(block_id, loop_info)
                    block_size_var = (
                        state.device_function.block_size_var(block_id) or "1"
                    )
                    if isinstance(origin_info.origin, TileBeginOrigin):
                        return begin_var
                    if isinstance(origin_info.origin, TileEndOrigin):
                        if loop_info is not None and loop_info.end_var_name is not None:
                            return env.backend.minimum_expr(
                                f"({begin_var}) + ({block_size_var})",
                                loop_info.end_var_name,
                            )
                        return f"({begin_var}) + ({block_size_var})"
                    if isinstance(origin_info.origin, TileCountOrigin):
                        end_var = (
                            loop_info.end_var_name
                            if loop_info is not None
                            and loop_info.end_var_name is not None
                            else f"({begin_var}) + ({block_size_var})"
                        )
                        extent = f"({end_var}) - ({begin_var})"
                        return env.backend.cdiv_expr(
                            extent, block_size_var, is_device=True
                        )
                    if isinstance(origin_info.origin, TileIdOrigin):
                        if block_size_var == "1":
                            return begin_var
                        return f"({begin_var}) // ({block_size_var})"
                    return state.sympy_expr(expr)
        block_id = env.get_block_id(idx)
        if block_id is not None:
            used_block_ids.add(block_id)
            return index_var_for_block_id(block_id, idx)
        if expr is not None:
            return state.sympy_expr(expr)
        raise exc.BackendUnsupported("cute", f"unlowerable symbolic index: {idx}")

    def active_loop_info(block_id: int) -> LoopDimInfo | None:
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].block_id_to_info.get(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            return grid_state.block_id_to_info.get(block_id)
        return None

    def active_local_coord(block_id: int) -> str | None:
        from .._compiler.cute.cute_reshape import _grid_local_coord_expr

        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            thread_axis = loops[-1].block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        return None

    def tile_begin_expr(block_id: int, loop_info: LoopDimInfo | None) -> str:
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return state.codegen.offset_var(block_id)
        begin_var = "0"
        if loop_info is not None and loop_info.begin_var_name is not None:
            begin_var = loop_info.begin_var_name
        global_index = active_index_var(block_id)
        local_coord = active_local_coord(block_id)
        if global_index is not None and local_coord is not None:
            return state.codegen.lift(
                expr_from_string(f"({global_index}) - ({local_coord})"),
                dce=True,
                prefix="tile_begin",
            ).id
        if global_index is not None:
            return global_index
        return begin_var

    def active_index_var(block_id: int) -> str | None:
        if (override := _cute_index_override(state, block_id)) is not None:
            return override
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
        return None

    def resolve_active_slice_block_id(
        size: object,
        used_block_ids: set[int],
    ) -> int | None:
        candidates = _matching_block_ids(env, size)
        active_candidates = [
            block_id
            for block_id in candidates
            if active_index_var(block_id) is not None
        ]
        active_unused_candidates = [
            block_id for block_id in active_candidates if block_id not in used_block_ids
        ]
        if len(active_unused_candidates) == 1:
            return active_unused_candidates[0]
        if len(active_candidates) == 1:
            return active_candidates[0]
        if len(active_unused_candidates) > 1:
            reduction_unused = [
                block_id
                for block_id in active_unused_candidates
                if env.block_sizes[block_id].reduction
            ]
            if len(reduction_unused) == 1:
                return reduction_unused[0]
        if len(active_candidates) > 1:
            reduction_active = [
                block_id
                for block_id in active_candidates
                if env.block_sizes[block_id].reduction
            ]
            if len(reduction_active) == 1:
                return reduction_active[0]
        return None

    def index_var_for_block_id(block_id: int, size: object) -> str:
        if (idx_var := active_index_var(block_id)) is not None:
            return idx_var

        raise exc.BackendUnsupported(
            "cute",
            (
                "indexing dimension is not active in this scope "
                f"(block_id={block_id}, size={size})"
            ),
        )

    def local_coord_for_block_id(block_id: int, begin_var: str) -> str | None:
        if (local_coord := active_local_coord(block_id)) is not None:
            return local_coord
        if (idx_var := active_index_var(block_id)) is not None:
            return f"({idx_var}) - ({begin_var})"
        return None

    def tile_with_offset_index_expr(tile_info: TileWithOffsetInfo) -> str:
        block_id = tile_info.block_id
        begin_var = tile_begin_expr(block_id, active_loop_info(block_id))
        local_coord = local_coord_for_block_id(block_id, begin_var)
        if local_coord is None:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "indexing dimension is not active in this scope "
                    f"(block_id={block_id})"
                ),
            )
        offset_expr = state.device_function.literal_expr(tile_info.offset)
        return f"({begin_var}) + cutlass.Int32({offset_expr}) + ({local_coord})"

    used_block_ids = {
        block_id
        for idx in subscript
        if isinstance(idx, torch.SymInt)
        if (block_id := env.get_block_id(idx)) is not None
    }
    result = []
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        ast_idx = None
        if ast_subscript is not None:
            ast_idx = ast_subscript[pos]
        if idx is None:
            continue
        if (
            tensor is not None
            and tensor_dim < tensor.ndim
            and env.known_equal(tensor.shape[tensor_dim], 1)
            and not (isinstance(idx, slice) and idx == slice(None))
        ):
            result.append("0")
            tensor_dim += 1
            continue
        if (
            tile_info := _get_tile_with_offset_info(
                idx, getattr(state, "fx_node", None), pos
            )
        ) is not None and tile_info.block_size is not None:
            used_block_ids.add(tile_info.block_id)
            result.append(tile_with_offset_index_expr(tile_info))
            tensor_dim += 1
            continue
        if isinstance(idx, torch.SymInt):
            result.append(symint_index_expr(idx, used_block_ids))
            tensor_dim += 1
        elif isinstance(idx, int):
            result.append(str(idx))
            tensor_dim += 1
        elif isinstance(idx, torch.Tensor):
            from .._compiler.cute.indexing import CuteAffineRangeIndex

            if isinstance(ast_idx, CuteAffineRangeIndex):
                raise exc.BackendUnsupported(
                    "cute",
                    "affine hl.arange() indexing is only supported in CuTe packed-matmul load fusion",
                )
            if not isinstance(ast_idx, ast.AST):
                raise exc.BackendUnsupported(
                    "cute", f"tensor index without AST at position {pos}"
                )
            lifted = state.codegen.lift(ast_idx, dce=True, prefix="index")
            index_dtype = env.backend.dtype_str(env.index_dtype)
            result.append(f"{index_dtype}({lifted.id})")
            tensor_dim += 1
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor is None:
                raise exc.BackendUnsupported("cute", "slice indexing without tensor")
            dim_size = tensor.shape[tensor_dim]
            block_id = resolve_active_slice_block_id(dim_size, used_block_ids)
            if block_id is not None:
                idx_var = active_index_var(block_id)
                assert idx_var is not None
                used_block_ids.add(block_id)
                result.append(idx_var)
                tensor_dim += 1
                continue
            if inactive_singleton_slice_expr is not None and env.known_equal(
                dim_size, 1
            ):
                result.append(inactive_singleton_slice_expr)
                tensor_dim += 1
                continue
            if inactive_slice_expr is None:
                raise exc.BackendUnsupported(
                    "cute",
                    (
                        "indexing dimension is not active in this scope "
                        f"(tensor_dim={pos}, size={dim_size})"
                    ),
                )
            result.append(inactive_slice_expr)
            tensor_dim += 1
        else:
            raise exc.BackendUnsupported("cute", f"index type: {type(idx)}")
    return result


def _cute_index_tuple(index_exprs: list[str]) -> str:
    if len(index_exprs) == 1:
        return f"({index_exprs[0]},)"
    return f"({', '.join(index_exprs)})"


def _cute_scalar_pointer_expr(tensor_name: str, index_exprs: list[str]) -> str:
    env = CompileEnvironment.current()
    index_dtype = env.index_type()
    offset = " + ".join(
        f"({index_dtype}({index}) * {index_dtype}({tensor_name}.layout.stride[{dim}]))"
        for dim, index in enumerate(index_exprs)
    )
    return f"({tensor_name}.iterator + {offset})"


def _cute_scalar_storage_dtype(dtype: torch.dtype) -> str:
    if dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return "cutlass.Uint8"
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _cute_scalar_load_expr(
    tensor_name: str,
    index_exprs: list[str],
    dtype: torch.dtype,
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}[{', '.join(index_exprs)}]"
    if dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return (
            f"cute.arch.load({_cute_scalar_pointer_expr(tensor_name, index_exprs)}, "
            "cutlass.Uint8)"
        )
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.load()"


def _cute_scalar_store_expr(
    tensor_name: str, index_exprs: list[str], value: str
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}.__setitem__({_cute_index_tuple(index_exprs)}, {value})"
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.store({value})"


# Maximum bytes per vector load/store transaction (LDG.128/STG.128).
_CUTE_VECTOR_MAX_BYTES = 16

# Dtype -> (cutlass scalar type name, max vector width).  Used for the
# ``vec`` mode that issues an explicit
# ``cute.arch.load(ptr, ir.VectorType.get([V], elem.mlir_type))`` and folds
# the result via ``_cute_pre_vec_fold``.
_CUTE_VECTOR_DTYPES: dict[torch.dtype, tuple[str, int]] = {
    torch.float32: ("cutlass.Float32", _CUTE_VECTOR_MAX_BYTES // 4),
    torch.float16: ("cutlass.Float16", _CUTE_VECTOR_MAX_BYTES // 2),
    torch.bfloat16: ("cutlass.BFloat16", _CUTE_VECTOR_MAX_BYTES // 2),
}

# ``unroll`` mode loads bf16/fp16 inputs as Uint16 vectors and bitcasts each
# extracted lane back to the original dtype.  This avoids the CuTe DSL
# crash that fires when subscripting a bf16/fp16 vector value.  Cutlass
# scalar type for the extracted lane is paired with the vec-element type
# name used in ``ir.VectorType.get``.
_CUTE_VECTOR_UNROLL_DTYPES: dict[torch.dtype, str] = {
    torch.float16: "cutlass.Float16",
    torch.bfloat16: "cutlass.BFloat16",
}


def _cute_vector_load_expr(
    tensor_name: str,
    index_exprs: list[str],
    dtype: torch.dtype,
    *,
    vec_width: int,
) -> str:
    elem_str, _ = _CUTE_VECTOR_DTYPES[dtype]
    ptr = _cute_scalar_pointer_expr(tensor_name, index_exprs)
    return (
        f"cute.arch.load({ptr}, ir.VectorType.get([{vec_width}], {elem_str}.mlir_type))"
    )


def _cute_vector_store_expr(
    tensor_name: str,
    index_exprs: list[str],
    value: str,
    dtype: torch.dtype,
    *,
    vec_width: int,
) -> str:
    elem_str, _ = _CUTE_VECTOR_DTYPES[dtype]
    ptr = _cute_scalar_pointer_expr(tensor_name, index_exprs)
    return (
        f"cute.arch.store({ptr}, {value}, "
        f"ir.VectorType.get([{vec_width}], {elem_str}.mlir_type))"
    )


def _cute_register_unroll_vec_hoist(
    state: CodegenState,
    strategy: object,  # LoopedReductionStrategy at runtime
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
) -> str:
    """Register a Uint16 vec load to be hoisted above the constexpr V-loop
    in the active lane body and return the per-element extract expression.

    The hoist runs once per outer-lane iter; the constexpr V-loop's body
    receives ``hoist_var[vi].bitcast(dtype)`` (a scalar) so the existing
    cast/mul/accumulate pipeline keeps working unchanged.
    """
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
    base_index_var = getattr(strategy, "_cute_lane_base_index_var", None)
    lane_body = getattr(strategy, "_cute_lane_body", None)
    assert isinstance(base_index_var, str)
    assert isinstance(lane_body, list)
    # The inner reduction-axis index_expr is the last entry; swap it with
    # the per-lane base so the vec load points at the start of the V-wide
    # chunk this thread owns.
    base_exprs = list(index_exprs)
    base_exprs[-1] = base_index_var
    base_ptr_expr = _cute_scalar_pointer_expr(tensor_name, base_exprs)
    cache_key = (tensor_name, base_ptr_expr)
    cache = getattr(strategy, "_cute_lane_vec_loads", None)
    if cache is None:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_loads = cache
    if cache_key not in cache:
        hoist_var = state.device_function.new_var(
            f"_unroll_vec_{len(cache)}", dce=False
        )
        cache[cache_key] = (hoist_var, tensor.dtype)
        hoist_stmt = statement_from_string(
            f"{hoist_var} = cute.arch.load({base_ptr_expr}, "
            f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type))"
        )
        # Insert the hoist just BEFORE the constexpr V-loop (the last entry
        # in lane_body).  ``lane_body[-1]`` is the constexpr loop.
        lane_body.insert(len(lane_body) - 1, hoist_stmt)
    else:
        hoist_var, _ = cache[cache_key]
    # The constexpr V-loop's target var is the last element's loop var.
    constexpr_loop = lane_body[-1]
    assert isinstance(constexpr_loop, ast.For)
    assert isinstance(constexpr_loop.target, ast.Name)
    vec_lane_var = constexpr_loop.target.id
    return f"cutlass.Uint16({hoist_var}[{vec_lane_var}]).bitcast({elem_dtype})"


def _cute_register_tile_unroll_vec_hoist(
    state: CodegenState,
    strategy: object,  # BlockSizeTileStrategy (CuteNDTileStrategy)
    block_id: int,
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
) -> str:
    """Tile-loop variant of ``_cute_register_unroll_vec_hoist`` for
    ``CuteNDTileStrategy`` lane loops.

    Splices a single ``cute.arch.load(base_ptr, Uint16x V)`` into the
    outer-lane body (above the constexpr V-loop) and returns the
    per-element bitcast expression ``hoist_var[vi].bitcast(dtype)`` so
    the existing scalar pipeline keeps working.
    """
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
    base_var_by_block = getattr(strategy, "_cute_lane_base_index_var_by_block", {})
    lane_body_by_block = getattr(strategy, "_cute_lane_body_by_block", {})
    vec_lane_var_by_block = getattr(strategy, "_cute_vec_lane_var_by_block", {})
    base_index_var = base_var_by_block.get(block_id)
    lane_body = lane_body_by_block.get(block_id)
    vec_lane_var = vec_lane_var_by_block.get(block_id)
    assert isinstance(base_index_var, str)
    assert isinstance(lane_body, list)
    assert isinstance(vec_lane_var, str)
    # The inner reduction-axis index_expr is the last entry; swap it
    # with the per-lane base so the vec load points at the start of the
    # V-wide chunk this thread owns.
    base_exprs = list(index_exprs)
    base_exprs[-1] = base_index_var
    base_ptr_expr = _cute_scalar_pointer_expr(tensor_name, base_exprs)
    cache_key = (tensor_name, base_ptr_expr)
    cache_by_block = getattr(strategy, "_cute_lane_vec_loads_by_block", None)
    if cache_by_block is None:
        cache_by_block = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_loads_by_block = cache_by_block
    cache = cache_by_block.setdefault(block_id, {})
    if cache_key not in cache:
        hoist_var = state.device_function.new_var(
            f"_tile_unroll_vec_{block_id}_{len(cache)}", dce=False
        )
        cache[cache_key] = (hoist_var, tensor.dtype)
        # Guard the LDG against per-thread OOB: on the very last grid
        # block + tail outer-tile iter, a thread whose vec base equals
        # ``numel`` would otherwise read past the end of the underlying
        # allocation (the next row doesn't exist for the last grid
        # block).  Use an "anchor pointer" fallback for the unsafe
        # threads: it points inside the tensor (specifically at the
        # per-thread base of the FIRST outer-tile iter, which is the
        # ``base_ptr_expr`` with the outer-lane index folded to 0).  The
        # fetched bytes are then ignored downstream by the per-lane
        # mask gate that wraps the bitcast result.
        env_local = CompileEnvironment.current()
        numel = env_local.block_sizes[block_id].numel
        numel_expr = state.sympy_expr(numel)
        # Build the "anchor" pointer: same index_exprs but with the
        # inner reduction-axis index forced to 0.  This is the
        # ``tile_offset == 0, lane_var == 0, vec_lane_var == 0`` base
        # for the very first outer-tile iter, which is always in-bounds
        # for any grid block.
        anchor_exprs = list(index_exprs)
        anchor_exprs[-1] = "0"
        anchor_ptr_expr = _cute_scalar_pointer_expr(tensor_name, anchor_exprs)
        guarded_ptr = (
            f"({base_ptr_expr} if {base_index_var} < {numel_expr} "
            f"else {anchor_ptr_expr})"
        )
        hoist_stmt = statement_from_string(
            f"{hoist_var} = cute.arch.load({guarded_ptr}, "
            f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type))"
        )
        # Insert the hoist just BEFORE the constexpr V-loop (the last
        # entry in lane_body).
        lane_body.insert(len(lane_body) - 1, hoist_stmt)
    else:
        hoist_var, _ = cache[cache_key]
    return f"cutlass.Uint16({hoist_var}[{vec_lane_var}]).bitcast({elem_dtype})"


def _cute_register_tile_unroll_vec_hoist_split2(
    state: CodegenState,
    strategy: object,  # BlockSizeTileStrategy (CuteNDTileStrategy)
    block_id: int,
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
) -> str:
    """Split-2 variant of ``_cute_register_tile_unroll_vec_hoist`` for V=8
    on fp16/bf16.

    The CuTe DSL's ``nvvm.load.ext`` ICEs at V=8 for these dtypes, so the
    full 16-byte LDG.128 is decomposed into TWO back-to-back V=4 loads
    (lanes 0-3 and 4-7).  The SASS scheduler is free to overlap the two
    LDGs, so the per-thread bytes-per-load grows from 8 (V=4) to the
    full 16 (effective V=8) without invoking the DSL bug.

    Returns a per-vec-lane expression of the form::

        (
            cutlass.Uint16(_tile_unroll_vec_ < n > _ < m > _a[vi]).bitcast(dtype)
            if vi < 4
            else cutlass.Uint16(_tile_unroll_vec_ < n > _ < m > _b[vi - 4]).bitcast(
                dtype
            )
        )

    Because ``vec_lane_var`` is the target of a ``cutlass.range_constexpr(8)``
    loop, it is a Python-int constant at each unrolled iter, so the
    ``if vi < 4`` branch folds away at trace time and the emitted SASS
    contains only the active load's extract.
    """
    assert vec_width == 8, (
        "tile_unroll_split2 expects V=8 (4+4); other widths use tile_unroll"
    )
    half = vec_width // 2
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
    base_var_by_block = getattr(strategy, "_cute_lane_base_index_var_by_block", {})
    lane_body_by_block = getattr(strategy, "_cute_lane_body_by_block", {})
    vec_lane_var_by_block = getattr(strategy, "_cute_vec_lane_var_by_block", {})
    base_index_var = base_var_by_block.get(block_id)
    lane_body = lane_body_by_block.get(block_id)
    vec_lane_var = vec_lane_var_by_block.get(block_id)
    assert isinstance(base_index_var, str)
    assert isinstance(lane_body, list)
    assert isinstance(vec_lane_var, str)
    base_exprs = list(index_exprs)
    base_exprs[-1] = base_index_var
    base_ptr_expr_a = _cute_scalar_pointer_expr(tensor_name, base_exprs)
    # The second-half pointer points 4 elements past the first.  Build
    # it by substituting ``base_index_var + half`` for the inner index.
    base_exprs_b = list(index_exprs)
    base_exprs_b[-1] = f"({base_index_var} + {half})"
    base_ptr_expr_b = _cute_scalar_pointer_expr(tensor_name, base_exprs_b)
    cache_key = (tensor_name, base_ptr_expr_a, "split2")
    cache_by_block = getattr(strategy, "_cute_lane_vec_loads_by_block", None)
    if cache_by_block is None:
        cache_by_block = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_loads_by_block = cache_by_block
    cache = cache_by_block.setdefault(block_id, {})
    if cache_key not in cache:
        slot = len(cache)
        hoist_var_a = state.device_function.new_var(
            f"_tile_unroll_vec_{block_id}_{slot}_a", dce=False
        )
        hoist_var_b = state.device_function.new_var(
            f"_tile_unroll_vec_{block_id}_{slot}_b", dce=False
        )
        # Stash both names plus the split marker so this entry doesn't
        # collide with the V=4 cache_key shape.  Downstream readers
        # don't introspect this tuple — it's just a sentinel.
        cache[cache_key] = ((hoist_var_a, hoist_var_b), tensor.dtype)
        env_local = CompileEnvironment.current()
        numel = env_local.block_sizes[block_id].numel
        numel_expr = state.sympy_expr(numel)
        anchor_exprs = list(index_exprs)
        anchor_exprs[-1] = "0"
        anchor_ptr_expr = _cute_scalar_pointer_expr(tensor_name, anchor_exprs)
        # The first-half OOB guard checks the same V-aligned base used by
        # the V=4 path; the second-half pointer is ``base + 4`` and only
        # needs guarding when ``base + 4 < numel``.  Reuse the same
        # anchor pointer for both halves' fallbacks (the per-element
        # mask gate downstream drops any anchor-fetched bytes anyway).
        guarded_ptr_a = (
            f"({base_ptr_expr_a} if {base_index_var} < {numel_expr} "
            f"else {anchor_ptr_expr})"
        )
        guarded_ptr_b = (
            f"({base_ptr_expr_b} if ({base_index_var} + {half}) < {numel_expr} "
            f"else {anchor_ptr_expr})"
        )
        hoist_stmt_a = statement_from_string(
            f"{hoist_var_a} = cute.arch.load({guarded_ptr_a}, "
            f"ir.VectorType.get([{half}], cutlass.Uint16.mlir_type))"
        )
        hoist_stmt_b = statement_from_string(
            f"{hoist_var_b} = cute.arch.load({guarded_ptr_b}, "
            f"ir.VectorType.get([{half}], cutlass.Uint16.mlir_type))"
        )
        # Insert both hoists just BEFORE the constexpr V-loop (the last
        # entry in lane_body).  Emit them back-to-back so the SASS
        # scheduler can issue the two LDGs together.
        lane_body.insert(len(lane_body) - 1, hoist_stmt_a)
        lane_body.insert(len(lane_body) - 1, hoist_stmt_b)
    else:
        (hoist_var_a, hoist_var_b), _ = cache[cache_key]
    return (
        f"(cutlass.Uint16({hoist_var_a}[{vec_lane_var}]).bitcast({elem_dtype}) "
        f"if {vec_lane_var} < {half} "
        f"else cutlass.Uint16({hoist_var_b}[{vec_lane_var} - {half}]).bitcast({elem_dtype}))"
    )


def _cute_vector_load_ctx(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_exprs: list[str],
    extra_mask: ast.AST | None,
) -> tuple[int, int, str] | None:
    """Return (vec_width, lane_block_id, mode) when a vec load may be emitted.

    ``mode`` is one of ``"vec"`` (explicit ``cute.arch.load(..., V)``) or
    ``"unroll"`` (per-element scalar bitcast inside a constexpr V-loop).
    Returns None when any predicate for a 128-bit gmem load fails, in which
    case the caller falls back to ``_cute_scalar_load_expr``.
    """
    from .._compiler.reduction_strategy import LoopedReductionStrategy

    env = CompileEnvironment.current()
    if env.backend.name != "cute":
        return None
    if extra_mask is not None:
        return None
    if "None" in index_exprs:
        return None
    if (
        tensor.dtype not in _CUTE_VECTOR_DTYPES
        and tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES
    ):
        return None
    # Only enable the vec path when the load's result eventually feeds a
    # reduction op.  The consume-sweep mixes the loaded vector with scalar
    # values (e.g. the post-reduction inverse-RMS), and broadcasting
    # scalar->vec is not supported by the CuTe DSL today.  When the load's
    # immediate user is a dtype cast (``to(torch.float32)``), the
    # ``"unroll"`` mode further down keeps the strategy on a per-element
    # scalar pipeline and the explicit-vec path is skipped — the explicit
    # ``cute.arch.load(ptr, ir.VectorType.get([V], dtype.mlir_type))`` form
    # would otherwise crash inside the CuTe DSL when subscripting bf16/fp16
    # vectors.
    fx_node = state.fx_node
    if fx_node is None:
        return None
    visited: set[torch.fx.Node] = set()
    pending = list(fx_node.users.keys())
    feeds_reduction = False
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        target_name = getattr(user.target, "__name__", "") or ""
        target_qualname = getattr(user.target, "_qualname", "") or ""
        if (
            "reduction" in target_name
            or "_inductor_lowering_extra" in target_name
            or "reduction" in target_qualname
        ):
            feeds_reduction = True
            break
        pending.extend(user.users.keys())
    # Note: ``feeds_reduction`` is required ONLY for the ``vec`` mode below;
    # the ``unroll`` mode also applies to the consume sweep where the load
    # result feeds an elementwise pipeline (no reduction).
    # The innermost dim of the load must be the reduction lane axis and
    # the tensor must be stride-1 in that dim so that consecutive lane
    # iters fetch consecutive bytes.
    try:
        if int(tensor.stride(-1)) != 1:
            return None
    except (TypeError, ValueError):
        return None
    # Locate the innermost (last) non-None subscript and pull the active
    # block_id off it.  Slices resolve to the matching tensor-dim block via
    # the strategy that's currently active for that block.
    inner_block_id: int | None = None
    tensor_dim = 0
    for idx in subscript:
        if idx is None:
            continue
        if isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None:
                inner_block_id = bid
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor_dim < tensor.ndim:
                dim_size = tensor.shape[tensor_dim]
                for cand_bid, bs in enumerate(env.block_sizes):
                    if not isinstance(bs.size, (int, torch.SymInt)):
                        continue
                    bs_numel = bs.numel
                    # Try a few candidate forms for the size equality
                    # check: sympy.Integer (most common via specialize()),
                    # int, and torch.SymInt all flow through known_equal
                    # after we coerce to plain int when possible.
                    bs_int: int | torch.SymInt | None
                    if isinstance(bs_numel, (int, torch.SymInt)):
                        bs_int = bs_numel
                    else:
                        try:
                            bs_int = int(bs_numel)
                        except (TypeError, ValueError):
                            bs_int = None
                    if bs_int is None:
                        continue
                    dim_int: int | torch.SymInt | None
                    if isinstance(dim_size, (int, torch.SymInt)):
                        dim_int = dim_size
                    else:
                        try:
                            dim_int = int(dim_size)
                        except (TypeError, ValueError):
                            dim_int = None
                    if dim_int is None:
                        continue
                    if env.known_equal(
                        bs_int, dim_int
                    ) and state.codegen.active_device_loops.get(cand_bid):
                        inner_block_id = cand_bid
                        break
        tensor_dim += 1
    if inner_block_id is None:
        return None
    loops = state.codegen.active_device_loops.get(inner_block_id)
    if not loops:
        return None
    strategy = getattr(loops[-1], "strategy", None)
    if isinstance(strategy, LoopedReductionStrategy):
        vec_width = getattr(strategy, "_cute_reduction_vec_width", 1)
        if vec_width <= 1:
            return None
        if strategy._mask_var is not None:
            return None
        if strategy._cute_reduction_lane_extent <= 0:
            return None
        mode = getattr(strategy, "_cute_reduction_vec_mode", "vec")
        if mode == "vec":
            if not feeds_reduction:
                return None
            if tensor.dtype not in _CUTE_VECTOR_DTYPES:
                return None
            return vec_width, inner_block_id, "vec"
        if mode == "unroll":
            if tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES:
                return None
            # The CuTe DSL's ``nvvm.load.ext`` only supports vec sizes 2
            # and 4 for bf16/fp16 (V=8 raises ICE).  Cap effective V
            # here so the autotuner's V=8 seed still compiles instead
            # of crashing.
            if vec_width > 4:
                return None
            # Need a lane base index var + a constexpr V-loop var; both
            # are set up by the strategy's codegen_device_loop.
            if (
                getattr(strategy, "_cute_lane_base_index_var", None) is None
                or getattr(strategy, "_cute_lane_body", None) is None
            ):
                return None
            return vec_width, inner_block_id, "unroll"
        return None
    # CuTe N-D tile strategy with lane loops: vec is set up per-block in
    # ``CuteNDTileStrategy.__init__`` when the autotuner picks
    # ``cute_vector_widths[block_id]`` > 1 and EPT is divisible by V.  Mode
    # is forced to ``"unroll"`` (per-element bitcast) for fp16/bf16 since
    # subscripting a bf16/fp16 vector in the CuTe DSL is unsafe; fp32
    # could in principle use ``"vec"`` but the per-element pipeline runs
    # most of the consume-sweep code after a cast, so unroll is the
    # robust choice.
    from .._compiler.tile_strategy import BlockSizeTileStrategy

    if isinstance(strategy, BlockSizeTileStrategy):
        vec_by_block = getattr(strategy, "_cute_lane_vec_width_by_block", None)
        if not isinstance(vec_by_block, dict):
            return None
        vec_width = vec_by_block.get(inner_block_id, 1)
        if vec_width <= 1:
            return None
        if tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES:
            return None
        # The CuTe DSL's ``nvvm.load.ext`` ICEs at V=8 for fp16/bf16, so
        # widths > 4 cannot use a single ``cute.arch.load``.  V=8 still
        # gets full LDG.128 throughput via the ``tile_unroll_split2``
        # mode: two back-to-back ``cute.arch.load(..., V=4)`` calls
        # (covering vec lanes 0-3 and 4-7) emit as two LDG.64s that the
        # SASS scheduler can overlap.  Wider Vs (16, 32, ...) are not
        # supported.
        if vec_width > 8:
            return None
        if vec_width == 8 and vec_width % 4 != 0:
            return None
        base_var_by_block = getattr(
            strategy, "_cute_lane_base_index_var_by_block", None
        )
        lane_body_by_block = getattr(strategy, "_cute_lane_body_by_block", None)
        vec_lane_var_by_block = getattr(strategy, "_cute_vec_lane_var_by_block", None)
        if (
            not isinstance(base_var_by_block, dict)
            or not isinstance(lane_body_by_block, dict)
            or not isinstance(vec_lane_var_by_block, dict)
            or inner_block_id not in base_var_by_block
            or inner_block_id not in lane_body_by_block
            or inner_block_id not in vec_lane_var_by_block
        ):
            return None
        # When the per-thread vec base could straddle the tensor edge
        # (e.g. ``numel`` not a multiple of V), the masked-tail iter
        # could load garbage in some lanes.  Gate the per-element mask
        # path correctly by requiring ``numel % V == 0`` so partial-vec
        # straddles are impossible.
        numel = env.block_sizes[inner_block_id].numel
        if not env.known_multiple(numel, vec_width):
            return None
        if vec_width == 8:
            return vec_width, inner_block_id, "tile_unroll_split2"
        return vec_width, inner_block_id, "tile_unroll"
    return None


def _cute_stack_tensor_offset_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    subscript: list[object],
    ast_subscript: list[object] | tuple[object, ...],
) -> str:
    env = CompileEnvironment.current()
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor_like,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in index_exprs:
        raise exc.BackendUnsupported("cute", "inactive stack tensor load dimension")
    index_dtype = env.index_type()
    terms = []
    for dim, index in enumerate(index_exprs):
        stride = tensor_like.stride(dim)
        stride_expr = (
            str(stride) if isinstance(stride, int) else state.sympy_expr(stride)
        )
        terms.append(f"({index_dtype}({index}) * {index_dtype}({stride_expr}))")
    return " + ".join(terms) if terms else "0"


def _cute_stack_tensor_mask_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
) -> str | None:
    terms = []
    tensor_mask = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor_like,
        include_tensor_index_masks=False,
    )
    if tensor_mask is not None:
        terms.append(tensor_mask)
    stack_mask = _cute_combined_mask(
        state,
        [slice(None)] * dev_ptrs.ndim,
        None,
        tensor=dev_ptrs,
    )
    if stack_mask is not None and stack_mask not in terms:
        terms.append(stack_mask)
    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _cute_stack_tensor_pointer_expr(
    target_dtype: str,
    dev_ptrs_ast: ast.AST,
    offset_expr: str,
) -> ast.AST:
    return expr_from_string(
        f"(cute.make_ptr({target_dtype}, cutlass.Int64({{base}}), "
        f"cute.AddressSpace.gmem) + ({offset_expr}))",
        base=dev_ptrs_ast,
    )


def _codegen_cute_store_stack_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: tuple[object, ...] | list[object],
    ast_subscript: tuple[object, ...] | list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    if value_node.op != "call_function" or value_node.target is not load:
        return None
    stack_arg = value_node.args[0]
    if not isinstance(stack_arg, tuple) or len(stack_arg) != 2:
        return None
    ptr_node = stack_arg[1]
    if (
        not isinstance(ptr_node, torch.fx.Node)
        or ptr_node.op != "call_function"
        or ptr_node.target is not load
        or len(ptr_node.args) < 2
    ):
        return None
    dev_ptrs = (
        ptr_node.args[0].meta.get("val")
        if isinstance(ptr_node.args[0], torch.fx.Node)
        else None
    )
    ptr_subscript = ptr_node.args[1]
    if not isinstance(dev_ptrs, torch.Tensor) or not isinstance(
        ptr_subscript, (list, tuple)
    ):
        return None
    tensor_like_node = stack_arg[0]
    tensor_like = (
        tensor_like_node.meta.get("val")
        if isinstance(tensor_like_node, torch.fx.Node)
        else tensor_like_node
    )
    if not isinstance(tensor_like, torch.Tensor):
        return None

    if (
        dev_ptrs.ndim == 2
        and len(ptr_subscript) == 2
        and all(isinstance(idx, slice) and idx == slice(None) for idx in ptr_subscript)
        and len(subscript) >= 3
        and isinstance(subscript[0], slice)
        and subscript[0] == slice(None)
        and isinstance(subscript[1], slice)
        and subscript[1] == slice(None)
    ):
        stack_value_subscript = value_node.args[1]
        if not isinstance(stack_value_subscript, (list, tuple)):
            return None
        stack_value_subscript_proxy = map_arg(
            stack_value_subscript, lambda arg: arg.meta["val"]
        )
        stack_value_subscript_ast = map_arg(
            stack_value_subscript, lambda arg: state.env[arg]
        )
        tensor_offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*stack_value_subscript_proxy],
            [*stack_value_subscript_ast],
        )
        target_index_exprs = _cute_index_exprs(
            state,
            [*subscript],
            ast_subscript,
            tensor=tensor,
            inactive_singleton_slice_expr="0",
        )
        if len(target_index_exprs) != tensor.ndim:
            return None
        first_stack_index = target_index_exprs[0]
        target_tail = target_index_exprs[2:]
        loop_var = state.device_function.new_var("stack_dim", dce=True)
        env = CompileEnvironment.current()
        index_dtype = env.index_type()
        dev_ptrs_name = state.device_function.tensor_arg(dev_ptrs).name
        tensor_name = state.device_function.tensor_arg(tensor).name
        target_dtype = env.backend.dtype_str(tensor.dtype)
        dev_ptr_offset = (
            f"{index_dtype}({first_stack_index}) * "
            f"{index_dtype}({dev_ptrs.stride(0)}) + "
            f"{index_dtype}({loop_var}) * {index_dtype}({dev_ptrs.stride(1)})"
        )
        stack_ptr_expr = (
            f"(cute.make_ptr({target_dtype}, "
            f"cutlass.Int64(({dev_ptrs_name}.iterator + {dev_ptr_offset}).load()), "
            f"cute.AddressSpace.gmem) + ({tensor_offset_expr}))"
        )
        target_indices = [first_stack_index, loop_var, *target_tail]
        store_expr = _cute_scalar_store_expr(
            tensor_name,
            target_indices,
            f"({stack_ptr_expr}).load()",
        )
        mask_expr = _cute_combined_mask(state, [*subscript], extra_mask, tensor=tensor)
        if mask_expr is None:
            body = f"    {store_expr}"
        else:
            body = f"    if {mask_expr}:\n        {store_expr}"
        state.add_statement(
            statement_from_string(
                f"for {loop_var} in range({dev_ptrs.size(1)}):\n{body}"
            )
        )
        return ast.Constant(value=None)

    ptr_subscript_proxy = map_arg(ptr_subscript, lambda arg: arg.meta["val"])
    ptr_subscript_ast = map_arg(ptr_subscript, lambda arg: state.env[arg])
    ptr_index_exprs = _cute_index_exprs(
        state,
        [*ptr_subscript_proxy],
        [*ptr_subscript_ast],
        tensor=dev_ptrs,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in ptr_index_exprs:
        return None

    target_index_exprs = _cute_index_exprs(
        state,
        [*subscript],
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    ptr_pos = 0
    rewritten_index_exprs = []
    for idx, index_expr in zip(subscript, target_index_exprs, strict=True):
        if isinstance(idx, slice) and idx == slice(None):
            replacement = (
                ptr_index_exprs[ptr_pos] if ptr_pos < len(ptr_index_exprs) else None
            )
            ptr_pos += 1
            rewritten_index_exprs.append(
                replacement if replacement is not None else index_expr
            )
        else:
            if ptr_pos < len(ptr_subscript_proxy) and not (
                isinstance(ptr_subscript_proxy[ptr_pos], slice)
                and ptr_subscript_proxy[ptr_pos] == slice(None)
            ):
                ptr_pos += 1
            rewritten_index_exprs.append(index_expr)

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, rewritten_index_exprs, "{value}"),
        value=value,
    )
    mask_expr = _cute_combined_mask(state, [*subscript], extra_mask, tensor=tensor)
    if mask_expr is None:
        return store_expr
    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    state.add_statement(
        ast.fix_missing_locations(
            ast.If(
                test=mask_ast,
                body=[ast.Expr(value=store_expr)],
                orelse=[],
            )
        )
    )
    return ast.Constant(value=None)


def _cute_affine_range_block_id(state: CodegenState, affine: object) -> int | None:
    from .._compiler.cute.indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    env = CompileEnvironment.current()
    base_meta = getattr(affine.base, "meta", {})
    base_val = base_meta.get("val") if isinstance(base_meta, dict) else None
    block_id = env.resolve_block_id(base_val) if base_val is not None else None
    if block_id is None:
        codegen = base_meta.get("codegen") if isinstance(base_meta, dict) else None
        if isinstance(codegen, ast.Name) and codegen.id.startswith("_BLOCK_SIZE_"):
            with contextlib.suppress(ValueError):
                block_id = int(codegen.id.removeprefix("_BLOCK_SIZE_"))
    if block_id is None:
        return None
    if state.fx_node is not None:
        return env.resolve_codegen_block_id(
            block_id, state.codegen, state.fx_node.graph
        )
    return block_id


def _cute_affine_range_expr(
    state: CodegenState,
    affine: object,
    lane_var: str,
    *,
    dtype: torch.dtype | None = None,
) -> str | None:
    from .._compiler.cute.indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None
    index_var = _cute_active_index_var(state, block_id)
    if index_var is None:
        return None
    expr = f"({affine.factor}) * ({index_var}) + cutlass.Int32({lane_var})"
    if dtype is not None:
        expr = f"{CompileEnvironment.current().backend.dtype_str(dtype)}({expr})"
    return expr


def _codegen_cute_affine_range_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from .._compiler.ast_extension import create
    from .._compiler.cute.indexing import CuteAffineRangeIndex

    affine_positions = [
        (pos, idx)
        for pos, idx in enumerate(ast_subscript)
        if isinstance(idx, CuteAffineRangeIndex)
    ]
    if len(affine_positions) != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    _pos, affine = affine_positions[0]
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None

    lane_var = state.device_function.new_var("affine_lane", dce=True)
    index_expr = _cute_affine_range_expr(
        state, affine, lane_var, dtype=CompileEnvironment.current().index_dtype
    )
    if index_expr is None:
        return None
    backend = CompileEnvironment.current().backend
    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor):
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        ast_source_subscript = list(
            map_arg(tuple(source_subscript), lambda arg: state.env[arg])
        )
        (source_affine,) = ast_source_subscript
        if not isinstance(source_affine, CuteAffineRangeIndex):
            return None
        if source_affine.factor != affine.factor:
            return None
        source_index_expr = _cute_affine_range_expr(
            state,
            source_affine,
            lane_var,
            dtype=CompileEnvironment.current().index_dtype,
        )
        if source_index_expr is None:
            return None
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index_expr}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, CuteAffineRangeIndex):
        value_expr = _cute_affine_range_expr(state, value, lane_var, dtype=value.dtype)
        if value_expr is None:
            return None
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = (
        f"{tensor_name}.__setitem__({_cute_index_tuple([index_expr])}, {value_expr})"
    )
    mask_var = _cute_active_mask_var(state, block_id)
    if mask_var is not None:
        store_expr = f"{store_expr} if {mask_var} else None"

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({affine.factor})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _codegen_cute_affine_reshape_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Lower a 2-D affine-row store fed by a reshape/stack chain.

    Handles ``out[(begin*K):(begin*K + block*K), tile_n] = reshaped`` where the
    leading index is a ``CuteAffineRangeIndex`` (factor ``K``) over the m-tile,
    the trailing index is the n-tile, and the value is a row-major shape chain
    (e.g. ``stack([a, b], dim=1).reshape(block*K, block_n)``).

    Each m-tile thread owns row ``m_local`` of the source; the reshaped tensor
    has ``K`` rows per source row, so the thread loops ``s in range(K)`` and
    writes the value resolved at flat index ``(K*m_local + s)*block_n + n_local``
    to output row ``K*m_global + s``, column ``n_global``.
    """
    from .._compiler.ast_extension import create
    from .._compiler.cute.cute_reshape import _get_block_local_coord
    from .._compiler.cute.cute_reshape import resolve_cute_shape_chain_value_at
    from .._compiler.cute.indexing import CuteAffineRangeIndex
    from .._compiler.cute.indexing import is_cute_shape_chain_target
    from .._compiler.generate_ast import GenerateAST

    if (
        tensor.ndim != 2
        or len(subscript) != 2
        or len(ast_subscript) != 2
        or extra_mask is not None
        or value_node is None
        or not isinstance(state.codegen, GenerateAST)
    ):
        return None
    affine = ast_subscript[0]
    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    n_index = subscript[1]
    if not isinstance(n_index, torch.SymInt):
        return None
    env = CompileEnvironment.current()
    block_id_n = env.get_block_id(n_index)
    if block_id_n is None:
        return None
    block_id_m = _cute_affine_range_block_id(state, affine)
    if block_id_m is None:
        return None

    if value_node.op != "call_function" or not is_cute_shape_chain_target(
        value_node.target
    ):
        return None
    value_val = value_node.meta.get("val")
    if not isinstance(value_val, torch.Tensor) or value_val.ndim != 2:
        return None

    m_global = _cute_active_index_var(state, block_id_m)
    n_global = _cute_active_index_var(state, block_id_n)
    if m_global is None or n_global is None:
        return None
    m_local = _get_block_local_coord(state.codegen, block_id_m)
    n_local = _get_block_local_coord(state.codegen, block_id_n)
    if m_local is None or n_local is None:
        return None
    block_n = state.device_function.resolved_block_size(block_id_n)
    if not isinstance(block_n, int):
        return None

    factor = affine.factor
    lane_var = state.device_function.new_var("affine_lane", dce=True)
    row_local = f"cutlass.Int32({factor}) * ({m_local}) + cutlass.Int32({lane_var})"
    flat_index = (
        f"(({row_local}) * cutlass.Int32({block_n})) + ({n_local})"
        if block_n != 1
        else f"({row_local}) + ({n_local})"
    )
    value_ast = resolve_cute_shape_chain_value_at(state, value_node, flat_index)
    if value_ast is None:
        return None

    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(ast.unparse(value_ast), target_dtype)

    # Bind the resolved (possibly select-based) value to a variable so the CuTe
    # DSL sees the stack `ifexp` as its own assignment rather than nested inside
    # the `.store(...)` call / masked store ternary.
    value_var = state.device_function.new_var("affine_value", dce=True)

    row_index = (
        f"{index_dtype}(cutlass.Int32({factor}) * ({m_global}) "
        f"+ cutlass.Int32({lane_var}))"
    )
    col_index = f"{index_dtype}({n_global})"
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = _cute_scalar_store_expr(tensor_name, [row_index, col_index], value_var)

    store_stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    mask_parts = [
        mask
        for mask in (
            _cute_active_mask_var(state, block_id_m),
            _cute_active_mask_var(state, block_id_n),
        )
        if mask is not None
    ]
    if mask_parts:
        # Use a guard statement (not a ternary) so the CuTe DSL accepts the
        # device-value mask condition.
        mask_ast = expr_from_string(" and ".join(mask_parts))
        assert isinstance(mask_ast, ast.expr)
        store_stmt = ast.fix_missing_locations(
            ast.If(test=mask_ast, body=[store_stmt], orelse=[])
        )

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({factor})"),
        body=[
            statement_from_string(f"{value_var} = {value_expr}"),
            store_stmt,
        ],
        orelse=[],
        type_comment=None,
    )


def _is_cute_affine_range_load_for_store(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
) -> bool:
    from .._compiler.cute.indexing import CuteAffineRangeIndex
    from .._compiler.cute.indexing import match_cute_affine_range_iota

    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and isinstance(store_subscript[0], torch.fx.Node)
            and match_cute_affine_range_iota(store_subscript[0]) is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and len(subscript) == 1
        and len(ast_subscript) == 1
        and isinstance(ast_subscript[0], CuteAffineRangeIndex)
    )


def _cute_positive_1d_slice_bounds(
    tensor: torch.Tensor, index: object
) -> tuple[int, int, int, int] | None:
    if not isinstance(index, slice) or index == slice(None):
        return None
    with contextlib.suppress(TypeError):
        dim_size = int(tensor.shape[0])
        start, stop, step = index.indices(dim_size)
        if step <= 0:
            return None
        length = max(0, (stop - start + step - 1) // step)
        return start, stop, step, length
    return None


def _is_cute_strided_slice_load_for_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> bool:
    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        target_node = user.args[0]
        if not isinstance(target_node, torch.fx.Node):
            return False
        target_tensor = target_node.meta.get("val")
        if not isinstance(target_tensor, torch.Tensor) or target_tensor.ndim != 1:
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and _cute_positive_1d_slice_bounds(target_tensor, store_subscript[0])
            is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and tensor.ndim == 1
        and len(subscript) == 1
        and _cute_positive_1d_slice_bounds(tensor, subscript[0]) is not None
    )


def _codegen_cute_strided_slice_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from .._compiler.ast_extension import create

    if tensor.ndim != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    target_bounds = _cute_positive_1d_slice_bounds(tensor, subscript[0])
    if target_bounds is None:
        return None
    target_start, _target_stop, target_step, target_length = target_bounds

    env = CompileEnvironment.current()
    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    loop_var = state.device_function.new_var("slice_idx", dce=True)
    target_index = f"{index_dtype}({target_start} + {loop_var} * {target_step})"

    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor) or source_tensor.ndim != 1:
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        source_bounds = _cute_positive_1d_slice_bounds(
            source_tensor, source_subscript[0]
        )
        if source_bounds is None:
            return None
        source_start, _source_stop, source_step, source_length = source_bounds
        if source_length != target_length:
            return None
        source_index = f"{index_dtype}({source_start} + {loop_var} * {source_step})"
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_name = state.device_function.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    store_expr = f"{target_name}.__setitem__(({target_index},), {value_expr})"
    return create(
        ast.For,
        target=create(ast.Name, id=loop_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({target_length})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _cute_combined_mask(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    tensor: torch.Tensor | None = None,
    *,
    include_tensor_index_masks: bool = True,
) -> str | None:
    env = CompileEnvironment.current()
    terms: list[str] = []

    def mask_var_for_block_id(block_id: int) -> str | None:
        if _cute_index_override(state, block_id) is not None:
            return None
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.mask_var(block_id)
        return None

    def active_index_var(block_id: int) -> str | None:
        if (override := _cute_index_override(state, block_id)) is not None:
            return override
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
        return None

    def active_local_coord(block_id: int) -> str | None:
        from .._compiler.cute.cute_reshape import _grid_local_coord_expr

        if _cute_index_override(state, block_id) is not None:
            return None
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            thread_axis = loops[-1].block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        return None

    def tile_begin_expr(block_id: int) -> str:
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return state.codegen.offset_var(block_id)
        global_index = active_index_var(block_id)
        local_coord = active_local_coord(block_id)
        if global_index is not None and local_coord is not None:
            return state.codegen.lift(
                expr_from_string(f"({global_index}) - ({local_coord})"),
                dce=True,
                prefix="tile_begin",
            ).id
        if global_index is not None:
            return global_index
        return "0"

    def tile_with_offset_mask_terms(
        tile_info: TileWithOffsetInfo,
        tensor_dim: int,
    ) -> list[str]:
        block_id = tile_info.block_id
        local_coord = active_local_coord(block_id)
        begin_var = tile_begin_expr(block_id)
        if local_coord is None:
            if (idx_var := active_index_var(block_id)) is None:
                raise exc.BackendUnsupported(
                    "cute",
                    (
                        "indexing dimension is not active in this scope "
                        f"(block_id={block_id})"
                    ),
                )
            local_coord = f"({idx_var}) - ({begin_var})"

        tile_terms = []
        if tile_info.block_size is not None:
            block_size_expr = state.device_function.literal_expr(tile_info.block_size)
            tile_terms.append(f"({local_coord}) < cutlass.Int32({block_size_expr})")
        if tensor is not None and tensor_dim < tensor.ndim:
            offset_expr = state.device_function.literal_expr(tile_info.offset)
            dim_size = _cute_tensor_dim_size_expr(state, tensor, tensor_dim)
            tile_terms.append(
                f"(({begin_var}) + cutlass.Int32({offset_expr}) + "
                f"({local_coord})) < {dim_size}"
            )
        return tile_terms

    if extra_mask is not None:
        terms.append(state.codegen.lift(extra_mask, dce=True, prefix="mask").id)

    seen: set[int] = set()
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        block_id: int | None = None
        if idx is None:
            continue
        if (
            tile_info := _get_tile_with_offset_info(
                idx, getattr(state, "fx_node", None), pos
            )
        ) is not None and tile_info.block_size is not None:
            seen.add(tile_info.block_id)
            for term in tile_with_offset_mask_terms(tile_info, tensor_dim):
                if term not in terms:
                    terms.append(term)
            tensor_dim += 1
            continue
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif isinstance(idx, slice) and idx == slice(None) and tensor is not None:
            for bid in _matching_block_ids(env, tensor.shape[tensor_dim]):
                if bid not in seen and mask_var_for_block_id(bid) is not None:
                    block_id = bid
                    break
        elif isinstance(idx, torch.Tensor):
            if not include_tensor_index_masks:
                for dim_size in idx.shape:
                    for bid in _matching_block_ids(env, dim_size):
                        if bid in seen or not env.is_jagged_tile(bid):
                            continue
                        mask_var = mask_var_for_block_id(bid)
                        if mask_var is not None:
                            seen.add(bid)
                            if mask_var not in terms:
                                terms.append(mask_var)
                            break
                tensor_dim += 1
                continue
            for dim_size in idx.shape:
                for bid in _matching_block_ids(env, dim_size):
                    if bid in seen:
                        continue
                    mask_var = mask_var_for_block_id(bid)
                    if mask_var is not None:
                        seen.add(bid)
                        if mask_var not in terms:
                            terms.append(mask_var)
                        break
                else:
                    continue
            tensor_dim += 1
            continue
        else:
            tensor_dim += 1
            continue
        if block_id is None or block_id in seen:
            tensor_dim += 1
            continue
        seen.add(block_id)
        if (mask_var := mask_var_for_block_id(block_id)) is not None:
            if mask_var not in terms:
                terms.append(mask_var)
        tensor_dim += 1

    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _cute_tensor_dim_size_expr(
    state: CodegenState, tensor: torch.Tensor, dim: int
) -> str:
    return state.device_function.tensor_size(tensor, dim).name


def _cute_tile_begin_expr(state: CodegenState, idx: object) -> str:
    env = CompileEnvironment.current()

    def active_index_var(block_id: int) -> str | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
        return None

    def active_local_coord(block_id: int) -> str | None:
        from .._compiler.cute.cute_reshape import _grid_local_coord_expr

        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            thread_axis = loops[-1].block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        return None

    def tile_begin_from_block_id(block_id: int) -> str:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return state.codegen.offset_var(block_id)
        global_index = active_index_var(block_id)
        local_coord = active_local_coord(block_id)
        if global_index is not None and local_coord is not None:
            return state.codegen.lift(
                expr_from_string(f"({global_index}) - ({local_coord})"),
                dce=True,
                prefix="tile_begin",
            ).id
        if global_index is not None:
            return global_index
        return "0"

    if isinstance(idx, int):
        return str(idx)
    if not isinstance(idx, torch.SymInt):
        raise exc.BackendUnsupported("cute", f"tile base index type: {type(idx)}")

    expr = _symint_expr(idx)
    if expr is not None:
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is not None and isinstance(origin_info.origin, TileBeginOrigin):
            return tile_begin_from_block_id(origin_info.origin.block_id)
    block_id = env.get_block_id(idx)
    if block_id is not None:
        return tile_begin_from_block_id(block_id)
    if expr is not None:
        return state.sympy_expr(expr)
    raise exc.BackendUnsupported("cute", f"unlowerable tile base index: {idx}")


def _codegen_cute_store_tcgen05_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_name: str,
    epilogue_chain: Tcgen05UnaryEpilogueChain | None = None,
) -> list[ast.AST] | ast.AST | None:
    df = state.device_function
    candidate_names = df.variable_aliases(value_name)
    tcgen05_value = df.cute_state.get_tcgen05_store_value(candidate_names)
    if tcgen05_value is None:
        return None
    if extra_mask is not None:
        if tcgen05_value.pure_matmul_role_lifecycle:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 pure role-lifecycle store cannot use an extra store mask",
            )
        return None
    if tensor.ndim != 2:
        if tcgen05_value.pure_matmul_role_lifecycle:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 pure role-lifecycle store requires a rank-2 tensor target",
            )
        return None
    if tcgen05_value.pure_matmul_role_lifecycle:
        if epilogue_chain is not None:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 pure role-lifecycle supports only identity pure-matmul stores",
            )
    # When one matmul accumulator fans out to multiple output stores (e.g.
    # aux = pre-activation and out = gelu(pre)), the per-matmul TMA-store
    # atom/tensor kernel-arg names allocated in cute_mma are shared by every
    # store site. Emitting them verbatim at each site produces duplicate kernel
    # parameters (SyntaxError) and binds both device epilogues to the same TMA
    # descriptor. The secondary store gets fresh per-store descriptor names so
    # each store threads its own TMA descriptor; the first store keeps the
    # original names. The secondary store also reuses the accumulator the first
    # store already consumed: the accumulator TMEM stays live until the
    # one-shot teardown frees it, so the secondary store reads it directly
    # without re-running the accumulator pipeline's consumer wait/release/advance
    # (those would hang waiting on a producer that has already drained) and
    # without re-emitting the matmul drain / TMEM-free teardown.
    is_secondary_store = (
        tcgen05_value.use_tma_store_epilogue
        and not tcgen05_value.pure_matmul_role_lifecycle
        and df.cute_state.tcgen05_tma_store_names_already_emitted(tcgen05_value)
    )
    if is_secondary_store:
        tcgen05_value = dataclasses.replace(
            tcgen05_value,
            tma_store_atom=df.new_var("tcgen05_tma_store_atom"),
            tma_store_tensor=df.new_var("tcgen05_tma_store_tensor"),
        )
    tcgen05_lifecycle = tcgen05_value.lifecycle_context
    tcgen05_pure_matmul_object = tcgen05_value.pure_matmul_object

    # Snapshot the accumulator consumer-state stage index. The primary store
    # captures it before advancing the consumer state; fan-out stores read the
    # same live TMEM stage through the snapshot rather than the already-advanced
    # live index. For single-store kernels the assignment is unused and DCE
    # drops it, so the generated code is unchanged.
    tcgen05_acc_stage_index_var, tcgen05_acc_stage_index_is_primary = (
        df.cute_state.get_or_create_tcgen05_acc_stage_index_var(
            tcgen05_lifecycle.acc_consumer_state,
            df.new_var,
        )
    )
    # The snapshot is captured at top level (before the store's control-flow
    # block) by the primary store so fan-out stores can read it; CuTe DSL
    # forbids defining a value inside one control-flow block and reading it in
    # another. For single-store kernels the assignment is unused and DCE drops
    # it, keeping generated code unchanged.
    tcgen05_acc_stage_index_top_level_stmts = (
        [
            statement_from_string(
                f"{tcgen05_acc_stage_index_var} = "
                f"{tcgen05_lifecycle.acc_consumer_state}.index"
            )
        ]
        if tcgen05_acc_stage_index_is_primary
        else []
    )
    # The primary store keeps reading the live consumer index so single-store
    # codegen is byte-identical; only fan-out stores route through the snapshot.
    tcgen05_acc_stage_index_expr = (
        f"{tcgen05_lifecycle.acc_consumer_state}.index"
        if not is_secondary_store
        else tcgen05_acc_stage_index_var
    )

    # Backstop for callers that bypass Config.normalize() validation;
    # see _tcgen05_epi_warp_count docstring and cute_plan.md.
    if tcgen05_value.epi_warp_count != 4:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 SIMT-store epilogue requires "
            f"tcgen05_num_epi_warps=4 (got {tcgen05_value.epi_warp_count}). "
            "CUTLASS tmem_warp_shape_mn=(4,1) hard-codes a 4-warp t2r "
            "partition for the supported tcgen05 path; per-warp "
            "tcgen05.ld semantics make the partition uncoverable by "
            "fewer warps. Lifts when the c_pipeline-driven multi-warp "
            "epilogue lands (see cute_plan.md).",
        )

    backend = CompileEnvironment.current().backend
    tensor_name = df.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    # The matmul plan computed `tcgen05_epi_tile` (role-local t2r
    # partition) with `epi_elem_dtype_str`; the store path below
    # recomputes `tcgen05_store_epi_tile` with `target_dtype`. They must
    # match or `compute_epilogue_tile_shape` selects different `tile_n`
    # values on the two sides and the t2r / r2s SMEM staging silently
    # corrupts. The loud-failure backstop covers cases where MMA-codegen-
    # time forward-tracing of the matmul fx_node could not pin a unique
    # store target dtype.
    if (
        tcgen05_value.epi_elem_dtype_str
        and tcgen05_value.epi_elem_dtype_str != target_dtype
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 epilogue element-type mismatch: matmul plan was set "
            f"up with epi_elem_dtype_str={tcgen05_value.epi_elem_dtype_str!r} "
            f"but the store target tensor dtype is {target_dtype!r}.",
        )
    base_indices = [_cute_tile_begin_expr(state, idx) for idx in subscript]
    if len(base_indices) != 2:
        if tcgen05_value.pure_matmul_role_lifecycle:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 pure role-lifecycle store requires a rank-2 tile store",
            )
        return None
    m_size = _cute_tensor_dim_size_expr(state, tensor, 0)
    n_size = _cute_tensor_dim_size_expr(state, tensor, 1)
    tile_coord_m = f"({base_indices[0]}) // cutlass.Int32({tcgen05_value.bm})"
    tile_coord_n = f"({base_indices[1]}) // cutlass.Int32({tcgen05_value.bn})"
    full_tile = df.new_var("tcgen05_full_tile")

    gmem_tile = df.new_var("tcgen05_gC")
    coord_tile = df.new_var("tcgen05_cC")
    tcgc_base = df.new_var("tcgen05_tCgC_base")
    tccc_base = df.new_var("tcgen05_tCcC_base")
    tcgc = df.new_var("tcgen05_tCgC")
    tcgc_planned = df.new_var("tcgen05_tCgC_planned")
    tccc = df.new_var("tcgen05_tCcC")
    tacc = df.new_var("tcgen05_tAcc")
    epi_tile = df.new_var("tcgen05_store_epi_tile")
    tiled_copy_t2r = df.new_var("tcgen05_tiled_copy_t2r")
    thr_copy_t2r = df.new_var("tcgen05_thr_copy_t2r")
    ttr_tacc_base = df.new_var("tcgen05_tTR_tAcc_base")
    tcgc_epi = df.new_var("tcgen05_tCgC_epi")
    tccc_epi = df.new_var("tcgen05_tCcC_epi")
    ttr_gc = df.new_var("tcgen05_tTR_gC")
    ttr_cc = df.new_var("tcgen05_tTR_cC")
    ttr_racc = df.new_var("tcgen05_tTR_rAcc")
    ttr_rd = df.new_var("tcgen05_tTR_rD")
    ttr_tacc_stage = df.new_var("tcgen05_tTR_tAcc_stage")
    ttr_tacc = df.new_var("tcgen05_tTR_tAcc")
    ttr_gc_grouped = df.new_var("tcgen05_tTR_gC_grouped")
    ttr_cc_grouped = df.new_var("tcgen05_tTR_cC_grouped")
    ttr_tacc_mn = df.new_var("tcgen05_tTR_tAcc_mn")
    ttr_gc_subtile = df.new_var("tcgen05_tTR_gC_subtile")
    ttr_cc_subtile = df.new_var("tcgen05_tTR_cC_subtile")
    acc_vec = df.new_var("tcgen05_acc_vec")
    kernel_desc = df.new_var("tcgen05_kernel_desc")
    mcld = df.new_var("tcgen05_mcld")
    num_bits = df.new_var("tcgen05_num_bits")
    simt_atom = df.new_var("tcgen05_simt_atom")
    smem_d_layout = df.new_var("tcgen05_sD_layout")
    smem_d_ptr = df.new_var("tcgen05_sD_ptr")
    smem_d = df.new_var("tcgen05_sD")
    tiled_copy_r2s = df.new_var("tcgen05_tiled_copy_r2s")
    trs_rd = df.new_var("tcgen05_tRS_rD")
    trs_racc = df.new_var("tcgen05_tRS_rAcc")
    trs_sd = df.new_var("tcgen05_tRS_sD")
    bsg_sd = df.new_var("tcgen05_bSG_sD")
    bsg_gd_partitioned = df.new_var("tcgen05_bSG_gD_partitioned")
    bsg_gd = df.new_var("tcgen05_bSG_gD")
    c_buffer = df.new_var("tcgen05_c_buffer")
    epilog_sync_barrier = df.new_var("tcgen05_epilog_sync_barrier")
    c_pipeline_producer_group = df.new_var("tcgen05_c_pipeline_producer_group")
    c_pipeline = df.new_var("tcgen05_c_pipeline")
    subtile_count = df.new_var("tcgen05_subtile_count")
    # Workstream A Stage 4 (cycle 93, Path B): the C-store producer->consumer
    # edge over the C-ring SMEM (``tRS_sD``, depth ``c_stage_count``). Producer
    # = the 4 epi warps (arrive after R2S + ``fence_view_async_shared``);
    # consumer = the single store warp (waits, issues the TMA-D, releases the
    # SMEM stage). Replaces the second ``epilog_sync_barrier`` (R2S-visible)
    # CTA-wide barrier with a cheaper cross-warp pipeline edge that lets the
    # epi warps proceed to the next subtile while the store warp drains.
    c_store_edge_barriers = df.new_var("tcgen05_c_store_edge_barriers")
    c_store_edge_producer_group = df.new_var("tcgen05_c_store_edge_producer_group")
    c_store_edge_consumer_group = df.new_var("tcgen05_c_store_edge_consumer_group")
    c_store_edge = df.new_var("tcgen05_c_store_edge")
    c_store_edge_producer_state = df.new_var("tcgen05_c_store_edge_producer_state")
    c_store_edge_consumer_state = df.new_var("tcgen05_c_store_edge_consumer_state")
    # Separate consumer state for the LAGGED release. The store warp's TMA-D is
    # an async bulk copy that reads the C-ring SMEM stage; the stage may not be
    # reused (epi R2S overwrite) until that read completes. ``c_pipeline``
    # (PipelineTmaStore) tracks store completion via ``cp_async_bulk_wait_group``
    # (read=True), which after committing store i and waiting drains every store
    # except the ``c_stages - 1`` most recent. So the store warp releases the
    # C-ring stage from ``c_stages - 1`` subtiles ago (provably drained), lagging
    # the consumer-wait by ``c_stages - 1``. This leaves exactly one free stage
    # (edge depth ``c_stages``), giving the ~1-subtile store/T2R overlap the
    # acc_stages=2 bound permits. The first ``c_stages - 1`` releases are
    # suppressed (no drained stage yet); the trailing stages release naturally
    # in subsequent tiles as the global subtile index advances.
    c_store_edge_release_state = df.new_var("tcgen05_c_store_edge_release_state")
    epi_warp_ids = ", ".join(
        f"cutlass.Int32({i})" for i in range(tcgen05_value.epi_warp_count)
    )
    if tcgen05_value.epi_warp_count == 1:
        epi_warp_ids += ","

    # Per-aux-step plumbing: per-thread auxiliary tensor reads at
    # the splice site. For each ``_AuxiliaryTensorStep`` in the
    # chain we register the auxiliary tensor as a kernel arg,
    # allocate fresh AST var names for the partitioning chain, and
    # later (inside each per-thread splice site) emit per-subtile
    # ``aux_loaded = ...`` lines that the chain renderer references.
    # Static-full TMA-store tiles use the historical direct
    # ``ttr_aux_subtile.load()`` form. SIMT-store edge tiles use a
    # predicated GMEM-to-register copy first, so the aux read observes
    # the same runtime predicate as the output store.
    aux_steps_in_chain: tuple[_AuxiliaryTensorStep, ...] = (
        epilogue_chain.auxiliary_tensor_steps if epilogue_chain is not None else ()
    )

    aux_step_records: list[_AuxStepRecord] = []
    for aux_idx, aux_step in enumerate(aux_steps_in_chain):
        aux_tensor_node = aux_step.load_node.args[0]
        assert isinstance(aux_tensor_node, torch.fx.Node)
        aux_torch_tensor = aux_tensor_node.meta.get("val")
        assert isinstance(aux_torch_tensor, torch.Tensor)
        aux_tensor_name = df.tensor_arg(aux_torch_tensor).name
        aux_dtype = backend.dtype_str(aux_torch_tensor.dtype)
        aux_dtype_bits = aux_torch_tensor.dtype.itemsize * 8
        # Aux tensors must be passed through to the device function as
        # placeholder args so the wrapper plumbs them into the cute
        # kernel signature (the role-local persistent path otherwise
        # treats unreferenced tensors as captures, which doesn't work
        # for tensors only read inside a per-subtile loop body).
        df.placeholder_args.add(aux_tensor_name)
        # Broadcast aux steps need a fresh AST var for the 2-D view
        # of the rank-1 underlying tensor (stride 0 on the orthogonal
        # axis). Exact-shape aux steps leave ``aux_view2d`` as None.
        aux_view2d = (
            df.new_var(f"tcgen05_aux_view2d_{aux_idx}")
            if aux_step.broadcast_axis is not None
            else None
        )
        aux_step_records.append(
            _AuxStepRecord(
                aux_tensor_name=aux_tensor_name,
                broadcast_axis=aux_step.broadcast_axis,
                aux_tile=df.new_var(f"tcgen05_aux_tile_{aux_idx}"),
                aux_part_base=df.new_var(f"tcgen05_tCgAux_base_{aux_idx}"),
                aux_xfm=df.new_var(f"tcgen05_tCgAux_xfm_{aux_idx}"),
                aux_planned=df.new_var(f"tcgen05_tCgAux_planned_{aux_idx}"),
                aux_epi=df.new_var(f"tcgen05_tCgAux_epi_{aux_idx}"),
                aux_dtype=aux_dtype,
                aux_dtype_bits=aux_dtype_bits,
                aux_extent=(
                    aux_torch_tensor.shape[0]
                    if (
                        aux_step.broadcast_axis == 1
                        and isinstance(aux_torch_tensor.shape[0], int)
                    )
                    else None
                ),
                ttr_aux=df.new_var(f"tcgen05_tTR_gAux_{aux_idx}"),
                ttr_aux_grouped=df.new_var(f"tcgen05_tTR_gAux_grouped_{aux_idx}"),
                ttr_aux_subtile=df.new_var(f"tcgen05_tTR_gAux_subtile_{aux_idx}"),
                aux_rmem=df.new_var(f"tcgen05_aux_rmem_{aux_idx}"),
                aux_loaded=df.new_var(f"tcgen05_aux_loaded_{aux_idx}"),
                aux_view2d=aux_view2d,
            )
        )

    # Pyrefly does not preserve the non-None ``tcgen05_value`` narrowing
    # inside the nested source-formatter closures, so keep local
    # string aliases for attributes the closures read.
    tcgen05_aux_bm = tcgen05_value.bm
    tcgen05_aux_bn = tcgen05_value.bn
    tcgen05_aux_thr_mma = tcgen05_value.thr_mma
    tcgen05_aux_epi_tidx = tcgen05_value.epi_tidx
    tcgen05_aux_epi_active = tcgen05_lifecycle.epi_active
    tcgen05_aux_epi_warp_count = tcgen05_value.epi_warp_count
    tcgen05_aux_epilogue_rest_mode = tcgen05_value.epilogue_rest_mode
    tcgen05_aux_use_tma_store_epilogue = tcgen05_value.use_tma_store_epilogue
    tcgen05_explicit_store_tile_expr: str | None = None
    if tcgen05_value.has_explicit_epilogue_tile:
        assert tcgen05_value.explicit_epi_tile_m is not None
        assert tcgen05_value.explicit_d_store_box_n is not None
        tcgen05_explicit_store_tile_expr = tcgen05_explicit_d_store_tile_expr(
            tcgen05_value.explicit_epi_tile_m,
            tcgen05_value.explicit_d_store_box_n,
        )

    # C-input warp productive-body gate (``cute_plan.md`` §7.5.3.2
    # cycle 2b producer + consumer flip). When the matmul plan has
    # ``has_c_input_warp`` AND a non-empty ``aux_tensor_descriptors``
    # tuple AND the aux pipeline plan was registered by
    # ``cute_mma._codegen_cute_mma``, the consumer-side per-thread
    # GMEM aux LDG flips to an SMEM read from the
    # ``c_pipeline_aux``-staged ring populated by the C-input warp's
    # cooperative copy. The producer body in
    # ``program_id._build_c_input_warp_role_local_while`` writes
    # ONE ``epi_tile`` subtile of the per-CTA aux region
    # (``(bm_per_cta, bn)`` under 2cta; ``(bm, bn)`` otherwise) per
    # stage per subtile iteration under ``producer_acquire`` /
    # ``producer_commit`` framing; the consumer issues one
    # ``consumer_wait`` / lane-0-gated ``consumer_release`` pair
    # per subtile and feeds the SMEM stage into Quack's
    # ``tiled_copy_s2r`` flow (``make_tiled_copy_D`` against
    # ``tiled_copy_t2r`` →  ``partition_S(sC_ring)`` → per-
    # subtile ``cute.copy(s2r, sC[..., stage], rmem)`` →
    # ``rmem.load()``). Gate-closed configs keep the historical
    # GMEM path byte-identical.
    aux_matmul_plan = df.cute_state.matmul_plan
    aux_pipeline_plan_obj = df.cute_state.aux_pipeline_plan
    # Workstream A Stage 4 (cycle 93, Path B): when the plan carries a store
    # warp, the per-subtile R2S->TMA-D tail is split by warp role and the
    # second epilogue barrier is replaced by the C-store pipeline edge. The
    # store warp drains the TMA-D so the 4 epi warps proceed to the next
    # subtile's T2R. ``store_warps=0`` keeps the original fused tail unchanged
    # (the production path; byte-identical codegen).
    has_store_warp = aux_matmul_plan is not None and aux_matmul_plan.has_store_warp
    store_warp_predicate = (
        f"{tcgen05_value.warp_idx} == cutlass.Int32({aux_matmul_plan.store_warp_id})"
        if aux_matmul_plan is not None and has_store_warp
        else ""
    )
    # Match each store-side record to its descriptor by
    # ``load_node`` FX-node identity rather than positional
    # index. The descriptor walker dedups by ``store_value_node``
    # at MMA-codegen time, so a single-store kernel's
    # descriptors and records share the same ``load_node``
    # values in some permutation. The matmul plan's
    # ``aux_single_store_value`` gate (in ``cute_mma`` and the
    # ``program_id`` role-local-while admission) only allocates
    # the producer-side pipeline when every descriptor shares
    # one ``store_value_node``, so the multi-store fan-out
    # wedge (producer commits to rings the per-store consumer
    # never releases) cannot occur — the productive body
    # closes its gate at MMA-codegen time and the consumer
    # path here falls back to GMEM. Broadcast row-vector aux loads are
    # deliberately not staged by the C-input producer, so the per-record lookup
    # below allows a mixed chain: matched exact-shape records read from SMEM,
    # unmatched records keep the direct GMEM path.
    aux_step_load_nodes: tuple = (
        tuple(rec_step.load_node for rec_step in aux_steps_in_chain)
        if aux_step_records
        else ()
    )
    aux_ring_index_by_step: list[int | None] = []
    aux_descriptor_load_nodes: tuple = (
        tuple(d.load_node for d in aux_matmul_plan.c_input_aux_tensor_descriptors)
        if aux_matmul_plan is not None
        else ()
    )
    for step_load_node in aux_step_load_nodes:
        try:
            aux_ring_index_by_step.append(
                aux_descriptor_load_nodes.index(step_load_node)
            )
        except ValueError:
            aux_ring_index_by_step.append(None)
    aux_has_staged_steps = any(
        ring_idx is not None for ring_idx in aux_ring_index_by_step
    )
    # Workstream A Stage 5 (cycle 94, the merge): the aux SMEM ring producer is
    # the C-input warp normally (SIMT or TMA), or the store warp under the merge
    # — but the store warp is TMA-ONLY (there is no SIMT store-warp producer;
    # ``store_warps=1 + SIMT aux`` falls back to direct-GMEM aux). The epi-warp
    # consumer reads the staged ring whenever a producer is present. The
    # ``aux_pipeline_plan_obj is not None`` term already closes this gate for
    # ``store_warps=1 + SIMT`` (``cute_mma`` never allocates the plan there);
    # the explicit ``use_tma_load`` term on the store-warp branch makes the
    # TMA-only requirement local and defensive.
    aux_producer_warp_present = aux_matmul_plan is not None and (
        aux_matmul_plan.has_c_input_warp
        or (
            aux_matmul_plan.has_store_warp
            and aux_pipeline_plan_obj is not None
            and aux_pipeline_plan_obj.use_tma_load
        )
    )
    use_aux_smem_source = (
        aux_step_records
        and aux_matmul_plan is not None
        and aux_producer_warp_present
        and bool(aux_matmul_plan.c_input_aux_tensor_descriptors)
        and aux_pipeline_plan_obj is not None
        and aux_has_staged_steps
        # Multi-store fan-out gate (same predicate as the
        # producer-side allocator + role-local-while
        # admission). Without this guard the producer fires
        # ``producer_commit`` on rings whose only matching
        # consumer-store is a different per-store-codegen
        # invocation — the per-store splice site here only
        # releases its own subset, leaving the unmatched rings
        # uncommitted and deadlocking the producer once a CTA
        # wraps the pipeline depth.
        and len(
            {d.store_value_node for d in aux_matmul_plan.c_input_aux_tensor_descriptors}
        )
        <= 1
    )
    if use_aux_smem_source:
        assert aux_pipeline_plan_obj is not None
        aux_pipeline_name = aux_pipeline_plan_obj.pipeline
        aux_consumer_state_name = aux_pipeline_plan_obj.consumer_state
        aux_pipeline_uses_tma_load = aux_pipeline_plan_obj.use_tma_load
        all_rings = aux_pipeline_plan_obj.rings
        aux_ring_smem_names: tuple[str | None, ...] = tuple(
            all_rings[ring_idx].smem if ring_idx is not None else None
            for ring_idx in aux_ring_index_by_step
        )
    else:
        aux_pipeline_name = ""
        aux_consumer_state_name = ""
        aux_pipeline_uses_tma_load = False
        aux_ring_smem_names = tuple(None for _ in aux_step_records)

    rowvec_aux_stage_records: list[_RowvecAuxStageRecord | None] = []
    for aux_idx, rec in enumerate(aux_step_records):
        copy_bits = 128
        copy_elems = _tcgen05_rowvec_aux_stage_copy_elems(
            rec.aux_dtype_bits,
            tcgen05_aux_bn,
            rec.aux_extent,
            copy_bits=copy_bits,
        )
        if (
            tcgen05_value.partial_output_tma_store
            and tcgen05_value.use_tma_store_epilogue
            and rec.broadcast_axis == 1
            and copy_elems is not None
        ):
            assert rec.aux_extent is not None
            rowvec_aux_stage_records.append(
                _RowvecAuxStageRecord(
                    smem_layout=df.new_var(f"tcgen05_aux_rowvec_smem_layout_{aux_idx}"),
                    smem_ptr=df.new_var(f"tcgen05_aux_rowvec_smem_ptr_{aux_idx}"),
                    smem=df.new_var(f"tcgen05_aux_rowvec_smem_{aux_idx}"),
                    tiled_copy=df.new_var(f"tcgen05_aux_rowvec_tiled_copy_{aux_idx}"),
                    thr_copy=df.new_var(f"tcgen05_aux_rowvec_thr_copy_{aux_idx}"),
                    gmem_tile=df.new_var(f"tcgen05_aux_rowvec_gmem_tile_{aux_idx}"),
                    gmem_part=df.new_var(f"tcgen05_aux_rowvec_gmem_part_{aux_idx}"),
                    smem_part=df.new_var(f"tcgen05_aux_rowvec_smem_part_{aux_idx}"),
                    coord=df.new_var(f"tcgen05_aux_rowvec_coord_{aux_idx}"),
                    limit=df.new_var(f"tcgen05_aux_rowvec_limit_{aux_idx}"),
                    pred=df.new_var(f"tcgen05_aux_rowvec_pred_{aux_idx}"),
                    copy_bits=copy_bits,
                    copy_elems=copy_elems,
                    aux_extent=rec.aux_extent,
                )
            )
        else:
            rowvec_aux_stage_records.append(None)
    partial_tma_needs_full_tile_guard = tcgen05_value.partial_output_tma_store and any(
        # ``aux_ring_smem_names`` and ``rowvec_aux_stage_records`` are both
        # positionally aligned with ``aux_step_records``.
        name is None and rowvec_aux_stage_records[aux_idx] is None
        for aux_idx, name in enumerate(aux_ring_smem_names)
    )

    def _rowvec_aux_smem_setup_lines() -> list[str]:
        """Emit compact per-tile SMEM allocation for staged row-vector aux."""

        lines: list[str] = []
        for aux_idx, rec in enumerate(aux_step_records):
            stage = rowvec_aux_stage_records[aux_idx]
            if stage is None:
                continue
            lines.extend(
                [
                    (
                        f"{stage.smem_layout} = cute.make_layout("
                        f"({tcgen05_aux_bn},), stride=(1,))"
                    ),
                    (
                        f"{stage.smem_ptr} = cute.arch.alloc_smem("
                        f"{rec.aux_dtype}, cute.cosize({stage.smem_layout}), "
                        "alignment=128)"
                    ),
                    (
                        f"{stage.smem} = cute.make_tensor("
                        f"{stage.smem_ptr}, {stage.smem_layout})"
                    ),
                ]
            )
        return lines

    def _rowvec_aux_copy_lines() -> list[str]:
        """Emit the predicated GMEM-to-SMEM copy for staged row-vector aux."""

        lines: list[str] = []
        for aux_idx, rec in enumerate(aux_step_records):
            stage = rowvec_aux_stage_records[aux_idx]
            if stage is None:
                continue
            lines.append(
                f"if {tcgen05_aux_epi_active}:\n"
                f"    {stage.tiled_copy} = cute.make_tiled_copy_tv("
                f"cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                f"{rec.aux_dtype}, num_bits_per_copy={stage.copy_bits}), "
                f"cute.make_layout({tcgen05_aux_epi_warp_count * 32}), "
                f"cute.make_layout({stage.copy_elems}))\n"
                f"    {stage.thr_copy} = {stage.tiled_copy}.get_slice("
                f"{tcgen05_aux_epi_tidx})\n"
                f"    {stage.gmem_tile} = cute.local_tile("
                f"{rec.aux_tensor_name}, ({tcgen05_aux_bn},), "
                f"({tile_coord_n},))\n"
                f"    {stage.gmem_part} = {stage.thr_copy}.partition_S("
                f"{stage.gmem_tile})\n"
                f"    {stage.smem_part} = {stage.thr_copy}.partition_D("
                f"{stage.smem})\n"
                f"    {stage.coord} = {stage.thr_copy}.partition_S("
                f"cute.make_identity_tensor({tcgen05_aux_bn}))\n"
                f"    {stage.limit} = min({n_size} - ({base_indices[1]}), "
                f"cutlass.Int32({stage.aux_extent}) - ({base_indices[1]}), "
                f"cutlass.Int32({tcgen05_aux_bn}))\n"
                f"    {stage.pred} = cute.make_rmem_tensor("
                f"(1, cute.size({stage.smem_part}.shape[1])), cutlass.Boolean)\n"
                f"    for _rowvec_i in cutlass.range("
                f"cute.size({stage.smem_part}.shape[1]), unroll_full=True):\n"
                f"        {stage.pred}[0, _rowvec_i] = "
                f"{stage.coord}[0, _rowvec_i] < {stage.limit}\n"
                f"    cute.copy({stage.tiled_copy}, {stage.gmem_part}, "
                f"{stage.smem_part}, pred={stage.pred})\n"
                f"    cute.arch.fence_acq_rel_cta()\n"
                f"    {epilog_sync_barrier}.arrive_and_wait()"
            )
        return lines

    def _simt_edge_coord_subtile_source(indent: str) -> str:
        return (
            f"{indent}{coord_tile} = cute.local_tile("
            f"cute.make_identity_tensor(({m_size}, {n_size})), "
            f"({tcgen05_aux_bm}, {tcgen05_aux_bn}), "
            f"({tile_coord_m}, {tile_coord_n}))\n"
            f"{indent}{tccc_base} = {tcgen05_aux_thr_mma}.partition_C("
            f"{coord_tile})\n"
            f"{indent}{tccc} = "
            "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tccc_base})\n"
            f"{indent}{tccc_epi} = cute.flat_divide({tccc}, {epi_tile})\n"
            f"{indent}{ttr_cc} = {thr_copy_t2r}.partition_D({tccc_epi})\n"
            f"{indent}{ttr_cc_grouped} = cute.group_modes({ttr_cc}, 3, "
            f"cute.rank({ttr_cc}))\n"
            f"{indent}{ttr_cc_subtile} = {ttr_cc_grouped}[(None, None, None, "
            f"cutlass.Int32(_tcgen05_subtile))]\n"
        )

    def _simt_edge_scalar_copy_source(
        indent: str, src: str, dst: str, *, include_coord_setup: bool = True
    ) -> str:
        # General SIMT edge copies keep the scalar loop unless the call site
        # retile below can build a predicate with one lane per logical element.
        return (
            (_simt_edge_coord_subtile_source(indent) if include_coord_setup else "")
            + f"{indent}for _edge_i in range(cute.size({src}.shape)):\n"
            f"{indent}    _coord = {ttr_cc_subtile}[_edge_i]\n"
            f"{indent}    if cute.elem_less(_coord, ({m_size}, {n_size})):\n"
            f"{indent}        {dst}[_edge_i] = {src}[_edge_i]\n"
        )

    def _simt_edge_logical_divide_copy_source(
        indent: str,
        src: str,
        dst: str,
        *,
        include_coord_setup: bool = True,
        var_prefix: str = "tcgen05_edge",
        copy_atom: str | None = None,
    ) -> str:
        # Shared edge-only vector copy emitter. The make_layout(1) retile gives
        # cute.copy a per-element predicate, while var_prefix/copy_atom let the
        # same shape drive D stores or exact-aux G2R register loads.
        copy_atom = copy_atom or simt_atom
        edge_src = df.new_var(f"{var_prefix}_src")
        edge_dst = df.new_var(f"{var_prefix}_dst")
        edge_coord = df.new_var(f"{var_prefix}_coord")
        edge_pred = df.new_var(f"{var_prefix}_pred")
        return (
            (_simt_edge_coord_subtile_source(indent) if include_coord_setup else "")
            + f"{indent}{edge_src} = cute.logical_divide({src}, cute.make_layout(1))\n"
            f"{indent}{edge_dst} = cute.logical_divide({dst}, cute.make_layout(1))\n"
            f"{indent}{edge_coord} = cute.logical_divide({ttr_cc_subtile}, cute.make_layout(1))\n"
            f"{indent}{edge_pred} = cute.make_rmem_tensor((1, {edge_src}.shape[1]), cutlass.Boolean)\n"
            f"{indent}for _edge_i in range(cute.size({edge_src}.shape[1])):\n"
            f"{indent}    _coord = {edge_coord}[0, _edge_i]\n"
            f"{indent}    {edge_pred}[0, _edge_i] = cute.elem_less(_coord, ({m_size}, {n_size}))\n"
            f"{indent}cute.copy({copy_atom}, {edge_src}, {edge_dst}, pred={edge_pred})\n"
        )

    def _aux_tile_setup_lines(
        *,
        thr_copy_t2r_var: str,
        define_thr_copy_t2r: bool,
        force_gmem_aux: bool = False,
        retile_for_r2s: bool = False,
    ) -> list[str]:
        """Emit the per-output-tile aux partitioning lines.

        Each line goes once per output tile, before the per-subtile
        loop. Mirrors the existing ``tcgc -> tcgc_planned -> tcgc_epi
        -> ttr_gc -> ttr_gc_grouped`` pipeline used for the result D
        tensor, but partitions a separate auxiliary GMEM tensor per
        chain step. Calls ``thr_mma.partition_C`` and
        ``thr_copy_t2r.partition_D`` against the aux tile so the
        per-thread layout matches D's layout exactly — both the
        exact-shape (``residual[tile_m, tile_n]``) and rank-1
        broadcast (``bias[tile_n]`` / ``bias[tile_m]``) forms feed
        the same downstream pipeline.

        For the broadcast form the helper first builds a 2-D view
        of the underlying rank-1 tensor with stride 0 on the
        orthogonal axis (see :class:`_AuxiliaryTensorStep` for the
        canonical contract).

        When ``define_thr_copy_t2r`` is True the helper emits the
        ``thr_copy_t2r = tiled_copy_t2r.get_slice(...)`` line first
        (the TMA-store path does not otherwise create
        ``thr_copy_t2r``); the SIMT path passes False because it
        already creates the slice as part of its existing partition
        pipeline. ``retile_for_r2s`` mirrors Quack's SM100 epilogue
        visitor layout: TMA-store chains read aux operands in the
        R2S-retiled layout so the chain carrier can be ``tRS_rAcc`` /
        ``tRS_rD`` instead of the raw T2R fragment layout.
        ``force_gmem_aux`` is used by the hybrid edge-only
        SIMT path: C-input staging is only safe for full tiles because
        the producer-side bulk copy is not predicated for M/N fringes.
        """
        lines: list[str] = []
        if not aux_step_records:
            return lines
        if define_thr_copy_t2r:
            lines.append(
                f"{thr_copy_t2r_var} = "
                f"{tiled_copy_t2r}.get_slice({tcgen05_aux_epi_tidx})"
            )
        for aux_idx, rec in enumerate(aux_step_records):
            staged_ring_name = aux_ring_smem_names[aux_idx]
            rowvec_stage = rowvec_aux_stage_records[aux_idx]
            if (
                use_aux_smem_source
                and staged_ring_name is not None
                and not force_gmem_aux
            ):
                # C-input warp productive-body gate is open for this exact-shape
                # descriptor: build the Quack-style SMEM->register path. Rowvec
                # broadcast records are not staged and fall through to the GMEM
                # partition setup below.
                assert aux_matmul_plan is not None
                ring_idx = aux_ring_index_by_step[aux_idx]
                assert ring_idx is not None
                aux_dtype_str = backend.dtype_str(
                    aux_matmul_plan.c_input_aux_tensor_descriptors[
                        ring_idx
                    ].host_tensor_val.dtype
                )
                tiled_copy_s2r_var = f"{rec.aux_tile}_tiled_copy_s2r"
                thr_copy_s2r_var = f"{rec.aux_tile}_thr_copy_s2r"
                tsr_sc_var = f"{rec.aux_tile}_tSR_sC"
                trs_rc_var = f"{rec.aux_tile}_tRS_rC"
                tsr_rc_var = f"{rec.aux_tile}_tSR_rC"
                rmem_shape_expr = (
                    f"{trs_rd}.layout" if retile_for_r2s else f"{ttr_racc}.shape"
                )
                lines.extend(
                    [
                        (
                            f"{tiled_copy_s2r_var} = "
                            f"cute.make_tiled_copy_D("
                            f"cute.make_copy_atom("
                            f"cute.nvgpu.CopyUniversalOp(), "
                            f"{aux_dtype_str}), "
                            f"{tiled_copy_t2r})"
                        ),
                        (
                            f"{thr_copy_s2r_var} = "
                            f"{tiled_copy_s2r_var}.get_slice("
                            f"{tcgen05_aux_epi_tidx})"
                        ),
                        (
                            f"{tsr_sc_var} = "
                            f"{thr_copy_s2r_var}.partition_S("
                            f"{staged_ring_name})"
                        ),
                        (
                            f"{trs_rc_var} = cute.make_rmem_tensor("
                            f"{rmem_shape_expr}, {aux_dtype_str})"
                        ),
                        (f"{tsr_rc_var} = {tiled_copy_s2r_var}.retile({trs_rc_var})"),
                    ]
                )
                continue

            if rec.broadcast_axis is None:
                # Exact-shape rank-2 aux: slice the per-tile region
                # of the underlying 2-D tensor directly.
                source_for_local_tile = rec.aux_tensor_name
                aux_tile_is_local = False
            elif rowvec_stage is not None:
                assert rec.broadcast_axis == 1
                assert rec.aux_view2d is not None
                # The compact SMEM rowvec is allocated and populated per output
                # tile, so its 2-D broadcast view is already tile-sized.
                lines.append(
                    f"{rec.aux_view2d} = cute.make_tensor("
                    f"{rowvec_stage.smem}.iterator, "
                    f"cute.make_layout(({tcgen05_bm}, {tcgen05_bn}), "
                    f"stride=(0, 1)))"
                )
                source_for_local_tile = rec.aux_view2d
                aux_tile_is_local = True
            else:
                # M-axis (row) broadcast aux: build a 2-D logical view
                # over the underlying tensor's ``.iterator`` with
                # stride 0 on the leading (M) axis and stride 1 on the
                # trailing (N) axis. Stride 0 on M causes every lane
                # "owning" output ``(m, n)`` to read the same source
                # element regardless of m, which is the broadcast
                # semantic shared by two accepted forms:
                #   * ``broadcast_axis == 1`` — a bare rank-1 tensor
                #     ``bias[tile_n]`` with shape ``(N,)`` (rank-1 RHS
                #     aligns to the trailing axis under PyTorch
                #     broadcasting).
                #   * ``broadcast_axis == 0`` — an explicit ``(1, N)``
                #     tensor ``bias[tile_m, tile_n]`` (row 0 broadcasts
                #     over M).
                # Both have the same contiguous N-major memory layout
                # (element ``(0, n)`` at offset ``n``), so the
                # stride-(0, 1) view over ``.iterator`` is identical
                # and feeds the same ``partition_C → flat_divide →
                # partition_D`` pipeline used by exact-shape aux.
                # Mirrors Quack's ``RowVecLoad`` epilogue
                # (``quack/quack/epi_ops.py``). The classifier
                # (``aux_tensor_load_kind``) admits only these two
                # broadcast shapes; everything else drops to the
                # loud-failure backstop.
                assert rec.broadcast_axis in (0, 1)
                assert rec.aux_view2d is not None
                lines.append(
                    f"{rec.aux_view2d} = cute.make_tensor("
                    f"{rec.aux_tensor_name}.iterator, "
                    f"cute.make_layout(({m_size}, {n_size}), "
                    f"stride=(0, 1)))"
                )
                source_for_local_tile = rec.aux_view2d
                aux_tile_is_local = False
            if aux_tile_is_local:
                lines.append(f"{rec.aux_tile} = {source_for_local_tile}")
            else:
                lines.append(
                    f"{rec.aux_tile} = cute.local_tile("
                    f"{source_for_local_tile}, ({tcgen05_bm}, {tcgen05_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                )
            lines.extend(
                [
                    (
                        f"{rec.aux_part_base} = "
                        f"{tcgen05_thr_mma}.partition_C({rec.aux_tile})"
                    ),
                    (
                        f"{rec.aux_xfm} = "
                        "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                        f"{rec.aux_part_base})"
                    ),
                    (
                        f"{rec.aux_planned} = cute.make_tensor("
                        f"{rec.aux_xfm}.iterator, "
                        f"cute.append(cute.append(cute.append({rec.aux_xfm}.layout, "
                        f"{tcgen05_aux_epilogue_rest_mode}), "
                        f"{tcgen05_aux_epilogue_rest_mode}), "
                        f"{tcgen05_aux_epilogue_rest_mode}))"
                    ),
                    (
                        f"{rec.aux_epi} = cute.flat_divide("
                        f"{rec.aux_planned}, {epi_tile})"
                    ),
                    (f"{rec.ttr_aux} = {thr_copy_t2r_var}.partition_D({rec.aux_epi})"),
                    *(
                        [f"{rec.ttr_aux} = {tiled_copy_r2s}.retile({rec.ttr_aux})"]
                        if retile_for_r2s
                        else []
                    ),
                    (
                        f"{rec.ttr_aux_grouped} = cute.group_modes("
                        f"{rec.ttr_aux}, 3, cute.rank({rec.ttr_aux}))"
                    ),
                ]
            )
        return lines

    def _aux_subtile_load_source(
        prelude_indent: str,
        *,
        force_simt_edge_aux: bool = False,
        safe_direct_aux_with_full_tile: bool = False,
    ) -> str:
        """Per-subtile aux GMEM-load source lines (one per aux step).

        Each step emits the per-thread GMEM subtile slice of
        ``tTR_gAux_grouped_<idx>`` followed by a ``.load()`` call
        into the per-subtile ``tcgen05_aux_loaded_*`` local. Goes
        inside the per-subtile loop body. The slice depends on
        ``_tcgen05_subtile`` so it cannot be hoisted out of the
        loop entirely. Splice sites choose where to place this
        block: the default TMA-store path keeps it after the
        c_pipeline acquire, acc ``consumer_wait``, and t2r
        async TMEM→reg copy so residual and bias fragments are
        not live through the store-prefix waits. SIMT fallback
        concatenates it with the chain prelude because it does not
        use the TMA aux-pipeline shape; diagnostic helper paths keep
        the same flat prelude order for unary chains and reject aux
        chains at validation time.

        Cycle 39 (GPU 6) replan note: an alternative form that
        pre-loads all subtile aux into a per-thread register
        tensor outside the per-subtile loop (``cute.autovec_copy``
        from ``tTR_gAux_grouped_<idx>`` into a fresh
        ``tTR_rAux_<idx>``) was tested. The single cooperative
        LDG fired before the per-subtile loop, but the multi-
        subtile register tensor pushed local-memory spills from
        356k to 1.17M and grew kernel duration from 308 µs to
        332 µs. The per-subtile GMEM load form below pays one
        LDG per chain-add but the compiler IR / SASS scheduler
        already lifts the LDG ahead of the chain-add given the
        independent dependency graph.

        Cycle 69 found a related spill tradeoff inside the default
        TMA-store body: placing the per-subtile aux LDG after the
        acquire/T2R prefix removes most local-memory spill traffic,
        so that path no longer uses the older top-of-loop hoist.
        """
        if not aux_step_records:
            return ""
        lines: list[str] = []
        force_simt_edge_coord_emitted = False
        if use_aux_smem_source and not force_simt_edge_aux:
            # C-input warp productive-body gate is open: per-subtile
            # SMEM ring staging. Each subtile iteration waits on
            # ``c_pipeline_aux`` for the producer warp to fill the
            # active stage, then issues one filtered
            # ``cute.copy(tiled_copy_s2r, tSR_sC[..., stage], tSR_rC)``
            # per descriptor to load the active stage into the
            # per-thread register tensor (Quack's
            # ``epilog_smem_load_and_partition`` flow from
            # ``quack/gemm_sm100.py``: ``tiled_copy_s2r`` is built via
            # ``make_tiled_copy_D`` against ``tiled_copy_t2r``;
            # ``tSR_sC = thr_copy_s2r.partition_S(sC_ring)`` selects
            # the SMEM source; ``tSR_rC`` is a re-layout view of the
            # same register memory as ``tRS_rC``). The chain reads
            # ``tRS_rC.load()`` (== ``aux_loaded``). The post-copy
            # lane-0-gated release plus state advance run in the
            # same per-subtile iteration so the producer can refill
            # the same stage on the very next persistent tile
            # (matches the consumer cooperative-group arrive count
            # of ``epi_warp_count`` set by
            # ``_emit_tcgen05_aux_pipeline_setup``).
            #
            # Note: ``partition_D(smem_stage).load()`` on
            # ``thr_copy_t2r`` (an earlier prior-subagent variant)
            # produced a deadlocking SMEM read — TMEM→reg-shaped
            # partition_D applied to a SMEM tensor does not
            # compose with the producer's
            # ``make_tiled_copy_tv`` cooperative copy in a way the
            # mbarrier handshake recognizes. The Quack-style
            # ``tiled_copy_s2r`` flow is the canonical CUTLASS-DSL
            # pattern.
            lines.append(
                f"{prelude_indent}{aux_pipeline_name}.consumer_wait("
                f"{aux_consumer_state_name})\n"
            )
            if aux_pipeline_uses_tma_load:
                # TMA producer writes arrive through the async proxy; after the
                # pipeline wait, fence that view before generic SMEM reads.
                # The warp sync mirrors CUTLASS/Quack's TMA-load consumer
                # sequence so every lane observes the fenced view before the
                # per-lane SMEM->register copy below.
                lines.extend(
                    [
                        f"{prelude_indent}cute.arch.fence_view_async_shared()\n",
                        f"{prelude_indent}cute.arch.sync_warp()\n",
                    ]
                )
            for aux_idx, rec in enumerate(aux_step_records):
                if aux_ring_smem_names[aux_idx] is None:
                    continue
                tiled_copy_s2r_var = f"{rec.aux_tile}_tiled_copy_s2r"
                tsr_sc_var = f"{rec.aux_tile}_tSR_sC"
                trs_rc_var = f"{rec.aux_tile}_tRS_rC"
                tsr_rc_var = f"{rec.aux_tile}_tSR_rC"
                lines.extend(
                    [
                        (
                            # The S2R visitor layout can carry zero/unused lanes;
                            # filtering keeps the residual SMEM read footprint
                            # aligned with the lanes that feed the R2S fragment.
                            f"{prelude_indent}cute.copy("
                            f"{tiled_copy_s2r_var}, "
                            f"cute.filter_zeros({tsr_sc_var}[None, None, None, "
                            f"{aux_consumer_state_name}.index]), "
                            f"cute.filter_zeros({tsr_rc_var}))\n"
                        ),
                        (f"{prelude_indent}{rec.aux_loaded} = {trs_rc_var}.load()\n"),
                    ]
                )
            lines.extend(
                [
                    (
                        f"{prelude_indent}with cute.arch.elect_one():\n"
                        f"{prelude_indent}    {aux_pipeline_name}.consumer_release("
                        f"{aux_consumer_state_name})\n"
                    ),
                    emit_pipeline_advance(
                        aux_consumer_state_name, indent=prelude_indent
                    )
                    + "\n",
                ]
            )
        for aux_idx, rec in enumerate(aux_step_records):
            rowvec_stage = rowvec_aux_stage_records[aux_idx]
            if (
                use_aux_smem_source
                and not force_simt_edge_aux
                and aux_ring_smem_names[aux_idx] is not None
            ):
                continue
            if force_simt_edge_aux:
                include_coord_setup = not force_simt_edge_coord_emitted
                force_simt_edge_coord_emitted = True
                if rec.broadcast_axis is None:
                    edge_aux_copy_source = _simt_edge_logical_divide_copy_source(
                        prelude_indent,
                        rec.ttr_aux_subtile,
                        rec.aux_rmem,
                        include_coord_setup=include_coord_setup,
                        var_prefix=f"{rec.aux_rmem}_edge",
                        copy_atom=simt_edge_aux_atoms[aux_idx],
                    )
                else:
                    # Rowvec broadcast stayed scalar in the cycle-74 ablation:
                    # vectorizing it did not reduce stack pressure or runtime.
                    edge_aux_copy_source = _simt_edge_scalar_copy_source(
                        prelude_indent,
                        rec.ttr_aux_subtile,
                        rec.aux_rmem,
                        include_coord_setup=include_coord_setup,
                    )
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_rmem} = "
                    f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.shape, "
                    f"{rec.aux_dtype})\n"
                    f"{prelude_indent}{rec.aux_rmem}.fill(0)\n"
                    + edge_aux_copy_source
                    + f"{prelude_indent}{rec.aux_loaded} = "
                    f"{rec.aux_rmem}.load()\n"
                )
                continue
            if rowvec_stage is None and (
                safe_direct_aux_with_full_tile or not tcgen05_aux_use_tma_store_epilogue
            ):
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_loaded} = cute.full("
                    f"{rec.ttr_aux_subtile}.shape, 0, {rec.aux_dtype})\n"
                    f"{prelude_indent}if {full_tile}:\n"
                    f"{prelude_indent}    {rec.aux_loaded} = "
                    f"{rec.ttr_aux_subtile}.load()\n"
                    f"{prelude_indent}else:\n"
                    f"{prelude_indent}    {rec.aux_rmem} = "
                    f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.shape, "
                    f"{rec.aux_dtype})\n"
                    f"{prelude_indent}    {rec.aux_rmem}.fill(0)\n"
                    f"{_simt_edge_scalar_copy_source(prelude_indent + '    ', rec.ttr_aux_subtile, rec.aux_rmem)}"
                    f"{prelude_indent}    {rec.aux_loaded} = "
                    f"{rec.aux_rmem}.load()\n"
                )
                continue
            if rowvec_stage is not None and not force_simt_edge_aux:
                # Row-vector staging broadcasts through a stride-0 M mode; filter
                # that layout so the SMEM read does not reload duplicate lanes.
                lines.append(
                    f"{prelude_indent}{rec.ttr_aux_subtile} = "
                    f"{rec.ttr_aux_grouped}"
                    "[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"{prelude_indent}{rec.aux_rmem} = "
                    f"cute.make_rmem_tensor({rec.ttr_aux_subtile}.layout, "
                    f"{rec.aux_dtype})\n"
                    f"{prelude_indent}cute.autovec_copy("
                    f"cute.filter_zeros({rec.ttr_aux_subtile}), "
                    f"cute.filter_zeros({rec.aux_rmem}))\n"
                    f"{prelude_indent}{rec.aux_loaded} = {rec.aux_rmem}.load()\n"
                )
                continue
            lines.extend(
                [
                    (
                        f"{prelude_indent}{rec.ttr_aux_subtile} = "
                        f"{rec.ttr_aux_grouped}"
                        f"[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    ),
                    (
                        f"{prelude_indent}{rec.aux_loaded} = "
                        f"{rec.ttr_aux_subtile}.load()\n"
                    ),
                ]
            )
        return "".join(lines)

    # Render the per-thread carrier expression for the accumulator
    # vector. The identity epilogue (no chain or empty chain) emits
    # the original `rAcc.load().to(target_dtype)` line. When a
    # chain is present, hoist `rAcc.load()` to a local TensorSSA so
    # the chain reads the loaded vector once; for chains with
    # auxiliary-tensor steps, also emit per-subtile aux-load lines
    # that bind the aux locals the chain references. Each splice
    # site below uses the appropriate carrier name (`ttr_racc` for
    # the SIMT path, `trs_racc` for the TMA path, and
    # `tcgen05_tRS_rAcc` for the @cute.jit module helper). The
    # returned snippet is a sequence of zero-or-more prelude
    # statements (each newline-terminated, indented with
    # `prelude_indent`) plus the assignment expression for
    # `tcgen05_acc_vec`.
    def _splice_acc_vec(
        carrier_name: str,
        prelude_indent: str,
        *,
        force_simt_edge_aux: bool = False,
        safe_direct_aux_with_full_tile: bool = False,
    ) -> tuple[str, str, str]:
        """Return ``(early_aux_prelude, late_prelude, assignment_rhs)``.

        ``early_aux_prelude`` is the per-subtile auxiliary-tensor LDG
        block (``ttr_aux_subtile = ...``; ``aux_loaded = .load()``) and
        is empty when the chain has no aux steps. ``late_prelude``
        holds the ``acc_loaded = carrier.load()`` and the chain-step
        renderings. ``assignment_rhs`` is the right-hand side of
        ``acc_vec = ...`` (without leading whitespace or the trailing
        newline). Both preludes are empty for the identity epilogue
        (no chain) — in that case ``assignment_rhs`` is the original
        ``carrier.load().to(target_dtype)`` expression.

        Each chain step renders into a fresh ``tcgen05_chain_step*``
        local so chain composition stays linear in source size — the
        relu template duplicates ``{inner}`` 5 times, so without per-
        step binding a 3-deep relu chain would emit 125x duplication
        and pessimize parse / IR-build time. Per-step locals keep
        the rendered source O(N) in chain depth and CuTe CSEs the
        loads at compile.

        Auxiliary-tensor chain steps additionally emit per-aux-step
        ``ttr_aux_subtile = ...`` slice + ``aux_loaded = ...`` lines
        (the per-tile aux setup runs once per output tile and is
        emitted by the splice site's surrounding scaffolding via
        ``_aux_tile_setup_lines()``). Splitting the aux LDG out of
        the chain prelude lets each splice site place the GMEM load
        where it best fits its live ranges. The default TMA-store
        splice now inserts it after the c_pipeline acquire, acc
        ``consumer_wait``, and t2r async TMEM→reg copy so residual
        and bias fragments are not live through those prefix waits.
        SIMT-store edge tiles use the same aux prelude, but route
        the aux load through a predicated copy before rendering the
        chain.
        """
        load_expr = f"{carrier_name}.load()"
        if epilogue_chain is None or not epilogue_chain.steps:
            return ("", "", f"{load_expr}.to({target_dtype})")
        loaded = df.new_var("tcgen05_acc_loaded")
        prelude_load = f"{prelude_indent}{loaded} = {load_expr}\n"
        early_aux_prelude = _aux_subtile_load_source(
            prelude_indent,
            force_simt_edge_aux=force_simt_edge_aux,
            safe_direct_aux_with_full_tile=safe_direct_aux_with_full_tile,
        )
        aux_locals: tuple[str, ...] = tuple(rec.aux_loaded for rec in aux_step_records)
        chain_prelude, final_expr = epilogue_chain.render_prelude_and_expr(
            loaded,
            df.new_var,
            prelude_indent,
            aux_locals_by_step=aux_locals or None,
        )
        return (
            early_aux_prelude,
            prelude_load + chain_prelude,
            f"({final_expr}).to({target_dtype})",
        )

    if tcgen05_value.use_tma_store_epilogue:
        df.placeholder_args.add(tensor_name)
        df.wrapper_only_params.extend(
            [tcgen05_value.tma_store_atom, tcgen05_value.tma_store_tensor]
        )
        if tcgen05_value.use_role_local_epi and tcgen05_value.role_local_tile_counter:
            df.cute_state.register_tcgen05_epi_role_tile_counter(
                tcgen05_value.role_local_tile_counter,
                increment_per_tile=not tcgen05_value.tma_store_full_tiles_only,
            )
        d_tma_plan: dict[str, object] = {
            "kind": "tcgen05_d_tma",
            "d_name": tensor_name,
            "bm": tcgen05_value.bm,
            "bn": tcgen05_value.bn,
            "c_stage_count": tcgen05_value.c_stage_count,
            "output_dtype": target_dtype,
            "kernel_args": [
                tcgen05_value.tma_store_atom,
                tcgen05_value.tma_store_tensor,
            ],
            **(
                {
                    "epi_tile_m": tcgen05_value.explicit_epi_tile_m,
                    "epi_tile_n": tcgen05_value.explicit_epi_tile_n,
                    "d_store_box_n": tcgen05_value.explicit_d_store_box_n,
                }
                if tcgen05_value.has_explicit_epilogue_tile
                else {}
            ),
        }
        state.codegen.cute_wrapper_plans.append(d_tma_plan)

    tcgen05_bm = tcgen05_value.bm
    tcgen05_bn = tcgen05_value.bn
    tcgen05_bk = tcgen05_value.bk
    tcgen05_epilog_sync_barrier_id = tcgen05_value.epilog_sync_barrier_id
    tcgen05_c_stage_count = tcgen05_value.c_stage_count
    tcgen05_is_two_cta = tcgen05_lifecycle.is_two_cta
    tcgen05_thr_mma = tcgen05_value.thr_mma
    full_tile_expr = (
        f"({base_indices[0]}) + cutlass.Int32({tcgen05_bm}) <= {m_size} "
        f"and ({base_indices[1]}) + cutlass.Int32({tcgen05_bn}) <= {n_size}"
    )

    def store_common_setup(
        gmem_tensor: str, *, include_full_tile: bool
    ) -> tuple[list[str], list[str]]:
        epi_tile_expr = tcgen05_explicit_store_tile_expr or (
            tcgen05_default_epilogue_tile_expr(
                tcgen05_bm,
                tcgen05_bn,
                target_dtype,
                c_layout="cutlass.utils.layout.LayoutEnum.ROW_MAJOR",
            )
        )
        static_setup = [
            (
                f"{kernel_desc} = type('Tcgen05KernelDesc', (), {{"
                f"'cta_tile_shape_mnk': ({tcgen05_bm}, {tcgen05_bn}, {tcgen05_bk}), "
                "'c_layout': cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                f"'c_dtype': {target_dtype}, "
                "'acc_dtype': cutlass.Float32, "
                f"'epilog_sync_bar_id': cutlass.Int32({tcgen05_epilog_sync_barrier_id}), "
                f"'epilogue_warp_id': ({epi_warp_ids}), "
                f"'num_c_stage': cutlass.Int32({tcgen05_c_stage_count}), "
                f"'use_2cta_instrs': {tcgen05_is_two_cta!s}"
                "})()"
            ),
            (
                # The fallback helper must receive the D-output dtype through
                # ``layout_c=`` / ``elem_ty_c=`` so it selects the same
                # with-source branch as the matmul-plan ``tcgen05_epi_tile``.
                # The explicit path instead uses the D-store box field directly.
                # Keep both forms in lockstep with the wrapper-side TMA atom.
                f"{epi_tile} = {epi_tile_expr}"
            ),
        ]
        tile_setup: list[str] = []
        if include_full_tile:
            tile_setup.append(f"{full_tile} = {full_tile_expr}")
        tile_setup.extend(
            [
                (
                    f"{gmem_tile} = cute.local_tile("
                    f"{gmem_tensor}, ({tcgen05_bm}, {tcgen05_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                ),
                f"{tcgc_base} = {tcgen05_thr_mma}.partition_C({gmem_tile})",
            ]
        )
        return static_setup, tile_setup

    simt_edge_only = tcgen05_value.tma_store_full_tiles_only
    simt_edge_aux_atoms: dict[int, str] = {}
    simt_edge_aux_atom_setup: list[str] = []
    if simt_edge_only:
        for aux_idx, rec in enumerate(aux_step_records):
            if rec.broadcast_axis is None:
                edge_aux_atom = df.new_var(f"{rec.aux_rmem}_edge_atom")
                simt_edge_aux_atoms[aux_idx] = edge_aux_atom
                # Use a per-aux atom typed to the aux dtype. Reusing the
                # output SIMT atom here was spill-free but slower on the
                # measured Target8 edge path.
                simt_edge_aux_atom_setup.append(
                    f"{edge_aux_atom} = "
                    f"cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                    f"{rec.aux_dtype})"
                )
    simt_static_store_setup, simt_tile_store_setup = store_common_setup(
        tensor_name, include_full_tile=not simt_edge_only
    )
    simt_early_aux, simt_late_prelude, simt_acc_vec_rhs = _splice_acc_vec(
        ttr_racc,
        "        ",
        force_simt_edge_aux=tcgen05_value.tma_store_full_tiles_only,
    )
    simt_acc_vec_prelude = simt_early_aux + simt_late_prelude
    tma_static_store_setup, tma_tile_store_setup = store_common_setup(
        tcgen05_value.tma_store_tensor,
        include_full_tile=partial_tma_needs_full_tile_guard,
    )
    # Role-local TMA stores reuse one C pipeline across work tiles. Static-full
    # kernels increment this counter once per role-local tile; hybrid
    # output-edge kernels increment it only in the full-tile branch so SIMT
    # fallback edge tiles do not perturb the C-pipeline SMEM stage sequence.
    tma_c_buffer_expr = "cutlass.Int32(_tcgen05_subtile)"
    if tcgen05_value.role_local_tile_counter:
        tma_c_buffer_expr = (
            f"{tcgen05_value.role_local_tile_counter} * "
            f"cutlass.Int32({subtile_count}) + cutlass.Int32(_tcgen05_subtile)"
        )
    simt_store_edge_coord_preloaded = simt_edge_only and bool(aux_steps_in_chain)
    if simt_edge_only:
        simt_store_copy_source = _simt_edge_logical_divide_copy_source(
            "        ",
            ttr_rd,
            ttr_gc_subtile,
            include_coord_setup=not simt_store_edge_coord_preloaded,
        )
    else:
        simt_store_copy_source = (
            f"        if {full_tile}:\n"
            f"            cute.copy({simt_atom}, {ttr_rd}, {ttr_gc_subtile})\n"
            f"        else:\n"
            f"{_simt_edge_scalar_copy_source('            ', ttr_rd, ttr_gc_subtile)}"
        )
    simt_store_body_core = [
        *simt_static_store_setup,
        *simt_tile_store_setup,
        (
            f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgc_base})"
        ),
        (
            f"{tcgc_planned} = cute.make_tensor("
            f"{tcgc}.iterator, "
            f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}))"
        ),
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
        (
            f"{tiled_copy_t2r}, {ttr_tacc_base}, {ttr_racc} = "
            "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition("
            f"{kernel_desc}, {tcgen05_value.epi_tidx}, {tacc}, {tcgc_planned}, {epi_tile}, {tcgen05_lifecycle.is_two_cta!s})"
        ),
        f"{thr_copy_t2r} = {tiled_copy_t2r}.get_slice({tcgen05_value.epi_tidx})",
        f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
        f"{ttr_gc} = {thr_copy_t2r}.partition_D({tcgc_epi})",
        (
            f"{ttr_tacc_stage} = {ttr_tacc_base}["
            f"(None, None, None, None, None, {tcgen05_acc_stage_index_expr})]"
        ),
        *(
            []
            if is_secondary_store
            else [
                (
                    f"if {tcgen05_lifecycle.epi_active}:\n"
                    f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})"
                )
            ]
        ),
        f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
        f"{ttr_gc_grouped} = cute.group_modes({ttr_gc}, 3, cute.rank({ttr_gc}))",
        # Per-aux-step partitioning lines (one chain per auxiliary
        # tensor). No-op when the chain has no aux steps; generated
        # source is byte-identical to the unary-chain shape for
        # unary chains and to the identity-store golden for identity
        # stores.
        *_aux_tile_setup_lines(
            thr_copy_t2r_var=thr_copy_t2r,
            define_thr_copy_t2r=False,
            force_gmem_aux=simt_edge_only,
        ),
        (
            f"{ttr_racc} = cute.make_rmem_tensor("
            f"{ttr_gc_grouped}[(None, None, None, 0)].shape, cutlass.Float32)"
        ),
        f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})",
        (
            f"{mcld} = cute.max_common_layout("
            f"{ttr_rd}.layout, {ttr_gc_grouped}[(None, None, None, 0)].layout)"
        ),
        (
            f"{num_bits} = min("
            f"{ttr_gc_grouped}.iterator.alignment * 8, "
            f"cute.size({mcld}) * {target_dtype}.width, 256)"
        ),
        (
            f"{simt_atom} = cute.make_copy_atom("
            f"cute.nvgpu.CopyR2GOp(), {target_dtype}, "
            f"num_bits_per_copy={num_bits}, "
            f"l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)"
        ),
        *simt_edge_aux_atom_setup,
        f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
        (
            # Per-subtile loop: TMEM->reg (t2r) first, then reg->GMEM (SIMT
            # store). On the last subtile we release the acc consumer slot
            # *before* the GMEM store so the next mainloop tile's MMA can
            # producer_acquire the TMEM stage and begin issuing UMMAs while
            # this tile's epilogue is still draining to GMEM. This mirrors the
            # release-acc-inside-the-subtile-loop pattern in Quack's sm100
            # gemm epilogue. Without c_pipeline SMEM staging we can only
            # release after the final t2r (not per-subtile), but even one
            # tile of overlap measurably improves the wide tcgen05 path on
            # B200. `cutlass.range(..., unroll_full=True)` keeps the loop
            # statically unrolled so `tiled_copy_t2r` (a TiledCopy that wraps
            # a tcgen05 tmem_load atom) is not captured as an scf.for iter_arg
            # — the cute-to-nvvm pass cannot legalize that conversion through
            # iter_args and aborts during compile.
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"    if {tcgen05_lifecycle.epi_active}:\n"
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        {ttr_gc_subtile} = {ttr_gc_grouped}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
            f"{simt_acc_vec_prelude}"
            f"        {acc_vec} = {simt_acc_vec_rhs}\n"
            f"        {ttr_rd}.store({acc_vec})\n"
            # The secondary fan-out store reuses the still-live accumulator and
            # must not release it; the primary store owns the release + advance.
            + (
                ""
                if is_secondary_store
                else (
                    f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
                    # `cute.copy(t2r, ...)` issues async TMEM->reg loads.
                    # Releasing the acc consumer slot lets the MMA producer
                    # re-acquire the TMEM stage and issue UMMAs that overwrite
                    # TMEM, so we must fence the in-flight async TMEM loads
                    # first to avoid a race on the last subtile's `ttr_racc` /
                    # `ttr_rd` data. This matches Quack's sm100 gemm
                    # fence-before-release pattern.
                    f"            cute.arch.fence_view_async_tmem_load()\n"
                    f"            with cute.arch.elect_one():\n"
                    f"                {tcgen05_lifecycle.acc_pipeline}.consumer_release({tcgen05_lifecycle.acc_consumer_state})\n"
                )
            )
            + f"{simt_store_copy_source}"
            # Advance is a per-thread local state update, so it intentionally
            # stays outside elect_one; only the mbarrier release is elected.
            + (
                ""
                if is_secondary_store
                else (
                    f"if {tcgen05_lifecycle.epi_active}:\n"
                    + emit_pipeline_advance(
                        tcgen05_lifecycle.acc_consumer_state, indent="    "
                    )
                )
            )
        ),
    ]
    # Workstream A Stage 4 (cycle 93, Path B): C-store producer->consumer edge.
    # Mirrors ``_emit_tcgen05_aux_pipeline_setup``'s SIMT PipelineAsync shape.
    # producer_arrive_count = ``epi_warp_count`` (per-warp: each of the 4 epi
    # warps arrives once via ``elect_one`` after R2S + fence); consumer_arrive
    # _count = 1 (the single store warp); num_stages = ``c_stage_count`` so the
    # store can lag up to ``c_stages`` subtiles behind the epi warps' T2R/R2S.
    # Producer (epi ``producer_commit``) AND consumer (store ``consumer_wait`` /
    # ``consumer_release``) BOTH land in this commit so the ring is never a
    # one-sided handshake that wedges only after wrapping the depth (the
    # cycle-2a partial-handshake lesson).
    c_store_edge_setup = (
        [
            (
                f"{c_store_edge_barriers} = cute.arch.alloc_smem("
                f"cutlass.Int64, cutlass.Int32({tcgen05_value.c_stage_count * 2}))"
            ),
            (
                f"{c_store_edge_producer_group} = cutlass.pipeline.CooperativeGroup("
                f"cutlass.pipeline.Agent.Thread, "
                f"cutlass.Int32({tcgen05_value.epi_warp_count}))"
            ),
            (
                f"{c_store_edge_consumer_group} = cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))"
            ),
            (
                f"{c_store_edge} = cutlass.pipeline.PipelineAsync.create("
                f"num_stages={tcgen05_value.c_stage_count}, "
                f"producer_group={c_store_edge_producer_group}, "
                f"consumer_group={c_store_edge_consumer_group}, "
                f"barrier_storage={c_store_edge_barriers})"
            ),
            (
                f"{c_store_edge_producer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Producer, "
                f"{tcgen05_value.c_stage_count})"
            ),
            (
                f"{c_store_edge_consumer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, "
                f"{tcgen05_value.c_stage_count})"
            ),
            (
                f"{c_store_edge_release_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, "
                f"{tcgen05_value.c_stage_count})"
            ),
        ]
        if has_store_warp
        else []
    )
    tma_store_pipeline_setup = [
        (
            f"{epilog_sync_barrier} = cutlass.pipeline.NamedBarrier("
            f"barrier_id=cutlass.Int32({tcgen05_value.epilog_sync_barrier_id}), "
            f"num_threads=cutlass.Int32({tcgen05_value.epi_warp_count * 32}))"
        ),
        *c_store_edge_setup,
        (
            f"{c_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
            f"cutlass.pipeline.Agent.Thread, cutlass.Int32({tcgen05_value.epi_warp_count * 32}))"
        ),
        (
            f"{c_pipeline} = cutlass.pipeline.PipelineTmaStore.create("
            f"num_stages={tcgen05_value.c_stage_count}, "
            f"producer_group={c_pipeline_producer_group})"
        ),
    ]
    c_acquire_placement = state.device_function.config.get(
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
        TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    )
    acc_wait_placement = state.device_function.config.get(
        TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
        TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    )
    c_store_mode = state.device_function.config.get(
        TCGEN05_C_STORE_MODE_CONFIG_KEY,
        TCGEN05_C_STORE_MODE_NORMAL,
    )
    epilogue_layout = state.device_function.config.get(
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_NORMAL,
    )
    diagnose_first_c_acquire_in_loop = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
    )
    diagnose_later_c_acquire_before_barrier = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER
    )
    diagnose_acc_wait_before_subtile_loop = (
        acc_wait_placement == TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
    )
    diagnose_skip_epilogue_store = (
        c_store_mode == TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
    )
    diagnose_split_first_t2r = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
    )
    diagnose_split_acc_t2r_store_tail = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL
    )
    diagnose_module_helper_acc_t2r = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R
    )
    diagnose_module_helper_store_tail = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL
    )
    diagnose_split_epilogue_layout = (
        diagnose_split_first_t2r
        or diagnose_split_acc_t2r_store_tail
        or diagnose_module_helper_acc_t2r
        or diagnose_module_helper_store_tail
    )
    if tcgen05_pure_matmul_object is not None and diagnose_split_epilogue_layout:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' does not support "
            f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r}",
        )
    if tcgen05_pure_matmul_object is not None and has_store_warp:
        # Workstream A Stage 4 (cycle 93) wires the store-warp tail split into
        # the non-pure ROLE_LOCAL_WITH_SCHEDULER path only. The pure-matmul
        # role-lifecycle object renders its own tail (``render_tma_store_tail
        # _region``) and is gated out here so a store warp never silently lands
        # on the unsplit pure tail (a correctness break). Stage 5 may wire it.
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' does not support "
            "tcgen05_warp_spec_store_warps>0 (Workstream A Stage 4 wires the "
            "store-warp epilogue split into the non-pure WITH_SCHEDULER path)",
        )
    # The diagnostic split / module-helper epilogue layouts route the
    # per-subtile tail through helpers that emit ONLY the ``if epi_active``
    # half under ``has_store_warp`` (and ``module_helper_store_tail`` keeps the
    # OLD two-barrier warp-0 ``c_pipeline`` tail while the main path suppressed
    # the matching acquires) — so the C-store edge would have no consumer and
    # wedge once the ring wraps, or the ``c_pipeline`` commit/acquire counts
    # mismatch. They are diagnostic-only source-boundary layouts; production
    # uses the DEFAULT layout, so reject the combination loudly (same guard
    # class as the pure-matmul tail above). ``split_first_t2r`` routes through
    # ``tma_store_subtile_body`` and IS handled by the Stage-4 split, so it is
    # intentionally excluded.
    if has_store_warp and (
        diagnose_split_acc_t2r_store_tail
        or diagnose_module_helper_acc_t2r
        or diagnose_module_helper_store_tail
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} does not "
            "support tcgen05_warp_spec_store_warps>0 (the diagnostic split / "
            "module-helper epilogue layouts do not emit the store-warp tail "
            "half of the Workstream A Stage 4 split; use the default layout)",
        )
    if tcgen05_pure_matmul_object is not None:
        pure_c_store_pipeline = Tcgen05TmaStorePipelineParams(
            c_pipeline=c_pipeline,
            warp_idx=tcgen05_value.warp_idx,
        )
        tma_store_pipeline_tail = (
            tcgen05_pure_matmul_object.render_c_store_pipeline_tail(
                pure_c_store_pipeline
            )
        )
        tma_store_first_subtile_acquire = (
            tcgen05_pure_matmul_object.render_c_store_pre_loop_acquire_lines(
                pure_c_store_pipeline,
                first_c_acquire_in_loop=diagnose_first_c_acquire_in_loop,
            )
        )
        tma_store_loop_first_subtile_acquire = (
            tcgen05_pure_matmul_object.render_c_store_loop_first_acquire(
                pure_c_store_pipeline,
                first_c_acquire_in_loop=diagnose_first_c_acquire_in_loop,
            )
        )
        tma_store_loop_later_subtile_acquire = (
            tcgen05_pure_matmul_object.render_c_store_loop_later_acquire(
                pure_c_store_pipeline,
                later_c_acquire_before_barrier=(
                    diagnose_later_c_acquire_before_barrier
                ),
            )
        )
        tma_store_loop_late_later_subtile_acquire = (
            tcgen05_pure_matmul_object.render_c_store_loop_late_later_acquire(
                pure_c_store_pipeline,
                later_c_acquire_before_barrier=(
                    diagnose_later_c_acquire_before_barrier
                ),
            )
        )
    else:
        # Workstream A Stage 4 (cycle 93, Path B): the ``c_pipeline``
        # (PipelineTmaStore) producer lifecycle is per-warp — its
        # ``producer_acquire`` is a ``cp_async_bulk_wait_group`` and its
        # ``producer_commit`` a ``cp_async_bulk_commit_group``, both scoped to
        # the warp that ISSUES the TMA-D bulk copy. So when a store warp owns
        # the TMA-D, the entire ``c_pipeline`` lifecycle (acquire + commit +
        # tail) moves onto the store warp: its ``wait_group`` reuse guard lives
        # in the store-warp tail (after the TMA-D + commit, gating the lagged
        # release), the epi warps' historical store-prefix acquire lines are
        # dropped (the C-ring is gated by the cross-warp C-store edge instead),
        # and ``producer_tail`` (final ``wait_group(0)``) stays on the store warp.
        c_pipeline_owner_predicate = (
            store_warp_predicate
            if has_store_warp
            else f"{tcgen05_value.warp_idx} == cutlass.Int32(0)"
        )
        first_acquire_role_gate = (
            f"{tcgen05_lifecycle.epi_active} and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0)"
        )
        tma_store_pipeline_tail = (
            f"if {c_pipeline_owner_predicate}:\n    {c_pipeline}.producer_tail()"
        )
        tma_store_first_subtile_acquire = (
            []
            if (diagnose_first_c_acquire_in_loop or has_store_warp)
            else [
                (f"if {first_acquire_role_gate}:\n    {c_pipeline}.producer_acquire()")
            ]
        )
        tma_store_loop_first_subtile_acquire = (
            (
                f"        if _tcgen05_subtile == 0 and "
                f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                f"            {c_pipeline}.producer_acquire()\n"
            )
            if (diagnose_first_c_acquire_in_loop and not has_store_warp)
            else ""
        )
        tma_store_loop_later_subtile_acquire = (
            ""
            if (diagnose_later_c_acquire_before_barrier or has_store_warp)
            else (
                f"        if _tcgen05_subtile != 0 and "
                f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                f"            {c_pipeline}.producer_acquire()\n"
            )
        )
        tma_store_loop_late_later_subtile_acquire = (
            (
                f"        if _tcgen05_subtile != 0 and "
                f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                f"            {c_pipeline}.producer_acquire()\n"
            )
            if (diagnose_later_c_acquire_before_barrier and not has_store_warp)
            else ""
        )
    if diagnose_split_epilogue_layout:
        if not (
            tcgen05_value.use_role_local_epi and tcgen05_value.use_tma_store_epilogue
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} "
                "requires the "
                "role-local TMA-store tcgen05 epilogue",
            )
        if not tcgen05_lifecycle.is_two_cta:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} requires "
                "CtaGroup.TWO",
            )
        # Conservative proxy for the validated static-full CtaGroup.TWO
        # two-or-more-subtile envelope; the exact subtile count is only
        # available after the CUTLASS epilogue partitioning below.
        if tcgen05_value.bn < TCGEN05_TWO_CTA_BLOCK_N:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} is only "
                f"validated for CtaGroup.TWO block_n >= {TCGEN05_TWO_CTA_BLOCK_N}",
            )
        # The diagnostic split-epilogue layouts emit the per-thread
        # chain into separate ``@cute.jit`` helpers (module-helper
        # layouts) or split source boundaries; the auxiliary-tensor
        # splice site needs per-tile aux setup that is not currently
        # plumbed into those helper signatures. Reject the
        # combination loudly so a user does not silently get a
        # kernel that drops the aux read. The diagnostic layouts
        # are only used for source-boundary investigation and do not
        # block any production path.
        if (
            diagnose_module_helper_acc_t2r
            or diagnose_module_helper_store_tail
            or diagnose_split_first_t2r
            or diagnose_split_acc_t2r_store_tail
        ) and aux_steps_in_chain:
            raise exc.BackendUnsupported(
                "cute",
                "auxiliary-tensor epilogue (e.g. "
                "`out[tile] = (acc + residual[tile]).to(dtype)`) is "
                f"not plumbed through {TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}="
                f"{epilogue_layout!r}. The diagnostic split-epilogue "
                "layouts are only used for source-boundary "
                "investigation; drop the layout config to use the "
                "default production layout.",
            )
    tma_store_split_first_subtile_acquire = (
        (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_first_c_acquire_in_loop
        else ""
    )
    tma_store_pre_loop_acc_wait = (
        [
            (
                f"if {tcgen05_lifecycle.epi_active}:\n"
                f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})"
            )
        ]
        if diagnose_acc_wait_before_subtile_loop and not is_secondary_store
        else []
    )
    tma_store_loop_acc_wait = (
        ""
        if diagnose_acc_wait_before_subtile_loop or is_secondary_store
        else (
            f"        if _tcgen05_subtile == 0:\n"
            f"            {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
        )
    )
    tma_store_split_first_acc_wait = (
        ""
        if diagnose_acc_wait_before_subtile_loop
        else (
            f"        {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
        )
    )
    tma_store_split_tail_later_subtile_acquire = (
        ""
        if diagnose_later_c_acquire_before_barrier
        else (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
    )
    tma_store_split_tail_late_later_subtile_acquire = (
        (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_later_c_acquire_before_barrier
        else ""
    )
    # Pyrefly does not preserve the non-None tcgen05_value narrowing inside
    # the nested source formatter, so keep local string aliases for attributes
    # read only by that closure.
    tcgen05_epi_active = tcgen05_lifecycle.epi_active
    tcgen05_acc_pipeline = tcgen05_lifecycle.acc_pipeline
    tcgen05_acc_consumer_state = tcgen05_lifecycle.acc_consumer_state
    tcgen05_warp_idx = tcgen05_value.warp_idx
    tcgen05_tma_store_atom = tcgen05_value.tma_store_atom
    # Locals for the store-warp tail closure (Pyrefly drops the non-None
    # tcgen05_value narrowing inside nested source formatters; see above).
    tcgen05_role_local_tile_counter = tcgen05_value.role_local_tile_counter

    def tma_store_acc_t2r_region_body(
        *, acc_wait: str, allow_aux_chain: bool = False
    ) -> str:
        """Return the t2r/math/store-source region.

        The aux prelude is rendered inside ``body`` immediately after
        the TMEM→register copy and before ``acc.load()`` / fused math.
        Keeping residual and bias fragments out of the acquire/T2R
        prefix shortens their live ranges through the R2S store path;
        the long-scoreboard overlap from the older hoist was less
        valuable on the packed Target8 epilogue than eliminating the
        resulting local-memory spills.
        """
        assert allow_aux_chain or not aux_steps_in_chain, (
            "diagnostic / module-helper layouts reject aux-tensor chains at "
            "validate time; use allow_aux_chain=True only for the default TMA "
            "store body that threads the aux LDG through the main T2R body."
        )
        carrier = trs_racc
        store_target = trs_rd
        early_aux_prelude, late_prelude, rhs = _splice_acc_vec(
            carrier,
            "        ",
            safe_direct_aux_with_full_tile=partial_tma_needs_full_tile_guard,
        )
        # The secondary fan-out store reuses the still-live accumulator TMEM and
        # must not release it: the primary store already owns the accumulator
        # pipeline consumer release, and the one-shot teardown frees the TMEM
        # after every store has read it.
        acc_release = (
            ""
            if is_secondary_store
            else (
                f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
                f"            cute.arch.fence_view_async_tmem_load()\n"
                f"            with cute.arch.elect_one():\n"
                f"                {tcgen05_acc_pipeline}.consumer_release({tcgen05_acc_consumer_state})\n"
            )
        )
        return (
            f"{acc_wait}"
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
            f"{early_aux_prelude}"
            f"{late_prelude}"
            f"        {acc_vec} = {rhs}\n"
            f"{acc_release}"
            f"        {store_target}.store({acc_vec})\n"
        )

    def tma_store_tail_params(
        *, late_later_subtile_acquire: str
    ) -> Tcgen05TmaStoreTailParams:
        return Tcgen05TmaStoreTailParams(
            late_later_subtile_acquire=late_later_subtile_acquire,
            epilog_sync_barrier=epilog_sync_barrier,
            c_buffer=c_buffer,
            c_buffer_expr=tma_c_buffer_expr,
            c_stage_count=tcgen05_c_stage_count,
            tiled_copy_r2s=tiled_copy_r2s,
            trs_rd=trs_rd,
            trs_sd=trs_sd,
            warp_idx=tcgen05_warp_idx,
            tma_store_atom=tcgen05_tma_store_atom,
            bsg_sd=bsg_sd,
            bsg_gd=bsg_gd,
            c_pipeline=c_pipeline,
        )

    def tma_store_tail_region(*, late_later_subtile_acquire: str) -> str:
        if tcgen05_pure_matmul_object is not None:
            return tcgen05_pure_matmul_object.render_tma_store_tail_region(
                tma_store_tail_params(
                    late_later_subtile_acquire=late_later_subtile_acquire
                )
            )
        if has_store_warp:
            # Path B epi-warp tail (inside ``if epi_active:``): acquire the
            # C-store edge stage (wait until the store warp released it, i.e.
            # the prior TMA-D reading this physical C-ring slot completed),
            # barrier-1 (intra-epi convergence), R2S, fence, then a C-store-edge
            # PRODUCER commit in place of the second CTA barrier. The TMA-D +
            # ``c_pipeline`` lifecycle move to the store warp's tail
            # (``tma_store_store_warp_tail_region``). The epi warps drop
            # straight into the next subtile's T2R after committing — that is
            # the store/T2R overlap. The producer cooperative group is per-warp
            # (count ``epi_warp_count``), so ``producer_acquire`` is a full-warp
            # wait on every epi warp and ``producer_commit`` arrives once per
            # warp via ``elect_one``.
            return (
                f"        {c_store_edge}.producer_acquire({c_store_edge_producer_state})\n"
                f"        {epilog_sync_barrier}.arrive_and_wait()\n"
                f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
                f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
                f"        cute.arch.fence_view_async_shared()\n"
                f"        with cute.arch.elect_one():\n"
                f"            {c_store_edge}.producer_commit({c_store_edge_producer_state})\n"
                f"        {c_store_edge_producer_state}.advance()\n"
            )
        return (
            f"{late_later_subtile_acquire}"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
            f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
            f"        cute.arch.fence_view_async_shared()\n"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        if {tcgen05_warp_idx} == cutlass.Int32(0):\n"
            f"            cute.copy({tcgen05_tma_store_atom}, {bsg_sd}[(None, {c_buffer})], {bsg_gd}[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            f"            {c_pipeline}.producer_commit()\n"
        )

    def tma_store_store_warp_tail_region() -> str:
        # Path B store-warp tail (inside ``if store_warp_predicate:``): consume
        # the C-store edge, issue the TMA-D, and recycle the C-ring SMEM stage
        # with a ``c_stages - 1`` lagged release so a stage is only freed for
        # the epi producer AFTER its TMA-D read has provably completed.
        #
        # Ordering (per subtile, ``S`` = ``c_buffer``):
        #  1. ``consumer_wait``: the epi warps' R2S of stage ``S`` has landed.
        #  2. TMA-D ``S`` -> GMEM + ``c_pipeline.producer_commit`` (commit_group).
        #  3. ``c_pipeline.producer_acquire`` = ``cp_async_bulk_wait_group(
        #     c_stages - 1, read=True)``: after committing store i this drains
        #     every store except the ``c_stages - 1`` most recent, i.e. proves
        #     store ``i - (c_stages - 1)`` finished reading its SMEM stage.
        #  4. release that proven-drained stage (the ``release_state``, which
        #     lags the wait ``consumer_state`` by ``c_stages - 1``). Suppressed
        #     for the first ``c_stages - 1`` global subtiles (nothing drained
        #     yet); the trailing stages release naturally as later tiles' global
        #     subtile index advances, and the final unreleased stores drain via
        #     ``c_pipeline.producer_tail`` after the loop.
        lag = tcgen05_c_stage_count - 1
        global_subtile = (
            f"({tcgen05_role_local_tile_counter} * "
            f"cutlass.Int32({subtile_count}) + cutlass.Int32(_tcgen05_subtile))"
            if tcgen05_role_local_tile_counter
            else "cutlass.Int32(_tcgen05_subtile)"
        )
        return (
            f"        {c_store_edge}.consumer_wait({c_store_edge_consumer_state})\n"
            f"        {c_store_edge_consumer_state}.advance()\n"
            f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
            f"        cute.copy({tcgen05_tma_store_atom}, {bsg_sd}[(None, {c_buffer})], {bsg_gd}[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            f"        {c_pipeline}.producer_commit()\n"
            f"        {c_pipeline}.producer_acquire()\n"
            f"        if {global_subtile} >= cutlass.Int32({lag}):\n"
            f"            with cute.arch.elect_one():\n"
            f"                {c_store_edge}.consumer_release({c_store_edge_release_state})\n"
            f"            {c_store_edge_release_state}.advance()\n"
        )

    def tma_store_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
        late_later_subtile_acquire: str,
    ) -> str:
        # The aux LDG depends on ``_tcgen05_subtile`` and stays inside
        # the per-subtile T2R body. It intentionally runs after the
        # c_pipeline acquire and TMEM→register copy so the residual/bias
        # fragments are not live through the store-prefix waits.
        t2r_body = tma_store_acc_t2r_region_body(
            acc_wait=acc_wait,
            allow_aux_chain=True,
        )
        if has_store_warp:
            # Path B: the epi warps own T2R/R2S + the C-store producer commit;
            # the store warp (a SEPARATE ``if``, NOT under ``epi_active``) owns
            # the TMA-D + ``c_pipeline`` commit/acquire (its ``cp_async_bulk
            # _wait_group`` reuse guard) + the lagged edge release. The C-ring
            # acquire/commit move WHOLLY onto the store warp (PipelineTmaStore
            # is per-warp commit-group state), so the epi warps never touch
            # ``c_pipeline``; their store-prefix acquire lines are dropped.
            return (
                f"    if {tcgen05_epi_active}:\n"
                f"{t2r_body}"
                f"{tma_store_tail_region(late_later_subtile_acquire='')}"
                f"    if {store_warp_predicate}:\n"
                f"{tma_store_store_warp_tail_region()}"
            )
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{t2r_body}"
            f"{tma_store_tail_region(late_later_subtile_acquire=late_later_subtile_acquire)}"
        )

    def indented_diagnostic_region(source: str) -> str:
        if not source:
            return "            pass\n"
        return "".join(f"    {line}" for line in source.splitlines(keepends=True))

    def tma_store_helper_boundary_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
        late_later_subtile_acquire: str,
    ) -> str:
        acquire_region = f"{first_subtile_acquire}{later_subtile_acquire}"
        acc_region = tma_store_acc_t2r_region_body(acc_wait=acc_wait)
        tail_region = tma_store_tail_region(
            late_later_subtile_acquire=late_later_subtile_acquire
        )
        # These constant-true blocks are diagnostic source boundaries. The
        # generated-code AST round trip preserves them, while emitted comments
        # are not reliable line-info anchors.
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"        if True:\n"
            f"{indented_diagnostic_region(acquire_region)}"
            f"        if True:\n"
            f"{indented_diagnostic_region(acc_region)}"
            f"        if True:\n"
            f"{indented_diagnostic_region(tail_region)}"
        )

    module_acc_t2r_helper_name = (
        df.unique_name("tcgen05_acc_t2r_region")
        if diagnose_module_helper_acc_t2r
        else ""
    )
    module_store_tail_helper_name = (
        df.unique_name("tcgen05_store_tail_region")
        if diagnose_module_helper_store_tail
        else ""
    )

    def tma_store_module_acc_t2r_helper_source(*, acc_wait: str) -> str:
        # Aux-tensor chains are rejected for the diagnostic module-helper
        # layouts (see the ``BackendUnsupported`` raise above), so
        # ``module_early_aux`` is always empty here. Concatenating it
        # with ``module_late_prelude`` preserves the prior flat-prelude
        # source order for unary chains and identity stores in this
        # diagnostic layout.
        module_early_aux, module_late_prelude, rhs = _splice_acc_vec(
            "tcgen05_tRS_rAcc", "    "
        )
        prelude = module_early_aux + module_late_prelude
        return (
            "@cute.jit\n"
            f"def {module_acc_t2r_helper_name}("
            "_tcgen05_subtile, "
            "tcgen05_acc_pipeline, "
            "tcgen05_acc_consumer_state, "
            "tcgen05_tTR_tAcc, "
            "tcgen05_tiled_copy_t2r, "
            "tcgen05_tTR_rAcc, "
            "tcgen05_tRS_rAcc, "
            "tcgen05_tRS_rD, "
            "tcgen05_subtile_count"
            "):\n"
            f"{acc_wait}"
            "    tcgen05_tTR_tAcc_mn = tcgen05_tTR_tAcc[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            "    cute.copy(tcgen05_tiled_copy_t2r, tcgen05_tTR_tAcc_mn, tcgen05_tTR_rAcc)\n"
            f"{prelude}"
            f"    tcgen05_acc_vec = {rhs}\n"
            "    if _tcgen05_subtile == tcgen05_subtile_count - 1:\n"
            "        cute.arch.fence_view_async_tmem_load()\n"
            "        with cute.arch.elect_one():\n"
            "            tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)\n"
            "    tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        )

    def tma_store_module_acc_t2r_helper_call() -> str:
        return (
            f"        {module_acc_t2r_helper_name}("
            f"_tcgen05_subtile, "
            f"{tcgen05_acc_pipeline}, "
            f"{tcgen05_acc_consumer_state}, "
            f"{ttr_tacc}, "
            f"{tiled_copy_t2r}, "
            f"{ttr_racc}, "
            f"{trs_racc}, "
            f"{trs_rd}, "
            f"{subtile_count})\n"
        )

    def tma_store_module_helper_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        late_later_subtile_acquire: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_module_acc_t2r_helper_call()}"
            f"{tma_store_tail_region(late_later_subtile_acquire=late_later_subtile_acquire)}"
        )

    def tma_store_module_tail_helper_source(*, late_later_subtile_acquire: str) -> str:
        return (
            "@cute.jit\n"
            f"def {module_store_tail_helper_name}("
            "_tcgen05_subtile, "
            "tcgen05_tma_c_buffer_index, "
            "tcgen05_epilog_sync_barrier, "
            "tcgen05_tiled_copy_r2s, "
            "tcgen05_tRS_rD, "
            "tcgen05_tRS_sD, "
            "tcgen05_tma_store_atom, "
            "tcgen05_bSG_sD, "
            "tcgen05_bSG_gD, "
            "tcgen05_c_pipeline, "
            "tcgen05_warp_idx"
            "):\n"
            f"{late_later_subtile_acquire}"
            "    tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            f"    tcgen05_c_buffer = tcgen05_tma_c_buffer_index % cutlass.Int32({tcgen05_c_stage_count})\n"
            "    cute.copy(tcgen05_tiled_copy_r2s, tcgen05_tRS_rD, tcgen05_tRS_sD[(None, None, None, tcgen05_c_buffer)])\n"
            "    cute.arch.fence_view_async_shared()\n"
            "    tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            "    if tcgen05_warp_idx == cutlass.Int32(0):\n"
            "        cute.copy(tcgen05_tma_store_atom, tcgen05_bSG_sD[(None, tcgen05_c_buffer)], tcgen05_bSG_gD[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            "        tcgen05_c_pipeline.producer_commit()"
        )

    def tma_store_module_tail_helper_call() -> str:
        return (
            f"        {module_store_tail_helper_name}("
            f"_tcgen05_subtile, "
            f"{tma_c_buffer_expr}, "
            f"{epilog_sync_barrier}, "
            f"{tiled_copy_r2s}, "
            f"{trs_rd}, "
            f"{trs_sd}, "
            f"{tcgen05_tma_store_atom}, "
            f"{bsg_sd}, "
            f"{bsg_gd}, "
            f"{c_pipeline}, "
            f"{tcgen05_warp_idx})\n"
        )

    def tma_store_module_tail_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_acc_t2r_region_body(acc_wait=acc_wait)}"
            f"{tma_store_module_tail_helper_call()}"
        )

    if diagnose_split_first_t2r:
        tma_store_split_first_subtile_body = tma_store_subtile_body(
            first_subtile_acquire=tma_store_split_first_subtile_acquire,
            later_subtile_acquire="",
            acc_wait=tma_store_split_first_acc_wait,
            late_later_subtile_acquire="",
        )
        tma_store_split_tail_subtile_body = tma_store_subtile_body(
            first_subtile_acquire="",
            later_subtile_acquire=tma_store_split_tail_later_subtile_acquire,
            acc_wait="",
            late_later_subtile_acquire=(
                tma_store_split_tail_late_later_subtile_acquire
            ),
        )
        # Diagnostic-only scaffolding: reuse the one-indent subtile formatter
        # for a static first subtile without changing production source layout.
        # The tail loop maps split-loop indices back to logical subtile ids 1..N-1;
        # unroll_full=True keeps those subtile values compile-time constants.
        tma_store_subtile_loop = (
            "if True:\n"
            f"    _tcgen05_subtile = 0\n"
            f"{tma_store_split_first_subtile_body}"
            f"for _tcgen05_split_subtile in cutlass.range({subtile_count} - 1, unroll_full=True):\n"
            f"    _tcgen05_subtile = _tcgen05_split_subtile + 1\n"
            f"{tma_store_split_tail_subtile_body}"
        )
    elif diagnose_split_acc_t2r_store_tail:
        tma_store_helper_boundary_body = tma_store_helper_boundary_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_helper_boundary_body}"
        )
    elif diagnose_module_helper_acc_t2r:
        module_helper_acc_wait = (
            ""
            if diagnose_acc_wait_before_subtile_loop
            else (
                "    if _tcgen05_subtile == 0:\n"
                "        tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)\n"
            )
        )
        state.codegen.module_statements.append(
            statement_from_string(
                tma_store_module_acc_t2r_helper_source(acc_wait=module_helper_acc_wait)
            )
        )
        tma_store_module_helper_body = tma_store_module_helper_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_module_helper_body}"
        )
    elif diagnose_module_helper_store_tail:
        module_tail_late_later_subtile_acquire = (
            (
                "    if _tcgen05_subtile != 0 and "
                "tcgen05_warp_idx == cutlass.Int32(0):\n"
                "        tcgen05_c_pipeline.producer_acquire()\n"
            )
            if diagnose_later_c_acquire_before_barrier
            else ""
        )
        state.codegen.module_statements.append(
            statement_from_string(
                tma_store_module_tail_helper_source(
                    late_later_subtile_acquire=module_tail_late_later_subtile_acquire
                )
            )
        )
        tma_store_module_tail_body = tma_store_module_tail_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_module_tail_body}"
        )
    else:
        tma_store_default_subtile_body = tma_store_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_default_subtile_body}"
        )
    tma_store_smem_setup = [
        # Must match the wrapper-side `tcgen05_d_tma` TMA atom layout in
        # `helion/runtime/__init__.py`; both describe one D SMEM stage.
        (
            f"{smem_d_layout} = cutlass.utils.blackwell_helpers.make_smem_layout_epi("
            f"{target_dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
            f"{epi_tile}, {tcgen05_value.c_stage_count})"
        ),
        (
            f"{smem_d_ptr} = cute.arch.alloc_smem("
            f"{target_dtype}, cute.cosize({smem_d_layout}.outer), alignment=1024)"
        ),
        (
            f"{smem_d} = cute.make_tensor("
            f"cute.recast_ptr({smem_d_ptr}, {smem_d_layout}.inner, dtype={target_dtype}), "
            f"{smem_d_layout}.outer)"
        ),
        *_rowvec_aux_smem_setup_lines(),
    ]
    tma_store_acc_layout_setup = [
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
    ]
    tma_store_role_invariant_setup = [
        *tma_static_store_setup,
        *tma_store_smem_setup,
        *tma_store_acc_layout_setup,
    ]
    suppressed_store_body_core = [
        (
            # Diagnostic-only invalid-output mode. Keep the accumulator
            # pipeline draining so persistent kernels do not deadlock, but
            # suppress C-pipeline acquire/commit, R2S/SMEM work, and TMA D
            # stores to bound whether hot waits are tied to the C-store path.
            f"if {tcgen05_lifecycle.epi_active}:\n"
            f"    {tcgen05_lifecycle.acc_pipeline}.consumer_wait({tcgen05_lifecycle.acc_consumer_state})\n"
            f"    with cute.arch.elect_one():\n"
            f"        {tcgen05_lifecycle.acc_pipeline}.consumer_release({tcgen05_lifecycle.acc_consumer_state})\n"
            + emit_pipeline_advance(
                tcgen05_lifecycle.acc_consumer_state,
                indent="    ",
            )
        )
    ]
    # C-input warp aux pipeline consumer-wait + lane-0-gated
    # consumer-release framing (``cute_plan.md`` §7.5.3.2 cycle 2b).
    # Gate-closed configs (default ``c_input_warps=0`` or no aux
    # residual) keep the historical GMEM aux path. When the gate
    # fires, the wait/release pair runs once per *subtile* of the
    # per-output-tile aux region: per-subtile staging keeps the
    # SMEM ring footprint at one ``epi_tile`` chunk per stage
    # rather than one ``(bm, bn)`` chunk, which is essential to
    # fit cluster_m=2 + ``tcgen05_ab_stages=3`` in the 228 KB
    # B200 SMEM cap. The wait begins the aux-load block emitted by
    # ``_aux_subtile_load_source`` (before any ``.load()`` from the
    # SMEM ring); the default TMA-store path now splices that block
    # after the c_pipeline acquire and T2R copy to keep aux fragments
    # out of the store-prefix live range. The release + ``advance``
    # happen at the bottom of the same per-subtile iteration (after
    # the chain has consumed ``aux_loaded``). Lane-0 gating mirrors
    # the per-warp consumer arrive count
    # (``epi_warp_count``) allocated on the aux pipeline.

    # Static-full role-local stores have no dynamic full-tile branch, so all
    # C-store invariant setup can be hoisted once. Scheduler-backed hybrid
    # output-edge stores split full and fringe tiles into separate role-local
    # scheduler phases, which gives the full-tile phase the same hoist shape.
    # The monolithic hybrid path still keeps descriptor/SMEM layout setup
    # inside its dynamic full-tile branch.
    split_hybrid_tma_store_role = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and tcgen05_value.tma_store_full_tiles_only
        and aux_matmul_plan is not None
        and aux_matmul_plan.has_scheduler_warp
        # CLC publishes a single hardware-scheduled stream today. The
        # full/edge split below requires the scheduler warp to publish two
        # static streams with a sentinel between them.
        and not aux_matmul_plan.is_clc_persistent
        and not diagnose_skip_epilogue_store
    )
    hoist_tma_store_resources = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and (not tcgen05_value.tma_store_full_tiles_only or split_hybrid_tma_store_role)
        and not diagnose_skip_epilogue_store
    )
    hoist_hybrid_tma_store_pipeline = (
        tcgen05_value.use_role_local_epi
        and tcgen05_value.use_tma_store_epilogue
        and tcgen05_value.tma_store_full_tiles_only
        and not split_hybrid_tma_store_role
        and not diagnose_skip_epilogue_store
    )
    tma_store_body_setup_core = [
        *(tma_static_store_setup if not hoist_tma_store_resources else []),
        *(
            tma_store_pipeline_setup
            if not (hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline)
            else []
        ),
        *(tma_store_smem_setup if not hoist_tma_store_resources else []),
        *_rowvec_aux_copy_lines(),
        *tma_store_first_subtile_acquire,
        *tma_tile_store_setup,
        (
            f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgc_base})"
        ),
        (
            f"{tcgc_planned} = cute.make_tensor("
            f"{tcgc}.iterator, "
            f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}))"
        ),
        *(tma_store_acc_layout_setup if not hoist_tma_store_resources else []),
        (
            f"{tiled_copy_t2r}, {ttr_tacc_base}, {ttr_racc} = "
            "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition("
            f"{kernel_desc}, {tcgen05_value.epi_tidx}, {tacc}, {tcgc_planned}, {epi_tile}, {tcgen05_lifecycle.is_two_cta!s})"
        ),
        (f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})"),
        (
            f"{tiled_copy_r2s}, {trs_rd}, {trs_sd} = "
            "cutlass.utils.gemm.sm100.epilogue_smem_copy_and_partition("
            f"{kernel_desc}, {tiled_copy_t2r}, {ttr_rd}, "
            f"{tcgen05_value.epi_tidx}, {smem_d})"
        ),
        f"{trs_racc} = {tiled_copy_r2s}.retile({ttr_racc})",
        f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
        # Per-aux-step partitioning lines (one chain per auxiliary
        # tensor). No-op when the chain has no aux steps; the TMA
        # path requires an explicit ``thr_copy_t2r`` slice because
        # (unlike the SIMT path) the TMA path does not otherwise
        # create one — the t2r partition is consumed directly by
        # the SMEM-staged store, never via partition_D. The aux
        # load needs partition_D to compute a per-thread GMEM read
        # for the auxiliary tile so we create the slice here.
        # When the C-input warp productive-body gate is open the
        # source switches from per-tile GMEM to the per-subtile
        # SMEM ring stage (see ``_aux_tile_setup_lines`` SMEM
        # branch); the partition pipeline is layout-only and
        # compiles unchanged, and the per-subtile ``consumer_wait``
        # / lane-0-gated ``consumer_release`` are emitted by
        # ``_aux_subtile_load_source`` inside the per-subtile loop.
        *_aux_tile_setup_lines(
            thr_copy_t2r_var=thr_copy_t2r,
            define_thr_copy_t2r=True,
            retile_for_r2s=True,
        ),
        (
            f"{bsg_sd}, {bsg_gd_partitioned} = cute.nvgpu.cpasync.tma_partition("
            f"{tcgen05_value.tma_store_atom}, 0, cute.make_layout(1), "
            f"cute.group_modes({smem_d}, 0, 2), "
            f"cute.group_modes({tcgc_epi}, 0, 2))"
        ),
        (
            f"{bsg_gd} = {bsg_gd_partitioned}["
            f"(None, None, None, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))]"
        ),
        f"{bsg_gd} = cute.group_modes({bsg_gd}, 1, cute.rank({bsg_gd}))",
        (
            f"{ttr_tacc_stage} = {ttr_tacc_base}["
            f"(None, None, None, None, None, {tcgen05_acc_stage_index_expr})]"
        ),
        f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
        f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
        *tma_store_pre_loop_acc_wait,
    ]
    # Warp 0 pre-acquires the first TMA-store SMEM stage before per-tile
    # C-store setup. The subtile loop acquires only later stages, so C-stage
    # waits can overlap setup, the first acc-pipeline wait, and the other epi
    # warps' TMEM load/conversion work on later subtile iterations. Most
    # alternate placements are diagnostics, but the edge+K-tail production seed
    # uses the measured first_in_loop / before_subtile_loop pair.
    # tcgen05_c_acquire_placement=first_in_loop moves only that first acquire
    # into the subtile loop; later acquires and the accumulator wait keep their
    # default order. The diagnostic later_before_barrier placement keeps the
    # first acquire in production position and moves only later-subtile
    # acquires just before the first epilogue barrier.
    # tcgen05_acc_wait_placement=before_subtile_loop keeps both C acquire sites
    # in production position and moves only the accumulator consumer wait
    # before the subtile loop. A CTA-scoped named barrier ensures all epi warps
    # have observed warp 0's acquire before they write SMEM; a second barrier
    # ensures the SMEM writes and Quack-style async-shared fence are visible
    # before warp 0 issues and commits the TMA operation. Compute the SMEM ring
    # index after the first barrier so the acquire/barrier/index order stays
    # aligned with Quack's TMA-store epilogue.
    # The accumulator consumer state advances after the loop, matching Quack's
    # call-site ordering while preserving the early release. After warp 0
    # commits the TMA store, the next subtile's producer_acquire plus the first
    # named barrier are enough to keep all epi warps from writing a reused SMEM
    # stage too early. Avoiding a post-commit barrier matches Quack's epilogue
    # loop. The split_first_t2r diagnostic emits the first static subtile as a
    # standalone source block, then loops over later subtile work. It is a
    # layout discriminator for the hot acc-wait/T2R SASS row; the default
    # production source shape remains the single loop.
    # Advance is a per-thread local state update, so it intentionally stays
    # outside elect_one; only the mbarrier release is elected.
    tma_store_pipeline_tail_lines = (
        [tma_store_pipeline_tail]
        if not (hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline)
        else []
    )
    if tcgen05_pure_matmul_object is not None:
        tma_store_body_core = tcgen05_pure_matmul_object.build_tma_store_body_core(
            Tcgen05TmaStoreBodyCoreParams(
                setup_lines=tma_store_body_setup_core,
                subtile_loop=Tcgen05TmaStoreSubtileLoopParams(
                    subtile_count=subtile_count,
                    epi_active=tcgen05_epi_active,
                    first_subtile_acquire=tma_store_loop_first_subtile_acquire,
                    later_subtile_acquire=tma_store_loop_later_subtile_acquire,
                    acc_t2r_region_body=tma_store_acc_t2r_region_body(
                        acc_wait=tma_store_loop_acc_wait,
                        allow_aux_chain=True,
                    ),
                    tail=tma_store_tail_params(
                        late_later_subtile_acquire=(
                            tma_store_loop_late_later_subtile_acquire
                        ),
                    ),
                ),
                pipeline_tail_lines=tma_store_pipeline_tail_lines,
            )
        )
    else:
        # The secondary fan-out store does not own the accumulator consumer
        # state, so it must not advance it (the primary store advances once).
        tma_store_acc_advance = (
            ""
            if is_secondary_store
            else (
                f"if {tcgen05_lifecycle.epi_active}:\n"
                + emit_pipeline_advance(
                    tcgen05_lifecycle.acc_consumer_state,
                    indent="    ",
                )
            )
        )
        tma_store_body_core = [
            *tma_store_body_setup_core,
            tma_store_subtile_loop + tma_store_acc_advance,
            *tma_store_pipeline_tail_lines,
        ]
    tma_store_full_tile_body_core = list(tma_store_body_core)
    if (
        tcgen05_value.tma_store_full_tiles_only
        and tcgen05_value.role_local_tile_counter
    ):
        tma_store_full_tile_body_core.append(
            f"{tcgen05_value.role_local_tile_counter} = "
            f"{tcgen05_value.role_local_tile_counter} + cutlass.Int32(1)"
        )
    tma_store_body_source = "\n".join(tma_store_full_tile_body_core)
    simt_store_body_source = "\n".join(simt_store_body_core)
    hybrid_tma_store_body_core = [
        f"{full_tile} = {full_tile_expr}",
        (
            f"if {full_tile}:\n"
            f"{textwrap.indent(tma_store_body_source, '    ')}\n"
            "else:\n"
            f"{textwrap.indent(simt_store_body_source, '    ')}"
        ),
    ]
    if diagnose_skip_epilogue_store:
        store_body_core = suppressed_store_body_core
    elif tcgen05_value.tma_store_full_tiles_only:
        store_body_core = hybrid_tma_store_body_core
    elif tcgen05_value.use_tma_store_epilogue:
        store_body_core = tma_store_body_core
    else:
        store_body_core = simt_store_body_core
    main_stmts: list[ast.AST]
    if tcgen05_value.use_role_local_epi:
        # These setup statements intentionally remain virtual-pid-independent.
        # The persistent splitter hoists pipeline state before the role-local
        # scheduler loops. Scheduler-backed hybrid stores keep descriptor and
        # layout Python objects inside the epilogue role prelude so they do
        # not leak across unrelated dynamic warp-role ``if`` regions.
        tma_store_pipeline_hoisted_stmts = (
            [statement_from_string(line) for line in tma_store_pipeline_setup]
            if (hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline)
            else []
        )
        tma_store_role_invariant_stmts = (
            [statement_from_string(line) for line in tma_store_role_invariant_setup]
            if hoist_tma_store_resources
            else []
        )
        if split_hybrid_tma_store_role:
            tma_store_hoisted_stmts = tma_store_pipeline_hoisted_stmts
        elif hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline:
            tma_store_hoisted_stmts = [
                *tma_store_pipeline_hoisted_stmts,
                *tma_store_role_invariant_stmts,
            ]
        else:
            tma_store_hoisted_stmts = []
        if tcgen05_pure_matmul_object is not None:
            assert not split_hybrid_tma_store_role, (
                "pure lifecycle is admitted only for static-full pure matmul"
            )
            assert not hoist_hybrid_tma_store_pipeline, (
                "pure lifecycle does not use hybrid edge TMA-store pipeline setup"
            )
            main_stmts = tcgen05_pure_matmul_object.emit_store_role_stmts(
                df.cute_state,
                tma_store_hoisted_stmts=tma_store_hoisted_stmts,
                store_body_core=store_body_core,
            )
        elif split_hybrid_tma_store_role:
            sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
            sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
            full_main_stmt = statement_from_string(
                "if True:\n"
                + textwrap.indent("\n".join(tma_store_full_tile_body_core), "    ")
            )
            edge_main_stmt = statement_from_string(
                "if True:\n" + textwrap.indent("\n".join(simt_store_body_core), "    ")
            )
            df.cute_state.register_tcgen05_per_tile_stmts(
                [sync_before_stmt, full_main_stmt, edge_main_stmt, sync_after_stmt]
            )
            df.cute_state.register_tcgen05_epi_role_full_edge_stmts(
                full_tile_stmts=[full_main_stmt],
                edge_tile_stmts=[edge_main_stmt],
            )
            # `cute.arch.alloc_smem` is a CuTe DSL static allocation even
            # though it is represented as a statement. Keeping the descriptor,
            # layout, and allocation statements in the epi-role prelude scopes
            # CuTe Python objects away from unrelated warp-role branches
            # without making the shared-memory reservation data-dependent on
            # the runtime epi-warp predicate.
            df.cute_state.register_tcgen05_epi_role_prelude_stmts(
                tma_store_role_invariant_stmts
            )
            main_stmts = [
                *tcgen05_acc_stage_index_top_level_stmts,
                *tma_store_hoisted_stmts,
                *tma_store_role_invariant_stmts,
                sync_before_stmt,
                full_main_stmt,
                edge_main_stmt,
                sync_after_stmt,
            ]
        else:
            sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
            sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
            main_stmt = statement_from_string(
                "if True:\n" + textwrap.indent("\n".join(store_body_core), "    ")
            )
            df.cute_state.register_tcgen05_per_tile_stmts(
                [sync_before_stmt, main_stmt, sync_after_stmt]
            )
            df.cute_state.register_tcgen05_epi_role_stmts([main_stmt])
            main_stmts = [
                *tcgen05_acc_stage_index_top_level_stmts,
                *tma_store_hoisted_stmts,
                sync_before_stmt,
                main_stmt,
                sync_after_stmt,
            ]
    else:
        store_body = [
            "cute.arch.sync_threads()",
            *store_body_core,
            "cute.arch.sync_threads()",
        ]
        main_stmt = statement_from_string(
            "if True:\n" + textwrap.indent("\n".join(store_body), "    ")
        )
        main_stmts = [*tcgen05_acc_stage_index_top_level_stmts, main_stmt]
    # Pipeline drain + TMEM dealloc are one-shot cleanup. They must run
    # AFTER all tiles have been processed (in the persistent path) and
    # naturally land at the end of the kernel in the non-persistent path.
    # Keep them as separate statements so the persistent splitter can
    # extract them via the post-loop registration below.
    tma_store_post_loop_tail = ""
    if hoist_tma_store_resources or hoist_hybrid_tma_store_pipeline:
        # Role-local persistent epilogues reuse the C-store pipeline across
        # scheduler-recycled work tiles. Draining it inside each tile would
        # serialize the next tile's epilogue against this tile's TMA stores.
        # The tail must run before TMEM dealloc setup below.
        tma_store_post_loop_tail = tma_store_pipeline_tail
    if is_secondary_store:
        # The matmul drain + TMEM-free teardown is one-shot and owned by the
        # primary store; the secondary fan-out store emits only its store body.
        post_loop_stmts = []
    elif tcgen05_pure_matmul_object is not None:
        post_loop_stmts = tcgen05_pure_matmul_object.emit_store_post_loop_stmts(
            df.cute_state,
            candidate_names,
            tma_store_pipeline_tail=tma_store_post_loop_tail,
        )
    else:
        post_loop_lines = tcgen05_lifecycle.render_store_post_loop_lines(
            tma_store_pipeline_tail=tma_store_post_loop_tail
        )
        post_loop_stmts = [statement_from_string(line) for line in post_loop_lines]
        df.cute_state.register_tcgen05_post_loop_stmts(post_loop_stmts)
    return [*main_stmts, *post_loop_stmts]


def _codegen_cute_store_loaded_index_trailing_slices(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from .._compiler.ast_extension import create

    if value_node.target is not load or len(value_node.args) < 2:
        return None
    source_tensor_node = value_node.args[0]
    if not isinstance(source_tensor_node, torch.fx.Node):
        return None
    source_tensor = source_tensor_node.meta.get("val")
    if not isinstance(source_tensor, torch.Tensor):
        return None
    source_subscript = value_node.args[1]
    if not isinstance(source_subscript, (list, tuple)) or not source_subscript:
        return None
    indexer = source_subscript[0]
    if not isinstance(indexer, torch.fx.Node):
        return None
    indexer_value = indexer.meta.get("val")
    if not isinstance(indexer_value, torch.Tensor) or indexer_value.ndim == 0:
        return None
    trailing_source = [*source_subscript[1:]]
    if not trailing_source or not all(idx == slice(None) for idx in trailing_source):
        return None
    if len(subscript) != indexer_value.ndim + len(trailing_source):
        return None
    trailing_store = subscript[indexer_value.ndim :]
    if not all(idx == slice(None) for idx in trailing_store):
        return None

    ast_source_subscript = list(
        map_arg(tuple(source_subscript), lambda arg: state.env[arg])
    )
    index_exprs = _cute_index_exprs(
        state,
        [indexer_value],
        [ast_source_subscript[0]],
        tensor=source_tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != 1:
        return None

    prefix_subscript = [*subscript[: indexer_value.ndim]]
    prefix_ast_subscript = [*ast_subscript[: indexer_value.ndim]]
    target_prefix = _cute_index_exprs(
        state,
        prefix_subscript,
        prefix_ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(target_prefix) != indexer_value.ndim:
        return None

    env = CompileEnvironment.current()
    index_dtype = env.backend.dtype_str(env.index_dtype)
    source_loop_vars = [
        state.device_function.new_var("slice_idx", dce=True) for _ in trailing_source
    ]
    source_indices = [
        index_exprs[0],
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    target_indices = [
        *target_prefix,
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    if len(source_indices) != source_tensor.ndim or len(target_indices) != tensor.ndim:
        return None

    source_name = state.device_function.tensor_arg(source_tensor).name
    target_name = state.device_function.tensor_arg(tensor).name
    source_dtype = env.backend.dtype_str(source_tensor.dtype)
    target_dtype = env.backend.dtype_str(tensor.dtype)
    source_mask = _cute_combined_mask(
        state,
        [indexer_value],
        None,
        tensor=source_tensor,
    )
    target_mask = _cute_combined_mask(
        state,
        prefix_subscript,
        extra_mask,
        tensor=tensor,
    )
    masks = [mask for mask in (source_mask, target_mask) if mask is not None]
    mask_expr = " and ".join(f"({mask})" for mask in masks) if masks else None
    load_expr = f"{source_name}[{', '.join(source_indices)}]"
    if mask_expr is not None:
        load_expr = f"({load_expr} if {mask_expr} else {source_dtype}(0))"
    store_expr = (
        f"{target_name}.__setitem__({_cute_index_tuple(target_indices)}, "
        f"{env.backend.ast_to_dtype_expr(load_expr, target_dtype)})"
    )
    if mask_expr is not None:
        store_expr = f"{store_expr} if {mask_expr} else None"

    tensor_dim = 0
    for idx in prefix_subscript:
        block_id = None
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif idx == slice(None) and tensor_dim < tensor.ndim:
            block_id = next(
                (
                    candidate
                    for candidate in _matching_block_ids(env, tensor.shape[tensor_dim])
                    if candidate in state.codegen.active_device_loops
                ),
                None,
            )
        tensor_dim += 1
        if block_id is None:
            continue
        axis = None
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            axis = grid_state.block_thread_axes.get(block_id)
        if axis is None:
            loops = state.codegen.active_device_loops.get(block_id)
            if loops:
                axis = loops[-1].block_thread_axes.get(block_id)
        if axis is None or not (0 <= axis < 3):
            continue
        block_size = state.device_function.resolved_block_size(block_id)
        if not isinstance(block_size, int):
            continue
        state.codegen.max_thread_block_dims[axis] = max(
            state.codegen.max_thread_block_dims[axis],
            block_size,
        )
        state.codegen.referenced_thread_block_dims[axis] = max(
            state.codegen.referenced_thread_block_dims[axis],
            block_size,
        )

    stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    for loop_var, source_pos in reversed(
        [*zip(source_loop_vars, range(1, len(source_subscript)), strict=True)]
    ):
        extent = _cute_tensor_dim_size_expr(state, source_tensor, source_pos)
        stmt = create(
            ast.For,
            target=create(ast.Name, id=loop_var, ctx=ast.Store()),
            iter=expr_from_string(f"range({extent})"),
            body=[stmt],
            orelse=[],
            type_comment=None,
        )
    state.add_statement(stmt)
    return ast.Constant(value=None)


def _cute_expand_broadcast_dim(value_node: torch.fx.Node) -> int | None:
    """Return the dim an ``aten.expand`` broadcasts (input size 1 -> >1).

    Returns ``None`` unless ``value_node`` is an ``aten.expand`` whose value has
    exactly one broadcast dimension — i.e. the expanded value carries a stride-0
    mode at exactly one position whose pre-expand extent was 1. This is the
    signal that the stored value replicates one source element across that dim.
    """
    if value_node.target is not torch.ops.aten.expand.default:
        return None
    input_arg = value_node.args[0]
    if not isinstance(input_arg, torch.fx.Node):
        return None
    out_val = value_node.meta.get("val")
    in_val = input_arg.meta.get("val")
    if not isinstance(out_val, torch.Tensor) or not isinstance(in_val, torch.Tensor):
        return None
    if out_val.ndim != in_val.ndim:
        return None
    env = CompileEnvironment.current()
    broadcast_dims = [
        dim
        for dim in range(out_val.ndim)
        if env.known_equal(in_val.shape[dim], 1)
        and not env.known_equal(out_val.shape[dim], 1)
        and out_val.stride(dim) == 0
    ]
    if len(broadcast_dims) != 1:
        return None
    return broadcast_dims[0]


def _cute_block_tile_begin_expr(state: CodegenState, block_id: int) -> str | None:
    """Return the *per-block* tile start for a tile mapped onto a thread axis.

    In the CuTe SIMT model a tile dimension is spread across a thread axis, so
    the strategy's ``index_var`` is the per-*thread* global index
    (``pid * block + thread_idx[axis]``). Subtracting the thread-local coordinate
    yields the per-*block* tile base (``pid * block``), shared by every thread in
    the tile — the correct anchor for a broadcast lane loop. Returns ``None`` when
    the block id has no active thread axis in this scope.
    """
    from .._compiler.cute.cute_reshape import _grid_local_coord_expr

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    loop_state = loops[-1]
    thread_axis = loop_state.block_thread_axes.get(block_id)
    global_index = loop_state.strategy.index_var(block_id)
    if thread_axis is None or global_index is None:
        return None
    local_coord = _grid_local_coord_expr(state.codegen, block_id, thread_axis)
    return state.codegen.lift(
        expr_from_string(f"({global_index}) - ({local_coord})"),
        dce=True,
        prefix="tile_begin",
    ).id


def _cute_unsqueeze_expand_load_source(
    value_node: torch.fx.Node, broadcast_dim: int
) -> torch.fx.Node | None:
    """Return the ``hl.load`` feeding ``expand(val[..., None, ...])``.

    Walks ``value_node`` (an ``aten.expand``) back through a single
    unsqueeze-style subscript op (``val[:, None, :]`` inserting the broadcast dim)
    to the originating ``hl.load``. Returns ``None`` unless the chain is exactly
    that shape, so the caller falls back to the load-agnostic path.
    """
    from .view_ops import subscript as subscript_op

    inner = value_node.args[0]
    if not isinstance(inner, torch.fx.Node):
        return None
    if inner.op == "call_function" and inner.target is subscript_op:
        index_arg = inner.args[1] if len(inner.args) > 1 else None
        if not isinstance(index_arg, (list, tuple)):
            return None
        # Exactly one ``None`` (the inserted broadcast dim) at ``broadcast_dim``.
        none_positions = [pos for pos, entry in enumerate(index_arg) if entry is None]
        if none_positions != [broadcast_dim]:
            return None
        load_node = inner.args[0]
    else:
        load_node = inner
    if (
        isinstance(load_node, torch.fx.Node)
        and load_node.op == "call_function"
        and load_node.target is load
        and len(load_node.args) >= 2
    ):
        return load_node
    return None


def _codegen_cute_store_expand_broadcast_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    """Lower a store whose value is broadcast across a reused tile dimension.

    Handles the pattern::

        val = hl.load(src, [tile, hl.arange(k)])  # (block, k)
        val_3d = val[:, None, :].expand(block, block, k)  # stride-0 middle dim
        hl.store(out, [idx[tile], tile.index, hl.arange(k)], val_3d)

    Here ``tile`` appears twice in the store index — once as a tensor indexer
    (``idx[tile]``) and once as the bare tile index (``tile.index``) — while the
    value is broadcast (stride 0) along the second (``tile.index``) position. The
    generic SIMT store lowers both positions onto ``tile``'s single thread axis,
    so each thread only writes the ``a == b`` diagonal of the ``(block, block)``
    block. Instead emit a sequential lane loop over the broadcast position so a
    thread holding ``val[a]`` writes the full ``out[idx[a], begin+b, :]`` row for
    every ``b`` in the tile, filling the block. ``val`` is broadcast, so every
    lane reads the same per-thread register.

    Returns ``None`` (a strict no-op) unless every gate matches, so existing
    kernels are byte-for-byte unchanged.
    """
    env = CompileEnvironment.current()
    broadcast_dim = _cute_expand_broadcast_dim(value_node)
    if broadcast_dim is None:
        return None
    if broadcast_dim >= len(subscript):
        return None
    broadcast_idx = subscript[broadcast_dim]
    # The broadcast position must be a bare tile index (a SymInt block id), and
    # that same block id must be reused by another (tensor) index position — the
    # collision the generic path mis-handles.
    if not isinstance(broadcast_idx, torch.SymInt):
        return None
    broadcast_block_id = env.get_block_id(broadcast_idx)
    if broadcast_block_id is None:
        return None
    block_size = state.device_function.resolved_block_size(broadcast_block_id)
    if not isinstance(block_size, int) or block_size <= 1:
        return None
    reused = False
    for pos, idx in enumerate(subscript):
        if pos == broadcast_dim:
            continue
        if isinstance(idx, torch.Tensor):
            for dim_size in idx.shape:
                if broadcast_block_id in _matching_block_ids(env, dim_size):
                    reused = True
                    break
        if reused:
            break
    if not reused:
        return None

    # Walk the value chain ``expand -> unsqueeze(None) -> load`` to recover the
    # source load. The stored value is a per-thread register holding ``val[a, c]``
    # whose coordinates live on the *load*'s thread axes; the store's own free
    # ``hl.arange`` index entries are distinct nodes that the synthetic-axis
    # machinery assigns to *different* axes. Reusing the load's coordinate for
    # those non-broadcast positions keeps the register and the store address on
    # the same thread axis (otherwise thread ``(a, c_load, c_store)`` would write
    # ``out[..., c_store] = val[a, c_load]`` for ``c_load != c_store``).
    load_node = _cute_unsqueeze_expand_load_source(value_node, broadcast_dim)
    load_coords: list[str] | None = None
    load_subscript_proxy: tuple[object, ...] | None = None
    if load_node is not None:
        load_tensor_node = load_node.args[0]
        load_subscript = load_node.args[1]
        if isinstance(load_tensor_node, torch.fx.Node) and isinstance(
            load_subscript, (list, tuple)
        ):
            load_tensor = load_tensor_node.meta.get("val")
            if isinstance(load_tensor, torch.Tensor):
                load_subscript_proxy = tuple(
                    map_arg([*load_subscript], lambda arg: arg.meta["val"])
                )
                load_subscript_ast = map_arg(
                    [*load_subscript], lambda arg: state.env[arg]
                )
                load_coords = _cute_index_exprs(
                    state,
                    [*load_subscript_proxy],
                    [*load_subscript_ast],
                    tensor=load_tensor,
                    inactive_singleton_slice_expr="0",
                )
                if len(load_coords) != load_tensor.ndim:
                    load_coords = None
                    load_subscript_proxy = None

    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != tensor.ndim or "None" in index_exprs:
        return None

    # Re-align each non-broadcast free-``hl.arange`` store position onto the
    # load's matching coordinate. Value dim ``d`` maps to load dim ``d`` before
    # the unsqueezed broadcast dim and ``d - 1`` after it. Only positions where
    # *both* the store and the matching load entry are free ``hl.arange`` index
    # tensors are remapped — a tensor *indexer* (``idx[tile]``) keeps its own
    # coordinate.
    if load_coords is not None and load_subscript_proxy is not None:
        for pos, idx in enumerate(subscript):
            if pos == broadcast_dim or not isinstance(idx, torch.Tensor):
                continue
            load_dim = pos if pos < broadcast_dim else pos - 1
            if not (0 <= load_dim < len(load_coords)):
                continue
            if isinstance(load_subscript_proxy[load_dim], torch.Tensor):
                index_exprs[pos] = load_coords[load_dim]

    # Replace the broadcast position's coordinate (currently the reused tile's
    # per-thread global index) with ``block_begin + lane`` so the lane loop sweeps
    # the full tile block, identically for every thread in the tile. ``block_begin``
    # is the *per-block* tile start (``global_index - local_coord``); in the CuTe
    # SIMT model the tile is mapped onto a thread axis, so the bare offset var
    # still carries the per-thread ``thread_idx`` lane and must be stripped.
    block_begin = _cute_block_tile_begin_expr(state, broadcast_block_id)
    if block_begin is None:
        return None
    lane_var = state.device_function.new_var("bcast_lane", dce=True)
    index_dtype = env.index_type()
    broadcast_coord = f"({block_begin}) + {index_dtype}({lane_var})"
    index_exprs[broadcast_dim] = broadcast_coord

    backend = env.backend
    target_dtype = backend.dtype_str(tensor.dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, index_exprs, "{value}"),
        value=value,
    )

    # Base mask excludes the broadcast position (its bound is enforced by the lane
    # bound below); other positions keep their tile/tensor masks.
    base_subscript = [
        slice(None) if pos == broadcast_dim else idx
        for pos, idx in enumerate(subscript)
    ]
    mask_expr = _cute_combined_mask(state, base_subscript, extra_mask, tensor=tensor)
    dim_size = _cute_tensor_dim_size_expr(state, tensor, broadcast_dim)
    lane_bound = f"({broadcast_coord}) < {dim_size}"
    mask_expr = lane_bound if mask_expr is None else f"({mask_expr}) and {lane_bound}"

    from .._compiler.ast_extension import create

    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    body_stmt: ast.stmt = ast.fix_missing_locations(
        ast.If(
            test=mask_ast,
            body=[ast.Expr(value=store_expr)],
            orelse=[],
        )
    )
    loop_stmt = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({block_size})"),
        body=[body_stmt],
        orelse=[],
        type_comment=None,
    )
    state.add_statement(loop_stmt)
    return ast.Constant(value=None)


def _codegen_cute_store_permute_lane_loops(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from .._compiler.cute.cute_reshape import _coords_from_flat_index
    from .._compiler.cute.cute_reshape import _flat_index_from_coords
    from .._compiler.cute.cute_reshape import _get_dim_local_coord
    from .._compiler.cute.cute_reshape import _get_tile_shape
    from .._compiler.cute.cute_reshape import _permute_reorders_active_dims
    from .._compiler.cute.cute_reshape import _shape_op_needs_materialization
    from .._compiler.cute.cute_reshape import _store_permute_info
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.tile_strategy import DeviceGridState

    if not isinstance(state.codegen, GenerateAST):
        return None
    grid_state = state.codegen.current_grid_state
    if not isinstance(grid_state, DeviceGridState) or not grid_state.has_lane_loops():
        return None
    if _shape_op_needs_materialization(value_node):
        return None

    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    index_tuple = _cute_index_tuple(index_exprs)
    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    tensor_name = state.device_function.tensor_arg(tensor).name

    input_node: torch.fx.Node
    output_val = value_node.meta.get("val")
    read_flat: str
    input_shape: list[int]

    info = _store_permute_info(value_node)
    if info is not None:
        input_node, perm = info
        input_val = input_node.meta.get("val")
        if not isinstance(input_val, torch.Tensor) or not isinstance(
            output_val, torch.Tensor
        ):
            return None
        if not _permute_reorders_active_dims(state.codegen, input_val, perm):
            return None
        source_tensor_node = input_node.args[0] if input_node.args else None
        source_extra_mask = input_node.args[2] if len(input_node.args) > 2 else None
        if (
            input_node.op == "call_function"
            and input_node.target is load
            and isinstance(source_tensor_node, torch.fx.Node)
            and source_extra_mask is None
        ):
            source_tensor = source_tensor_node.meta.get("val")
            if isinstance(source_tensor, torch.Tensor):
                reordered_subscript = [
                    subscript[perm.index(i)] for i in range(len(perm))
                ]
                reordered_ast_subscript = (
                    [ast_subscript[perm.index(i)] for i in range(len(perm))]
                    if isinstance(ast_subscript, (list, tuple))
                    else None
                )
                source_index_exprs = _cute_index_exprs(
                    state,
                    reordered_subscript,
                    ast_subscript=reordered_ast_subscript,
                    tensor=source_tensor,
                    inactive_singleton_slice_expr="0",
                )
                source_index_tuple = _cute_index_tuple(source_index_exprs)
                source_name = state.device_function.tensor_arg(source_tensor).name
                source_mask = _cute_combined_mask(
                    state,
                    reordered_subscript,
                    None,
                    tensor=source_tensor,
                )
                source_dtype = CompileEnvironment.current().backend.dtype_str(
                    source_tensor.dtype
                )
                return expr_from_string(
                    (
                        f"({tensor_name}.__setitem__({index_tuple}, "
                        f"({source_name}[{source_index_tuple}] if {source_mask} else {source_dtype}(0))) "
                        f"if {mask_expr} else None)"
                    )
                    if source_mask is not None and mask_expr is not None
                    else (
                        f"{tensor_name}.__setitem__({index_tuple}, "
                        f"{source_name}[{source_index_tuple}] if {source_mask} else {source_dtype}(0))"
                        if source_mask is not None
                        else (
                            f"({tensor_name}.__setitem__({index_tuple}, {source_name}[{source_index_tuple}]) "
                            f"if {mask_expr} else None)"
                            if mask_expr is not None
                            else f"{tensor_name}.__setitem__({index_tuple}, {source_name}[{source_index_tuple}])"
                        )
                    )
                )
            raise exc.BackendUnsupported("cute", "permute lane-loop source tensor")
        env = CompileEnvironment.current()
        df = state.device_function
        input_shape = _get_tile_shape(input_val, env, df.config)
        output_shape = _get_tile_shape(output_val, env, df.config)
        src_coords = [
            _get_dim_local_coord(state.codegen, input_val, i)
            for i in range(len(input_shape))
        ]
        current_flat = _flat_index_from_coords(src_coords, input_shape)
        output_coords = _coords_from_flat_index(current_flat, output_shape)
        read_coords = [output_coords[perm.index(i)] for i in range(len(perm))]
        read_flat = _flat_index_from_coords(read_coords, input_shape)
    elif value_node.target in {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    }:
        input_arg = value_node.args[0]
        if not isinstance(input_arg, torch.fx.Node):
            return None
        input_node = input_arg
        input_val = input_node.meta.get("val")
        if not isinstance(input_val, torch.Tensor) or not isinstance(
            output_val, torch.Tensor
        ):
            return None
        env = CompileEnvironment.current()
        df = state.device_function
        input_shape = _get_tile_shape(input_val, env, df.config)
        output_shape = _get_tile_shape(output_val, env, df.config)
        if input_shape == output_shape:
            return None
        input_non_unit = [s for s in input_shape if s != 1]
        output_non_unit = [s for s in output_shape if s != 1]
        if input_non_unit == output_non_unit:
            return None
        src_coords = [
            _get_dim_local_coord(state.codegen, input_val, i)
            for i in range(len(input_shape))
        ]
        current_flat = _flat_index_from_coords(src_coords, input_shape)
        output_coords = [
            _get_dim_local_coord(state.codegen, output_val, i)
            for i in range(len(output_shape))
        ]
        read_flat = _flat_index_from_coords(output_coords, output_shape)
    else:
        return None

    env = CompileEnvironment.current()
    df = state.device_function
    input_numel = 1
    for size in input_shape:
        input_numel *= size

    dtype_str = env.backend.dtype_str(input_val.dtype)
    smem_ptr = df.new_var("permute_smem_ptr")
    smem = df.new_var("permute_smem")
    state.codegen.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {input_numel})"
        )
    )
    state.codegen.add_statement(
        statement_from_string(
            f"{smem} = cute.make_tensor({smem_ptr}, ({input_numel},))"
        )
    )

    read_expr = (
        f"{df.tensor_arg(tensor).name}.__setitem__({index_tuple}, {smem}[{read_flat}])"
        if mask_expr is None
        else (
            f"({df.tensor_arg(tensor).name}.__setitem__({index_tuple}, {smem}[{read_flat}]) "
            f"if {mask_expr} else None)"
        )
    )
    return expr_from_string(
        f"({smem}.__setitem__({current_flat}, {{value}}), "
        f"cute.arch.sync_threads(), "
        f"{read_expr})",
        value=value,
    )


@_decorators.codegen(store, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton.
    # This produces tl.store(ptr + offset, val, mask) in the AST;
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_store_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_store(state, tensor, [*subscript], value, extra_mask)
    raise exc.BackendUnsupported("metal", f"store target type: {type(tensor)}")


def _try_splice_tcgen05_unary_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Splice attempt for ``out[tile] = chain(acc).to(x.dtype)``.

    Returns the splice-completion sentinel (``ast.Constant(value=None)``)
    on a successful splice (the caller should return it directly), and
    ``None`` if the splice did not fire — the caller should continue to
    the loud-failure backstop or the SIMT fallback.

    Splice is attempted only when the kernel has a tcgen05-registered
    matmul fx_node (``cute_state.matmul_fx_nodes`` non-empty), the
    store value has a backing FX node, the store target is a 2-D
    ``torch.Tensor``, and the chain analyzer accepts the value chain
    (returning ``(chain, anchor)`` for a non-empty chain rooted at
    a tcgen05 matmul). Chains the whitelist rejects (broadcast aux
    loads, reductions, kwarg-bearing binaries, etc.) leave the
    analyzer returning ``None`` and the splice does not fire — the
    loud-failure backstop then catches them.
    """
    cute_state = state.device_function.cute_state
    if not cute_state.matmul_fx_nodes:
        return None
    if value_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    analyzed = analyze_tcgen05_unary_epilogue_chain(
        state, value_node, output_global_shape=tuple(tensor.shape)
    )
    if analyzed is None:
        return None
    chain, anchor = analyzed
    assert chain.steps
    anchor_result_var = cute_state.matmul_fx_node_result_vars.get(anchor)
    if anchor_result_var is None:
        return None
    rewritten_stmt = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        anchor_result_var,
        epilogue_chain=chain,
    )
    if rewritten_stmt is None:
        return None
    stmts = rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
    for stmt in stmts:
        state.add_statement(stmt)
    return ast.Constant(value=None)


@_decorators.codegen(store, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    raw_value = state.ast_args[2]
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    value_node = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        maybe_value_node = state.fx_node.args[2]
        if isinstance(maybe_value_node, torch.fx.Node):
            value_node = maybe_value_node

    if isinstance(tensor, torch.Tensor):
        affine_range_store = _codegen_cute_affine_range_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            raw_value,
            extra_mask,
            value_node,
        )
        if affine_range_store is not None:
            state.add_statement(affine_range_store)
            return ast.Constant(value=None)
        affine_reshape_store = _codegen_cute_affine_reshape_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value_node,
        )
        if affine_reshape_store is not None:
            state.add_statement(affine_reshape_store)
            return ast.Constant(value=None)
        strided_slice_store = _codegen_cute_strided_slice_store(
            state,
            tensor,
            subscript,
            raw_value,
            extra_mask,
            value_node,
        )
        if strided_slice_store is not None:
            state.add_statement(strided_slice_store)
            return ast.Constant(value=None)

    value = state.ast_arg(2)

    if value_node is not None:
        if value_node.op == "call_function":
            if isinstance(tensor, torch.Tensor):
                rewritten_stmt = _codegen_cute_store_stack_load(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_loaded_index_trailing_slices(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_expand_broadcast_tile(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_permute_lane_loops(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
            from .._compiler.cute.cute_reshape import codegen_cute_store_permute

            rewritten = codegen_cute_store_permute(state, value, value_node)
            if rewritten is not None:
                value = rewritten

    if isinstance(tensor, tuple):
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        value = expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value,
        )
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        store_expr = expr_from_string(
            "({ptr}).store({value})", ptr=ptr_expr, value=value
        )
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if mask_expr is None:
            return store_expr
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(store_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=store_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"store target type: {type(tensor)}")

    _log_cute_layout(state, "store")

    if isinstance(value, ast.Name):
        rewritten_stmt = _codegen_cute_store_tcgen05_tile(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value.id,
        )
        if rewritten_stmt is not None:
            stmts = (
                rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
            )
            for stmt in stmts:
                state.add_statement(stmt)
            return ast.Constant(value=None)

    # Try to splice a whitelisted chain epilogue
    # (`out[tile] = chain(acc).to(x.dtype)`) into the role-local
    # tcgen05 epilogue's per-thread T2R loop. Implementation in
    # ``_try_splice_tcgen05_unary_epilogue``. Chains the whitelist
    # rejects (broadcast aux loads, reductions, etc.) leave the
    # splice off and fall through to the loud-failure backstop
    # below.
    spliced = _try_splice_tcgen05_unary_epilogue(
        state, tensor, subscript, ast_subscript, extra_mask, value_node
    )
    if spliced is not None:
        return spliced

    # Loud-failure backstop for fused-epilogue stores that follow a
    # tcgen05 matmul. The tcgen05 grid-emission path (in `program_id.py`)
    # does not bind the per-block-id `indices_<n>` / `mask_<n>` variable
    # names that the SIMT-fallback store path expects, so falling through
    # here would emit a kernel that crashes inside the cute DSL with
    # `name 'mask_0' is not defined`. Detect the pattern here — any
    # store value whose FX user chain transitively reaches a
    # tcgen05-registered matmul fx node — and raise a structured error
    # so the caller sees the actionable message instead of a cute-DSL
    # crash. Fixing this requires either (a) extending the tcgen05 grid
    # to emit per-block-id index/mask vars, or (b) per-subtile lambda
    # emission in `_codegen_cute_store_tcgen05_tile`.
    if (
        state.device_function.cute_state.matmul_fx_nodes
        and value_node is not None
        and reach_tcgen05_matmul_anchors(state, value_node)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 MMA path does not yet emit per-block-id indices "
            "and masks for non-whitelisted fused epilogues that follow "
            "the MMA. The store target's value chain depends on a "
            "tcgen05 matmul result through ops the chain analyzer "
            "rejects (e.g. aux tensors with a 3-D underlying shape "
            "and a static collapse like `aux3d[tile_m, tile_n, 0]`, "
            "loads whose index expression is not exactly the "
            "carrier tile-id symbol, non-scalar binary ops, "
            "`aten.add.Tensor` with `alpha=k`, or an intermediate "
            "`.to(d_inter)` cast where `d_inter` differs from the "
            "store-target dtype). Identity stores "
            "(`out[tile] = acc.to(x.dtype)`), whitelisted unary chains "
            "(relu/tanh/exp/log/sqrt/abs/neg + scalar add/sub/mul/div "
            "on the accumulator carrier), exact-shape 2-D "
            "auxiliary-tensor binary ops (`acc + residual[tile_m, "
            "tile_n]`), and rank-1 trailing-axis (rowvec) broadcast "
            "aux loads (`acc + bias[tile_n]`) all work via the "
            "fused-epilogue splice path. The leading-axis rank-1 "
            "form (`acc + bias[tile_m]`) is rejected because a bare "
            "rank-1 RHS aligns to the trailing axis under PyTorch "
            "broadcasting; an explicit colvec broadcast must be "
            "written with `bias[tile_m][:, None]` / "
            "`.unsqueeze(-1)`.",
        )

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    topk_lane_expr: object | None = None
    topk_k: object | None = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        value_node = state.fx_node.args[2]
        if (
            isinstance(value_node, torch.fx.Node)
            and value_node.target is operator.getitem
            and isinstance(value_node.args[0], torch.fx.Node)
            and value_node.args[0].target is torch.ops.aten.topk.default
        ):
            topk_lane_expr = value_node.args[0].meta.get("cute_topk_lane_expr")
            topk_k = value_node.args[0].meta.get("cute_topk_k")
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        index_exprs[-1] = topk_lane_expr
    store_uses_pointer = "None" not in index_exprs
    store_expr = _cute_scalar_store_expr(tensor_name, index_exprs, "{value}")
    assign_expr = expr_from_string(store_expr, value=value)

    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        topk_mask = f"({topk_lane_expr}) < {topk_k}"
        mask_expr = topk_mask if mask_expr is None else f"({mask_expr}) and {topk_mask}"
    if mask_expr is None:
        return assign_expr
    if store_uses_pointer:
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(assign_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=assign_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    return expr_from_string(
        f"({store_expr} if {mask_expr} else None)",
        value=value,
    )


# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    from .ref_tile import RefTile

    # Normalize indices and identify tensor indices
    indices = []
    tensor_idx_positions = []
    for i, idx in enumerate(index):
        if isinstance(idx, RefTile):
            idx = idx.index
        # pyrefly: ignore [bad-argument-type]
        indices.append(idx)
        if isinstance(idx, torch.Tensor):
            tensor_idx_positions.append(i)

    # Handle broadcasting for multiple tensor indices
    if len(tensor_idx_positions) > 1:
        grids = torch.meshgrid(
            # pyrefly: ignore [bad-argument-type]
            *(indices[i] for i in tensor_idx_positions),
            indexing="ij",
        )
        for i, grid in zip(tensor_idx_positions, grids, strict=False):
            # pyrefly: ignore [unsupported-operation]
            indices[i] = grid

    if extra_mask is not None:
        mask = extra_mask.to(torch.bool)

        # Check bounds for tensor indices
        for i, idx in enumerate(indices):
            if isinstance(idx, torch.Tensor):
                mask = mask & (idx >= 0) & (idx < tensor.shape[i])
        mask_count = int(mask.sum().item())
        if mask_count == 0:
            return

        # Use index_put_ for masked stores
        valid_indices = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                valid_indices.append(idx[mask].long())
            else:
                idx_val = int(idx) if isinstance(idx, torch.SymInt) else idx
                valid_indices.append(
                    # pyrefly: ignore [no-matching-overload]
                    torch.full(
                        (mask_count,), idx_val, dtype=torch.long, device=tensor.device
                    )
                )

        if isinstance(value, torch.Tensor):
            values = value[mask]
        else:
            val = int(value) if isinstance(value, torch.SymInt) else value
            values = torch.full(
                (mask_count,), val, dtype=tensor.dtype, device=tensor.device
            )

        # Check for duplicate indices - this is undefined behavior in Triton
        if valid_indices:
            stacked = torch.stack(valid_indices, dim=1)
            unique_count = stacked.unique(dim=0).size(0)
            if unique_count < stacked.size(0):
                raise exc.DuplicateStoreIndicesError(
                    "hl.store with duplicate indices has undefined behavior in compiled mode. "
                    "The order in which values are written to the same memory location is "
                    "non-deterministic and may vary between Triton versions and backends."
                )

        tensor.index_put_(tuple(valid_indices), values, accumulate=False)
        return

    # Simple assignment
    tensor[tuple(indices)] = (  # pyrefly: ignore[unsupported-operation]
        int(value) if isinstance(value, torch.SymInt) else value
    )


@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def load(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    This function is equivalent to `tensor[index]` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range. It also accepts an optional
    `eviction_policy` which is forwarded to the underlying Triton `tl.load`
    call to control the cache eviction behavior (e.g., "evict_last").

    Args:
        tensor: The tensor / stack tensor to load from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
        eviction_policy: Optional Triton load eviction policy to hint cache behavior
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(load)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> tuple[torch.Tensor | tuple, list[object], torch.Tensor | None, str | None]:
    from .tile_proxy import Tile

    index = Tile._tiles_to_sizes_for_index(index)
    if isinstance(tensor, StackTensor):
        return (tuple(tensor), index, extra_mask, eviction_policy)
    assert isinstance(tensor, torch.Tensor)
    return (tensor, index, extra_mask, eviction_policy)


@_decorators.register_fake(load)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        target_shape = SubscriptIndexing.compute_shape(tensor, index)
        env = CompileEnvironment.current()
        env.backend.process_fake_tensor_load(tensor, index)
        return env.new_index_result(tensor, target_shape)
    if isinstance(tensor, tuple):
        tensor_like, dev_ptrs = tensor
        assert isinstance(tensor_like, torch.Tensor)
        assert isinstance(dev_ptrs, torch.Tensor)
        tensor_shape = SubscriptIndexing.compute_shape(tensor_like, index)
        target_shape = list(dev_ptrs.size()) + tensor_shape
        return tensor_like.new_empty(target_shape)
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


def _maybe_materialize_tile_index_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> ast.AST | None:
    """If this load is on a ``tile.index`` value (e.g. ``tile_m.index[:, None]``),
    emit the inline ``indices_<bid>[<sub>]`` expression and return it.
    Returns ``None`` otherwise.

    ``tile.index`` tensors are synthesized inside the kernel — they aren't
    registered in ``tensor_to_origin`` — so the regular load path's
    ``tensor_arg`` lookup would ``KeyError``.  Supported subscript entries
    are ``None`` (new axis) and ``slice(None)`` (full slice).
    """
    from ..language import tile_index

    tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
    if not (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        return None

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
            raise AssertionError(f"Unexpected index type in tile_index load: {idx}")
    return expr_from_string(f"{base_var}[{', '.join(parts)}]")


@_decorators.codegen(load, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None

    device_fn = state.device_function
    load_idx = device_fn.device_load_index
    device_fn.device_load_index += 1

    # If no explicit eviction_policy and we're in device code, use tunable
    if eviction_policy is None and state.codegen.on_device:
        policies = state.config.load_eviction_policies
        if load_idx < len(policies):
            policy_value = policies[load_idx]
            eviction_policy = _EVICTION_POLICY_MAP.get(policy_value, policy_value)

    if eviction_policy is not None:
        assert isinstance(eviction_policy, str)
        eviction_policy = ast.Constant(value=eviction_policy)

    cache_modifier = None
    if state.codegen.on_device:
        modifier_idx = device_fn.device_load_cache_modifier_index
        device_fn.device_load_cache_modifier_index += 1
        modifiers = state.config.load_cache_modifiers
        if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
            cache_modifier = ast.Constant(value=modifiers[modifier_idx])

    if isinstance(tensor, torch.Tensor):
        tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
        if tile_index_result is not None:
            return tile_index_result

        # Use the shared memory op index for indexing strategy
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)

        if state.codegen.load_transform is not None:
            return state.codegen.load_transform(
                state,
                tensor,
                [*subscript],
                extra_mask,
                eviction_policy,
                cache_modifier,
                strategy.codegen_load,
            )

        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, cache_modifier
        )
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack loads (multi-tensor device pointers);
        # fall through to the unfused path regardless of load_transform.
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state,
            tensor,
            dev_ptrs_ast,
            [*subscript],
            extra_mask,
            eviction_policy,
            cache_modifier,
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
    if tile_index_result is not None:
        return tile_index_result

    return pallas_codegen.load_expr(state, list(subscript), tensor)


@_decorators.codegen(load, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton.
    # This produces tl.load(ptr + offset, mask, other=0) in the AST;
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None
    assert isinstance(eviction_policy, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_load_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, None
        )
    raise exc.BackendUnsupported("metal", f"load tensor type: {type(tensor)}")


def _cute_load_feeds_sort_or_scan(load_node: object) -> bool:
    """Return True if ``load_node`` feeds a sort/topk/_associative_scan.

    Direct users (sort/topk and the scalar ``_associative_scan`` path) are
    matched immediately.  For a tuple ``_associative_scan`` the index stream is
    typically a ``load`` that flows through a chain of dtype-cast / shape ops
    (e.g. ``indices[tile].float().unsqueeze(1).expand_as(vals)``) before
    reaching the scan.  To recover a scalar load for that stream we follow the
    forward chain through those pass-through ops.
    """
    from torch.fx.node import Node

    from .._compiler.cute.indexing import is_cute_shape_chain_target

    if not isinstance(load_node, Node):
        return False

    passthrough_targets = (torch.ops.prims.convert_element_type.default,)
    seen: set[Node] = set()
    stack: list[Node] = [load_node]
    while stack:
        node = stack.pop()
        for user in node.users:
            if not isinstance(user, Node):
                continue
            target = user.target
            if (
                target in (torch.ops.aten.sort.default, torch.ops.aten.topk.default)
                or getattr(target, "__name__", None) == "_associative_scan"
            ):
                return True
            if (
                is_cute_shape_chain_target(target) or target in passthrough_targets
            ) and user not in seen:
                seen.add(user)
                stack.append(user)
    return False


@_decorators.codegen(load, "cute")
def _(state: CodegenState) -> object:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, tuple):
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        load_expr = f"({ast.unparse(ptr_expr)}).load()"
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if tensor_like.dtype is torch.bool:
            load_expr = f"({load_expr} != cutlass.Uint8(0))"
            if mask_expr is None:
                return expr_from_string(load_expr)
            return expr_from_string(
                f"({load_expr} if {mask_expr} else cutlass.Boolean(0))"
            )
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else {target_dtype}(0))")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"load tensor type: {type(tensor)}")

    _log_cute_layout(state, "load")

    from ..language import tile_index

    tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
    if (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        env = CompileEnvironment.current()
        block_id = env.get_block_id(tensor.size(0))
        if block_id is None:
            raise exc.BackendUnsupported("cute", "tile_index load block id")
        index_var = _cute_active_index_var(state, block_id)
        if index_var is None:
            raise exc.BackendUnsupported("cute", "inactive tile_index load")
        for idx in subscript:
            if idx is None or idx == slice(None):
                continue
            raise exc.BackendUnsupported(
                "cute", f"tile_index load index type: {type(idx)}"
            )
        return expr_from_string(index_var)

    cute_state = state.device_function.cute_state
    if cute_state.suppress_root_lane_loops or (
        state.fx_node is not None
        and cute_state.is_collective_handled_load(state.fx_node.name)
    ):
        zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

    packed_affine_lhs = _maybe_codegen_cute_packed_affine_lhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_affine_lhs is not None:
        return packed_affine_lhs

    packed_rhs_load = _maybe_codegen_cute_packed_rhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_rhs_load is not None:
        return packed_rhs_load

    if _is_cute_affine_range_load_for_store(state, subscript, ast_subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")
    if _is_cute_strided_slice_load_for_store(state, tensor, subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

    tensor_name = state.device_function.tensor_arg(tensor).name
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    mask_expr = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor,
        include_tensor_index_masks=False,
    )
    vec_ctx = _cute_vector_load_ctx(state, tensor, subscript, index_exprs, extra_mask)
    if vec_ctx is not None:
        vec_width, vec_block_id, vec_mode = vec_ctx
        from .._compiler.reduction_strategy import LoopedReductionStrategy

        loops = state.codegen.active_device_loops.get(vec_block_id)
        strategy = loops[-1].strategy if loops else None
        if vec_mode == "vec":
            load_expr = _cute_vector_load_expr(
                tensor_name, index_exprs, tensor.dtype, vec_width=vec_width
            )
            # The mask is deferred to the post-fold scalar in
            # codegen_reduction.  The vec load itself is unconditional; the
            # mask is recorded on the active LoopedReductionStrategy and
            # applied around the folded sum.
            if isinstance(strategy, LoopedReductionStrategy):
                strategy._cute_emitted_vec_load = True
                if mask_expr is not None:
                    strategy._cute_pending_vec_masks.append(mask_expr)
            mask_expr = None
        elif vec_mode == "unroll":
            # Register (or reuse) a hoisted U16 vec load for this (tensor,
            # base_index) pair, then return ``hoist_var[vi].bitcast(dtype)``
            # so the existing scalar pipeline sees a scalar of the original
            # dtype.
            assert isinstance(strategy, LoopedReductionStrategy)
            load_expr = _cute_register_unroll_vec_hoist(
                state,
                strategy,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
            )
        elif vec_mode == "tile_unroll":
            # Same hoist protocol as ``LoopedReductionStrategy``'s
            # ``unroll`` mode but for ``CuteNDTileStrategy`` lane loops.
            from .._compiler.tile_strategy import BlockSizeTileStrategy

            assert isinstance(strategy, BlockSizeTileStrategy)
            load_expr = _cute_register_tile_unroll_vec_hoist(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
            )
        else:
            assert vec_mode == "tile_unroll_split2"
            # V=8 fp16/bf16: emit two back-to-back ``cute.arch.load(...,
            # V=4)`` calls (lanes 0-3 and 4-7).  Works around the CuTe
            # DSL's ``nvvm.load.ext`` ICE on V=8 while still issuing the
            # full LDG.128 of bytes-per-thread-per-outer-iter.
            from .._compiler.tile_strategy import BlockSizeTileStrategy

            assert isinstance(strategy, BlockSizeTileStrategy)
            load_expr = _cute_register_tile_unroll_vec_hoist_split2(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
            )
    else:
        load_expr = _cute_scalar_load_expr(tensor_name, index_exprs, tensor.dtype)
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else cutlass.Boolean(0))")
    if state.fx_node is not None and _cute_load_feeds_sort_or_scan(state.fx_node):
        from .._compiler.cute.indexing import CuteSortableLoad

        tensor_dim = 0
        sort_index_pos = -1
        for idx in subscript:
            if idx is None:
                continue
            if tensor_dim == tensor.ndim - 1:
                sort_index_pos = tensor_dim
                break
            tensor_dim += 1
        if sort_index_pos < 0:
            raise exc.BackendUnsupported("cute", "sort/topk input rank")
        sortable_load = CuteSortableLoad(
            expr=expr_from_string(
                load_expr
                if mask_expr is None
                else f"({load_expr} if {mask_expr} else {_cute_scalar_storage_dtype(tensor.dtype)}(0))"
            ),
            tensor_name=tensor_name,
            index_exprs=tuple(index_exprs),
            sort_index_pos=sort_index_pos,
            mask_expr=mask_expr,
            dtype=tensor.dtype,
        )
        state.fx_node.meta["cute_sortable_load"] = sortable_load
        return sortable_load.expr
    if mask_expr is None:
        return expr_from_string(load_expr)
    zero = _cute_scalar_storage_dtype(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")


@_decorators.get_masked_value(load)
def _(node: torch.fx.Node) -> int:
    return 0  # loads are always masked to 0


# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(load)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    from .ref_tile import RefTile

    if extra_mask is None:
        # Convert RefTiles to indices
        indices = [idx.index if isinstance(idx, RefTile) else idx for idx in index]
        # Use meshgrid for Cartesian product when we have multiple tensor indices
        tensor_idxs = [
            i for i, idx in enumerate(indices) if isinstance(idx, torch.Tensor)
        ]
        if len(tensor_idxs) > 1:
            # pyrefly: ignore [bad-argument-type]
            grids = torch.meshgrid(*(indices[i] for i in tensor_idxs), indexing="ij")
            for i, grid in zip(tensor_idxs, grids, strict=False):
                indices[i] = grid
        # pyrefly: ignore [bad-argument-type, bad-index]
        return tensor[tuple(indices)]

    # Create zero result matching mask shape
    result = torch.zeros(extra_mask.shape, dtype=tensor.dtype, device=tensor.device)

    # Process indices: convert RefTiles and clamp tensor indices
    orig_indices, safe_indices, is_tensor_mask = [], [], []
    for i, idx in enumerate(index):
        if isinstance(idx, RefTile):
            idx = idx.index  # Convert RefTile to tensor

        if isinstance(idx, torch.Tensor):
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            orig_indices.append(idx)
            safe_indices.append(torch.clamp(idx, 0, dim_size - 1))
            is_tensor_mask.append(True)
        else:
            orig_indices.append(idx)
            safe_indices.append(idx)
            is_tensor_mask.append(False)

    # Apply broadcasting if we have multiple tensor indices
    tensor_positions = [i for i, is_tensor in enumerate(is_tensor_mask) if is_tensor]

    if len(tensor_positions) > 1:
        # Add unsqueeze operations for broadcasting
        broadcast_indices = []
        for i, (idx, is_tensor) in enumerate(
            zip(safe_indices, is_tensor_mask, strict=False)
        ):
            if is_tensor:
                new_idx = idx
                # Add dimension for each other tensor index
                for j, other_pos in enumerate(tensor_positions):
                    if other_pos != i:
                        new_idx = new_idx.unsqueeze(j if other_pos < i else -1)
                broadcast_indices.append(new_idx)
            else:
                broadcast_indices.append(idx)
        values = tensor[tuple(broadcast_indices)]
    else:
        values = tensor[tuple(safe_indices)]

    # Build validity mask
    valid_mask = extra_mask.clone()
    for i, (orig_idx, is_tensor) in enumerate(
        zip(orig_indices, is_tensor_mask, strict=False)
    ):
        if is_tensor:
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            in_bounds = (orig_idx >= 0) & (orig_idx < dim_size)
            # Broadcast to match mask shape by adding dimensions
            # Count how many tensor indices come before and after this one
            n_before = sum(1 for j in range(i) if is_tensor_mask[j])
            n_after = sum(
                1 for j in range(i + 1, len(is_tensor_mask)) if is_tensor_mask[j]
            )

            # Add dimensions: n_after dimensions at the end, n_before at the beginning
            for _ in range(n_after):
                in_bounds = in_bounds.unsqueeze(-1)
            for _ in range(n_before):
                in_bounds = in_bounds.unsqueeze(0)
            valid_mask = valid_mask & in_bounds

    return torch.where(valid_mask, values, result)


# ============================================================================
# NKI backend load/store codegen
#
# Ported verbatim from fix-nki-kernel-compilation. Six NKI helpers followed by
# the @_decorators.codegen(load, "nki") and @_decorators.codegen(store, "nki")
# implementations. The most complex piece of the backend; do not simplify.
# Most heavy imports are lazy (inside function bodies); IndirectAP/DynamicAP
# are imported at module top.
# ============================================================================

def _nki_shifted_tile_subscript(
    fx_node: object,
    state: "CodegenState",
    env: "CompileEnvironment",
) -> str | None:
    """Recognize ``tile.index ± const`` subscripts and return a shifted slice.

    For patterns like ``y[:, tile1.index - x.size(1)]``, the index tensor is
    ``tile1.index - x.size(1)`` — a 1D tile of integers. Instead of emitting
    an indirect gather, observe that this is just a UNIFORM shift by a
    constant: the gather of ``y[p, tile1.index - C]`` is equivalent to the
    slice ``y[p, offset_1 - C : offset_1 - C + block_size]``.

    Supports:
      - ``aten.sub.Tensor(tile_index, const_int)`` → ``offset - const``
      - ``aten.add.Tensor(tile_index, const_int)`` → ``offset + const``
      - ``aten.sub.Tensor(const_int, tile_index)`` → ``const - offset``
        (only valid for `block_size == 1`; otherwise the slice direction
        is reversed which dma_copy doesn't support)

    Returns a slice string ``"start:end"`` or None if the pattern doesn't
    apply.
    """
    if not isinstance(fx_node, torch.fx.Node):
        return None

    target = getattr(fx_node, "target", None)
    target_name = str(target) if target else ""
    import operator as _operator
    is_add = (
        "add.Tensor" in target_name
        or "add.Scalar" in target_name
        or target is torch.ops.aten.add.Tensor
        or target is torch.ops.aten.add.Scalar
        or target is _operator.add
    )
    is_sub = (
        "sub.Tensor" in target_name
        or "sub.Scalar" in target_name
        or target is torch.ops.aten.sub.Tensor
        or target is torch.ops.aten.sub.Scalar
        or target is _operator.sub
    )
    if not (is_add or is_sub):
        return None

    args = fx_node.args
    if len(args) != 2:
        return None

    def _get_tile_index_block_id(node: object) -> int | None:
        """If ``node`` is ``tile.index`` (FX value is a 1D SymInt-tensor),
        return its block_id.  tile.index gets lowered as a tensor-valued
        node whose shape is ``[block_size]`` symbolically.
        Also recognizes scalar grid loop variables (block_size == 1).
        """
        if not isinstance(node, torch.fx.Node):
            return None
        val = node.meta.get("val")
        # Scalar grid loop variable (hl.grid): val is int or SymInt, block_size=1
        if isinstance(val, (int, torch.SymInt)):
            node_ast = state.codegen.ast_for_fx_node(node)
            if isinstance(node_ast, ast.AST):
                node_expr = ast.unparse(node_ast)
                for bid in state.codegen.active_device_loops:
                    try:
                        block_size = int(
                            env.block_sizes[bid].from_config_assert(state.config)
                        )
                    except Exception:
                        continue
                    if block_size == 1 and node_expr == state.codegen.offset_var(bid):
                        return bid
        if isinstance(val, torch.Tensor) and val.ndim == 1:
            size = val.size(0)
            if isinstance(size, torch.SymInt):
                bid = env.get_block_id(size)
                if bid is not None:
                    return bid
                sym_expr = size._sympy_()
                free = getattr(sym_expr, "free_symbols", None)
                if free:
                    for sym in free:
                        bid = env.get_block_id(sym)
                        if bid is not None:
                            return bid
                try:
                    size_hint = int(env.size_hint(size))
                except Exception:
                    size_hint = -1
                for bid in state.codegen.active_device_loops:
                    try:
                        block_size = int(
                            env.block_sizes[bid].from_config_assert(state.config)
                        )
                    except Exception:
                        continue
                    if block_size == size_hint:
                        return bid
            node_target = getattr(node, "target", None)
            node_target_name = str(node_target) if node_target else ""
            if "tile_index" in node_target_name and len(
                state.codegen.active_device_loops
            ) == 1:
                return next(iter(state.codegen.active_device_loops))
        return None

    def _get_const_int(node: object) -> int | None:
        if isinstance(node, int):
            return node
        if isinstance(node, torch.SymInt):
            try:
                return int(node._sympy_())
            except Exception:
                # Don't use size_hint here — it gives a fake concrete value
                # that may not represent the actual runtime value (e.g. 8192
                # for a loop variable that will be 0..1 at runtime).
                return None
        if isinstance(node, torch.fx.Node):
            val = node.meta.get("val")
            if isinstance(val, int):
                return val
            if isinstance(val, torch.SymInt):
                try:
                    return int(val._sympy_())
                except Exception:
                    return None  # Same: don't use size_hint
        try:
            return int(node)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return None

    lhs_bid = _get_tile_index_block_id(args[0])
    rhs_const = _get_const_int(args[1]) if lhs_bid is not None else None

    if lhs_bid is not None and lhs_bid in state.codegen.active_device_loops:
        offset_var = state.codegen.offset_var(lhs_bid)
        block_size = int(env.block_sizes[lhs_bid].from_config_assert(state.config))
        # Get the shift expression — either numeric const or AST expression
        # (e.g. tile_c.begin * chunk_size → "offset_5 * 128")
        _rhs_expr: str | None = None
        if rhs_const is not None:
            _rhs_expr = str(rhs_const)
        elif isinstance(args[1], torch.fx.Node):
            _rhs_ast = state.codegen.ast_for_fx_node(args[1])
            if isinstance(_rhs_ast, ast.AST):
                _candidate = ast.unparse(_rhs_ast)
                # Only use as a shift if it's a scalar expression — not an
                # SBUF tile variable. SBUF tiles (iota indices, loaded tiles,
                # etc.) cannot be used as integer offsets in DMA slice
                # expressions. Check both the sbuf_shapes dict and variable
                # naming conventions (_nki_*, indices_*).
                _sbuf_shapes = getattr(
                    getattr(state, "device_function", None), "_nki_sbuf_shapes", {}
                )
                # Block SBUF tile variables from being used as DMA slice offsets.
                # SBUF tiles cannot be used as Python integers in DMA slice
                # expressions — they cause NKI compile errors like
                # "'add' expected (int, int) ... got (int, object)".
                _is_sbuf_tile = (
                    _candidate in _sbuf_shapes
                    or _candidate.startswith(("_nki_", "indices_"))
                )
                if not _is_sbuf_tile:
                    _rhs_expr = _candidate
        if _rhs_expr is not None:
            if is_sub:
                start = f"{offset_var} - {_rhs_expr}"
            else:  # is_add
                start = f"{offset_var} + {_rhs_expr}"
            # Build the end expression.  When the shift is a numeric constant,
            # fold it with block_size arithmetically to avoid emitting
            # "(offset_var + N) + block_size" which NKI's symbolic tracer
            # cannot evaluate for compound affine expressions.
            try:
                _rhs_int = int(_rhs_expr)
                if is_sub:
                    end = f"{offset_var} - {_rhs_int - block_size}"
                else:
                    end = f"{offset_var} + {_rhs_int + block_size}"
            except (ValueError, TypeError):
                end = f"{start}+{block_size}"
            return f"{start}:{end}"

    # Reverse: const ± tile
    rhs_bid = _get_tile_index_block_id(args[1])
    lhs_const = _get_const_int(args[0]) if rhs_bid is not None else None
    if rhs_bid is not None and rhs_bid in state.codegen.active_device_loops:
        offset_var = state.codegen.offset_var(rhs_bid)
        block_size = int(env.block_sizes[rhs_bid].from_config_assert(state.config))
        # Get the LHS expression — either a numeric const or an AST expression
        # (e.g. tile_c.begin * chunk_size → "offset_5 * 128")
        _lhs_expr: str | None = None
        if lhs_const is not None:
            _lhs_expr = str(lhs_const)
        elif isinstance(args[0], torch.fx.Node):
            _lhs_ast = state.codegen.ast_for_fx_node(args[0])
            if isinstance(_lhs_ast, ast.AST):
                _candidate_lhs = ast.unparse(_lhs_ast)
                _sbuf_shapes_rev = getattr(
                    getattr(state, "device_function", None), "_nki_sbuf_shapes", {}
                )
                _is_sbuf_tile_lhs = (
                    _candidate_lhs in _sbuf_shapes_rev
                    or _candidate_lhs.startswith(("_nki_", "indices_"))
                )
                if not _is_sbuf_tile_lhs:
                    _lhs_expr = _candidate_lhs
        if _lhs_expr is not None:
            if is_add:
                start = f"{_lhs_expr} + {offset_var}"
                # Fold numeric LHS with block_size to avoid double-addition.
                try:
                    _lhs_int = int(_lhs_expr)
                    end = f"{offset_var} + {_lhs_int + block_size}"
                except (ValueError, TypeError):
                    end = f"{start}+{block_size}"
                return f"{start}:{end}"
            # const - tile: only safe when block_size == 1
            if is_sub and block_size == 1:
                return f"{_lhs_expr} - {offset_var}"

    return None


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
    from .._compiler.ast_extension import statement_from_string
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


def _nki_lookup_sbuf_shape_dtype(
    state: "CodegenState", name: str
) -> tuple[list[int] | None, str]:
    device_fn = state.device_function
    shape = device_fn._nki_sbuf_shapes.get(name)
    if shape is None:
        lookup = name
        while "_copy" in lookup:
            lookup = lookup[: lookup.rfind("_copy")]
            shape = device_fn._nki_sbuf_shapes.get(lookup)
            if shape is not None:
                name = lookup
                break
    dtype = getattr(device_fn, "_nki_sbuf_dtypes", {}).get(name, "nl.int32")
    return shape, dtype


def _nki_as_uint32_p1_vector(
    state: "CodegenState", name: str, p_count: int
) -> str | None:
    """Return an SBUF ``[P, 1]`` uint32 vector-offset tile for ``name``.

    Helion tile indices usually materialize as row vectors ``[1, P]`` while
    NKI vector indirection expects one row id per partition, ``[P, 1]``.
    """

    from .._compiler.ast_extension import statement_from_string

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

    tr_psum = device_fn.new_var("_ig_tr_psum", dce=True)
    tr_sbuf = device_fn.new_var("_ig_tr_sbuf", dce=True)
    device_fn._nki_sbuf_shapes[tr_sbuf] = [p_count, 1]
    device_fn._nki_sbuf_dtypes[tr_sbuf] = "nl.uint32"

    if dtype in int_dtypes or dtype == "nl.uint32":
        cast_in = device_fn.new_var("_ig_cast_in", dce=True)
        device_fn._nki_sbuf_shapes[cast_in] = [1, p_count]
        device_fn._nki_sbuf_dtypes[cast_in] = "nl.float32"
        state.codegen.add_statement(
            statement_from_string(
                f"{cast_in} = nl.ndarray([1, {p_count}], nl.float32, buffer=nl.sbuf)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.activation(dst={cast_in}, op=nl.copy, data={name})"
            )
        )
        state.codegen.add_statement(
            statement_from_string(
                f"{tr_psum} = nl.ndarray([{p_count}, 1], nl.float32, buffer=nl.psum)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.nc_transpose(dst={tr_psum}, data={cast_in})")
        )
    else:
        state.codegen.add_statement(
            statement_from_string(
                f"{tr_psum} = nl.ndarray([{p_count}, 1], {dtype}, buffer=nl.psum)"
            )
        )
        state.codegen.add_statement(
            statement_from_string(f"nisa.nc_transpose(dst={tr_psum}, data={name})")
        )

    state.codegen.add_statement(
        statement_from_string(
            f"{tr_sbuf} = nl.ndarray([{p_count}, 1], nl.uint32, buffer=nl.sbuf)"
        )
    )
    state.codegen.add_statement(
        statement_from_string(
            f"nisa.tensor_scalar(dst={tr_sbuf}, data={tr_psum}, op0=nl.add, operand0=0.0)"
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
                from .._compiler.compile_environment import CompileEnvironment as _CE
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
                from .._compiler.ast_extension import statement_from_string

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
                from .._compiler.ast_extension import statement_from_string

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


def _nki_subscript_block_id(
    sub_val: object,
    fx_node_i: object,
    env: CompileEnvironment,
) -> int | None:
    """Return the block_id a subscript element corresponds to, or None.

    Tries in order:
      1. The live SymInt in ``sub_val`` via ``env.get_block_id``.
      2. The FX subscript node's ``meta["val"]`` (which might still be a
         live SymInt even if ``sub_val`` has been concretized).
      3. Free sympy symbols of either — match each symbol to a registered
         block size via ``env.get_block_id``.
      4. The FX node's name (Helion names symnode getters after their debug
         names ``block_size_N`` / ``rdim_N`` per compile_environment.py:245).
      5. The FX node's first positional arg (which for ``_get_symnode`` is
         the debug name string).

    This is the canonical NKI-backend subscript→block_id resolver. It does
    NOT consult ``active_device_loops``: positional heuristics there are
    fragile (e.g. broke for ``y[:, tile_n]`` when subscripts concretize).
    """
    import sympy as _sp

    # 1. Direct SymInt lookup.
    if isinstance(sub_val, torch.SymInt):
        bid = env.get_block_id(sub_val)
        if bid is not None:
            return bid
        sym_expr = sub_val._sympy_()
        # Only match a bare symbol (e.g. u2), not a compound expression
        # like u2 + 1. Compound expressions containing a block symbol should
        # be handled by _nki_shifted_tile_subscript instead.
        if isinstance(sym_expr, _sp.Symbol):
            bid = env.get_block_id(sym_expr)
            if bid is not None:
                return bid

    # 1b. FakeTensor subscript (1D tile): look at size(0) for the block_id.
    # Handles compound subscripts like "tile_c * chunk_size + tile_m.index"
    # which trace as 1D FakeTensors whose size is the tile_m block_size.
    if isinstance(sub_val, torch.Tensor) and sub_val.ndim == 1:
        size0 = sub_val.size(0)
        if isinstance(size0, torch.SymInt):
            bid = env.get_block_id(size0)
            if bid is not None:
                return bid
            sym_expr0 = size0._sympy_()
            if isinstance(sym_expr0, _sp.Symbol):
                bid = env.get_block_id(sym_expr0)
                if bid is not None:
                    return bid
            # Also try free symbols (for expressions like u2 in tensor size)
            free0 = getattr(sym_expr0, "free_symbols", None)
            if free0:
                for sym in free0:
                    bid = env.get_block_id(sym)
                    if bid is not None:
                        return bid

    # 2. FX node's meta["val"] (survives even if sub_val was concretized).
    if isinstance(fx_node_i, torch.fx.Node):
        fx_val = fx_node_i.meta.get("val")
        if isinstance(fx_val, torch.SymInt):
            bid = env.get_block_id(fx_val)
            if bid is not None:
                return bid
            sym_expr = fx_val._sympy_()
            # Same as above: only match bare symbols.
            if isinstance(sym_expr, _sp.Symbol):
                bid = env.get_block_id(sym_expr)
                if bid is not None:
                    return bid
        elif isinstance(fx_val, _sp.Expr):
            free = getattr(fx_val, "free_symbols", None)
            if free:
                for sym in free:
                    bid = env.get_block_id(sym)
                    if bid is not None:
                        return bid

        # 3. FX node name: "block_size_N" -> block_id N. Tile symnodes get
        #    this name via device_ir._get_proxy_slot's debug_name.
        node_name = getattr(fx_node_i, "name", None) or ""
        if node_name.startswith("block_size_"):
            try:
                cand = int(node_name.removeprefix("block_size_").split("_")[0])
                if 0 <= cand < len(env.block_sizes):
                    return cand
            except (ValueError, IndexError):
                pass
        elif node_name.startswith("rdim_"):
            try:
                cand = int(node_name.removeprefix("rdim_").split("_")[0])
                if 0 <= cand < len(env.block_sizes):
                    return cand
            except (ValueError, IndexError):
                pass

        # 4. FX node args[0] may be the debug name string (when target is
        #    _get_symnode).
        args = getattr(fx_node_i, "args", ())
        if args:
            first = args[0]
            if isinstance(first, str):
                for prefix in ("block_size_", "rdim_"):
                    if first.startswith(prefix):
                        try:
                            cand = int(first.removeprefix(prefix).split("_")[0])
                            if 0 <= cand < len(env.block_sizes):
                                return cand
                        except (ValueError, IndexError):
                            pass

    return None


@_decorators.codegen(load, "nki")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import create
    from .._compiler.ast_extension import expr_from_string
    from .._compiler.ast_extension import statement_from_string

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
    from .tile_ops import tile_index

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
    _ret_bufs = getattr(device_fn, "_nki_return_buffers", {})
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
                    from .._compiler.ast_extension import statement_from_string as _sfs3d
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
            from .tile_ops import tile_id as _tile_id_fn
            try:
                from .tile_ops import tile_begin as _tile_begin_fn
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
                        _sbuf_shapes_dyn = getattr(
                            getattr(state, "device_function", None), "_nki_sbuf_shapes", {}
                        )
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
                _sbuf_shapes_ck = getattr(
                    getattr(state, "device_function", None), "_nki_sbuf_shapes", {}
                )
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
                mask_dtype = getattr(device_fn, "_nki_sbuf_dtypes", {}).get(
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
            _active_lps_fg = getattr(state.codegen, "active_device_loops", {})
            from .._compiler.compile_environment import CompileEnvironment as _CE_fg
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
            from ..language._tracing_ops import _mask_to as _mask_to_fn
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
                _active_loops = getattr(state.codegen, "active_device_loops", {})
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
                    _sbuf = getattr(device_fn, "_nki_sbuf_shapes", {})
                    _matching = [(k, v) for k, v in _sbuf.items() if v == [p_count, k_count]]
                    print(f"  SBUF vars with shape [{p_count},{k_count}]: {_matching[:5]}")
                    _mask_like = [(k, v) for k, v in _sbuf.items() if ("cmp" in k or "mask" in k or "pred" in k)]
                    print(f"  Mask-like SBUF vars: {_mask_like[:10]}")
                    from .._compiler.compile_environment import CompileEnvironment as _CE_dbg
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
                    from .._compiler.compile_environment import CompileEnvironment as _CE_jg
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
                        _ret_bufs = getattr(device_fn, "_nki_return_buffers", {})
                        if _store_tensor_id in _ret_bufs:
                            _store_out_buf = _ret_bufs[_store_tensor_id]["buf_name"]
                        else:
                            # Pre-allocate the return buffer for this output tensor
                            try:
                                from .._compiler.host_function import HostFunction as _HF
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
            from .._compiler.aten_lowering import Lowering as _BaseLowering

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

                from .._compiler.ast_extension import statement_from_string as _sfs_sg
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
                from .._compiler.ast_extension import statement_from_string as _sfs_ig
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
            mask_dtype = getattr(device_fn, "_nki_sbuf_dtypes", {}).get(
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


@_decorators.codegen(store, "nki")
def _(state: CodegenState) -> None:
    from .._compiler.ast_extension import create
    from .._compiler.ast_extension import expr_from_string
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.host_function import HostFunction

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
            from .tile_ops import tile_id as _tile_id_fn
            try:
                from .tile_ops import tile_begin as _tile_begin_fn
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
                        _sbuf_shapes_st = getattr(device_fn, "_nki_sbuf_shapes", {})
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
            if (
                tensor_dim_idx == 0
                and tensor.dim() == 3
                and isinstance(sub_val, torch.Tensor)
                and len(subscript) >= 3
            ):
                # Look ahead: next subscript must be scalar, then slice
                _remaining_subs = [s for j, s in enumerate(subscript) if j > i and s is not None]
                if len(_remaining_subs) >= 2:
                    _next_sub = _remaining_subs[0]
                    _next_next_sub = _remaining_subs[1]
                    _next_is_scalar = not isinstance(_next_sub, (slice, torch.Tensor))
                    _next_next_is_slice = isinstance(_next_next_sub, slice)
                    if _next_is_scalar and _next_next_is_slice and isinstance(fx_node_i, torch.fx.Node):
                        import sympy as _sp_store3d
                        _bs_subs_store3d: dict[_sp_store3d.Symbol, int] = {}
                        for _bid in range(len(env.block_sizes)):
                            _bs = env.block_sizes[_bid]
                            _bs_subs_store3d[_bs.symbol()] = int(_bs.from_config_assert(state.config))
                        h_size = tensor.size(1)
                        d_size = tensor.size(2)
                        h_int = int(h_size._sympy_().subs(_bs_subs_store3d)) if isinstance(h_size, torch.SymInt) else int(h_size)
                        d_int = int(d_size._sympy_().subs(_bs_subs_store3d)) if isinstance(d_size, torch.SymInt) else int(d_size)
                        # Get scalar head expression
                        _head_expr_s: str | None = None
                        # Find the FX node for the next subscript
                        _next_fx_idx = i + 1
                        while _next_fx_idx < len(subscript) and subscript[_next_fx_idx] is None:
                            _next_fx_idx += 1
                        _next_fx = fx_subscript[_next_fx_idx] if fx_subscript and _next_fx_idx < len(fx_subscript) else None
                        if isinstance(_next_fx, torch.fx.Node):
                            _next_ast = state.codegen.ast_for_fx_node(_next_fx)
                            if isinstance(_next_ast, ast.AST):
                                _head_expr_s = ast.unparse(_next_ast)
                        if _head_expr_s is None and isinstance(_next_sub, (int, bool)):
                            _head_expr_s = str(int(_next_sub))
                        elif _head_expr_s is None and isinstance(_next_sub, torch.SymInt):
                            _head_expr_s = str(int(_next_sub._sympy_().subs(_bs_subs_store3d)))
                        # Get row-gather sentinel for vec index
                        _p_count_s = None
                        # Determine partition_dim from the value being stored
                        val_name = ast.unparse(value) if isinstance(value, ast.AST) else str(value)
                        val_shape = device_fn._nki_sbuf_shapes.get(val_name)
                        if val_shape and len(val_shape) >= 1:
                            _p_count_s = val_shape[0]
                        _row_scatter_3d = _nki_row_index_gather(fx_node_i, state, _p_count_s)
                        if _row_scatter_3d is not None and _head_expr_s is not None:
                            if isinstance(_row_scatter_3d, IndirectAP):
                                _vec_var_s = _row_scatter_3d.vec_var
                                _flat_var_s = device_fn.new_var("_3d_flat_idx_store", dce=True)
                                device_fn._nki_sbuf_shapes[_flat_var_s] = [_p_count_s, 1]
                                device_fn._nki_sbuf_dtypes[_flat_var_s] = "nl.uint32"
                                state.codegen.add_statement(statement_from_string(
                                    f"{_flat_var_s} = nl.ndarray([{_p_count_s}, 1], nl.uint32, buffer=nl.sbuf)"
                                ))
                                state.codegen.add_statement(statement_from_string(
                                    f"nisa.tensor_copy(dst={_flat_var_s}, src={_vec_var_s})"
                                ))
                                if h_int != 1:
                                    state.codegen.add_statement(statement_from_string(
                                        f"nisa.tensor_scalar(dst={_flat_var_s}, data={_flat_var_s}, op0=nl.multiply, operand0={h_int}, op1=None)"
                                    ))
                                state.codegen.add_statement(statement_from_string(
                                    f"nisa.tensor_scalar(dst={_flat_var_s}, data={_flat_var_s}, op0=nl.add, operand0={_head_expr_s}, op1=None)"
                                ))
                                slice_parts = [IndirectAP(vec_var=_flat_var_s, p_count=_p_count_s, pattern=None), f"0:{d_int}"]
                                is_scalar_dim_s = [False, False]
                                total_rows_s = int(tensor.numel()) // d_int
                                hbm_dim_size_strs_s = [str(total_rows_s), str(d_int)]
                                partition_offset_var = None
                                # Skip remaining dims
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

                from .._compiler.ast_extension import statement_from_string as _sfs_ss
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
                    from .._compiler.ast_extension import expr_from_string as _efrom
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
                                from .._compiler.ast_extension import create as _ast_create
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

