from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators
from .stack_tensor import StackTensor

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = ["load", "store"]

# Map short config names to full Triton API names for eviction policies
_EVICTION_POLICY_MAP = {
    "": None,
    "first": "evict_first",
    "last": "evict_last",
}


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
    index = Tile._tiles_to_sizes(index)

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
        device_fn.device_store_index += 1
        # Use the shared memory op index for indexing strategy
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_store(state, tensor, [*subscript], value, extra_mask)
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    from .._compiler.ast_extension import statement_from_string

    tensor = state.proxy_arg(0)
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    name = state.device_function.tensor_arg(tensor).name
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    state.codegen.add_statement(
        statement_from_string(f"{name}[...] = {{value}}", value=value)
    )


def _cute_index_exprs(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...] | None = None,
    tensor: torch.Tensor | None = None,
) -> list[str]:
    env = CompileEnvironment.current()
    result = []
    for pos, idx in enumerate(subscript):
        ast_idx = None
        if ast_subscript is not None:
            ast_idx = ast_subscript[pos]
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
            if block_id is not None:
                result.append(state.codegen.index_var(block_id))
            else:
                result.append(state.sympy_expr(idx._sympy_()))
        elif isinstance(idx, int):
            result.append(str(idx))
        elif isinstance(idx, torch.Tensor):
            if not isinstance(ast_idx, ast.AST):
                raise exc.BackendUnsupported(
                    "cute", f"tensor index without AST at position {pos}"
                )
            lifted = state.codegen.lift(ast_idx, dce=True, prefix="index")
            result.append(lifted.id)
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor is None:
                raise exc.BackendUnsupported("cute", "slice indexing without tensor")
            block_id = env.resolve_block_id(tensor.shape[pos])
            if block_id is None:
                raise exc.BackendUnsupported(
                    "cute", f"slice indexing on non-block dimension {pos}"
                )
            result.append(state.codegen.index_var(block_id))
        elif idx is None:
            raise exc.BackendUnsupported("cute", "None indexing")
        else:
            raise exc.BackendUnsupported("cute", f"index type: {type(idx)}")
    return result


def _cute_combined_mask(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    tensor: torch.Tensor | None = None,
) -> str | None:
    env = CompileEnvironment.current()
    terms: list[str] = []

    if extra_mask is not None:
        terms.append(state.codegen.lift(extra_mask, dce=True, prefix="mask").id)

    seen: set[int] = set()
    for pos, idx in enumerate(subscript):
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif isinstance(idx, slice) and idx == slice(None) and tensor is not None:
            block_id = env.resolve_block_id(tensor.shape[pos])
        else:
            continue
        if block_id is None or block_id in seen:
            continue
        seen.add(block_id)
        if (mask_var := state.codegen.mask_var(block_id)) is not None:
            if mask_var not in terms:
                terms.append(mask_var)

    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


@_decorators.codegen(store, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, tuple):
        raise exc.BackendUnsupported("cute", "stack tensor store")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"store target type: {type(tensor)}")

    tensor_name = state.device_function.tensor_arg(tensor).name
    index_exprs = _cute_index_exprs(state, subscript, ast_subscript, tensor=tensor)
    index_tuple = (
        f"({index_exprs[0]},)"
        if len(index_exprs) == 1
        else f"({', '.join(index_exprs)})"
    )
    assign_expr = expr_from_string(
        f"{tensor_name}.__setitem__({index_tuple}, {{value}})", value=value
    )

    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if mask_expr is None:
        return assign_expr
    return expr_from_string(
        f"({tensor_name}.__setitem__({index_tuple}, {{value}}) if {mask_expr} else None)",
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

    index = Tile._tiles_to_sizes(index)
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
        return env.new_index_result(tensor, target_shape)
    if isinstance(tensor, tuple):
        tensor_like, dev_ptrs = tensor
        assert isinstance(tensor_like, torch.Tensor)
        assert isinstance(dev_ptrs, torch.Tensor)
        tensor_shape = SubscriptIndexing.compute_shape(tensor_like, index)
        target_shape = list(dev_ptrs.size()) + tensor_shape
        return tensor_like.new_empty(target_shape)
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


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

    if isinstance(tensor, torch.Tensor):
        # If tile_index(...) is being broadcast-only indexed
        from ..language import tile_index

        tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
        if (
            isinstance(tensor_node, torch.fx.Node)
            and tensor_node.op == "call_function"
            and tensor_node.target == tile_index
        ):
            # tile.index tensors are not real memory accesses; materialize the
            # block index variable with the requested broadcast/reshape.
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

        # Use the shared memory op index for indexing strategy
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy
        )
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state, tensor, dev_ptrs_ast, [*subscript], extra_mask, eviction_policy
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    name = state.device_function.tensor_arg(tensor).name
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_load_index += 1
    device_fn.device_memory_op_index += 1
    return expr_from_string(f"{name}[...]")


@_decorators.codegen(load, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, tuple):
        raise exc.BackendUnsupported("cute", "stack tensor load")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"load tensor type: {type(tensor)}")

    tensor_name = state.device_function.tensor_arg(tensor).name
    index_exprs = _cute_index_exprs(state, subscript, ast_subscript, tensor=tensor)
    load_expr = f"{tensor_name}[{', '.join(index_exprs)}]"
    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if mask_expr is None:
        return expr_from_string(load_expr)
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")


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
    is_add = (
        "add.Tensor" in target_name
        or "add.Scalar" in target_name
        or target is torch.ops.aten.add.Tensor
        or target is torch.ops.aten.add.Scalar
    )
    is_sub = (
        "sub.Tensor" in target_name
        or "sub.Scalar" in target_name
        or target is torch.ops.aten.sub.Tensor
        or target is torch.ops.aten.sub.Scalar
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
                _rhs_expr = ast.unparse(_rhs_ast)
        if _rhs_expr is not None:
            if is_sub:
                start = f"{offset_var} - {_rhs_expr}"
            else:  # is_add
                start = f"{offset_var} + {_rhs_expr}"
            return f"{start}:{start}+{block_size}"

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
                _lhs_expr = ast.unparse(_lhs_ast)
        if _lhs_expr is not None:
            if is_add:
                start = f"{_lhs_expr} + {offset_var}"
                return f"{start}:{start}+{block_size}"
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
        # Transpose requires float; cast if int
        if _dt in ("nl.int32", "nl.int16", "nl.uint32", "nl.uint16"):
            # Cast to float32 for transpose
            cast_in = device_fn.new_var("_ig_cast_in", dce=True)
            state.codegen.add_statement(statement_from_string(
                f"{cast_in} = nl.ndarray([1, {P}], nl.float32, buffer=nl.sbuf)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.memset({cast_in}, value=0)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.tensor_tensor(dst={cast_in}, data1={cast_in}, data2={lhs_name}, op=nl.add)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"{tr_psum} = nl.ndarray([{P}, 1], nl.float32, buffer=nl.psum)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.nc_transpose(dst={tr_psum}, data={cast_in})"
            ))
            state.codegen.add_statement(statement_from_string(
                f"{tr_sbuf} = nl.ndarray([{P}, 1], nl.uint32, buffer=nl.sbuf)"
            ))
            state.codegen.add_statement(statement_from_string(
                f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
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
                f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
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
    return f"__AP_VEC_OFFSET__{vec_offset_var}__{pattern}__"


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
            statement_from_string(f"nisa.memset({cast_in}, value=0)")
        )
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.tensor_tensor(dst={cast_in}, data1={cast_in}, "
                f"data2={name}, op=nl.add)"
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
        statement_from_string(f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})")
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
                    return f"__AP_ROW_GATHER__{vec_offset_var}__"

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
                        return f"__AP_ROW_GATHER__{vec_offset_var}__"

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
    return f"__AP_ROW_GATHER__{vec_offset_var}__"


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
                sentinel_prefix = "__AP_ROW_GATHER__"
                if _row_gather_3d.startswith(sentinel_prefix):
                    vec_var = _row_gather_3d[len(sentinel_prefix):].rstrip("_")
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
                    # Now emit the 2D gather: [__AP_ROW_GATHER__flat_var__, 0:D]
                    slice_parts = [f"__AP_ROW_GATHER__{flat_var}__", f"0:{d_int}"]
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
            elif block_id in _dyn_loops:
                _counter = _dyn_loops[block_id]["counter"]
                slice_parts.append(f"__DYN_AP__{_counter}__{int(block_size)}")
            elif (
                tdi == 0
                and tensor.dim() == 2
                and isinstance(_sub_val_load, torch.Tensor)
            ):
                # Check if the subscript is a non-contiguous gather (indirect load).
                # When block_id was found via size match but the subscript is a gather
                # (e.g. sorted_to_orig_token_idx[indices] or torch.where result),
                # use the row gather (.ap() with vector_offset) mechanism.
                row_gather = _nki_row_index_gather(fx_node_tdi_check, state, partition_dim)
                if row_gather is not None:
                    if tdi == 0:
                        partition_offset_var = row_gather.split(":", 1)[0].strip() if ":" in row_gather else row_gather
                    slice_parts.append(row_gather)
                    is_scalar_dim.append(False)
                    continue
                slice_parts.append(f"{offset_var}:{offset_var}+{int(block_size)}")
            else:
                slice_parts.append(f"{offset_var}:{offset_var}+{int(block_size)}")
            is_scalar_dim.append(False)
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
                                    slice_parts.append(f"__DYN_AP__{_counter_c}__{_slice_len}")
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
            return expr_from_string(
                _emit_flat_masked(sbuf_name, [p_count, f_count])
            )

        def _try_emit_flat_gather_sum_dim1(index_node: torch.fx.Node) -> ast.AST | None:
            index_val = index_node.meta.get("val")
            if not isinstance(index_val, torch.Tensor) or index_val.ndim != 3:
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
                _walk_result = _walk_chain(state.fx_node)
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
                return None

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

        flat_gather_2d = _try_emit_flat_gather_2d(fx_subscript[0])
        if flat_gather_2d is not None:
            return flat_gather_2d

        fused_gather = _try_emit_flat_gather_sum_dim1(fx_subscript[0])
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

        # Replace leading slice_parts with one combined slice
        flat_slice = f"({flat_offset}):({flat_offset}) + {flat_block_size}"
        slice_parts = [flat_slice] + [slice_parts[-1]]

        # Fix partition_offset_var to point to the flat offset expression
        partition_offset_var = f"({flat_offset})"

        # Squeeze output_shape to 2D using flat_block_size from the slice computation.
        # flat_block_size is derived from the actual DMA slice ranges and is always
        # correct. output_shape may have extra leading trivial dimensions (block_size=1)
        # from outer loop tiles, making it unreliable for partition_dim computation.
        partition_dim = flat_block_size
        free_dims = [_resolve_dim(output_shape[-1])]
        output_shape = [partition_dim] + free_dims
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
            if _p.startswith("__AP_ROW_GATHER__"):
                vec_offset = _p[len("__AP_ROW_GATHER__") :].rstrip("_")
                vec_shape = device_fn._nki_sbuf_shapes.get(vec_offset)
                if vec_shape is None or len(vec_shape) < 1:
                    raise NotImplementedError(
                        f"Unknown row-gather vector offset shape for {vec_offset}"
                    )
                p_count = int(vec_shape[0])
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
            if _p.startswith("__AP_VEC_OFFSET__"):
                _rest = _p[len("__AP_VEC_OFFSET__"):]
                # Format: var__pattern__
                parts_split = _rest.rsplit("__", 2)
                # Actually encoding: var__PATTERN__ — only 2 __ separators
                # but pattern may contain __ too. Use a safer split.
                # Split at first __ only (var)
                _vp = _rest.split("__", 1)
                if len(_vp) == 2:
                    var_name, rest2 = _vp
                    # rest2 ends with "__", strip
                    pattern_str = rest2.rstrip("_")
                else:
                    continue
                return f"{name_str}.ap(pattern={pattern_str}, vector_offset={var_name}, indirect_dim=0)"
        has_dyn = any(p.startswith("__DYN_AP__") for p in parts)
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
            if p.startswith("__DYN_AP__"):
                assert dyn_dim_idx is None, "multiple dynamic dims not yet supported"
                dyn_dim_idx = i
                # Format: __DYN_AP__counter__size
                _, counter_name, size_str = p.split("__", 2)[1:] + [""]
                # Actually format: f"__DYN_AP__{_counter}__{int(block_size)}"
                _rest = p[len("__DYN_AP__"):]
                counter_name, size_str = _rest.rsplit("__", 1)
                dyn_counter = counter_name
                dyn_size = int(size_str)
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
            if part.startswith(("__DYN_AP__", "__AP_VEC_OFFSET__")):
                continue
            if ":" not in part:
                continue
            start, end = part.split(":", 1)
            start = start.strip()
            end = end.strip()
            if not start or not end:
                continue
            dim_size_str = hbm_dim_size_strs[dim_idx]
            checks.append(f"({start}) >= 0")
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
                    # Check tile_offset + block_size <= stride (inner bound)
                    checks.append(f"({_tile_offset}) + {_block_size} <= {_stride_str}")
        if not checks:
            return None
        return " and ".join(checks)

    def _slice_info(part: str, dim_idx: int) -> tuple[str, str, int, str] | None:
        if dim_idx >= len(hbm_dim_size_strs) or ":" not in part:
            return None
        if part.startswith(("__DYN_AP__", "__AP_VEC_OFFSET__", "__AP_ROW_GATHER__")):
            return None
        start, end = part.split(":", 1)
        start = start.strip()
        end = end.strip()
        if not start or not end:
            return None
        count: int | None = None
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
        cases: list[ast.If] = []
        infos = [_slice_info(part, i) for i, part in enumerate(parts)]
        if any(info is None for info in infos):
            return cases
        for tail_dim, tail_info in enumerate(infos):
            assert tail_info is not None
            start, end, count, dim_size_str = tail_info
            checks = [f"({start}) >= 0", f"({start}) < {dim_size_str}", f"({end}) > {dim_size_str}"]
            dst_parts: list[str] = []
            src_parts: list[str] = []
            for dim_idx, info in enumerate(infos):
                assert info is not None
                dim_start, dim_end, dim_count, dim_size = info
                if dim_idx == tail_dim:
                    src_parts.append(f"{dim_start}:{dim_size}")
                    dst_parts.append(f"0:{dim_size} - ({dim_start})")
                else:
                    checks.append(f"({dim_start}) >= 0")
                    checks.append(f"({dim_end}) <= {dim_size}")
                    src_parts.append(parts[dim_idx])
                    dst_parts.append(f"0:{dim_count}")
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
                if ":" in _part0 and not _part0.startswith(("__DYN_AP__", "__AP_VEC_OFFSET__")):
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
            if orig_slice.startswith(("__DYN_AP__", "__AP_VEC_OFFSET__")):
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
            part_slice_str = ", ".join(part_slice_parts)
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
            p.startswith(("__DYN_AP__", "__AP_VEC_OFFSET__", "__AP_ROW_GATHER__"))
            for p in slice_parts
        )
        if _has_dyn_2d:
            if tensor.dim() == 1 and len(slice_parts) == 1:
                hbm_src_expr = _build_hbm_src(name, ["0:1", slice_parts[0]])
            else:
                hbm_src_expr = _build_hbm_src(name, slice_parts)
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
                slice_parts.append(f"__DYN_AP__{_counter_st}__{int(block_size)}")
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
                            sentinel_prefix = "__AP_ROW_GATHER__"
                            if _row_scatter_3d.startswith(sentinel_prefix):
                                _vec_var_s = _row_scatter_3d[len(sentinel_prefix):].rstrip("_")
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
                                slice_parts = [f"__AP_ROW_GATHER__{_flat_var_s}__", f"0:{d_int}"]
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
            if ":" in sp:
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

        flat_slice = f"({flat_offset_s}) : ({flat_offset_s}) + {flat_block_size_s}"
        slice_parts = [flat_slice] + [slice_parts[-1]]
        partition_offset_var = f"({flat_offset_s})"
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
                ret_buf_name = device_fn.new_var("nki_return_buf")
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
            part_slice_str = ", ".join(part_slice_parts)
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

        slice_str = ", ".join(adjusted_slice_parts)
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
            if dim_idx >= len(hbm_dim_size_strs_s) or ":" not in part:
                return None
            if part.startswith(("__DYN_AP__", "__AP_ROW_GATHER__")):
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
            checks: list[str] = []
            for dim_idx, part in enumerate(parts):
                info = _store_slice_info(part, dim_idx)
                if info is None:
                    return None
                start, end, _width, dim_size_str = info
                checks.append(f"({start}) >= 0")
                checks.append(f"({end}) <= {dim_size_str}")
            return " and ".join(checks) if checks else None

        def _single_tail_store_cases(
            dst_base: str, parts: list[str], src_name: str
        ) -> list[ast.If]:
            cases: list[ast.If] = []
            infos = [_store_slice_info(part, i) for i, part in enumerate(parts)]
            if any(info is None for info in infos):
                return cases
            for tail_dim, tail_info in enumerate(infos):
                assert tail_info is not None
                start, end, _width, dim_size_str = tail_info
                checks = [
                    f"({start}) >= 0",
                    f"({start}) < {dim_size_str}",
                    f"({end}) > {dim_size_str}",
                ]
                dst_parts: list[str] = []
                src_parts: list[str] = []
                for dim_idx, info in enumerate(infos):
                    assert info is not None
                    dim_start, dim_end, dim_width, dim_size = info
                    if dim_idx == tail_dim:
                        dst_parts.append(f"{dim_start}:{dim_size}")
                        src_parts.append(f"0:{dim_size} - ({dim_start})")
                    else:
                        checks.append(f"({dim_start}) >= 0")
                        checks.append(f"({dim_end}) <= {dim_size}")
                        dst_parts.append(parts[dim_idx])
                        src_parts.append(f"0:{dim_width}")
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
            _has_dyn_store = "__DYN_AP__" in slice_str
            _has_row_store = "__AP_ROW_GATHER__" in slice_str
            if _has_row_store:
                _dst_parts = [p.strip() for p in slice_str.split(",")]
                _row_part = next(
                    p for p in _dst_parts if p.startswith("__AP_ROW_GATHER__")
                )
                _vec_offset = _row_part[len("__AP_ROW_GATHER__") :].rstrip("_")
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
                _free_part = _dst_parts[1] if len(_dst_parts) > 1 else f"0:{_f_total}"
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
                _dst_parts = [p.strip() for p in slice_str.split(",")]
                # Inline the same builder logic as _build_hbm_src (can't use it
                # here because it closes over `tensor` / `_resolve_dim` in load).
                _dyn_idx = None
                _dyn_counter = None
                _dyn_size = 0
                for _i, _p in enumerate(_dst_parts):
                    if _p.startswith("__DYN_AP__"):
                        assert _dyn_idx is None, "multi-dyn store not supported"
                        _dyn_idx = _i
                        _rest = _p[len("__DYN_AP__"):]
                        _dyn_counter, _size_str = _rest.rsplit("__", 1)
                        _dyn_size = int(_size_str)
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
                    if _dyn_idx == 1:
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
        # pyrefly: ignore [bad-argument-type]
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
