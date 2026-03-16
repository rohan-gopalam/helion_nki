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


@_decorators.codegen(load, "nki")
def _(state: CodegenState) -> ast.AST:
    from .._compiler.ast_extension import create
    from .._compiler.ast_extension import statement_from_string

    NKI_PARTITION_MAX = 128

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    name = state.device_function.tensor_arg(tensor).name
    device_fn = state.device_function
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
        return int(s._sympy_().subs(_bs_subs))

    # NKI fix: when a SymInt subscript lost its BlockSizeOrigin (concretized by
    # inductor), compute_shape eliminates the dimension.  Detect this case:
    # if the tensor is 2D, subscript has a SymInt for dim 0, and output_shape
    # has only 1 element, the first dim was incorrectly eliminated.  Re-insert
    # it using the active outer tile loop's block_size.
    if (
        tensor.dim() == 2
        and len(output_shape) == 1
        and len(subscript) >= 2
        and isinstance(subscript[0], torch.SymInt)
    ):
        # Find the tile loop block_id for dim 0 from active device loops
        for bid in state.codegen.active_device_loops.keys():
            if bid < len(env.block_sizes) and not env.block_sizes[bid].reduction:
                bs_val = int(env.block_sizes[bid].from_config_assert(state.config))
                # This is the outer tile loop; re-insert its block_size as dim 0
                output_shape = [env.block_sizes[bid].var, *output_shape]
                break

    partition_dim = _resolve_dim(output_shape[0])
    free_dims = [_resolve_dim(s) for s in output_shape[1:]]
    dtype_str = backend.dtype_str(tensor.dtype)

    # Build slices from the subscript list - process only actual tensor dimensions
    fx_subscript = (
        state.fx_node.args[1]
        if state.fx_node is not None and len(state.fx_node.args) >= 2
        else None
    )
    slice_parts: list[str] = []
    partition_offset_var: str | None = None
    _used_block_ids_load: set[int] = set()

    # Track which subscript elements map to tensor dimensions
    tensor_dim_idx = 0
    for i, sub_val in enumerate(subscript):
        # Skip None subscripts (newaxis/broadcasting dimensions)
        if sub_val is None:
            continue

        # Stop if we've processed all tensor dimensions
        if tensor_dim_idx >= tensor.dim():
            break

        block_id = None
        # Try to get block_id from FX node metadata
        if fx_subscript is not None and i < len(fx_subscript):
            fx_node_i = fx_subscript[i]
            if isinstance(fx_node_i, torch.fx.Node):
                val = fx_node_i.meta.get("val")
                if val is not None:
                    block_id = env.get_block_id(val)

        # NKI fix: when get_block_id fails for a SymInt subscript (concretized
        # hint value instead of symbolic), try to match the active tile loop by
        # checking which non-reduction block_id iterates over this dimension.
        if block_id is None and isinstance(sub_val, torch.SymInt):
            for bid in state.codegen.active_device_loops.keys():
                if bid < len(env.block_sizes) and not env.block_sizes[bid].reduction:
                    if bid not in _used_block_ids_load:
                        block_id = bid
                    break

        # For slice(None), check if this is a reduction dimension
        if block_id is None and isinstance(sub_val, slice):
            # Check if this dimension matches a reduction block size
            dim_size = tensor.size(tensor_dim_idx)
            for bid in state.codegen.active_device_loops.keys():
                if bid < len(env.block_sizes):
                    block_size = env.block_sizes[bid].from_config_assert(state.config)
                    # Skip trivial block_size=1 (matches everything via divisibility)
                    if block_size <= 1:
                        continue
                    # Match by size - if dimension is divisible by block size, it's likely the reduction
                    if isinstance(dim_size, int) and block_size > 0 and dim_size % block_size == 0:
                        # Match for: dim 1 of 2D tensors, or dim 0 of 1D tensors
                        if (tensor_dim_idx == 1 and tensor.dim() == 2) or (
                            tensor_dim_idx == 0 and tensor.dim() == 1
                        ):
                            block_id = bid
                            break

        # Build the slice for this tensor dimension
        if block_id is not None:
            _used_block_ids_load.add(block_id)
        if block_id is not None and block_id in state.codegen.active_device_loops:
            offset_var = state.codegen.offset_var(block_id)
            block_size = env.block_sizes[block_id].from_config_assert(state.config)
            if tensor_dim_idx == 0:
                partition_offset_var = offset_var
            slice_parts.append(f"{offset_var}:{offset_var}+{int(block_size)}")
        else:
            # Fixed slice for this dimension
            size_i = tensor.size(tensor_dim_idx)
            size_str = (
                state.sympy_expr(size_i._sympy_())
                if isinstance(size_i, torch.SymInt)
                else str(size_i)
            )
            slice_parts.append(f"0:{size_str}")

        # Advance to next tensor dimension
        tensor_dim_idx += 1

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
                # end_str is like "offset_0+128" or "offset_0 + 128"
                plus_idx = end_str.find("+")
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
                flat_offset_parts.append(f"{off} * {multiplier}")
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

        # Squeeze output_shape to 2D: combine leading dims
        flat_partition = 1
        for dim_i in range(tensor.dim() - 1):
            flat_partition *= _resolve_dim(output_shape[dim_i]) if dim_i < len(output_shape) else 1
        if len(output_shape) > 2:
            output_shape = [flat_partition] + [output_shape[-1]]
        partition_dim = _resolve_dim(output_shape[0])
        free_dims = [_resolve_dim(s) for s in output_shape[1:]]

    if partition_dim > NKI_PARTITION_MAX and free_dims in ([], [1]):
        # 1D / [1, N] path: tensor is 1D (reshaped to [1, N] at kernel entry).
        # Allocate a single [1, partition_dim] SBUF and one DMA copy.
        device_fn._nki_sbuf_shapes[sbuf_name] = [1, partition_dim]
        state.codegen.add_statement(
            statement_from_string(
                f"{sbuf_name} = nl.ndarray([1, {partition_dim}], "
                f"{dtype_str}, buffer=nl.sbuf)"
            )
        )
        # Build the source slice: name[0:1, orig_slice] to match the [1, N] reshape
        orig_slice = slice_parts[0] if slice_parts else f"0:{partition_dim}"
        hbm_src_1d = f"{name}[0:1, {orig_slice}]"
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.dma_copy(dst={sbuf_name}, src={hbm_src_1d})"
            )
        )
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
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={tile_var}, src={name}[{part_slice_str}])"
                )
            )
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
        slice_str = ", ".join(slice_parts)
        # For 1D tensors reshaped to [1, N] at kernel entry, the DMA source
        # needs 2D indexing. Prepend "0:1" for the partition dimension.
        if tensor.dim() == 1 and len(slice_parts) == 1:
            slice_str = f"0:1, {slice_str}"
        hbm_src_expr = f"{name}[{slice_str}]"
        state.codegen.add_statement(
            statement_from_string(
                f"nisa.dma_copy(dst={sbuf_name}, src={hbm_src_expr})"
            )
        )
        # Track HBM source for partition-broadcast codegen in tensor_tensor
        if not hasattr(device_fn, "_nki_hbm_sources"):
            device_fn._nki_hbm_sources = {}
        device_fn._nki_hbm_sources[sbuf_name] = hbm_src_expr
    return expr_from_string(sbuf_name)


@_decorators.codegen(store, "nki")
def _(state: CodegenState) -> None:
    from .._compiler.ast_extension import create
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.host_function import HostFunction

    NKI_PARTITION_MAX = 128

    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    value = state.ast_arg(2)
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
    partition_offset_var: str | None = None
    _used_block_ids_store: set[int] = set()
    tensor_dim_idx = 0
    for i, sub_val in enumerate(subscript):
        if sub_val is None:
            continue
        if tensor_dim_idx >= tensor.dim():
            break
        block_id = None
        if fx_subscript is not None and i < len(fx_subscript):
            fx_node_i = fx_subscript[i]
            if isinstance(fx_node_i, torch.fx.Node):
                val = fx_node_i.meta.get("val")
                if val is not None:
                    block_id = env.get_block_id(val)

        # NKI fix: fallback for concretized SymInt subscripts
        if block_id is None and isinstance(sub_val, torch.SymInt):
            for bid in state.codegen.active_device_loops.keys():
                if bid < len(env.block_sizes) and not env.block_sizes[bid].reduction:
                    if bid not in _used_block_ids_store:
                        block_id = bid
                        break

        # NKI fix: fallback for slice subscripts — match reduction blocks (skip block_size=1)
        if block_id is None and isinstance(sub_val, slice):
            dim_size = tensor.size(tensor_dim_idx) if tensor_dim_idx < tensor.dim() else None
            if dim_size is not None:
                for bid in state.codegen.active_device_loops.keys():
                    if bid < len(env.block_sizes):
                        block_size = env.block_sizes[bid].from_config_assert(state.config)
                        if block_size <= 1:
                            continue
                        if isinstance(dim_size, int) and block_size > 0 and dim_size % block_size == 0:
                            if (tensor_dim_idx == 1 and tensor.dim() == 2) or (
                                tensor_dim_idx == 0 and tensor.dim() == 1
                            ):
                                block_id = bid
                                break

        if block_id is not None:
            _used_block_ids_store.add(block_id)
        if block_id is not None and block_id in state.codegen.active_device_loops:
            offset_var = state.codegen.offset_var(block_id)
            block_size = env.block_sizes[block_id].from_config_assert(state.config)
            # For 1D tensors, don't set partition_offset_var here.
            # The partition vs free layout decision is deferred to HBM
            # allocation based on the value's SBUF shape.
            if tensor_dim_idx == 0 and tensor.dim() != 1:
                partition_offset_var = offset_var
            slice_parts.append(f"{offset_var} : {offset_var}+{int(block_size)}")
        else:
            size_i = tensor.size(tensor_dim_idx) if tensor_dim_idx < tensor.dim() else sub_val
            size_str = (
                state.sympy_expr(size_i._sympy_())
                if isinstance(size_i, torch.SymInt)
                else str(size_i)
            )
            slice_parts.append(f"0 : {size_str}")
        tensor_dim_idx += 1

    # 3D+ tensors: reshaped to 2D at kernel entry.
    # Combine leading slice_parts into one flat partition slice.
    if tensor.dim() > 2 and len(slice_parts) > 2:
        import sympy as _sympy_store

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
                off_str, end_str = sp.split(":")
                off_str = off_str.strip()
                leading_offsets_s.append(off_str)
                plus_idx = end_str.find("+")
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
                flat_offset_parts_s.append(f"{off} * {multiplier}")
            else:
                flat_offset_parts_s.append(off)
        flat_offset_s = " + ".join(flat_offset_parts_s)
        flat_block_size_s = 1
        for bs in leading_block_sizes_s:
            flat_block_size_s *= bs

        flat_slice = f"({flat_offset_s}) : ({flat_offset_s}) + {flat_block_size_s}"
        slice_parts = [flat_slice] + [slice_parts[-1]]
        partition_offset_var = f"({flat_offset_s})"

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

                # NKI requires 2D buffers (partition, free). For 1D output
                # tensors, choose layout based on the value's SBUF shape:
                # - [P, 1] value (from reduction): use [N, 1] HBM (partition-axis)
                # - [1, F] value (element-wise): use [1, N] HBM (free-axis)
                host_reshape = None
                # Check if we're storing to a view of a base tensor
                base_shapes = getattr(device_fn, "_nki_base_tensor_shapes", {})
                base_shape = base_shapes.get(tensor_id)
                if tensor.dim() == 1:
                    val_sbuf_shape = device_fn._nki_sbuf_shapes.get(value_name)
                    use_partition_layout = (
                        val_sbuf_shape is not None
                        and len(val_sbuf_shape) >= 2
                        and val_sbuf_shape[0] > 1
                        and val_sbuf_shape[1] == 1
                    )
                    if use_partition_layout:
                        shape_str = f"{shape_parts[0]}, 1"
                    else:
                        shape_str = f"1, {shape_parts[0]}"
                    if base_shape is not None:
                        # Reshape NKI result to the base tensor's shape
                        host_reshape = f"[{', '.join(str(d) for d in base_shape)}]"
                    else:
                        host_reshape = f"[{shape_parts[0]}]"
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
        if numel == 1:
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={name}, src={{value}})", value=value
                )
            )
        else:
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={name}[{slice_str}], src={{value}})", value=value
                )
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
