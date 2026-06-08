from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["arange", "full", "zeros"]


def zeros(
    shape: list[object],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Return a device-tensor filled with zeros.

    Equivalent to ``hl.full(shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype)``.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For output tensor creation, use ``torch.zeros()`` with proper device placement.

    Args:
        shape: A list of sizes (or tile indices which are implicitly converted to sizes)
        dtype: Data type of the tensor (default: torch.float32)
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with zeros

    Examples:

        .. code-block:: python

            @helion.kernel
            def process_kernel(input: torch.Tensor) -> torch.Tensor:
                result = torch.empty_like(input)

                for tile in hl.tile(input.size(0)):
                    buffer = hl.zeros([tile], dtype=input.dtype)  # Local buffer
                    buffer += input[tile]  # Add input values to buffer
                    result[tile] = buffer

                return result

    See Also:
        - :func:`~helion.language.full`: For filling with arbitrary values
        - :func:`~helion.language.arange`: For creating sequences
    """
    return full(
        shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype, device=device
    )


@_decorators.api(tiles_as_sizes=True)
def full(
    shape: list[object],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a device-tensor filled with a specified value.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For output tensor creation, use ``torch.full()`` with proper device placement.

    Args:
        shape: A list of sizes (or tile indices which are implicitly converted to sizes)
        value: The value to fill the tensor with
        dtype: The data type of the tensor (default: torch.float32)
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with value

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(input: torch.Tensor) -> torch.Tensor:
                result = torch.empty_like(input)

                for tile in hl.tile(input.size(0)):
                    # Create local buffer filled with initial value
                    buffer = hl.full([tile], 0.0, dtype=input.dtype)
                    buffer += input[tile]  # Add input values to buffer
                    result[tile] = buffer

                return result

    See Also:
        - :func:`~helion.language.zeros`: For filling with zeros
        - :func:`~helion.language.arange`: For creating sequences
    """
    raise NotInsideKernel


@_decorators.register_fake(full)
def _full_fake(
    shape: list[int | torch.SymInt],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.codegen(full, "common")
def _full_codegen(state: CodegenState) -> ast.AST:
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)
    shape_dims = state.device_function.tile_strategy.shape_dims(fake_value.size())
    env = CompileEnvironment.current()
    backend = env.backend

    # NKI two-line pattern: ndarray then nisa.memset(dst, value=...)
    nki_memset = getattr(backend, "full_memset_stmt", None)
    if nki_memset is not None and shape_dims:
        # 2D accumulator layout normalization for NKI. ``hl.full([1, N])``
        # inside a tile loop is almost always a reduction accumulator
        # (per-row scalars); reductions with keepdim=True produce [N, 1]
        # SBUF layout. Transpose the allocation to [N, 1] so element-wise
        # ops line up without a numpy-broadcast to [N, N].
        #
        # More aggressive guard: fire when users include a REDUCTION op
        # (amax/amin/sum/mean/etc) which produces [N, 1] that we'd want
        # to combine with this accumulator. Pure [1, N] broadcast vectors
        # (e.g. bias for matmul) don't see reduction users.
        fx_node = state.fx_node
        if fx_node is not None and backend.name == "nki" and len(shape_dims) == 2:
            try:
                resolved = [int(d) for d in shape_dims]
            except (TypeError, ValueError):
                resolved = None
            if resolved is not None and resolved[0] == 1 and resolved[1] > 1:
                # Heuristic: if this hl.full is combined with a reduction
                # result (amax/amin/sum/mean) anywhere in the kernel body
                # (possibly in a nested _for_loop / _if subgraph), the
                # reduction's SBUF layout will be [N, 1]. Scan the ENTIRE
                # device_ir for reduction ops whose dim=-1 reduction target
                # produces a [N] vector that would combine with our full.
                _reduction_ops = (
                    "amax", "amin", "max_dim", "min_dim",
                    "sum.dim_intlist", "mean.dim",
                )
                transpose = False
                try:
                    from .._compiler.host_function import HostFunction as _HF
                    import sympy as _sp_r
                    device_ir = _HF.current().device_ir
                    env = CompileEnvironment.current()
                    _bs_subs: dict[_sp_r.Symbol, int] = {}
                    if state.config is not None:
                        for _bs in env.block_sizes:
                            _cfg = _bs.from_config(state.config)
                            if isinstance(_cfg, int):
                                _bs_subs[_bs.symbol()] = _cfg
                    for ginfo in device_ir.graphs:
                        g = ginfo.graph
                        for n in g.nodes:
                            if n.op != "call_function":
                                continue
                            t_name = str(getattr(n.target, "__name__", n.target)).lower()
                            if not any(r in t_name for r in _reduction_ops):
                                continue
                            # Resolve symbolic shape to concrete ints via
                            # block-size substitution so we can compare
                            # against resolved[1] (our N).
                            val = n.meta.get("val")
                            if not isinstance(val, torch.Tensor):
                                continue
                            import sympy as _sp_r
                            vs: list[int] = []
                            for d in val.shape:
                                if isinstance(d, int):
                                    vs.append(d)
                                elif isinstance(d, torch.SymInt):
                                    try:
                                        vs.append(int(d._sympy_().subs(_bs_subs)))
                                    except Exception:
                                        try:
                                            vs.append(env.size_hint(d))
                                        except Exception:
                                            vs.append(-1)
                                else:
                                    vs.append(-1)
                            # If the reduction result is already the same
                            # logical row tile as this accumulator ([1, N]),
                            # keep the accumulator row-major.  Partition-axis
                            # NKI reductions produce exactly this layout; the
                            # old blanket transpose to [N, 1] caused later
                            # binary ops to over-broadcast [N, 1] x [1, N].
                            if (
                                len(vs) == 2
                                and resolved[0] == 1
                                and vs[0] == 1
                                and vs[1] == resolved[1]
                            ):
                                continue
                            if resolved[1] in vs:
                                transpose = True
                                break
                        if transpose:
                            break
                except Exception:
                    pass

                if transpose:
                    shape_dims = [shape_dims[1], shape_dims[0]]

        var = state.device_function.new_var("_nki_full", dce=True)
        ndarray_expr = backend.full_expr(shape_dims, "0", fake_value.dtype)
        state.add_statement(statement_from_string(f"{var} = {ndarray_expr}"))
        proxy_value = state.proxy_arg(1)
        if isinstance(proxy_value, (int, float, bool)):
            value_str = state.device_function.literal_expr(proxy_value)
            state.add_statement(statement_from_string(nki_memset(var, value_str)))
        else:
            value_ast = state.ast_arg(1)
            state.add_statement(
                statement_from_string(nki_memset(var, "{val}"), val=value_ast)
            )
        # Register SBUF shape for multi-user detection on copy vars
        if hasattr(state.device_function, "_nki_sbuf_shapes"):

            def _resolve_nki_shape_dim(dim: object) -> int:
                if isinstance(dim, int):
                    return dim
                if isinstance(dim, torch.SymInt):
                    import sympy as _sp_shape

                    _shape_subs: dict[_sp_shape.Symbol, int] = {}
                    if state.config is not None:
                        for _bs_shape in env.block_sizes:
                            _cfg_shape = _bs_shape.from_config(state.config)
                            if isinstance(_cfg_shape, int):
                                _shape_subs[_bs_shape.symbol()] = _cfg_shape
                    return int(dim._sympy_().subs(_shape_subs))
                if isinstance(dim, str):
                    for (
                        block_id,
                    ), var_name in state.device_function.block_size_var_cache.items():
                        if var_name == dim:
                            return int(
                                env.block_sizes[block_id].from_config_assert(
                                    state.config
                                )
                            )
                    return int(dim)
                return int(dim)

            try:
                resolved = [_resolve_nki_shape_dim(d) for d in shape_dims]
                if len(resolved) == 1:
                    resolved.append(1)  # full_expr adds trailing 1 for 1D
                if len(resolved) >= 2:
                    state.device_function._nki_sbuf_shapes[var] = resolved
            except (TypeError, ValueError):
                pass
        return expr_from_string(var)

    # Check if the value is static (literal) or dynamic (node)
    proxy_value = state.proxy_arg(1)
    if isinstance(proxy_value, (int, float, bool)):
        # For static values, use literal_expr to preserve special representations like float('-inf')
        value_str = state.device_function.literal_expr(proxy_value)
        return expr_from_string(
            backend.full_expr(shape_dims, value_str, fake_value.dtype)
        )
    # For dynamic values, use ast_arg to get the proper AST representation
    value_ast = state.ast_arg(1)
    return expr_from_string(
        backend.full_expr(shape_dims, "{value}", fake_value.dtype), value=value_ast
    )


@_decorators.codegen(full, "pallas")
def _full_codegen_pallas(state: CodegenState) -> ast.AST:
    """Pallas codegen for hl.full / hl.zeros.

    Always lowers to a plain ``jnp.full`` bound to a fresh local, regardless
    of pallas_loop_type.  The previous emit_pipeline/fori_loop path returned
    a bare scratch ref AST, which broke any downstream arithmetic on the
    result outside the inner loop -- e.g. ``acc = hl.zeros(...); acc += x``
    emitted ``scratch + x`` and JAX raised
    ``'AbstractRef' object has no attribute '_add'`` at trace time
    (``Refs`` don't support arithmetic; only ``ref[...]`` reads).

    When the result is loop-carried, ``_setup_loop_carried_state`` allocates
    a scratch buffer at the loop boundary and copies the init value in --
    no scratch needed at the ``hl.zeros`` site itself.
    """
    return full._codegen["common"](state)  # pyrefly: ignore[missing-attribute]


@_decorators.get_masked_value(full)
def _(
    node: torch.fx.Node,
) -> float | bool | None:
    value = node.args[1]
    if isinstance(value, (int, float, bool)):
        return value
    # Return None for dynamic values (like tensor elements)
    return None


@_decorators.ref(full)
def _(
    shape: list[int | RefTile],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    processed_shape = []
    for s in shape:
        if isinstance(s, RefTile):
            processed_shape.append(s.end - s.begin)
        else:
            processed_shape.append(s)
    env = CompileEnvironment.current()
    return torch.full(
        processed_shape,
        value,
        dtype=dtype,
        device=env.device if device is None else device,
    )


def arange(
    *args: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """
    Same as `torch.arange()`, but defaults to same device as the current kernel.

    Creates a 1D tensor containing a sequence of integers in the specified range,
    automatically using the current kernel's device and index dtype.

    Args:
        args: Positional arguments passed to torch.arange(start, end, step).
        dtype: Data type of the result tensor (defaults to kernel's index dtype)
        device: Device must match the current compile environment device
        kwargs: Additional keyword arguments passed to torch.arange

    Returns:
        torch.Tensor: 1D tensor containing the sequence

    See Also:
        - :func:`~helion.language.tile_index`: For getting tile indices
        - :func:`~helion.language.zeros`: For creating zero-filled tensors
        - :func:`~helion.language.full`: For creating constant-filled tensors
    """
    env = CompileEnvironment.current()
    if dtype is None:
        dtype = env.index_dtype
    # pyrefly: ignore [no-matching-overload]
    return torch.arange(
        *args,
        **kwargs,
        dtype=dtype,
        device=env.device if device is None else device,
    )
