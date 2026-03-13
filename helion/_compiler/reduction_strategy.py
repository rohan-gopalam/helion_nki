from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._prims_common import get_computation_dtype

from ..autotuner.config_fragment import integer_power_of_two
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import find_block_size_symbols
from .host_function import HostFunction
from .inductor_lowering import install_inductor_kernel_handlers
from .tile_strategy import CompactedShape
from .tile_strategy import DeviceLoopState
from .tile_strategy import PersistentReductionState
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState


def _dtype_str(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _acc_type(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.acc_type(dtype)


class ReductionStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        mask_var: str | None,
        block_size_var: str | None,
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=[block_index],
        )
        self._mask_var = mask_var
        if block_size_var is not None:
            fn.block_size_var_cache[(block_index,)] = block_size_var

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._mask_var

    @property
    def block_index(self) -> int:
        return self.block_ids[0]

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].numel

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        return shapes

    def _reduction_thread_count(self) -> int:
        """Return threads used for this reduction on thread-aware backends."""
        return 0

    def thread_axes_used(self) -> int:
        return 1 if self._reduction_thread_count() > 0 else 0

    def thread_block_sizes(self) -> list[int]:
        count = self._reduction_thread_count()
        return [count] if count > 0 else []

    def _get_thread_axis(self) -> int:
        """Compute the thread axis index for this reduction strategy.

        Some backends place reduction strategies first so reduction threads share
        a warp. Others keep the natural strategy order.
        """
        env = CompileEnvironment.current()
        if env.backend.reduction_axis_first():
            axis = 0
            for strategy in self.fn.tile_strategy.strategies:
                if strategy is self:
                    break
                if isinstance(strategy, ReductionStrategy):
                    axis += strategy.thread_axes_used()
            return axis
        axis = 0
        for strategy in self.fn.tile_strategy.strategies:
            if strategy is self:
                break
            axis += strategy.thread_axes_used()
        return axis

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        raise NotImplementedError

    def call_reduction_function(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        backend = CompileEnvironment.current().backend
        if backend.is_indexed_reduction(reduction_type):
            index_var = self.index_var(self.block_index)
            return self.call_indexed_reduction(
                input_name,
                self.broadcast_str(index_var, fake_input, dim),
                reduction_type,
                dim,
                fake_output,
            )
        return backend.reduction_expr(
            input_name,
            reduction_type,
            dim,
            block_size_var=self.block_size_var(self.block_index),
            fake_input=fake_input,
            fake_output=fake_output,
        )

    def _index_init_expr(self, block_size_var: str, dtype: str, block_idx: int) -> str:
        env = CompileEnvironment.current()
        backend = env.backend
        size = env.block_sizes[block_idx].size
        if isinstance(size, int) and size == 0:
            return backend.reduction_index_zero_expr(dtype)
        if isinstance(size, torch.SymInt) and env.known_equal(size, 0):
            return backend.reduction_index_zero_expr(dtype)
        return backend.reduction_index_expr(
            block_size_var, dtype, block_idx, axis=self._get_thread_axis()
        )

    def call_indexed_reduction(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        fake_output: torch.Tensor,
    ) -> str:
        env = CompileEnvironment.current()
        return env.backend.argreduce_result_expr(
            input_name,
            index_value,
            reduction_type,
            dim,
            fake_output.dtype,
            block_size_var=self.block_size_var(self.block_index),
            index_dtype=env.index_dtype,
        )

    def maybe_reshape(
        self,
        expr: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        size = [*fake_input.size()]
        size.pop(dim)
        if [*fake_output.size()] == size:
            return expr
        shape = self.fn.tile_strategy.shape_str([*fake_output.size()])
        return CompileEnvironment.current().backend.reshape_expr(expr, shape)

    def broadcast_str(self, base: str, fake_input: torch.Tensor, dim: int) -> str:
        input_size = [*fake_input.size()]
        expand = self.fn.tile_strategy.expand_str(input_size, dim)
        shape = self.fn.tile_strategy.shape_str(input_size)
        return CompileEnvironment.current().backend.broadcast_to_expr(
            f"{base}{expand}", shape
        )


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        if isinstance(numel, (int, sympy.Integer)) and integer_power_of_two(int(numel)):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_RDIM_SIZE_{block_index}"),
        )
        self.offset_vars[block_index] = "0"
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                # pyrefly: ignore [no-matching-overload]
                size_hint = int(env.shape_env.size_hint(numel))
            else:
                size_hint = env.size_hint(numel)
            self._thread_count = next_power_of_2(min(size_hint, max_threads))
        else:
            self._thread_count = 0

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        backend = env.backend
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        index_var = self.index_var(block_idx)
        mask_var = self._mask_var
        block_size_var = self.block_size_var(self.block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            if isinstance(numel, sympy.Integer):
                # Static size - issue statement immediately
                stmt = statement_from_string(
                    f"{block_size_var} = {next_power_of_2(int(numel))}"
                )
                state.codegen.host_statements.append(stmt)
            else:
                # Check for block size dependencies
                block_mapping, _ = find_block_size_symbols(numel)
                if block_mapping:
                    # Defer issuing statement until block sizes are known
                    state.device_function.deferred_rdim_defs.append(
                        (block_size_var, numel)
                    )
                else:
                    # No dependencies - issue statement immediately
                    expr_str = HostFunction.current().sympy_expr(numel)
                    stmt = statement_from_string(
                        f"{block_size_var} = {backend.next_power_of_2_host_expr(expr_str)}"
                    )
                    state.codegen.host_statements.append(stmt)
        state.add_statement(
            f"{index_var} = {self._index_init_expr(block_size_var, env.index_type(), block_idx)}"
        )
        if mask_var is not None:
            state.add_statement(
                f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
            )
        # Extract end_var_name from the numel expression
        from .tile_strategy import LoopDimInfo

        end_var_name = self.fn.sympy_expr(numel)
        block_id_to_info = {
            self.block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        state.codegen.set_active_loops(
            PersistentReductionState(self, block_id_to_info=block_id_to_info)
        )

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend
        numel = env.block_sizes[self.block_index].numel
        if isinstance(numel, sympy.Integer) and numel == 0:
            default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
            assert isinstance(default, (float, int, bool))
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                backend.full_expr(shape_dims, constant_repr(default), fake_output.dtype)
            )
        expr = self.call_reduction_function(
            input_name,
            reduction_type,
            dim,
            fake_input,
            fake_output,
        )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))


class LoopedReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        block_size: int,
    ) -> None:
        env = CompileEnvironment.current()
        if env.known_multiple(env.block_sizes[block_index].numel, block_size):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_REDUCTION_BLOCK_{block_index}"),
        )
        self.offset_vars[block_index] = fn.new_var(f"roffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"rindex_{block_index}", dce=True)
        self.block_size = block_size
        assert block_size > 1
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            self._thread_count = next_power_of_2(min(block_size, max_threads))
        else:
            self._thread_count = 0

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        env = CompileEnvironment.current()
        block_index = self.block_index
        numel = env.block_sizes[block_index].numel
        offset_var = self.offset_var(block_index)
        index_var = self.index_var(block_index)
        block_size_var = self.block_size_var(block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {self.block_size!r}")
            )
        body: list[ast.AST] = [
            statement_from_string(
                f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)}"
            ),
        ]
        if (mask_var := self._mask_var) is not None:
            body.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {state.sympy_expr(numel)}"
                )
            )

        for_node = create(
            ast.For,
            target=create(ast.Name, id=offset_var, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(
                    state.config,
                    [self.block_index],
                    begin="0",
                    end=state.sympy_expr(numel),
                    step=block_size_var,
                ),
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        # Extract end_var_name from the actual numel expression used in the range()
        from .tile_strategy import LoopDimInfo

        end_var_name = state.sympy_expr(numel)
        block_id_to_info = {
            block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
        )

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        with install_inductor_kernel_handlers(state.codegen, {}):
            env = CompileEnvironment.current()
            backend = env.backend
            device_loop = state.codegen.active_device_loops[self.block_index][-1]
            assert isinstance(device_loop, DeviceLoopState)
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_input.size()])
            acc_dtype = get_computation_dtype(fake_input.dtype)  # promote fp16 to fp32
            default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
            assert isinstance(default, (float, int, bool))
            assert state.fx_node is not None
            acc = self.fn.new_var(f"{state.fx_node.name}_acc", dce=True)
            acc_full = backend.full_expr(shape_dims, constant_repr(default), acc_dtype)
            device_loop.outer_prefix.append(
                statement_from_string(f"{acc} = {acc_full}")
            )
            nki_memset = getattr(backend, "full_memset_stmt", None)
            if nki_memset is not None:
                device_loop.outer_prefix.append(
                    statement_from_string(nki_memset(acc, constant_repr(default)))
                )
            # Register SBUF shape for NKI multi-user detection on copy vars
            if backend.name == "nki" and hasattr(self.fn, "_nki_sbuf_shapes"):
                resolved_dims = []
                for d in shape_dims:
                    try:
                        resolved_dims.append(int(d))
                    except (TypeError, ValueError):
                        break
                if len(resolved_dims) >= 2:
                    self.fn._nki_sbuf_shapes[acc] = resolved_dims
            result = self.fn.new_var(state.fx_node.name, dce=True)
            if not backend.is_indexed_reduction(reduction_type):
                combine_expr = backend.reduction_combine_expr(
                    reduction_type, acc, input_name, acc_dtype
                )
                state.add_statement(f"{acc} = {combine_expr}")

                # For NKI backend, we need to handle the final reduction specially
                # to ensure tensor_reduce happens after the loop, not inside it
                if backend.name == "nki":
                    # For NKI, we defer the final reduction to outer_suffix
                    # The accumulator updates are already handled by the combine_expr above

                    # Then add the final reduction to outer_suffix (after the loop)
                    # We directly generate the NKI reduction code here
                    _NKI_REDUCTION_OPS = {
                        "sum": "nl.add",
                        "max": "nl.maximum",
                        "min": "nl.min",
                        "prod": "nl.mul",
                        "mean": "nl.add",
                    }
                    op = _NKI_REDUCTION_OPS.get(reduction_type)
                    if op is not None:
                        # Generate the final reduction code for outer_suffix
                        reduction_stmts = []
                        nki_reduce_var = self.fn.new_var("nki_reduce", dce=True)

                        # Allocate result buffer.
                        # Use shape_dims (already resolved for this tile strategy)
                        # to determine the partition dimension, since these are
                        # already resolved to concrete config values.
                        device_fn = getattr(state.codegen, "device_function", None)
                        sbuf_shape = None
                        if device_fn is not None and hasattr(device_fn, "_nki_sbuf_shapes"):
                            # Try input_name first, then the accumulator variable
                            sbuf_shape = device_fn._nki_sbuf_shapes.get(input_name)
                            if sbuf_shape is None:
                                sbuf_shape = device_fn._nki_sbuf_shapes.get(acc)

                        # Also try resolving from shape_dims (the tile strategy dims).
                        # shape_dims[0] might be a numeric string or a variable name
                        # like "_BLOCK_SIZE_1" — resolve using config values.
                        if sbuf_shape is None and shape_dims:
                            _dim_str = shape_dims[0]
                            try:
                                sbuf_shape = [int(_dim_str)]
                            except (ValueError, TypeError):
                                # Build a lookup dict: var_str → config value
                                _local_vars: dict[str, int] = {}
                                for _bid2 in range(len(env.block_sizes)):
                                    _bs2 = env.block_sizes[_bid2]
                                    # Get the generated variable name for this block size
                                    _sym = _bs2.symbol()
                                    _cfg_val = int(_bs2.from_config_assert(state.config))
                                    # The var name in generated code comes from the sympy symbol
                                    _local_vars[str(_sym)] = _cfg_val
                                    # Also try common naming patterns
                                    for _dbg in _bs2.debug_names:
                                        _local_vars[_dbg] = _cfg_val
                                try:
                                    sbuf_shape = [int(eval(_dim_str, {}, _local_vars))]  # noqa: S307
                                except Exception:
                                    pass

                        if sbuf_shape is not None and len(sbuf_shape) >= 1:
                            part_size = sbuf_shape[0]
                            dst_shape = f"[{part_size}, 1]"
                        elif fake_input is not None and fake_input.ndim >= 2:
                            # Resolve partition dim through config block sizes
                            # to avoid using the hint (which may differ from
                            # the actual configured block size)
                            part_size_sym = fake_input.size(0)
                            import sys
                            print(f"[REDUCE_DBG] input={input_name} fake_input.shape={list(fake_input.shape)} part_size_sym={part_size_sym} type={type(part_size_sym).__name__} sbuf_shapes_has={input_name in (device_fn._nki_sbuf_shapes if device_fn else {})}", file=sys.stderr)
                            if isinstance(part_size_sym, torch.SymInt):
                                import sympy as _sympy
                                _bs_subs: dict[_sympy.Symbol, int] = {}
                                for _bid in range(len(env.block_sizes)):
                                    _bs = env.block_sizes[_bid]
                                    _bs_subs[_bs.symbol()] = int(
                                        _bs.from_config_assert(state.config)
                                    )
                                part_size = int(part_size_sym._sympy_().subs(_bs_subs))
                            else:
                                part_size = int(part_size_sym)
                                # The hint might differ from the configured block size.
                                # Check if this value matches any block size's hint, and
                                # if so use the configured value instead.
                                for _bid in range(len(env.block_sizes)):
                                    _bs = env.block_sizes[_bid]
                                    if not _bs.reduction:
                                        _hint_val = env.size_hint(_bs.var._sympy_())
                                        _cfg_val = int(_bs.from_config_assert(state.config))
                                        if int(_hint_val) == part_size and _cfg_val != part_size:
                                            part_size = _cfg_val
                                            break
                            dst_shape = f"[{part_size}, 1]"
                        else:
                            dst_shape = "[1, 1]"
                        reduction_stmts.append(
                            f"{nki_reduce_var} = nl.ndarray({dst_shape}, nl.float32, buffer=nl.sbuf)"
                        )

                        # Perform the reduction
                        reduction_stmts.append(
                            f"nisa.tensor_reduce(dst={nki_reduce_var}, op={op}, data={acc}, axis={dim}, keepdims=True)"
                        )

                        # Handle mean reduction (scale by 1/N)
                        if reduction_type == "mean" and fake_input is not None:
                            reduction_extent = fake_input.size(dim)
                            if isinstance(reduction_extent, sympy.Basic):
                                reduction_extent = env.size_hint(reduction_extent)
                            reduction_extent = int(reduction_extent)
                            if reduction_extent > 0:
                                scale = 1.0 / reduction_extent
                                reduction_stmts.append(
                                    f"nisa.tensor_scalar(dst={nki_reduce_var}, data={nki_reduce_var}, "
                                    f"op0=nl.multiply, operand0={repr(scale)}, op1=None)"
                                )

                        # Extract the result with proper shape
                        if fake_output is not None and fake_output.ndim == fake_input.ndim - 1 and dim == 1:
                            expr = f"{nki_reduce_var}[:, 0]"
                        else:
                            expr = nki_reduce_var

                        # Add all statements to outer_suffix
                        for stmt in reduction_stmts:
                            device_loop.outer_suffix.append(statement_from_string(stmt))

                        # Final assignment with shape/cast
                        expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
                        expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
                        device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))
                    else:
                        # Fallback for unsupported reduction types
                        expr = self.call_reduction_function(
                            acc, reduction_type, dim, fake_input, fake_output
                        )
                        expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
                        expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
                        device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))
                else:
                    # Non-NKI backends: original behavior
                    expr = self.call_reduction_function(
                        acc, reduction_type, dim, fake_input, fake_output
                    )
                    # Ensure the final reduction result matches torch.* dtype semantics
                    expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
                    expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
                    device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))
            else:
                acc_index = self.fn.new_var(f"{state.fx_node.name}_acc_index", dce=True)
                index_dtype = env.index_dtype
                device_loop.outer_prefix.append(
                    statement_from_string(
                        f"{acc_index} = {backend.reduction_index_init_expr(shape_dims, index_dtype)}"
                    )
                )
                index = self.broadcast_str(
                    self.index_var(self.block_index), fake_input, dim
                )
                for stmt in backend.argreduce_loop_update_statements(
                    reduction_type=reduction_type,
                    acc=acc,
                    acc_index=acc_index,
                    value=input_name,
                    index=index,
                ):
                    state.add_statement(stmt)
                expr = self.call_indexed_reduction(
                    acc,
                    acc_index,
                    reduction_type,
                    dim,
                    fake_output,
                )
                # Ensure the final reduction result matches torch.* dtype semantics
                expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
                expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
                device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))

            # Optional: emit a dtype static assert right after the assignment when enabled
            if env.settings.debug_dtype_asserts and backend.name != "nki":
                device_loop.outer_suffix.append(
                    statement_from_string(
                        f"tl.static_assert({result}.dtype == {_dtype_str(fake_output.dtype)})"
                    )
                )
            return expr_from_string(result)


class BlockReductionStrategy(ReductionStrategy):
    """This is used when we are reducing over a tile rather than an entire tensor."""

    def __init__(
        self,
        state: CodegenState,
        block_index: int,
    ) -> None:
        super().__init__(
            fn=state.device_function,
            block_index=block_index,
            mask_var=state.codegen.mask_var(block_index),
            block_size_var=None,
        )
        self.offset_vars[block_index] = "0"
        # Store reference to codegen to access existing index variables
        self._codegen = state.codegen

    def index_var(self, block_idx: int) -> str:
        # Use the existing index variable from the active device loop
        # instead of the newly created one from TileStrategy.__init__
        return self._codegen.index_var(block_idx)

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
        assert isinstance(default, (float, int, bool))
        env = CompileEnvironment.current()
        dim_size = fake_input.size(dim)
        is_zero_dim = False
        if (
            isinstance(dim_size, int)
            and dim_size == 0
            or isinstance(dim_size, torch.SymInt)
            and env.known_equal(dim_size, 0)
        ):
            is_zero_dim = True
        if is_zero_dim:
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            backend = env.backend
            nki_memset = getattr(backend, "full_memset_stmt", None)
            if nki_memset is not None:
                var = self.fn.new_var("zero_dim_default", dce=True)
                ndarray_expr = backend.full_expr(shape_dims, "0", fake_output.dtype)
                state.add_statement(
                    statement_from_string(f"{var} = {ndarray_expr}")
                )
                state.add_statement(
                    statement_from_string(
                        nki_memset(var, constant_repr(default))
                    )
                )
                return expr_from_string(var)
            return expr_from_string(
                backend.full_expr(
                    shape_dims, constant_repr(default), fake_output.dtype
                )
            )
        expr = self.call_reduction_function(
            input_name,
            reduction_type,
            dim,
            fake_input,
            fake_output,
        )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
