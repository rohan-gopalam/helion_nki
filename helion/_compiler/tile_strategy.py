from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
import weakref

import sympy
import torch

from .. import exc
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy
from .host_function import HostFunction
from .program_id import FlatProgramIDs
from .program_id import ForEachProgramID
from .program_id import L2GroupingProgramIDs
from .program_id import NKIProgramIDs
from .program_id import PersistentBlockedProgramIDs
from .program_id import PersistentInterleavedProgramIDs
from .program_id import PIDInfo
from .program_id import ProgramIDs
from .program_id import XYZProgramIDs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


@dataclasses.dataclass
class LoopDimInfo:
    end_var_name: str | None
    end_expr: sympy.Expr | None

    def is_end_matching(self, size: int | torch.SymInt) -> bool:
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        hint = CompileEnvironment.current().shape_env.size_hint
        # TODO(jansel): current check is based on size hints, may need to guard here in the future
        return hint(expected) == hint(self.end_expr)


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy
    block_id_to_info: dict[int, LoopDimInfo]

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids


@dataclasses.dataclass
class DeviceLoopState(DeviceLoopOrGridState):
    for_node: ast.For
    inner_statements: list[ast.AST]
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DeviceGridState(DeviceLoopOrGridState):
    # lane_loops entries may be:
    #   (lane_var, extent)
    #   (lane_var, body_prefix_stmts, extent)
    # The NKI backend uses the 3-tuple form to inject nisa.iota index setup
    # at the top of each loop body where ``lane_var`` is in scope.
    lane_loops: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for entry in reversed(self.lane_loops):
            if len(entry) == 3:
                lane_var, body_prefix, extent = entry
            else:
                lane_var, extent = entry
                body_prefix = []
            iter_expr = extent if isinstance(extent, str) else f"range({extent})"
            body_with_prefix: list[ast.AST] = [*body_prefix, *wrapped]
            wrapped = [
                create(
                    ast.For,
                    target=create(ast.Name, id=lane_var, ctx=ast.Store()),
                    iter=expr_from_string(iter_expr),
                    body=body_with_prefix,
                    orelse=[],
                    type_comment=None,
                )
            ]
        return wrapped


class PersistentReductionState(DeviceLoopOrGridState):
    pass


def _nki_body_leading_count(
    state: object, block_ids: list[int], env: object
) -> int:
    """How many of this loop's block_sizes contribute to the partition
    dim of the largest 3D+ allocation in the kernel body.

    Scans the FX graph (via ``state.fx_node``'s enclosing graph) for
    creation ops (hl.full/hl.zeros and similar) with ndim >= 3 whose
    leading dim symbols match this loop's block_ids. Returns the max
    number of leading dims whose symbols are in our block_ids.

    For a pure 2D loop with 2D body allocations only, returns 0.
    For a 2D loop with a 3D body allocation like [tile_b, tile_m, D]
    (both block_ids in leading dims), returns 2.
    """
    block_id_set = set(block_ids)
    best = 0

    # Walk EVERY sub-graph in device_ir (including inner loop bodies that
    # aren't root graphs). Inner for loops (e.g. the tile_v inner loop in
    # grpo_loss) own their own graphs and contain 3D loads that the outer
    # root graph doesn't see.
    try:
        codegen = getattr(state, "codegen", None)
        hf = getattr(codegen, "host_function", None) if codegen is not None else None
        if hf is not None:
            device_ir = hf.device_ir
            for graph_info in device_ir.graphs:
                _best = _count_leading_block_id_matches(graph_info.graph, block_id_set, env)
                if _best > best:
                    best = _best
            return best
    except Exception:
        pass

    # Fallback: state.fx_node.graph
    fx_node = getattr(state, "fx_node", None)
    if fx_node is not None and hasattr(fx_node, "graph"):
        return _count_leading_block_id_matches(fx_node.graph, block_id_set, env)
    return best


def _count_leading_block_id_matches(graph, block_id_set, env) -> int:
    """For each node in ``graph`` whose meta['val'] is a 3D+ tensor, count
    how many of its leading dims (all but the last) have symbols that
    are block-size symbols for the loop identified by ``block_id_set``.
    Return the max count seen.
    """
    best = 0
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        if val.ndim < 3:
            continue
        # Count how many LEADING dims (all but the last) are themselves
        # this-loop's block_size symbols.
        leading_matches = 0
        for d in val.shape[:-1]:
            if not isinstance(d, torch.SymInt):
                break
            sym_expr = d._sympy_()
            if not hasattr(sym_expr, "free_symbols"):
                break
            matched = False
            for sym in sym_expr.free_symbols:
                for bid in block_id_set:
                    try:
                        bs_info = env.block_sizes[bid]
                    except IndexError:
                        continue
                    if sym == bs_info.symbol():
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                break
            leading_matches += 1
        if leading_matches > best:
            best = leading_matches
    return best


def _backend_loop_index_statements(
    backend: object,
    *,
    offset_var: str,
    block_size_var: str,
    dtype: str,
    axis: int,
    index_var: str,
) -> list[ast.stmt]:
    """Return the statement(s) that compute ``index_var`` from ``offset_var``.

    Most backends produce a single assignment ``index_var = <expr>``. The NKI
    backend needs an SBUF ndarray allocation plus a nisa.iota fill (so the
    index is a per-position vector, not a scalar). Backends with that kind
    of multi-statement setup implement ``loop_index_statements`` and
    ``grid_index_statements``; everything else falls back to the single
    ``{index_var} = {loop_index_expr(...)}`` shape.
    """
    fn = getattr(backend, "loop_index_statements", None)
    if fn is not None:
        stmts = fn(
            offset_var=offset_var,
            block_size_var=block_size_var,
            dtype=dtype,
            axis=axis,
            index_var=index_var,
        )
        return [
            s if isinstance(s, ast.stmt) else statement_from_string(s)
            for s in stmts
        ]
    idx_expr = backend.loop_index_expr(offset_var, block_size_var, dtype, axis=axis)
    return [statement_from_string(f"{index_var} = {idx_expr}")]


def _backend_grid_index_statements(
    backend: object,
    *,
    offset_var: str,
    block_size_var: str,
    dtype: str,
    axis: int,
    index_var: str,
) -> list[ast.stmt]:
    """Parallel to ``_backend_loop_index_statements`` for codegen_grid."""
    fn = getattr(backend, "grid_index_statements", None)
    if fn is not None:
        stmts = fn(
            offset_var=offset_var,
            block_size_var=block_size_var,
            dtype=dtype,
            axis=axis,
            index_var=index_var,
        )
        return [
            s if isinstance(s, ast.stmt) else statement_from_string(s)
            for s in stmts
        ]
    idx_expr = backend.grid_index_expr(offset_var, block_size_var, dtype, axis=axis)
    return [statement_from_string(f"{index_var} = {idx_expr}")]


class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    block_ids: list[int]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_ids = block_ids
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_ids
        }
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"offset_{block_idx}", dce=True)
            for block_idx in block_ids
        }

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def offset_var(self, block_idx: int) -> str:
        return self.offset_vars[block_idx]

    def index_var(self, block_idx: int) -> str:
        return self.index_vars[block_idx]

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def block_size_var(self, block_idx: int) -> str | None:
        return self.fn.block_size_var_cache.get((block_idx,))

    def supports_index_rank_expansion(self) -> bool:
        """Whether index expressions produced by this strategy are tensor-shaped."""
        return True

    def thread_axes_used(self) -> int:
        return 0

    def thread_block_sizes(self) -> list[int]:
        """Return the thread block size for each thread axis this strategy uses."""
        return []

    @staticmethod
    def get_tl_range_kwargs(config: Config, block_idx: int) -> list[str]:
        """Get the range_extra string for loop unroll factor and num_stages based on config."""
        env = CompileEnvironment.current()
        kwargs = []

        range_unroll_factor = env.config_spec.range_unroll_factors.config_get(
            config.range_unroll_factors, block_idx, 0
        )
        range_warp_specialize = env.config_spec.range_warp_specialize.config_get(
            config.range_warp_specializes, block_idx, None
        )
        range_num_stages = env.config_spec.range_num_stages.config_get(
            config.range_num_stages, block_idx, 0
        )
        num_stages = config.num_stages

        if config.indexing == "tensor_descriptor":
            # Tensor descriptor + multi-stage pipelines in addition to unrolling tend to cause
            # CUDA "misaligned address" or "unspecified launch failure" errors.
            if range_num_stages > 0:
                range_num_stages = 0
            if range_unroll_factor > 0 and num_stages > 1:
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            # Unrolling can cause CUDA IMA with pipelining
            # We want to ensure new step size + pipeline is within bounds
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
            step = range_unroll_factor * block_size
            last_offset = ((loop_numel - 1) // block_size) * block_size
            remainder = loop_numel - last_offset
            range_num_stages = min(
                max(1, int(math.ceil(remainder / step))), range_num_stages
            )

        if range_unroll_factor > 0:
            kwargs.append(f"loop_unroll_factor={range_unroll_factor}")
        if range_warp_specialize is not None:
            kwargs.append(f"warp_specialize={range_warp_specialize}")
        if range_num_stages > 0:
            kwargs.append(f"num_stages={range_num_stages}")

        range_multi_buffer = env.config_spec.range_multi_buffers.config_get(
            config.range_multi_buffers, block_idx, None
        )
        if range_multi_buffer is not None:
            kwargs.append(f"disallow_acc_multi_buffer={not range_multi_buffer}")

        range_flatten = env.config_spec.range_flattens.config_get(
            config.range_flattens, block_idx, None
        )
        if range_flatten is not None:
            kwargs.append(f"flatten={range_flatten}")

        dpf_range = config.get("_triton_range_id_data_partition_factor", None)
        dpf_value = config.get("_triton_range_value_data_partition_factor", None)

        if dpf_range is not None and dpf_value is not None and dpf_range == block_idx:
            kwargs.append(f"data_partition_factor={dpf_value}")

        return kwargs

    @staticmethod
    def get_range_call_str(
        config: Config,
        block_ids: list[int],
        *,
        begin: str | None = None,
        end: str,
        step: str | None = None,
    ) -> str:
        env = CompileEnvironment.current()

        # Allow backend to override the range expression entirely
        backend_range = env.backend.range_str(begin, end, step)
        if backend_range is not None:
            return backend_range

        use_static_range = all(
            env.config_spec.static_ranges.config_get(
                config.static_ranges, block_idx, None
            )
            is True
            for block_idx in block_ids
        )

        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)

        if use_static_range:
            return f"tl.static_range({', '.join(range_args)})"

        range_kwargs = TileStrategy.get_tl_range_kwargs(config, block_ids[0])
        return f"tl.range({', '.join(range_args + range_kwargs)})"

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    def _create_block_id_info_dict(
        self,
        state: CodegenState,
        use_proxy_ends: bool = False,
        ends_override: list[object] | None = None,
    ) -> dict[int, LoopDimInfo]:
        """Helper to create block_id_to_info dictionary with end bounds.

        Args:
            state: The codegen state
            use_proxy_ends: If True, use proxy_ends from state.proxy_args (for device loops)
            ends_override: If provided, use these ends instead of block_sizes.numel (for data-dependent bounds)
        """
        env = CompileEnvironment.current()
        block_id_to_info = {}

        if use_proxy_ends:
            _, _, proxy_ends, _ = state.proxy_args
            assert isinstance(proxy_ends, list)
            for block_idx, end in zip(self.block_ids, proxy_ends, strict=True):
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                else:
                    end_expr = None
                block_id_to_info[block_idx] = LoopDimInfo(
                    end_var_name=None, end_expr=end_expr
                )
        elif ends_override is not None:
            # Data-dependent bounds: use the provided ends
            for block_id, end in zip(self.block_ids, ends_override, strict=True):
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                    end_var_name = state.sympy_expr(end_expr)
                else:
                    # Tensor (data-dependent) - end_expr is None, but we still need end_var
                    end_expr = None
                    end_var_name = None
                block_id_to_info[block_id] = LoopDimInfo(
                    end_var_name=end_var_name, end_expr=end_expr
                )
        else:
            for block_id in self.block_ids:
                block_size_info = env.block_sizes[block_id]
                if block_size_info.size is None:
                    # Data-dependent bound - skip numel, it will be handled elsewhere
                    end_expr = None
                    end_var_name = None
                else:
                    end_expr = block_size_info.numel
                    end_var_name = state.sympy_expr(end_expr)
                block_id_to_info[block_id] = LoopDimInfo(
                    end_var_name=end_var_name, end_expr=end_expr
                )

        return block_id_to_info

    def _setup_block_size_constexpr(
        self,
        state: CodegenState,
        block_size_var: str,
        block_size: SymIntLike,
        block_idx: int | None = None,
    ) -> None:
        """Helper to setup constexpr block size variable on host.
        For NKI, block sizes are inlined as literals (no kernel params); block_idx required.
        """
        env = CompileEnvironment.current()
        if env.backend.name == "nki" and block_idx is not None:
            literal = env.block_sizes[block_idx].from_config_assert(state.config)
            state.device_function.block_size_var_cache[(block_idx,)] = str(int(literal))
            return
        state.device_function.constexpr_arg_with_host_def(block_size_var, block_size)


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=block_ids,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_ids: list[_T]) -> list[_T]:
        if len(block_ids) <= 1:
            return block_ids
        order = self.loop_order
        assert len(order) == len(block_ids), (
            f"Invalid order length: {len(order)} != {len(block_ids)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_ids[i] for i in reversed(order)]

    def _get_data_dependent_numel(
        self, state: CodegenState, end: object, begin: object
    ) -> sympy.Expr | str:
        """Get numel for data-dependent bounds using the tensor end value.

        When the tile bound is a tensor (data-dependent), we need to pass
        the tensor to the kernel and use it to compute the number of elements.
        Returns either a sympy.Expr or a string expression.
        """
        from .device_function import DeviceFunction

        device_function = DeviceFunction.current()

        if isinstance(end, torch.Tensor):
            # For tensor bounds, we need to add it as a kernel argument
            # and load the scalar value
            tensor_arg = device_function.tensor_arg(end)
            end_expr = CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        elif isinstance(end, (int, torch.SymInt)):
            end_expr = device_function.sympy_expr(_to_sympy(end))
        else:
            raise NotImplementedError(f"Unsupported end type: {type(end)}")

        if begin == 0:
            # Simple case: numel = end
            return end_expr  # type: ignore[return-value]
        if isinstance(begin, torch.Tensor):
            begin_arg = device_function.tensor_arg(begin)
            begin_expr = CompileEnvironment.current().backend.scalar_load_expr(
                begin_arg.name
            )
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        if isinstance(begin, (int, torch.SymInt)):
            begin_expr = device_function.sympy_expr(_to_sympy(begin))
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        raise NotImplementedError(f"Unsupported begin type: {type(begin)}")

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()

    def _fold_tile_end_op(
        self,
        state: CodegenState,
        end: object,
        block_size: int | torch.SymInt,
    ) -> sympy.Expr | None:
        """
        Compute more precise end bound for the pattern:

            for outer in hl.tile(...):
                for inner in hl.tile(outer.begin, outer.end):
                    ...
        """
        if isinstance(end, (int, torch.SymInt)):
            end = _to_sympy(end)
        elif not isinstance(end, sympy.Expr):
            return None

        var_info = state.device_function.expr_to_var_info.get(end)
        if var_info is None or not isinstance(block_size, int):
            return end

        from ..language.tile_ops import tile_end

        env = CompileEnvironment.current()
        fx_node = var_info.fx_node
        # check for the case where we have the same end bound a parent loop
        if (
            fx_node is not None
            and fx_node.target is tile_end
            and isinstance(arg := fx_node.args[0], torch.fx.Node)
            and (block_id := env.get_block_id(arg.meta["val"])) is not None
            and (device_loops := state.codegen.active_device_loops.get(block_id))
            and (loop_info := device_loops[-1].block_id_to_info.get(block_id))
            is not None
            # TODO(jansel): when parent block size is a SymInt, we fail to apply this optimization should fix this
            and isinstance(
                parent_block_size := env.block_sizes[block_id].from_config(
                    state.config
                ),
                int,
            )
            # If our block size is larger than the parent, then their will be gaps in the iteration space
            and block_size <= parent_block_size
        ):
            # Replace our end bound (a SymInt) will the parent loop's end bound
            return loop_info.end_expr
        return end

    def select_pid_strategy(self) -> ProgramIDs:
        pid_type = self.fn.config.pid_type
        if pid_type == "xyz":
            assert 1 < len(self.block_ids) <= 3
            return XYZProgramIDs()
        if pid_type == "persistent_blocked":
            return PersistentBlockedProgramIDs()
        if pid_type == "persistent_interleaved":
            return PersistentInterleavedProgramIDs()
        assert pid_type == "flat"
        return FlatProgramIDs()


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    # pyrefly: ignore [bad-override]
    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_ids, block_size, loop_order)
        env = CompileEnvironment.current()
        if not env.backend.force_tile_mask() and env.known_multiple(
            functools.reduce(
                operator.mul, [env.block_sizes[i].numel for i in block_ids]
            ),
            block_size,
        ):
            self._mask_var = None
        else:
            self._mask_var: str | None = self.new_var("mask", dce=True)
        self._offsets_var = self.new_var("offsets", dce=True)

        key = (*self.block_ids,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_ids:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_ids))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def mask_var(self, block_idx: int) -> str | None:
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_ids)]

    def thread_axes_used(self) -> int:
        return int(self._uses_thread_axis())

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis() or not isinstance(self.block_size, int):
            return []
        return [self.block_size]

    def _uses_thread_axis(self) -> bool:
        return not (isinstance(self.block_size, int) and self.block_size == 1)

    def _codegen_common(
        self, state: CodegenState
    ) -> tuple[str, str, sympy.Expr, list[ast.AST]]:
        offsets_var = self._offsets_var
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        statements = []

        # pyrefly: ignore [bad-assignment]
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            mask_terms = [f"{offsets_var} < ({state.sympy_expr(total_numel)})"]
            thread_mask = env.backend.thread_in_tile_mask_expr(
                block_size_var, axis=self._flat_thread_axis()
            )
            if thread_mask is not None:
                mask_terms.insert(0, f"({thread_mask})")
            mask_expr = " and ".join(mask_terms)
            statements.append(statement_from_string(f"{mask_var} = {mask_expr}"))
        # pyrefly: ignore [bad-return]
        return block_size_var, offsets_var, total_numel, statements

    def _flat_thread_axis(self) -> int:
        """Compute the thread axis for this flattened strategy.

        For CuTe, reduction strategies occupy earlier axes.
        """
        from .reduction_strategy import ReductionStrategy

        env = CompileEnvironment.current()
        if not env.backend.reduction_axis_first():
            return 0
        axis = 0
        for strategy in self.fn.tile_strategy.strategies:
            if isinstance(strategy, ReductionStrategy):
                axis += strategy.thread_axes_used()
        return axis

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))

        state.add_statement(
            env.backend.arange_expr(
                offsets_var,
                pid_var,
                block_size_var,
                dtype,
                axis=self._flat_thread_axis(),
            )
        )
        state.codegen.statements_stack[-1].extend(statements)

        pids.codegen(state)

        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(self, block_id_to_info=block_id_to_info)

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()
        lid = self.new_var("lid")
        numel_str = state.sympy_expr(total_numel)
        end_var = env.backend.cdiv_expr(numel_str, block_size_var, is_device=True)
        arange_expr = env.backend.arange_expr(
            offsets_var, lid, block_size_var, dtype, axis=self._flat_thread_axis()
        )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=(
                body := [
                    statement_from_string(arange_expr),
                    *statements,
                ]
            ),
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)

        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
        )

    @classmethod
    def update_allow_flattened(cls, shape: Sequence[sympy.Expr]) -> None:
        env = CompileEnvironment.current()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = env.get_block_id(x)
            if block_idx is not None:
                used_indices[block_idx] = i
        flatten_loops = env.config_spec.flatten_loops
        for spec in [*flatten_loops]:
            block_ids = spec.block_ids
            if not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            ):
                flatten_loops.disable_block_id(block_ids[0])
                continue
            for i, j in itertools.pairwise(block_ids):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    flatten_loops.disable_block_id(block_ids[0])
                    break

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # Keep axis structure intact for multi-phase kernels (e.g., barrier) to
        # avoid mismatched ranks in downstream reductions.
        if len(HostFunction.current().device_ir.root_ids) > 1:
            return shapes

        env = CompileEnvironment.current()
        # Filter out unit-sized blocks that don't need compacting
        compact_block_ids = [
            block_id
            for block_id in self.block_ids
            if not (
                isinstance(env.block_sizes[block_id].size, int)
                and env.block_sizes[block_id].size == 1
            )
        ]
        if not compact_block_ids:
            return shapes

        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            # Check if this starts our flattened sequence
            if len(shape.block_ids) != 1 or shape.block_ids[0] != compact_block_ids[0]:
                output.append(shape)
                continue

            # Try to collect the full sequence
            group_shapes = [shape]
            found_complete_sequence = True
            for expected in compact_block_ids[1:]:
                if (
                    shape_queue
                    and len(shape_queue[0].block_ids) == 1
                    and shape_queue[0].block_ids[0] == expected
                ):
                    group_shapes.append(shape_queue.popleft())
                else:
                    # Partial match - don't combine
                    found_complete_sequence = False
                    output.extend(group_shapes)
                    break

            if found_complete_sequence:
                # Full match - combine into one
                for s in group_shapes[1:]:
                    shape = shape.combine(s)
                output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    # pyrefly: ignore [bad-override]
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_ids, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_ids, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def _uses_thread_axis(self, block_size: SymIntLike) -> bool:
        return not (isinstance(block_size, int) and block_size == 1)

    def thread_axes_used(self) -> int:
        return sum(
            1 for block_size in self.block_size if self._uses_thread_axis(block_size)
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if self._uses_thread_axis(bs) and isinstance(bs, int):
                sizes.append(bs)
        return sizes

    def _thread_axis_offset(self, state: CodegenState) -> int:
        from .reduction_strategy import ReductionStrategy

        seen: set[int] = set()
        offset = 0
        env = CompileEnvironment.current()
        reduction_axis_first = env.backend.reduction_axis_first()
        if reduction_axis_first:
            # Reduction strategies claim axis 0, so grid/loop
            # strategies must offset past them.
            for strategy in self.fn.tile_strategy.strategies:
                if isinstance(strategy, ReductionStrategy):
                    offset += strategy.thread_axes_used()
        for loops in state.codegen.active_device_loops.values():
            for loop_state in loops:
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                if reduction_axis_first and isinstance(
                    loop_state.strategy, ReductionStrategy
                ):
                    # Reduction axes are already accounted for above.
                    continue
                offset += loop_state.strategy.thread_axes_used()
        return offset

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis(block_size_by_id[block_id]):
                axis += 1
        return mapping

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)

        if env.backend.name == "nki":
            return self._codegen_grid_nki(state, block_ids, block_sizes, begins, ends)

        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end) in enumerate(
            reversed(
                self._reorder([*zip(block_ids, block_sizes, begins, ends, strict=True)])
            )
        ):
            block_size_info = env.block_sizes[block_idx]
            # Handle data-dependent bounds: if size is None, use the end value from proxy_args
            if block_size_info.size is None:
                # Data-dependent bound - use the tensor end value
                numel = self._get_data_dependent_numel(state, end, begin)
            else:
                numel = block_size_info.numel
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if block_size != 1:
                # NKI: set cache to literal first so device and launcher grid use literal
                block_size_var_for_constexpr = self.block_size_var(block_idx)
                assert block_size_var_for_constexpr is not None
                self._setup_block_size_constexpr(
                    state,
                    block_size_var_for_constexpr,
                    block_size,
                    block_idx=block_idx,
                )
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")
            axis = thread_axis_offset + thread_axis_map[block_idx]
            uses_thread_axis = self._uses_thread_axis(block_size)
            bs = block_size_var if uses_thread_axis else "1"
            for _stmt in _backend_grid_index_statements(
                env.backend,
                offset_var=offset_var,
                block_size_var=bs,
                dtype=dtype,
                axis=axis,
                index_var=index_var,
            ):
                state.add_statement(_stmt)
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, numel
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        # Only use ends_override if there are data-dependent (tensor) bounds
        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(self, block_id_to_info=block_id_to_info)

    def _codegen_grid_nki(
        self,
        state: CodegenState,
        block_ids: list[int],
        block_sizes: list[int],
        begins: list[object],
        ends: list[object],
    ) -> DeviceGridState:
        """NKI: emit nl.affine_range loops instead of program_id grid parallelism.

        NKI SBUF is 2D: the partition dim is capped at 128. Helion reshapes
        3D+ inputs [B, M, K] → [B*M, K] at kernel entry, so the effective
        partition extent of a tile group is ``prod(block_sizes[:-1])``.

        To keep user configs running unchanged when they are "too big" for
        NKI (e.g. ``block_sizes=[128, 128, 128]`` for a 3D kernel gives a
        flattened partition of 16384), we auto-clamp the leading tile dims
        downward — 1 at a time from the outermost — until the flattened
        partition product fits in 128. The outermost loop then iterates
        more times with a smaller step, preserving correctness. The user's
        config is treated as an upper bound; we only reduce, never grow.
        """
        env = CompileEnvironment.current()
        _NKI_PMAX = 128
        _orig_block_sizes = list(block_sizes)
        block_sizes = list(block_sizes)  # make mutable

        # Scan the FX graph for creation ops with higher-rank shapes. If any
        # allocation has N>=2 leading dims (plus a trailing free dim) whose
        # symbols match this loop's block_ids, the flattened partition
        # extent of that allocation is the product of our block_sizes at
        # those positions — and must fit in 128. We collect the maximum
        # number of leading block_sizes that would contribute to a body
        # allocation's partition, then cap the product of those
        # block_sizes at _NKI_PMAX.
        _body_leading_count = _nki_body_leading_count(state, block_ids, env)

        # NKI SBUF is 2D with partition dim ≤128. Helion reshapes 3D+ tiles
        # [B, M, K] to [B*M, K] at allocation time, so when a kernel body
        # allocates a tile like ``hl.zeros([tile_b, tile_m, head_dim])``,
        # the flattened partition is ``tile_b * tile_m``. If the user's
        # config declares a product of tile sizes that exceeds 128, clamp
        # the outermost dims down to 1 (one at a time) until the product
        # fits. This keeps user configs "just working" when they're too
        # big for NKI — the outer loop just iterates more times with a
        # smaller step. The final tile dim (free axis) is never clamped.
        if len(block_sizes) >= 2:
            def _resolve_bs(bs: object) -> int | None:
                if isinstance(bs, (int, bool)):
                    return int(bs)
                if isinstance(bs, torch.SymInt):
                    try:
                        return env.size_hint(bs)
                    except Exception:
                        return None
                return None

            resolved = [_resolve_bs(bs) for bs in block_sizes]
            if all(r is not None for r in resolved):
                def _prod_leading() -> int:
                    p = 1
                    for r in resolved[:-1]:
                        assert r is not None
                        p *= r
                    return p

                target = _NKI_PMAX

                def _prod_body_leading() -> int:
                    # How many of THIS loop's block_sizes end up as
                    # partition dims for the largest body allocation. For
                    # a 3D+ tile loop (e.g. bmm [B,M,N]), the kernel
                    # reshape flattens [B,M,K]→[B*M,K], so all but the
                    # last tile dim are partition → N-1 leading. For a 2D
                    # tile loop where the body allocates a 3D tile like
                    # ``hl.zeros([tile_b, tile_m, head_dim])``, the
                    # allocation's partition = tile_b * tile_m, i.e. BOTH
                    # loop block_sizes become partition. The helper
                    # ``_nki_body_leading_count`` scans the FX graph for
                    # these 3D+ allocations and returns the max count.
                    count = max(len(block_sizes) - 1, _body_leading_count)
                    count = min(count, len(block_sizes))  # can't exceed loop rank
                    p = 1
                    for r in resolved[:count]:
                        assert r is not None
                        p *= r
                    return p

                for i in range(len(block_sizes)):
                    # Leave at least the last dim uncollapsed so we don't
                    # accidentally force the free axis to 1.
                    if i == len(block_sizes) - 1 and _prod_body_leading() <= target:
                        break
                    while _prod_body_leading() > target:
                        if resolved[i] is None or resolved[i] <= 1:
                            break
                        new_bs = max(1, resolved[i] // 2)
                        if new_bs == resolved[i]:
                            new_bs = 1
                        resolved[i] = new_bs
                        block_sizes[i] = new_bs
                    if _prod_body_leading() <= target:
                        break

                # If the body uses a 3D allocation of the same shape as the
                # tile variables (common pattern like hl.zeros([tile_b,
                # tile_m, head_dim])), the second-to-last block_size also
                # contributes to the flattened partition. We already cap
                # leading dims above; as a safety net, if prod_all > cap
                # and prod_leading ≤ cap, also clamp the last leading-ish
                # dim once. In practice this case is rare and the above
                # loop handles it.
                if block_sizes != _orig_block_sizes:
                    import sys as _sys
                    print(
                        f"[helion-nki] auto-clamped block_sizes "
                        f"{_orig_block_sizes} → {block_sizes} "
                        f"(NKI partition ≤ {_NKI_PMAX})",
                        file=_sys.stderr,
                    )
                    # Propagate the clamp into state.config so any code
                    # that re-derives the block size via
                    # env.block_sizes[bid].from_config_assert(state.config)
                    # (e.g. hl.full / hl.zeros body allocations, mm sizing)
                    # sees the reduced value.
                    try:
                        config_block_sizes = state.config.config.get("block_sizes")
                        if isinstance(config_block_sizes, list):
                            for bid, new_bs in zip(block_ids, block_sizes, strict=True):
                                try:
                                    idx = env.config_spec.block_sizes.block_id_to_index(bid)
                                except Exception:
                                    continue
                                config_block_sizes[idx] = new_bs
                    except Exception:
                        pass

        lane_loops: list[tuple[str, list[ast.stmt], str]] = []

        # Clamp block_size to loop extent. NKI's dma_copy uses
        # ``offset:offset+block_size`` which goes OOB when block_size > extent
        # even though the outer range limits iterations. For tensors where
        # dim size is small (e.g. x_offsets [B+1=33]), reduce block_size.
        _clamped: list[object] = []
        for bsz, begin, end, block_idx in zip(block_sizes, begins, ends, block_ids, strict=True):
            _numel = env.block_sizes[block_idx].numel
            _num_int: int | None = None
            if isinstance(_numel, (int, bool)):
                _num_int = int(_numel)
            elif isinstance(_numel, torch.SymInt):
                try:
                    _num_int = env.size_hint(_numel)
                except Exception:
                    _num_int = None
            elif isinstance(_numel, sympy.Expr):
                try:
                    _num_int = int(_numel)
                except (TypeError, ValueError):
                    try:
                        _num_int = int(env.size_hint(_numel))
                    except Exception:
                        _num_int = None
            if _num_int is not None and isinstance(bsz, (int, bool)) and int(bsz) > _num_int:
                _clamped.append(_num_int)
            else:
                _clamped.append(bsz)
        if _clamped != list(block_sizes):
            # Propagate to config
            try:
                config_block_sizes = state.config.config.get("block_sizes")
                if isinstance(config_block_sizes, list):
                    for bid, new_bs in zip(block_ids, _clamped, strict=True):
                        try:
                            idx = env.config_spec.block_sizes.block_id_to_index(bid)
                        except Exception:
                            continue
                        config_block_sizes[idx] = new_bs
            except Exception:
                pass
            block_sizes = _clamped

        for block_idx, block_size, begin, end in zip(
            block_ids, block_sizes, begins, ends, strict=True
        ):
            block_size_info = env.block_sizes[block_idx]
            numel = block_size_info.numel
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)

            self._setup_block_size_constexpr(
                state,
                self.block_size_var(block_idx),
                block_size,
                block_idx=block_idx,
            )

            def _safe_int(x: object) -> int | None:
                if isinstance(x, (int, bool)):
                    return int(x)
                if isinstance(x, float):
                    return int(x)
                if isinstance(x, torch.SymInt):
                    return env.size_hint(x)
                if isinstance(x, sympy.Expr):
                    try:
                        return int(x)
                    except (TypeError, ValueError):
                        if x.free_symbols:
                            try:
                                return int(env.size_hint(x))
                            except Exception:
                                return None
                        return None
                return None

            begin_int = _safe_int(begin)
            begin_str = str(begin_int) if begin_int is not None else "0"
            if numel is not None:
                numel_int = _safe_int(numel)
                end_str = str(numel_int) if numel_int is not None else state.sympy_expr(_to_sympy(numel))
            else:
                end_int = _safe_int(end)
                end_str = str(end_int) if end_int is not None else str(end)
            step_int = _safe_int(block_size)
            step_str = str(step_int) if step_int is not None else "1"
            range_expr = f"nl.affine_range({begin_str}, {end_str}, {step_str})"

            # Emit the per-position index tile (nisa.iota) at the top of the
            # loop body so tile.index / tile.index-based masks can be
            # evaluated against a real SBUF vector. ``bs`` matches what the
            # generic codegen_device_loop path uses — see _BaseNDTileStrategy.
            uses_thread_axis = self._uses_thread_axis(block_size)
            bs = self.block_size_var(block_idx) if uses_thread_axis else "1"
            if bs is None:
                bs = "1"
            dtype_str = env.index_type()
            idx_stmts = _backend_loop_index_statements(
                env.backend,
                offset_var=offset_var,
                block_size_var=bs,
                dtype=dtype_str,
                axis=0,
                index_var=index_var,
            )
            lane_loops.append((offset_var, idx_stmts, range_expr))

        nki_pids = NKIProgramIDs()
        state.device_function.set_pid(nki_pids)

        block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(
            self, block_id_to_info=block_id_to_info, lane_loops=lane_loops
        )

    def _to_ast(self, x: object, to_dtype: str | None = None) -> ast.AST:
        if isinstance(x, ast.AST):
            if to_dtype:
                cast_expr = CompileEnvironment.current().backend.ast_to_dtype_expr(
                    "{value}", to_dtype
                )
                return expr_from_string(cast_expr, value=x)
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        if isinstance(x, torch.SymInt):
            return self._to_ast(x._sympy_())
        if isinstance(x, torch.Tensor):
            # Handle tensor values (for data-dependent bounds)
            # For scalar tensors, we need to load the value using tl.load
            from .device_function import DeviceFunction

            tensor_arg = DeviceFunction.current().tensor_arg(x)
            return expr_from_string(
                CompileEnvironment.current().backend.scalar_load_expr(tensor_arg.name)
            )
        if isinstance(x, str):
            # Already a string expression (for data-dependent numel)
            return expr_from_string(x)
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        _, begins, ends, _ = state.ast_args
        _, _, proxy_ends, _ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        assert isinstance(proxy_ends, list)
        # For NKI device loops, use dimension size as loop end (e.g. x.shape[1] for K)
        # so the range is 0..dim_size with step block_size, not 0..128.
        # Only apply when begin == 0 (top-level loops).  For nested loops with a
        # non-zero begin (e.g. inner tile of hl.tile(mb_cta.begin, mb_cta.end)),
        # bs_info.size is the range *length* (32), not the absolute end
        # (offset_0 + 32), so replacing ends[i] with bs_info.size would give a
        # wrong constant stop value and make all but the first outer iteration empty.
        if env.backend.name == "nki":
            ends = list(ends)
            proxy_ends = list(proxy_ends)
            for i, block_idx in enumerate(block_ids):
                begin_i = begins[i]
                begin_is_zero = (
                    (isinstance(begin_i, int) and begin_i == 0)
                    or (
                        isinstance(begin_i, torch.SymInt)
                        and begin_i._sympy_() == 0
                    )
                )
                if not begin_is_zero:
                    continue  # keep the original absolute end for nested loops
                bs_info = env.block_sizes[block_idx]
                if isinstance(bs_info.size, (int, torch.SymInt)):
                    ends[i] = bs_info.size
                    proxy_ends[i] = bs_info.size
            # Clamp block_size to loop extent: NKI's dma_copy uses
            # [offset:offset+block_size] which can go OOB when block_size
            # exceeds tensor dim (e.g. x_offsets[33] with block_size=64).
            _clamped_bs = list(block_sizes)
            for i, block_idx in enumerate(block_ids):
                try:
                    _numel = env.block_sizes[block_idx].numel
                except (AssertionError, AttributeError):
                    # Dynamic (tensor) bound: can't clamp at compile time
                    continue
                _num_int: int | None = None
                if isinstance(_numel, (int, bool)):
                    _num_int = int(_numel)
                elif isinstance(_numel, torch.SymInt):
                    try:
                        _num_int = env.size_hint(_numel)
                    except Exception:
                        _num_int = None
                elif isinstance(_numel, sympy.Expr):
                    try:
                        _num_int = int(_numel)
                    except (TypeError, ValueError):
                        try:
                            _num_int = int(env.size_hint(_numel))
                        except Exception:
                            _num_int = None
                cur_bs = block_sizes[i]
                if (_num_int is not None
                        and isinstance(cur_bs, (int, bool))
                        and int(cur_bs) > _num_int):
                    _clamped_bs[i] = _num_int
            if _clamped_bs != list(block_sizes):
                try:
                    config_block_sizes = state.config.config.get("block_sizes")
                    if isinstance(config_block_sizes, list):
                        for bid, new_bs in zip(block_ids, _clamped_bs, strict=True):
                            try:
                                idx = env.config_spec.block_sizes.block_id_to_index(bid)
                            except Exception:
                                continue
                            config_block_sizes[idx] = new_bs
                except Exception:
                    pass
                block_sizes = _clamped_bs
        block_id_to_info = {}
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for block_idx, block_size, begin, end, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if block_size != 1:
                block_size_var_for_constexpr = self.block_size_var(block_idx)
                assert block_size_var_for_constexpr is not None
                self._setup_block_size_constexpr(
                    state,
                    block_size_var_for_constexpr,
                    block_size,
                    block_idx=block_idx,
                )
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=block_size_var,
                    ),
                    begin=self._to_ast(begin, to_dtype=dtype),
                    end=self._to_ast(end, to_dtype=dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            assert for_node.body is body
            uses_thread_axis = self._uses_thread_axis(block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            bs = block_size_var if uses_thread_axis else "1"
            idx_stmts = _backend_loop_index_statements(
                env.backend,
                offset_var=offset_var,
                block_size_var=bs,
                dtype=dtype,
                axis=axis,
                index_var=index_var,
            )
            extra_body = list(idx_stmts)
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            # pyrefly: ignore [unsupported-operation]
            body[:] = [*extra_body, *body]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
            block_id_to_info=block_id_to_info,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self.mask_vars: dict[int, str | None] = {}
        self.l2_grouping = l2_grouping

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars.get(block_idx)

    def _setup_mask(
        self,
        state: CodegenState,
        block_idx: int,
        block_size: SymIntLike,
        index_var: str,
        end: object,
    ) -> ast.stmt | None:
        if (
            CompileEnvironment.current()
            .block_sizes[block_idx]
            .known_multiple(block_size)
        ):
            self.mask_vars[block_idx] = None
            return None
        self.mask_vars[block_idx] = mask_var = self.fn.new_var(
            f"mask_{block_idx}", dce=True
        )
        return statement_from_string(
            f"{mask_var} = ({index_var}) < {{end}}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class CuteNDTileStrategy(NDTileStrategy):
    """CuTe N-D tile strategy using the standard tile pipeline."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
        elements_per_thread: list[int] | None = None,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order, l2_grouping)
        assert isinstance(block_size, list)
        if elements_per_thread is None:
            elements_per_thread = [1 for _ in block_ids]
        assert len(elements_per_thread) == len(block_ids)
        self.elements_per_thread = elements_per_thread
        self._lane_var_by_block: dict[int, str] = {}
        for block_id, ept in zip(block_ids, elements_per_thread, strict=True):
            if ept > 1:
                self._lane_var_by_block[block_id] = self.fn.new_var(f"lane_{block_id}")

    def _ept_for_block(self, block_id: int) -> int:
        idx = self.block_ids.index(block_id)
        return self.elements_per_thread[idx]

    def _thread_extent_for_axis(
        self, block_id: int, block_size: SymIntLike
    ) -> SymIntLike:
        ept = self._ept_for_block(block_id)
        if ept == 1:
            return block_size
        if not isinstance(block_size, int):
            raise exc.BackendUnsupported(
                "cute",
                "elements_per_thread requires static ND block sizes for cute",
            )
        if block_size % ept != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "elements_per_thread must divide block size for cute axis "
                    f"{block_id}: {ept} does not divide {block_size}"
                ),
            )
        return block_size // ept

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def _thread_axis_map_with_ept(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                axis += 1
        return mapping

    def thread_axes_used(self) -> int:
        return sum(
            1
            for block_idx, block_size in zip(
                self.block_ids, self.block_size, strict=True
            )
            if self._uses_thread_axis_for_block(block_idx, block_size)
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            thread_extent = self._thread_extent_for_axis(
                block_id, block_size_by_id[block_id]
            )
            if self._uses_thread_axis_for_block(
                block_id, block_size_by_id[block_id]
            ) and isinstance(thread_extent, int):
                sizes.append(thread_extent)
        return sizes

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if all(ept == 1 for ept in self.elements_per_thread):
            return super().codegen_grid(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)

        lane_setup_statements: list[ast.AST] = []
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map_with_ept()
        for i, (block_idx, block_size, begin, end) in enumerate(
            reversed(
                self._reorder([*zip(block_ids, block_sizes, begins, ends, strict=True)])
            )
        ):
            block_size_info = env.block_sizes[block_idx]
            if block_size_info.size is None:
                numel = self._get_data_dependent_numel(state, end, begin)
            else:
                numel = block_size_info.numel
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(
                    state, block_size_var, block_size, block_idx=block_idx
                )
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")

            ept = self._ept_for_block(block_idx)
            uses_thread_axis = self._uses_thread_axis_for_block(block_idx, block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            if uses_thread_axis:
                idx_expr = f"{offset_var} + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {ept}"
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                idx_expr = f"{idx_expr} + cutlass.Int32({lane_var})"
            lane_setup_statements.append(
                statement_from_string(f"{index_var} = {idx_expr}")
            )

            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, numel
            )
            if mask_statement is not None:
                lane_setup_statements.append(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = [
            (self._lane_var_by_block[block_id], self._ept_for_block(block_id))
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_setup_statements=lane_setup_statements,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if all(ept == 1 for ept in self.elements_per_thread):
            return super().codegen_device_loop(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = user_body = []
        lane_loops = [
            (self._lane_var_by_block[block_id], self._ept_for_block(block_id))
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        for lane_var, extent in reversed(lane_loops):
            lane_for = create(
                ast.For,
                target=create(ast.Name, id=lane_var, ctx=ast.Store()),
                iter=expr_from_string(f"range({extent})"),
                body=body,
                orelse=[],
                type_comment=None,
            )
            body = [lane_for]
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        _, begins, ends, _ = state.ast_args
        _, _, proxy_ends, _ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map_with_ept()
        index_setup: list[ast.stmt] = []
        for block_idx, block_size, begin, end, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if block_size != 1:
                block_size_var_for_constexpr = self.block_size_var(block_idx)
                assert block_size_var_for_constexpr is not None
                self._setup_block_size_constexpr(
                    state,
                    block_size_var_for_constexpr,
                    block_size,
                    block_idx=block_idx,
                )
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=block_size_var,
                    ),
                    begin=self._to_ast(begin, to_dtype=dtype),
                    end=self._to_ast(end, to_dtype=dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            ept = self._ept_for_block(block_idx)
            uses_thread_axis = self._uses_thread_axis_for_block(block_idx, block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            if uses_thread_axis:
                idx_expr = f"{offset_var} + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {ept}"
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                idx_expr = f"{idx_expr} + cutlass.Int32({lane_var})"
            index_setup.append(statement_from_string(f"{index_var} = {idx_expr}"))
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                index_setup.append(mask_statement)
            body = [for_node]
        assert for_node is not None
        # Run index/mask setup once per loop-offset and per-lane before user body.
        user_body[:0] = index_setup
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
        )

    def supports_index_rank_expansion(self) -> bool:
        return False


class CuteFlattenedTileStrategy(FlattenedTileStrategy):
    """Flattened CuTe strategy: scalar index per thread over a flattened tile."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        elements_per_thread: int = 1,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self.elements_per_thread = elements_per_thread
        self._lane_var: str | None = None
        if elements_per_thread > 1:
            self._lane_var = self.new_var("lane", dce=False)

    def _thread_extent(self) -> SymIntLike:
        if self.elements_per_thread == 1:
            return self.block_size
        if not isinstance(self.block_size, int):
            raise exc.BackendUnsupported(
                "cute",
                "elements_per_thread requires static flattened block sizes for cute",
            )
        if self.block_size % self.elements_per_thread != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "elements_per_thread must divide flattened block size for cute: "
                    f"{self.elements_per_thread} does not divide {self.block_size}"
                ),
            )
        return self.block_size // self.elements_per_thread

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if not isinstance(thread_extent, int):
            return []
        return [thread_extent]

    def _uses_thread_axis(self) -> bool:
        thread_extent = self._thread_extent()
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if self.elements_per_thread == 1:
            return super().codegen_grid(state)

        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + cutlass.Int32({self._lane_var})"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        axis = self._flat_thread_axis()
        state.add_statement(
            f"{offsets_base_var} = ({pid_var}) * ({block_size_var}) + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {self.elements_per_thread}"
        )
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)
        block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = []
        if self._lane_var is not None:
            lane_loops = [(self._lane_var, self.elements_per_thread)]
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_setup_statements=lane_setup_statements,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if self.elements_per_thread == 1:
            return super().codegen_device_loop(state)

        env = CompileEnvironment.current()
        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + cutlass.Int32({self._lane_var})"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        lid = self.new_var("lid")
        end_var = env.backend.cdiv_expr(
            state.sympy_expr(total_numel), block_size_var, is_device=True
        )
        axis = self._flat_thread_axis()
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        user_body[:0] = lane_setup_statements
        if self._lane_var is not None:
            lane_for = create(
                ast.For,
                target=create(ast.Name, id=self._lane_var, ctx=ast.Store()),
                iter=expr_from_string(f"range({self.elements_per_thread})"),
                body=body,
                orelse=[],
                type_comment=None,
            )
            body = [lane_for]
        body[:0] = [
            statement_from_string(
                f"{offsets_base_var} = {lid} * ({block_size_var}) + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {self.elements_per_thread}"
            )
        ]
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
        )

    def offset_var(self, block_idx: int) -> str:
        return self._offsets_var

    def supports_index_rank_expansion(self) -> bool:
        return False


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_ids: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_ids=[*self.block_ids, *other.block_ids],
        )
