from __future__ import annotations

import ast
from collections import defaultdict
import dataclasses
import itertools
import math
import threading
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Protocol
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._inductor.codegen.triton import TritonPrinter
from torch.fx.graph import _Namespace

from .._compat import get_tensor_descriptor_fn_name
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import ReadWrites
from .ast_read_writes import ast_rename
from .ast_read_writes import dead_assignment_elimination
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .output_header import reserved_names
from .source_location import SyntheticLocation
from .variable_origin import BlockSizeOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .device_ir import HelperFunctionGraphInfo
    from .generate_ast import GenerateAST
    from .indexing_strategy import IndexingStrategy
    from .program_id import ProgramIDs

    _P = TypeVar("_P", bound="TensorPropertyArg")

    class _TLS(Protocol):
        functions: list[DeviceFunction]


tls: _TLS = cast("_TLS", threading.local())


class VarInfo(NamedTuple):
    """Information about a variable derived from a sympy expression."""

    name: str
    fx_node: torch.fx.Node


def find_block_size_symbols(
    expr: sympy.Expr,
) -> tuple[dict[sympy.Symbol, int], set[sympy.Symbol]]:
    """
    Find block size symbols in a sympy expression.

    Returns:
        tuple of (block_size_mapping, non_block_size_symbols) where:
        - block_size_mapping: dict mapping block size symbols to their block_id
        - non_block_size_symbols: set of symbols that are NOT block sizes
    """
    if not isinstance(expr, sympy.Expr):
        return {}, set()

    hf = HostFunction.current()
    block_sizes = {}
    non_block_size_symbols = set()

    for symbol in expr.free_symbols:
        # pyrefly: ignore [no-matching-overload]
        origin_info = hf.expr_to_origin.get(symbol)
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            # pyrefly: ignore [bad-argument-type]
            non_block_size_symbols.add(symbol)
        else:
            # pyrefly: ignore [unsupported-operation]
            block_sizes[symbol] = origin_info.origin.block_id

    # pyrefly: ignore[bad-return]
    return block_sizes, non_block_size_symbols


def contains_only_block_size_symbols(expr: sympy.Expr) -> bool:
    """Check if expression contains only block size symbols (no other variables)."""
    _, non_block = find_block_size_symbols(expr)
    return len(non_block) == 0


@dataclasses.dataclass
class Argument:
    name: str  # in the device function

    def host_str(self) -> str:
        raise NotImplementedError

    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name)

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)],)


@dataclasses.dataclass
class TensorArg(Argument):
    fake_value: torch.Tensor
    _host_str: str | None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError("TensorArg has no host representation")
        return self._host_str


@dataclasses.dataclass
class TensorDescriptorArg(TensorArg):
    # Permutation applied to make stride==1 dimension last
    permutation: list[int] | None = None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError(
                "TensorDescriptorArg is device-only and has no host representation"
            )
        return self._host_str

    @property
    def inverse_permutation(self) -> list[int]:
        """Get the inverse permutation to undo the applied permutation."""
        if (permutation := self.permutation) is None:
            raise RuntimeError("TensorDescriptorArg.permutation is None")
        inverse_perm = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse_perm[p] = i
        return inverse_perm


@dataclasses.dataclass
class TensorPropertyArg(Argument):
    tensor_arg: TensorArg
    dim: int

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)], self.tensor_arg.name, self.dim)


class TensorSizeArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.size({self.dim})"


class TensorStrideArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.stride({self.dim})"


@dataclasses.dataclass
class NumericArgument(Argument):
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


class ConstExprArg(NumericArgument):
    def arg_def_node(self) -> ast.arg:
        return create_arg(
            self.name, CompileEnvironment.current().backend.constexpr_type
        )


@dataclasses.dataclass
class SymbolArgument(NumericArgument):
    pass


class StaticShape(Argument):
    def __init__(self, val: int) -> None:
        super().__init__(repr(val))


_sort_order: dict[type[Argument], int] = {
    TensorDescriptorArg: 0,
    TensorArg: 0,
    TensorSizeArg: 1,
    TensorStrideArg: 2,
    SymbolArgument: 3,
    ConstExprArg: 4,
}


def _is_literal_constexpr(arg: ConstExprArg) -> bool:
    """Check if a constexpr arg has a known literal value that can be inlined at module level."""
    host_str = arg.host_str()
    if host_str == arg.name:
        return False
    try:
        ast.literal_eval(host_str)
        return True
    except (ValueError, SyntaxError):
        return False


class DeviceFunction:
    def __init__(
        self,
        name: str,
        config: Config,
        codegen: GenerateAST,
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.codegen = codegen
        self.arguments: list[Argument] = []
        self.preamble: list[ast.AST] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArg] = {}
        self._tensor_descriptor_args: dict[
            tuple[torch.Tensor, str], TensorDescriptorArg
        ] = {}
        self._expr_args: dict[sympy.Expr, SymbolArgument] = {}
        self._constexpr_args: dict[str, ConstExprArg] = {}
        self._constexpr_host_defs: set[str] = set()
        self._tensor_properties: dict[
            tuple[type[TensorPropertyArg], torch.Tensor, int], TensorPropertyArg
        ] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )
        self.pid: ProgramIDs | None = None
        self.namespace: _Namespace = _Namespace()
        self.namespace._used_names.update(reserved_names())
        self.namespace._used_names.update(
            # used by triton run() method
            [
                "grid",
                "warmup",
                "num_warps",
                "num_stages",
            ]
            + (
                ["num_ctas", "occupancy"]
                if CompileEnvironment.current().backend_name == "tileir"
                else []
            )
            + [
                x.removeprefix("_triton_config_")
                for x in config
                if x.startswith("_triton_config_")
            ]
        )
        self._variable_renames: dict[str, list[str]] = {}
        self._nki_tile_lists: dict[str, list[str]] = {}  # var → [var_0, var_1, ...]
        self._nki_sbuf_shapes: dict[str, list[int]] = {}  # var → [p, f]
        # Loop-index SBUF tiles are generated from a scalar offset plus iota.
        # Keep the scalar origin so index arithmetic such as ``tile.index % S``
        # can lower to NKI tensor ops even when S is a lifted local variable.
        self._nki_iota_offsets: dict[str, str] = {}
        self._nki_iota_block_sizes: dict[str, str] = {}
        # Names of kernel parameters that are Python scalars (float, int,
        # bool) — used by NKIOpOverrides to route binary ops to
        # tensor_scalar instead of tensor_tensor when the operand is a
        # host scalar.  Populated in codegen_function_def when building
        # the parameter list.
        self._nki_scalar_arg_names: set[str] = set()
        # Per-SBUF-var dtype tracking, used by broadcast / comparison codegen
        # to allocate matching output buffers and pass through nc_transpose /
        # nc_matmul constraints that require dst.dtype == data.dtype.
        self._nki_sbuf_dtypes: dict[str, str] = {}
        # HBM-source tracking: maps an SBUF var back to the HBM expression it
        # was DMA-loaded from, so partition-broadcast codegen can re-DMA with
        # a per-partition layout instead of tiling the existing SBUF tile.
        self._nki_hbm_sources: dict[str, str] = {}
        # Host tensor argument dtype overrides needed by NKI lowering. This is
        # used when an HBM dtype has broken/unsupported DMA semantics but the
        # Helion program immediately casts the value before computation.
        self._nki_host_arg_casts: dict[str, str] = {}
        # Loop-carry buffers that must not be modified in-place inside nested
        # device loops.
        self._nki_protected_vars: set[str] = set()
        # Per-output-tensor return buffer info for multi-output kernels.
        # Keyed by id(tensor). Keys are ints; values map to buf_name etc.
        self._nki_return_buffers: dict[int, dict] = {}
        # Tile lists where tiles represent the FREE dimension (not partition).
        # These must be distributed along dim 1 when stored to a 2D buffer.
        self._nki_free_dim_tile_lists: set[str] = set()
        # PSUM-reuse alias map: keys are the SBUF result variable names that
        # _nki_dot *would* have created for a matmul; values are the PSUM
        # buffer names the matmul actually writes to. Populated by _nki_dot
        # when the matmul's FX node carries meta["nki_keep_in_psum"]=True.
        # Consumer ops (_nki_tensor_tensor / _nki_tensor_scalar /
        # _nki_activation) resolve a data operand through this map to read
        # from PSUM directly and skip the intermediate SBUF copy.
        self._nki_psum_aliases: dict[str, str] = {}
        # FX-node-name -> SBUF result var, so a consumer (which only has
        # its predecessor's FX node name via meta["nki_read_psum_from"])
        # can look up the SBUF name that _nki_dot produced.
        self._nki_fx_matmul_vars: dict[str, str] = {}
        self.dce_vars: list[str] = []
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}
        self.expr_to_var_info: dict[sympy.Expr, VarInfo] = {}
        self.deferred_rdim_defs: list[tuple[str, sympy.Expr]] = []

        from .helper_function import HelperFunctionManager

        self.helper_manager = HelperFunctionManager()

        from .tile_dispatch import TileStrategyDispatch

        self.tile_strategy: TileStrategyDispatch = TileStrategyDispatch(self, config)

        # Store indexing config to lazily create strategies per load/store
        self._indexing_config = config.indexing
        self.indexing_strategies: list[IndexingStrategy] = []

        self.rng_seed_count = 0
        self.device_load_index = 0
        self.device_store_index = 0
        # Single counter for both loads and stores for indexing assignment
        self.device_memory_op_index = 0
        self.rng_seed_buffer_param_name = None

    def get_indexing_strategy(self, index: int) -> IndexingStrategy:
        from .indexing_strategy import IndexingStrategy
        from .indexing_strategy import PointerIndexingStrategy

        # Expand strategies list if needed
        while len(self.indexing_strategies) <= index:
            idx = len(self.indexing_strategies)

            if isinstance(self._indexing_config, str):
                # Single string: all loads/stores use the same strategy
                if not self.indexing_strategies:
                    strategy = IndexingStrategy.select(self._indexing_config)
                else:
                    strategy = self.indexing_strategies[0]
            elif isinstance(self._indexing_config, list) and self._indexing_config:
                # List: one strategy per load/store
                assert idx < len(self._indexing_config), (
                    f"Load/Store operation {idx} exceeds indexing config length "
                    f"{len(self._indexing_config)}. Please specify indexing for all loads and stores."
                )
                strategy = IndexingStrategy.select(self._indexing_config[idx])
            else:
                # Empty/default: use pointer
                strategy = PointerIndexingStrategy()

            self.indexing_strategies.append(strategy)

        return self.indexing_strategies[index]

    def has_rng_ops(self) -> bool:
        """Check if this kernel uses any RNG operations."""
        return self.rng_seed_count > 0 and self.rng_seed_buffer_param_name is not None

    def allocate_rng_seed(self) -> int:
        """Allocate a new RNG seed index and ensure buffer argument exists.

        Returns:
            The seed index for this RNG operation.
        """
        seed_index = self.rng_seed_count
        self.rng_seed_count += 1

        # Ensure seed buffer parameter name exists
        if self.rng_seed_buffer_param_name is None:
            # pyrefly: ignore [bad-assignment]
            self.rng_seed_buffer_param_name = self.new_var("rng_seed_buffer")

        return seed_index

    def block_size_var(self, block_id: int) -> str | None:
        key = (block_id,)
        env = CompileEnvironment.current()

        # Block size var could be used outside of a hl.tile loop, and at that point
        # no tile strategy has populated the cache yet, so we must lazily create
        # the constexpr argument here and lift it as device function argument;
        # later strategies will reuse the cached name or intentionally replace it
        # (e.g. flattened loops, reductions).
        if key not in self.block_size_var_cache:
            block_value = env.block_sizes[block_id].from_config(self.config)

            if block_value is None:
                return None

            var_name = self.new_var(f"_BLOCK_SIZE_{block_id}")
            self.block_size_var_cache[key] = var_name
            self.constexpr_arg_with_host_def(var_name, block_value)
        else:
            # Some strategies pre-populate the cache with a symbol name only.
            # Ensure the constexpr argument/host-side definition exists.
            var_name = self.block_size_var_cache[key]
            if var_name.isidentifier() and var_name not in self._constexpr_args:
                block_value = env.block_sizes[block_id].from_config(self.config)
                if block_value is not None:
                    self.constexpr_arg_with_host_def(var_name, block_value)

        return self.block_size_var_cache[key]

    def try_map_block_symbols_to_vars(self, expr: sympy.Expr) -> sympy.Expr | None:
        """Try to map all block size symbols in expression to their variable names.

        Returns:
            - The expression with symbols replaced if ALL symbols are block sizes and have variables
            - None if the expression contains non-block symbols or unmapped block symbols
        """
        block_mapping, non_block_symbols = find_block_size_symbols(expr)

        # Can't map if there are non-block symbols
        if non_block_symbols:
            return None

        # No symbols to map - return as-is
        if not block_mapping:
            return expr

        # Try to map all block symbols to their variables
        var_map = {}
        for symbol, block_id in block_mapping.items():
            block_var = self.block_size_var(block_id)
            if not block_var:
                # Can't map this block symbol - fail
                return None
            var_map[symbol] = sympy.Symbol(block_var, integer=True)

        # Successfully mapped all symbols
        # pyrefly: ignore [bad-return]
        return expr.xreplace(var_map)

    def merge_variable_names(self, a: str, b: str) -> None:
        name_group = [
            *self._variable_renames.get(a, [a]),
            *self._variable_renames.get(b, [b]),
        ]
        for n in name_group:
            self._variable_renames[n] = name_group
        tile_vars: list[str] | None = None
        for n in name_group:
            if n in self._nki_tile_lists:
                tile_vars = self._nki_tile_lists[n]
                break
        if tile_vars is not None:
            for n in name_group:
                self._nki_tile_lists[n] = tile_vars

    def set_pid(self, pid: ProgramIDs) -> None:
        assert self.pid is None, "pid already set"
        self.pid = pid

    def sympy_expr(self, expr: sympy.Expr) -> str:
        env = CompileEnvironment.current()
        expr = env.specialize_expr(env.shape_env.simplify(expr))
        if not expr.free_symbols:
            return texpr(expr)
        if expr in self.expr_to_var_info:
            return self.expr_to_var_info[expr].name
        expr_to_origin = HostFunction.current().expr_to_origin
        if expr in expr_to_origin:
            return self._lift_sympy_arg(expr)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            if sym in self.expr_to_var_info:
                replacements[sym] = sympy.Symbol(
                    self.expr_to_var_info[sym].name, integer=True
                )
            else:
                assert sym in expr_to_origin, f"no origin found for {sym.name}"
                replacements[sym] = sympy.Symbol(
                    self._lift_sympy_arg(sym), integer=True
                )
        # pyrefly: ignore [bad-argument-type]
        return texpr(expr.xreplace(replacements))

    def _lift_sympy_arg(self, expr: sympy.Expr) -> str:
        origin = HostFunction.current().expr_to_origin[expr]
        if isinstance(origin.origin, TensorSizeOrigin):
            assert origin.fake_value is not None
            arg = self.tensor_size(
                origin.fake_value,
                origin.origin.key,
            )
            return arg.name
        if isinstance(origin.origin, BlockSizeOrigin):
            result = self.block_size_var(origin.origin.block_id)
            assert result is not None
            return result
        if isinstance(origin.origin, GridOrigin):
            return self.codegen.offset_var(origin.origin.block_id)
        return self.expr_arg(expr, origin.origin).name

    def user_sympy_expr(self, expr: sympy.Expr) -> str:
        """A sympy expression that flows into user computations."""
        expr_to_origin = HostFunction.current().expr_to_origin
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda s: s.name):
            assert isinstance(sym, sympy.Symbol)
            origin_info = expr_to_origin.get(sym)
            if origin_info is None:
                continue
            origin = origin_info.origin
            if isinstance(origin, BlockSizeOrigin):
                replacements[sym] = self.tile_strategy.user_size(origin.block_id)
        if replacements:
            # pyrefly: ignore [bad-assignment]
            expr = expr.xreplace(replacements)
        return self.sympy_expr(expr)

    def literal_expr(self, expr: object) -> str:
        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.sympy_expr(expr._sympy_())
        if isinstance(expr, sympy.Expr):
            return self.sympy_expr(expr)
        if isinstance(expr, float) and not math.isfinite(expr):
            return f"float('{expr}')"
        return repr(expr)

    def unique_name(self, prefix: str, dce: bool = False) -> str:
        return self.new_var(f"{prefix}_{next(self._unique_counter[prefix])}", dce=dce)

    def new_var(self, name: str, *, dce: bool = False) -> str:
        name = self.namespace.create_name(name, None)
        if dce:
            self.dce_vars.append(name)
        return name

    def register_tile_list(self, var_name: str, tile_vars: list[str]) -> None:
        """Register var_name as a tile list backed by the given individual variables."""
        if len(tile_vars) > 1:
            self._nki_tile_lists[var_name] = tile_vars

    def get_tile_list_vars(self, var_name: str) -> list[str] | None:
        """Return individual tile variable names if var_name is a tile list, else None.

        Also searches through the variable rename chain in case var_name is an
        alias (e.g. phi-merged name) that hasn't been registered directly.
        """
        result = self._nki_tile_lists.get(var_name)
        if result is not None:
            return result
        rename_group = self._variable_renames.get(var_name)
        if rename_group:
            for alias in rename_group:
                result = self._nki_tile_lists.get(alias)
                if result is not None:
                    self._nki_tile_lists[var_name] = result
                    return result
        return None

    def get_tile_list_count(self, var_name: str) -> int:
        tile_vars = self._nki_tile_lists.get(var_name)
        return len(tile_vars) if tile_vars else 1

    def propagate_tile_list(self, src: str, dst: str) -> None:
        tile_vars = self._nki_tile_lists.get(src)
        if tile_vars:
            self._nki_tile_lists[dst] = tile_vars

    def tensor_arg(
        self, fake_value: torch.Tensor, prefer_name: str | None = None
    ) -> TensorArg:
        host_function = HostFunction.current()
        if fake_value not in host_function.tensor_to_origin:
            origin = self._recover_captured_tensor_origin(fake_value, host_function)
            if origin is not None:
                host_function.tensor_to_origin[fake_value] = origin
        if fake_value not in self._tensor_args:
            origin = host_function.tensor_to_origin[fake_value]
            arg = TensorArg(
                self.new_var(prefer_name or origin.suggest_var_name()),
                fake_value,
                origin.host_str(),
            )
            self.arguments.append(arg)
            self._tensor_args[fake_value] = arg
        return self._tensor_args[fake_value]

    @staticmethod
    def _recover_captured_tensor_origin(
        fake_value: torch.Tensor,
        host_function: HostFunction,
    ) -> Origin | None:
        """Recover origins for tensors captured by helper function closures.

        Dynamo can represent a captured tensor inside a nested helper graph as an
        ``aten.empty_strided`` constant. That produces a tensor with matching
        metadata but no direct entry in ``tensor_to_origin``. Restrict recovery
        to renamed origins (globals/closures), so ordinary kernel parameters
        continue to use the explicit ``host_tensor`` path.
        """

        candidates: dict[str, Origin] = {}
        fake_shape = tuple(fake_value.size())
        fake_stride = tuple(fake_value.stride())
        for known_tensor, origin in host_function.tensor_to_origin.items():
            if not origin.needs_rename():
                continue
            if known_tensor.dtype != fake_value.dtype:
                continue
            if tuple(known_tensor.size()) != fake_shape:
                continue
            if tuple(known_tensor.stride()) != fake_stride:
                continue
            candidates[origin.host_str()] = origin
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        return None

    def tensor_descriptor_arg(
        self, fake_value: torch.Tensor, block_size: list[int | torch.SymInt]
    ) -> TensorDescriptorArg:
        host_function = HostFunction.current()
        block_size_expr = ", ".join(map(self.literal_expr, block_size))
        key = (fake_value, block_size_expr)
        if key not in self._tensor_descriptor_args:
            origin = host_function.tensor_to_origin[fake_value]
            desc_name = self.new_var(origin.suggest_var_name() + "_desc")
            env = CompileEnvironment.current()

            # Find which dimension has stride==1
            stride_one_dim = [*map(env.size_hint, fake_value.stride())].index(1)

            # Determine if we need permutation (stride==1 dimension is not last)
            permutation = None
            if stride_one_dim != fake_value.ndim - 1:
                # Create permutation to move stride==1 dimension to last position
                permutation = [*range(fake_value.ndim)]
                permutation.pop(stride_one_dim)
                permutation.append(stride_one_dim)

            # Create the regular tensor arg and size/stride args
            tensor_arg = self.tensor_arg(fake_value)
            size_args = [
                self.tensor_size(fake_value, i) for i in range(fake_value.ndim)
            ]
            stride_args = [
                self.tensor_stride(fake_value, i) for i in range(fake_value.ndim)
            ]

            # Apply permutation if needed
            if permutation is not None:
                size_args = [size_args[i] for i in permutation]
                stride_args = [stride_args[i] for i in permutation]
                block_size = [block_size[i] for i in permutation]
                # Update block_size_expr for the permuted order
                block_size_expr = ", ".join(map(self.literal_expr, block_size))

            # Add tl.make_tensor_descriptor call to preamble
            sizes = ", ".join([arg.name for arg in size_args])
            strides = ", ".join([arg.name for arg in stride_args])

            tensor_descriptor_fn_name = get_tensor_descriptor_fn_name()
            descriptor_stmt = statement_from_string(
                f"{desc_name} = {tensor_descriptor_fn_name}({tensor_arg.name}, [{sizes}], [{strides}], [{block_size_expr}])"
            )
            self.preamble.append(descriptor_stmt)

            arg = TensorDescriptorArg(
                desc_name,
                fake_value,
                None,  # No host_str since this is device-only
                permutation,
            )
            # Don't add to self.arguments since this is device-only
            self._tensor_descriptor_args[key] = arg
        return self._tensor_descriptor_args[key]

    def expr_arg(self, sym: sympy.Expr, origin: Origin) -> SymbolArgument:
        if sym not in self._expr_args:
            arg = SymbolArgument(
                name=self.new_var(origin.suggest_var_name()),
                _host_str=origin.host_str(),
            )
            self.arguments.append(arg)
            self._expr_args[sym] = arg
            # NKI: register Python-scalar kernel parameters so the
            # NKIOpOverrides cast / binary-op paths can detect them and
            # route to memset/tensor_scalar instead of tensor_tensor.
            self._nki_scalar_arg_names.add(arg.name)
        return self._expr_args[sym]

    def constexpr_arg(self, name: str, value: object | None = None) -> bool:
        """Create a constexpr argument, returns True if created, False if already exists."""
        if name in self._constexpr_args:
            return False
        host_str = name if value is None else self._format_constexpr_value(value)
        self._constexpr_args[name] = rv = ConstExprArg(name, host_str)
        self.arguments.append(rv)
        return True

    def constexpr_arg_with_host_def(self, name: str, value: object) -> None:
        """Create a constexpr argument and add its host-side definition if needed."""
        created = self.constexpr_arg(name, value)
        host_expr = self._constexpr_args[name].host_str()
        if created or name not in self._constexpr_host_defs:
            self.codegen.host_statements.append(
                statement_from_string(f"{name} = {host_expr}")
            )
        self._constexpr_host_defs.add(name)

    def _format_constexpr_value(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return repr(value)

        # Extract sympy expression from torch symbolic types
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value._sympy_()

        # Handle sympy expressions (sanitize by replacing triton_helpers functions)
        if isinstance(value, sympy.Expr):
            # type: ignore [missing-attribute]
            sanitized = value.replace(
                lambda node: (
                    isinstance(node, sympy.Function)
                    and getattr(node.func, "__name__", "")
                    == "triton_helpers.div_floor_integer"
                ),
                lambda node: sympy.floor(node.args[0] / node.args[1]),
            ).replace(
                lambda node: (
                    isinstance(node, sympy.Function)
                    and getattr(node.func, "__name__", "")
                    == "triton_helpers.remainder_integer"
                ),
                lambda node: sympy.Mod(node.args[0], node.args[1]),
            )
            expr = cast("sympy.Expr", sanitized)
            return HostFunction.current().sympy_expr(expr)

        return HostFunction.current().literal_expr(value)

    def _tensor_property(
        self,
        prop_cls: type[_P],
        fake_value: torch.Tensor,
        dim: int,
        prefix: str,
    ) -> _P:
        # TODO(jansel): dedupe based on sympy expressions
        key = (prop_cls, fake_value, dim)
        if key not in self._tensor_properties:
            arg = self.tensor_arg(fake_value)
            prop = prop_cls(f"{arg.name}_{prefix}_{dim}", arg, dim)
            self.arguments.append(prop)
            self._tensor_properties[key] = prop
        return cast("_P", self._tensor_properties[key])

    def tensor_size(self, fake_value: torch.Tensor, dim: int) -> Argument:
        if isinstance(v := fake_value.size(dim), int) or isinstance(
            v._sympy_(), sympy.Integer
        ):
            return StaticShape(int(v))
        return self._tensor_property(TensorSizeArg, fake_value, dim, "size")

    def tensor_stride(self, fake_value: torch.Tensor, dim: int) -> Argument:
        v = fake_value.stride(dim)
        env = CompileEnvironment.current()
        # Check if this stride was explicitly specialized
        source = env.input_sources.get(fake_value)
        if (
            isinstance(source, LocalSource)
            and (source.local_name, dim) in env.specialized_strides
        ):
            return StaticShape(int(v))
        if isinstance(v, int):
            if env.settings.static_shapes:
                return StaticShape(v)
        return self._tensor_property(TensorStrideArg, fake_value, dim, "stride")

    def sorted_args(self) -> list[Argument]:
        self.arguments.sort(key=lambda arg: arg.sort_key())
        return self.arguments

    def _register_nki_dynamic_tensor_size_args(self) -> None:
        env = CompileEnvironment.current()
        if env.backend.name != "nki" or env.settings.static_shapes:
            return
        for arg in list(self.arguments):
            if isinstance(arg, TensorArg):
                for dim, size in enumerate(arg.fake_value.size()):
                    if isinstance(size, torch.SymInt):
                        expr = env.shape_env.simplify(size._sympy_())
                        if expr.free_symbols:
                            self.tensor_size(arg.fake_value, dim)

    def _nki_return_statements(self) -> list[ast.stmt]:
        """Generate return statement(s) for NKI kernel with one or more output buffers."""
        bufs = getattr(self, "_nki_return_buffers", None)
        if not bufs:
            single = getattr(self, "_nki_return_buffer_name", None)
            if single is not None:
                return [statement_from_string(f"return {single}")]
            return []
        buf_names = [info["buf_name"] for info in bufs.values()]
        if len(buf_names) == 1:
            return [statement_from_string(f"return {buf_names[0]}")]
        return [statement_from_string(f"return ({', '.join(buf_names)})")]

    @staticmethod
    def _is_nki_sbuf_allocation(stmt: ast.stmt) -> str | None:
        if (
            not isinstance(stmt, ast.Assign)
            or len(stmt.targets) != 1
            or not isinstance(stmt.targets[0], ast.Name)
            or not isinstance(stmt.value, ast.Call)
            or not isinstance(stmt.value.func, ast.Attribute)
            or not isinstance(stmt.value.func.value, ast.Name)
            or stmt.value.func.value.id != "nl"
            or stmt.value.func.attr != "ndarray"
        ):
            return None
        for keyword in stmt.value.keywords:
            if (
                keyword.arg == "buffer"
                and isinstance(keyword.value, ast.Attribute)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == "nl"
                and keyword.value.attr == "sbuf"
            ):
                return stmt.targets[0].id
        return None

    def _should_rewrite_nki_sbuf_reassign(
        self,
        target: str,
        source: str,
        visible_allocs: dict[str, int],
        depth: int,
    ) -> bool:
        if target == source or not target.startswith("_nki") or "_copy" in target:
            return False
        target_depth = visible_allocs.get(target)
        source_depth = visible_allocs.get(source)
        if target_depth is None or source_depth is None:
            return False
        if target_depth > depth or source_depth > depth:
            return False
        target_shape = self._nki_sbuf_shapes.get(target)
        source_shape = self._nki_sbuf_shapes.get(source)
        if target_shape is None or source_shape is None or target_shape != source_shape:
            return False
        target_dtype = self._nki_sbuf_dtypes.get(target)
        source_dtype = self._nki_sbuf_dtypes.get(source)
        return not (
            target_dtype is not None
            and source_dtype is not None
            and target_dtype != source_dtype
        )

    def _rewrite_nki_sbuf_reassignments(
        self, statements: list[ast.stmt], visible_allocs: dict[str, int], depth: int
    ) -> None:
        index = 0
        while index < len(statements):
            stmt = statements[index]
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Name)
                and self._should_rewrite_nki_sbuf_reassign(
                    stmt.targets[0].id, stmt.value.id, visible_allocs, depth
                )
            ):
                statements[index] = statement_from_string(
                    f"nisa.tensor_copy(dst={stmt.targets[0].id}, src={stmt.value.id})"
                )
                index += 1
                continue

            if allocated := self._is_nki_sbuf_allocation(stmt):
                existing_depth = visible_allocs.get(allocated)
                if (
                    existing_depth is not None
                    and existing_depth < depth
                    and allocated in self._nki_sbuf_shapes
                    and "_copy" not in allocated
                ):
                    del statements[index]
                    continue
                visible_allocs[allocated] = depth

            if isinstance(stmt, (ast.For, ast.While)):
                self._rewrite_nki_sbuf_reassignments(
                    stmt.body, visible_allocs.copy(), depth + 1
                )
                self._rewrite_nki_sbuf_reassignments(
                    stmt.orelse, visible_allocs.copy(), depth + 1
                )
            elif isinstance(stmt, ast.If):
                self._rewrite_nki_sbuf_reassignments(
                    stmt.body, visible_allocs.copy(), depth + 1
                )
                self._rewrite_nki_sbuf_reassignments(
                    stmt.orelse, visible_allocs.copy(), depth + 1
                )
            index += 1

    def codegen_function_def(self) -> list[ast.stmt]:
        prefix = []
        if self._tensor_descriptor_args:
            prefix.append(
                statement_from_string("helion.runtime.set_triton_allocator()")
            )

        env = CompileEnvironment.current()
        backend = env.backend
        self._register_nki_dynamic_tensor_size_args()

        sorted_arguments = self.sorted_args()

        # Separate constexpr args: inline those with known literal values at
        # module level, keep dynamic ones as function parameters
        constexpr_to_inline = [
            arg
            for arg in sorted_arguments
            if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg)
        ]
        inlined_names = {arg.name for arg in constexpr_to_inline}
        param_args = [
            arg
            for arg in sorted_arguments
            if not isinstance(arg, ConstExprArg) or arg.name not in inlined_names
        ]

        args = [arg.arg_def_node() for arg in param_args]
        if self.has_rng_ops():
            # Add the seed buffer as a pointer parameter to kernel signature
            assert self.rng_seed_buffer_param_name is not None
            args.append(create_arg(self.rng_seed_buffer_param_name))

        # Generate inlined constexpr assignments at module level
        # (e.g., _BLOCK_SIZE_0 = tl.constexpr(256))
        # Use SyntheticLocation to suppress source origin comments on these statements
        with SyntheticLocation():
            for arg in constexpr_to_inline:
                self.codegen.module_statements.append(
                    statement_from_string(
                        backend.inline_constexpr(arg.name, arg.host_str())
                    )
                )

        # Generate preamble to dereference scalar refs (e.g., Pallas 0-dim tensors)
        scalar_preamble: list[ast.AST] = []
        for arg in param_args:
            scalar_preamble.extend(backend.scalar_arg_preamble(arg))

        tensor_shape_preamble: list[ast.AST] = []
        if backend.name == "nki":
            # Normalize tensor argument views at kernel entry to the 2D logical
            # shape expected by codegen. Dynamic kernels use lifted tensor-size
            # arguments here so one bound kernel can cover multiple input sizes.
            def _nki_shape_part(arg: TensorArg, dim: int, s: object) -> str:
                if isinstance(s, int):
                    return str(s)
                if isinstance(s, torch.SymInt):
                    size_arg = self.tensor_size(arg.fake_value, dim)
                    return size_arg.name
                return self.literal_expr(s)

            # Collect names of non-tensor scalar parameters (e.g.
            # ``alpha: float``, ``scale: float``) so binary ops can
            # route ``tile * alpha`` to ``tensor_scalar`` instead of
            # ``tensor_tensor``.
            for arg in param_args:
                if not isinstance(arg, TensorArg):
                    name = getattr(arg, "name", None)
                    if isinstance(name, str):
                        self._nki_scalar_arg_names.add(name)

            for arg in param_args:
                if isinstance(arg, TensorArg):
                    # int64 args are cast to int32 at the launcher (see
                    # default_nki_launcher) and also in _to_fake_tensor so
                    # the traced kernel is declared with int32. No in-kernel
                    # cast needed here.
                    shape_parts = [
                        _nki_shape_part(arg, dim, s)
                        for dim, s in enumerate(arg.fake_value.size())
                    ]
                    if arg.fake_value.dim() == 1:
                        # Reshape [N] → [1, N] so the large dimension is the
                        # free axis.  This avoids tile-list fragmentation for
                        # broadcast vectors like weight/bias and keeps the
                        # partition dimension at 1 for natural broadcasting.
                        shape_parts = ["1"] + shape_parts
                    elif arg.fake_value.dim() > 2:
                        # Reshape 3D+ tensors to 2D by combining all leading
                        # dims into the first dim.  E.g. [B,M,K] → [B*M, K].
                        # NKI SBUF only supports 2D tiles; DMA requires src/dst
                        # dims to match. The batch loop uses b_idx*dim + offset
                        # to index into this flattened tensor.
                        leading = shape_parts[:-1]
                        flat = " * ".join(leading)
                        shape_parts = [f"({flat})"] + shape_parts[-1:]
                    tensor_shape_preamble.append(
                        statement_from_string(
                            f"{arg.name} = {arg.name}.reshape([{', '.join(shape_parts)}])"
                        )
                    )

        fn_def = ast_rename(
            create(
                ast.FunctionDef,
                name=self.name,
                args=create_arguments(args),
                body=[
                    *scalar_preamble,
                    *tensor_shape_preamble,
                    *self.preamble,
                    *self.body,
                    *self._nki_return_statements(),
                ],
                decorator_list=[expr_from_string(backend.function_decorator)]
                if backend.function_decorator
                else [],
                type_params=[],
            ),
            {k: v[0] for k, v in self._variable_renames.items()},
        )
        if backend.name == "nki":
            self._rewrite_nki_sbuf_reassignments(fn_def.body, {}, 0)

        return [
            *prefix,
            fn_def,
        ]

    def codegen_function_call(self) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend
        self._register_nki_dynamic_tensor_size_args()

        args: list[str] = []
        tensor_host_args: list[str] = []
        for arg in self.sorted_args():
            # Skip constexpr args that are inlined at module level
            if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg):
                continue
            if isinstance(arg, ConstExprArg) and arg.name in self._constexpr_host_defs:
                host_arg = arg.name
            else:
                host_arg = arg.host_str()
            if isinstance(arg, TensorArg):
                tensor_host_args.append(host_arg)
            host_arg = backend.transform_host_arg(arg, host_arg, tensor_host_args)
            args.append(host_arg)

        pid = self.pid
        assert pid is not None

        call_grid_expr = pid.codegen_grid()
        call_args = backend.build_launcher_args(
            args,
            tensor_host_args=tensor_host_args,
            has_rng_ops=self.has_rng_ops(),
            config=self.config,
            has_barrier=env.has_barrier,
        )
        bufs = getattr(self, "_nki_return_buffers", None)
        call_str = f"_launcher({self.name}, {{call_grid_expr}}, {', '.join(call_args)})"
        if bufs and len(bufs) > 1:
            result_var = "_nki_result"
            call_str = f"{result_var} = {call_str}"
            post_stmts: list[ast.stmt] = []
            for i, info in enumerate(bufs.values()):
                host_var = info["host_var"]
                reshape = info["host_reshape"]
                if reshape is not None:
                    post_stmts.append(
                        statement_from_string(
                            f"{host_var} = {result_var}[{i}].reshape({reshape})"
                        )
                    )
                else:
                    post_stmts.append(
                        statement_from_string(f"{host_var} = {result_var}[{i}]")
                    )
            self._nki_post_call_stmts = post_stmts
        else:
            return_host_var = getattr(self, "_nki_return_host_var", None)
            return_host_reshape = getattr(self, "_nki_return_host_reshape", None)
            if return_host_var is not None:
                if return_host_reshape is not None:
                    call_str = f"{return_host_var} = {call_str}.reshape({return_host_reshape})"
                else:
                    call_str = f"{return_host_var} = " + call_str
        # TODO(jansel): we should run CSE this statement
        call_statement = statement_from_string(
            call_str,
            call_grid_expr=call_grid_expr,
        )
        assert isinstance(call_statement, ExtendedAST)
        # Mark the kernel call we can find it in codegen_precompile_def
        call_statement._is_kernel_call = True
        return call_statement

    def dead_code_elimination(self) -> None:
        """
        Remove variables that are not used in the function body.
        """

        for _ in range(8):
            rw = ReadWrites.from_list([*self.preamble, *self.body])
            dead_assignment_elimination(self.body, self.dce_vars, 1, rw)
            dead_assignment_elimination(self.preamble, self.dce_vars, 1, rw)

        # drop any unused args
        args_to_remove = {
            arg.name
            for arg in self.arguments
            # pyrefly: ignore [unbound-name]
            if arg.name not in rw.reads
        }
        if args_to_remove:
            self.arguments = [
                arg for arg in self.arguments if arg.name not in args_to_remove
            ]
            for cache in cast(
                "list[dict[object, Argument]]",
                [
                    self._tensor_args,
                    self._tensor_descriptor_args,
                    self._expr_args,
                    self._tensor_properties,
                ],
            ):
                for k, v in [*cache.items()]:
                    if v.name in args_to_remove:
                        del cache[k]

    def register_helper_function(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> None:
        """Register a helper function to be generated at global scope."""
        name = self.namespace.create_name(helper_graph_info.name, None)
        self.helper_manager.register_helper_function(helper_graph_info, name)

    def codegen_helper_functions(self) -> list[ast.stmt]:
        """Generate helper function definitions at global scope."""
        return self.helper_manager.codegen_helper_functions()

    def flush_deferred_rdim_defs(self, codegen: GenerateAST) -> None:
        """Add all deferred RDIM definitions to host statements."""
        backend = CompileEnvironment.current().backend
        for var_name, expr in self.deferred_rdim_defs:
            expr_str = HostFunction.current().sympy_expr(expr)
            stmt = statement_from_string(
                f"{var_name} = {backend.next_power_of_2_host_expr(expr_str)}"
            )
            codegen.host_statements.append(stmt)
        self.deferred_rdim_defs.clear()

    def __enter__(self) -> None:
        try:
            tls.functions.append(self)
        except AttributeError:
            tls.functions = [self]

    def __exit__(self, *args: object) -> None:
        tls.functions.pop()

    @staticmethod
    def current() -> DeviceFunction:
        try:
            return tls.functions[-1]
        except (AttributeError, IndexError):
            raise NoCurrentFunction from None


class HelionTritonPrinter(TritonPrinter):
    """Custom Triton printer that avoids wrapping float literals in tl.full().

    Inductor's default TritonPrinter prints SymPy Float as a 0-D Triton value
    via tl.full([], <val>, tl.float64). We override this to emit the raw numeric
    literal, letting downstream type promotion and casts handle dtype.
    """

    def _print_Float(self, expr: sympy.Expr) -> str:
        return str(expr)

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert expr.func.__name__ == "ToFloat" and len(expr.args) == 1
        # pyrefly: ignore [missing-attribute]
        return f"{self._print(expr.args[0])} + 0.0"


class HelionNKIPrinter(HelionTritonPrinter):
    """Python-only expression printer for the NKI backend.

    TritonPrinter emits ``triton_helpers.div_floor_integer`` and
    ``triton_helpers.remainder_integer`` for FloorDiv / Mod on runtime
    integer expressions, both of which are unresolvable inside a
    @nki.jit kernel. We override to emit plain ``(a) // (b)`` and
    ``(a) % (b)`` which evaluate the same way on host-Python integers
    and are accepted by the NKI parser.
    """

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, y = expr.args
        return f"({self._print(x)} // {self._print(y)})"

    def _print_Mod(self, expr: sympy.Expr) -> str:
        x, y = expr.args
        return f"({self._print(x)} % {self._print(y)})"

    # sympy.floor(a / b) over int-valued args maps to a floordiv too.
    def _print_floor(self, expr: sympy.Expr) -> str:
        (arg,) = expr.args
        if arg.is_Rational or (
            isinstance(arg, sympy.Mul)
            and len(arg.args) == 2
            and arg.args[1].is_Pow
        ):
            # a / b style — fall through to default, which may still
            # produce a helper. We keep this guard conservative.
            pass
        return super()._print_floor(expr)  # type: ignore[misc]


def texpr(expr: sympy.Expr) -> str:
    try:
        from .compile_environment import CompileEnvironment
        env = CompileEnvironment.current()
        if env.backend.name == "nki":
            return HelionNKIPrinter().doprint(expr)
    except Exception:
        pass
    return HelionTritonPrinter().doprint(expr)
