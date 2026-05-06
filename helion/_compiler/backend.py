from __future__ import annotations

import abc
import ast
import functools
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import torch

from .. import exc
from .ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from torch._inductor.ops_handler import OpsHandler

    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .device_function import Argument
    from .device_function import DeviceFunction
    from .tile_strategy import TileStrategy

    InductorOpOverrides = OpsHandler[Any]


class Backend(abc.ABC):
    """Abstract base class for Helion code generation backends.

    Each backend is responsible for defining:
    - How types are represented in generated code
    - What imports are needed in generated code
    - What decorators and annotations are used on generated functions
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend name used for codegen dispatch (e.g., 'triton')."""
        ...

    @property
    def codegen_name(self) -> str:
        """Backend name used to look up registered codegen functions."""
        return self.name

    @abc.abstractmethod
    def dtype_str(self, dtype: torch.dtype) -> str:
        """Convert a torch dtype to a backend-specific type string.

        For example, Triton returns 'tl.float32' for torch.float32.
        """
        ...

    @abc.abstractmethod
    def acc_type(self, dtype: torch.dtype) -> str:
        """Get the accumulator type string for reductions.

        Some backends may promote certain types for numerical stability
        during reductions (e.g., fp16 -> fp32).
        """
        ...

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        """Get the index type string for the given dtype.

        Defaults to dtype_str, but backends may override for special handling.
        """
        return self.dtype_str(index_dtype)

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        raise exc.BackendUnsupported(self.name, "program IDs")

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
        return f"(({numel}) + ({block_size}) - 1) // ({block_size})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        """Generate a backend-specific type cast expression."""
        return f"tl.cast({expr_str}, {dtype_str})"

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        """Generate a backend-specific range expression, or None to use the default."""
        return None

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        """Generate a backend-specific arange expression for loop offsets."""
        return f"{offsets_var} = {lid} * {block_size_var} + tl.arange(0, {block_size_var}).to({dtype})"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Generate backend-specific grid index expression from an offset."""
        return f"({offset_var} + tl.arange(0, ({block_size_var}))).to({dtype})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Generate backend-specific device-loop index expression from an offset."""
        return f"{offset_var} + tl.arange(0, ({block_size_var})).to({dtype})"

    def scalar_load_expr(self, tensor_name: str) -> str:
        """Load scalar value from a tensor argument."""
        return f"tl.load({tensor_name})"

    def ast_to_dtype_expr(self, expr_str: str, dtype_str: str) -> str:
        """Generate dtype conversion expression for AST values."""
        return self.cast_expr(expr_str, dtype_str)

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        """Optional per-thread mask restricting active threads to tile width."""
        return None

    def max_reduction_threads(self) -> int | None:
        """Maximum threads for a single warp-level reduction, or None if unlimited."""
        return None

    def reduction_axis_first(self) -> bool:
        """Whether reduction strategies should occupy the first (lowest) thread axes."""
        return False

    def force_tile_mask(self) -> bool:
        """Whether tile strategies must emit explicit masks for all tiles."""
        return False

    def supports_config_key(self, key: str) -> bool:
        from ..autotuner.config_spec import BACKEND_SPECIFIC_KEYS

        return key not in BACKEND_SPECIFIC_KEYS

    def supports_block_ptr_indexing(self) -> bool:
        return True

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        return {}

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        """Generate a backend-specific conditional select expression."""
        return f"tl.where({mask}, {true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        """Generate a backend-specific minimum expression."""
        return f"tl.minimum({a}, {b})"

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        """Generate a backend-specific arange expression for reduction index setup."""
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def zeros_expr(self, shape: str, dtype: str) -> str:
        """Generate a backend-specific zeros expression."""
        return f"tl.zeros({shape}, {dtype})"

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        raise exc.BackendUnsupported(self.name, "full tensor creation")

    def reshape_expr(self, expr: str, shape: str) -> str:
        return f"tl.reshape({expr}, {shape})"

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return f"tl.broadcast_to({expr}, {shape})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        """Generate the index expression for a reduction dimension.

        For Triton this is tl.arange; for CuTe it maps to a thread index.
        """
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        """Generate the zero-length index expression for an empty reduction."""
        return f"tl.zeros([0], {dtype})"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        """Generate a host-side next-power-of-2 expression."""
        return f"triton.next_power_of_2({expr})"

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        """Generate the combine expression for looped reductions."""
        from torch._inductor.ir import get_reduction_combine_fn

        combine_fn = get_reduction_combine_fn(reduction_type, dtype)
        return str(combine_fn(acc, val))

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        fake_input: torch.Tensor | None = None,
        fake_output: torch.Tensor | None = None,
    ) -> str:
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        """Whether this reduction type tracks an auxiliary index state."""
        return False

    def reduction_index_init_expr(
        self, shape_dims: list[str], index_dtype: torch.dtype
    ) -> str:
        """Initial accumulator value for index-carrying reductions."""
        return self.full_expr(
            shape_dims, repr(torch.iinfo(index_dtype).max), index_dtype
        )

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
    ) -> str:
        raise exc.BackendUnsupported(self.name, "argmin/argmax reductions")

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        raise exc.BackendUnsupported(self.name, "argmin/argmax reductions")

    def inductor_op_overrides(self) -> InductorOpOverrides:
        raise exc.BackendUnsupported(self.name, "Inductor OpOverrides")

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(
            self.cast_expr("{x}", self.dtype_str(target_dtype)),
            x=x,
        )

    @property
    @abc.abstractmethod
    def function_decorator(self) -> str:
        """Expression string for the kernel function decorator.

        For example, Triton returns 'triton.jit'.
        """
        ...

    @property
    @abc.abstractmethod
    def constexpr_type(self) -> str:
        """Type annotation string for compile-time constant arguments.

        For example, Triton returns 'tl.constexpr'.
        """
        ...

    def inline_constexpr(self, name: str, value: str) -> str:
        """Return the source for a module-level inlined constexpr assignment.

        For example, Triton returns '_BLOCK_SIZE_0 = tl.constexpr(256)'.
        """
        return f"{name} = {self.constexpr_type}({value})"

    @property
    @abc.abstractmethod
    def default_launcher_name(self) -> str:
        """Name of the default host-side launcher symbol for this backend."""
        ...

    @property
    @abc.abstractmethod
    def library_imports(self) -> dict[str, str]:
        """Mapping of short names to import statements for generated code.

        Keys are the short names used in generated code (e.g., 'tl'),
        values are the corresponding import statements.
        """
        ...

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        return []

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        """Transform a host argument expression before passing to the launcher.

        Backends can override this to wrap certain argument types.
        Called during codegen for each argument in sorted order.
        """
        return host_str

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        """Generate preamble statements for scalar arguments in the device function.

        Backends can override to dereference scalar refs, etc.
        """
        return []

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
    ) -> list[str]:
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from .compile_environment import CompileEnvironment
        from .tile_strategy import FlattenedTileStrategy
        from .tile_strategy import NDTileStrategy

        env = CompileEnvironment.current()
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        loop_order = env.config_spec.loop_orders.config_get(
            config.loop_orders, block_ids[0]
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )

        if block_size_infos[0].is_flattened(config):
            block_size = functools.reduce(
                operator.mul, [bs.from_config_assert(config) for bs in block_size_infos]
            )
            return FlattenedTileStrategy(
                fn,
                block_ids,
                block_size=block_size,
                loop_order=loop_order,
            )

        return NDTileStrategy(
            fn,
            block_ids,
            block_size=[bs.from_config_assert(config) for bs in block_size_infos],
            loop_order=loop_order,
            l2_grouping=l2_grouping,
        )

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        raise exc.BackendUnsupported(self.name, "autotune")


class TritonBackend(Backend):
    """Triton code generation backend."""

    @property
    def name(self) -> str:
        return "triton"

    def supports_config_key(self, key: str) -> bool:
        if key in {"waves_per_eu", "matrix_instr_nonkdim"}:
            from .._compat import supports_amd_cdna_tunables

            return supports_amd_cdna_tunables()
        return super().supports_config_key(key)

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from .._compat import supports_amd_cdna_tunables
        from ..autotuner.config_fragment import EnumFragment

        if not supports_amd_cdna_tunables():
            return {}
        return {
            "waves_per_eu": EnumFragment(choices=(1, 2, 3, 4)),
            "matrix_instr_nonkdim": EnumFragment(choices=(0, 16, 32)),
        }

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.utils import triton_type

        return triton_type(dtype)

    def acc_type(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.triton import triton_acc_type

        return triton_acc_type(dtype)

    @property
    def function_decorator(self) -> str:
        return "triton.jit"

    @property
    def constexpr_type(self) -> str:
        return "tl.constexpr"

    @property
    def default_launcher_name(self) -> str:
        return "_default_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "triton": "import triton",
            "tl": "import triton.language as tl",
            "triton_helpers": "from torch._inductor.runtime import triton_helpers",
            "tl_math": "from torch._inductor.runtime.triton_helpers import math as tl_math",
            "libdevice": "from torch._inductor.runtime.triton_compat import libdevice",
            "_default_launcher": "from helion.runtime import default_launcher as _default_launcher",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        if index_dtype != "tl.int32":
            return f"tl.program_id({dim}).to({index_dtype})"
        return f"tl.program_id({dim})"

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
        if is_device:
            return f"tl.cdiv({numel}, {block_size})"
        return f"triton.cdiv({numel}, {block_size})"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.triton import TritonOverrides

        return TritonOverrides()

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return f"{offset_var} + tl.zeros([1], {dtype})"
        return f"({offset_var} + tl.arange(0, ({block_size_var}))).to({dtype})"

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
    ) -> str:
        if reduction_type in {"sum", "max", "min"}:
            return f"tl.{reduction_type}({input_name}, {dim})"
        if reduction_type == "prod":
            return f"triton_helpers.prod({input_name}, {dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        return reduction_type in {"argmin", "argmax"}

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
    ) -> str:
        helper = "max" if reduction_type == "argmax" else "min"
        return (
            f"triton_helpers.{helper}_with_index("
            f"{input_name}, {index_value}, {dim})[1].to({self.dtype_str(output_dtype)})"
        )

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        helper = "maximum" if reduction_type == "argmax" else "minimum"
        return [
            (
                f"{acc}, {acc_index} = "
                f"triton_helpers.{helper}_with_index({acc}, {acc_index}, {value}, {index})"
            )
        ]

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        return (
            f"tl.full([{', '.join(shape_dims)}], {value_expr}, {self.dtype_str(dtype)})"
        )

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from .._compat import supports_maxnreg

        # Workaround for triton bug: warp_specialize requires at least 4 warps
        # See: https://github.com/triton-lang/triton/issues/7354
        num_warps = config.num_warps
        if any(config.range_warp_specializes):
            num_warps = max(4, num_warps)

        args = [
            f"num_warps={num_warps}",
            f"num_stages={config.num_stages}",
            *(["launch_cooperative_grid=True"] if has_barrier else []),
        ] + [
            f"{x.removeprefix('_triton_config_')}={config[x]}"
            for x in config
            if x.startswith("_triton_config_")
        ]

        for key in ("waves_per_eu", "matrix_instr_nonkdim", "num_ctas", "occupancy"):
            if key in config:
                args.append(f"{key}={config[key]}")

        if "maxnreg" in config and config["maxnreg"] is not None and supports_maxnreg():
            args.append(f"maxnreg={config['maxnreg']}")

        return args

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
    ) -> list[str]:
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        force = force or bound_kernel.settings.force_autotune
        if not force and bound_kernel.kernel.configs:
            if len(bound_kernel.kernel.configs) == 1:
                (config,) = bound_kernel.kernel.configs
            else:
                # We have finite predetermined configs, no need to precompile
                bound_kernel.settings.autotune_precompile = None

                from ..autotuner import FiniteSearch

                config = FiniteSearch(
                    bound_kernel, args, bound_kernel.configs
                ).autotune()
        else:
            bound_kernel.settings.check_autotuning_disabled()
            config = bound_kernel.settings.autotuner_fn(
                bound_kernel, args, **kwargs
            ).autotune(skip_cache=force)
        return config


class TileIRBackend(TritonBackend):
    """TileIR code generation backend (extends Triton)."""

    @property
    def name(self) -> str:
        return "tileir"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def supports_config_key(self, key: str) -> bool:
        # Override TritonBackend/Backend rejections for tileir-specific tunables
        if key in {"num_ctas", "occupancy"}:
            return True
        return super().supports_config_key(key)

    def supports_block_ptr_indexing(self) -> bool:
        return False

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from ..autotuner.config_fragment import PowerOfTwoFragment

        return {
            **super().tunable_fragments(),
            "num_ctas": PowerOfTwoFragment(1, 2, 1),
            "occupancy": PowerOfTwoFragment(1, 8, 1),
        }


class NKIOpOverrides:
    """NKI op overrides for Inductor codegen (elementwise ops).
    When _codegen_state is set, emits nisa.tensor_tensor(dst, data1, data2, op);
    otherwise falls back to (a + b) etc. and relies on nl/language support.
    """

    @staticmethod
    def _squeeze_shape_2d(shape: list[int]) -> list[int]:
        """Squeeze a shape to 2D for NKI SBUF. Drops leading 1-dims, then
        flattens remaining leading dims if still > 2D."""
        while len(shape) > 2 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) > 2:
            flat = 1
            for d in shape[:-1]:
                flat *= d
            shape = [flat, shape[-1]]
        return shape

    @staticmethod
    def _resolve_psum_alias(state: object, operand: object) -> object:
        """Resolve a possibly SBUF-named operand through the PSUM alias map.

        The FX-graph fusion pass (nki_fusion.annotate_psum_reuse) may have
        tagged an upstream matmul so its final PSUM→SBUF copy is skipped.
        _nki_dot registers an entry in ``device_function._nki_psum_aliases``
        mapping the matmul's virtual SBUF result name to the real PSUM
        buffer. This helper swaps the SBUF name for the PSUM name so the
        consumer reads directly from PSUM (legal for Vector/Scalar ops).

        Accepts AST nodes, strings, and arbitrary objects; only rewrites
        when the name matches an alias. All other operands pass through.
        """
        aliases = getattr(state.device_function, "_nki_psum_aliases", None)
        if not aliases:
            return operand
        import ast as _ast
        if isinstance(operand, _ast.AST):
            try:
                name = _ast.unparse(operand)
            except Exception:
                return operand
        elif isinstance(operand, str):
            name = operand
        else:
            return operand
        psum_name = aliases.get(name)
        if psum_name is None:
            return operand
        # Return a plain string — callers always interpolate operands
        # into f-strings via str()/ast.unparse(), so this is transparent.
        return psum_name

    @staticmethod
    def _nki_tensor_tensor(a: object, b: object, op: str, prefix: str) -> str:
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

        # PSUM-reuse fusion: if either operand is a matmul result whose final
        # PSUM→SBUF copy was elided, rewrite the operand to read from PSUM.
        # Safe because nisa.tensor_tensor runs on the Vector/GpSimd Engine,
        # which can read both SBUF and PSUM.
        a = NKIOpOverrides._resolve_psum_alias(state, a)
        b = NKIOpOverrides._resolve_psum_alias(state, b)

        # Layout reconcile: when one operand is [1, N] (FX-declared shape
        # that wasn't transposed) and the other is [N, 1] (reduction
        # output or transposed accumulator) AND the FX val for the output
        # has a single non-trivial dim (logically a vector), transpose
        # the [N, 1] operand to [1, N] so tensor_tensor sees matching
        # shapes. This catches the downstream-use-of-transposed-accumulator
        # case without over-broadcasting.
        def _layout_reconcile_transpose(name: object, target_shape: list[int],
                                         dtype: str) -> object:
            """If ``name`` SBUF is [N, 1] and target_shape is [1, N], emit
            nc_transpose and return the new var name (as a string)."""
            if not isinstance(name, str):
                name_str = str(name)
            else:
                name_str = name
            cur_shape = state.device_function._nki_sbuf_shapes.get(name_str)
            if (
                cur_shape is not None
                and len(cur_shape) == 2
                and len(target_shape) == 2
                and cur_shape[0] == target_shape[1]
                and cur_shape[1] == target_shape[0]
                and cur_shape[0] > 1
                and target_shape[0] > 0
                and target_shape[1] > 0
            ):
                from .ast_extension import statement_from_string as _sfs
                transpose_valid = {
                    "nl.float8_e4m3",
                    "nl.float8_e5m2",
                    "nl.bfloat16",
                    "nl.float16",
                    "nl.tfloat32",
                    "nl.float32",
                }
                src_dtype = state.device_function._nki_sbuf_dtypes.get(
                    name_str, dtype
                )
                transpose_src = name_str
                transpose_dtype = dtype
                if src_dtype not in transpose_valid:
                    cast_in = state.device_function.new_var("_tr_cast", dce=True)
                    state.device_function._nki_sbuf_shapes[cast_in] = list(cur_shape)
                    state.device_function._nki_sbuf_dtypes[cast_in] = "nl.float32"
                    state.add_statement(
                        _sfs(
                            f"{cast_in} = nl.ndarray([{cur_shape[0]}, {cur_shape[1]}], "
                            "nl.float32, buffer=nl.sbuf)"
                        )
                    )
                    state.add_statement(
                        _sfs(f"nisa.tensor_copy(dst={cast_in}, src={name_str})")
                    )
                    transpose_src = cast_in
                    transpose_dtype = "nl.float32"
                tr_psum = state.device_function.new_var("_tr_psum", dce=True)
                tr_sbuf = state.device_function.new_var("_tr_sbuf", dce=True)
                state.device_function._nki_sbuf_shapes[tr_sbuf] = list(target_shape)
                state.device_function._nki_sbuf_dtypes[tr_sbuf] = dtype
                state.add_statement(
                    _sfs(
                        f"{tr_psum} = nl.ndarray([{target_shape[0]}, {target_shape[1]}], "
                        f"{transpose_dtype}, buffer=nl.psum)"
                    )
                )
                state.add_statement(
                    _sfs(f"nisa.nc_transpose(dst={tr_psum}, data={transpose_src})")
                )
                state.add_statement(
                    _sfs(
                        f"{tr_sbuf} = nl.ndarray([{target_shape[0]}, {target_shape[1]}], "
                        f"{dtype}, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    _sfs(f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})")
                )
                return tr_sbuf
            return name

        # Layout reconcile: when one operand is [1, N] and the other is
        # [N, 1], they are both semantically 1D vectors. Reconcile by
        # transposing one to match the other. Prefer [1, N] direction
        # since downstream stores typically expect row-major.
        def _lookup_shape(name: object) -> list[int] | None:
            s = state.device_function._nki_sbuf_shapes.get(str(name))
            if s is not None:
                return s
            # Strip _copy suffixes
            _lk = str(name)
            while "_copy" in _lk:
                _lk = _lk[:_lk.rfind("_copy")]
                s = state.device_function._nki_sbuf_shapes.get(_lk)
                if s is not None:
                    return s
            return None

        a_shape_r = _lookup_shape(a)
        b_shape_r = _lookup_shape(b)
        if (
            a_shape_r is not None and b_shape_r is not None
            and len(a_shape_r) == 2 and len(b_shape_r) == 2
            and a_shape_r[0] != b_shape_r[0]
            and a_shape_r[0] == b_shape_r[1]
            and a_shape_r[1] == b_shape_r[0]
            and max(a_shape_r[0], a_shape_r[1]) > 1
            and max(b_shape_r[0], b_shape_r[1]) > 1
            and 1 in {a_shape_r[0], a_shape_r[1]}
            and 1 in {b_shape_r[0], b_shape_r[1]}
        ):
            # Both are 1D vectors in different layouts. Transpose the
            # [N, 1] one to [1, N] to match its partner.
            try:
                from torch._inductor.virtualized import V as _V_tr
                _cn = _V_tr.current_node
                _cn_val = _cn.meta.get("val") if _cn is not None else None
            except Exception:
                _cn_val = None
            if isinstance(_cn_val, torch.Tensor):
                _rdt = env.backend.dtype_str(_cn_val.dtype)
            else:
                _rdt = "nl.float32"
            # Pick target: prefer the [1, N] layout (row-major)
            if a_shape_r[0] == 1:
                target = a_shape_r
                b = _layout_reconcile_transpose(b, target, _rdt)
            else:
                target = b_shape_r
                a = _layout_reconcile_transpose(a, target, _rdt)

        def _emit_partition_broadcast(hbm_src: str, f_count: int, p_count: int,
                                      target_dtype_str: str,
                                      src_var_name: str = "") -> str | None:
            """Broadcast HBM [1, F] data to [P, F] SBUF buffer.
            Returns the SBUF variable name or None. Used by both multi-user and partition-mismatch paths.
            """
            from .ast_extension import create as _create, expr_from_string as _efrom
            if not hasattr(state.device_function, "_nki_hbm_sources"):
                state.device_function._nki_hbm_sources = {}
            sbuf_dtypes = getattr(state.device_function, "_nki_sbuf_dtypes", {})
            hbm_dtype_str = sbuf_dtypes.get(src_var_name, "nl.float16")
            bcast_native = state.device_function.new_var(
                "_nki_bcast_fp16" if hbm_dtype_str == "nl.float16" else "_nki_bcast_fp32",
                dce=True,
            )
            state.device_function._nki_sbuf_shapes[bcast_native] = [p_count, f_count]
            state.device_function._nki_hbm_sources[bcast_native] = hbm_src
            if not hasattr(state.device_function, "_nki_sbuf_dtypes"):
                state.device_function._nki_sbuf_dtypes = {}
            state.device_function._nki_sbuf_dtypes[bcast_native] = hbm_dtype_str
            state.add_statement(statement_from_string(
                f"{bcast_native} = nl.ndarray([{p_count}, {f_count}], {hbm_dtype_str}, buffer=nl.sbuf)"
            ))
            p_loop_var = state.device_function.new_var("_p_bcast")
            state.add_statement(_create(
                ast.For,
                target=_create(ast.Name, id=p_loop_var, ctx=ast.Store()),
                iter=_efrom(f"nl.affine_range({p_count})"),
                body=[statement_from_string(
                    f"nisa.dma_copy(dst={bcast_native}[{p_loop_var}:{p_loop_var}+1, 0:{f_count}], "
                    f"src={hbm_src})"
                )],
                orelse=[],
            ))
            if hbm_dtype_str == target_dtype_str:
                return bcast_native
            bcast_final = state.device_function.new_var("_nki_bcast", dce=True)
            state.device_function._nki_sbuf_shapes[bcast_final] = [p_count, f_count]
            state.device_function._nki_hbm_sources[bcast_final] = hbm_src
            state.device_function._nki_sbuf_dtypes[bcast_final] = target_dtype_str
            state.add_statement(statement_from_string(
                f"{bcast_final} = nl.ndarray([{p_count}, {f_count}], {target_dtype_str}, buffer=nl.sbuf)"
            ))
            state.add_statement(statement_from_string(f"nisa.memset({bcast_final}, value=0)"))
            state.add_statement(statement_from_string(
                f"nisa.tensor_tensor(dst={bcast_final}, data1={bcast_final}, "
                f"data2={bcast_native}, op=nl.add)"
            ))
            return bcast_final

        # Use ast.unparse for AST nodes to get the variable name string,
        # since str(ast.Name(id='x')) returns an object repr, not 'x'.
        dst = ast.unparse(a) if isinstance(a, ast.AST) else str(a)
        b_str = ast.unparse(b) if isinstance(b, ast.AST) else str(b)

        def _lookup_cross_shape(name: str) -> list[int] | None:
            shape = state.device_function._nki_sbuf_shapes.get(name)
            if shape is not None:
                return shape
            lookup = name
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                shape = state.device_function._nki_sbuf_shapes.get(lookup)
                if shape is not None:
                    return shape
            return None

        def _lookup_cross_dtype(name: str, default: str) -> str:
            dtypes = state.device_function._nki_sbuf_dtypes
            if name in dtypes:
                return dtypes[name]
            lookup = name
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                if lookup in dtypes:
                    return dtypes[lookup]
            return default

        try:
            from torch._inductor.virtualized import V as _V_cross

            _cross_node = _V_cross.current_node or state.fx_node
            _cross_val = _cross_node.meta.get("val") if _cross_node is not None else None
        except Exception:
            _cross_val = None
        a_cross_shape = _lookup_cross_shape(dst)
        b_cross_shape = _lookup_cross_shape(b_str)
        if (
            isinstance(_cross_val, torch.Tensor)
            and a_cross_shape is not None
            and b_cross_shape is not None
            and len(a_cross_shape) == 2
            and len(b_cross_shape) == 2
            and a_cross_shape[0] == 1
            and b_cross_shape[0] == 1
            and _cross_val.ndim >= 2
        ):
            import sympy as _sp_cross

            _cross_subs: dict[_sp_cross.Symbol, int] = {}
            if state.config is not None:
                for _bs_cross in env.block_sizes:
                    _cfg_cross = _bs_cross.from_config(state.config)
                    if isinstance(_cfg_cross, int):
                        _cross_subs[_bs_cross.symbol()] = _cfg_cross

            def _resolve_cross_dim(dim: object) -> int:
                if isinstance(dim, int):
                    return dim
                if isinstance(dim, torch.SymInt):
                    try:
                        return int(dim._sympy_().subs(_cross_subs))
                    except (TypeError, ValueError):
                        return int(env.size_hint(dim))
                return int(dim)

            cross_shape = NKIOpOverrides._squeeze_shape_2d(
                [_resolve_cross_dim(d) for d in _cross_val.shape]
            )
            if (
                len(cross_shape) >= 2
                and cross_shape[0] > 1
                and cross_shape[1] > 1
                and {a_cross_shape[1], b_cross_shape[1]}
                == {cross_shape[0], cross_shape[1]}
            ):
                from .ast_extension import create as _create
                from .ast_extension import expr_from_string as _efrom

                p_target, f_target = cross_shape[0], cross_shape[1]
                out_dtype = (
                    "nl.int32"
                    if op.startswith("nl.bitwise")
                    else env.backend.dtype_str(_cross_val.dtype)
                )

                def _row_to_col(src: str, p_count: int, dtype: str) -> str:
                    int_dtypes = {
                        "nl.int32",
                        "nl.int16",
                        "nl.int8",
                        "nl.uint32",
                        "nl.uint16",
                        "nl.uint8",
                        "nl.bool_",
                    }
                    transpose_src = src
                    transpose_dtype = dtype
                    if dtype in int_dtypes:
                        cast_in = state.device_function.new_var(
                            "_nki_cross_cast", dce=True
                        )
                        state.device_function._nki_sbuf_shapes[cast_in] = [1, p_count]
                        state.device_function._nki_sbuf_dtypes[cast_in] = "nl.float32"
                        state.add_statement(
                            statement_from_string(
                                f"{cast_in} = nl.ndarray([1, {p_count}], "
                                "nl.float32, buffer=nl.sbuf)"
                            )
                        )
                        state.add_statement(
                            statement_from_string(
                                f"nisa.tensor_copy(dst={cast_in}, src={src})"
                            )
                        )
                        transpose_src = cast_in
                        transpose_dtype = "nl.float32"
                    tr_psum = state.device_function.new_var(
                        "_nki_cross_tr_psum", dce=True
                    )
                    tr_sbuf = state.device_function.new_var(
                        "_nki_cross_tr_sbuf", dce=True
                    )
                    state.device_function._nki_sbuf_shapes[tr_sbuf] = [p_count, 1]
                    state.device_function._nki_sbuf_dtypes[tr_sbuf] = dtype
                    state.add_statement(
                        statement_from_string(
                            f"{tr_psum} = nl.ndarray([{p_count}, 1], "
                            f"{transpose_dtype}, buffer=nl.psum)"
                        )
                    )
                    state.add_statement(
                        statement_from_string(
                            f"nisa.nc_transpose(dst={tr_psum}, data={transpose_src})"
                        )
                    )
                    state.add_statement(
                        statement_from_string(
                            f"{tr_sbuf} = nl.ndarray([{p_count}, 1], "
                            f"{dtype}, buffer=nl.sbuf)"
                        )
                    )
                    state.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
                        )
                    )
                    return tr_sbuf

                def _replicate_col(src: str, dtype: str) -> str:
                    out = state.device_function.new_var("_nki_cross_bcast", dce=True)
                    state.device_function._nki_sbuf_shapes[out] = [
                        p_target,
                        f_target,
                    ]
                    state.device_function._nki_sbuf_dtypes[out] = dtype
                    state.add_statement(
                        statement_from_string(
                            f"{out} = nl.ndarray([{p_target}, {f_target}], "
                            f"{dtype}, buffer=nl.sbuf)"
                        )
                    )
                    f_loop = state.device_function.new_var("_f_bcast")
                    state.add_statement(
                        _create(
                            ast.For,
                            target=_create(ast.Name, id=f_loop, ctx=ast.Store()),
                            iter=_efrom(f"nl.affine_range({f_target})"),
                            body=[
                                statement_from_string(
                                    f"nisa.tensor_copy(dst={out}[0:{p_target}, {f_loop}:{f_loop}+1], "
                                    f"src={src}[0:{p_target}, 0:1])"
                                )
                            ],
                            orelse=[],
                        )
                    )
                    return out

                def _replicate_row(src: str, dtype: str) -> str:
                    out = state.device_function.new_var("_nki_cross_bcast", dce=True)
                    state.device_function._nki_sbuf_shapes[out] = [
                        p_target,
                        f_target,
                    ]
                    state.device_function._nki_sbuf_dtypes[out] = dtype
                    state.add_statement(
                        statement_from_string(
                            f"{out} = nl.broadcast_to({src}, "
                            f"shape=({p_target}, {f_target}))"
                        )
                    )
                    return out

                def _cast_cross_bitwise(
                    name: str, shape: list[int], dtype: str
                ) -> tuple[str, str]:
                    if not op.startswith("nl.bitwise") or dtype == out_dtype:
                        return name, dtype
                    casted = state.device_function.new_var(
                        "_nki_cross_bitcast", dce=True
                    )
                    state.device_function._nki_sbuf_shapes[casted] = list(shape)
                    state.device_function._nki_sbuf_dtypes[casted] = out_dtype
                    shape_str = ", ".join(str(d) for d in shape)
                    state.add_statement(
                        statement_from_string(
                            f"{casted} = nl.ndarray([{shape_str}], "
                            f"{out_dtype}, buffer=nl.sbuf)"
                        )
                    )
                    state.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={casted}, src={name})"
                        )
                    )
                    return casted, out_dtype

                def _expand_cross(name: str, shape: list[int]) -> str:
                    dtype = _lookup_cross_dtype(name, out_dtype)
                    name, dtype = _cast_cross_bitwise(name, shape, dtype)
                    if shape[1] == p_target:
                        return _replicate_col(_row_to_col(name, p_target, dtype), out_dtype)
                    return _replicate_row(name, dtype)

                a_expanded = _expand_cross(dst, a_cross_shape)
                b_expanded = _expand_cross(b_str, b_cross_shape)
                out = state.device_function.new_var(prefix, dce=True)
                state.device_function._nki_sbuf_shapes[out] = [p_target, f_target]
                state.device_function._nki_sbuf_dtypes[out] = out_dtype
                state.add_statement(
                    statement_from_string(
                        f"{out} = nl.ndarray([{p_target}, {f_target}], "
                        f"{out_dtype}, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={out}, data1={a_expanded}, "
                        f"data2={b_expanded}, op={op})"
                    )
                )
                return out

        dst_tile_vars = state.device_function.get_tile_list_vars(dst)
        b_tile_vars = state.device_function.get_tile_list_vars(b_str)

        # If the first operand's buffer must be preserved, writing the result
        # back into 'a' in-place would corrupt future reads of 'a'.
        # Detect two cases and allocate a fresh output buffer instead:
        #
        # 1. The first operand FX node has multiple users in the current graph
        #    (the value is needed again later in this iteration).
        #
        # 2. The destination variable name is a second-level Helion copy var
        #    (e.g. "_nki_cast_copy_0"): these are outer-loop read-only captures
        #    that alias the outer-scope SBUF buffer. In-place writes would corrupt
        #    the buffer for subsequent loop iterations. (First-level "_copy" vars
        #    like "_nki_full_copy" are accumulators and should remain in-place.)
        from torch._inductor.virtualized import V

        cur_node = V.current_node or state.fx_node
        _is_second_level_copy = "_copy_" in dst and dst[-1:].isdigit()
        if _is_second_level_copy and cur_node is not None:
            _users = getattr(cur_node, "users", {})
            _is_loop_output = (
                len(_users) == 1
                and any(getattr(user, "op", None) == "output" for user in _users)
            )
            if _is_loop_output and b_tile_vars is None and dst_tile_vars is None:
                _base_dst = dst
                _sbuf_shapes = state.device_function._nki_sbuf_shapes
                while _base_dst not in _sbuf_shapes and "_copy" in _base_dst:
                    _base_dst = _base_dst[:_base_dst.rfind("_copy")]
                if _base_dst in _sbuf_shapes and _base_dst != dst:
                    state.add_statement(
                        statement_from_string(
                            f"nisa.tensor_tensor(dst={_base_dst}, data1={_base_dst}, data2={b}, op={op})"
                        )
                    )
                    return _base_dst
        # (debug removed)
        a_node_has_other_users = _is_second_level_copy or (
            cur_node is not None
            and len(cur_node.args) >= 1
            and hasattr(cur_node.args[0], "users")
            and len(cur_node.args[0].users) > 1
        )
        if a_node_has_other_users:
            shape_list = getattr(state.device_function, "_nki_sbuf_shapes", {}).get(dst)
            out_val = cur_node.meta.get("val") if cur_node else None
            # Fallback: derive shape from the output fake tensor with config substitution
            # ONLY for second-level copy vars (outer-scope loop captures like _nki_cast_copy_0).
            # Do NOT use this fallback for the general users>1 path: when V.current_node is the
            # surrounding reduction node its output shape is the reduced shape (e.g. [1]), not the
            # accumulator shape ([1, 4096]), and allocating a buffer with the wrong shape breaks NKI.
            # When V.current_node is None (NKI codegen path), derive shape
            # from the source variable's registered SBUF shape.
            # Copy vars (e.g. _nki_full_copy_0) aren't registered, but
            # we can strip suffixes to find the original var.
            if shape_list is None and _is_second_level_copy:
                # Try direct lookup first
                _lookup_name = dst
                _sbuf_shapes = state.device_function._nki_sbuf_shapes
                # Strip copy suffixes: _nki_full_copy_0 → _nki_full_copy → _nki_full
                while _lookup_name not in _sbuf_shapes and "_copy" in _lookup_name:
                    idx = _lookup_name.rfind("_copy")
                    _lookup_name = _lookup_name[:idx]
                src_shape = _sbuf_shapes.get(_lookup_name)
                if src_shape is not None and len(src_shape) >= 2:
                    shape_list = list(src_shape)
                    if out_val is None:
                        out_val = torch.empty(1)  # just needs to be non-None and a Tensor
            if shape_list is None and _is_second_level_copy and out_val is not None and isinstance(out_val, torch.Tensor):
                from .compile_environment import CompileEnvironment as _CE

                _env2 = _CE.current()
                _state2 = getattr(_env2, "_codegen_state", None)
                if _state2 is not None and _state2.config is not None:
                    import sympy as _sp2

                    _subs2: dict[_sp2.Symbol, int] = {}
                    for _bs2 in _env2.block_sizes:
                        _c2 = _bs2.from_config(_state2.config)
                        if isinstance(_c2, int):
                            _subs2[_bs2.symbol()] = _c2
                    _resolved2 = []
                    for _d2 in out_val.shape:
                        if isinstance(_d2, torch.SymInt):
                            try:
                                _resolved2.append(int(_d2._sympy_().subs(_subs2)))
                            except (TypeError, ValueError):
                                _resolved2.append(int(_env2.size_hint(_d2)))
                        else:
                            _resolved2.append(int(_d2))
                    if len(_resolved2) >= 2:  # only use if 2-D (NKI requirement)
                        shape_list = _resolved2
            # Squeeze 3D+ shapes to 2D for NKI SBUF
            if shape_list is not None:
                shape_list = NKIOpOverrides._squeeze_shape_2d(shape_list)
            if shape_list is not None and out_val is not None and isinstance(out_val, torch.Tensor):
                # Derive dtype: prefer FX node output, fallback to float32
                if hasattr(out_val, 'dtype') and out_val.numel() > 1:
                    dtype_str = env.backend.dtype_str(out_val.dtype)
                elif cur_node is not None and isinstance(cur_node.meta.get("val"), torch.Tensor):
                    dtype_str = env.backend.dtype_str(cur_node.meta["val"].dtype)
                else:
                    dtype_str = "nl.float32"

                # If operands have mismatched partition dims, the output must use
                # the larger partition count (the broadcast result is [P, F]).
                def _lookup_alloc_shape(name: str) -> list[int] | None:
                    shape = state.device_function._nki_sbuf_shapes.get(name)
                    if shape is not None:
                        return shape
                    lookup = name
                    while "_copy" in lookup:
                        lookup = lookup[: lookup.rfind("_copy")]
                        shape = state.device_function._nki_sbuf_shapes.get(lookup)
                        if shape is not None:
                            return shape
                    return None

                a_sbuf_shape_alloc = _lookup_alloc_shape(dst)
                b_sbuf_shape_alloc = _lookup_alloc_shape(b_str)
                if (
                    a_sbuf_shape_alloc is not None
                    and b_sbuf_shape_alloc is not None
                    and len(a_sbuf_shape_alloc) >= 2
                    and len(b_sbuf_shape_alloc) >= 2
                    and a_sbuf_shape_alloc[0] == 1
                    and b_sbuf_shape_alloc[0] == 1
                    and out_val is not None
                    and isinstance(out_val, torch.Tensor)
                    and out_val.ndim >= 2
                ):
                    _fx_out_shape = NKIOpOverrides._squeeze_shape_2d(
                        [
                            env.size_hint(d) if isinstance(d, torch.SymInt) else int(d)
                            for d in out_val.shape
                        ]
                    )
                    if (
                        len(_fx_out_shape) >= 2
                        and _fx_out_shape[0] == a_sbuf_shape_alloc[1]
                        and _fx_out_shape[1] == b_sbuf_shape_alloc[1]
                    ):
                        shape_list = [
                            _fx_out_shape[0],
                            _fx_out_shape[1],
                            *list(shape_list[2:]),
                        ]
                if (
                    b_sbuf_shape_alloc is not None
                    and len(shape_list) >= 2
                    and len(b_sbuf_shape_alloc) >= 2
                    and shape_list[0] != b_sbuf_shape_alloc[0]
                ):
                    # Use the larger partition count for the result shape
                    max_p = max(shape_list[0], b_sbuf_shape_alloc[0])
                    shape_list = [max_p] + list(shape_list[1:])
                if (
                    b_sbuf_shape_alloc is not None
                    and len(shape_list) >= 2
                    and len(b_sbuf_shape_alloc) >= 2
                    and shape_list[1] != b_sbuf_shape_alloc[1]
                ):
                    max_f = max(shape_list[1], b_sbuf_shape_alloc[1])
                    shape_list = [shape_list[0], max_f, *list(shape_list[2:])]
                if (
                    b_sbuf_shape_alloc is not None
                    and len(shape_list) >= 2
                    and len(b_sbuf_shape_alloc) >= 2
                    and {shape_list[0], b_sbuf_shape_alloc[0]}
                    == {1, max(shape_list[0], b_sbuf_shape_alloc[0])}
                    and {shape_list[1], b_sbuf_shape_alloc[1]}
                    == {1, max(shape_list[1], b_sbuf_shape_alloc[1])}
                ):
                    shape_list = [
                        max(shape_list[0], b_sbuf_shape_alloc[0]),
                        max(shape_list[1], b_sbuf_shape_alloc[1]),
                        *list(shape_list[2:]),
                    ]

                shape_str = ", ".join(str(d) for d in shape_list)
                new_dst = state.device_function.new_var(prefix, dce=True)
                # Register the new variable's SBUF shape for downstream cast_ast
                state.device_function._nki_sbuf_shapes[new_dst] = list(shape_list)
                state.add_statement(
                    statement_from_string(
                        f"{new_dst} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
                    )
                )

                if b_tile_vars is not None:
                    # tile-list variant: emit per-tile ops into new_dst
                    for i, bv in enumerate(b_tile_vars):
                        state.add_statement(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={new_dst}, data1={a}, data2={bv}, op={op})"
                            )
                        )
                else:
                    # Check for partition broadcast before emitting tensor_tensor.
                    # Use the copy-stripping lookup so _nki_full_copy_0 etc.
                    # resolve to their original var's shape.
                    def _lookup_inner(name: str) -> list[int] | None:
                        shape = state.device_function._nki_sbuf_shapes.get(name)
                        if shape is not None:
                            return shape
                        _lk = name
                        while "_copy" in _lk:
                            _lk = _lk[:_lk.rfind("_copy")]
                            shape = state.device_function._nki_sbuf_shapes.get(_lk)
                            if shape is not None:
                                return shape
                        return None
                    _new_a_shape = _lookup_inner(str(a))
                    _new_b_shape = _lookup_inner(b_str)

                    # SBUF-replicate broadcast helper (mirrors the outer
                    # path). Used for [1,F]×[P,1] and similar pure-SBUF
                    # mid-kernel allocations that have no HBM source.
                    def _inner_sbuf_replicate(src_var: str, src_shape: list[int],
                                              p_target: int, f_target: int,
                                              dtype: str) -> str:
                        from .ast_extension import (
                            create as _create,
                            expr_from_string as _efrom,
                        )
                        out_var = state.device_function.new_var("_nki_sbuf_bcast", dce=True)
                        state.device_function._nki_sbuf_shapes[out_var] = [p_target, f_target]
                        state.device_function._nki_sbuf_dtypes[out_var] = dtype
                        state.add_statement(
                            statement_from_string(
                                f"{out_var} = nl.ndarray([{p_target}, {f_target}], {dtype}, buffer=nl.sbuf)"
                            )
                        )
                        src_p, src_f = src_shape[0], src_shape[1]
                        if src_p == 1 and src_f == f_target:
                            state.add_statement(
                                statement_from_string(
                                    f"{out_var} = nl.broadcast_to({src_var}, "
                                    f"shape=({p_target}, {f_target}))"
                                )
                            )
                        elif src_p == p_target and src_f == 1:
                            f_loop = state.device_function.new_var("_f_bcast")
                            state.add_statement(_create(
                                ast.For,
                                target=_create(ast.Name, id=f_loop, ctx=ast.Store()),
                                iter=_efrom(f"nl.affine_range({f_target})"),
                                body=[statement_from_string(
                                    f"nisa.tensor_copy(dst={out_var}[0:{p_target}, {f_loop}:{f_loop}+1], "
                                    f"src={src_var}[0:{p_target}, 0:1])"
                                )],
                                orelse=[],
                            ))
                        else:
                            # Scalar or already-matching; do a straight copy
                            state.add_statement(statement_from_string(
                                f"nisa.tensor_copy(dst={out_var}, src={src_var})"
                            ))
                        return out_var

                    def _inner_transpose_row_to_col(
                        src_var: str, src_shape: list[int], dtype: str
                    ) -> str:
                        p_target = src_shape[1]
                        int_dtypes = {
                            "nl.int32",
                            "nl.int16",
                            "nl.int8",
                            "nl.uint32",
                            "nl.uint16",
                            "nl.uint8",
                            "nl.bool_",
                        }
                        transpose_src = src_var
                        transpose_dtype = dtype
                        if dtype in int_dtypes:
                            cast_in = state.device_function.new_var(
                                "_nki_cross_cast", dce=True
                            )
                            state.device_function._nki_sbuf_shapes[cast_in] = [
                                1,
                                p_target,
                            ]
                            state.device_function._nki_sbuf_dtypes[cast_in] = (
                                "nl.float32"
                            )
                            state.add_statement(
                                statement_from_string(
                                    f"{cast_in} = nl.ndarray([1, {p_target}], "
                                    "nl.float32, buffer=nl.sbuf)"
                                )
                            )
                            state.add_statement(
                                statement_from_string(
                                    f"nisa.tensor_copy(dst={cast_in}, src={src_var})"
                                )
                            )
                            transpose_src = cast_in
                            transpose_dtype = "nl.float32"

                        tr_psum = state.device_function.new_var(
                            "_nki_cross_tr_psum", dce=True
                        )
                        tr_sbuf = state.device_function.new_var(
                            "_nki_cross_tr_sbuf", dce=True
                        )
                        state.device_function._nki_sbuf_shapes[tr_sbuf] = [
                            p_target,
                            1,
                        ]
                        state.device_function._nki_sbuf_dtypes[tr_sbuf] = dtype
                        state.add_statement(
                            statement_from_string(
                                f"{tr_psum} = nl.ndarray([{p_target}, 1], "
                                f"{transpose_dtype}, buffer=nl.psum)"
                            )
                        )
                        state.add_statement(
                            statement_from_string(
                                f"nisa.nc_transpose(dst={tr_psum}, data={transpose_src})"
                            )
                        )
                        state.add_statement(
                            statement_from_string(
                                f"{tr_sbuf} = nl.ndarray([{p_target}, 1], "
                                f"{dtype}, buffer=nl.sbuf)"
                            )
                        )
                        state.add_statement(
                            statement_from_string(
                                f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
                            )
                        )
                        return tr_sbuf

                    # Full numpy broadcast: [1,F]×[P,1] or similar. Only
                    # apply when the FX-derived new_dst allocation shape
                    # ALSO matches the broadcast target (i.e. downstream
                    # codegen agrees the result is [P, F], not [1, F] or
                    # [P, 1]). Without this guard, we over-broadcast and
                    # break downstream tensor_scalar calls that assume
                    # operand0 is scalar-like AND accumulator stores that
                    # expect [1, F] or [P, 1] shape.
                    # Additional guard: the shape_list (FX-derived output
                    # shape) must match the broadcast target AND the dst
                    # must not be reused as an accumulator (i.e. the op is
                    # assigning a new var, not updating an existing one).
                    _ptgt_check = max(_new_a_shape[0] or 0, _new_b_shape[0] or 0) if _new_a_shape and _new_b_shape else 0
                    _ftgt_check = max(_new_a_shape[1] or 0, _new_b_shape[1] or 0) if _new_a_shape and _new_b_shape else 0
                    _full_inner = (
                        _new_a_shape is not None and _new_b_shape is not None
                        and len(_new_a_shape) >= 2 and len(_new_b_shape) >= 2
                        and {_new_a_shape[0], _new_b_shape[0]} == {1, _ptgt_check}
                        and {_new_a_shape[1], _new_b_shape[1]} == {1, _ftgt_check}
                        and _ptgt_check > 1
                        and _ftgt_check > 1
                        and len(shape_list) >= 2
                        and shape_list[0] == _ptgt_check
                        and shape_list[1] == _ftgt_check
                    )
                    if _full_inner:
                        _ptgt = max(_new_a_shape[0], _new_b_shape[0])
                        _ftgt = max(_new_a_shape[1], _new_b_shape[1])
                        _dt = dtype_str
                        _abc = (
                            str(a) if _new_a_shape == [_ptgt, _ftgt]
                            else _inner_sbuf_replicate(str(a), _new_a_shape, _ptgt, _ftgt, _dt)
                        )
                        _bbc = (
                            b_str if _new_b_shape == [_ptgt, _ftgt]
                            else _inner_sbuf_replicate(b_str, _new_b_shape, _ptgt, _ftgt, _dt)
                        )
                        state.add_statement(statement_from_string(
                            f"nisa.tensor_tensor(dst={new_dst}, data1={_abc}, data2={_bbc}, op={op})"
                        ))
                        return new_dst

                    _cross_row_inner = (
                        _new_a_shape is not None and _new_b_shape is not None
                        and len(_new_a_shape) >= 2 and len(_new_b_shape) >= 2
                        and _new_a_shape[0] == 1
                        and _new_b_shape[0] == 1
                        and len(shape_list) >= 2
                        and shape_list[0] == _new_a_shape[1]
                        and shape_list[1] == _new_b_shape[1]
                        and shape_list[0] > 1
                        and shape_list[1] > 1
                    )
                    if _cross_row_inner:
                        _ptgt = shape_list[0]
                        _ftgt = shape_list[1]
                        _a_dtype = state.device_function._nki_sbuf_dtypes.get(
                            str(a), dtype_str
                        )
                        _b_dtype = state.device_function._nki_sbuf_dtypes.get(
                            b_str, dtype_str
                        )
                        _a_col = _inner_transpose_row_to_col(
                            str(a), _new_a_shape, _a_dtype
                        )
                        _abc = _inner_sbuf_replicate(
                            _a_col, [_ptgt, 1], _ptgt, _ftgt, dtype_str
                        )
                        _bbc = _inner_sbuf_replicate(
                            b_str, _new_b_shape, _ptgt, _ftgt, _b_dtype
                        )
                        state.add_statement(statement_from_string(
                            f"nisa.tensor_tensor(dst={new_dst}, data1={_abc}, "
                            f"data2={_bbc}, op={op})"
                        ))
                        return new_dst

                    _new_mismatch = (
                        _new_a_shape and _new_b_shape
                        and len(_new_a_shape) >= 2 and len(_new_b_shape) >= 2
                        and _new_a_shape[0] != _new_b_shape[0]
                    )
                    if _new_mismatch:
                        # Use broadcast path: determine which operand has partition=1
                        _hbm_srcs = getattr(state.device_function, "_nki_hbm_sources", {})
                        # Get target dtype from FX node output
                        _inline_dtype_str = "nl.float32"
                        if cur_node is not None:
                            _iv = cur_node.meta.get("val")
                            if isinstance(_iv, torch.Tensor):
                                _inline_dtype_str = env.backend.dtype_str(_iv.dtype)
                        if _new_a_shape[0] == 1 and str(a) in _hbm_srcs:
                            # Broadcast a to [P, F]
                            _p = _new_b_shape[0]
                            _f = _new_a_shape[1]
                            _bcast_v = _emit_partition_broadcast(
                                _hbm_srcs[str(a)], _f, _p, _inline_dtype_str, str(a))
                            if _bcast_v:
                                state.add_statement(statement_from_string(
                                    f"nisa.tensor_tensor(dst={new_dst}, data1={_bcast_v}, "
                                    f"data2={b}, op={op})"
                                ))
                            else:
                                state.add_statement(statement_from_string(
                                    f"nisa.tensor_tensor(dst={new_dst}, data1={a}, data2={b}, op={op})"
                                ))
                        elif _new_b_shape[0] == 1 and b_str in _hbm_srcs:
                            # Broadcast b to [P, F]
                            _p = _new_a_shape[0]
                            _f = _new_b_shape[1]
                            _bcast_v = _emit_partition_broadcast(
                                _hbm_srcs[b_str], _f, _p, _inline_dtype_str, b_str)
                            if _bcast_v:
                                state.add_statement(statement_from_string(
                                    f"nisa.tensor_tensor(dst={new_dst}, data1={a}, "
                                    f"data2={_bcast_v}, op={op})"
                                ))
                            else:
                                state.add_statement(statement_from_string(
                                    f"nisa.tensor_tensor(dst={new_dst}, data1={a}, data2={b}, op={op})"
                                ))
                        else:
                            _part_bcast_a = str(a)
                            _part_bcast_b = b_str
                            if _new_a_shape[0] == 1 and _new_b_shape[0] > 1:
                                _part_bcast_a = _inner_sbuf_replicate(
                                    str(a),
                                    _new_a_shape,
                                    _new_b_shape[0],
                                    max(_new_a_shape[1], _new_b_shape[1]),
                                    dtype_str,
                                )
                            elif _new_b_shape[0] == 1 and _new_a_shape[0] > 1:
                                _part_bcast_b = _inner_sbuf_replicate(
                                    b_str,
                                    _new_b_shape,
                                    _new_a_shape[0],
                                    max(_new_a_shape[1], _new_b_shape[1]),
                                    dtype_str,
                                )
                            state.add_statement(statement_from_string(
                                f"nisa.tensor_tensor(dst={new_dst}, data1={_part_bcast_a}, "
                                f"data2={_part_bcast_b}, op={op})"
                            ))
                    else:
                        state.add_statement(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={new_dst}, data1={a}, data2={b}, op={op})"
                            )
                        )
                return new_dst

        # Check for partition-dimension broadcasting.
        # nisa.tensor_tensor requires matching partition counts on trn1, so we must
        # explicitly broadcast the [1, F] operand to [P, F] via per-partition DMA.
        # Handle both cases: a=[P,F] b=[1,F] AND a=[1,F] b=[P,F].
        def _lookup_sbuf_shape(name: str) -> list[int] | None:
            """SBUF shape lookup with copy-var suffix stripping.

            ``_nki_full_copy_0`` is a read-only copy of ``_nki_full``; we
            register the shape on the original but need to find it when
            the op references the copy.  Strip ``_copy``/``_copy_N``
            suffixes and retry.
            """
            shape = state.device_function._nki_sbuf_shapes.get(name)
            if shape is not None:
                return shape
            _lk = name
            while "_copy" in _lk:
                _lk = _lk[:_lk.rfind("_copy")]
                shape = state.device_function._nki_sbuf_shapes.get(_lk)
                if shape is not None:
                    return shape
            return None

        a_sbuf_shape = _lookup_sbuf_shape(dst)
        b_sbuf_shape = _lookup_sbuf_shape(b_str)
        _has_partition_mismatch = (
            dst_tile_vars is None
            and b_tile_vars is None
            and a_sbuf_shape is not None
            and b_sbuf_shape is not None
            and len(a_sbuf_shape) >= 2
            and len(b_sbuf_shape) >= 2
            and a_sbuf_shape[0] != b_sbuf_shape[0]
        )
        # Which operand needs broadcasting (the one with partition=1)
        _a_needs_broadcast = _has_partition_mismatch and a_sbuf_shape[0] == 1
        b_needs_broadcast = _has_partition_mismatch and b_sbuf_shape[0] == 1

        # Free-dim broadcast: [P, 1] × [P, F] or vice versa. NKI
        # tensor_tensor also requires matching free-dim extents.
        _has_free_mismatch = (
            dst_tile_vars is None
            and b_tile_vars is None
            and a_sbuf_shape is not None
            and b_sbuf_shape is not None
            and len(a_sbuf_shape) >= 2
            and len(b_sbuf_shape) >= 2
            and a_sbuf_shape[0] == b_sbuf_shape[0]
            and a_sbuf_shape[1] != b_sbuf_shape[1]
        )
        _a_needs_free_broadcast = _has_free_mismatch and a_sbuf_shape[1] == 1
        _b_needs_free_broadcast = _has_free_mismatch and b_sbuf_shape[1] == 1

        # Full numpy-style: [1, F] × [P, 1] → [P, F] (or reverse). Emit
        # two SBUF-replicating copies to produce matching-shape operands.
        _has_full_bcast = (
            dst_tile_vars is None
            and b_tile_vars is None
            and a_sbuf_shape is not None
            and b_sbuf_shape is not None
            and len(a_sbuf_shape) >= 2
            and len(b_sbuf_shape) >= 2
            and {a_sbuf_shape[0], b_sbuf_shape[0]} == {1, max(a_sbuf_shape[0], b_sbuf_shape[0])}
            and {a_sbuf_shape[1], b_sbuf_shape[1]} == {1, max(a_sbuf_shape[1], b_sbuf_shape[1])}
            and 1 in (a_sbuf_shape[0], b_sbuf_shape[0])
            and 1 in (a_sbuf_shape[1], b_sbuf_shape[1])
            and (a_sbuf_shape[0] != 1 or b_sbuf_shape[0] != 1)
            and (a_sbuf_shape[1] != 1 or b_sbuf_shape[1] != 1)
        )

        def _emit_broadcast_op(bcast_src_name: str, bcast_src_shape: list,
                               data1_name: str, data2_name: str,
                               dst_name: str, p_count: int,
                               target_dtype_str: str = "nl.float32") -> bool:
            """Broadcast bcast_src [1, F] to [P, F] via per-partition DMA, then tensor_tensor.
            Returns True if broadcast was successfully emitted."""
            hbm_sources = getattr(state.device_function, "_nki_hbm_sources", {})
            hbm_src = hbm_sources.get(bcast_src_name)
            if hbm_src is None:
                return False
            f_count = bcast_src_shape[1]
            bcast_var = _emit_partition_broadcast(
                hbm_src, f_count, p_count, target_dtype_str, bcast_src_name)
            if bcast_var is None:
                return False
            # Replace the original broadcast operand with the expanded version
            real_data1 = bcast_var if bcast_src_name == data1_name else data1_name
            real_data2 = bcast_var if bcast_src_name == data2_name else data2_name
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={dst_name}, data1={real_data1}, "
                    f"data2={real_data2}, op={op})"
                )
            )
            return True

        # Derive target dtype from the FX output node
        _bcast_dtype_str = "nl.float32"
        if cur_node is not None:
            _out_val = cur_node.meta.get("val")
            if isinstance(_out_val, torch.Tensor):
                _bcast_dtype_str = env.backend.dtype_str(_out_val.dtype)

        def _emit_sbuf_replicate(src_var: str, src_shape: list[int],
                                  p_target: int, f_target: int,
                                  dtype_str: str) -> str:
            """Replicate an SBUF [1, F] or [P, 1] or [1, 1] source into a
            [p_target, f_target] SBUF tile using per-partition tensor_copy.

            Handles the pure-SBUF broadcast case (no HBM source required).
            """
            from .ast_extension import create as _create, expr_from_string as _efrom
            out_var = state.device_function.new_var("_nki_sbuf_bcast", dce=True)
            state.device_function._nki_sbuf_shapes[out_var] = [p_target, f_target]
            state.device_function._nki_sbuf_dtypes[out_var] = dtype_str
            state.add_statement(
                statement_from_string(
                    f"{out_var} = nl.ndarray([{p_target}, {f_target}], "
                    f"{dtype_str}, buffer=nl.sbuf)"
                )
            )
            src_p, src_f = src_shape[0], src_shape[1]
            if src_p == 1 and src_f == f_target:
                # [1, F] → [P, F]: replicate the row into every partition.
                state.add_statement(
                    statement_from_string(
                        f"{out_var} = nl.broadcast_to({src_var}, "
                        f"shape=({p_target}, {f_target}))"
                    )
                )
            elif src_p == p_target and src_f == 1:
                # [P, 1] → [P, F]: replicate the column across the free dim.
                # We use a per-free-index nisa.tensor_copy from [P,1] src
                # to [P,f:f+1] dst slice.
                f_loop = state.device_function.new_var("_f_bcast")
                state.add_statement(
                    _create(
                        ast.For,
                        target=_create(ast.Name, id=f_loop, ctx=ast.Store()),
                        iter=_efrom(f"nl.affine_range({f_target})"),
                        body=[
                            statement_from_string(
                                f"nisa.tensor_copy(dst={out_var}[0:{p_target}, {f_loop}:{f_loop}+1], "
                                f"src={src_var}[0:{p_target}, 0:1])"
                            )
                        ],
                        orelse=[],
                    )
                )
            elif src_p == 1 and src_f == 1:
                # Scalar tile → full: memset to the single value... but
                # we don't have the value here. Fall back to nested copy.
                p_loop = state.device_function.new_var("_p_bcast")
                f_loop = state.device_function.new_var("_f_bcast")
                state.add_statement(
                    _create(
                        ast.For,
                        target=_create(ast.Name, id=p_loop, ctx=ast.Store()),
                        iter=_efrom(f"nl.affine_range({p_target})"),
                        body=[
                            statement_from_string(
                                f"nisa.tensor_copy(dst={out_var}[{p_loop}:{p_loop}+1, 0:{f_target}], "
                                f"src={src_var}[0:1, 0:1])"
                            )
                        ],
                        orelse=[],
                    )
                )
            return out_var

        # Full numpy broadcast: [1, F] × [P, 1] → [P, F]. Replicate both
        # operands to the full [P, F] shape before tensor_tensor.
        # GUARD: only fire if the current FX node's output val confirms a
        # [P, F] result. If the FX trace thinks the output is [1, F] or
        # [P, 1] (kernel accumulator), broadcasting would break semantics.
        _allow_full_bcast = _has_full_bcast
        if _allow_full_bcast and cur_node is not None:
            _cn_val = cur_node.meta.get("val")
            if isinstance(_cn_val, torch.Tensor):
                _cn_shape = NKIOpOverrides._squeeze_shape_2d(
                    [env.size_hint(d) if isinstance(d, torch.SymInt) else int(d) for d in _cn_val.shape]
                )
                _ptgt = max(a_sbuf_shape[0], b_sbuf_shape[0])
                _ftgt = max(a_sbuf_shape[1], b_sbuf_shape[1])
                # Must match the broadcast target shape.
                if not (len(_cn_shape) >= 2 and _cn_shape[0] == _ptgt and _cn_shape[1] == _ftgt):
                    _allow_full_bcast = False

        if _allow_full_bcast:
            p_target = max(a_sbuf_shape[0], b_sbuf_shape[0])
            f_target = max(a_sbuf_shape[1], b_sbuf_shape[1])
            a_bcast = dst if a_sbuf_shape == [p_target, f_target] else _emit_sbuf_replicate(
                dst, a_sbuf_shape, p_target, f_target, _bcast_dtype_str
            )
            b_bcast = b_str if b_sbuf_shape == [p_target, f_target] else _emit_sbuf_replicate(
                b_str, b_sbuf_shape, p_target, f_target, _bcast_dtype_str
            )
            new_dst_full = state.device_function.new_var(prefix, dce=True)
            state.device_function._nki_sbuf_shapes[new_dst_full] = [p_target, f_target]
            state.device_function._nki_sbuf_dtypes[new_dst_full] = _bcast_dtype_str
            state.add_statement(
                statement_from_string(
                    f"{new_dst_full} = nl.ndarray([{p_target}, {f_target}], "
                    f"{_bcast_dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={new_dst_full}, data1={a_bcast}, "
                    f"data2={b_bcast}, op={op})"
                )
            )
            return new_dst_full

        if b_needs_broadcast:
            p_count = a_sbuf_shape[0]
            # Try HBM-source broadcast first; it's cheaper when available.
            if _emit_broadcast_op(b_str, b_sbuf_shape, dst, b_str, dst, p_count,
                                   _bcast_dtype_str):
                return dst
            # Fall back to SBUF replication for mid-kernel allocations
            # (e.g. reductions, hl.full outputs) that have no HBM origin.
            b_bcast = _emit_sbuf_replicate(
                b_str, b_sbuf_shape, p_count, b_sbuf_shape[1], _bcast_dtype_str
            )
            new_dst_b = state.device_function.new_var(prefix, dce=True)
            state.device_function._nki_sbuf_shapes[new_dst_b] = [p_count, b_sbuf_shape[1]]
            state.device_function._nki_sbuf_dtypes[new_dst_b] = _bcast_dtype_str
            state.add_statement(
                statement_from_string(
                    f"{new_dst_b} = nl.ndarray([{p_count}, {b_sbuf_shape[1]}], "
                    f"{_bcast_dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={new_dst_b}, data1={a}, "
                    f"data2={b_bcast}, op={op})"
                )
            )
            return new_dst_b
        elif _a_needs_broadcast:
            # a=[1,F] needs broadcast to match b=[P,F]; also need new dest buffer
            p_count = b_sbuf_shape[0]
            new_dst2 = state.device_function.new_var(prefix, dce=True)
            out_shape2 = [p_count, a_sbuf_shape[1]]
            state.device_function._nki_sbuf_shapes[new_dst2] = out_shape2
            state.add_statement(
                statement_from_string(
                    f"{new_dst2} = nl.ndarray([{p_count}, {a_sbuf_shape[1]}], "
                    f"{_bcast_dtype_str}, buffer=nl.sbuf)"
                )
            )
            if _emit_broadcast_op(dst, a_sbuf_shape, dst, b_str, new_dst2, p_count,
                                   _bcast_dtype_str):
                return new_dst2
            a_bcast = _emit_sbuf_replicate(
                dst, a_sbuf_shape, p_count, a_sbuf_shape[1], _bcast_dtype_str
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={new_dst2}, data1={a_bcast}, "
                    f"data2={b_str}, op={op})"
                )
            )
            return new_dst2
        elif _has_free_mismatch:
            # Free-dim mismatch: either [P,1] or [P,F] shapes; replicate
            # the [P,1] operand across the free axis.
            p_target = a_sbuf_shape[0]
            f_target = max(a_sbuf_shape[1], b_sbuf_shape[1])
            a_bcast = dst if a_sbuf_shape[1] == f_target else _emit_sbuf_replicate(
                dst, a_sbuf_shape, p_target, f_target, _bcast_dtype_str
            )
            b_bcast = b_str if b_sbuf_shape[1] == f_target else _emit_sbuf_replicate(
                b_str, b_sbuf_shape, p_target, f_target, _bcast_dtype_str
            )
            new_dst_f = state.device_function.new_var(prefix, dce=True)
            state.device_function._nki_sbuf_shapes[new_dst_f] = [p_target, f_target]
            state.device_function._nki_sbuf_dtypes[new_dst_f] = _bcast_dtype_str
            state.add_statement(
                statement_from_string(
                    f"{new_dst_f} = nl.ndarray([{p_target}, {f_target}], "
                    f"{_bcast_dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={new_dst_f}, data1={a_bcast}, "
                    f"data2={b_bcast}, op={op})"
                )
            )
            return new_dst_f

        if dst_tile_vars is not None:
            for i, dv in enumerate(dst_tile_vars):
                bv = b_tile_vars[i] if b_tile_vars else b_str
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={dv}, data1={dv}, data2={bv}, op={op})"
                    )
                )
        else:
            state.add_statement(
                f"nisa.tensor_tensor(dst={dst}, data1={a}, data2={b}, op={op})"
            )
        return dst

    @classmethod
    def _resolve_scalar_operand(cls, x: object) -> object:
        """Resolve operand to scalar value if it's a const or var holding const."""
        if (
            type(x).__name__ == "OpsValue"
            or type(x).__module__.startswith("torch._inductor")
        ):
            x = str(x)
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, str):
            try:
                float(x)
                return x  # numeric string like "1.0"
            except (ValueError, TypeError):
                pass
            # Var holding constant (e.g. v_0 = 1.0 from lift)?
            from .compile_environment import CompileEnvironment

            env = CompileEnvironment.current()
            state = getattr(env, "_codegen_state", None)
            if state is not None and hasattr(state, "codegen"):
                cg = state.codegen
                if hasattr(cg, "get_var_constant_value"):
                    val = cg.get_var_constant_value(x)
                    if val is not None:
                        return val
            # Kernel-parameter scalar (e.g. ``alpha: float`` passed to
            # the kernel). Recognize by matching the var name against
            # registered Python-scalar arguments on DeviceFunction.
            if state is not None:
                dev_fn = state.device_function
                scalar_args = getattr(dev_fn, "_nki_scalar_arg_names", None)
                if scalar_args is not None and x in scalar_args:
                    # Return the name itself as a "numeric-like" token so
                    # callers emit it verbatim. Keep as string so
                    # tensor_scalar gets it as operand0=<name>.
                    return x
        return x

    @classmethod
    def _is_scalar_param_name(cls, x: object) -> bool:
        """Return True if ``x`` names a Python-scalar kernel parameter."""
        if not isinstance(x, str):
            return False
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return False
        dev_fn = state.device_function
        scalar_args = getattr(dev_fn, "_nki_scalar_arg_names", None)
        return scalar_args is not None and x in scalar_args

    @classmethod
    def _is_scalar_operand(cls, x: object) -> bool:
        resolved = cls._resolve_scalar_operand(x)
        return isinstance(resolved, (int, float, bool)) or cls._is_scalar_param_name(
            resolved
        )

    @staticmethod
    def _nki_tensor_scalar(
        data: object,
        operand0: object,
        op0: str,
        *,
        reverse0: bool = False,
    ) -> str:
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

        # PSUM-reuse fusion: rewrite data operand to PSUM name if upstream
        # matmul had its final copy elided. nisa.tensor_scalar runs on the
        # Vector / Scalar / GpSimd Engine; all accept PSUM input.
        data = NKIOpOverrides._resolve_psum_alias(state, data)

        def _scalar_for_emit(x: object) -> object:
            """Emit scalar as Python literal (1.0 not '1.0') for nisa.tensor_scalar."""
            if isinstance(x, str):
                try:
                    return float(x) if "." in x or "e" in x.lower() else int(x)
                except (ValueError, TypeError):
                    pass
            return x

        dst = str(data)
        dst_tile_vars = state.device_function.get_tile_list_vars(dst)
        operand_tile_vars = state.device_function.get_tile_list_vars(str(operand0))

        reverse_part = ", reverse0=True" if reverse0 else ""

        # Multi-user protection: if dst is a second-level copy var or the FX input has
        # multiple users, tensor_scalar must NOT modify the dst in-place.
        # Check ALL tensor args, not just args[0] — reverse0=True ops (e.g.
        # 1 - probs) pass probs as args[1], and if probs has multiple users
        # we still need to allocate a new dst to avoid overwriting it.
        from torch._inductor.virtualized import V as _V_ts
        _ts_node = _V_ts.current_node or state.fx_node

        def _recover_tensor_scalar_operand(operand: object) -> object:
            is_mask_literal = isinstance(operand, (int, float, bool))
            if isinstance(operand, str):
                try:
                    float(operand)
                except (TypeError, ValueError):
                    pass
                else:
                    is_mask_literal = True
            if not is_mask_literal or _ts_node is None:
                return operand
            scalar_arg_index = 0 if reverse0 else 1
            args = getattr(_ts_node, "args", ())
            if len(args) <= scalar_arg_index:
                return operand
            fx_arg = args[scalar_arg_index]
            if not hasattr(fx_arg, "meta"):
                return operand
            if not isinstance(fx_arg.meta.get("val"), torch.Tensor):
                return operand
            fx_ast = state.codegen.ast_for_fx_node(fx_arg)
            if isinstance(fx_ast, ast.AST):
                recovered = ast.unparse(fx_ast)
                if recovered:
                    return recovered
            return operand

        operand0 = _recover_tensor_scalar_operand(operand0)
        _ts_is_copy = "_copy_" in dst and dst[-1:].isdigit()
        if _ts_is_copy and _ts_node is not None and dst_tile_vars is None:
            _ts_users = getattr(_ts_node, "users", {})
            _ts_is_loop_output = (
                len(_ts_users) == 1
                and any(getattr(user, "op", None) == "output" for user in _ts_users)
            )
            if _ts_is_loop_output:
                _ts_base_dst = dst
                _ts_sbuf_shapes = state.device_function._nki_sbuf_shapes
                while _ts_base_dst not in _ts_sbuf_shapes and "_copy" in _ts_base_dst:
                    _ts_base_dst = _ts_base_dst[:_ts_base_dst.rfind("_copy")]
                if _ts_base_dst in _ts_sbuf_shapes and _ts_base_dst != dst:
                    _operand_emit_ts = _scalar_for_emit(operand0)
                    state.add_statement(
                        statement_from_string(
                            f"nisa.tensor_scalar(dst={_ts_base_dst}, data={_ts_base_dst}, "
                            f"op0={op0}, operand0={_operand_emit_ts}, op1=None{reverse_part})"
                        )
                    )
                    return _ts_base_dst
        _ts_any_arg_multi_user = False
        if _ts_node is not None:
            for _arg in getattr(_ts_node, "args", ()):
                if hasattr(_arg, "users") and len(_arg.users) > 1:
                    _ts_any_arg_multi_user = True
                    break
        _ts_multi_user = _ts_is_copy or _ts_any_arg_multi_user
        if _ts_multi_user and dst_tile_vars is None:
            _ts_shape = state.device_function._nki_sbuf_shapes.get(dst)
            if _ts_shape is None and _ts_is_copy:
                _lk = dst
                _sb = state.device_function._nki_sbuf_shapes
                while _lk not in _sb and "_copy" in _lk:
                    _lk = _lk[:_lk.rfind("_copy")]
                _ts_shape = _sb.get(_lk)
            if _ts_shape is not None and len(_ts_shape) >= 2:
                _ts_shape = NKIOpOverrides._squeeze_shape_2d(list(_ts_shape))
                _ts_out_val = _ts_node.meta.get("val") if _ts_node else None
                _ts_dtype = "nl.float32"
                if _ts_out_val is not None and isinstance(_ts_out_val, torch.Tensor):
                    _ts_dtype = env.backend.dtype_str(_ts_out_val.dtype)
                _ts_new_dst = state.device_function.new_var("_nki_ts_out", dce=True)
                state.device_function._nki_sbuf_shapes[_ts_new_dst] = list(_ts_shape)
                _ts_shape_str = ", ".join(str(d) for d in _ts_shape)
                state.add_statement(
                    statement_from_string(
                        f"{_ts_new_dst} = nl.ndarray([{_ts_shape_str}], {_ts_dtype}, buffer=nl.sbuf)"
                    )
                )

                # Layout reconcile: operand0 must be [P, 1] layout for
                # partition-scalar broadcast. Transpose [1, P] → [P, 1] if
                # needed. Look up operand0's SBUF shape.
                def _ts_lookup_shape_mu(name: str) -> list[int] | None:
                    s = state.device_function._nki_sbuf_shapes.get(name)
                    if s is not None:
                        return s
                    _lk = name
                    while "_copy" in _lk:
                        _lk = _lk[:_lk.rfind("_copy")]
                        s = state.device_function._nki_sbuf_shapes.get(_lk)
                        if s is not None:
                            return s
                    return None

                def _ts_lookup_dtype_mu(name: str) -> str:
                    dtypes = state.device_function._nki_sbuf_dtypes
                    if name in dtypes:
                        return dtypes[name]
                    _lk = name
                    while "_copy" in _lk:
                        _lk = _lk[:_lk.rfind("_copy")]
                        if _lk in dtypes:
                            return dtypes[_lk]
                    return "nl.float32"

                _operand_str_mu = str(operand0) if not isinstance(operand0, str) else operand0
                _op0_shape_mu = _ts_lookup_shape_mu(_operand_str_mu)
                _operand_for_emit_mu = operand0
                if (
                    _op0_shape_mu is not None
                    and len(_op0_shape_mu) == 2
                    and len(_ts_shape) == 2
                    and _op0_shape_mu[0] == 1
                    and _op0_shape_mu[1] > 1
                    and _op0_shape_mu[1] == _ts_shape[0]
                ):
                    _dt_mu = state.device_function._nki_sbuf_dtypes.get(
                        _operand_str_mu, "nl.float32")
                    _tr_psum_mu = state.device_function.new_var("_ts_tr_psum", dce=True)
                    _tr_sbuf_mu = state.device_function.new_var("_ts_tr_sbuf", dce=True)
                    state.device_function._nki_sbuf_shapes[_tr_sbuf_mu] = [_op0_shape_mu[1], 1]
                    state.device_function._nki_sbuf_dtypes[_tr_sbuf_mu] = _dt_mu
                    state.add_statement(statement_from_string(
                        f"{_tr_psum_mu} = nl.ndarray([{_op0_shape_mu[1]}, 1], {_dt_mu}, buffer=nl.psum)"
                    ))
                    state.add_statement(statement_from_string(
                        f"nisa.nc_transpose(dst={_tr_psum_mu}, data={_operand_str_mu})"
                    ))
                    state.add_statement(statement_from_string(
                        f"{_tr_sbuf_mu} = nl.ndarray([{_op0_shape_mu[1]}, 1], {_dt_mu}, buffer=nl.sbuf)"
                    ))
                    state.add_statement(statement_from_string(
                        f"nisa.tensor_copy(dst={_tr_sbuf_mu}, src={_tr_psum_mu})"
                    ))
                    _operand_for_emit_mu = _tr_sbuf_mu

                _operand_name_mu = str(_operand_for_emit_mu)
                _operand_shape_mu = _ts_lookup_shape_mu(_operand_name_mu)
                _operand_dtype_mu = _ts_lookup_dtype_mu(_operand_name_mu)
                if (
                    _operand_shape_mu is not None
                    and _operand_dtype_mu
                    in {"nl.int32", "nl.int16", "nl.int8", "nl.uint32", "nl.uint16", "nl.uint8"}
                ):
                    _cast_mu = state.device_function.new_var("_ts_operand_f", dce=True)
                    _shape_mu = ", ".join(str(d) for d in _operand_shape_mu)
                    state.device_function._nki_sbuf_shapes[_cast_mu] = list(_operand_shape_mu)
                    state.device_function._nki_sbuf_dtypes[_cast_mu] = "nl.float32"
                    state.add_statement(statement_from_string(
                        f"{_cast_mu} = nl.ndarray([{_shape_mu}], nl.float32, buffer=nl.sbuf)"
                    ))
                    state.add_statement(statement_from_string(
                        f"nisa.tensor_copy(dst={_cast_mu}, src={_operand_for_emit_mu})"
                    ))
                    _operand_for_emit_mu = _cast_mu

                _operand_emit_ts = _scalar_for_emit(_operand_for_emit_mu)
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_scalar(dst={_ts_new_dst}, data={data}, "
                        f"op0={op0}, operand0={_operand_emit_ts}, op1=None{reverse_part})"
                    )
                )
                # Mark this new variable as "protected" — activations on it should
                # not modify it in-place since it carries state between iterations
                if not hasattr(state.device_function, "_nki_protected_vars"):
                    state.device_function._nki_protected_vars = set()
                state.device_function._nki_protected_vars.add(_ts_new_dst)
                return _ts_new_dst

        if dst_tile_vars is not None:
            if operand_tile_vars is not None and len(operand_tile_vars) != len(dst_tile_vars):
                raise exc.BackendUnsupported(
                    "nki",
                    "tensor_scalar tile-list operand length mismatch between data and operand0",
                )
            for i, dv in enumerate(dst_tile_vars):
                operand_expr = (
                    operand_tile_vars[i]
                    if operand_tile_vars is not None
                    else _scalar_for_emit(operand0)
                )
                state.add_statement(
                    statement_from_string(
                        "nisa.tensor_scalar("
                        f"dst={dv}, data={dv}, op0={op0}, operand0={operand_expr}, op1=None{reverse_part})"
                    )
                )
            return dst

        if operand_tile_vars is not None:
            raise exc.BackendUnsupported(
                "nki",
                "tensor_scalar does not support non-tile-list data with tile-list operand0",
            )

        # Layout reconcile: tensor_scalar needs operand0 in [P, 1] layout
        # (per-partition scalar). When operand0 is a SBUF tensor in [1, N]
        # layout but semantically represents one value per partition (e.g.
        # lse [1, N] used as per-row scalar in [N, V] × [N, 1] broadcast),
        # transpose [1, N] → [N, 1] so tensor_scalar can consume it.
        def _ts_lookup_shape(name: str) -> list[int] | None:
            s = state.device_function._nki_sbuf_shapes.get(name)
            if s is not None:
                return s
            _lk = name
            while "_copy" in _lk:
                _lk = _lk[:_lk.rfind("_copy")]
                s = state.device_function._nki_sbuf_shapes.get(_lk)
                if s is not None:
                    return s
            return None

        def _ts_lookup_dtype(name: str) -> str:
            dtypes = state.device_function._nki_sbuf_dtypes
            if name in dtypes:
                return dtypes[name]
            _lk = name
            while "_copy" in _lk:
                _lk = _lk[:_lk.rfind("_copy")]
                if _lk in dtypes:
                    return dtypes[_lk]
            return "nl.float32"

        operand_str = str(operand0) if not isinstance(operand0, str) else operand0
        op0_shape = _ts_lookup_shape(operand_str)
        dst_shape = _ts_lookup_shape(dst)
        operand_for_emit = operand0
        if (
            op0_shape is not None
            and dst_shape is not None
            and len(op0_shape) == 2
            and len(dst_shape) == 2
            and op0_shape[0] == 1
            and op0_shape[1] > 1
            and op0_shape[1] == dst_shape[0]
        ):
            # Need [P, 1] but have [1, P] — emit nc_transpose
            _dt = state.device_function._nki_sbuf_dtypes.get(operand_str, "nl.float32")
            tr_psum = state.device_function.new_var("_ts_tr_psum", dce=True)
            tr_sbuf = state.device_function.new_var("_ts_tr_sbuf", dce=True)
            state.device_function._nki_sbuf_shapes[tr_sbuf] = [op0_shape[1], 1]
            state.device_function._nki_sbuf_dtypes[tr_sbuf] = _dt
            state.add_statement(statement_from_string(
                f"{tr_psum} = nl.ndarray([{op0_shape[1]}, 1], {_dt}, buffer=nl.psum)"
            ))
            state.add_statement(statement_from_string(
                f"nisa.nc_transpose(dst={tr_psum}, data={operand_str})"
            ))
            state.add_statement(statement_from_string(
                f"{tr_sbuf} = nl.ndarray([{op0_shape[1]}, 1], {_dt}, buffer=nl.sbuf)"
            ))
            state.add_statement(statement_from_string(
                f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
            ))
            operand_for_emit = tr_sbuf

        operand_name = str(operand_for_emit)
        operand_shape = _ts_lookup_shape(operand_name)
        operand_dtype = _ts_lookup_dtype(operand_name)
        if (
            operand_shape is not None
            and operand_dtype
            in {"nl.int32", "nl.int16", "nl.int8", "nl.uint32", "nl.uint16", "nl.uint8"}
        ):
            cast_operand = state.device_function.new_var("_ts_operand_f", dce=True)
            operand_shape_str = ", ".join(str(d) for d in operand_shape)
            state.device_function._nki_sbuf_shapes[cast_operand] = list(operand_shape)
            state.device_function._nki_sbuf_dtypes[cast_operand] = "nl.float32"
            state.add_statement(statement_from_string(
                f"{cast_operand} = nl.ndarray([{operand_shape_str}], nl.float32, buffer=nl.sbuf)"
            ))
            state.add_statement(statement_from_string(
                f"nisa.tensor_copy(dst={cast_operand}, src={operand_for_emit})"
            ))
            operand_for_emit = cast_operand

        operand_emit = _scalar_for_emit(operand_for_emit)
        state.add_statement(
            statement_from_string(
                "nisa.tensor_scalar("
                f"dst={dst}, data={data}, op0={op0}, operand0={operand_emit}, op1=None{reverse_part})"
            )
        )
        return dst

    @classmethod
    def _nki_binary_op(
        cls,
        a: object,
        b: object,
        *,
        op_tensor_tensor: str,
        op_tensor_scalar: str,
        allow_tensor_tensor: bool = True,
    ) -> str:
        from torch._inductor.virtualized import V as _V_bin

        _bin_node = _V_bin.current_node
        if _bin_node is None:
            from .compile_environment import CompileEnvironment as _CE_bin

            _bin_state = getattr(_CE_bin.current(), "_codegen_state", None)
            _bin_node = getattr(_bin_state, "fx_node", None)
        if _bin_node is not None and cls._used_only_as_memory_index(_bin_node):
            # The NKI load/store lowering pattern-matches the original FX
            # index expression directly. Emitting the arithmetic expression as
            # a real SBUF op here is both unnecessary and can be invalid for
            # broadcasted index expressions such as starts[:, None] + i[None, :].
            return "0"

        if _bin_node is not None:
            from .compile_environment import CompileEnvironment as _CE_bin

            _bin_state = getattr(_CE_bin.current(), "_codegen_state", None)

            def _recover_tensor_ast_operand(operand: object, arg_index: int) -> object:
                is_mask_literal = isinstance(operand, (int, float, bool))
                if isinstance(operand, str):
                    try:
                        float(operand)
                    except (TypeError, ValueError):
                        pass
                    else:
                        is_mask_literal = True
                if not is_mask_literal:
                    return operand
                if _bin_state is None or len(_bin_node.args) <= arg_index:
                    return operand
                fx_arg = _bin_node.args[arg_index]
                if not hasattr(fx_arg, "meta"):
                    return operand
                if not isinstance(fx_arg.meta.get("val"), torch.Tensor):
                    return operand
                fx_ast = _bin_state.codegen.ast_for_fx_node(fx_arg)
                if isinstance(fx_ast, ast.AST):
                    recovered = ast.unparse(fx_ast)
                    if recovered:
                        return recovered
                return operand

            # Some tile loads have masked-value metadata of 0. When a scalar-like
            # tensor view such as mean_mb[:, None] reaches Inductor binary
            # lowering, the operand can arrive as that neutral literal even
            # though the real SBUF load already exists. Recover the AST so NKI
            # tensor_scalar uses the loaded tile as operand0.
            a = _recover_tensor_ast_operand(a, 0)
            b = _recover_tensor_ast_operand(b, 1)

        if _bin_node is not None and op_tensor_tensor == "nl.add":
            from .ast_extension import statement_from_string
            from .compile_environment import CompileEnvironment as _CE_acc

            _acc_state = getattr(_CE_acc.current(), "_codegen_state", None)
            if _acc_state is not None:
                _users = getattr(_bin_node, "users", {})
                _is_loop_output = (
                    len(_users) == 1
                    and any(getattr(user, "op", None) == "output" for user in _users)
                )
                _a_name = ast.unparse(a) if isinstance(a, ast.AST) else str(a)
                _b_name = ast.unparse(b) if isinstance(b, ast.AST) else str(b)

                def _lookup_loop_acc_shape(name: str) -> list[int] | None:
                    shape = _acc_state.device_function._nki_sbuf_shapes.get(name)
                    if shape is not None:
                        return shape
                    lookup = name
                    while "_copy" in lookup:
                        lookup = lookup[: lookup.rfind("_copy")]
                        shape = _acc_state.device_function._nki_sbuf_shapes.get(
                            lookup
                        )
                        if shape is not None:
                            return shape
                    return None

                if (
                    _is_loop_output
                    and "_copy_" in _a_name
                    and _a_name[-1:].isdigit()
                    and _acc_state.device_function.get_tile_list_vars(_a_name) is None
                    and _acc_state.device_function.get_tile_list_vars(_b_name) is None
                ):
                    _base_name = _a_name
                    _sbuf_shapes = _acc_state.device_function._nki_sbuf_shapes
                    while _base_name not in _sbuf_shapes and "_copy" in _base_name:
                        _base_name = _base_name[: _base_name.rfind("_copy")]
                    _base_shape = _lookup_loop_acc_shape(_base_name)
                    _b_shape = _lookup_loop_acc_shape(_b_name)
                    if (
                        _base_name != _a_name
                        and _base_shape is not None
                        and _b_shape is not None
                        and list(_base_shape) == list(_b_shape)
                    ):
                        _acc_state.add_statement(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={_base_name}, "
                                f"data1={_base_name}, data2={_b_name}, "
                                f"op={op_tensor_tensor})"
                            )
                        )
                        return _base_name

        a_is_scalar = cls._is_scalar_operand(a)
        b_is_scalar = cls._is_scalar_operand(b)

        if a_is_scalar and b_is_scalar:
            raise exc.BackendUnsupported(
                "nki",
                "both operands are host scalars; expected at least one tile operand",
            )

        # tensor <op> scalar
        if not a_is_scalar and b_is_scalar:
            return cls._nki_tensor_scalar(a, cls._resolve_scalar_operand(b), op_tensor_scalar)

        # scalar <op> tensor
        if a_is_scalar and not b_is_scalar:
            reverse0 = op_tensor_scalar in {"nl.subtract", "nl.divide"}
            return cls._nki_tensor_scalar(b, cls._resolve_scalar_operand(a), op_tensor_scalar, reverse0=reverse0)

        # tensor <op> scalar-like FX rhs (e.g. runtime scalar kernel arg)
        if not a_is_scalar and not b_is_scalar and cls._is_scalar_like_tensor(b):
            return cls._nki_tensor_scalar(a, b, op_tensor_scalar)

        # scalar-like FX lhs <op> tensor.  This comes up for dynamic row
        # index construction such as ``group_start + tile.index`` where the
        # group start is a [1, 1] SBUF tile and the tile index is [1, P].
        if not a_is_scalar and not b_is_scalar and _bin_node is not None:
            lhs_shape = cls._shape_from_node_arg(_bin_node.args[0])
            if lhs_shape is not None:
                def _shape_is_scalar_like(shape: tuple[object, ...]) -> bool:
                    if len(shape) == 0:
                        return True
                    if all(d == 1 for d in shape):
                        return True
                    from .compile_environment import CompileEnvironment
                    env = CompileEnvironment.current()
                    state = getattr(env, "_codegen_state", None)
                    if state is None or state.config is None:
                        return False
                    import sympy as _sympy
                    bs_subs: dict[_sympy.Symbol, int] = {}
                    for bs in env.block_sizes:
                        cfg = bs.from_config(state.config)
                        if isinstance(cfg, int):
                            bs_subs[bs.symbol()] = cfg
                    resolved: list[int] = []
                    for dim in shape:
                        if isinstance(dim, torch.SymInt):
                            try:
                                resolved.append(int(dim._sympy_().subs(bs_subs)))
                            except (TypeError, ValueError):
                                resolved.append(int(env.size_hint(dim)))
                        else:
                            resolved.append(int(dim))
                    return all(dim == 1 for dim in resolved)

                if _shape_is_scalar_like(lhs_shape):
                    reverse0 = op_tensor_scalar in {"nl.subtract", "nl.divide"}
                    return cls._nki_tensor_scalar(
                        b, a, op_tensor_scalar, reverse0=reverse0
                    )

        if (
            not a_is_scalar
            and not b_is_scalar
            and _bin_node is not None
            and op_tensor_tensor == "nl.add"
        ):
            from .ast_extension import statement_from_string
            from .compile_environment import CompileEnvironment as _CE_acc

            _acc_state = getattr(_CE_acc.current(), "_codegen_state", None)
            if _acc_state is not None:
                _users = getattr(_bin_node, "users", {})
                _is_loop_output = (
                    len(_users) == 1
                    and any(getattr(user, "op", None) == "output" for user in _users)
                )
                _a_name = ast.unparse(a) if isinstance(a, ast.AST) else str(a)
                _b_name = ast.unparse(b) if isinstance(b, ast.AST) else str(b)

                def _lookup_loop_acc_shape(name: str) -> list[int] | None:
                    shape = _acc_state.device_function._nki_sbuf_shapes.get(name)
                    if shape is not None:
                        return shape
                    lookup = name
                    while "_copy" in lookup:
                        lookup = lookup[: lookup.rfind("_copy")]
                        shape = _acc_state.device_function._nki_sbuf_shapes.get(
                            lookup
                        )
                        if shape is not None:
                            return shape
                    return None

                if (
                    _is_loop_output
                    and "_copy_" in _a_name
                    and _a_name[-1:].isdigit()
                    and _acc_state.device_function.get_tile_list_vars(_a_name) is None
                    and _acc_state.device_function.get_tile_list_vars(_b_name) is None
                ):
                    _base_name = _a_name
                    _sbuf_shapes = _acc_state.device_function._nki_sbuf_shapes
                    while _base_name not in _sbuf_shapes and "_copy" in _base_name:
                        _base_name = _base_name[: _base_name.rfind("_copy")]
                    _base_shape = _lookup_loop_acc_shape(_base_name)
                    _b_shape = _lookup_loop_acc_shape(_b_name)
                    if (
                        _base_name != _a_name
                        and _base_shape is not None
                        and _b_shape is not None
                        and list(_base_shape) == list(_b_shape)
                    ):
                        _acc_state.add_statement(
                            statement_from_string(
                                f"nisa.tensor_tensor(dst={_base_name}, "
                                f"data1={_base_name}, data2={_b_name}, "
                                f"op={op_tensor_tensor})"
                            )
                        )
                        return _base_name

        # Check for broadcast pattern: [M,N] op [M,1] after 3D squeeze
        # This handles cases like acc * alpha[:, :, None] where alpha is [1,M,1]
        if not a_is_scalar and not b_is_scalar:
            if _bin_node is not None and len(_bin_node.args) >= 2:
                _lhs_s = cls._shape_from_node_arg(_bin_node.args[0])
                _rhs_s = cls._shape_from_node_arg(_bin_node.args[1])
                if _lhs_s is not None and _rhs_s is not None:
                    _lhs_sq = tuple(cls._squeeze_shape_2d(list(_lhs_s)))
                    _rhs_sq = tuple(cls._squeeze_shape_2d(list(_rhs_s)))
                    if (len(_lhs_sq) == 2 and len(_rhs_sq) == 2
                            and _lhs_sq[0] == _rhs_sq[0] and _rhs_sq[1] == 1
                            and _lhs_sq[1] != 1):
                        return cls._nki_tensor_scalar(a, b, op_tensor_scalar)
                    # Also handle reverse: [M,1] op [M,N]
                    if (len(_lhs_sq) == 2 and len(_rhs_sq) == 2
                            and _lhs_sq[0] == _rhs_sq[0] and _lhs_sq[1] == 1
                            and _rhs_sq[1] != 1):
                        reverse0 = op_tensor_scalar in {"nl.subtract", "nl.divide"}
                        return cls._nki_tensor_scalar(b, a, op_tensor_scalar, reverse0=reverse0)

        if not a_is_scalar and not b_is_scalar and _bin_node is not None:
            from .ast_extension import create as _create
            from .ast_extension import expr_from_string as _efrom
            from .ast_extension import statement_from_string
            from .compile_environment import CompileEnvironment

            env = CompileEnvironment.current()
            state = getattr(env, "_codegen_state", None)

            def _lookup_sbuf_shape(name: str) -> list[int] | None:
                if state is None:
                    return None
                shape = state.device_function._nki_sbuf_shapes.get(name)
                if shape is not None:
                    return shape
                lookup = name
                while "_copy" in lookup:
                    lookup = lookup[: lookup.rfind("_copy")]
                    shape = state.device_function._nki_sbuf_shapes.get(lookup)
                    if shape is not None:
                        return shape
                return None

            def _lookup_sbuf_dtype(name: str, default: str) -> str:
                if state is None:
                    return default
                dtypes = state.device_function._nki_sbuf_dtypes
                if name in dtypes:
                    return dtypes[name]
                lookup = name
                while "_copy" in lookup:
                    lookup = lookup[: lookup.rfind("_copy")]
                    if lookup in dtypes:
                        return dtypes[lookup]
                return default

            def _emit_row_replicate(
                src: str, p_target: int, f_target: int, dtype: str
            ) -> str:
                assert state is not None
                out = state.device_function.new_var("_nki_cross_bcast", dce=True)
                state.device_function._nki_sbuf_shapes[out] = [p_target, f_target]
                state.device_function._nki_sbuf_dtypes[out] = dtype
                state.add_statement(
                    statement_from_string(
                        f"{out} = nl.broadcast_to({src}, "
                        f"shape=({p_target}, {f_target}))"
                    )
                )
                return out

            def _emit_col_replicate(
                src: str, p_target: int, f_target: int, dtype: str
            ) -> str:
                assert state is not None
                out = state.device_function.new_var("_nki_cross_bcast", dce=True)
                state.device_function._nki_sbuf_shapes[out] = [p_target, f_target]
                state.device_function._nki_sbuf_dtypes[out] = dtype
                state.add_statement(
                    statement_from_string(
                        f"{out} = nl.ndarray([{p_target}, {f_target}], "
                        f"{dtype}, buffer=nl.sbuf)"
                    )
                )
                f_loop = state.device_function.new_var("_f_bcast")
                state.add_statement(
                    _create(
                        ast.For,
                        target=_create(ast.Name, id=f_loop, ctx=ast.Store()),
                        iter=_efrom(f"nl.affine_range({f_target})"),
                        body=[
                            statement_from_string(
                                f"nisa.tensor_copy(dst={out}[0:{p_target}, {f_loop}:{f_loop}+1], "
                                f"src={src}[0:{p_target}, 0:1])"
                            )
                        ],
                        orelse=[],
                    )
                )
                return out

            def _emit_row_to_col(src: str, p_target: int, dtype: str) -> str:
                assert state is not None
                int_dtypes = {
                    "nl.int32",
                    "nl.int16",
                    "nl.int8",
                    "nl.uint32",
                    "nl.uint16",
                    "nl.uint8",
                }
                transpose_src = src
                transpose_dtype = dtype
                if dtype in int_dtypes:
                    cast_in = state.device_function.new_var(
                        "_nki_cross_cast", dce=True
                    )
                    state.device_function._nki_sbuf_shapes[cast_in] = [1, p_target]
                    state.device_function._nki_sbuf_dtypes[cast_in] = "nl.float32"
                    state.add_statement(
                        statement_from_string(
                            f"{cast_in} = nl.ndarray([1, {p_target}], "
                            "nl.float32, buffer=nl.sbuf)"
                        )
                    )
                    state.add_statement(
                        statement_from_string(
                            f"nisa.tensor_copy(dst={cast_in}, src={src})"
                        )
                    )
                    transpose_src = cast_in
                    transpose_dtype = "nl.float32"
                tr_psum = state.device_function.new_var(
                    "_nki_cross_tr_psum", dce=True
                )
                tr_sbuf = state.device_function.new_var(
                    "_nki_cross_tr_sbuf", dce=True
                )
                state.device_function._nki_sbuf_shapes[tr_sbuf] = [p_target, 1]
                state.device_function._nki_sbuf_dtypes[tr_sbuf] = dtype
                state.add_statement(
                    statement_from_string(
                        f"{tr_psum} = nl.ndarray([{p_target}, 1], "
                        f"{transpose_dtype}, buffer=nl.psum)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.nc_transpose(dst={tr_psum}, data={transpose_src})"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"{tr_sbuf} = nl.ndarray([{p_target}, 1], "
                        f"{dtype}, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
                    )
                )
                return tr_sbuf

            if state is not None:
                a_name = ast.unparse(a) if isinstance(a, ast.AST) else str(a)
                b_name = ast.unparse(b) if isinstance(b, ast.AST) else str(b)
                a_shape = _lookup_sbuf_shape(a_name)
                b_shape = _lookup_sbuf_shape(b_name)
                out_val = _bin_node.meta.get("val")
                if (
                    isinstance(out_val, torch.Tensor)
                    and a_shape is not None
                    and b_shape is not None
                    and len(a_shape) == 2
                    and len(b_shape) == 2
                    and a_shape[0] == 1
                    and b_shape[0] == 1
                    and out_val.ndim >= 2
                ):
                    out_shape = cls._squeeze_shape_2d(
                        [
                            env.size_hint(d) if isinstance(d, torch.SymInt) else int(d)
                            for d in out_val.shape
                        ]
                    )
                    if len(out_shape) >= 2 and out_shape[0] > 1 and out_shape[1] > 1:
                        p_target, f_target = out_shape[0], out_shape[1]
                        if {a_shape[1], b_shape[1]} == {p_target, f_target}:
                            out_dtype = (
                                "nl.int32"
                                if op_tensor_tensor.startswith("nl.bitwise")
                                else env.backend.dtype_str(out_val.dtype)
                            )

                            def _cast_for_bitwise(
                                name: str, shape: list[int], dtype: str
                            ) -> tuple[str, str]:
                                if not op_tensor_tensor.startswith("nl.bitwise"):
                                    return name, dtype
                                if dtype == out_dtype:
                                    return name, dtype
                                casted = state.device_function.new_var(
                                    "_nki_cross_bitcast", dce=True
                                )
                                state.device_function._nki_sbuf_shapes[casted] = list(
                                    shape
                                )
                                state.device_function._nki_sbuf_dtypes[casted] = (
                                    out_dtype
                                )
                                shape_str = ", ".join(str(d) for d in shape)
                                state.add_statement(
                                    statement_from_string(
                                        f"{casted} = nl.ndarray([{shape_str}], "
                                        f"{out_dtype}, buffer=nl.sbuf)"
                                    )
                                )
                                state.add_statement(
                                    statement_from_string(
                                        f"nisa.tensor_copy(dst={casted}, src={name})"
                                    )
                                )
                                return casted, out_dtype

                            def _expand(name: str, shape: list[int]) -> str:
                                dtype = _lookup_sbuf_dtype(name, out_dtype)
                                name, dtype = _cast_for_bitwise(name, shape, dtype)
                                if shape[1] == p_target:
                                    col = _emit_row_to_col(name, p_target, dtype)
                                    return _emit_col_replicate(
                                        col, p_target, f_target, out_dtype
                                    )
                                return _emit_row_replicate(
                                    name, p_target, f_target, dtype
                                )

                            a_expanded = _expand(a_name, a_shape)
                            b_expanded = _expand(b_name, b_shape)
                            out = state.device_function.new_var(
                                "nki_binary", dce=True
                            )
                            state.device_function._nki_sbuf_shapes[out] = [
                                p_target,
                                f_target,
                            ]
                            state.device_function._nki_sbuf_dtypes[out] = out_dtype
                            state.add_statement(
                                statement_from_string(
                                    f"{out} = nl.ndarray([{p_target}, {f_target}], "
                                    f"{out_dtype}, buffer=nl.sbuf)"
                                )
                            )
                            state.add_statement(
                                statement_from_string(
                                    f"nisa.tensor_tensor(dst={out}, "
                                    f"data1={a_expanded}, data2={b_expanded}, "
                                    f"op={op_tensor_tensor})"
                                )
                            )
                            return out

        # tensor <op> tensor
        if allow_tensor_tensor:
            return cls._nki_tensor_tensor(a, b, op_tensor_tensor, "nki_binary")

        raise exc.BackendUnsupported(
            "nki",
            f"single-op tensor_scalar path does not support tensor/tensor for op {op_tensor_scalar}",
        )

    @classmethod
    def _nki_tensor_tensor_fresh(
        cls, a: object, b: object, op: str, prefix: str
    ) -> str | None:
        """Emit a tensor/tensor op into a fresh SBUF when shapes already match."""
        import ast as _ast

        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return None
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return None

        a = cls._resolve_psum_alias(state, a)
        b = cls._resolve_psum_alias(state, b)
        a_name = _ast.unparse(a) if isinstance(a, _ast.AST) else str(a)
        b_name = _ast.unparse(b) if isinstance(b, _ast.AST) else str(b)

        def _lookup_shape(name: str) -> list[int] | None:
            shape = state.device_function._nki_sbuf_shapes.get(name)
            if shape is not None:
                return shape
            lookup = name
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                shape = state.device_function._nki_sbuf_shapes.get(lookup)
                if shape is not None:
                    return shape
            return None

        def _lookup_dtype(name: str) -> str | None:
            dtypes = state.device_function._nki_sbuf_dtypes
            dtype = dtypes.get(name)
            if dtype is not None:
                return dtype
            lookup = name
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                dtype = dtypes.get(lookup)
                if dtype is not None:
                    return dtype
            return None

        a_shape = _lookup_shape(a_name)
        b_shape = _lookup_shape(b_name)
        from torch._inductor.virtualized import V

        cur_node = V.current_node or state.fx_node
        out_val = cur_node.meta.get("val") if cur_node is not None else None
        if a_shape is None or b_shape is None or a_shape != b_shape:
            return None
        if isinstance(out_val, torch.Tensor):
            out_shape = cls._squeeze_shape_2d(list(out_val.shape))
            if len(out_shape) == 2 and list(out_shape) != list(a_shape):
                return None
        a_dtype = _lookup_dtype(a_name)
        b_dtype = _lookup_dtype(b_name)
        input_dtype = a_dtype or b_dtype
        if op.startswith("nl.bitwise") and input_dtype is not None:
            dtype_str = "nl.int32"
        elif isinstance(out_val, torch.Tensor):
            dtype_str = env.backend.dtype_str(out_val.dtype)
        else:
            dtype_str = input_dtype or "nl.int32"

        def _cast_bitwise_operand(name: str, dtype: str | None) -> str:
            if not op.startswith("nl.bitwise") or dtype == dtype_str:
                return name
            casted = state.device_function.new_var("_nki_bitwise_cast", dce=True)
            state.device_function._nki_sbuf_shapes[casted] = list(a_shape)
            state.device_function._nki_sbuf_dtypes[casted] = dtype_str
            state.add_statement(
                statement_from_string(
                    f"{casted} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(f"nisa.tensor_copy(dst={casted}, src={name})")
            )
            return casted

        shape_str = ", ".join(str(d) for d in a_shape)
        a_emit = _cast_bitwise_operand(a_name, a_dtype)
        b_emit = _cast_bitwise_operand(b_name, b_dtype)
        out = state.device_function.new_var(prefix, dce=True)
        state.device_function._nki_sbuf_shapes[out] = list(a_shape)
        state.device_function._nki_sbuf_dtypes[out] = dtype_str
        state.add_statement(
            statement_from_string(
                f"{out} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_tensor(dst={out}, data1={a_emit}, data2={b_emit}, op={op})"
            )
        )
        return out

    @staticmethod
    def _used_only_as_memory_index(node: object) -> bool:
        users = getattr(node, "users", None)
        if not users:
            return False

        def contains(thing: object, needle: object) -> bool:
            if thing is needle:
                return True
            if isinstance(thing, (list, tuple)):
                return any(contains(item, needle) for item in thing)
            if isinstance(thing, dict):
                return any(contains(item, needle) for item in thing.values())
            return False

        for user in users:
            if getattr(user, "op", None) != "call_function":
                return False
            target_name = str(getattr(getattr(user, "target", None), "__name__", ""))
            if target_name not in {"load", "store"}:
                return False
            args = getattr(user, "args", ())
            if len(args) < 2 or not contains(args[1], node):
                return False
        return True

    @staticmethod
    def _shape_from_node_arg(arg: object) -> tuple[object, ...] | None:
        if not hasattr(arg, "meta"):
            return None
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            return tuple(val.shape)
        return None

    @classmethod
    def _is_scalar_like_tensor(cls, x: object) -> bool:
        """Check if the RHS operand of the current binary node is scalar-like.

        Returns True if the RHS is a 0-dim tensor, a [1]-shaped tensor (or [1,1],
        etc.), or a non-tensor symbolic value (e.g. a scalar function parameter).

        SymInt dimensions are first checked statically, then resolved against the
        current config so that tensors like [u_inner, 1] with FixedBlockSizeSource(1)
        (config=1, hint=64) are correctly identified as scalar-like.
        """
        from torch._inductor.virtualized import V

        node = V.current_node
        if node is None or len(node.args) < 2:
            return False
        rhs = node.args[1]
        if not hasattr(rhs, "meta"):
            return False
        val = rhs.meta.get("val")
        if val is None:
            return False
        if not isinstance(val, torch.Tensor):
            return True
        shape = tuple(val.shape)
        if len(shape) == 0 or all(d == 1 for d in shape):
            return True
        # Slow path: resolve any SymInt dims using configured block size values.
        # Needed when a tile has block_size=1 (FixedBlockSizeSource) but its tracing
        # hint is >1, making shape dims appear non-unit until config substitution.
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        state = getattr(env, "_codegen_state", None)
        if state is None or state.config is None:
            return False
        import sympy as _sympy

        _bs_subs: dict[_sympy.Symbol, int] = {}
        for _bs in env.block_sizes:
            _cfg = _bs.from_config(state.config)
            if isinstance(_cfg, int):
                _bs_subs[_bs.symbol()] = _cfg
        resolved = []
        for d in shape:
            if isinstance(d, torch.SymInt):
                try:
                    resolved.append(int(d._sympy_().subs(_bs_subs)))
                except (TypeError, ValueError):
                    resolved.append(int(env.size_hint(d)))
            else:
                resolved.append(int(d))
        return all(d == 1 for d in resolved)

    @classmethod
    def _truediv_tensor_tensor_supported(cls) -> bool:
        """Check if tensor/tensor truediv can use reciprocal+tensor_scalar.

        Supports [M,N]/[M,1] broadcast and same-shape divisions where
        one dimension is 1 (NKI tensor_scalar broadcasts over free dim).
        Also handles 3D+ shapes with leading 1-dims (from batch_block=1).
        """
        from torch._inductor.virtualized import V

        node = V.current_node
        if node is None or len(node.args) < 2:
            return False

        lhs_shape = cls._shape_from_node_arg(node.args[0])
        rhs_shape = cls._shape_from_node_arg(node.args[1])
        if lhs_shape is None or rhs_shape is None:
            return False
        # Squeeze leading dims for 3D+ shapes (3D→2D at kernel entry).
        # Dims may be symbolic (SymInt) so just drop all leading dims to get last 2.
        lhs_shape = lhs_shape[-2:] if len(lhs_shape) >= 2 else lhs_shape
        rhs_shape = rhs_shape[-2:] if len(rhs_shape) >= 2 else rhs_shape
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            return False
        return lhs_shape[0] == rhs_shape[0] and rhs_shape[1] == 1

    @classmethod
    def _truediv_tensor_tensor_broadcast_supported(cls) -> bool:
        """True for tensor/tensor division where RHS numpy-broadcasts to LHS."""
        from torch._inductor.virtualized import V

        node = V.current_node
        if node is None or len(node.args) < 2:
            return False
        lhs_shape = cls._shape_from_node_arg(node.args[0])
        rhs_shape = cls._shape_from_node_arg(node.args[1])
        if lhs_shape is None or rhs_shape is None:
            return False

        lhs_shape = lhs_shape[-2:] if len(lhs_shape) >= 2 else lhs_shape
        rhs_shape = rhs_shape[-2:] if len(rhs_shape) >= 2 else rhs_shape
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            return False

        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        state = getattr(env, "_codegen_state", None)
        if state is None or state.config is None:
            return False
        import sympy as _sympy

        bs_subs: dict[_sympy.Symbol, int] = {}
        for bs in env.block_sizes:
            cfg = bs.from_config(state.config)
            if isinstance(cfg, int):
                bs_subs[bs.symbol()] = cfg

        def _resolve_dim(dim: object) -> int:
            if isinstance(dim, int):
                return dim
            if isinstance(dim, torch.SymInt):
                try:
                    return int(dim._sympy_().subs(bs_subs))
                except (TypeError, ValueError):
                    return int(env.size_hint(dim))
            return int(dim)

        lhs = [_resolve_dim(dim) for dim in lhs_shape]
        rhs = [_resolve_dim(dim) for dim in rhs_shape]
        return (
            all(rdim in (1, ldim) for ldim, rdim in zip(lhs, rhs, strict=True))
            and lhs != rhs
            and lhs[0] > 1
            and lhs[1] > 1
        )

    @classmethod
    def _same_shape_tensor_tensor(cls) -> bool:
        """True when both operands have the same shape (any dimensionality).
        Squeezes leading 1-dims for 3D+ shapes."""
        from torch._inductor.virtualized import V

        node = V.current_node
        if node is None or len(node.args) < 2:
            return False
        lhs_shape = cls._shape_from_node_arg(node.args[0])
        rhs_shape = cls._shape_from_node_arg(node.args[1])
        if lhs_shape is None or rhs_shape is None:
            return False
        # Use last 2 dims for 3D+ shapes (3D→2D at kernel entry)
        lhs_shape = lhs_shape[-2:] if len(lhs_shape) >= 2 else lhs_shape
        rhs_shape = rhs_shape[-2:] if len(rhs_shape) >= 2 else rhs_shape
        if len(lhs_shape) != len(rhs_shape):
            return False
        return all(a == b for a, b in zip(lhs_shape, rhs_shape))

    @classmethod
    def _subtract_tensor_tensor_supported(cls) -> bool:
        """True when [M,N] - [M,1] broadcast; use tensor_scalar instead of tensor_tensor."""
        return cls._truediv_tensor_tensor_supported()

    @staticmethod
    def _nki_reciprocal_operand(operand: object) -> object:
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        resolved = NKIOpOverrides._resolve_scalar_operand(operand)
        if isinstance(resolved, (int, float, bool)):
            value = float(resolved)
            if value == 0.0:
                raise exc.BackendUnsupported(
                    "nki",
                    "truediv with zero scalar denominator is unsupported",
                )
            return repr(1.0 / value)

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return operand
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return operand

        operand_str = str(operand)

        def _refine_reciprocal(recip_var: str, denom_var: str, shape: list[int]) -> None:
            shape_str = ", ".join(str(d) for d in shape)
            for _ in range(2):
                prod = state.device_function.new_var("nki_reciprocal_prod", dce=True)
                correction = state.device_function.new_var(
                    "nki_reciprocal_correction", dce=True
                )
                state.device_function._nki_sbuf_shapes[prod] = list(shape)
                state.device_function._nki_sbuf_shapes[correction] = list(shape)
                state.device_function._nki_sbuf_dtypes[prod] = "nl.float32"
                state.device_function._nki_sbuf_dtypes[correction] = "nl.float32"
                state.add_statement(
                    statement_from_string(
                        f"{prod} = nl.ndarray([{shape_str}], nl.float32, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={prod}, data1={denom_var}, "
                        f"data2={recip_var}, op=nl.multiply)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"{correction} = nl.ndarray([{shape_str}], nl.float32, "
                        "buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(f"nisa.memset({correction}, value=2.0)")
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={correction}, data1={correction}, "
                        f"data2={prod}, op=nl.subtract)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={recip_var}, data1={recip_var}, "
                        f"data2={correction}, op=nl.multiply)"
                    )
                )

        # Python-scalar kernel arg (e.g. ``temperature: float``). Emit
        # ``(1.0 / name)`` as a bare expression; the caller will use it
        # in a tensor_scalar as the scalar operand.
        scalar_args = getattr(state.device_function, "_nki_scalar_arg_names", set())
        if operand_str in scalar_args:
            return f"(1.0 / {operand_str})"

        operand_tile_vars = state.device_function.get_tile_list_vars(operand_str)
        if operand_tile_vars is not None:
            for ov in operand_tile_vars:
                state.add_statement(
                    statement_from_string(
                        f"nisa.activation(dst={ov}, op=nl.reciprocal, data={ov})"
                    )
                )
            return operand

        # Check for second-level copy vars OR protected vars (created by tensor_scalar
        # as new-value buffers that carry state between loop iterations).
        # Only protect inside nested loops (≥2 active device loops); outside the inner
        # loop the variable is terminal and can be modified in-place.
        _protected_vars = getattr(state.device_function, "_nki_protected_vars", set())
        _in_nested_loop = len(getattr(state.codegen, "active_device_loops", {})) >= 2
        _is_protected = operand_str in _protected_vars and _in_nested_loop
        _is_copy = "_copy_" in operand_str and operand_str[-1:].isdigit()
        if _is_copy or _is_protected:
            _lookup = operand_str
            _sbuf = state.device_function._nki_sbuf_shapes
            # For copy vars, strip suffix to find original shape
            while _lookup not in _sbuf and "_copy" in _lookup:
                idx = _lookup.rfind("_copy")
                _lookup = _lookup[:idx]
            src_shape = _sbuf.get(_lookup)
            # For protected vars, the shape is registered directly
            if src_shape is None and _is_protected:
                src_shape = _sbuf.get(operand_str)
            if src_shape is not None and len(src_shape) >= 2:
                shape_str = ", ".join(str(d) for d in src_shape)
                new_var = state.device_function.new_var("nki_reciprocal", dce=True)
                state.device_function._nki_sbuf_shapes[new_var] = list(src_shape)
                state.add_statement(
                    statement_from_string(
                        f"{new_var} = nl.ndarray([{shape_str}], nl.float32, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.activation(dst={new_var}, op=nl.reciprocal, data={operand_str})"
                    )
                )
                _refine_reciprocal(new_var, operand_str, list(src_shape))
                return new_var

        state.add_statement(
            statement_from_string(
                f"nisa.activation(dst={operand_str}, op=nl.reciprocal, data={operand_str})"
            )
        )
        return operand

    @staticmethod
    def _nki_activation(a: object, op: str, prefix: str) -> str:
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

        # PSUM-reuse fusion: rewrite input to PSUM name if upstream matmul
        # had its final copy elided. nisa.activation runs on Scalar Engine
        # which accepts PSUM input. Since the activation output must go to
        # SBUF (store codegen expects SBUF), we force a new output buffer.
        _orig_a_str = str(a) if not isinstance(a, (int, float)) else None
        a = NKIOpOverrides._resolve_psum_alias(state, a)
        _psum_aliased = _orig_a_str is not None and str(a) != _orig_a_str

        dst = str(a)
        dst_tile_vars = state.device_function.get_tile_list_vars(dst)

        # Check if the input FX node has other users or if the dst is a
        # second-level copy var (read-only outer-scope capture). In-place
        # modification would corrupt values needed by later ops or iterations.
        from torch._inductor.virtualized import V
        cur_node = V.current_node
        _is_second_level_copy = "_copy_" in dst and dst[-1:].isdigit()
        # Also check protected vars (loop-carry new-value buffers), but only inside nested loops
        _in_nested_loop_act = len(getattr(state.codegen, "active_device_loops", {})) >= 2
        _is_protected_act = (dst in getattr(state.device_function, "_nki_protected_vars", set())
                             and _in_nested_loop_act)
        need_new_buffer = _is_second_level_copy or _is_protected_act or _psum_aliased
        if not need_new_buffer and cur_node is not None and len(cur_node.args) >= 1:
            input_node = cur_node.args[0]
            if hasattr(input_node, "users") and len(input_node.users) > 1:
                need_new_buffer = True

        if need_new_buffer and dst_tile_vars is None:
            shape_list = state.device_function._nki_sbuf_shapes.get(dst)
            out_val = cur_node.meta.get("val") if cur_node else None
            # For copy vars, strip suffixes to find the original var's shape
            if shape_list is None and _is_second_level_copy:
                _lookup = dst
                _sbuf = state.device_function._nki_sbuf_shapes
                while _lookup not in _sbuf and "_copy" in _lookup:
                    idx = _lookup.rfind("_copy")
                    _lookup = _lookup[:idx]
                src_shape = _sbuf.get(_lookup)
                if src_shape is not None and len(src_shape) >= 2:
                    shape_list = list(src_shape)
                    if out_val is None:
                        out_val = torch.empty(1)
            # Derive shape from FX output tensor
            if shape_list is None and out_val is not None and isinstance(out_val, torch.Tensor) and out_val.numel() > 1:
                import sympy as _sp
                _subs: dict[_sp.Symbol, int] = {}
                for _bs in env.block_sizes:
                    _c = _bs.from_config(state.config) if hasattr(state, 'config') and state.config else None
                    if isinstance(_c, int):
                        _subs[_bs.symbol()] = _c
                _resolved = []
                for _d in out_val.shape:
                    if isinstance(_d, torch.SymInt):
                        try:
                            _resolved.append(int(_d._sympy_().subs(_subs)))
                        except (TypeError, ValueError):
                            _resolved.append(int(env.size_hint(_d)))
                    else:
                        _resolved.append(int(_d))
                if len(_resolved) >= 2:
                    shape_list = _resolved
            # Squeeze 3D+ shapes to 2D for NKI SBUF
            if shape_list is not None:
                shape_list = NKIOpOverrides._squeeze_shape_2d(shape_list)
            if shape_list is not None and out_val is not None:
                if hasattr(out_val, 'dtype') and out_val.numel() > 1:
                    dtype_str = env.backend.dtype_str(out_val.dtype)
                elif cur_node is not None and isinstance(cur_node.meta.get("val"), torch.Tensor):
                    dtype_str = env.backend.dtype_str(cur_node.meta["val"].dtype)
                else:
                    dtype_str = "nl.float32"
                shape_str = ", ".join(str(d) for d in shape_list)
                new_dst = state.device_function.new_var(prefix, dce=True)
                state.device_function._nki_sbuf_shapes[new_dst] = list(shape_list)
                state.add_statement(
                    statement_from_string(
                        f"{new_dst} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.activation(dst={new_dst}, op={op}, data={dst})"
                    )
                )
                return new_dst

        if dst_tile_vars is not None:
            for dv in dst_tile_vars:
                state.add_statement(
                    statement_from_string(
                        f"nisa.activation(dst={dv}, op={op}, data={dv})"
                    )
                )
        else:
            state.add_statement(
                statement_from_string(
                    f"nisa.activation(dst={dst}, op={op}, data={dst})"
                )
            )
        return dst

    def add(self, a: object, b: object) -> str:
        return self._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.add",
            op_tensor_scalar="nl.add",
            allow_tensor_tensor=True,
        )

    def sub(self, a: object, b: object) -> str:
        # [M,N] - [M,1] broadcast: use tensor_scalar (NKI tensor_tensor fails on shape mismatch)
        if (
            not self._is_scalar_operand(a)
            and not self._is_scalar_operand(b)
            and self._subtract_tensor_tensor_supported()
        ):
            return self._nki_tensor_scalar(a, b, "nl.subtract")
        return self._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.subtract",
            op_tensor_scalar="nl.subtract",
            allow_tensor_tensor=True,
        )

    def mul(self, a: object, b: object) -> str:
        if (
            not self._is_scalar_operand(a)
            and not self._is_scalar_operand(b)
            and self._truediv_tensor_tensor_supported()
        ):
            return self._nki_tensor_scalar(a, b, "nl.multiply")
        return self._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.multiply",
            op_tensor_scalar="nl.multiply",
            allow_tensor_tensor=True,
        )

    def maximum(self, a: object, b: object) -> str:
        return self._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.maximum",
            op_tensor_scalar="nl.maximum",
            allow_tensor_tensor=True,
        )

    def minimum(self, a: object, b: object) -> str:
        return self._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.minimum",
            op_tensor_scalar="nl.minimum",
            allow_tensor_tensor=True,
        )

    def neg(self, a: object) -> str:
        """neg(x) = -x via tensor_scalar multiply by -1."""
        return self._nki_tensor_scalar(a, -1.0, "nl.multiply")

    def where(self, mask: object, a: object, b: object) -> str:
        """NKI ``torch.where`` lowering.

        Uses ``nisa.tensor_copy_predicated`` (Trn2+) to conditionally select
        between the two operands. ``mask`` is treated as a predicate tile:
        where mask is truthy (non-zero), pick ``a``; else pick ``b``.

        Strategy:

        1. Allocate an output SBUF tile in the consumer's expected dtype.
        2. Initialize it from ``b`` (the "false" branch) via a normal copy.
        3. Overwrite selected positions with ``a`` using
           ``nisa.tensor_copy_predicated(src=a, dst=out, predicate=mask)``.

        Falls back to ``BackendUnsupported`` on trn1, where
        ``tensor_copy_predicated`` is not available. The scalar cases (b
        constant or a constant) use ``memset``/``nisa.tensor_scalar`` to
        materialize the scalar in the output buffer before the predicated
        overwrite.
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment
        from torch._inductor.virtualized import V

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            raise exc.BackendUnsupported(env.backend.name, "NKI-only where")
        state = getattr(env, "_codegen_state", None)
        if state is None:
            raise exc.BackendUnsupported(
                "nki", "where requires active codegen state"
            )

        # Resolve PSUM aliases on tensor operands so upstream PSUM results
        # are read directly.
        a = NKIOpOverrides._resolve_psum_alias(state, a)
        b = NKIOpOverrides._resolve_psum_alias(state, b)
        mask = NKIOpOverrides._resolve_psum_alias(state, mask)

        # Determine output shape/dtype from the current FX node.
        cur_node = V.current_node
        out_val = cur_node.meta.get("val") if cur_node is not None else None
        if not isinstance(out_val, torch.Tensor):
            raise exc.BackendUnsupported(
                "nki", "where: output shape unknown (no FX val metadata)"
            )

        import sympy as _sp_w
        _bs_subs_w: dict[_sp_w.Symbol, int] = {}
        if state.config is not None:
            for _bs in env.block_sizes:
                _cfg = _bs.from_config(state.config)
                if isinstance(_cfg, int):
                    _bs_subs_w[_bs.symbol()] = _cfg

        def _resolve_dim(d: object) -> int:
            if isinstance(d, int):
                return d
            if isinstance(d, torch.SymInt):
                try:
                    return int(d._sympy_().subs(_bs_subs_w))
                except (TypeError, ValueError):
                    return env.size_hint(d)
            return int(d)

        resolved_shape = [_resolve_dim(d) for d in out_val.shape]
        out_shape = NKIOpOverrides._squeeze_shape_2d(resolved_shape)
        if len(out_shape) < 2:
            out_shape = [1] + list(out_shape) if len(out_shape) == 1 else [1, 1]
        out_dtype_str = env.backend.dtype_str(out_val.dtype)

        def _emit_str(x: object) -> str:
            if isinstance(x, ast.AST):
                return ast.unparse(x)
            return str(x)

        mask_str = _emit_str(mask)
        a_str = _emit_str(a)
        b_str = _emit_str(b)

        # Allocate output buffer.
        dst_var = state.device_function.new_var("_nki_where_out", dce=True)
        state.device_function._nki_sbuf_shapes[dst_var] = list(out_shape)
        state.device_function._nki_sbuf_dtypes[dst_var] = out_dtype_str
        state.add_statement(
            statement_from_string(
                f"{dst_var} = nl.ndarray([{out_shape[0]}, {out_shape[1]}], "
                f"{out_dtype_str}, buffer=nl.sbuf)"
            )
        )

        def _try_as_scalar(name: str) -> tuple[bool, object]:
            """Return (True, value) if ``name`` resolves to a host scalar.

            Handles literal numbers ("0.0"), kernel-parameter scalars
            (``_nki_scalar_arg_names``), and lifted constant vars
            (``v_N = 0.0``).
            """
            try:
                return True, float(name)
            except (TypeError, ValueError):
                pass
            scalar_args = getattr(state.device_function, "_nki_scalar_arg_names", set())
            if name in scalar_args:
                return True, name  # emit name as-is
            cg = getattr(state, "codegen", None)
            if cg is not None and hasattr(cg, "get_var_constant_value"):
                val = cg.get_var_constant_value(name)
                if val is not None:
                    return True, val
            return False, None

        def _scalar_literal(value: object) -> object:
            if isinstance(value, float):
                if value == float("inf"):
                    return "float('inf')"
                if value == float("-inf"):
                    return "float('-inf')"
            return value

        is_scalar_b, b_val = _try_as_scalar(b_str)

        # Materialize the "false" branch into dst first.
        if is_scalar_b:
            if isinstance(b_val, float) and b_val == 0.0 and env.backend.dtype_str(out_val.dtype) != "nl.int32":
                state.add_statement(
                    statement_from_string(f"nisa.memset({dst_var}, value=0)")
                )
            else:
                state.add_statement(
                    statement_from_string(
                        f"nisa.memset({dst_var}, value={_scalar_literal(b_val)})"
                    )
                )
        else:
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy(dst={dst_var}, src={b_str})"
                )
            )

        # tensor_copy_predicated requires predicate dtype uint8/16/32.
        # The masks produced by comparison ops (nl.less/greater/etc.) are
        # int32 by default; cast to uint32 via a bitcast-style tensor_copy
        # into a uint32 SBUF tile so we don't lose the bit pattern.
        mask_shape = state.device_function._nki_sbuf_shapes.get(mask_str)
        if mask_shape is None:
            # Try to derive from FX meta
            mask_shape = list(out_shape)
        mask_src_var = mask_str
        if list(mask_shape) != list(out_shape):
            mask_src_var = state.device_function.new_var("_nki_pred_bcast", dce=True)
            mask_dtype = state.device_function._nki_sbuf_dtypes.get(
                mask_str, "nl.int32"
            )
            state.device_function._nki_sbuf_shapes[mask_src_var] = list(out_shape)
            state.device_function._nki_sbuf_dtypes[mask_src_var] = mask_dtype
            state.add_statement(
                statement_from_string(
                    f"{mask_src_var} = nl.broadcast_to({mask_str}, "
                    f"shape=({out_shape[0]}, {out_shape[1]}))"
                )
            )
            mask_shape = list(out_shape)
        mask_cast_var = state.device_function.new_var("_nki_pred_u32", dce=True)
        state.device_function._nki_sbuf_shapes[mask_cast_var] = list(mask_shape)
        state.device_function._nki_sbuf_dtypes[mask_cast_var] = "nl.uint32"
        state.add_statement(
            statement_from_string(
                f"{mask_cast_var} = nl.ndarray([{mask_shape[0]}, {mask_shape[1]}], "
                f"nl.uint32, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_copy(dst={mask_cast_var}, src={mask_src_var})"
            )
        )
        pred_var = mask_cast_var

        # Now predicated-copy the "true" branch into dst wherever mask is set.
        a_is_scalar, a_val = _try_as_scalar(a_str)

        if a_is_scalar:
            # tensor_copy_predicated supports a scalar src starting Trn2.
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy_predicated(dst={dst_var}, "
                    f"src={_scalar_literal(a_val)}, predicate={pred_var})"
                )
            )
        else:
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy_predicated(dst={dst_var}, "
                    f"src={a_str}, predicate={pred_var})"
                )
            )

        return dst_var

    def constant(self, value: int | float | bool, dtype: torch.dtype) -> str:
        """Return string repr for _unpack_opsvalue; _is_scalar_operand treats numeric strings as scalars."""
        from torch._inductor.codegen.simd import constant_repr

        return constant_repr(value)

    def truediv(self, a: object, b: object) -> str:
        a_is_scalar = self._is_scalar_operand(a)
        b_is_scalar = self._is_scalar_operand(b)

        if a_is_scalar and b_is_scalar:
            raise exc.BackendUnsupported(
                "nki",
                "truediv with two host scalars is unsupported in NKI codegen",
            )
        if not a_is_scalar and b_is_scalar:
            recip = self._nki_reciprocal_operand(b)
            return self._nki_tensor_scalar(a, recip, "nl.multiply")
        if a_is_scalar and not b_is_scalar:
            recip = self._nki_reciprocal_operand(b)
            return self._nki_tensor_scalar(recip, a, "nl.multiply")
        if self._truediv_tensor_tensor_supported():
            recip = self._nki_reciprocal_operand(b)
            return self._nki_tensor_scalar(a, recip, "nl.multiply")
        if self._is_scalar_like_tensor(b):
            recip = self._nki_reciprocal_operand(b)
            return self._nki_tensor_scalar(a, recip, "nl.multiply")
        if self._same_shape_tensor_tensor():
            recip = self._nki_reciprocal_operand(b)
            return self._nki_tensor_tensor(a, recip, "nl.multiply", "nki_div")
        if self._truediv_tensor_tensor_broadcast_supported():
            recip = self._nki_reciprocal_operand(b)
            return self._nki_tensor_tensor(a, recip, "nl.multiply", "nki_div")
        recip = self._nki_reciprocal_operand(b)
        return self._nki_tensor_tensor(a, recip, "nl.multiply", "nki_div")

    def div(self, a: object, b: object) -> str:
        return self.truediv(a, b)

    def relu(self, x: object) -> str:
        return self._nki_activation(x, "nl.relu", "nki_relu")

    def sigmoid(self, x: object) -> str:
        return self._nki_activation(x, "nl.sigmoid", "nki_sigmoid")

    def tanh(self, x: object) -> str:
        return self._nki_activation(x, "nl.tanh", "nki_tanh")

    def silu(self, x: object) -> str:
        return self._nki_activation(x, "nl.silu", "nki_silu")

    def silu_dx(self, x: object) -> str:
        return self._nki_activation(x, "nl.silu_dx", "nki_silu_dx")

    def gelu(self, x: object) -> str:
        return self._nki_activation(x, "nl.gelu", "nki_gelu")

    def gelu_dx(self, x: object) -> str:
        return self._nki_activation(x, "nl.gelu_dx", "nki_gelu_dx")

    def gelu_apprx_tanh(self, x: object) -> str:
        return self._nki_activation(x, "nl.gelu_apprx_tanh", "nki_gelu_apprx_tanh")

    def gelu_apprx_sigmoid(self, x: object) -> str:
        return self._nki_activation(
            x, "nl.gelu_apprx_sigmoid", "nki_gelu_apprx_sigmoid"
        )

    def gelu_apprx_sigmoid_dx(self, x: object) -> str:
        return self._nki_activation(
            x, "nl.gelu_apprx_sigmoid_dx", "nki_gelu_apprx_sigmoid_dx"
        )

    def softplus(self, x: object) -> str:
        return self._nki_activation(x, "nl.softplus", "nki_softplus")

    def mish(self, x: object) -> str:
        return self._nki_activation(x, "nl.mish", "nki_mish")

    def erf(self, x: object) -> str:
        return self._nki_activation(x, "nl.erf", "nki_erf")

    def exp(self, x: object) -> str:
        return self._nki_activation(x, "nl.exp", "nki_exp")

    def exp2(self, x: object) -> str:
        """exp2(x) = 2^x = exp(x * ln(2))."""
        import math
        scaled = self._nki_tensor_scalar(x, repr(math.log(2)), "nl.multiply")
        return self._nki_activation(scaled, "nl.exp", "nki_exp2")

    def log(self, x: object) -> str:
        return self._nki_activation(x, "nl.log", "nki_log")

    def sin(self, x: object) -> str:
        return self._nki_activation(x, "nl.sin", "nki_sin")

    def arctan(self, x: object) -> str:
        return self._nki_activation(x, "nl.arctan", "nki_arctan")

    def sqrt(self, x: object) -> str:
        return self._nki_activation(x, "nl.sqrt", "nki_sqrt")

    def rsqrt(self, x: object) -> str:
        return self._nki_activation(x, "nl.rsqrt", "nki_rsqrt")

    def reciprocal(self, x: object) -> str:
        return self._nki_activation(x, "nl.reciprocal", "nki_reciprocal")

    def sign(self, x: object) -> str:
        return self._nki_activation(x, "nl.sign", "nki_sign")

    def abs(self, x: object) -> str:
        return self._nki_activation(x, "nl.abs", "nki_abs")

    def square(self, x: object) -> str:
        return self._nki_activation(x, "nl.square", "nki_square")

    def copy(self, x: object) -> str:
        return self._nki_activation(x, "nl.copy", "nki_copy")

    # -------------------------------------------------------------------------
    # Fused ISA operations (Trn2/Trn3)
    # -------------------------------------------------------------------------

    @staticmethod
    def _nki_alloc_sbuf(
        shape_list: list[int],
        dtype_str: str,
        prefix: str,
        state: object,
    ) -> str:
        """Allocate a new SBUF buffer, register its shape, return var name."""
        from .ast_extension import statement_from_string

        var = state.device_function.new_var(prefix, dce=True)
        shape_str = ", ".join(str(d) for d in shape_list)
        state.device_function._nki_sbuf_shapes[var] = list(shape_list)
        state.add_statement(
            statement_from_string(
                f"{var} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        return var

    @staticmethod
    def _nki_tensor_scalar_reduce(
        data: object,
        op0: str,
        operand0: object,
        reduce_op: str,
        reduce_shape: list[int],
        *,
        reverse0: bool = False,
    ) -> tuple[str, str]:
        """Emit nisa.tensor_scalar_reduce and return (dst_var, reduce_res_var).

        Trn2/Trn3 fused instruction: computes ``result = data <op0> operand0``
        and simultaneously reduces result along the free axis into reduce_res.

        Returns the pair (dst_var, reduce_res_var) so the caller can use both
        the element-wise result and the reduction result.
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ("", "")
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ("", "")

        data_str = str(data)
        data_shape = state.device_function._nki_sbuf_shapes.get(data_str)

        cur_node = None
        try:
            from torch._inductor.virtualized import V
            _cn = V.current_node
            if hasattr(_cn, "meta"):
                cur_node = _cn
        except Exception:
            pass

        if data_shape is None and cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                data_shape = NKIOpOverrides._squeeze_shape_2d(list(val.shape))

        dtype_str = "nl.float32"
        if cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtype_str = env.backend.dtype_str(val.dtype)

        # Allocate output tile for the element-wise result
        dst_shape = data_shape if data_shape is not None else [1, 1]
        dst_var = NKIOpOverrides._nki_alloc_sbuf(dst_shape, dtype_str, "nki_ts_reduce_dst", state)

        # Allocate reduce_res tile [P, 1]
        reduce_var = NKIOpOverrides._nki_alloc_sbuf(reduce_shape, "nl.float32", "nki_ts_reduce_res", state)

        reverse_part = ", reverse0=True" if reverse0 else ""
        operand_emit = operand0
        if isinstance(operand0, str):
            try:
                operand_emit = float(operand0) if "." in operand0 or "e" in operand0.lower() else int(operand0)
            except (ValueError, TypeError):
                pass

        state.add_statement(
            statement_from_string(
                f"nisa.tensor_scalar_reduce(dst={dst_var}, data={data_str}, "
                f"op0={op0}, operand0={operand_emit}, "
                f"reduce_op={reduce_op}, reduce_res={reduce_var}{reverse_part})"
            )
        )
        return (dst_var, reduce_var)

    @staticmethod
    def _nki_activation_reduce(
        data: object,
        op: str,
        reduce_op: str,
        reduce_shape: list[int],
        *,
        bias: str | None = None,
        scale: float | str | None = None,
    ) -> tuple[str, str]:
        """Emit nisa.activation_reduce and return (dst_var, reduce_res_var).

        Trn2/Trn3 fused instruction: applies ``op`` activation to data (with
        optional scale/bias) and simultaneously reduces the result along the
        free axis into reduce_res.

        Returns the pair (dst_var, reduce_res_var).
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ("", "")
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ("", "")

        data_str = str(data)
        data_shape = state.device_function._nki_sbuf_shapes.get(data_str)

        cur_node = None
        try:
            from torch._inductor.virtualized import V
            _cn = V.current_node
            if hasattr(_cn, "meta"):
                cur_node = _cn
        except Exception:
            pass

        if data_shape is None and cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                data_shape = NKIOpOverrides._squeeze_shape_2d(list(val.shape))

        dtype_str = "nl.float32"
        if cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtype_str = env.backend.dtype_str(val.dtype)

        dst_shape = data_shape if data_shape is not None else [1, 1]
        dst_var = NKIOpOverrides._nki_alloc_sbuf(dst_shape, dtype_str, "nki_act_reduce_dst", state)
        reduce_var = NKIOpOverrides._nki_alloc_sbuf(reduce_shape, "nl.float32", "nki_act_reduce_res", state)

        extra_parts = []
        if bias is not None:
            extra_parts.append(f"bias={bias}")
        if scale is not None:
            extra_parts.append(f"scale={scale!r}" if isinstance(scale, float) else f"scale={scale}")
        extra_str = (", " + ", ".join(extra_parts)) if extra_parts else ""

        state.add_statement(
            statement_from_string(
                f"nisa.activation_reduce(dst={dst_var}, op={op}, data={data_str}, "
                f"reduce_op={reduce_op}, reduce_res={reduce_var}{extra_str})"
            )
        )
        return (dst_var, reduce_var)

    @staticmethod
    def _nki_scalar_tensor_tensor(
        data: object,
        op0: str,
        operand0: object,
        op1: str,
        operand1: object,
        *,
        reverse0: bool = False,
        reverse1: bool = False,
    ) -> str:
        """Emit nisa.scalar_tensor_tensor and return the dst var name.

        Trn2/Trn3 fused instruction: ``(data <op0> operand0) <op1> operand1``
        where operand0 is a scalar/[P,1] tile and operand1 is a full tile.
        Saves one SBUF round-trip versus chaining tensor_scalar + tensor_tensor.
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

        data_str = str(data)
        operand1_str = str(operand1)
        data_shape = state.device_function._nki_sbuf_shapes.get(data_str)

        cur_node = None
        try:
            from torch._inductor.virtualized import V
            _cn = V.current_node
            if hasattr(_cn, "meta"):
                cur_node = _cn
        except Exception:
            pass

        if data_shape is None and cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                data_shape = NKIOpOverrides._squeeze_shape_2d(list(val.shape))

        dtype_str = "nl.float32"
        if cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtype_str = env.backend.dtype_str(val.dtype)

        dst_shape = data_shape if data_shape is not None else [1, 1]
        dst_var = NKIOpOverrides._nki_alloc_sbuf(dst_shape, dtype_str, "nki_stt", state)

        operand0_emit = operand0
        if isinstance(operand0, str):
            try:
                operand0_emit = float(operand0) if "." in operand0 or "e" in operand0.lower() else int(operand0)
            except (ValueError, TypeError):
                pass

        extra = []
        if reverse0:
            extra.append("reverse0=True")
        if reverse1:
            extra.append("reverse1=True")
        extra_str = (", " + ", ".join(extra)) if extra else ""

        state.add_statement(
            statement_from_string(
                f"nisa.scalar_tensor_tensor(dst={dst_var}, data={data_str}, "
                f"op0={op0}, operand0={operand0_emit}, "
                f"op1={op1}, operand1={operand1_str}{extra_str})"
            )
        )
        return dst_var

    @staticmethod
    def _nki_tensor_tensor_scan(
        data0: object,
        data1: object,
        initial: object,
        op0: str,
        op1: str,
        *,
        reverse0: bool = False,
        reverse1: bool = False,
    ) -> str:
        """Emit nisa.tensor_tensor_scan and return the dst var name.

        Trn2/Trn3 fused scan instruction: computes a sequential scan across
        the free axis where each output element depends on its predecessor.
        Classic use-case: prefix-sum (op0=nl.add, op1=nl.add, initial=0).
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

        data0_str = str(data0)
        data1_str = str(data1)
        data_shape = state.device_function._nki_sbuf_shapes.get(data0_str)

        cur_node = None
        try:
            from torch._inductor.virtualized import V
            _cn = V.current_node
            if hasattr(_cn, "meta"):
                cur_node = _cn
        except Exception:
            pass

        if data_shape is None and cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                data_shape = NKIOpOverrides._squeeze_shape_2d(list(val.shape))

        dtype_str = "nl.float32"
        if cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtype_str = env.backend.dtype_str(val.dtype)

        dst_shape = data_shape if data_shape is not None else [1, 1]
        dst_var = NKIOpOverrides._nki_alloc_sbuf(dst_shape, dtype_str, "nki_scan", state)

        # initial can be a scalar constant or a tile name
        initial_emit = initial
        if isinstance(initial, str):
            try:
                initial_emit = float(initial) if "." in initial or "e" in initial.lower() else int(initial)
            except (ValueError, TypeError):
                pass

        extra = []
        if reverse0:
            extra.append("reverse0=True")
        if reverse1:
            extra.append("reverse1=True")
        extra_str = (", " + ", ".join(extra)) if extra else ""

        state.add_statement(
            statement_from_string(
                f"nisa.tensor_tensor_scan(dst={dst_var}, data0={data0_str}, data1={data1_str}, "
                f"initial={initial_emit}, op0={op0}, op1={op1}{extra_str})"
            )
        )
        return dst_var

    @staticmethod
    def _nki_tensor_scalar_cumulative(
        src: object,
        op0: str,
        op1: str,
        imm0: float | str,
        *,
        imm1: float | str | None = None,
        reduce_cmd: str = "nisa.reduce_cmd.reset_reduce",
    ) -> str:
        """Emit nisa.tensor_scalar_cumulative and return the dst var name.

        Trn2/Trn3 cumulative instruction: applies ``op0`` with scalar ``imm0``
        to each element of src, then performs cumulative ``op1`` into dst.
        Example: cumulative sum-of-squares via op0=nl.multiply (x*x) + op1=nl.add.
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

        src_str = str(src)
        data_shape = state.device_function._nki_sbuf_shapes.get(src_str)

        cur_node = None
        try:
            from torch._inductor.virtualized import V
            _cn = V.current_node
            if hasattr(_cn, "meta"):
                cur_node = _cn
        except Exception:
            pass

        if data_shape is None and cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                data_shape = NKIOpOverrides._squeeze_shape_2d(list(val.shape))

        dtype_str = "nl.float32"
        if cur_node is not None:
            val = cur_node.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtype_str = env.backend.dtype_str(val.dtype)

        dst_shape = data_shape if data_shape is not None else [1, 1]
        dst_var = NKIOpOverrides._nki_alloc_sbuf(dst_shape, dtype_str, "nki_tsc", state)

        imm0_emit = imm0 if isinstance(imm0, (int, float)) else repr(imm0)
        extra = [f"reduce_cmd={reduce_cmd}"]
        if imm1 is not None:
            imm1_emit = imm1 if isinstance(imm1, (int, float)) else repr(imm1)
            extra.insert(0, f"imm1={imm1_emit}")
        extra_str = ", " + ", ".join(extra)

        state.add_statement(
            statement_from_string(
                f"nisa.tensor_scalar_cumulative(dst={dst_var}, src={src_str}, "
                f"op0={op0}, op1={op1}, imm0={imm0_emit}{extra_str})"
            )
        )
        return dst_var

    # Index / scalar arithmetic — these operate on Python integers (tile offsets,
    # loop counters), not on SBUF tiles, so plain Python operators are correct
    # for the scalar/scalar case.  NKI does not provide tile-valued floordiv /
    # mod, so we keep Python semantics; tile-tile forms are rare in practice.
    @staticmethod
    def floordiv(a: object, b: object) -> str:
        # When ``a`` is an SBUF tile and ``b`` is a scalar constant,
        # emit (a * (1.0/b)) then floor, producing the integer floor div.
        # The result is an int32 tile. Uses two nisa.tensor_scalar calls
        # plus one nisa.activation(floor).
        a_scalar = NKIOpOverrides._is_scalar_operand(a)
        b_resolved = NKIOpOverrides._resolve_scalar_operand(b)
        from .compile_environment import CompileEnvironment

        env_for_size = CompileEnvironment.current()
        b_val: float | None = None
        try:
            b_val = float(b_resolved)
        except (TypeError, ValueError):
            try:
                b_val = float(env_for_size.size_hint(b_resolved))
            except Exception:
                b_val = None
        b_scalar = NKIOpOverrides._is_scalar_operand(b)
        if not b_scalar and b_val is not None:
            b_scalar = True
        if not b_scalar and isinstance(b_resolved, str):
            state_for_scalar = getattr(env_for_size, "_codegen_state", None)
            if state_for_scalar is not None:
                dev_fn = state_for_scalar.device_function
                is_tile_operand = (
                    b_resolved in dev_fn._nki_sbuf_shapes
                    or b_resolved in dev_fn._nki_tile_lists
                )
                if not is_tile_operand:
                    b_scalar = True
        if a_scalar and b_scalar:
            return f"{a} // {b}"
        if not a_scalar and b_scalar:
            # Tile // scalar: multiply by 1/scalar, floor, cast to int.
            from .ast_extension import statement_from_string
            env = env_for_size
            state = getattr(env, "_codegen_state", None)
            if state is None:
                return f"{a} // {b}"
            if b_val is None and not isinstance(b_resolved, str):
                return f"{a} // {b}"
            if b_val == 0.0:
                return f"{a} // {b}"
            a_str = str(NKIOpOverrides._resolve_scalar_operand(a))
            src_shape = state.device_function._nki_sbuf_shapes.get(a_str)
            if src_shape is None:
                # indices_N from loop_index_statements aren't registered
                # in _nki_sbuf_shapes. Try FX node's val to derive shape,
                # or assume a [1, block_size] iota layout.
                try:
                    from torch._inductor.virtualized import V as _V_fd
                    _cn = _V_fd.current_node
                    if _cn is not None:
                        val = _cn.meta.get("val")
                        if isinstance(val, torch.Tensor):
                            src_shape = NKIOpOverrides._squeeze_shape_2d(
                                [env.size_hint(d) if isinstance(d, torch.SymInt) else int(d) for d in val.shape]
                            )
                            if len(src_shape) == 1:
                                src_shape = [1, src_shape[0]]
                except Exception:
                    pass
            if src_shape is None or a_str.startswith("indices_"):
                block_size = state.device_function._nki_iota_block_sizes.get(a_str)
                if block_size is not None:
                    try:
                        src_shape = [1, int(block_size)]
                    except ValueError:
                        pass
            if src_shape is None or a_str.startswith("indices_"):
                try:
                    for block_id, loops in state.codegen.active_device_loops.items():
                        if not loops:
                            continue
                        strategy = loops[-1].strategy
                        if strategy.index_var(block_id) != a_str:
                            continue
                        block_size = env.block_sizes[block_id].from_config_assert(
                            state.config
                        )
                        src_shape = [1, int(block_size)]
                        break
                except Exception:
                    pass
            if src_shape is None or len(src_shape) < 2:
                return f"{a} // {b}"
            tmp_var = state.device_function.new_var("_nki_floordiv_f32", dce=True)
            out_var = state.device_function.new_var("_nki_floordiv_i32", dce=True)
            state.device_function._nki_sbuf_shapes[tmp_var] = list(src_shape)
            state.device_function._nki_sbuf_shapes[out_var] = list(src_shape)
            state.device_function._nki_sbuf_dtypes[tmp_var] = "nl.float32"
            state.device_function._nki_sbuf_dtypes[out_var] = "nl.int32"
            inv = 1.0 / b_val if b_val is not None else f"1.0 / ({b_resolved})"
            state.add_statement(
                statement_from_string(
                    f"{tmp_var} = nl.ndarray([{src_shape[0]}, {src_shape[1]}], nl.float32, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_scalar(dst={tmp_var}, data={a_str}, "
                    f"op0=nl.multiply, operand0={inv})"
                )
            )
            state.add_statement(
                statement_from_string(f"{out_var} = nl.floor({tmp_var}, dtype=nl.int32)")
            )
            return out_var
        return f"{a} // {b}"

    @staticmethod
    def mod(a: object, b: object) -> str:
        a_scalar = NKIOpOverrides._is_scalar_operand(a)
        b_resolved = NKIOpOverrides._resolve_scalar_operand(b)
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        b_val: float | None = None
        try:
            b_val = float(b_resolved)
        except (TypeError, ValueError):
            try:
                b_val = float(env.size_hint(b_resolved))
            except Exception:
                b_val = None
        b_scalar = NKIOpOverrides._is_scalar_operand(b) or b_val is not None
        state = getattr(env, "_codegen_state", None)
        if not b_scalar and isinstance(b_resolved, str) and state is not None:
            dev_fn = state.device_function
            is_tile_operand = (
                b_resolved in dev_fn._nki_sbuf_shapes
                or b_resolved in dev_fn._nki_tile_lists
            )
            if not is_tile_operand:
                b_scalar = True
        if not a_scalar and b_scalar:
            from .ast_extension import statement_from_string

            if state is None:
                return f"{a} % {b}"
            a_str = str(NKIOpOverrides._resolve_scalar_operand(a))
            src_shape = state.device_function._nki_sbuf_shapes.get(a_str)
            if src_shape is None or len(src_shape) < 2:
                return f"{a} % {b}"
            q = NKIOpOverrides.floordiv(a, b)
            b_operand: object
            if b_val is not None:
                b_operand = int(b_val) if float(b_val).is_integer() else b_val
            else:
                b_operand = b_resolved
            prod = state.device_function.new_var("_nki_mod_prod", dce=True)
            out = state.device_function.new_var("_nki_mod_i32", dce=True)
            state.device_function._nki_sbuf_shapes[prod] = list(src_shape)
            state.device_function._nki_sbuf_shapes[out] = list(src_shape)
            state.device_function._nki_sbuf_dtypes[prod] = "nl.int32"
            state.device_function._nki_sbuf_dtypes[out] = "nl.int32"
            state.add_statement(
                statement_from_string(
                    f"{prod} = nl.ndarray([{src_shape[0]}, {src_shape[1]}], nl.int32, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_scalar(dst={prod}, data={q}, "
                    f"op0=nl.multiply, operand0={b_operand})"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"{out} = nl.ndarray([{src_shape[0]}, {src_shape[1]}], nl.int32, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={out}, data1={a_str}, "
                    f"data2={prod}, op=nl.subtract)"
                )
            )
            return out
        return f"{a} % {b}"

    @staticmethod
    def remainder(a: object, b: object) -> str:
        return NKIOpOverrides.mod(a, b)

    @staticmethod
    def fmod(a: object, b: object) -> str:
        return NKIOpOverrides.mod(a, b)

    @staticmethod
    def _nki_scalar_arith(a: object, b: object, nl_op: str | None, py_op: str) -> str:
        """Arithmetic on tile-or-scalar operands.

        When both operands are host scalars, fall back to Python ``py_op``.
        When one is a tile and ``nl_op`` is provided, emit tensor_scalar.
        When one is a tile and ``nl_op`` is None (e.g. mod — NKI has no
        direct mod op on tiles), fall back to Python ``py_op``; the caller
        is responsible for ensuring this only runs on tile-index scalars.
        """
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        state = getattr(env, "_codegen_state", None)
        a_scalar = NKIOpOverrides._is_scalar_operand(a)
        b_scalar = NKIOpOverrides._is_scalar_operand(b)
        if a_scalar and b_scalar:
            return f"{a} {py_op} {b}"
        if nl_op is None or state is None:
            return f"{a} {py_op} {b}"
        # Tile-op: reuse the general binary path for consistency.
        if b_scalar and not a_scalar:
            return NKIOpOverrides._nki_tensor_scalar(a, b, nl_op)
        if a_scalar and not b_scalar:
            return NKIOpOverrides._nki_tensor_scalar(b, a, nl_op, reverse0=True)
        return NKIOpOverrides._nki_tensor_tensor(a, b, nl_op, "_nki_scalar_arith")

    @staticmethod
    def lt(a: object, b: object) -> str:
        return NKIOpOverrides._nki_compare(a, b, "nl.less", "<")

    @staticmethod
    def le(a: object, b: object) -> str:
        return NKIOpOverrides._nki_compare(a, b, "nl.less_equal", "<=")

    @staticmethod
    def gt(a: object, b: object) -> str:
        return NKIOpOverrides._nki_compare(a, b, "nl.greater", ">")

    @staticmethod
    def ge(a: object, b: object) -> str:
        return NKIOpOverrides._nki_compare(a, b, "nl.greater_equal", ">=")

    @staticmethod
    def eq(a: object, b: object) -> str:
        return NKIOpOverrides._nki_compare(a, b, "nl.equal", "==")

    @staticmethod
    def ne(a: object, b: object) -> str:
        return NKIOpOverrides._nki_compare(a, b, "nl.not_equal", "!=")

    @staticmethod
    def _nki_compare(a: object, b: object, nl_op: str, py_op: str) -> str:
        """Elementwise comparison on NKI tiles.

        When either operand is an SBUF tile, emits a tensor_tensor /
        tensor_scalar call with a boolean ``op``. When both operands are
        host scalars (Python ints or compile-time constants), falls back
        to the Python operator so downstream codegen can use the result
        as a bare Python value (e.g. in slice bounds).
        """
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        state = getattr(env, "_codegen_state", None)

        a_scalar = NKIOpOverrides._is_scalar_operand(a)
        b_scalar = NKIOpOverrides._is_scalar_operand(b)

        # Both host scalars: emit Python-level comparison.
        if a_scalar and b_scalar:
            return f"{a} {py_op} {b}"

        if state is None:
            return f"{a} {py_op} {b}"

        # Resolve PSUM aliases for tensor operands.
        if not a_scalar:
            a = NKIOpOverrides._resolve_psum_alias(state, a)
        if not b_scalar:
            b = NKIOpOverrides._resolve_psum_alias(state, b)

        def _emit_str(x: object) -> str:
            if isinstance(x, ast.AST):
                return ast.unparse(x)
            return str(x)

        a_str = _emit_str(a)
        b_str = _emit_str(b)

        # Determine output shape — prefer the FX-inferred output shape when
        # available (handles cross-axis broadcasts like [1,N] == [M,1] → [M,N]).
        def _cmp_lookup_shape(name: str) -> list[int] | None:
            s = state.device_function._nki_sbuf_shapes.get(name)
            if s is not None:
                return s
            _lk = name
            while "_copy" in _lk:
                _lk = _lk[:_lk.rfind("_copy")]
                s = state.device_function._nki_sbuf_shapes.get(_lk)
                if s is not None:
                    return s
            return None

        shape = None
        dtype_str = "nl.int32"  # bool-like (0/1) stored as int
        # Prefer FX output shape for cross-axis broadcast (e.g. [1,N]×[M,1])
        try:
            from torch._inductor.virtualized import V
            cur = V.current_node
            val = cur.meta.get("val") if cur is not None else None
            if isinstance(val, torch.Tensor):
                import sympy as _sp_cmp
                _bs_subs_c: dict[_sp_cmp.Symbol, int] = {}
                if state.config is not None:
                    for _bs in env.block_sizes:
                        _cfg = _bs.from_config(state.config)
                        if isinstance(_cfg, int):
                            _bs_subs_c[_bs.symbol()] = _cfg
                _fx_shape = []
                for _d in val.shape:
                    if isinstance(_d, int):
                        _fx_shape.append(_d)
                    elif isinstance(_d, torch.SymInt):
                        try:
                            _fx_shape.append(int(_d._sympy_().subs(_bs_subs_c)))
                        except (TypeError, ValueError):
                            _fx_shape.append(env.size_hint(_d))
                    else:
                        _fx_shape.append(int(_d))
                shape = NKIOpOverrides._squeeze_shape_2d(_fx_shape)
        except Exception:
            pass
        if shape is None:
            # Fallback: prefer the tensor operand's known SBUF shape.
            a_tshape = b_tshape = None
            if not a_scalar:
                a_tshape = _cmp_lookup_shape(_emit_str(a))
            if not b_scalar:
                b_tshape = _cmp_lookup_shape(_emit_str(b))
            if a_tshape is not None and b_tshape is None:
                shape = a_tshape
            elif b_tshape is not None and a_tshape is None:
                shape = b_tshape
            elif a_tshape is not None and b_tshape is not None:
                if len(a_tshape) >= 2 and len(b_tshape) >= 2:
                    shape = [max(a_tshape[0], b_tshape[0]), max(a_tshape[1], b_tshape[1])]
                else:
                    shape = a_tshape
        if shape is None or len(shape) == 0:
            shape = [1, 1]
        elif len(shape) == 1:
            shape = [1, int(shape[0])]

        def _cmp_lookup_dtype(name: str) -> str:
            dtypes = state.device_function._nki_sbuf_dtypes
            dtype = dtypes.get(name)
            if dtype is not None:
                return dtype
            lookup = name
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                dtype = dtypes.get(lookup)
                if dtype is not None:
                    return dtype
            return "nl.float32"

        def _cmp_match_data_layout(data_name: str) -> str:
            data_shape = _cmp_lookup_shape(data_name)
            if (
                data_shape is None
                or len(data_shape) != 2
                or len(shape) != 2
                or data_shape != [1, shape[0]]
                or shape[1] != 1
            ):
                return data_name
            data_dtype = _cmp_lookup_dtype(data_name)
            int_dtypes = {
                "nl.int32",
                "nl.int16",
                "nl.int8",
                "nl.uint32",
                "nl.uint16",
                "nl.uint8",
            }
            transpose_src = data_name
            transpose_dtype = data_dtype
            if data_dtype in int_dtypes:
                cast_in = state.device_function.new_var("_cmp_layout_cast", dce=True)
                state.device_function._nki_sbuf_shapes[cast_in] = [1, shape[0]]
                state.device_function._nki_sbuf_dtypes[cast_in] = "nl.float32"
                state.add_statement(
                    statement_from_string(
                        f"{cast_in} = nl.ndarray([1, {shape[0]}], "
                        "nl.float32, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(f"nisa.tensor_copy(dst={cast_in}, src={data_name})")
                )
                transpose_src = cast_in
                transpose_dtype = "nl.float32"
            tr_psum = state.device_function.new_var("_cmp_layout_tr_psum", dce=True)
            tr_sbuf = state.device_function.new_var("_cmp_layout_tr_sbuf", dce=True)
            state.device_function._nki_sbuf_shapes[tr_sbuf] = [shape[0], 1]
            state.device_function._nki_sbuf_dtypes[tr_sbuf] = data_dtype
            state.add_statement(
                statement_from_string(
                    f"{tr_psum} = nl.ndarray([{shape[0]}, 1], "
                    f"{transpose_dtype}, buffer=nl.psum)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.nc_transpose(dst={tr_psum}, data={transpose_src})"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"{tr_sbuf} = nl.ndarray([{shape[0]}, 1], "
                    f"{data_dtype}, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})")
            )
            return tr_sbuf

        dst_var = state.device_function.new_var("_nki_cmp", dce=True)
        state.device_function._nki_sbuf_shapes[dst_var] = list(shape)
        state.device_function._nki_sbuf_dtypes[dst_var] = dtype_str
        state.add_statement(
            statement_from_string(
                f"{dst_var} = nl.ndarray([{shape[0]}, {shape[1]}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        if a_scalar and not b_scalar:
            # (scalar op tensor)  — use tensor_scalar with reverse
            data_name = _cmp_match_data_layout(b_str)
            scalar_operand = NKIOpOverrides._resolve_scalar_operand(a_str)
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_scalar(dst={dst_var}, data={data_name}, "
                    f"op0={nl_op}, operand0={scalar_operand}, reverse0=True)"
                )
            )
        elif b_scalar and not a_scalar:
            data_name = _cmp_match_data_layout(a_str)
            scalar_operand = NKIOpOverrides._resolve_scalar_operand(b_str)
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_scalar(dst={dst_var}, data={data_name}, "
                    f"op0={nl_op}, operand0={scalar_operand})"
                )
            )
        else:
            # Check if we need cross-axis broadcast to produce output shape.
            # E.g. [1, N] op [1, N] output [N, N] means one operand is
            # partition-axis and the other is free-axis. Emit an SBUF
            # replicate to produce matching-shape operands.
            a_sh = _cmp_lookup_shape(a_str)
            b_sh = _cmp_lookup_shape(b_str)
            # ``indices_*`` names come from nisa.iota without a shape registration;
            # they're always [1, N] row vectors (N == output free dim).
            if a_sh is None and a_str.startswith("indices_") and len(shape) == 2:
                a_sh = [1, shape[1]]
            if b_sh is None and b_str.startswith("indices_") and len(shape) == 2:
                b_sh = [1, shape[1]]
            data1_name = a_str
            data2_name = b_str
            if (
                a_sh is not None
                and b_sh is not None
                and len(a_sh) == 2
                and len(b_sh) == 2
                and len(shape) == 2
                and shape[0] == 1
                and shape[1] > 1
                and a_sh == [1, 1]
                and b_sh == [1, shape[1]]
            ):
                scalar_bcast = state.device_function.new_var(
                    "_cmp_scalar_bcast", dce=True
                )
                scalar_dtype = _cmp_lookup_dtype(a_str)
                state.device_function._nki_sbuf_shapes[scalar_bcast] = [
                    1,
                    shape[1],
                ]
                state.device_function._nki_sbuf_dtypes[scalar_bcast] = scalar_dtype
                state.add_statement(
                    statement_from_string(
                        f"{scalar_bcast} = nl.broadcast_to({a_str}, "
                        f"shape=(1, {shape[1]}))"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={dst_var}, "
                        f"data1={scalar_bcast}, data2={b_str}, op={nl_op})"
                    )
                )
                return dst_var
            if (
                a_sh is not None
                and b_sh is not None
                and len(a_sh) == 2
                and len(b_sh) == 2
                and len(shape) == 2
                and shape[0] == 1
                and shape[1] > 1
                and b_sh == [1, 1]
                and a_sh == [1, shape[1]]
            ):
                scalar_bcast = state.device_function.new_var(
                    "_cmp_scalar_bcast", dce=True
                )
                scalar_dtype = _cmp_lookup_dtype(b_str)
                state.device_function._nki_sbuf_shapes[scalar_bcast] = [
                    1,
                    shape[1],
                ]
                state.device_function._nki_sbuf_dtypes[scalar_bcast] = scalar_dtype
                state.add_statement(
                    statement_from_string(
                        f"{scalar_bcast} = nl.broadcast_to({b_str}, "
                        f"shape=(1, {shape[1]}))"
                    )
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={dst_var}, data1={a_str}, "
                        f"data2={scalar_bcast}, op={nl_op})"
                    )
                )
                return dst_var
            if (
                a_sh is not None
                and b_sh is not None
                and len(a_sh) == 2
                and len(b_sh) == 2
                and len(shape) == 2
                and shape[0] > 1
                and shape[1] == 1
                and a_sh == [1, 1]
                and b_sh == [1, shape[0]]
            ):
                scalar_bcast = state.device_function.new_var(
                    "_cmp_scalar_bcast", dce=True
                )
                scalar_dtype = _cmp_lookup_dtype(a_str)
                state.device_function._nki_sbuf_shapes[scalar_bcast] = [
                    shape[0],
                    1,
                ]
                state.device_function._nki_sbuf_dtypes[scalar_bcast] = scalar_dtype
                state.add_statement(
                    statement_from_string(
                        f"{scalar_bcast} = nl.broadcast_to({a_str}, "
                        f"shape=({shape[0]}, 1))"
                    )
                )
                data_name = _cmp_match_data_layout(b_str)
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={dst_var}, data1={scalar_bcast}, "
                        f"data2={data_name}, op={nl_op})"
                    )
                )
                return dst_var
            if (
                a_sh is not None
                and b_sh is not None
                and len(a_sh) == 2
                and len(b_sh) == 2
                and len(shape) == 2
                and shape[0] > 1
                and shape[1] == 1
                and b_sh == [1, 1]
                and a_sh == [1, shape[0]]
            ):
                scalar_bcast = state.device_function.new_var(
                    "_cmp_scalar_bcast", dce=True
                )
                scalar_dtype = _cmp_lookup_dtype(b_str)
                state.device_function._nki_sbuf_shapes[scalar_bcast] = [
                    shape[0],
                    1,
                ]
                state.device_function._nki_sbuf_dtypes[scalar_bcast] = scalar_dtype
                state.add_statement(
                    statement_from_string(
                        f"{scalar_bcast} = nl.broadcast_to({b_str}, "
                        f"shape=({shape[0]}, 1))"
                    )
                )
                data_name = _cmp_match_data_layout(a_str)
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={dst_var}, data1={data_name}, "
                        f"data2={scalar_bcast}, op={nl_op})"
                    )
                )
                return dst_var
            if (
                a_sh is not None and b_sh is not None
                and len(a_sh) == 2 and len(b_sh) == 2
                and len(shape) == 2
                and shape[0] > 1
                and shape[1] > 1
                and (
                    (
                        list(a_sh) == list(b_sh)
                        and shape[0] != a_sh[0]
                        and shape[1] == a_sh[1]
                        and a_sh[0] == 1
                    )
                    or (
                        a_sh[0] == 1
                        and a_sh[1] == shape[1]
                        and b_sh[0] == 1
                        and b_sh[1] == shape[0]
                    )
                    or (
                        b_sh[0] == 1
                        and b_sh[1] == shape[1]
                        and a_sh[0] == 1
                        and a_sh[1] == shape[0]
                    )
                )
            ):
                # Both [1, N] but output is [P, N]: broadcast one to [P, N]
                # along partition (replicate the row), and transpose the
                # other to [P, 1] then broadcast to [P, N].
                # We choose: a stays [1, N] broadcast → [P, N]
                #           b transposed [1, N] → [N, 1], broadcast [N, N]
                # Actually simpler: replicate a row-wise to [P, N], and
                # transpose-then-broadcast b to [P, N] as well.
                # However the most common case is v_indices × completion_id.
                # v_indices is the same for every partition — replicate.
                # completion_id has one value per partition — needs transpose.
                # Heuristic: if a name starts with "indices_" it's the iota/v_idx row
                # and should be replicated; the other is partition-semantics
                # and should be transposed.
                from .ast_extension import (
                    create as _create,
                    expr_from_string as _efrom,
                )
                def _replicate_row(src: str, p_tgt: int, f_tgt: int, dt: str) -> str:
                    # Use nl.broadcast_to which uses nc_stream_shuffle for
                    # partition-dim broadcast (NKI-friendly).
                    out = state.device_function.new_var("_cmp_bcast", dce=True)
                    state.device_function._nki_sbuf_shapes[out] = [p_tgt, f_tgt]
                    state.device_function._nki_sbuf_dtypes[out] = dt
                    state.add_statement(statement_from_string(
                        f"{out} = nl.broadcast_to({src}, shape=({p_tgt}, {f_tgt}))"
                    ))
                    return out

                def _transpose_to_partition(src: str, p_tgt: int, f_tgt: int, dt: str) -> str:
                    # [1, N] → [N, 1] via transpose, then broadcast to [N, F_tgt]
                    # nc_transpose requires float dtype. For int types, cast
                    # to bfloat16, transpose, cast back.
                    _int_dtypes = {"nl.int32", "nl.int16", "nl.int8", "nl.uint32",
                                   "nl.uint16", "nl.uint8"}
                    if dt in _int_dtypes:
                        # Cast src to float32 in SBUF first
                        cast_in = state.device_function.new_var("_cmp_cast_in", dce=True)
                        state.device_function._nki_sbuf_shapes[cast_in] = [1, p_tgt]
                        state.device_function._nki_sbuf_dtypes[cast_in] = "nl.float32"
                        state.add_statement(statement_from_string(
                            f"{cast_in} = nl.ndarray([1, {p_tgt}], nl.float32, buffer=nl.sbuf)"
                        ))
                        state.add_statement(statement_from_string(
                            f"nisa.memset({cast_in}, value=0)"
                        ))
                        state.add_statement(statement_from_string(
                            f"nisa.tensor_tensor(dst={cast_in}, data1={cast_in}, data2={src}, op=nl.add)"
                        ))
                        # Transpose in float space
                        tr_psum = state.device_function.new_var("_cmp_tr_psum", dce=True)
                        state.add_statement(statement_from_string(
                            f"{tr_psum} = nl.ndarray([{p_tgt}, 1], nl.float32, buffer=nl.psum)"
                        ))
                        state.add_statement(statement_from_string(
                            f"nisa.nc_transpose(dst={tr_psum}, data={cast_in})"
                        ))
                        # Cast back to int in SBUF
                        tr_sbuf = state.device_function.new_var("_cmp_tr_sbuf", dce=True)
                        state.device_function._nki_sbuf_shapes[tr_sbuf] = [p_tgt, 1]
                        state.device_function._nki_sbuf_dtypes[tr_sbuf] = dt
                        state.add_statement(statement_from_string(
                            f"{tr_sbuf} = nl.ndarray([{p_tgt}, 1], {dt}, buffer=nl.sbuf)"
                        ))
                        state.add_statement(statement_from_string(
                            f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
                        ))
                    else:
                        tr_psum = state.device_function.new_var("_cmp_tr_psum", dce=True)
                        tr_sbuf = state.device_function.new_var("_cmp_tr_sbuf", dce=True)
                        state.device_function._nki_sbuf_shapes[tr_sbuf] = [p_tgt, 1]
                        state.device_function._nki_sbuf_dtypes[tr_sbuf] = dt
                        state.add_statement(statement_from_string(
                            f"{tr_psum} = nl.ndarray([{p_tgt}, 1], {dt}, buffer=nl.psum)"
                        ))
                        state.add_statement(statement_from_string(
                            f"nisa.nc_transpose(dst={tr_psum}, data={src})"
                        ))
                        state.add_statement(statement_from_string(
                            f"{tr_sbuf} = nl.ndarray([{p_tgt}, 1], {dt}, buffer=nl.sbuf)"
                        ))
                        state.add_statement(statement_from_string(
                            f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})"
                        ))
                    # Broadcast to [P, F]
                    out = state.device_function.new_var("_cmp_bcast", dce=True)
                    state.device_function._nki_sbuf_shapes[out] = [p_tgt, f_tgt]
                    state.device_function._nki_sbuf_dtypes[out] = dt
                    state.add_statement(statement_from_string(
                        f"{out} = nl.ndarray([{p_tgt}, {f_tgt}], {dt}, buffer=nl.sbuf)"
                    ))
                    f_loop = state.device_function.new_var("_cp_f")
                    state.add_statement(_create(
                        ast.For,
                        target=_create(ast.Name, id=f_loop, ctx=ast.Store()),
                        iter=_efrom(f"nl.affine_range({f_tgt})"),
                        body=[statement_from_string(
                            f"nisa.tensor_copy(dst={out}[0:{p_tgt}, {f_loop}:{f_loop}+1], "
                            f"src={tr_sbuf})"
                        )],
                        orelse=[],
                    ))
                    return out

                # Pick which operand to replicate vs transpose.
                # iota-named or repeat-patterns go row-wise; others go partition-wise.
                a_dt = state.device_function._nki_sbuf_dtypes.get(a_str, "nl.int32")
                b_dt = state.device_function._nki_sbuf_dtypes.get(b_str, "nl.int32")
                if a_str.startswith("indices_"):
                    data1_name = _replicate_row(a_str, shape[0], shape[1], a_dt)
                    data2_name = _transpose_to_partition(b_str, shape[0], shape[1], b_dt)
                elif b_str.startswith("indices_"):
                    data1_name = _transpose_to_partition(a_str, shape[0], shape[1], a_dt)
                    data2_name = _replicate_row(b_str, shape[0], shape[1], b_dt)
                else:
                    # Default: replicate a, transpose b
                    data1_name = _replicate_row(a_str, shape[0], shape[1], a_dt)
                    data2_name = _transpose_to_partition(b_str, shape[0], shape[1], b_dt)
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={dst_var}, data1={data1_name}, data2={data2_name}, op={nl_op})"
                )
            )
        return dst_var

    @classmethod
    def and_(cls, a: object, b: object) -> str:
        if cls._is_scalar_operand(a) and cls._is_scalar_operand(b):
            return f"{a} & {b}"
        if not cls._is_scalar_operand(a) and not cls._is_scalar_operand(b):
            fresh = cls._nki_tensor_tensor_fresh(
                a, b, "nl.bitwise_and", "nki_bitwise"
            )
            if fresh is not None:
                return fresh
        return cls._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.bitwise_and",
            op_tensor_scalar="nl.bitwise_and",
            allow_tensor_tensor=True,
        )

    @classmethod
    def or_(cls, a: object, b: object) -> str:
        if cls._is_scalar_operand(a) and cls._is_scalar_operand(b):
            return f"{a} | {b}"
        if not cls._is_scalar_operand(a) and not cls._is_scalar_operand(b):
            fresh = cls._nki_tensor_tensor_fresh(
                a, b, "nl.bitwise_or", "nki_bitwise"
            )
            if fresh is not None:
                return fresh
        return cls._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.bitwise_or",
            op_tensor_scalar="nl.bitwise_or",
            allow_tensor_tensor=True,
        )

    @classmethod
    def xor(cls, a: object, b: object) -> str:
        if cls._is_scalar_operand(a) and cls._is_scalar_operand(b):
            return f"{a} ^ {b}"
        if not cls._is_scalar_operand(a) and not cls._is_scalar_operand(b):
            fresh = cls._nki_tensor_tensor_fresh(
                a, b, "nl.bitwise_xor", "nki_bitwise"
            )
            if fresh is not None:
                return fresh
        return cls._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.bitwise_xor",
            op_tensor_scalar="nl.bitwise_xor",
            allow_tensor_tensor=True,
        )

    @staticmethod
    def lshift(a: object, b: object) -> str:
        return NKIOpOverrides._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.left_shift",
            op_tensor_scalar="nl.left_shift",
            allow_tensor_tensor=True,
        )

    @staticmethod
    def rshift(a: object, b: object) -> str:
        return NKIOpOverrides._nki_binary_op(
            a,
            b,
            op_tensor_tensor="nl.right_shift",
            op_tensor_scalar="nl.right_shift",
            allow_tensor_tensor=True,
        )

    @staticmethod
    def pow(a: object, b: object) -> str:
        return f"{a} ** {b}"

    # Inductor sometimes queries these alternate names.
    @staticmethod
    def bitwise_left_shift(a: object, b: object) -> str:
        return NKIOpOverrides.lshift(a, b)

    @staticmethod
    def bitwise_right_shift(a: object, b: object) -> str:
        return NKIOpOverrides.rshift(a, b)

    @staticmethod
    def bitwise_or(a: object, b: object) -> str:
        return NKIOpOverrides.or_(a, b)

    @staticmethod
    def bitwise_and(a: object, b: object) -> str:
        return NKIOpOverrides.and_(a, b)

    @staticmethod
    def bitwise_xor(a: object, b: object) -> str:
        return NKIOpOverrides.xor(a, b)


def _validate_nki_tensor_shape(shape: tuple, name: str, env: object) -> None:
    """Check partition dim (0) % 128 and free dim (1) % 512. Raise clear error if not."""
    if not shape:
        return
    # Partition dimension (axis 0) must be a multiple of 128
    s0 = shape[0]
    try:
        v0 = env.size_hint(s0) if isinstance(s0, (int, torch.SymInt)) else int(env.shape_env.size_hint(s0))
    except Exception:
        raise exc.BackendUnsupported(
            "nki",
            f"Tensor {name!r}: partition dimension (axis 0) must be a multiple of 128; "
            "could not resolve shape to an integer.",
        )
    if v0 % 128 != 0:
        raise exc.BackendUnsupported(
            "nki",
            f"Tensor {name!r}: partition dimension (axis 0) has size {v0}, "
            "which is not a multiple of 128. NKI requires exact-multiple tile shapes.",
        )
    # Free dimension (axis 1, if present) must be a multiple of 512
    if len(shape) < 2:
        return
    s1 = shape[1]
    try:
        v1 = env.size_hint(s1) if isinstance(s1, (int, torch.SymInt)) else int(env.shape_env.size_hint(s1))
    except Exception:
        raise exc.BackendUnsupported(
            "nki",
            f"Tensor {name!r}: free dimension (axis 1) must be a multiple of 512; "
            "could not resolve shape to an integer.",
        )
    if v1 % 512 != 0:
        raise exc.BackendUnsupported(
            "nki",
            f"Tensor {name!r}: free dimension (axis 1) has size {v1}, "
            "which is not a multiple of 512. NKI requires exact-multiple tile shapes.",
        )


class NKIBackend(Backend):
    """NKI (Neural Kernel Interface) code generation backend for Trainium."""

    def validate_nki_tensor_shapes(self, graph: torch.fx.Graph) -> None:
        """Run before NKI codegen: require partition dim % 128, free dim % 512."""
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        for node in graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    _validate_nki_tensor_shape(
                        tuple(val.shape), node.name or "input", env
                    )
        # Output: graph.output(result) — result is the single arg to the output node
        for node in graph.nodes:
            if node.op == "output":
                out_val = node.args[0]
                if isinstance(out_val, torch.fx.Node) and "val" in out_val.meta:
                    val = out_val.meta["val"]
                    if isinstance(val, torch.Tensor):
                        _validate_nki_tensor_shape(
                            tuple(val.shape), "output", env
                        )
                    elif isinstance(val, (list, tuple)):
                        for i, t in enumerate(val):
                            if isinstance(t, torch.Tensor):
                                _validate_nki_tensor_shape(
                                    tuple(t.shape), f"output[{i}]", env
                                )
                break

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        """NKI uses slice-based tiles; always use NDTileStrategy so pid/offset are emitted."""
        from .compile_environment import CompileEnvironment
        from .tile_strategy import NDTileStrategy

        class _NKINDTileStrategy(NDTileStrategy):
            def supports_index_rank_expansion(self) -> bool:
                # NKI handles broadcasting via partition/free axis semantics.
                # Adding [None, :] produces 3D indexing on 2D SBUF tiles.
                return False

        env = CompileEnvironment.current()
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        loop_order = env.config_spec.loop_orders.config_get(
            config.loop_orders, block_ids[0]
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )
        return _NKINDTileStrategy(
            fn,
            block_ids,
            block_size=[bs.from_config_assert(config) for bs in block_size_infos],
            loop_order=loop_order,
            l2_grouping=l2_grouping,
        )

    @property
    def name(self) -> str:
        return "nki"

    @property
    def codegen_name(self) -> str:
        return "nki"

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        """NKI: use nl.sequential_range with literal step (no tl.range / constexpr).

        When this loop has a dynamic (tensor-valued) bound, the tile_strategy
        has pre-emitted a register setup and stashed the register var name on
        ``device_function._nki_dyn_range_end_var``. Use dynamic_range instead.
        """
        begin_part = begin if begin is not None else "0"
        step_part = step if step is not None else "1"
        from .compile_environment import CompileEnvironment
        _env = CompileEnvironment.current()
        _state = getattr(_env, "_codegen_state", None)
        _dyn_reg = None
        if _state is not None:
            _dyn_reg = getattr(_state.device_function, "_nki_dyn_range_end_var", None)
        if _dyn_reg is not None:
            # Consume once so nested loops don't reuse.
            _state.device_function._nki_dyn_range_end_var = None
            return f"nl.dynamic_range({begin_part}, {_dyn_reg}, {step_part})"
        return f"nl.sequential_range({begin_part}, {end}, {step_part})"

    def dtype_str(self, dtype: torch.dtype) -> str:
        _DTYPE_MAP = {
            torch.float16: "nl.float16",
            torch.bfloat16: "nl.bfloat16",
            torch.float32: "nl.float32",
            torch.int8: "nl.int8",
            torch.int16: "nl.int16",
            torch.int32: "nl.int32",
            # NKI does not expose int64. Widen nothing at compile time; the
            # launcher (default_nki_launcher) and fake-tensor path cast
            # int64 args to int32 before they reach here, and intermediate
            # int64 FX values end up as int32 in generated code.
            torch.int64: "nl.int32",
            torch.uint8: "nl.uint8",
            torch.bool: "nl.bool_",
        }
        # FP8 (Trn2+). Some PyTorch builds lack certain fp8 aliases; guard.
        for _torch_name, _nki_name in (
            # This neuronx-cc build rejects F8E4M3FN HLO on TRN2 and does not
            # recognize the suggested compatibility flag, so Helion's NKI
            # path pre-casts PyTorch e4m3fn inputs to bfloat16 on the host.
            ("float8_e4m3fn", "nl.bfloat16"),
            ("float8_e4m3", "nl.float8_e4m3"),
            ("float8_e5m2", "nl.float8_e5m2"),
        ):
            _td = getattr(torch, _torch_name, None)
            if _td is not None:
                _DTYPE_MAP[_td] = _nki_name
        if dtype not in _DTYPE_MAP:
            raise exc.BackendUnsupported(self.name, f"dtype {dtype}")
        return _DTYPE_MAP[dtype]

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype in (torch.float16, torch.bfloat16):
            return "nl.float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        from .compile_environment import CompileEnvironment
        from helion.runtime.settings import get_neuron_target

        # Grab the active compilation environment
        env = CompileEnvironment.current()

        # Extract the config (falling back safely if it's nested in state)
        config = getattr(env, "config", None)
        if config is None:
            state = getattr(env, "_codegen_state", None)
            config = getattr(state, "config", None) if state else None

        config_target = getattr(config, "platform_target", None)
        if config_target is None:
            settings = getattr(env, "settings", None)
            if settings is None:
                state = getattr(env, "_codegen_state", None)
                settings = getattr(state, "settings", None) if state else None
            config_target = getattr(settings, "platform_target", None)

        # Resolve the target for our own downstream use (ISA gating, etc.), but
        # the NKI runtime reads NEURON_PLATFORM_TARGET_OVERRIDE directly — the
        # 'platform_target' kwarg on nki.jit is deprecated and now raises.
        get_neuron_target(config_target)

        return "nki.jit"

    @property
    def constexpr_type(self) -> str:
        return "int"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"(1 << (({expr}) - 1).bit_length())"

    @property
    def default_launcher_name(self) -> str:
        return "_default_nki_launcher"

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        from .device_function import DeviceFunction
        from .device_function import TensorArg

        if isinstance(arg, TensorArg):
            fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
            if fp8_e4m3fn is not None and arg.fake_value.dtype == fp8_e4m3fn:
                return f"{host_str}.to(torch.bfloat16)"
            cast_dtype = getattr(
                DeviceFunction.current(), "_nki_host_arg_casts", {}
            ).get(arg.name)
            if cast_dtype is not None:
                return f"{host_str}.to({cast_dtype})"
        return host_str

    def inductor_op_overrides(self) -> InductorOpOverrides:
        return NKIOpOverrides()

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "nki": "import nki",
            "nl": "import nki.language as nl",
            "nisa": "import nki.isa as nisa",
            "_default_nki_launcher": "from helion.runtime import default_nki_launcher as _default_nki_launcher",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        if index_dtype != "nl.int32":
            return f"nl.program_id({dim}).to({index_dtype})"
        return f"nl.program_id({dim})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        # NKI Beta 2: arange removed; use slicing or access patterns (.ap()).
        raise exc.BackendUnsupported(
            self.name,
            "arange is removed in NKI Beta 2; use Python slicing or access patterns",
        )

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return f"nl.full([1, 1], {offset_var}, dtype={dtype})"
        # Generate a [1, block_size] SBUF tile of offsets via nisa.iota.
        # The resulting variable supplants the scalar-offset default: any
        # downstream code that uses tile.index as a per-position tensor
        # (e.g. masks, gathers) sees an SBUF-resident vector instead of
        # a Python int. block_size_var is always a literal int at this
        # point because NKI requires compile-time tile sizes.
        return self._nki_iota_index_expr(
            offset_var=offset_var, block_size_var=block_size_var, dtype=dtype
        )

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return f"nl.full([1, 1], {offset_var}, dtype={dtype})"
        return self._nki_iota_index_expr(
            offset_var=offset_var, block_size_var=block_size_var, dtype=dtype
        )

    def _nki_iota_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str
    ) -> str:
        # Kept for backward-compat. The actual iota emission now happens via
        # ``loop_index_statements`` / ``grid_index_statements`` so alloc +
        # nisa.iota + ``index_var = idx_var`` are placed together inside the
        # for loop body where ``offset_var`` is in scope.
        return offset_var

    def loop_index_statements(
        self,
        *,
        offset_var: str,
        block_size_var: str,
        dtype: str,
        axis: int,
        index_var: str,
    ) -> list[str]:
        """Emit the statements that bind ``index_var`` to an SBUF iota tile.

        NKI has no ``nl.arange``; the equivalent for per-position tile
        indices is ``nisa.iota`` which fills an SBUF tile with an integer
        pattern. We allocate a fresh ``[1, block_size]`` tile and iota-fill
        it with ``offset_var`` as the start. The assignment
        ``index_var = <iota tile>`` keeps the generated code symmetric
        with other backends (``indices_N = ...``) so downstream codegen
        (masks, gather subscripts) finds the expected variable.
        """
        # If offset_var belongs to a dynamic_range loop, we can't use it as
        # a static iota offset (it's a register). Instead: iota(offset=0)
        # then add the SBUF counter tile.
        from .compile_environment import CompileEnvironment
        _env = CompileEnvironment.current()
        _state = getattr(_env, "_codegen_state", None)
        _dyn_counter = None
        _dyn_counter_float = None
        if _state is not None:
            _dyn_loops = getattr(_state.device_function, "_nki_dyn_loops", {})
            for _blk_info in _dyn_loops.values():
                if _blk_info.get("offset_var") == offset_var:
                    _dyn_counter = _blk_info["counter"]
                    _dyn_counter_float = _blk_info.get("counter_float")
                    break
            try:
                _idx_width = int(block_size_var)
            except (TypeError, ValueError):
                _idx_width = 1
            _state.device_function._nki_sbuf_shapes[index_var] = [1, _idx_width]
            _state.device_function._nki_sbuf_dtypes[index_var] = dtype
            _state.device_function._nki_iota_offsets[index_var] = (
                str(_dyn_counter_float or _dyn_counter)
                if _dyn_counter is not None
                else str(offset_var)
            )
            _state.device_function._nki_iota_block_sizes[index_var] = str(
                block_size_var
            )
        if _dyn_counter is not None:
            counter_operand = _dyn_counter_float or _dyn_counter
            if block_size_var == "1":
                return [
                    f"{index_var} = nl.ndarray([1, 1], {dtype}, buffer=nl.sbuf)",
                    f"nisa.memset({index_var}, value=0)",
                    f"nisa.tensor_scalar(dst={index_var}, data={index_var}, op0=nl.add, operand0={counter_operand})",
                ]
            return [
                f"{index_var} = nl.ndarray([1, {block_size_var}], {dtype}, buffer=nl.sbuf)",
                f"nisa.iota(dst={index_var}, pattern=[[1, {block_size_var}]], "
                f"offset=0, channel_multiplier=0)",
                f"nisa.tensor_scalar(dst={index_var}, data={index_var}, op0=nl.add, operand0={counter_operand})",
            ]
        if block_size_var == "1":
            return [
                f"{index_var} = nl.full([1, 1], {offset_var}, dtype={dtype})",
            ]
        return [
            f"{index_var} = nl.ndarray([1, {block_size_var}], {dtype}, buffer=nl.sbuf)",
            f"nisa.iota(dst={index_var}, pattern=[[1, {block_size_var}]], "
            f"offset={offset_var}, channel_multiplier=0)",
        ]

    # grid_index_statements mirrors loop_index_statements; the top-level grid
    # path has ``offset_var`` in scope at the location of the assignment.
    def grid_index_statements(
        self,
        *,
        offset_var: str,
        block_size_var: str,
        dtype: str,
        axis: int,
        index_var: str,
    ) -> list[str]:
        return self.loop_index_statements(
            offset_var=offset_var,
            block_size_var=block_size_var,
            dtype=dtype,
            axis=axis,
            index_var=index_var,
        )

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        # For simple cases where we can't emit statements, return as-is.
        # Real casting is done in cast_ast below.
        return expr_str

    def cast_ast(
        self,
        x: ast.AST,
        target_dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        src_shape: list[int | torch.SymInt] | None = None,
    ) -> ast.AST:
        """NKI dtype cast: allocate a new buffer, memset to 0, add the source
        into it.  tensor_tensor with mismatched dtypes does the conversion
        on Trainium hardware (e.g. fp16 src → fp32 dst)."""
        from .ast_extension import statement_from_string

        state = self._nki_codegen_state()
        if state is None:
            return x

        # Skip cast if source and target dtypes match or source is unknown
        if src_dtype is None or src_dtype == target_dtype:
            return x

        src_name = ast.unparse(x) if isinstance(x, ast.AST) else str(x)

        # Look up the SBUF tile shape registered during load codegen
        def _lookup_sbuf_shape(name: str) -> list[int | torch.SymInt] | None:
            shape = state.device_function._nki_sbuf_shapes.get(name)
            if shape is not None:
                return shape
            lookup = name
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                shape = state.device_function._nki_sbuf_shapes.get(lookup)
                if shape is not None:
                    return shape
            return None

        shape_dims = _lookup_sbuf_shape(src_name)

        if shape_dims is None:
            # Try to derive shape from the FX node being lowered.
            # V.current_node is the cast/to node; its args[0] is the source.
            # The source shape (in the FX trace) tells us the correct size.
            from torch._inductor.virtualized import V as _V
            _cur = _V.current_node
            _src_val = None
            if _cur is not None:
                # Try the first arg of the current node (cast input)
                if len(_cur.args) >= 1 and hasattr(_cur.args[0], "meta"):
                    _src_val = _cur.args[0].meta.get("val")
                if _src_val is None:
                    _src_val = _cur.meta.get("val")
            if _src_val is not None and isinstance(_src_val, torch.Tensor):
                from .compile_environment import CompileEnvironment
                env = CompileEnvironment.current()
                import sympy as _sp
                _subs = {}
                for _bs in env.block_sizes:
                    try:
                        _subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))
                    except Exception:
                        pass
                _resolved_dims = []
                for _d in _src_val.shape:
                    if isinstance(_d, torch.SymInt):
                        try:
                            _resolved_dims.append(int(_d._sympy_().subs(_subs)))
                        except Exception:
                            _resolved_dims.append(int(env.size_hint(_d)))
                    else:
                        _resolved_dims.append(int(_d))
                # Squeeze 3D+ shapes to 2D (NKI SBUF is always 2D)
                while len(_resolved_dims) > 2 and _resolved_dims[0] == 1:
                    _resolved_dims = _resolved_dims[1:]
                if len(_resolved_dims) > 2:
                    flat = 1
                    for d in _resolved_dims[:-1]:
                        flat *= d
                    _resolved_dims = [flat, _resolved_dims[-1]]
                if len(_resolved_dims) == 0:
                    shape_dims = [1, 1]  # scalar
                elif len(_resolved_dims) == 1:
                    # 1D: scalar per partition row → [N, 1] partition layout
                    shape_dims = [_resolved_dims[0], 1]
                else:
                    shape_dims = _resolved_dims

        if shape_dims is None:
            shape_dims = [1, 1]

        # Resolve symbolic dimensions to concrete values using config
        import sympy as _sympy
        from .compile_environment import CompileEnvironment
        env = CompileEnvironment.current()
        _bs_subs: dict[_sympy.Symbol, int] = {}
        for _bid in range(len(env.block_sizes)):
            _bs = env.block_sizes[_bid]
            _bs_subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))

        resolved = []
        for s in shape_dims:
            if isinstance(s, torch.SymInt):
                resolved.append(int(s._sympy_().subs(_bs_subs)))
            else:
                resolved.append(int(s))

        # NKI SBUF is always 2D. Squeeze 3D+ to 2D, pad 1D to 2D.
        while len(resolved) > 2 and resolved[0] == 1:
            resolved = resolved[1:]
        if len(resolved) > 2:
            flat = 1
            for d in resolved[:-1]:
                flat *= d
            resolved = [flat, resolved[-1]]
        if len(resolved) == 1:
            resolved.append(1)

        dtype_str = self.dtype_str(target_dtype)
        src_name = ast.unparse(x) if isinstance(x, ast.AST) else str(x)

        # Prefer the source SBUF tile's ACTUAL registered shape (e.g. the
        # matmul result [128, 128]) over the FX-derived abstract shape
        # (which may include an unsquashed batch/head dim that exceeds the
        # 128-partition NKI limit). If the src has a registered SBUF shape
        # and it's smaller than the FX-derived one, use it.
        src_sbuf_shape = _lookup_sbuf_shape(src_name)
        if (
            src_sbuf_shape is not None
            and len(src_sbuf_shape) == 2
            and len(resolved) == 2
            and resolved[0] > 128  # FX-derived partition dim exceeds hw cap
            and src_sbuf_shape[0] <= 128
        ):
            resolved = list(src_sbuf_shape)

        cast_var = state.device_function.new_var("_nki_cast", dce=True)
        src_tile_vars = state.device_function.get_tile_list_vars(src_name)

        # Register the cast var's shape so downstream casts and stores can look it up
        state.device_function._nki_sbuf_shapes[cast_var] = resolved

        # Propagate HBM source and dtype tracking through casts so partition-broadcast
        # can re-load from HBM even after dtype conversion
        _hbm_src_for_cast = getattr(state.device_function, "_nki_hbm_sources", {}).get(src_name)
        if _hbm_src_for_cast is not None:
            if not hasattr(state.device_function, "_nki_hbm_sources"):
                state.device_function._nki_hbm_sources = {}
            state.device_function._nki_hbm_sources[cast_var] = _hbm_src_for_cast
        if not hasattr(state.device_function, "_nki_sbuf_dtypes"):
            state.device_function._nki_sbuf_dtypes = {}
        state.device_function._nki_sbuf_dtypes[cast_var] = dtype_str

        def _mark_host_hbm_cast(hbm_src: str) -> bool:
            hbm_arg_name = hbm_src.split("[", 1)[0].split(".", 1)[0].strip()
            if not hbm_arg_name.isidentifier():
                return False
            state.device_function._nki_host_arg_casts[hbm_arg_name] = (
                "torch.bfloat16"
            )
            return True

        direct_hbm_cast_src: str | None = None
        if (
            src_dtype is torch.int16
            and target_dtype is torch.bfloat16
            and _hbm_src_for_cast is not None
        ):
            # NKI on this runtime corrupts direct int16 HBM DMA. For the
            # Helion pattern ``tensor_arg[...] .to(bfloat16)``, cast the host
            # tensor argument before launch and reload the source slice into a
            # bfloat16 tile here.
            if _mark_host_hbm_cast(_hbm_src_for_cast):
                direct_hbm_cast_src = _hbm_src_for_cast

        # Determine if src is a Python scalar (e.g. v_0 = 128, an integer constant)
        # vs an NKI tensor (e.g. mean_x_squared_extra holding an SBUF tile value).
        # A Python scalar has: no SBUF shape, no tile vars, no subscript brackets,
        # AND either (a) an integer source dtype, or (b) can be parsed as a literal number.
        _no_sbuf_shape = _lookup_sbuf_shape(src_name) is None
        _no_brackets = "[" not in src_name and "(" not in src_name
        _is_integer_dtype = (
            src_dtype is not None
            and src_dtype in (torch.int32, torch.int64, torch.int16, torch.int8,
                              torch.uint8, torch.bool)
        )
        _is_numeric_literal = False
        try:
            float(src_name)
            _is_numeric_literal = True
        except (ValueError, TypeError):
            pass

        # Use memset when the source is provably a Python scalar constant:
        # - integer dtype (tensor size, loop bound, etc.) on any buffer shape
        # - literal number like "1.0" or "128"
        # - a kernel-parameter scalar (e.g. ``alpha: float`` passed in).
        # Float NKI tensor expressions always go through tensor_tensor.
        _is_scalar_arg = (
            src_tile_vars is None
            and _no_sbuf_shape
            and _no_brackets
            and src_name in getattr(state.device_function, "_nki_scalar_arg_names", set())
        )
        src_is_scalar = (
            src_tile_vars is None
            and _no_sbuf_shape
            and _no_brackets
            and (_is_integer_dtype or _is_numeric_literal or _is_scalar_arg)
        )
        if _is_scalar_arg:
            return expr_from_string(src_name)

        if src_tile_vars is None:
            shape_parts = [str(d) for d in resolved]
            shape_str = ", ".join(shape_parts)
            state.codegen.add_statement(
                statement_from_string(
                    f"{cast_var} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
                )
            )

        if direct_hbm_cast_src is not None:
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.dma_copy(dst={cast_var}, src={direct_hbm_cast_src})"
                )
            )
        elif src_is_scalar:
            # For scalar-to-tensor cast, use memset with the scalar value directly
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.memset({cast_var}, value={src_name})"
                )
            )
        elif src_tile_vars is not None:
            # Each tile is [NKI_PARTITION_MAX=128, free]; the overall shape is
            # [n_tiles * 128, free]. Cast per-tile using the PER-TILE shape, not
            # the global shape.
            cast_tile_vars = []
            _NKI_PARTITION_MAX = 128
            per_tile_partition = min(resolved[0], _NKI_PARTITION_MAX) if len(resolved) >= 2 else resolved[0]
            per_tile_free = resolved[1] if len(resolved) >= 2 else 1
            per_tile_shape_str = f"{per_tile_partition}, {per_tile_free}"
            for i, tv in enumerate(src_tile_vars):
                ctv = f"{cast_var}_{i}"
                cast_tile_vars.append(ctv)
                state.codegen.add_statement(
                    statement_from_string(
                        f"{ctv} = nl.ndarray([{per_tile_shape_str}], {dtype_str}, buffer=nl.sbuf)"
                    )
                )
                tile_hbm_src = getattr(
                    state.device_function, "_nki_hbm_sources", {}
                ).get(tv)
                if (
                    src_dtype is torch.int16
                    and target_dtype is torch.bfloat16
                    and tile_hbm_src is not None
                    and _mark_host_hbm_cast(tile_hbm_src)
                ):
                    state.codegen.add_statement(
                        statement_from_string(
                            f"nisa.dma_copy(dst={ctv}, src={tile_hbm_src})"
                        )
                    )
                else:
                    state.codegen.add_statement(
                        statement_from_string(f"nisa.tensor_copy(dst={ctv}, src={tv})")
                    )
                # Register per-tile shape
                state.device_function._nki_sbuf_shapes[ctv] = [per_tile_partition, per_tile_free]
                state.device_function._nki_sbuf_dtypes[ctv] = dtype_str
            state.device_function.register_tile_list(cast_var, cast_tile_vars)
        else:
            state.codegen.add_statement(
                statement_from_string(f"nisa.tensor_copy(dst={cast_var}, src={src_name})")
            )

        return expr_from_string(cast_var)

    def _nki_codegen_state(self) -> object | None:
        """Return current codegen state when set (Option B, NKI statement-based codegen)."""
        if self.name != "nki":
            return None
        from .compile_environment import CompileEnvironment

        return getattr(CompileEnvironment.current(), "_codegen_state", None)

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        # Two-line pattern: (1) ndarray alloc, (2) nisa.memset(dst, value=...).
        # Callers must emit the second line via full_memset_stmt().
        if not shape_dims:
            return value_expr
        # NKI requires at least 2D (partition, free). Append trailing 1 if 1D.
        if len(shape_dims) == 1:
            shape_dims = shape_dims + ["1"]
        # NKI SBUF is always 2D. Squeeze leading 1-dims from 3D+ shapes,
        # then flatten remaining leading dims if still > 2D.
        while len(shape_dims) > 2:
            try:
                if int(shape_dims[0]) == 1:
                    shape_dims = shape_dims[1:]
                    continue
            except (ValueError, TypeError):
                pass
            # Flatten leading dims into one
            leading = shape_dims[:-1]
            flat = " * ".join(f"({d})" for d in leading)
            shape_dims = [flat, shape_dims[-1]]
        shape_str = ", ".join(shape_dims)
        dtype_str = self.dtype_str(dtype)
        return f"nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"

    def full_memset_stmt(self, var: str, value_expr: str) -> str:
        """Return the second line for full: nisa.memset(var, value=value_expr)."""
        return f"nisa.memset({var}, value={value_expr})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        # NKI SBUF tensors are always 2D; reshape is a no-op
        return expr

    def scalar_load_expr(self, tensor_name: str) -> str:
        raise exc.BackendUnsupported(
            self.name,
            "scalar tensor loads (nl.load removed; use dma_copy codegen path)",
        )

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        # Delegate to NKIOpOverrides.where, which emits
        # nisa.tensor_copy_predicated on Trn2+. This path fires when
        # codegen calls where_expr directly (rarely) rather than going
        # through the Inductor OpsHandler (which already dispatches to
        # NKIOpOverrides.where via getattr).
        from .ast_extension import statement_from_string

        state = self._nki_codegen_state()
        if state is None:
            raise exc.BackendUnsupported(
                self.name, "where_expr requires active codegen state"
            )
        overrides = self.inductor_op_overrides()
        return overrides.where(mask, true_val, false_val)

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        # No nisa.* API for broadcast_to is documented in NKI; add statement-based path when available.
        raise exc.BackendUnsupported(
            self.name, "nl.broadcast_to does not exist in nki.*"
        )

    def minimum_expr(self, a: str, b: str) -> str:
        state = self._nki_codegen_state()
        if state is not None:
            # nisa.tensor_tensor(dst, data1, data2, op) is used in this codebase; op name per NKI docs.
            result_var = state.device_function.new_var("nki_minimum", dce=True)
            state.add_statement(
                f"nisa.tensor_tensor(dst={result_var}, data1={a}, data2={b}, op=nl.minimum)"
            )
            return result_var
        raise exc.BackendUnsupported(
            self.name, "nl.minimum does not exist in nki.*"
        )

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        fake_input: torch.Tensor | None = None,
        fake_output: torch.Tensor | None = None,
    ) -> str:
        state = self._nki_codegen_state()
        if state is not None:
            _NKI_REDUCTION_OPS = {
                "sum": "nl.add",
                "max": "nl.maximum",
                "min": "nl.min",
                "prod": "nl.mul",
                "mean": "nl.add",
            }
            op = _NKI_REDUCTION_OPS.get(reduction_type)
            if op is None:
                raise exc.BackendUnsupported(
                    self.name, f"reduction {reduction_type!r} not mapped to NKI op"
                )

            from .compile_environment import CompileEnvironment

            env = CompileEnvironment.current()
            pre_reduced = getattr(
                state.device_function, "_nki_pre_reduced_loads", {}
            )
            pre_reduce_info = pre_reduced.get(input_name)
            if (
                pre_reduce_info is not None
                and reduction_type == "sum"
                and pre_reduce_info.get("dim") == dim
            ):
                return input_name

            def _resolve_reduction_extent(extent: object) -> int:
                if isinstance(extent, torch.SymInt):
                    if state.config is not None:
                        import sympy as _sympy_reduce_extent

                        bs_subs: dict[_sympy_reduce_extent.Symbol, int] = {}
                        for bs in env.block_sizes:
                            cfg = bs.from_config(state.config)
                            if isinstance(cfg, int):
                                bs_subs[bs.symbol()] = cfg
                        try:
                            return int(extent._sympy_().subs(bs_subs))
                        except (TypeError, ValueError):
                            return int(env.size_hint(extent))
                    return int(env.size_hint(extent))
                return int(extent)

            if fake_input is not None and dim < fake_input.ndim:
                reduction_extent = fake_input.size(dim)
                reduction_extent = _resolve_reduction_extent(reduction_extent)
                if int(reduction_extent) == 1:
                    return input_name

            logical_dim = dim
            nki_dim = dim
            leading_singleton = True
            if fake_input is not None and fake_input.ndim > 2:
                n_squeeze = fake_input.ndim - 2
                nki_dim = dim - n_squeeze
                if nki_dim == 0 and logical_dim > 0:
                    for lead_dim in range(logical_dim):
                        if _resolve_reduction_extent(fake_input.size(lead_dim)) != 1:
                            leading_singleton = False
                            break

            if nki_dim == 0:
                if not leading_singleton:
                    raise exc.BackendUnsupported(
                        self.name,
                        "partition-axis reduction with non-singleton leading "
                        "dimensions is unsupported in NKI lowering",
                    )
                input_shape = state.device_function._nki_sbuf_shapes.get(input_name)
                if (
                    fake_input is not None
                    and fake_input.ndim == 1
                    and input_shape is not None
                    and len(input_shape) == 2
                    and input_shape[0] == 1
                    and input_shape[1] > 1
                ):
                    if fake_output is not None:
                        dtype_str = env.backend.dtype_str(fake_output.dtype)
                    else:
                        dtype_str = env.backend.dtype_str(fake_input.dtype)
                    result_var = state.device_function.new_var("nki_reduce", dce=True)
                    state.device_function._nki_sbuf_shapes[result_var] = [1, 1]
                    state.device_function._nki_sbuf_dtypes[result_var] = dtype_str
                    state.add_statement(
                        f"{result_var} = nl.ndarray([1, 1], {dtype_str}, buffer=nl.sbuf)"
                    )
                    state.add_statement(
                        f"nisa.tensor_reduce(dst={result_var}, op={op}, data={input_name}, axis=1, keepdims=True)"
                    )
                    if reduction_type == "mean":
                        reduction_extent = fake_input.size(0)
                        if isinstance(reduction_extent, torch.SymInt):
                            reduction_extent = env.size_hint(reduction_extent)
                        reduction_extent_int = int(reduction_extent)
                        if reduction_extent_int <= 0:
                            raise exc.BackendUnsupported(
                                self.name,
                                f"mean reduction requires positive reduction extent, got {reduction_extent_int}",
                            )
                        scale = 1.0 / reduction_extent_int
                        state.add_statement(
                            "nisa.tensor_scalar("
                            f"dst={result_var}, data={result_var}, op0=nl.multiply, operand0={repr(scale)}, op1=None)"
                        )
                    return result_var

                # tensor_partition_reduce supports: add, max, bitwise_or, bitwise_and.
                # Use nl.max (not nl.maximum) per NKI naming convention.
                _NKI_PARTITION_REDUCE_OPS = {"sum": "nl.add", "max": "nl.max", "mean": "nl.add"}
                part_op = _NKI_PARTITION_REDUCE_OPS.get(reduction_type)
                if part_op is None:
                    raise exc.BackendUnsupported(
                        self.name,
                        f"partition-axis reduction {reduction_type!r} not supported by "
                        "nisa.tensor_partition_reduce (supports add, max)",
                    )
                if fake_input is not None:
                    if fake_input.ndim > 2:
                        free_size = _resolve_reduction_extent(fake_input.size(-1))
                    else:
                        free_size = (
                            _resolve_reduction_extent(fake_input.size(1))
                            if fake_input.ndim >= 2
                            else 1
                        )
                else:
                    free_size = 1
                dst_shape = f"[1, {free_size}]"
                if reduction_type == "mean":
                    dtype_str = "nl.float32"
                elif fake_output is not None:
                    dtype_str = env.backend.dtype_str(fake_output.dtype)
                elif fake_input is not None:
                    dtype_str = env.backend.dtype_str(fake_input.dtype)
                else:
                    dtype_str = "nl.float32"
                result_var = state.device_function.new_var("nki_part_reduce", dce=True)
                state.device_function._nki_sbuf_shapes[result_var] = [1, free_size]
                state.device_function._nki_sbuf_dtypes[result_var] = dtype_str
                state.add_statement(
                    f"{result_var} = nl.ndarray({dst_shape}, {dtype_str}, buffer=nl.sbuf)"
                )
                state.add_statement(
                    f"nisa.tensor_partition_reduce(dst={result_var}, op={part_op}, data={input_name})"
                )
                if reduction_type == "mean":
                    if fake_input is None:
                        raise exc.BackendUnsupported(
                            self.name,
                            "mean reduction requires fake_input to determine reduction extent",
                        )
                    reduction_extent_int = _resolve_reduction_extent(
                        fake_input.size(logical_dim)
                    )
                    if reduction_extent_int > 0:
                        scale = 1.0 / reduction_extent_int
                        state.add_statement(
                            "nisa.tensor_scalar("
                            f"dst={result_var}, data={result_var}, op0=nl.multiply, operand0={repr(scale)}, op1=None)"
                        )
                # In NKI, always keep the result as 2D [1, free_size] — do NOT
                # slice to [free_size] (1D).  The partition-reduce output is [1, N]
                # and downstream accumulators (e.g. grad_w_acc) are also [1, N].
                # Taking [0, :] would create a 1D tensor that mismatches the 2D
                # accumulator in nisa.tensor_tensor, producing silent wrong values.
                return result_var

            # dim >= 1: use tensor_reduce on the free axis.  For 3D+ shapes,
            # nki_dim accounts for squeezed leading singleton dimensions,
            # e.g. dim=2 on [1,M,N] -> dim=1 on [M,N].
            dim = nki_dim

            # Derive the partition size from fake_input.size(0) with SymInt→config
            # substitution so the configured block size is used rather than the trace
            # hint (e.g. an inner tile with block_size=1 has a default hint of 64,
            # but the actual partition dimension at runtime is 1).
            if fake_input is not None and fake_input.ndim >= 2:
                # For 3D+ inputs, partition dim is after squeezed leading dims
                part_dim_idx = max(0, fake_input.ndim - 2)
                part_size_sym = fake_input.size(part_dim_idx)
                if isinstance(part_size_sym, torch.SymInt) and state.config is not None:
                    import sympy as _sympy
                    _bs_subs: dict[_sympy.Symbol, int] = {}
                    for _bs in env.block_sizes:
                        _cfg = _bs.from_config(state.config)
                        if isinstance(_cfg, int):
                            _bs_subs[_bs.symbol()] = _cfg
                    _resolved = part_size_sym._sympy_().subs(_bs_subs)
                    try:
                        part_size = int(_resolved)
                    except (TypeError, ValueError):
                        part_size = int(env.size_hint(part_size_sym))
                else:
                    part_size = int(part_size_sym)
                dst_shape = f"[{part_size}, 1]"
            else:
                part_size = 1
                dst_shape = "[1, 1]"
            dtype_str = "nl.float32"
            result_var = state.device_function.new_var("nki_reduce", dce=True)
            state.add_statement(
                f"{result_var} = nl.ndarray({dst_shape}, {dtype_str}, buffer=nl.sbuf)"
            )
            state.add_statement(
                f"nisa.tensor_reduce(dst={result_var}, op={op}, data={input_name}, axis={dim}, keepdims=True)"
            )
            if reduction_type == "mean":
                if fake_input is None:
                    raise exc.BackendUnsupported(
                        self.name,
                        "mean reduction requires fake_input to determine reduction extent",
                    )
                reduction_extent = fake_input.size(logical_dim)
                if isinstance(reduction_extent, torch.SymInt):
                    reduction_extent = env.size_hint(reduction_extent)
                reduction_extent_int = int(reduction_extent)
                if reduction_extent_int <= 0:
                    raise exc.BackendUnsupported(
                        self.name,
                        f"mean reduction requires positive reduction extent, got {reduction_extent_int}",
                    )
                scale = 1.0 / reduction_extent_int
                state.add_statement(
                    "nisa.tensor_scalar("
                    f"dst={result_var}, data={result_var}, op0=nl.multiply, operand0={repr(scale)}, op1=None)"
                )
            # Register the reduction result's SBUF shape for downstream store codegen
            state.device_function._nki_sbuf_shapes[result_var] = [part_size, 1]

            if (
                fake_input is not None
                and fake_output is not None
                and fake_output.ndim == fake_input.ndim - 1
                and dim == 1
            ):
                # For vector reductions (partition_dim > 1), keep the 2D [P, 1]
                # result so the store codegen can use partition-axis DMA.
                # For scalar reductions (partition_dim == 1), squeeze to 1D.
                if part_size > 1:
                    return result_var
                return f"{result_var}[:, 0]"
            return result_var
        raise exc.BackendUnsupported(
            self.name,
            f"reduction {reduction_type!r} (use nisa.tensor_reduce, requires dst)",
        )

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"0"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return f"nl.zeros([0], {dtype})"

    # -------------------------------------------------------------------------
    # Fused ISA helpers exposed for external call-sites (e.g. reduction_strategy)
    # -------------------------------------------------------------------------

    def tensor_scalar_reduce_expr(
        self,
        data: str,
        op0: str,
        operand0: object,
        reduce_op: str,
        reduce_shape: list[int],
        *,
        reverse0: bool = False,
    ) -> tuple[str, str]:
        """Emit nisa.tensor_scalar_reduce, return (dst_var, reduce_res_var).

        Wraps NKIOpOverrides._nki_tensor_scalar_reduce for use outside the
        NKIOpOverrides class (e.g. from reduction_strategy.py).
        """
        return NKIOpOverrides._nki_tensor_scalar_reduce(
            data, op0, operand0, reduce_op, reduce_shape, reverse0=reverse0
        )

    def activation_reduce_expr(
        self,
        data: str,
        op: str,
        reduce_op: str,
        reduce_shape: list[int],
        *,
        bias: str | None = None,
        scale: float | str | None = None,
    ) -> tuple[str, str]:
        """Emit nisa.activation_reduce, return (dst_var, reduce_res_var).

        Wraps NKIOpOverrides._nki_activation_reduce for use outside the
        NKIOpOverrides class (e.g. from reduction_strategy.py).
        """
        return NKIOpOverrides._nki_activation_reduce(
            data, op, reduce_op, reduce_shape, bias=bias, scale=scale
        )

    def scalar_tensor_tensor_expr(
        self,
        data: str,
        op0: str,
        operand0: object,
        op1: str,
        operand1: str,
        *,
        reverse0: bool = False,
        reverse1: bool = False,
    ) -> str:
        """Emit nisa.scalar_tensor_tensor, return the dst var name."""
        return NKIOpOverrides._nki_scalar_tensor_tensor(
            data, op0, operand0, op1, operand1, reverse0=reverse0, reverse1=reverse1
        )

    def tensor_tensor_scan_expr(
        self,
        data0: str,
        data1: str,
        initial: object,
        op0: str,
        op1: str,
        *,
        reverse0: bool = False,
        reverse1: bool = False,
    ) -> str:
        """Emit nisa.tensor_tensor_scan, return the dst var name."""
        return NKIOpOverrides._nki_tensor_tensor_scan(
            data0, data1, initial, op0, op1, reverse0=reverse0, reverse1=reverse1
        )

    def tensor_scalar_cumulative_expr(
        self,
        src: str,
        op0: str,
        op1: str,
        imm0: float | str,
        *,
        imm1: float | str | None = None,
        reduce_cmd: str = "nisa.reduce_cmd.reset_reduce",
    ) -> str:
        """Emit nisa.tensor_scalar_cumulative, return the dst var name."""
        return NKIOpOverrides._nki_tensor_scalar_cumulative(
            src, op0, op1, imm0, imm1=imm1, reduce_cmd=reduce_cmd
        )

    def inductor_op_overrides(self) -> InductorOpOverrides:
        return NKIOpOverrides()

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        config = bound_kernel.config_spec.default_config()
        block_sizes = config.config.get("block_sizes")
        if isinstance(block_sizes, list):
            safe_block_sizes = list(block_sizes)
            for i, spec in enumerate(bound_kernel.config_spec.block_sizes):
                if i >= len(safe_block_sizes):
                    break
                if i == 0:
                    continue
                size_hint = int(max(spec.size_hint, 1))
                current = int(safe_block_sizes[i])
                safe = min(current, 128)
                while safe > 1 and size_hint % safe != 0:
                    safe //= 2
                safe_block_sizes[i] = max(safe, 1)
            config.config["block_sizes"] = safe_block_sizes
        return config


# Mapping from torch dtype to JAX dtype string (e.g., "jnp.float32")
_TORCH_TO_JAX_DTYPE: dict[str, str] = {
    "torch.float16": "jnp.float16",
    "torch.float32": "jnp.float32",
    "torch.float64": "jnp.float64",
    "torch.bfloat16": "jnp.bfloat16",
    "torch.int8": "jnp.int8",
    "torch.int16": "jnp.int16",
    "torch.int32": "jnp.int32",
    "torch.int64": "jnp.int64",
    "torch.uint8": "jnp.uint8",
    "torch.bool": "jnp.bool_",
    "torch.complex64": "jnp.complex64",
    "torch.complex128": "jnp.complex128",
}


class PallasBackend(Backend):
    """Pallas (JAX) code generation backend."""

    @property
    def name(self) -> str:
        return "pallas"

    def dtype_str(self, dtype: torch.dtype) -> str:
        key = str(dtype)
        if key not in _TORCH_TO_JAX_DTYPE:
            raise ValueError(f"Unsupported dtype for Pallas backend: {dtype}")
        return _TORCH_TO_JAX_DTYPE[key]

    def acc_type(self, dtype: torch.dtype) -> str:
        # Promote half-precision types to float32 for numerical stability
        if dtype in (torch.float16, torch.bfloat16):
            return "jnp.float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_pallas_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "jax": "import jax",
            "jnp": "import jax.numpy as jnp",
            "pl": "from jax.experimental import pallas as pl",
            "lax": "import jax.lax as lax",
            "_default_pallas_launcher": "from helion.runtime import default_pallas_launcher as _default_pallas_launcher",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"pl.program_id({dim})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"lax.convert_element_type({expr_str}, {dtype_str})"

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)
        return f"range({', '.join(range_args)})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"{offsets_var} = {lid} * {block_size_var} + jnp.arange(0, {block_size_var}, dtype={dtype})"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.pallas import PallasKernelOverrides

        return PallasKernelOverrides()

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(
            f"lax.convert_element_type({{x}}, {self.dtype_str(target_dtype)})", x=x
        )

    # TODO(oulgen): once https://github.com/jax-ml/jax/pull/35116 is merged
    # and released, swap to static_argnums API instead of converting scalars
    # to 0-dim tensors.
    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        from .device_function import SymbolArgument
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            device_expr = (
                f"{tensor_host_args[0]}.device" if tensor_host_args else "'cuda'"
            )
            return f"torch.tensor({host_str}, device={device_expr})"
        return host_str

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        from .ast_extension import statement_from_string
        from .device_function import SymbolArgument
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            return [statement_from_string(f"{arg.name} = {arg.name}[...]")]
        return []

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + jnp.arange(0, ({block_size_var}), dtype={dtype})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + jnp.arange(0, ({block_size_var}), dtype={dtype})"

    def scalar_load_expr(self, tensor_name: str) -> str:
        return f"{tensor_name}[0]"

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        return f"jnp.full([{', '.join(shape_dims)}], {value_expr}, {self.dtype_str(dtype)})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return f"jnp.reshape({expr}, {shape})"

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return f"jnp.broadcast_to({expr}, {shape})"

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
    ) -> str:
        if reduction_type in {"sum", "max", "min", "prod"}:
            return f"jnp.{reduction_type}({input_name}, axis={dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"jnp.where({mask}, {true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        return f"jnp.minimum({a}, {b})"

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        return f"jnp.arange(0, {block_size_var}, dtype={dtype})"

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return f"jnp.zeros({shape}, dtype={dtype})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"jnp.arange(0, {block_size_var}, dtype={dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return f"jnp.zeros([0], dtype={dtype})"

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        return bound_kernel.config_spec.default_config()


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend."""

    @property
    def name(self) -> str:
        return "cute"

    def supports_config_key(self, key: str) -> bool:
        if key == "elements_per_thread":
            return True
        return super().supports_config_key(key)

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        if (
            inductor_dtype := CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        ) is not None:
            return inductor_dtype

        raise ValueError(f"Unsupported dtype for Cute backend: {dtype}")

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype in (torch.float16, torch.bfloat16):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return "cute.kernel"

    @property
    def constexpr_type(self) -> str:
        return "cutlass.Constexpr"

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

    @property
    def default_launcher_name(self) -> str:
        return "_default_cute_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "cutlass": "import cutlass",
            "cute": "import cutlass.cute as cute",
            "_default_cute_launcher": "from helion.runtime import default_cute_launcher as _default_cute_launcher",
            "_next_power_of_2": "from helion._utils import next_power_of_2 as _next_power_of_2",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"cute.arch.block_idx()[{dim}]"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        return CuteDSLOpOverrides()

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{dtype_str}({expr_str})"

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(f"cutlass.Int32({begin})")
        range_args.append(f"cutlass.Int32({end})")
        if step is not None and step != "1":
            range_args.append(f"cutlass.Int32({step})")
        return f"range({', '.join(range_args)})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return (
            f"{offsets_var} = ({lid}) * ({block_size_var})"
            f" + cutlass.Int32(cute.arch.thread_idx()[{axis}])"
        )

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if axis >= 3 and block_size_var != "1":
            raise exc.BackendUnsupported(self.name, f"thread axis {axis}")
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return self.grid_index_expr(offset_var, block_size_var, dtype, axis=axis)

    def scalar_load_expr(self, tensor_name: str) -> str:
        return f"{tensor_name}[0]"

    def max_reduction_threads(self) -> int | None:
        return 32

    def reduction_axis_first(self) -> bool:
        return True

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        return True

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        # One element per thread: tile-shaped temporaries are scalars.
        return f"{self.dtype_str(dtype)}({value_expr})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return expr

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return expr

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return "cutlass.Int32(0)"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"_next_power_of_2({expr})"

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        if reduction_type == "sum":
            return f"({acc} + {val})"
        if reduction_type == "max":
            return f"cute.where({acc} > {val}, {acc}, {val})"
        if reduction_type == "min":
            return f"cute.where({acc} < {val}, {acc}, {val})"
        if reduction_type == "prod":
            return f"({acc} * {val})"
        raise exc.BackendUnsupported(self.name, f"reduction combine {reduction_type!r}")

    def _threads_for_block_size_var(self, block_size_var: str | None) -> int:
        # threads_in_group must be a Python int literal for CuTe DSL.
        from .reduction_strategy import ReductionStrategy
        from .tile_strategy import BlockSizeTileStrategy

        threads = 32
        strategies = self._get_strategies()
        if block_size_var is not None:
            for strategy in strategies:
                if not isinstance(strategy, ReductionStrategy):
                    continue
                strategy_bs_var = strategy.block_size_var(strategy.block_index)
                if strategy_bs_var != block_size_var:
                    continue
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc

            # Block reductions are keyed by a tile block-size var rather than a
            # ReductionStrategy var. Recover the tile width from the owning strategy.
            for strategy in strategies:
                if not isinstance(strategy, BlockSizeTileStrategy):
                    continue
                for idx, block_id in enumerate(strategy.block_ids):
                    strategy_bs_var = strategy.block_size_var(block_id)
                    if strategy_bs_var != block_size_var:
                        continue
                    block_size = strategy.block_size
                    if isinstance(block_size, list) and idx < len(block_size):
                        block_size = block_size[idx]
                    if isinstance(block_size, int) and block_size > 0:
                        return min(block_size, 32)
            return threads

        for strategy in strategies:
            if isinstance(strategy, ReductionStrategy):
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc
        return threads

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
    ) -> str:
        threads = self._threads_for_block_size_var(block_size_var)
        tg = f", threads_in_group={threads}"
        if reduction_type == "sum":
            return f"cute.arch.warp_reduction_sum({input_name}{tg})"
        if reduction_type == "max":
            return f"cute.arch.warp_reduction_max({input_name}{tg})"
        if reduction_type == "min":
            return (
                f"cute.arch.warp_reduction("
                f"{input_name}, lambda a, b: (a if a < b else b){tg})"
            )
        if reduction_type == "prod":
            return f"cute.arch.warp_reduction({input_name}, lambda a, b: (a * b){tg})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        return reduction_type in {"argmin", "argmax"}

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
    ) -> str:
        if index_dtype is None:
            raise exc.BackendUnsupported(self.name, "missing index_dtype for argreduce")
        value_reduction = "min" if reduction_type == "argmin" else "max"
        reduced_value = self.reduction_expr(
            input_name,
            value_reduction,
            dim,
            block_size_var=block_size_var,
        )
        index_dtype_str = self.index_type_str(index_dtype)
        max_index = self.cast_expr(repr(torch.iinfo(index_dtype).max), index_dtype_str)
        candidate_index = f"({index_value}) if (({input_name}) == ({reduced_value})) else ({max_index})"
        reduced_index = self.reduction_expr(
            candidate_index,
            "min",
            dim,
            block_size_var=block_size_var,
        )
        return self.cast_expr(reduced_index, self.dtype_str(output_dtype))

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        if reduction_type == "argmin":
            better = (
                f"(({value}) < ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        else:
            better = (
                f"(({value}) > ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        return [
            (
                f"{acc}, {acc_index} = "
                f"(({value}), ({index})) if ({better}) else (({acc}), ({acc_index}))"
            )
        ]

    def _get_strategies(self) -> list[TileStrategy]:
        """Get the current device function's strategies."""
        from .device_function import DeviceFunction

        try:
            return DeviceFunction.current().tile_strategy.strategies
        except Exception:
            return []

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from .device_function import DeviceFunction

        dims = DeviceFunction.current().tile_strategy.thread_block_dims()
        if dims[0] * dims[1] * dims[2] > 1024:
            raise exc.BackendUnsupported(
                self.name,
                f"thread block too large for cute kernel: {tuple(dims)}",
            )
        return [f"block=({dims[0]}, {dims[1]}, {dims[2]})"]

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
    ) -> list[str]:
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        if not tensor_host_args:
            raise exc.BackendUnsupported(self.name, "kernel launch without tensor args")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from .compile_environment import CompileEnvironment
        from .device_ir import ForLoopGraphInfo
        from .device_ir import ReductionLoopGraphInfo
        from .host_function import HostFunction
        from .tile_strategy import CuteFlattenedTileStrategy
        from .tile_strategy import CuteNDTileStrategy

        env = CompileEnvironment.current()
        device_ir = HostFunction.current().device_ir
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        flattened = block_size_infos[0].is_flattened(config)
        loop_order = env.config_spec.loop_orders.config_get(
            config.loop_orders, block_ids[0]
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )
        has_device_loops = any(
            isinstance(graph, ForLoopGraphInfo)
            and not isinstance(graph, ReductionLoopGraphInfo)
            for graph in device_ir.graphs
        )
        has_dynamic_shape = any(env.block_sizes[i].size is None for i in block_ids)
        elements_per_thread = [
            int(
                env.config_spec.elements_per_thread.config_get(
                    config.elements_per_thread, block_id, 1
                )
            )
            for block_id in block_ids
        ]
        if (
            has_device_loops
            or has_dynamic_shape
            or len(device_ir.grid_block_ids) != 1
            or (len(block_ids) > 1 and not flattened)
        ):
            nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
            int_positions = [
                i for i, bs in enumerate(nd_block_size) if isinstance(bs, int)
            ]
            static_threads = functools.reduce(
                operator.mul,
                (
                    int(nd_block_size[i]) // elements_per_thread[i]
                    for i in int_positions
                ),
                1,
            )
            if static_threads > 1024:
                raise exc.BackendUnsupported(
                    self.name,
                    f"thread block too large for cute kernel: {tuple(nd_block_size)}",
                )
            return CuteNDTileStrategy(
                fn,
                block_ids,
                block_size=nd_block_size,
                loop_order=loop_order,
                l2_grouping=l2_grouping,
                elements_per_thread=elements_per_thread,
            )
        flat_elements_per_thread = functools.reduce(
            operator.mul, elements_per_thread, 1
        )
        block_size = functools.reduce(
            operator.mul, [bs.from_config_assert(config) for bs in block_size_infos]
        )
        if isinstance(block_size, int):
            physical_threads = block_size // max(flat_elements_per_thread, 1)
            if physical_threads > 1024:
                raise exc.BackendUnsupported(
                    self.name,
                    f"thread block too large for cute kernel: {block_size}",
                )
        return CuteFlattenedTileStrategy(
            fn,
            block_ids,
            block_size=block_size,
            loop_order=loop_order,
            elements_per_thread=flat_elements_per_thread,
        )

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        return bound_kernel.config_spec.default_config()
