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
    def _nki_tensor_tensor(a: object, b: object, op: str, prefix: str) -> str:
        from .ast_extension import statement_from_string
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        if env.backend.name != "nki":
            return ""
        state = getattr(env, "_codegen_state", None)
        if state is None:
            return ""

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

        cur_node = V.current_node
        _is_second_level_copy = "_copy_" in dst and dst[-1:].isdigit()
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
                b_sbuf_shape_alloc = state.device_function._nki_sbuf_shapes.get(b_str)
                if (
                    b_sbuf_shape_alloc is not None
                    and len(shape_list) >= 2
                    and len(b_sbuf_shape_alloc) >= 2
                    and shape_list[0] != b_sbuf_shape_alloc[0]
                ):
                    # Use the larger partition count for the result shape
                    max_p = max(shape_list[0], b_sbuf_shape_alloc[0])
                    shape_list = [max_p] + list(shape_list[1:])

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
                    # Check for partition broadcast before emitting tensor_tensor
                    _new_a_shape = state.device_function._nki_sbuf_shapes.get(str(a))
                    _new_b_shape = state.device_function._nki_sbuf_shapes.get(b_str)
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
                            state.add_statement(statement_from_string(
                                f"nisa.tensor_tensor(dst={new_dst}, data1={a}, data2={b}, op={op})"
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
        a_sbuf_shape = state.device_function._nki_sbuf_shapes.get(dst)
        b_sbuf_shape = state.device_function._nki_sbuf_shapes.get(b_str)
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

        if b_needs_broadcast:
            p_count = a_sbuf_shape[0]
            if _emit_broadcast_op(b_str, b_sbuf_shape, dst, b_str, dst, p_count,
                                   _bcast_dtype_str):
                return dst
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
        return x

    @classmethod
    def _is_scalar_operand(cls, x: object) -> bool:
        resolved = cls._resolve_scalar_operand(x)
        return isinstance(resolved, (int, float, bool))

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
        from torch._inductor.virtualized import V as _V_ts
        _ts_node = _V_ts.current_node
        _ts_is_copy = "_copy_" in dst and dst[-1:].isdigit()
        _ts_multi_user = _ts_is_copy or (
            _ts_node is not None
            and len(_ts_node.args) >= 1
            and hasattr(_ts_node.args[0], "users")
            and len(_ts_node.args[0].users) > 1
        )
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
                _operand_emit_ts = _scalar_for_emit(operand0)
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

        operand_emit = _scalar_for_emit(operand0)
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

        # Check for broadcast pattern: [M,N] op [M,1] after 3D squeeze
        # This handles cases like acc * alpha[:, :, None] where alpha is [1,M,1]
        if not a_is_scalar and not b_is_scalar:
            from torch._inductor.virtualized import V as _V_bin
            _bin_node = _V_bin.current_node
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

        # tensor <op> tensor
        if allow_tensor_tensor:
            return cls._nki_tensor_tensor(a, b, op_tensor_tensor, "nki_binary")

        raise exc.BackendUnsupported(
            "nki",
            f"single-op tensor_scalar path does not support tensor/tensor for op {op_tensor_scalar}",
        )

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
        need_new_buffer = _is_second_level_copy or _is_protected_act
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
        raise exc.BackendUnsupported(
            "nki",
            "truediv tensor/tensor currently supports only vector-like rhs broadcast "
            "with shape [M, 1] over a 2D lhs tile.",
        )

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

    # Index / scalar arithmetic — these operate on Python integers (tile offsets,
    # loop counters), not on SBUF tiles, so plain Python operators are correct.
    @staticmethod
    def floordiv(a: object, b: object) -> str:
        return f"{a} // {b}"

    @staticmethod
    def mod(a: object, b: object) -> str:
        return f"{a} % {b}"

    @staticmethod
    def remainder(a: object, b: object) -> str:
        return f"{a} % {b}"

    @staticmethod
    def lt(a: object, b: object) -> str:
        return f"{a} < {b}"

    @staticmethod
    def le(a: object, b: object) -> str:
        return f"{a} <= {b}"

    @staticmethod
    def gt(a: object, b: object) -> str:
        return f"{a} > {b}"

    @staticmethod
    def ge(a: object, b: object) -> str:
        return f"{a} >= {b}"

    @staticmethod
    def eq(a: object, b: object) -> str:
        return f"{a} == {b}"

    @staticmethod
    def ne(a: object, b: object) -> str:
        return f"{a} != {b}"

    @staticmethod
    def and_(a: object, b: object) -> str:
        return f"{a} & {b}"

    @staticmethod
    def or_(a: object, b: object) -> str:
        return f"{a} | {b}"

    @staticmethod
    def xor(a: object, b: object) -> str:
        return f"{a} ^ {b}"

    @staticmethod
    def lshift(a: object, b: object) -> str:
        return f"{a} << {b}"

    @staticmethod
    def rshift(a: object, b: object) -> str:
        return f"{a} >> {b}"

    @staticmethod
    def pow(a: object, b: object) -> str:
        return f"{a} ** {b}"


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
        """NKI: use nl.sequential_range with literal step (no tl.range / constexpr)."""
        begin_part = begin if begin is not None else "0"
        step_part = step if step is not None else "1"
        return f"nl.sequential_range({begin_part}, {end}, {step_part})"

    def dtype_str(self, dtype: torch.dtype) -> str:
        _DTYPE_MAP = {
            torch.float16: "nl.float16",
            torch.bfloat16: "nl.bfloat16",
            torch.float32: "nl.float32",
            torch.int8: "nl.int8",
            torch.int16: "nl.int16",
            torch.int32: "nl.int32",
            torch.int64: "nl.int64",
            torch.uint8: "nl.uint8",
            torch.bool: "nl.bool_",
        }
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
        
        # Resolve the actual hardware target string
        resolved_target = get_neuron_target(config_target)
        
        # Bake the hardware target into the generated AST
        return f'nki.jit(platform_target="{resolved_target}")'

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_nki_launcher"

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
            return f"{offset_var} + nl.zeros([1], {dtype})"
        # Slice-based tile model: index is the scalar tile start offset.
        return offset_var

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return f"{offset_var} + nl.zeros([1], {dtype})"
        # Slice-based tile model: index is the scalar tile start offset.
        return offset_var

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
        shape_dims = state.device_function._nki_sbuf_shapes.get(src_name)

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

        shape_parts = [str(d) for d in resolved]
        shape_str = ", ".join(shape_parts)

        dtype_str = self.dtype_str(target_dtype)
        src_name = ast.unparse(x) if isinstance(x, ast.AST) else str(x)
        cast_var = state.device_function.new_var("_nki_cast", dce=True)

        # Register the cast var's shape so downstream casts and stores can look it up
        state.device_function._nki_sbuf_shapes[cast_var] = resolved

        # Propagate HBM source and dtype tracking through casts so partition-broadcast
        # can re-load from HBM even after dtype conversion
        _hbm_src_for_cast = getattr(state.device_function, "_nki_hbm_sources", {}).get(src_name)
        if _hbm_src_for_cast is not None:
            if not hasattr(state.device_function, "_nki_hbm_sources"):
                state.device_function._nki_hbm_sources = {}
            state.device_function._nki_hbm_sources[cast_var] = _hbm_src_for_cast
        # Also propagate the ORIGINAL (pre-cast) SBUF dtype for correct broadcast loads
        _src_dtype_str = getattr(state.device_function, "_nki_sbuf_dtypes", {}).get(src_name)
        if _src_dtype_str is not None:
            if not hasattr(state.device_function, "_nki_sbuf_dtypes"):
                state.device_function._nki_sbuf_dtypes = {}
            state.device_function._nki_sbuf_dtypes[cast_var] = _src_dtype_str

        src_tile_vars = state.device_function.get_tile_list_vars(src_name)
        # Determine if src is a Python scalar (e.g. v_0 = 128, an integer constant)
        # vs an NKI tensor (e.g. mean_x_squared_extra holding an SBUF tile value).
        # A Python scalar has: no SBUF shape, no tile vars, no subscript brackets,
        # AND either (a) an integer source dtype, or (b) can be parsed as a literal number.
        _no_sbuf_shape = state.device_function._nki_sbuf_shapes.get(src_name) is None
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
        # Float NKI tensor expressions always go through tensor_tensor.
        src_is_scalar = (
            src_tile_vars is None
            and _no_sbuf_shape
            and _no_brackets
            and (_is_integer_dtype or _is_numeric_literal)
        )

        # Emit: cast_var = nl.ndarray(shape, target_dtype, buffer=nl.sbuf)
        state.codegen.add_statement(
            statement_from_string(
                f"{cast_var} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
            )
        )

        if src_is_scalar:
            # For scalar-to-tensor cast, use memset with the scalar value directly
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.memset({cast_var}, value={src_name})"
                )
            )
        elif src_tile_vars is not None:
            cast_tile_vars = []
            for i, tv in enumerate(src_tile_vars):
                ctv = f"{cast_var}_{i}"
                cast_tile_vars.append(ctv)
                state.codegen.add_statement(
                    statement_from_string(
                        f"{ctv} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
                    )
                )
                state.codegen.add_statement(
                    statement_from_string(f"nisa.memset({ctv}, value=0)")
                )
                state.codegen.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={ctv}, data1={ctv}, data2={tv}, op=nl.add)"
                    )
                )
            state.device_function.register_tile_list(cast_var, cast_tile_vars)
        else:
            state.codegen.add_statement(
                statement_from_string(f"nisa.memset({cast_var}, value=0)")
            )
            state.codegen.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={cast_var}, data1={cast_var}, data2={src_name}, op=nl.add)"
                )
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
        raise exc.BackendUnsupported(
            self.name,
            "trn1 does not support tensor comparison ops needed for where/masking. "
            "Use slice-based approaches (separate DMA copies) instead.",
        )

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

            if dim == 0:
                _NKI_PARTITION_REDUCE_OPS = {"sum": "nl.add", "max": "nl.maximum", "mean": "nl.add"}
                part_op = _NKI_PARTITION_REDUCE_OPS.get(reduction_type)
                if part_op is None:
                    raise exc.BackendUnsupported(
                        self.name,
                        f"partition-axis reduction {reduction_type!r} not supported by "
                        "nisa.tensor_partition_reduce (supports add, max)",
                    )
                if fake_input is not None:
                    free_size = int(fake_input.size(1)) if fake_input.ndim >= 2 else 1
                else:
                    free_size = 1
                dst_shape = f"[1, {free_size}]"
                dtype_str = "nl.float32"
                result_var = state.device_function.new_var("nki_part_reduce", dce=True)
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
                    reduction_extent = fake_input.size(0)
                    if isinstance(reduction_extent, torch.SymInt):
                        reduction_extent = env.size_hint(reduction_extent)
                    reduction_extent_int = int(reduction_extent)
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

            # dim >= 1: use tensor_reduce on the free axis
            # For 3D+ shapes (batch_block=1), adjust dim to account for
            # squeezed leading dims: e.g. dim=2 on [1,M,N] → dim=1 on [M,N]
            if fake_input is not None and fake_input.ndim > 2:
                n_squeeze = fake_input.ndim - 2
                dim = dim - n_squeeze

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
                reduction_extent = fake_input.size(dim)
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
        return bound_kernel.config_spec.default_config()


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
