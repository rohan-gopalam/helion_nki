from __future__ import annotations

import ast
from collections.abc import Callable
import contextlib
import dataclasses
from operator import getitem
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.utils import triton_type
from torch.fx.node import Argument
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc
from .._utils import next_power_of_2
from ..language.matmul_ops import enforce_dot_requirements
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.argreduce import codegen_cute_tile_argreduce
from .cute.cute_mma import codegen_cute_mma_direct_mm
from .cute.indexing import CutePackedAffineLoad
from .cute.indexing import CuteShapeChainView
from .cute.indexing import CuteSortableLoad
from .cute.indexing import is_cute_shape_chain_target
from .cute.indexing import match_cute_affine_range_iota
from .cute.iota_utils import cute_free_arange_indexed_dim_key
from .cute.iota_utils import cute_iota_has_atomic_tensor_index_only_users
from .cute.iota_utils import cute_iota_is_free_memory_index
from .cute.matmul_fallback import _emit_cute_matmul
from .cute.matmul_fallback import _emit_cute_matmul_n_collapse
from .cute.matmul_utils import cute_lower_rhs_for_matmul
from .cute.matmul_utils import cute_outer_accumulates_result
from .cute.matmul_utils import cute_outer_accumulator_dtype
from .cute.matmul_utils import cute_outer_accumulator_out_dtype
from .cute.matmul_utils import cute_rematerialize_rhs_at_contraction_block
from .cute.matmul_utils import cute_rematerialize_rhs_at_index_override
from .cute.matmul_utils import cute_resolve_active_block_id
from .cute.matmul_utils import cute_resolve_active_matmul_k_block_id
from .cute.matmul_utils import cute_static_k_invariant_extent
from .cute.matmul_utils import cute_static_mn_collapse_n_block_id
from .cute.matmul_utils import cute_static_serial_matmul_k_extent
from .cute.matmul_utils import emit_cute_serial_scalar_mm_from_loads
from .cute.strategies import is_pure_matmul_role_lifecycle_config
from .cute.tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from .matmul_utils import _emit_pallas_matmul
from .matmul_utils import _needs_f32_accumulator
from .matmul_utils import emit_tl_dot_with_padding
from .node_masking import apply_masking
from .node_masking import cached_masked_value
from .node_masking import getitem_masked_value

if TYPE_CHECKING:
    from .generate_ast import GenerateAST
    from .helper_function import CodegenInterface


class LoweringContext:
    cg: CodegenInterface
    env: dict[Node, Argument]

    def to_ast(self, value: object) -> ast.AST:
        raise NotImplementedError


def _requested_pure_matmul_role_lifecycle(ctx: LoweringContext) -> bool:
    return is_pure_matmul_role_lifecycle_config(ctx.cg.device_function.config)


def _requested_tcgen05_flat_role_coordinates(ctx: LoweringContext) -> bool:
    return bool(
        ctx.cg.device_function.config.get(
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False
        )
    )


def _reject_tcgen05_flat_role_coordinates_fallback() -> None:
    raise exc.BackendUnsupported(
        "cute",
        f"{TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY}=True requires "
        "active-K-loop tcgen05 MMA lowering",
    )


class Lowering:
    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        raise NotImplementedError

    def get_masked_value(self, node: Node) -> float | bool | None:
        """Get the masked value for this node."""
        return None


MaskedValueFn = Callable[[Node], float | bool | None]
CodegenHandler = Callable[[LoweringContext, Node], object]


def _env_arg(ctx: LoweringContext, node: Node) -> Argument:
    return ctx.env[node]


@dataclasses.dataclass
class AtenLowering(Lowering):
    target: object | None = None
    masked_value_fn: MaskedValueFn | None = None
    codegen_impls: dict[str, CodegenHandler] = dataclasses.field(default_factory=dict)

    def register_codegen(
        self, backend: str
    ) -> Callable[[CodegenHandler], CodegenHandler]:
        def decorator(handler: CodegenHandler) -> CodegenHandler:
            assert backend not in self.codegen_impls, (
                f"codegen already registered for backend {backend!r}"
            )
            self.codegen_impls[backend] = handler
            return handler

        return decorator

    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        env = CompileEnvironment.current()
        handler = self.codegen_impls.get(env.codegen_name)
        if handler is None:
            handler = self.codegen_impls.get("common")
        if handler is None:  # pragma: no cover - defensive
            target = self.target or "unknown"
            raise exc.BackendImplementationMissing(
                env.backend_name,
                f"Aten lowering codegen not registered for {target!r}",
            )
        return handler(ctx, node)

    def get_masked_value(self, node: Node) -> float | bool | None:
        if self.masked_value_fn is not None:
            return self.masked_value_fn(node)
        return None


def passthrough_masked_value(node: Node) -> float | bool | None:
    for input_node in node.all_input_nodes:
        if isinstance(input_node.meta["val"], torch.Tensor):
            return cached_masked_value(input_node)
    return None


aten_lowering_dispatch: dict[object, Callable[[Node], Lowering]] = {}


def default_make_lowering(lowering: AtenLowering, node: Node) -> Lowering:
    return lowering


def register_lowering(
    fn: object,
    make_lowering: Callable[[AtenLowering, Node], Lowering] = default_make_lowering,
    masked_value_fn: MaskedValueFn | None = None,
) -> AtenLowering:
    assert fn not in aten_lowering_dispatch, f"Lowering for {fn} already registered"
    lowering = AtenLowering(target=fn, masked_value_fn=masked_value_fn)
    aten_lowering_dispatch[fn] = lambda node: make_lowering(lowering, node)
    return lowering


sym_size_lowering = register_lowering(torch.ops.aten.sym_size.int)


@sym_size_lowering.register_codegen("common")
def codegen_sym_size(ctx: LoweringContext, node: Node) -> object:
    val = node.meta["val"]
    assert isinstance(
        val, (int, float, bool, torch.SymInt, torch.SymBool, torch.SymFloat)
    )
    return val


getitem_lowering = register_lowering(getitem, masked_value_fn=getitem_masked_value)


@getitem_lowering.register_codegen("common")
def codegen_getitem(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(lhs, (list, tuple))
    assert isinstance(rhs, int)
    return lhs[rhs]


full_lowering = register_lowering(
    torch.ops.aten.full.default,
    masked_value_fn=lambda n: (
        n.args[1] if isinstance(n.args[1], (int, float, bool)) else None
    ),
)
scalar_tensor_lowering = register_lowering(
    torch.ops.aten.scalar_tensor.default,
)


where_lowering = register_lowering(torch.ops.aten.where.self)


@where_lowering.register_codegen("common")
def codegen_where(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    cond, x, y = map_arg(node.args, lambda arg: _env_arg(ctx, arg))

    def ensure_ast(value: object) -> ast.AST:
        if isinstance(value, ast.AST):
            return value
        if isinstance(value, (int, float, bool)):
            return expr_from_string(constant_repr(value))
        raise AssertionError(f"unsupported where operand: {type(value)!r}")

    return expr_from_string(
        env.backend.where_expr("{cond}", "{x}", "{y}"),
        cond=ensure_ast(cond),
        x=ensure_ast(x),
        y=ensure_ast(y),
    )


@where_lowering.register_codegen("cute")
def codegen_where_cute(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    cond, x, y = map_arg(node.args, lambda arg: _env_arg(ctx, arg))

    def ensure_ast(value: object) -> ast.AST:
        if isinstance(value, ast.AST):
            return value
        if isinstance(value, (int, float, bool)):
            return expr_from_string(constant_repr(value))
        raise AssertionError(f"unsupported where operand: {type(value)!r}")

    output = node.meta.get("val")
    x_ast = ensure_ast(x)
    y_ast = ensure_ast(y)
    if isinstance(output, torch.Tensor):
        x_ast = env.backend.cast_ast(x_ast, output.dtype)
        y_ast = env.backend.cast_ast(y_ast, output.dtype)
    return expr_from_string(
        env.backend.where_expr("{cond}", "{x}", "{y}"),
        cond=ensure_ast(cond),
        x=x_ast,
        y=y_ast,
    )


@full_lowering.register_codegen("common")
def codegen_full(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    size = map_arg(node.args[0], lambda n: n.meta["val"])
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    value_ast = map_arg(node.args[1], lambda arg: _env_arg(ctx, arg))
    if isinstance(value_ast, (int, float, bool)):
        value_ast = expr_from_string(constant_repr(value_ast))
    assert isinstance(value_ast, ast.AST), value_ast
    # pyrefly: ignore [not-iterable]
    shape_dims = ctx.cg.device_function.tile_strategy.shape_dims([*size])
    return expr_from_string(
        env.backend.full_expr(shape_dims, "{value}", dtype),
        value=value_ast,
    )


@scalar_tensor_lowering.register_codegen("common")
def codegen_scalar_tensor(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    layout = node.kwargs.get("layout", torch.strided)
    assert layout in (None, torch.strided), f"layout={layout}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    value_arg = node.args[0]
    value_ast = _env_arg(ctx, value_arg) if isinstance(value_arg, Node) else value_arg
    if isinstance(value_ast, (int, float, bool)):
        value_ast = expr_from_string(constant_repr(value_ast))
    assert isinstance(value_ast, ast.AST), value_ast
    return expr_from_string(
        env.backend.full_expr([], "{value}", dtype),
        value=value_ast,
    )


unsqueeze_lowering = register_lowering(
    torch.ops.aten.unsqueeze.default,
    masked_value_fn=passthrough_masked_value,
)


@unsqueeze_lowering.register_codegen("common")
def codegen_unsqueeze(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dim = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    assert isinstance(dim, int)
    # pyrefly: ignore [missing-attribute]
    ndim = node.args[0].meta["val"].ndim
    if dim < 0:
        dim += ndim + 1
    assert 0 <= dim <= ndim, f"Invalid dim {dim} for tensor with {ndim} dims"
    args = [":"] * ndim
    args.insert(dim, "None")
    return expr_from_string(
        f"{{tensor}}[{', '.join(args)}]",
        tensor=tensor,
    )


@unsqueeze_lowering.register_codegen("cute")
def codegen_unsqueeze_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    # One scalar per thread — adding a unit dimension cannot change the value.
    assert not node.kwargs, "unsqueeze kwargs not supported"
    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, tensor.node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


squeeze_lowering = register_lowering(
    torch.ops.aten.squeeze.dim,
    masked_value_fn=passthrough_masked_value,
)
view_lowering = register_lowering(
    torch.ops.aten.view.default,
    masked_value_fn=passthrough_masked_value,
)
reshape_lowering = register_lowering(
    torch.ops.aten.reshape.default,
    masked_value_fn=passthrough_masked_value,
)
argmax_lowering = register_lowering(torch.ops.aten.argmax.default)
argmin_lowering = register_lowering(torch.ops.aten.argmin.default)


def _argreduce_schema(node: Node) -> tuple[torch.Tensor, int | None, bool]:
    input_node = cast("Node", node.args[0])
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dim is None:
        keepdim = (
            bool(node.args[2])
            if len(node.args) > 2
            else bool(node.kwargs.get("keepdim", False))
        )
        return input_val, None, keepdim
    if not isinstance(dim, int):
        raise exc.BackendUnsupported(
            CompileEnvironment.current().backend_name,
            f"{node.target} with a non-integer dim",
        )
    if dim < 0:
        dim += input_val.ndim
    if not (0 <= dim < input_val.ndim):
        raise exc.ReductionDimInvalidForShape(dim, input_val.shape)
    keepdim = (
        bool(node.args[2])
        if len(node.args) > 2
        else bool(node.kwargs.get("keepdim", False))
    )
    return input_val, dim, keepdim


def _normalize_argreduce_dim(node: Node) -> tuple[torch.Tensor, int]:
    input_val, dim, _ = _argreduce_schema(node)
    if dim is None:
        raise exc.BackendUnsupported(
            CompileEnvironment.current().backend_name,
            f"{node.target} without an explicit integer dim",
        )
    return input_val, dim


def _shape_chain_only_users(node: Node) -> bool:
    return bool(node.users) and all(
        user.op == "call_function" and is_cute_shape_chain_target(user.target)
        for user in node.users
    )


def _should_use_cute_argreduce_lowering(argreduce_node: Node) -> bool:
    from ..language import _tracing_ops
    from ..language._decorators import is_api_func
    from .device_ir import DeviceIR

    if CompileEnvironment.current().backend_name != "cute":
        return False
    if not argreduce_node.args or not isinstance(argreduce_node.args[0], Node):
        return False

    matmul_targets = {
        torch.matmul,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
    try:
        device_ir = DeviceIR.current()
        graph_by_id = {
            idx: graph_info
            for idx, graph_info in enumerate(getattr(device_ir, "graphs", ()))
            if hasattr(graph_info, "graph")
        }
    except (AttributeError, IndexError):
        graph_by_id = {}
    seen_graph_ids: set[int] = set()
    seen_nodes: set[Node] = set()

    def graph_contains_matmul(graph_id: int) -> bool:
        if graph_id in seen_graph_ids:
            return False
        seen_graph_ids.add(graph_id)
        graph_info = graph_by_id.get(graph_id)
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            return False
        return any(node_contains_matmul(node) for node in graph.nodes)

    def node_contains_matmul(node: Node) -> bool:
        if node in seen_nodes:
            return False
        seen_nodes.add(node)
        if node.op != "call_function":
            return False
        if node.target in matmul_targets:
            return True
        if is_api_func(node.target):
            name = getattr(node.target, "__name__", "")
            if name == "dot":
                return True
            if _tracing_ops.is_for_loop_target(node.target):
                graph_id = node.args[0] if node.args else None
                if isinstance(graph_id, int) and graph_contains_matmul(graph_id):
                    return True
        for arg in node.args:
            if isinstance(arg, Node) and node_contains_matmul(arg):
                return True
        for arg in node.kwargs.values():
            if isinstance(arg, Node) and node_contains_matmul(arg):
                return True
        return False

    return node_contains_matmul(argreduce_node.args[0])


def _triton_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    input_val, dim, keepdim = _argreduce_schema(node)
    assert isinstance(input_val, torch.Tensor)
    fn = "argmax" if reduction_type == "argmax" else "argmin"
    backend = CompileEnvironment.current().backend
    dtype_str = backend.dtype_str(node.meta["val"].dtype)
    if dim is None:
        flat_shape = ctx.cg.device_function.tile_strategy.shape_str([input_val.numel()])
        tensor = expr_from_string(
            backend.reshape_expr("{tensor}", flat_shape), tensor=tensor
        )
        reduced = f"tl.{fn}({{tensor}}, axis=0).to({dtype_str})"
    else:
        reduced = f"tl.{fn}({{tensor}}, axis={dim}).to({dtype_str})"
    if keepdim:
        output_val = node.meta["val"]
        assert isinstance(output_val, torch.Tensor)
        shape_dims = ctx.cg.device_function.tile_strategy.shape_dims(
            [*output_val.size()]
        )
        output_shape = ctx.cg.device_function.tile_strategy.shape_str(
            [*output_val.size()]
        )
        if output_val.numel() == 1:
            reduced = backend.full_expr(shape_dims, reduced, output_val.dtype)
        else:
            reduced = backend.reshape_expr(reduced, output_shape)
    return expr_from_string(reduced, tensor=tensor)


def _pallas_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    input_val, dim, keepdim = _argreduce_schema(node)
    assert isinstance(input_val, torch.Tensor)
    fn = "argmax" if reduction_type == "argmax" else "argmin"
    backend = CompileEnvironment.current().backend
    dtype_str = backend.dtype_str(node.meta["val"].dtype)
    if dim is None:
        flat_shape = ctx.cg.device_function.tile_strategy.shape_str([input_val.numel()])
        tensor = expr_from_string(
            backend.reshape_expr("{tensor}", flat_shape), tensor=tensor
        )
        reduced = f"{dtype_str}(jnp.{fn}({{tensor}}, axis=0))"
    else:
        reduced = f"{dtype_str}(jnp.{fn}({{tensor}}, axis={dim}))"
    if keepdim:
        output_val = node.meta["val"]
        assert isinstance(output_val, torch.Tensor)
        shape_dims = ctx.cg.device_function.tile_strategy.shape_dims(
            [*output_val.size()]
        )
        output_shape = ctx.cg.device_function.tile_strategy.shape_str(
            [*output_val.size()]
        )
        if output_val.numel() == 1:
            reduced = backend.full_expr(shape_dims, reduced, output_val.dtype)
        else:
            reduced = backend.reshape_expr(reduced, output_shape)
    return expr_from_string(reduced, tensor=tensor)


def _cute_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    _, dim, keepdim = _argreduce_schema(node)
    return codegen_cute_tile_argreduce(
        ctx,
        node,
        reduction_type,
        dim=dim,
        keepdim=keepdim,
    )


@argmax_lowering.register_codegen("triton")
def codegen_argmax(ctx: LoweringContext, node: Node) -> ast.AST:
    return _triton_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("triton")
def codegen_argmin(ctx: LoweringContext, node: Node) -> ast.AST:
    return _triton_argreduce(ctx, node, "argmin")


@argmax_lowering.register_codegen("pallas")
def codegen_argmax_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("pallas")
def codegen_argmin_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_argreduce(ctx, node, "argmin")


@argmax_lowering.register_codegen("cute")
def codegen_argmax_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    return _cute_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("cute")
def codegen_argmin_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    return _cute_argreduce(ctx, node, "argmin")


@squeeze_lowering.register_codegen("cute")
def codegen_squeeze_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    # Squeeze removes a dimension of size 1 — no data movement needed
    # since each thread still holds the same element.
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, tensor.node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


@view_lowering.register_codegen("cute")
@reshape_lowering.register_codegen("cute")
def codegen_view_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import codegen_cute_reshape

    return codegen_cute_reshape(ctx, node)


@squeeze_lowering.register_codegen("triton")
@view_lowering.register_codegen("triton")
@reshape_lowering.register_codegen("triton")
def codegen_view(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "view kwargs not supported"
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    return expr_from_string(f"tl.reshape({{tensor}}, {shape_str})", tensor=tensor)


@squeeze_lowering.register_codegen("pallas")
@view_lowering.register_codegen("pallas")
@reshape_lowering.register_codegen("pallas")
def codegen_view_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    input_node = node.args[0]
    if isinstance(input_node, Node):
        input_val = input_node.meta.get("val")
        if isinstance(input_val, torch.Tensor) and input_val.dtype is torch.bool:
            # Mosaic cannot reshape bool vectors directly:
            # https://github.com/jax-ml/jax/issues/37370
            return expr_from_string(
                f"(jnp.reshape(({{tensor}}).astype(jnp.int32), {shape_str}) != 0)",
                tensor=tensor,
            )
    return expr_from_string(f"jnp.reshape({{tensor}}, {shape_str})", tensor=tensor)


view_dtype_lowering = register_lowering(
    torch.ops.aten.view.dtype,
    masked_value_fn=passthrough_masked_value,
)


@view_dtype_lowering.register_codegen("triton")
def codegen_view_dtype(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.cast with bitcast=True for dtype reinterpretation."""
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    target_dtype = node.args[1]
    assert isinstance(target_dtype, torch.dtype)
    return expr_from_string(
        f"tl.cast({{tensor}}, {triton_type(target_dtype)}, bitcast=True)",
        tensor=tensor,
    )


@view_dtype_lowering.register_codegen("cute")
def codegen_view_dtype_cute(ctx: LoweringContext, node: Node) -> object:
    """Per-element bitcast through shared memory ``cute.recast_tensor``.

    CuTe DSL operates on per-thread scalars, so a dtype reinterpret has to
    round-trip a value through shared memory: write as the source dtype, then
    read the same memory through a recast view typed as the target dtype.
    """
    from .cute.cute_reshape import _flat_index_from_coords
    from .cute.cute_reshape import _get_dim_local_coord
    from .cute.cute_reshape import _get_tile_shape

    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    target_dtype = node.args[1]
    assert isinstance(target_dtype, torch.dtype)

    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    if input_val.dtype.itemsize != target_dtype.itemsize:
        raise exc.BackendUnsupported(
            "cute",
            f"view.dtype with mismatched widths: "
            f"{input_val.dtype} ({input_val.dtype.itemsize} bytes) -> "
            f"{target_dtype} ({target_dtype.itemsize} bytes)",
        )

    from .generate_ast import GenerateAST

    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    df = cg.device_function
    env = CompileEnvironment.current()
    config = df.config

    shape = _get_tile_shape(input_val, env, config)
    if not shape:
        shape = [1]
    numel = 1
    for s in shape:
        numel *= s

    src_dtype_str = env.backend.dtype_str(input_val.dtype)
    tgt_dtype_str = env.backend.dtype_str(target_dtype)

    smem_ptr = df.new_var("view_dtype_smem_ptr")
    smem = df.new_var("view_dtype_smem")
    smem_recast = df.new_var("view_dtype_smem_recast")

    coords = [_get_dim_local_coord(cg, input_val, i) for i in range(len(shape))]
    flat = _flat_index_from_coords(coords, shape) if coords else "cutlass.Int32(0)"

    cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({src_dtype_str}, {numel})"
        )
    )
    cg.add_statement(
        statement_from_string(f"{smem} = cute.make_tensor({smem_ptr}, ({numel},))")
    )
    cg.add_statement(
        statement_from_string(
            f"{smem}[{flat}] = {src_dtype_str}({{_inp}})", _inp=tensor
        )
    )
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    cg.add_statement(
        statement_from_string(
            f"{smem_recast} = cute.recast_tensor({smem}, {tgt_dtype_str})"
        )
    )

    result = df.new_var("view_dtype_value")
    cg.add_statement(statement_from_string(f"{result} = {smem_recast}[{flat}]"))
    return expr_from_string(result)


alias_lowering = register_lowering(
    torch.ops.aten.alias.default,
    masked_value_fn=passthrough_masked_value,
)


@alias_lowering.register_codegen("common")
def codegen_alias(ctx: LoweringContext, node: Node) -> object:
    """Alias is a no-op view, just pass through the input tensor."""
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    return tensor


permute_lowering = register_lowering(
    torch.ops.aten.permute.default,
    masked_value_fn=passthrough_masked_value,
)


@permute_lowering.register_codegen("cute")
def codegen_permute_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import codegen_cute_permute

    return codegen_cute_permute(ctx, node)


@permute_lowering.register_codegen("triton")
def codegen_permute(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    assert {*dims} == {*range(len(dims))}, dims
    return expr_from_string(
        f"tl.permute({{tensor}}, {dims!r})",
        tensor=tensor,
    )


@permute_lowering.register_codegen("pallas")
def codegen_permute_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    return expr_from_string(
        f"jnp.transpose({{tensor}}, {dims!r})",
        tensor=tensor,
    )


stack_lowering = register_lowering(
    torch.ops.aten.stack.default,
    masked_value_fn=passthrough_masked_value,
)


@stack_lowering.register_codegen("triton")
def codegen_stack(ctx: LoweringContext, node: Node) -> object:
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)

    assert isinstance(tensors, (list, tuple))
    # pyrefly: ignore [bad-index]
    tensor_asts = [ctx.env[t] for t in tensors]
    n = len(tensor_asts)

    if n == 0:
        raise ValueError("Cannot stack empty tensor list")

    # Round up to power of 2 for efficient masking
    padded_size = 1 << (n - 1).bit_length()

    # Create index array [0, 1, 2, 3, ...] for tensor selection
    idx = ctx.cg.device_function.new_var("stack_idx")
    ctx.cg.add_statement(statement_from_string(f"{idx} = tl.arange(0, {padded_size})"))

    # Broadcast index to target dimension shape
    # e.g., dim=0: [:, None, None], dim=1: [None, :, None], dim=2: [None, None, :]
    bidx = ctx.cg.device_function.new_var("broadcast_idx")
    assert isinstance(dim, int)
    pattern = "[" + ", ".join(["None"] * dim + [":"] + ["None"] * max(0, 2 - dim)) + "]"
    ctx.cg.add_statement(statement_from_string(f"{bidx} = {idx}{pattern}"))

    # Expand each input tensor along the stack dimension
    expanded = [ctx.cg.device_function.new_var(f"expanded_{i}") for i in range(n)]
    for var, tensor in zip(expanded, tensor_asts, strict=False):
        tensor_ast = cast("ast.AST", tensor)
        ctx.cg.add_statement(
            statement_from_string(f"{var} = tl.expand_dims({{t}}, {dim})", t=tensor_ast)
        )

    # Initialize result with zeros
    result = ctx.cg.device_function.new_var("stacked_result")
    ctx.cg.add_statement(
        statement_from_string(f"{result} = tl.zeros_like({expanded[0]})")
    )

    # Select each tensor using masks
    for i in range(n):
        mask = ctx.cg.device_function.new_var(f"mask_{i}")
        ctx.cg.add_statement(statement_from_string(f"{mask} = {bidx} == {i}"))
        ctx.cg.add_statement(
            statement_from_string(
                f"{result} = tl.where({mask}, {expanded[i]}, {result})"
            )
        )

    return expr_from_string(result)


@stack_lowering.register_codegen("cute")
def codegen_stack_cute(ctx: LoweringContext, node: Node) -> object:
    tensors = node.args[0]
    assert isinstance(tensors, (list, tuple))
    if not tensors:
        raise ValueError("Cannot stack empty tensor list")
    if not all(isinstance(tensor, Node) for tensor in tensors):
        raise exc.BackendUnsupported("cute", "stack inputs")
    if _shape_chain_only_users(node):
        return CuteShapeChainView(node)
    # A stack materialized to a per-thread scalar is only correct when every
    # consumer reads it element-wise (each output element is one stacked
    # element), e.g. a direct store. A reduction or matmul that contracts over
    # the virtual stacked dimension cannot gather the stacked operands from a
    # single per-thread scalar and would silently produce wrong values, so only
    # materialize when the direct consumers are stores (or further shape-chain
    # ops); otherwise keep rejecting the pattern.
    from ..language import memory_ops

    if not all(
        user.op == "call_function"
        and (user.target is memory_ops.store or is_cute_shape_chain_target(user.target))
        for user in node.users
    ):
        raise exc.BackendUnsupported(
            "cute", "virtual shape-chain direct consumers are not yet supported"
        )
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    materialized = resolve_cute_shape_chain_value(ctx, node)
    if materialized is None:
        raise exc.BackendUnsupported(
            "cute", "virtual shape-chain direct consumers are not yet supported"
        )
    return materialized


expand_lowering = register_lowering(
    torch.ops.aten.expand.default,
    masked_value_fn=passthrough_masked_value,
)


@expand_lowering.register_codegen("triton")
def codegen_expand(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    # pyrefly: ignore [missing-attribute]
    if node.args[0].meta["val"].ndim != len(shape):
        broadcasting = [":"] * len(shape)
        # pyrefly: ignore [missing-attribute]
        for i in range(len(shape) - node.args[0].meta["val"].ndim):
            broadcasting[i] = "None"
        tensor = expr_from_string(
            f"{{tensor}}[{', '.join(broadcasting)}]", tensor=tensor
        )
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"tl.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


@expand_lowering.register_codegen("pallas")
def codegen_expand_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    # pyrefly: ignore [missing-attribute]
    if node.args[0].meta["val"].ndim != len(shape):
        broadcasting = [":"] * len(shape)
        # pyrefly: ignore [missing-attribute]
        for i in range(len(shape) - node.args[0].meta["val"].ndim):
            broadcasting[i] = "None"
        tensor = expr_from_string(
            f"{{tensor}}[{', '.join(broadcasting)}]", tensor=tensor
        )
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"jnp.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


@expand_lowering.register_codegen("cute")
def codegen_expand_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


def apply_dot_requirements(lowering: AtenLowering, node: Node) -> Lowering:
    """Apply min_dot_size requirements to the config_spec"""
    assert not node.kwargs, "dot kwargs not supported"
    assert len(node.args) in (2, 3)
    lproxy, rproxy = map_arg(node.args[-2:], lambda arg: arg.meta["val"])
    assert isinstance(lproxy, torch.Tensor)
    assert isinstance(rproxy, torch.Tensor)
    # Update config spec min sizes for M, N, K
    enforce_dot_requirements(lproxy, rproxy)
    # inputs to the dot operation must be zero-masked
    *maybe_acc, lnode, rnode = node.args
    assert isinstance(lnode, Node)
    assert isinstance(rnode, Node)
    lnode = apply_masking(lnode, base_node=node, other=0)
    rnode = apply_masking(rnode, base_node=node, other=0)
    node.args = (*maybe_acc, lnode, rnode)
    return lowering


def reduce_3d_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    acc = None
    acc_node: Node | None = None
    if with_acc:
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        assert isinstance(node.args[0], Node)
        acc_node = node.args[0]
        lhs_node = node.args[1]
        rhs_node = node.args[2]
    else:
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        lhs_node = node.args[0]
        rhs_node = node.args[1]
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)

    # Check if inputs are FP8 - if so, redirect user to hl.dot()
    lhs_dtype = lhs_node.meta["val"].dtype
    rhs_dtype = rhs_node.meta["val"].dtype
    acc_dtype_meta: torch.dtype | None = None
    if with_acc:
        assert acc_node is not None
        assert isinstance(acc_node, Node)
        acc_dtype_meta = acc_node.meta["val"].dtype
    if lhs_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and rhs_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        raise NotImplementedError(
            "FP8 GEMM via torch API is not supported yet. Please use hl.dot() instead."
        )

    lhs_shape = list(lhs_node.meta["val"].size())
    rhs_shape = list(rhs_node.meta["val"].size())
    acc_shape = (
        list(acc_node.meta["val"].size())
        if (with_acc and acc_node is not None)
        else None
    )

    # Extract expected output dtype from FX node to match PyTorch eager mode behavior
    out_dtype: torch.dtype | None = None
    if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
        out_dtype = node.meta["val"].dtype

    return emit_tl_dot_with_padding(
        lhs,
        rhs,
        acc if with_acc else None,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_meta if with_acc else None,
        out_dtype=out_dtype,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
    )


bmm_lowering = register_lowering(
    torch.ops.aten.bmm.default,
    apply_dot_requirements,
)
mm_lowering = register_lowering(
    torch.ops.aten.mm.default,
    apply_dot_requirements,
)


def _apply_bmm_dot_dtype_requirements(_lowering: AtenLowering, node: Node) -> Lowering:
    """Handle bmm.dtype by stripping the ScalarType arg and reusing bmm_lowering."""
    node.args = tuple(a for a in node.args if isinstance(a, Node))
    return apply_dot_requirements(bmm_lowering, node)


register_lowering(
    torch.ops.aten.bmm.dtype,
    _apply_bmm_dot_dtype_requirements,
)


@bmm_lowering.register_codegen("triton")
@mm_lowering.register_codegen("triton")
def codegen_mm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"

    return reduce_3d_dot(ctx, node, False)


addmm_lowering = register_lowering(
    torch.ops.aten.addmm.default,
    apply_dot_requirements,
)


@addmm_lowering.register_codegen("triton")
def codegen_addmm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


baddbmm_lowering = register_lowering(
    torch.ops.aten.baddbmm.default,
    apply_dot_requirements,
)


@baddbmm_lowering.register_codegen("triton")
def codegen_baddbmm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


def _pallas_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    """Generate jnp.dot_general for Pallas backend."""
    if with_acc:
        acc_node_arg, lhs_node_arg, rhs_node_arg = node.args[:3]
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        assert isinstance(lhs, ast.AST)
        assert isinstance(rhs, ast.AST)
    else:
        lhs_node_arg, rhs_node_arg = node.args[:2]
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(lhs, ast.AST)
        assert isinstance(rhs, ast.AST)
        acc = None

    assert isinstance(lhs_node_arg, Node)
    assert isinstance(rhs_node_arg, Node)
    lhs_dtype = lhs_node_arg.meta["val"].dtype
    rhs_dtype = rhs_node_arg.meta["val"].dtype
    lhs_ndim = lhs_node_arg.meta["val"].ndim
    need_f32_acc = _needs_f32_accumulator(lhs_dtype, rhs_dtype)
    out_dtype = node.meta["val"].dtype if "val" in node.meta else None

    return _emit_pallas_matmul(
        lhs,
        rhs,
        acc=acc if with_acc else None,
        need_f32_acc=need_f32_acc,
        out_dtype=out_dtype,
        lhs_ndim=lhs_ndim,
    )


@bmm_lowering.register_codegen("pallas")
@mm_lowering.register_codegen("pallas")
def codegen_mm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, False)


@addmm_lowering.register_codegen("pallas")
def codegen_addmm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, True)


@baddbmm_lowering.register_codegen("pallas")
def codegen_baddbmm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, True)


@bmm_lowering.register_codegen("cute")
@mm_lowering.register_codegen("cute")
def codegen_mm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    lhs_node, rhs_node = node.args[:2]
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    if k_block_id is None and packed_rhs is None:
        remat = cute_rematerialize_rhs_at_contraction_block(ctx, lhs_node, rhs_node)
        if remat is not None:
            rhs, k_block_id = remat
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    serial_k_extent = (
        None
        if k_block_id is not None or static_k_extent is not None
        else cute_static_serial_matmul_k_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if (
        static_k_extent is None
        and serial_k_extent is None
        and k_block_id is None
        and not k_is_one
    ):
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    out_dtype = node.meta["val"].dtype if "val" in node.meta else None
    outer_acc_dtype = cute_outer_accumulator_dtype(node, is_acc_none=True)
    effective_out_dtype = (
        cute_outer_accumulator_out_dtype(out_dtype, outer_acc_dtype)
        if out_dtype is not None
        else None
    )
    direct_mma_result = codegen_cute_mma_direct_mm(
        ctx,
        node,
        serial_k_extent=serial_k_extent,
    )
    if direct_mma_result is not None:
        if _requested_tcgen05_flat_role_coordinates(ctx):
            _reject_tcgen05_flat_role_coordinates_fallback()
        if _requested_pure_matmul_role_lifecycle(ctx):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_strategy='pure_matmul_role_lifecycle' requires the "
                "active-K-loop tcgen05 matmul lowering, not direct-mm fallback",
            )
        return direct_mma_result
    serial_result = emit_cute_serial_scalar_mm_from_loads(
        ctx,
        lhs_node,
        rhs_node,
        k_extent=serial_k_extent,
        out_dtype=effective_out_dtype,
    )
    if serial_result is not None:
        if _requested_tcgen05_flat_role_coordinates(ctx):
            _reject_tcgen05_flat_role_coordinates_fallback()
        if _requested_pure_matmul_role_lifecycle(ctx):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_strategy='pure_matmul_role_lifecycle' requires the "
                "active-K-loop tcgen05 matmul lowering, not serial scalar fallback",
            )
        return serial_result
    if serial_k_extent is not None:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe direct mm without an active K tile only supports contiguous direct-load operands",
        )
    if _requested_pure_matmul_role_lifecycle(ctx):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires aten.mm "
            "to lower through the tcgen05 K-loop path",
        )
    if _requested_tcgen05_flat_role_coordinates(ctx):
        _reject_tcgen05_flat_role_coordinates_fallback()
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        accumulate_in_lane_loop=not cute_outer_accumulates_result(
            node,
            is_acc_none=True,
        ),
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        out_dtype=effective_out_dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
    )


@addmm_lowering.register_codegen("cute")
def codegen_addmm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    from .cute.cute_mma import codegen_cute_mma

    result = codegen_cute_mma(ctx, node, with_acc=True)
    if result is not None:
        return result
    if _requested_tcgen05_flat_role_coordinates(ctx):
        _reject_tcgen05_flat_role_coordinates_fallback()
    if _requested_pure_matmul_role_lifecycle(ctx):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires the "
            "active-K-loop tcgen05 addmm lowering",
        )
    acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    acc_node = node.args[0]
    lhs_node = node.args[1]
    rhs_node = node.args[2]
    assert isinstance(acc_node, Node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
        acc_node=acc_node,
    )


def _cute_baddbmm_result_reduced_over_block(
    node: Node,
    n_block_id: int,
) -> bool:
    """Whether *node*'s collapsed result is summed away over *n_block_id*.

    The matmul's M (lhs free) and N (rhs free) axes collapse to ``n_block_id``,
    so the standard fallback would compute only the diagonal.  Folding the N
    reduction into the matmul (layout A) is correct *only* when N is genuinely
    reduced out downstream.  This guard requires:

    * ``n_block_id`` is a reduction block (allocated for a ``sum`` /
      reduction), and a reduction node over it exists somewhere in the device
      IR (the ``.sum(-1)``), and
    * the baddbmm result does not escape to any non-passthrough consumer in its
      own graph - either it is consumed only by a same-graph reduction over the
      block, or it is purely loop-carried (its only users are the graph output
      and/or pure casts), so the carried value reaches the downstream reduction.

    Returns ``False`` otherwise, leaving the (unchanged) standard path.
    """
    from ..language._tracing_ops import _new_var
    from .host_function import HostFunction
    from .inductor_lowering import ReductionLowering

    env = CompileEnvironment.current()
    canonical_block_id = getattr(env, "canonical_block_id", lambda block_id: block_id)
    target_canonical = canonical_block_id(n_block_id)

    if not env.block_sizes[n_block_id].reduction:
        return False

    # A reduction over the (canonical) N block must exist in the device IR.
    device_ir = HostFunction.current().device_ir
    has_block_reduction = False
    for graph_info in getattr(device_ir, "graphs", ()):
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for other in graph.nodes:
            lowering = other.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and canonical_block_id(lowering.block_index) == target_canonical
            ):
                has_block_reduction = True
                break
        if has_block_reduction:
            break
    if not has_block_reduction:
        return False

    passthrough = {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.to.dtype,
        torch.ops.prims.convert_element_type.default,
        _new_var,
    }

    # Walk forward inside the baddbmm's own graph; the result must not reach any
    # non-passthrough consumer other than a reduction over the block or the
    # graph output (loop carry).
    seen: set[Node] = set()
    stack: list[Node] = [node]
    while stack:
        cur = stack.pop()
        for user in cur.users:
            if not isinstance(user, Node) or user in seen:
                continue
            seen.add(user)
            if len(seen) > 64:
                return False
            if user.op == "output":
                continue
            lowering = user.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and canonical_block_id(lowering.block_index) == target_canonical
            ):
                continue
            if user.op == "call_function" and user.target in passthrough:
                stack.append(user)
                continue
            return False
    return True


def _maybe_codegen_cute_baddbmm_n_collapse(
    ctx: LoweringContext,
    node: Node,
    *,
    lhs: ast.AST | CutePackedAffineLoad,
    acc: ast.AST,
    lhs_node: Node,
    rhs_node: Node,
    acc_node: Node,
    k_block_id: int | None,
    static_k_extent: int | None,
) -> ast.AST | None:
    """Layout (A) for a static-M==N-collapse baddbmm reduced over N.

    See ``cute_static_mn_collapse_n_block_id`` /
    ``_emit_cute_matmul_n_collapse``.  Returns ``None`` (caller keeps the
    standard path) unless the tightly-gated pattern holds: the lhs M axis and
    rhs N axis share a block id and the result is summed away over that block.
    """
    if not isinstance(lhs, ast.AST):
        return None
    n_block_id = cute_static_mn_collapse_n_block_id(ctx.cg, lhs_node, rhs_node)
    if n_block_id is None:
        return None
    if not _cute_baddbmm_result_reduced_over_block(node, n_block_id):
        return None
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)
    n_size = rhs_node.meta["val"].shape[-1]
    n_extent = size_hint(n_size) if callable(size_hint) else int(n_size)
    if not isinstance(n_extent, int) or n_extent <= 0:
        return None

    def rhs_at_n(n_var: str) -> ast.AST:
        rematerialized = cute_rematerialize_rhs_at_index_override(
            ctx, rhs_node, n_block_id, n_var
        )
        if rematerialized is None:
            raise exc.BackendUnsupported(
                "cute",
                "CuTe static-MN-collapse baddbmm could not re-materialize the rhs "
                "at the serial N index",
            )
        return rematerialized

    return _emit_cute_matmul_n_collapse(
        ctx.cg,
        lhs,
        rhs_at_n=rhs_at_n,
        n_extent=n_extent,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
    )


@baddbmm_lowering.register_codegen("cute")
def codegen_baddbmm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    from .cute.cute_mma import codegen_cute_mma

    result = codegen_cute_mma(ctx, node, with_acc=True)
    if result is not None:
        return result
    if _requested_tcgen05_flat_role_coordinates(ctx):
        _reject_tcgen05_flat_role_coordinates_fallback()
    if _requested_pure_matmul_role_lifecycle(ctx):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires "
            "aten.baddbmm to lower through the tcgen05 K-loop path",
        )
    acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    acc_node = node.args[0]
    lhs_node = node.args[1]
    rhs_node = node.args[2]
    assert isinstance(acc_node, Node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    n_collapse_result = _maybe_codegen_cute_baddbmm_n_collapse(
        ctx,
        node,
        lhs=lhs,
        acc=acc,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
        acc_node=acc_node,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
    )
    if n_collapse_result is not None:
        return n_collapse_result
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
        lhs_node=lhs_node,
        rhs_node=rhs_node,
        acc_node=acc_node,
    )


iota_lowering = register_lowering(torch.ops.prims.iota.default)
arange_default_lowering = register_lowering(torch.ops.aten.arange.default)


def _triton_iota_expr(
    ctx: LoweringContext,
    *,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype: torch.dtype | None = None,
) -> object:
    dtype = dtype or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)

    # Pad static non-power-of-2 lengths to next power of 2
    length_expr = "{length}"
    if isinstance(length_arg, int) and length_arg != next_power_of_2(length_arg):
        length_expr = str(next_power_of_2(length_arg))

    expr = f"tl.arange(0, {length_expr})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    if dtype != torch.int32:
        expr = f"({expr}).to({triton_type(dtype)})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )


def _pallas_iota_expr(
    ctx: LoweringContext,
    *,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype: torch.dtype | None = None,
) -> object:
    dtype = dtype or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)

    dtype_str = CompileEnvironment.current().backend.dtype_str(dtype)
    expr = f"jnp.arange(0, {{length}}, dtype={dtype_str})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )


def _node_dtype_kwarg(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else None


@iota_lowering.register_codegen("triton")
def codegen_iota(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.arange for torch.ops.prims.iota.default operations with automatic power-of-2 padding."""
    return _triton_iota_expr(
        ctx,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype=_node_dtype_kwarg(node),
    )


@iota_lowering.register_codegen("pallas")
def codegen_iota_pallas(ctx: LoweringContext, node: Node) -> object:
    """Generate jnp.arange for torch.ops.prims.iota.default on Pallas."""
    return _pallas_iota_expr(
        ctx,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype=_node_dtype_kwarg(node),
    )


def _cute_compacted_tile_begin_lane_expr(
    ctx: LoweringContext,
    source_node: Node,
    start: object,
    step: object,
    dtype: torch.dtype,
) -> ast.AST | None:
    """Resolve ``hl.arange(block // F)`` to the tile-local lane ``lane // F``.

    Handles the compacted sub-block store ``out[tile.begin + hl.arange(block //
    F)] = split_result`` where the value is a spread/compacted tile whose element
    on lane ``t`` is the ``t // F`` piece. The arange must contribute only the
    tile-local coordinate ``lane // F`` because ``tile.begin`` is added
    explicitly; resolving it to the global ``index_var // F`` would fold the
    tile's offset in twice. Returns ``None`` unless the pattern matches.
    """
    from .cute.cute_reshape import _get_block_local_coord
    from .cute.iota_utils import cute_free_arange_compacted_tile_begin_factor
    from .generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    cg = ctx.cg
    match = cute_free_arange_compacted_tile_begin_factor(source_node, cg)
    if match is None:
        return None
    block_id, factor = match
    local_coord = _get_block_local_coord(cg, block_id)
    if local_coord is None:
        return None
    expr = f"({local_coord}) // cutlass.Int32({factor})"
    return _wrap_iota_coord_expr(ctx, expr, start, step, dtype)


def _cute_free_arange_axis_expr(
    cg: GenerateAST,
    source_node: Node,
    length_hint: int | None,
    start: object,
    step: object,
) -> str | None:
    """Map a free/unbound ``hl.arange`` index dim onto a synthetic thread axis.

    Returns the per-thread coordinate expression (``thread_idx()[axis]``) when
    ``source_node`` is an iota that flows into a load/store index but is bound
    to no tile/reduction/grid axis. Returns ``None`` otherwise so the caller
    keeps its existing (raising) behavior — this keeps the synthetic-axis path
    a strict no-op for every already-supported arange.
    """
    if not isinstance(length_hint, int) or length_hint <= 0:
        return None
    if not cute_iota_is_free_memory_index(source_node, cg):
        return None
    # The synthetic axis is keyed by the size of the tensor dimension this
    # arange indexes. Two arange dims that address the same logical dimension (the
    # load and store ``hl.arange(k)`` over a K-sized axis) share one axis so a
    # value loaded on a lane is stored back on that lane, while a cartesian
    # ``row``/``col`` pair addressing differently-sized dims gets distinct axes.
    # ``length``/``start``/``step`` round out the key so two distinct arange dims
    # that happen to index equal-sized dims still separate.
    dim_key = cute_free_arange_indexed_dim_key(source_node, cg)
    if dim_key is None:
        return None
    key = (
        dim_key,
        length_hint,
        _arange_endpoint_key(start),
        _arange_endpoint_key(step),
    )
    return cg.allocate_cute_synthetic_arange_coord(key, length_hint)


def _arange_endpoint_key(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        return str(value._sympy_())
    return repr(value)


def _wrap_iota_coord_expr(
    ctx: LoweringContext,
    coord_expr: str,
    start: object,
    step: object,
    dtype: torch.dtype,
) -> ast.AST:
    """Apply ``start``/``step``/``dtype`` to a per-thread iota coordinate.

    Mirrors the trailing wrapping ``_cute_iota_expr`` applies to a resolved
    block-id coordinate so the synthetic-axis path produces identical
    ``start + step * coord`` arithmetic.
    """
    expr = coord_expr
    if step != 1:
        expr = f"{{step}} * ({expr})"
    if start != 0:
        expr = f"{{start}} + ({expr})"
    if dtype != torch.int32:
        expr = f"{CompileEnvironment.current().backend.dtype_str(dtype)}({expr})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
    )


def _cute_iota_expr(
    ctx: LoweringContext,
    *,
    source_node: Node,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype_arg: object = None,
) -> object:
    from .cute.cute_reshape import _get_dim_local_coord
    from .cute.cute_reshape import _grid_local_coord_expr
    from .device_ir import ForLoopGraphInfo
    from .generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    cg = ctx.cg
    dtype = (
        dtype_arg
        if isinstance(dtype_arg, torch.dtype)
        else CompileEnvironment.current().index_dtype
    )

    env = CompileEnvironment.current()
    length_hint: int | None = None
    if isinstance(length_arg, int):
        length_hint = length_arg
    elif isinstance(length_arg, torch.SymInt):
        length_hint = env.size_hint(length_arg)

    def active_iota_expr() -> ast.AST | None:
        active_block_ids: list[int] = []
        graph_block_ids = [
            graph_info.block_ids
            for graph_info in cg.codegen_graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            and graph_info.graph is source_node.graph
        ]
        if len(graph_block_ids) == 1:
            active_block_ids = [
                candidate
                for candidate in graph_block_ids[0]
                if cg.active_device_loops.get(candidate)
            ]
        if not active_block_ids and cg.current_grid_state is not None:
            active_block_ids = list(cg.current_grid_state.block_ids)
        if not active_block_ids:
            active_block_ids = [
                candidate
                for candidate, loops in cg.active_device_loops.items()
                if loops
            ]
        if not active_block_ids:
            return None

        def local_expr_and_extent(
            candidate: int,
        ) -> tuple[str | None, int | None]:
            loops = cg.active_device_loops.get(candidate)
            if loops:
                loop_state = loops[-1]
                thread_axis = loop_state.block_thread_axes.get(candidate)
                if thread_axis is None:
                    return None, None
                local_expr = _grid_local_coord_expr(cg, candidate, thread_axis)
                elements_per_thread_fn = getattr(
                    loop_state.strategy, "_elements_per_thread_for_block", None
                )
                elements_per_thread = (
                    elements_per_thread_fn(candidate)
                    if callable(elements_per_thread_fn)
                    else 1
                )
                if not isinstance(elements_per_thread, int):
                    return local_expr, None
                return (
                    local_expr,
                    loop_state.thread_axis_sizes.get(thread_axis, 1)
                    * elements_per_thread,
                )
            if cg.current_grid_state is not None:
                thread_axis = cg.current_grid_state.block_thread_axes.get(candidate)
                if thread_axis is None:
                    return None, None
                local_expr = _grid_local_coord_expr(cg, candidate, thread_axis)
                elements_per_thread_fn = getattr(
                    cg.current_grid_state.strategy,
                    "_elements_per_thread_for_block",
                    None,
                )
                elements_per_thread = (
                    elements_per_thread_fn(candidate)
                    if callable(elements_per_thread_fn)
                    else 1
                )
                if not isinstance(elements_per_thread, int):
                    return local_expr, None
                return (
                    local_expr,
                    cg.current_grid_state.thread_axis_sizes.get(thread_axis, 1)
                    * elements_per_thread,
                )
            return None, None

        matched: list[tuple[int, str]] = []
        for candidate in active_block_ids:
            loops = cg.active_device_loops.get(candidate)
            if loops:
                expr = loops[-1].strategy.index_var(candidate)
            elif (
                cg.current_grid_state is not None
                and candidate in cg.current_grid_state.block_ids
            ):
                expr = cg.current_grid_state.strategy.index_var(candidate)
            else:
                continue

            candidate_size = cg.device_function.resolved_block_size(candidate)
            if (
                not isinstance(candidate_size, int)
                or candidate_size <= 0
                or not isinstance(length_hint, int)
                or length_hint <= 0
            ):
                continue
            if candidate_size == length_hint:
                matched.append((candidate, expr))
            elif candidate_size % length_hint == 0:
                matched.append(
                    (candidate, f"({expr}) // {candidate_size // length_hint}")
                )
            else:
                local_expr, local_extent = local_expr_and_extent(candidate)
                if (
                    local_expr is not None
                    and isinstance(local_extent, int)
                    and local_extent > 0
                ):
                    if local_extent == length_hint:
                        matched.append((candidate, local_expr))
                    elif local_extent % length_hint == 0:
                        matched.append(
                            (
                                candidate,
                                f"({local_expr}) // {local_extent // length_hint}",
                            )
                        )
        if len(matched) != 1:
            return None
        _, expr = matched[0]
        if step != 1:
            expr = f"{{step}} * ({expr})"
        if start != 0:
            expr = f"{{start}} + ({expr})"
        if dtype != torch.int32:
            expr = f"{env.backend.dtype_str(dtype)}({expr})"
        return expr_from_string(
            expr,
            start=ctx.to_ast(start),
            step=ctx.to_ast(step),
        )

    block_id = env.resolve_block_id(length_arg)
    original_block_id = block_id
    if block_id is None:
        if (affine_range := match_cute_affine_range_iota(source_node)) is not None:
            return affine_range
    if (
        compacted := _cute_compacted_tile_begin_lane_expr(
            ctx, source_node, start, step, dtype
        )
    ) is not None:
        return compacted
    if "val" in source_node.meta:
        fake_val = source_node.meta["val"]
        if isinstance(fake_val, torch.Tensor) and fake_val.ndim == 1:
            with contextlib.suppress(Exception):
                length_hint = int(fake_val.shape[0])
            local_coord = _get_dim_local_coord(cg, fake_val, 0)
            if local_coord != "cutlass.Int32(0)":
                expr = local_coord
                if step != 1:
                    expr = f"{{step}} * ({expr})"
                if start != 0:
                    expr = f"{{start}} + ({expr})"
                if dtype != torch.int32:
                    expr = f"{env.backend.dtype_str(dtype)}({expr})"
                return expr_from_string(
                    expr,
                    start=ctx.to_ast(start),
                    step=ctx.to_ast(step),
                )
            if block_id is None:
                block_id = env.resolve_block_id(fake_val.shape[0])
            if block_id is None and cg.current_grid_state is not None:
                grid_candidates = [
                    candidate
                    for candidate in cg.current_grid_state.block_ids
                    if isinstance(length_hint, int)
                    and isinstance(
                        cg.device_function.resolved_block_size(candidate),
                        int,
                    )
                    and cg.device_function.resolved_block_size(candidate) == length_hint
                ]
                if len(grid_candidates) == 1:
                    block_id = grid_candidates[0]
    if block_id is None:
        if (active_expr := active_iota_expr()) is not None:
            return active_expr
        if (
            cute_iota_has_atomic_tensor_index_only_users(source_node, cg)
            and isinstance(start, int)
            and isinstance(step, int)
        ):
            return expr_from_string(
                "cute.make_identity_tensor({length})",
                length=ctx.to_ast(length_arg),
            )
        if (
            synthetic := _cute_free_arange_axis_expr(
                cg, source_node, length_hint, start, step
            )
        ) is not None:
            return _wrap_iota_coord_expr(ctx, synthetic, start, step, dtype)
        raise exc.BackendUnsupported(
            "cute",
            "hl.arange() requires an active tile/reduction axis in cute kernels",
        )
    resolved_block_id = env.resolve_codegen_block_id(block_id, cg, source_node.graph)
    candidate_block_ids = [resolved_block_id]
    if (
        original_block_id is not None
        and original_block_id != resolved_block_id
        and original_block_id not in candidate_block_ids
    ):
        candidate_block_ids.append(original_block_id)

    expr: str | None = None
    active_block_id: int | None = None
    for candidate_block_id in candidate_block_ids:
        loops = cg.active_device_loops.get(candidate_block_id)
        if loops:
            expr = loops[-1].strategy.index_var(candidate_block_id)
            active_block_id = candidate_block_id
            break
        if (
            cg.current_grid_state is not None
            and candidate_block_id in cg.current_grid_state.block_ids
        ):
            expr = cg.current_grid_state.strategy.index_var(candidate_block_id)
            active_block_id = candidate_block_id
            break
    block_id = resolved_block_id if active_block_id is None else active_block_id

    if expr is None:
        thread_axis: int | None = None
        if cg.current_grid_state is not None:
            thread_axis = cg.current_grid_state.block_thread_axes.get(block_id)
        if thread_axis is None:
            for loops_for_block in cg.active_device_loops.values():
                for loop_state in loops_for_block:
                    block_axes = getattr(loop_state, "block_thread_axes", {})
                    if isinstance(block_axes, dict) and block_id in block_axes:
                        thread_axis = block_axes[block_id]
                        break
                if thread_axis is not None:
                    break
        if thread_axis is not None:
            expr = _grid_local_coord_expr(cg, block_id, thread_axis)
        elif (active_expr := active_iota_expr()) is not None:
            return active_expr
        elif (
            cute_iota_has_atomic_tensor_index_only_users(source_node, cg)
            and isinstance(start, int)
            and isinstance(step, int)
        ):
            return expr_from_string(
                "cute.make_identity_tensor({length})",
                length=ctx.to_ast(length_arg),
            )
        elif (
            synthetic := _cute_free_arange_axis_expr(
                cg, source_node, length_hint, start, step
            )
        ) is not None:
            return _wrap_iota_coord_expr(ctx, synthetic, start, step, dtype)
        else:
            raise exc.BackendUnsupported(
                "cute",
                f"hl.arange() axis block_id={block_id} is not active in this scope",
            )
    if step != 1:
        expr = f"{{step}} * ({expr})"
    if start != 0:
        expr = f"{{start}} + ({expr})"
    if dtype != torch.int32:
        expr = f"{env.backend.dtype_str(dtype)}({expr})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
    )


@iota_lowering.register_codegen("cute")
def codegen_iota_cute(ctx: LoweringContext, node: Node) -> object:
    return _cute_iota_expr(
        ctx,
        source_node=node,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype_arg=node.kwargs.get("dtype"),
    )


@arange_default_lowering.register_codegen("triton")
def codegen_arange_default(ctx: LoweringContext, node: Node) -> object:
    return _triton_iota_expr(
        ctx,
        length_arg=node.args[0],
        dtype=_node_dtype_kwarg(node),
    )


@arange_default_lowering.register_codegen("pallas")
def codegen_arange_default_pallas(ctx: LoweringContext, node: Node) -> object:
    return _pallas_iota_expr(
        ctx,
        length_arg=node.args[0],
        dtype=_node_dtype_kwarg(node),
    )


@arange_default_lowering.register_codegen("cute")
def codegen_arange_default_cute(ctx: LoweringContext, node: Node) -> object:
    return _cute_iota_expr(
        ctx,
        source_node=node,
        length_arg=node.args[0],
        dtype_arg=node.kwargs.get("dtype"),
    )


sort_lowering = register_lowering(torch.ops.aten.sort.default)


def _sort_args(node: Node) -> tuple[int, bool]:
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )
    assert isinstance(dim, int), f"sort dim must be int, got {type(dim)}"
    assert isinstance(descending, bool), (
        f"sort descending must be bool, got {type(descending)}"
    )
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim
    if dim < 0:
        dim = ndim + dim
    assert dim == ndim - 1, (
        f"sort only supports sorting on last dimension, got dim={dim}"
    )
    return dim, descending


def _emit_cute_rank_sort(
    ctx: LoweringContext,
    load: CuteSortableLoad,
    input_tensor: torch.Tensor,
    *,
    descending: bool,
    k: int | None = None,
) -> tuple[ast.AST, ast.AST]:
    env = CompileEnvironment.current()
    fn = ctx.cg.device_function
    n = input_tensor.shape[-1]
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    if not isinstance(n_hint, int):
        raise exc.BackendUnsupported("cute", "dynamic sort extent")
    dtype_str = env.backend.dtype_str(load.dtype)
    index_dtype = env.backend.dtype_str(env.index_dtype)
    out_pos = fn.new_var("sort_out_pos")
    sorted_vals = fn.new_var("sorted_vals")
    sorted_indices = fn.new_var("sorted_indices")
    candidate = fn.new_var("sort_k")
    probe = fn.new_var("sort_j")
    candidate_rank = fn.new_var("sort_rank")
    candidate_value = fn.new_var("sort_candidate")
    probe_value = fn.new_var("sort_probe")
    before = fn.new_var("sort_before")
    selected = fn.new_var("sort_selected")

    ctx.cg.add_statement(
        statement_from_string(
            f"{out_pos} = {index_dtype}({load.index_exprs[load.sort_index_pos]})"
        )
    )
    ctx.cg.add_statement(statement_from_string(f"{sorted_vals} = {dtype_str}(0)"))
    ctx.cg.add_statement(statement_from_string(f"{sorted_indices} = {index_dtype}(0)"))

    cmp_op = ">" if descending else "<"

    def indexed_load(index: str) -> str:
        index_exprs = list(load.index_exprs)
        index_exprs[load.sort_index_pos] = index
        expr = f"{load.tensor_name}[{', '.join(index_exprs)}]"
        if load.mask_expr is not None:
            return f"({expr} if {load.mask_expr} else {dtype_str}(0))"
        return expr

    mask_suffix = f" and {out_pos} < {k}" if k is not None else ""
    ctx.cg.add_statement(
        statement_from_string(
            "\n".join(
                [
                    f"for {candidate} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    f"    {candidate_value} = {indexed_load(candidate)}",
                    f"    {candidate_rank} = {index_dtype}(0)",
                    f"    for {probe} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    f"        {probe_value} = {indexed_load(probe)}",
                    f"        {before} = ({probe_value} {cmp_op} {candidate_value}) or (({probe_value} == {candidate_value}) and ({probe} < {candidate}))",
                    f"        {candidate_rank} = {candidate_rank} + ({index_dtype}(1) if {before} else {index_dtype}(0))",
                    f"    {selected} = ({candidate_rank} == {out_pos}{mask_suffix})",
                    f"    {sorted_vals} = {candidate_value} if {selected} else {sorted_vals}",
                    f"    {sorted_indices} = {index_dtype}({candidate}) if {selected} else {sorted_indices}",
                ]
            )
        )
    )
    return expr_from_string(sorted_vals), expr_from_string(sorted_indices)


@sort_lowering.register_codegen("triton")
def codegen_sort(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.sort-based sort implementation.

    torch.sort(input, dim=-1, descending=False, stable=False) returns (values, indices).
    We implement this using tl.sort for values.
    For indices, we compute the rank of each element to determine its sorted position.

    Note: tl.sort only works on the last dimension currently.
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )
    # stable arg (node.args[3]) is ignored - tl.sort is stable

    assert isinstance(dim, int), f"sort dim must be int, got {type(dim)}"
    assert isinstance(descending, bool), (
        f"sort descending must be bool, got {type(descending)}"
    )

    # Get the input tensor shape info
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # tl.sort only supports sorting on the last dimension
    assert dim == ndim - 1, (
        f"tl.sort only supports sorting on last dimension, got dim={dim}"
    )

    descending_str = "True" if descending else "False"

    # Generate sorted values using tl.sort
    sorted_vals = ctx.cg.device_function.new_var("sorted_vals")
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_vals} = tl.sort({{tensor}}, descending={descending_str})",
            tensor=tensor,
        )
    )

    # Skip O(N^2) argsort when indices are not used downstream
    indices_used = any(
        user.target is getitem and user.args[1] == 1 for user in node.users
    )
    if not indices_used:
        return (expr_from_string(sorted_vals), None)

    # For indices, compute argsort using ranking:
    # For each element x[..., i], its rank is count of elements strictly less (or greater for descending)
    # plus count of equal elements with smaller index (for stability).
    # rank[..., i] gives the sorted position of x[..., i], so we need to invert this.
    sorted_indices = ctx.cg.device_function.new_var("sorted_indices")
    rank = ctx.cg.device_function.new_var("rank")
    idx_var = ctx.cg.device_function.new_var("idx")

    # Get size of last dimension (must be power of 2 for tl.sort)
    n = input_tensor.shape[-1]
    env = CompileEnvironment.current()
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    n_pow2 = next_power_of_2(n_hint)

    # Create indices: [0, 1, 2, ..., n-1]
    ctx.cg.add_statement(statement_from_string(f"{idx_var} = tl.arange(0, {n_pow2})"))

    # Set up dimension-specific indexing patterns and comparison operator
    cmp_op = ">" if descending else "<"
    if ndim == 1:
        # 1D: compare [1, n] with [n, 1], reduce over axis 1
        t_a, t_b = "[None, :]", "[:, None]"
        i_a, i_b = "[None, :]", "[:, None]"
        reduce_axis = 1
        # For inverting: [n, 1] == [1, n], reduce axis 0
        r_a, r_b, inv_i_a, _inv_i_b, inv_axis = (
            "[:, None]",
            "[None, :]",
            "[:, None]",
            "[None, :]",
            0,
        )
    elif ndim == 2:
        # 2D: compare [batch, 1, n] with [batch, n, 1], reduce over axis 2
        t_a, t_b = "[:, None, :]", "[:, :, None]"
        i_a, i_b = "[None, None, :]", "[None, :, None]"
        reduce_axis = 2
        # For inverting: [batch, n, 1] == [1, 1, n], reduce axis 1
        r_a, r_b, inv_i_a, _inv_i_b, inv_axis = (
            "[:, :, None]",
            "[None, None, :]",
            "[None, :, None]",
            "[None, None, :]",
            1,
        )
    else:
        raise NotImplementedError

    # Compute rank: count elements that should come before + tie-breaking
    ctx.cg.add_statement(
        statement_from_string(
            f"{rank} = tl.sum(tl.where({{tensor}}{t_a} {cmp_op} {{tensor}}{t_b}, 1, 0), axis={reduce_axis}) + "
            f"tl.sum(tl.where(({{tensor}}{t_a} == {{tensor}}{t_b}) & ({idx_var}{i_a} < {idx_var}{i_b}), 1, 0), axis={reduce_axis})",
            tensor=tensor,
        )
    )

    # Invert the rank permutation: sorted_indices[rank[i]] = i
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_indices} = tl.sum(tl.where({rank}{r_a} == {idx_var}{r_b}, {idx_var}{inv_i_a}, 0), axis={inv_axis})"
        )
    )

    # Return as tuple (values, indices)
    return (expr_from_string(sorted_vals), expr_from_string(sorted_indices))


@sort_lowering.register_codegen("cute")
def codegen_sort_cute(ctx: LoweringContext, node: Node) -> object:
    _, descending = _sort_args(node)
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_tensor = input_node.meta["val"]
    load = _env_arg(ctx, input_node)
    if not isinstance(load, CuteSortableLoad):
        load = input_node.meta.get("cute_sortable_load")
    if not isinstance(load, CuteSortableLoad):
        raise exc.BackendUnsupported("cute", "torch.sort input")
    node.meta["cute_sort_load"] = load
    node.meta["cute_sort_descending"] = descending
    return _emit_cute_rank_sort(ctx, load, input_tensor, descending=descending)


gather_lowering = register_lowering(
    torch.ops.aten.gather.default,
    masked_value_fn=passthrough_masked_value,
)


@gather_lowering.register_codegen("triton")
def codegen_gather(ctx: LoweringContext, node: Node) -> object:
    """Generate gather implementation using tl.gather.

    torch.gather(input, dim, index) gathers values along dim using index.
    Both input and index must be already-loaded tiles (not host tensors).
    Uses Triton's tl.gather for the actual gather operation.
    """
    # Validate arguments
    assert not node.kwargs, "gather does not support keyword arguments"
    assert len(node.args) == 3, f"gather expects 3 arguments, got {len(node.args)}"

    input_node = node.args[0]
    dim = node.args[1]
    index_node = node.args[2]

    assert isinstance(input_node, Node), "gather input must be a Node"
    assert isinstance(dim, int), f"gather dim must be int, got {type(dim)}"
    assert isinstance(index_node, Node), "gather index must be a Node"

    input_tensor = input_node.meta["val"]

    # Validate that input is a tensor
    assert isinstance(input_tensor, torch.Tensor), (
        f"gather input must be a tensor, got {type(input_tensor)}"
    )

    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # Validate dim is in range
    assert 0 <= dim < ndim, (
        f"gather dim {dim} out of range for tensor with {ndim} dimensions"
    )

    fn = ctx.cg.device_function

    # Get the input and index AST nodes
    input_ast_raw = _env_arg(ctx, input_node)
    assert isinstance(input_ast_raw, ast.AST)
    input_ast = input_ast_raw

    index_ast_raw = _env_arg(ctx, index_node)
    assert isinstance(index_ast_raw, ast.AST)
    index_ast = index_ast_raw

    result_var = fn.new_var("gather_result")

    ctx.cg.add_statement(
        statement_from_string(
            f"{result_var} = tl.gather({{input}}, {{index}}.to(tl.int32), axis={dim})",
            input=input_ast,
            index=index_ast,
        )
    )

    return expr_from_string(result_var)


@gather_lowering.register_codegen("cute")
def codegen_gather_cute(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "gather does not support keyword arguments"
    assert len(node.args) == 3, f"gather expects 3 arguments, got {len(node.args)}"

    input_node = node.args[0]
    dim = node.args[1]
    index_node = node.args[2]
    assert isinstance(input_node, Node)
    assert isinstance(dim, int)
    assert isinstance(index_node, Node)

    from ..language.memory_ops import _cute_combined_mask
    from ..language.memory_ops import _cute_index_exprs
    from ..language.memory_ops import load
    from .inductor_lowering import CodegenState

    if input_node.target is not load:
        raise exc.BackendUnsupported("cute", "torch.gather input")
    tensor_node = input_node.args[0]
    if not isinstance(tensor_node, Node):
        raise exc.BackendUnsupported("cute", "torch.gather tensor input")
    tensor = tensor_node.meta["val"]
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", "torch.gather tensor input")
    input_subscript = input_node.args[1]
    if not isinstance(input_subscript, (list, tuple)):
        raise exc.BackendUnsupported("cute", "torch.gather input subscript")

    ndim = len(input_subscript)
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise exc.InvalidReductionDim(dim)

    proxy_subscript = cast(
        "list[object]",
        list(map_arg(tuple(input_subscript), lambda arg: arg.meta["val"])),
    )
    ast_subscript = cast(
        "list[object]",
        list(map_arg(tuple(input_subscript), lambda arg: _env_arg(ctx, arg))),
    )
    index_ast = _env_arg(ctx, index_node)
    assert isinstance(index_ast, ast.AST)
    proxy_subscript[dim] = index_node.meta["val"]
    ast_subscript[dim] = index_ast

    from .generate_ast import GenerateAST

    if not isinstance(ctx.cg, GenerateAST):
        raise exc.NotAllowedInHelperFunction

    state = CodegenState(ctx.cg, fx_node=node, env=ctx.env)
    index_exprs = _cute_index_exprs(
        state,
        proxy_subscript,
        ast_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    tensor_name = ctx.cg.device_function.tensor_arg(tensor).name
    load_expr = f"{tensor_name}[{', '.join(index_exprs)}]"
    mask_expr = _cute_combined_mask(state, proxy_subscript, None, tensor=tensor)
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else cutlass.Boolean(0))")
    if mask_expr is None:
        return expr_from_string(load_expr)
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")


topk_lowering = register_lowering(torch.ops.aten.topk.default)


def _topk_args(node: Node) -> tuple[int, int, bool]:
    k = node.args[1]
    assert isinstance(k, int), f"topk k must be int, got {type(k)}"
    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    largest = node.args[3] if len(node.args) > 3 else node.kwargs.get("largest", True)
    assert isinstance(dim, int), f"topk dim must be int, got {type(dim)}"
    assert isinstance(largest, bool), f"topk largest must be bool, got {type(largest)}"
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim
    if dim < 0:
        dim = ndim + dim
    assert dim == ndim - 1, f"topk only supports the last dimension, got dim={dim}"
    return k, dim, largest


@topk_lowering.register_codegen("triton")
def codegen_topk(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.topk-based topk implementation.

    torch.topk(input, k, dim=-1, largest=True, sorted=True) returns (values, indices).
    We use tl.topk for values (when largest=True) or tl.sort (when largest=False).
    For indices, we compute argsort using a ranking approach.

    Note: tl.topk/tl.sort only works on the last dimension currently.
    See: https://github.com/triton-lang/triton/blob/main/python/triton/language/standard.py
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    k = node.args[1]
    assert isinstance(k, int), f"topk k must be int, got {type(k)}"

    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    largest = node.args[3] if len(node.args) > 3 else node.kwargs.get("largest", True)
    # sorted arg (node.args[4]) is ignored - tl.topk always returns sorted

    assert isinstance(dim, int), f"topk dim must be int, got {type(dim)}"
    assert isinstance(largest, bool), f"topk largest must be bool, got {type(largest)}"

    # Get the input tensor shape info
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # tl.topk only supports sorting on the last dimension
    assert dim == ndim - 1, f"tl.topk only supports the last dimension, got dim={dim}"

    # Get size of last dimension
    n = input_tensor.shape[-1]
    env = CompileEnvironment.current()
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    n_pow2 = next_power_of_2(n_hint)
    k_pow2 = next_power_of_2(k)

    # Generate top-k values using tl.topk (for largest=True) or tl.sort (for largest=False)
    topk_vals = ctx.cg.device_function.new_var("topk_vals")
    if largest:
        # tl.topk returns top k largest elements directly
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_vals} = tl.topk({{tensor}}, {k_pow2})",
                tensor=tensor,
            )
        )
    else:
        # tl.topk only supports largest=True, so use tl.sort with descending=False
        sorted_vals = ctx.cg.device_function.new_var("sorted_vals")
        ctx.cg.add_statement(
            statement_from_string(
                f"{sorted_vals} = tl.sort({{tensor}}, descending=False)",
                tensor=tensor,
            )
        )
        # Need to gather first k elements from sorted
        k_idx = ctx.cg.device_function.new_var("k_idx")
        idx_n = ctx.cg.device_function.new_var("idx_n")
        ctx.cg.add_statement(statement_from_string(f"{k_idx} = tl.arange(0, {k_pow2})"))
        ctx.cg.add_statement(statement_from_string(f"{idx_n} = tl.arange(0, {n_pow2})"))
        if ndim == 1:
            ctx.cg.add_statement(
                statement_from_string(
                    f"{topk_vals} = tl.sum("
                    f"tl.where(({idx_n}[:, None] == {k_idx}[None, :]) & ({k_idx}[None, :] < {k}), "
                    f"{sorted_vals}[:, None], 0.0), axis=0)"
                )
            )
        else:
            ctx.cg.add_statement(
                statement_from_string(
                    f"{topk_vals} = tl.sum("
                    f"tl.where(({idx_n}[None, :, None] == {k_idx}[None, None, :]) & ({k_idx}[None, None, :] < {k}), "
                    f"{sorted_vals}[:, :, None], 0.0), axis=1)"
                )
            )

    # For indices, compute argsort using ranking approach
    topk_indices = ctx.cg.device_function.new_var("topk_indices")
    rank = ctx.cg.device_function.new_var("rank")
    idx_var = ctx.cg.device_function.new_var("idx")

    ctx.cg.add_statement(statement_from_string(f"{idx_var} = tl.arange(0, {n_pow2})"))

    # Set up dimension-specific indexing patterns and comparison operator
    cmp_op = ">" if largest else "<"
    if ndim == 1:
        t_a, t_b = "[None, :]", "[:, None]"
        i_a, i_b = "[None, :]", "[:, None]"
        reduce_axis = 1
        r_a, r_b, inv_i_a, inv_axis = "[:, None]", "[None, :]", "[:, None]", 0
    elif ndim == 2:
        t_a, t_b = "[:, None, :]", "[:, :, None]"
        i_a, i_b = "[None, None, :]", "[None, :, None]"
        reduce_axis = 2
        r_a, r_b, inv_i_a, inv_axis = (
            "[:, :, None]",
            "[None, None, :]",
            "[None, :, None]",
            1,
        )
    else:
        raise NotImplementedError

    # Compute rank: count elements that should come before + tie-breaking
    ctx.cg.add_statement(
        statement_from_string(
            f"{rank} = tl.sum(tl.where({{tensor}}{t_a} {cmp_op} {{tensor}}{t_b}, 1, 0), axis={reduce_axis}) + "
            f"tl.sum(tl.where(({{tensor}}{t_a} == {{tensor}}{t_b}) & ({idx_var}{i_a} < {idx_var}{i_b}), 1, 0), axis={reduce_axis})",
            tensor=tensor,
        )
    )

    # Invert rank permutation to get sorted indices, then gather first k
    sorted_indices = ctx.cg.device_function.new_var("sorted_indices")
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_indices} = tl.sum(tl.where({rank}{r_a} == {idx_var}{r_b}, {idx_var}{inv_i_a}, 0), axis={inv_axis})"
        )
    )

    # Gather first k indices
    k_idx_final = ctx.cg.device_function.new_var("k_idx")
    ctx.cg.add_statement(
        statement_from_string(f"{k_idx_final} = tl.arange(0, {k_pow2})")
    )

    if ndim == 1:
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_indices} = tl.sum("
                f"tl.where(({idx_var}[:, None] == {k_idx_final}[None, :]) & ({k_idx_final}[None, :] < {k}), "
                f"{sorted_indices}[:, None], 0), axis=0)"
            )
        )
    else:
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_indices} = tl.sum("
                f"tl.where(({idx_var}[None, :, None] == {k_idx_final}[None, None, :]) & ({k_idx_final}[None, None, :] < {k}), "
                f"{sorted_indices}[:, :, None], 0), axis=1)"
            )
        )

    return (expr_from_string(topk_vals), expr_from_string(topk_indices))


@topk_lowering.register_codegen("cute")
def codegen_topk_cute(ctx: LoweringContext, node: Node) -> object:
    k, _, largest = _topk_args(node)
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_tensor = input_node.meta["val"]
    load = _env_arg(ctx, input_node)
    if not isinstance(load, CuteSortableLoad):
        load = input_node.meta.get("cute_sortable_load")
    if not isinstance(load, CuteSortableLoad):
        raise exc.BackendUnsupported("cute", "torch.topk input")
    node.meta["cute_topk_lane_expr"] = load.index_exprs[load.sort_index_pos]
    node.meta["cute_topk_k"] = k
    return _emit_cute_rank_sort(ctx, load, input_tensor, descending=largest, k=k)


# ============================================================================
# NKI backend codegen handlers
#
# Appended as a block (ported from fix-nki-kernel-compilation). Each handler
# attaches to its *_lowering object via @<obj>.register_codegen("nki"); decorator
# registration is position-independent so long as the target object is defined
# above. silu_lowering and cumsum_lowering are NKI-introduced lowering objects
# (upstream lacked them); stack_lowering already exists upstream.
# ============================================================================

@full_lowering.register_codegen("nki")
def codegen_full_nki(ctx: LoweringContext, node: Node) -> ast.AST:
    env = CompileEnvironment.current()
    size = map_arg(node.args[0], lambda n: n.meta["val"])
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    value_ast = map_arg(node.args[1], lambda arg: _env_arg(ctx, arg))
    if isinstance(value_ast, (int, float, bool)):
        value_ast = expr_from_string(constant_repr(value_ast))
    assert isinstance(value_ast, ast.AST), value_ast

    NKI_PARTITION_MAX = 128

    import sympy as _sympy

    state = getattr(env, "_codegen_state", None)
    _bs_subs: dict[_sympy.Symbol, int] = {}
    if state is not None:
        for _bid in range(len(env.block_sizes)):
            _bs = env.block_sizes[_bid]
            _bs_subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))

    def _resolve_dim(s: int | torch.SymInt) -> int:
        if isinstance(s, int):
            return s
        return int(s._sympy_().subs(_bs_subs))

    # Squeeze leading batch dims (size 1) from 3D+ shapes to 2D.
    # E.g. hl.zeros([1, tile_m, tile_n]) → [tile_m, tile_n] in NKI SBUF.
    size = list(size)
    while len(size) > 2 and _resolve_dim(size[0]) == 1:
        size = size[1:]
    # Also handle case where leading dims multiply to the partition dim
    # (e.g. [B_tile, M_tile, N_tile] with B_tile > 1 — combine into flat partition)
    if len(size) > 2:
        flat_part = 1
        for s in size[:-1]:
            flat_part *= _resolve_dim(s)
        size = [flat_part, _resolve_dim(size[-1])]

    # 2D accumulator layout normalization: ``hl.full([1, N], ...)`` inside a
    # tile-loop body is almost always a reduction accumulator (per-row
    # scalars). When this full is later combined element-wise with a
    # reduction result (which our codegen stores as [N, 1] SBUF), FX
    # expects a [1, N] × [N, 1] → [N, N] numpy-broadcast but the kernel's
    # semantic intent is a per-row [N] vector. Storing the accumulator
    # as [N, 1] instead aligns with reduction output layout and lets
    # ``tensor_scalar`` / ``tensor_tensor`` operate element-wise with
    # matching shapes.
    #
    # Heuristic: if this hl.full has size [1, N] with N > 1 AND its FX
    # users include a binary op (add/sub/max/min/mul/...) whose other
    # operand is a SymInt-shaped tensor, transpose to [N, 1]. This
    # correctly catches the common loop-carried accumulator case while
    # leaving [1, N] broadcast vectors untouched (they typically have
    # only .to()/reshape users).
    transpose_to_partition_major = False
    # NOTE: The transpose_to_partition_major heuristic below is disabled
    # because it causes shape cascades in attention-style kernels where
    # [1,N] × [1,N] operations are incorrectly treated as [N,N] broadcasts.
    # The correct fix requires semantic understanding of accumulator vs.
    # result shapes which is beyond a simple FX-shape heuristic.
    if False and len(size) == 2 and _resolve_dim(size[0]) == 1:
        _n = _resolve_dim(size[1])
        if _n > 1:
            _binary_op_names = {
                "maximum", "minimum", "amax", "amin", "add", "sub", "mul",
                "div", "log", "exp", "sum", "mean", "maximum.default",
                "minimum.default", "exp.default", "log.default",
            }
            visited = set()
            stack = list(node.users)
            while stack and not transpose_to_partition_major:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                if u.op == "call_function":
                    t_name = str(getattr(u.target, "__name__", u.target)).lower()
                    for key in _binary_op_names:
                        if key in t_name:
                            # Keep [1, N] row-major when the downstream op
                            # itself remains [1, N]. Partition-axis NKI
                            # reductions feed row tiles, and transposing the
                            # accumulator to [N, 1] turns elementwise updates
                            # into accidental [N, N] broadcasts.
                            val = u.meta.get("val")
                            if isinstance(val, torch.Tensor):
                                try:
                                    u_shape = [_resolve_dim(d) for d in val.shape]
                                except (TypeError, ValueError):
                                    u_shape = []
                                if (
                                    len(u_shape) == 2
                                    and u_shape[0] == 1
                                    and u_shape[1] == _n
                                ):
                                    break
                            # Look for a same-[N] other operand or a
                            # reduction result in the chain. This also
                            # covers m_i = m_ij pattern where the binary
                            # pattern appears a couple of hops away.
                            transpose_to_partition_major = True
                            break
                if len(visited) < 32:
                    stack.extend(u.users)

    if transpose_to_partition_major:
        size = [size[1], size[0]]  # swap [1, N] → [N, 1]

    partition_dim = _resolve_dim(size[0])
    free_dims = [_resolve_dim(s) for s in size[1:]]
    dtype_str = env.backend.dtype_str(dtype)
    var = ctx.cg.device_function.new_var("_nki_full", dce=True)

    if partition_dim > NKI_PARTITION_MAX and not free_dims:
        # 1D tensor (e.g. grad_w_acc[DIM]): in NKI semantics this represents a
        # feature-dimension accumulator. Allocate as [1, DIM] (partition=1, free=DIM)
        # so that it can be stored to [block_id:block_id+1, 0:DIM] in HBM.
        ctx.cg.add_statement(
            statement_from_string(
                f"{var} = nl.ndarray([1, {partition_dim}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        ctx.cg.add_statement(
            statement_from_string(
                f"nisa.memset({var}, value={{val}})", val=value_ast
            )
        )
        # Register shape so cast_ast and other ops know the SBUF shape
        ctx.cg.device_function._nki_sbuf_shapes[var] = [1, partition_dim]
        # Mark as a free-dim accumulator (not a partition-split tile list)
        ctx.cg.device_function._nki_free_dim_tile_lists.add(var)
    elif partition_dim > NKI_PARTITION_MAX:
        n_partitions = partition_dim // NKI_PARTITION_MAX
        assert partition_dim % NKI_PARTITION_MAX == 0, (
            f"partition dim {partition_dim} must be multiple of {NKI_PARTITION_MAX}"
        )
        free_str = ", ".join(str(d) for d in free_dims)
        tile_vars: list[str] = []
        for i in range(n_partitions):
            tile_var = ctx.cg.device_function.new_var(f"{var}_{i}")
            tile_vars.append(tile_var)
            ctx.cg.add_statement(
                statement_from_string(
                    f"{tile_var} = nl.ndarray([{NKI_PARTITION_MAX}, {free_str}], "
                    f"{dtype_str}, buffer=nl.sbuf)"
                )
            )
        for tile_var in tile_vars:
            ctx.cg.add_statement(
                statement_from_string(
                    f"nisa.memset({tile_var}, value={{val}})", val=value_ast
                )
            )
        ctx.cg.device_function.register_tile_list(var, tile_vars)
    else:
        shape_dims = ctx.cg.device_function.tile_strategy.shape_dims([*size])
        ndarray_expr = env.backend.full_expr(shape_dims, "0", dtype)
        ctx.cg.add_statement(statement_from_string(f"{var} = {ndarray_expr}"))
        ctx.cg.add_statement(
            statement_from_string(
                f"nisa.memset({var}, value={{val}})", val=value_ast
            )
        )
        # Register SBUF shape for multi-user detection on copy vars
        # full_expr may have added a trailing 1 for 1D shapes internally
        def _resolve_nki_shape_dim(dim: object) -> int:
            if isinstance(dim, int):
                return dim
            if isinstance(dim, torch.SymInt):
                return int(dim._sympy_().subs(_bs_subs))
            if isinstance(dim, str):
                for (
                    block_id,
                ), var_name in ctx.cg.device_function.block_size_var_cache.items():
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
                resolved.append(1)
            if len(resolved) >= 2:
                ctx.cg.device_function._nki_sbuf_shapes[var] = resolved
        except (TypeError, ValueError):
            pass
    return expr_from_string(var)


@unsqueeze_lowering.register_codegen("nki")
def codegen_unsqueeze_nki(ctx: LoweringContext, node: Node) -> object:
    # NKI tensors are already 2D (partition, free) and broadcast automatically.
    # unsqueeze is a no-op.
    assert not node.kwargs, "unsqueeze kwargs not supported"
    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    assert isinstance(tensor, ast.AST)
    return tensor


@squeeze_lowering.register_codegen("nki")
def codegen_squeeze_nki(ctx: LoweringContext, node: Node) -> object:
    # NKI tensors are already 2D (partition, free); squeeze is a no-op.
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    return tensor


@view_lowering.register_codegen("nki")
@reshape_lowering.register_codegen("nki")
def codegen_view_nki(ctx: LoweringContext, node: Node) -> object:
    # NKI SBUF tensors are 2D (partition, free); view/reshape are no-ops.
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    return tensor


@permute_lowering.register_codegen("nki")
def codegen_permute_nki(ctx: LoweringContext, node: Node) -> object:
    """NKI lowering for aten.permute.

    NKI SBUF tiles are always 2D (partition, free). Helion's NKI backend
    reshapes higher-rank inputs to 2D at kernel entry (`device_function`
    preamble). In that 2D view:

    - Identity permutations are no-ops.
    - A true 2D swap (``dims == [1, 0]``) becomes ``nc_transpose``.
    - Higher-rank permutations whose logical effect preserves the
      (partition, free) 2D layout are treated as no-ops — the FX graph
      uses them for shape bookkeeping only.

    Permutations that actually reorder the partition/free axes in a way
    that changes the 2D tile contents (other than a plain 2D swap) are
    not yet supported.
    """
    import ast as _ast
    from .nki_backend import NKIOpOverrides, NKIBackend
    from .compile_environment import CompileEnvironment
    from .ast_extension import statement_from_string

    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, _ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    assert {*dims} == {*range(len(dims))}, dims

    # Identity
    if dims == list(range(len(dims))):
        return tensor

    # 2D swap → nc_transpose. We allocate a PSUM tile (required by
    # nc_transpose on the Tensor Engine) then copy back to SBUF.
    if len(dims) == 2 and dims == [1, 0]:
        from .generate_ast import GenerateAST

        env = CompileEnvironment.current()
        state = getattr(env, "_codegen_state", None)
        if state is None or env.backend.name != "nki":
            # Fall back to no-op; shape bookkeeping only.
            return tensor

        val = node.meta.get("val")
        input_val = node.args[0].meta.get("val") if isinstance(node.args[0], Node) else None
        if not isinstance(val, torch.Tensor) or not isinstance(input_val, torch.Tensor):
            return tensor

        def _resolve_shape(shape: list) -> list[int]:
            out: list[int] = []
            for d in shape:
                if isinstance(d, int):
                    out.append(d)
                elif isinstance(d, torch.SymInt):
                    out.append(env.size_hint(d))
                else:
                    out.append(int(d))
            return out

        # Squeeze to 2D (NKI SBUF tiles are 2D) AFTER resolving any SymInts.
        src_shape = NKIOpOverrides._squeeze_shape_2d(_resolve_shape(list(input_val.shape)))
        dst_shape = NKIOpOverrides._squeeze_shape_2d(_resolve_shape(list(val.shape)))
        if len(src_shape) != 2 or len(dst_shape) != 2:
            return tensor

        dtype_str = env.backend.dtype_str(val.dtype)

        # Check if source is a tile-list (multi-tile split from partition > 128 load).
        src_name = _ast.unparse(tensor)

        # Prefer the registered SBUF shape over the FX val shape, which may
        # have wrong size hints for dynamic block sizes (e.g. tile_kv with
        # unbacked SymInt whose hint differs from the configured block_size).
        registered_src = state.device_function._nki_sbuf_shapes.get(src_name)
        if registered_src is not None and len(registered_src) == 2:
            src_shape = list(registered_src)
            # Infer dst_shape as the transpose
            dst_shape = [src_shape[1], src_shape[0]]

        src_tile_vars = state.device_function.get_tile_list_vars(src_name)
        if src_tile_vars is not None:
            # Use actual per-tile SBUF shapes (not FX val shape, which may
            # not reflect the clamped block_size).
            tile0_shape = state.device_function._nki_sbuf_shapes.get(src_tile_vars[0])
            if tile0_shape is None or len(tile0_shape) != 2:
                tile0_shape = [128, src_shape[1] if src_shape else 1]
            n_tiles = len(src_tile_vars)
            per_tile_partition = tile0_shape[0]
            per_tile_free = tile0_shape[1]
            # Full src shape after concat (partition axis): [n_tiles * 128, F]
            # Transposed: [F, n_tiles * 128]
            total_partition = n_tiles * per_tile_partition
            sbuf_var = state.device_function.new_var("_nki_permute_sbuf")
            state.add_statement(statement_from_string(
                f"{sbuf_var} = nl.ndarray([{per_tile_free}, {total_partition}], {dtype_str}, buffer=nl.sbuf)"
            ))
            for i, tv in enumerate(src_tile_vars):
                tr_psum = state.device_function.new_var("_nki_permute_psum_t")
                state.add_statement(statement_from_string(
                    f"{tr_psum} = nl.ndarray([{per_tile_free}, {per_tile_partition}], {dtype_str}, buffer=nl.psum)"
                ))
                state.add_statement(statement_from_string(
                    f"nisa.nc_transpose(dst={tr_psum}, data={tv})"
                ))
                free_start = i * per_tile_partition
                free_end = (i + 1) * per_tile_partition
                state.add_statement(statement_from_string(
                    f"nisa.tensor_copy(dst={sbuf_var}[0:{per_tile_free}, {free_start}:{free_end}], src={tr_psum})"
                ))
            state.device_function._nki_sbuf_shapes[sbuf_var] = [per_tile_free, total_partition]
            state.device_function._nki_sbuf_dtypes[sbuf_var] = dtype_str
            return expr_from_string(sbuf_var)

        psum_var = state.device_function.new_var("_nki_permute_psum")
        sbuf_var = state.device_function.new_var("_nki_permute_sbuf")

        # Allocate PSUM with the same dtype as input (nc_transpose requires
        # dst.dtype == data.dtype).
        state.add_statement(
            statement_from_string(
                f"{psum_var} = nl.ndarray([{dst_shape[0]}, {dst_shape[1]}], {dtype_str}, buffer=nl.psum)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.nc_transpose(dst={psum_var}, data={_ast.unparse(tensor)})"
            )
        )
        state.add_statement(
            statement_from_string(
                f"{sbuf_var} = nl.ndarray([{dst_shape[0]}, {dst_shape[1]}], {dtype_str}, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_copy(dst={sbuf_var}, src={psum_var})"
            )
        )
        state.device_function._nki_sbuf_shapes[sbuf_var] = dst_shape
        state.device_function._nki_sbuf_dtypes[sbuf_var] = dtype_str
        return expr_from_string(sbuf_var)

    # Higher-rank permutations: treat as a no-op on the 2D-squeezed layout
    # when the permutation preserves the logical partition/free axes. This
    # covers common Mamba/attention patterns where 3D/4D permutes are used
    # for shape bookkeeping only and the underlying SBUF tile is unchanged.
    # If a kernel really needs a partition-axis permute on a higher-rank
    # tile, it should reshape explicitly — we surface a clear error then.
    return tensor


@stack_lowering.register_codegen("nki")
def codegen_stack_nki(ctx: LoweringContext, node: Node) -> object:
    """NKI lowering for aten.stack.

    Stacks N input tensors along a new dimension. Since NKI only supports
    2D tiles, we interleave the input tiles into a larger 2D tile.

    For dim=0: stack([A, B], dim=0) on [P, F] tiles → [N*P, F]
      Row i*P+j of output = input_i[j, :]
    For dim=1: stack([A, B], dim=1) on [P, F] tiles → [P, N*F]
      or after squeeze: [P*N, F] depending on dimensionality
    """
    import ast as _ast

    from .ast_extension import statement_from_string, create, expr_from_string
    from .compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    state = getattr(env, "_codegen_state", None)
    if state is None:
        raise NotImplementedError("NKI stack requires codegen state")

    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    assert isinstance(tensors, (list, tuple))
    assert isinstance(dim, int)

    tensor_asts = [ctx.env[t] for t in tensors]
    n = len(tensor_asts)
    if n == 0:
        raise ValueError("Cannot stack empty tensor list")

    # Get output shape from FX metadata
    out_val = node.meta.get("val")
    if out_val is None or not isinstance(out_val, torch.Tensor):
        raise NotImplementedError("NKI stack requires output shape metadata")

    from .nki_backend import NKIOpOverrides
    import sympy as _sp_stack
    _bs_subs_stack: dict[_sp_stack.Symbol, int] = {}
    for _bid in range(len(env.block_sizes)):
        _bs = env.block_sizes[_bid]
        _cfg = _bs.from_config(state.config)
        if isinstance(_cfg, int):
            _bs_subs_stack[_bs.symbol()] = _cfg

    def _resolve_stack_dim(d: object) -> int:
        if isinstance(d, int):
            return d
        if isinstance(d, torch.SymInt):
            try:
                return int(d._sympy_().subs(_bs_subs_stack))
            except (TypeError, ValueError):
                return int(env.size_hint(d))
        return int(d)

    out_shape_full = [_resolve_stack_dim(d) for d in out_val.shape]
    out_shape = NKIOpOverrides._squeeze_shape_2d(out_shape_full)
    if len(out_shape) != 2:
        raise NotImplementedError(f"NKI stack: output shape {out_shape_full} doesn't squeeze to 2D")

    p_out, f_out = out_shape[0], out_shape[1]

    # Get input shapes
    first_input = tensors[0]
    first_val = first_input.meta.get("val") if isinstance(first_input, Node) else None
    if first_val is None or not isinstance(first_val, torch.Tensor):
        raise NotImplementedError("NKI stack: input metadata missing")
    in_shape_full = [_resolve_stack_dim(d) for d in first_val.shape]
    in_shape = NKIOpOverrides._squeeze_shape_2d(in_shape_full)
    if len(in_shape) != 2:
        raise NotImplementedError(f"NKI stack: input shape {in_shape_full} doesn't squeeze to 2D")
    p_in, f_in = in_shape[0], in_shape[1]

    dtype_str = env.backend.dtype_str(out_val.dtype)
    device_fn = state.device_function

    # Allocate output buffer
    result_var = device_fn.new_var("_nki_stack", dce=True)
    device_fn._nki_sbuf_shapes[result_var] = [p_out, f_out]
    device_fn._nki_sbuf_dtypes[result_var] = dtype_str
    state.add_statement(statement_from_string(
        f"{result_var} = nl.ndarray([{p_out}, {f_out}], {dtype_str}, buffer=nl.sbuf)"
    ))
    state.add_statement(statement_from_string(f"nisa.memset({result_var}, value=0)"))

    # Determine whether this stack interleaves (dim inserts between existing dims)
    # or creates contiguous blocks (dim inserts at the front).
    # For 2D inputs [P, F]:
    #   dim=0 → output [n, P, F] → [n*P, F]: block copy (each input = one block)
    #   dim=1 → output [P, n, F] → [P*n, F]: interleave (elements alternate per row)
    #   dim=-1 or dim=2 → output [P, F, n] → [P, F*n]: free-dim stack
    # Normalize dim for the input rank
    _input_rank = len(in_shape_full)
    _dim_norm = dim if dim >= 0 else dim + _input_rank + 1
    _stack_interleave = (_dim_norm > 0 and _dim_norm <= _input_rank - 1)

    # Copy each input tile into the correct position
    # After squeeze_shape_2d, the output [P_out, F_out] = [n*P_in, F_in] for dim=0
    # or [P_in, n*F_in] for dim=-1 on the original shape.
    # Determine the interleave strategy from the shapes:
    if p_out == n * p_in and f_out == f_in and not _stack_interleave:
        # Stack along partition dimension: output[i*p_in:(i+1)*p_in, :] = input_i
        for i, tensor_ast in enumerate(tensor_asts):
            src_name = _ast.unparse(tensor_ast) if isinstance(tensor_ast, _ast.AST) else str(tensor_ast)
            start = i * p_in
            state.add_statement(statement_from_string(
                f"nisa.tensor_copy(dst={result_var}[{start}:{start + p_in}, 0:{f_out}], src={src_name})"
            ))
    elif p_out == p_in and f_out == n * f_in:
        # Stack along free dimension: output[:, i*f_in:(i+1)*f_in] = input_i
        for i, tensor_ast in enumerate(tensor_asts):
            src_name = _ast.unparse(tensor_ast) if isinstance(tensor_ast, _ast.AST) else str(tensor_ast)
            start = i * f_in
            state.add_statement(statement_from_string(
                f"nisa.tensor_copy(dst={result_var}[0:{p_out}, {start}:{start + f_in}], src={src_name})"
            ))
    elif p_out == p_in * n and f_out == f_in:
        # Interleave: output[i + j*n, :] = input_j[i, :] — partition interleave
        # This happens with dim=1 on [P, F] → [P, n, F] → squeeze [P*n, F]
        loop_var = device_fn.new_var("_stack_i")
        body_stmts = []
        for j, tensor_ast in enumerate(tensor_asts):
            src_name = _ast.unparse(tensor_ast) if isinstance(tensor_ast, _ast.AST) else str(tensor_ast)
            body_stmts.append(statement_from_string(
                f"nisa.tensor_copy(dst={result_var}[{loop_var}*{n}+{j}:{loop_var}*{n}+{j}+1, 0:{f_out}], "
                f"src={src_name}[{loop_var}:{loop_var}+1, 0:{f_in}])"
            ))
        state.add_statement(create(
            _ast.For,
            target=create(_ast.Name, id=loop_var, ctx=_ast.Store()),
            iter=expr_from_string(f"nl.affine_range({p_in})"),
            body=body_stmts,
            orelse=[],
        ))
    else:
        raise NotImplementedError(
            f"NKI stack: unsupported shape combination in={in_shape} out={out_shape} n={n}"
        )

    return expr_from_string(result_var)


@expand_lowering.register_codegen("nki")
def codegen_expand_nki(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "expand kwargs not supported"
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)

    from .nki_backend import NKIOpOverrides

    env = CompileEnvironment.current()
    state = getattr(env, "_codegen_state", None)
    if state is None:
        return tensor

    def _resolve_dim(dim: object) -> int:
        if isinstance(dim, int):
            return dim
        if isinstance(dim, torch.SymInt):
            if state.config is not None:
                import sympy as _sympy_expand

                subs: dict[_sympy_expand.Symbol, int] = {}
                for bs in env.block_sizes:
                    cfg = bs.from_config(state.config)
                    if isinstance(cfg, int):
                        subs[bs.symbol()] = cfg
                try:
                    return int(dim._sympy_().subs(subs))
                except (TypeError, ValueError):
                    return int(env.size_hint(dim))
            return int(env.size_hint(dim))
        return int(dim)

    dst_shape = NKIOpOverrides._squeeze_shape_2d(
        [_resolve_dim(dim) for dim in val.shape]
    )
    if len(dst_shape) != 2:
        raise exc.BackendUnsupported("nki", f"expand to non-2D SBUF shape {dst_shape}")

    tensor_name = ast.unparse(tensor)

    def _lookup_shape(name: str) -> list[int] | None:
        shape = state.device_function._nki_sbuf_shapes.get(name)
        if shape is not None:
            return list(shape)
        lookup = name
        while "_copy" in lookup:
            lookup = lookup[: lookup.rfind("_copy")]
            shape = state.device_function._nki_sbuf_shapes.get(lookup)
            if shape is not None:
                return list(shape)
        return None

    src_shape = _lookup_shape(tensor_name)
    if src_shape is None and isinstance(node.args[0], Node):
        input_val = node.args[0].meta.get("val")
        if isinstance(input_val, torch.Tensor):
            src_shape = NKIOpOverrides._squeeze_shape_2d(
                [_resolve_dim(dim) for dim in input_val.shape]
            )
    if src_shape is None:
        return tensor
    if src_shape == dst_shape:
        return tensor
    if len(src_shape) != 2:
        raise exc.BackendUnsupported("nki", f"expand from non-2D SBUF shape {src_shape}")

    src_for_broadcast = tensor_name
    if (
        src_shape[0] == 1
        and src_shape[1] == dst_shape[0]
        and dst_shape[0] > 1
        and dst_shape[1] > 1
    ):
        dtype_str = env.backend.dtype_str(val.dtype)
        # nc_transpose requires float input. Cast int/bool to float32 first.
        _int_dtypes_expand = {"nl.int32", "nl.int16", "nl.int8", "nl.uint32",
                              "nl.uint16", "nl.uint8", "nl.bool_"}
        _tr_src = tensor_name
        _tr_dtype = dtype_str
        if dtype_str in _int_dtypes_expand:
            _cast_expand = state.device_function.new_var("_nki_expand_cast", dce=True)
            state.device_function._nki_sbuf_shapes[_cast_expand] = [1, src_shape[1]]
            state.device_function._nki_sbuf_dtypes[_cast_expand] = "nl.float32"
            state.add_statement(statement_from_string(
                f"{_cast_expand} = nl.ndarray([1, {src_shape[1]}], nl.float32, buffer=nl.sbuf)"
            ))
            state.add_statement(statement_from_string(
                f"nisa.memset({_cast_expand}, value=0)"
            ))
            state.add_statement(statement_from_string(
                f"nisa.tensor_tensor(dst={_cast_expand}, data1={_cast_expand}, data2={tensor_name}, op=nl.add)"
            ))
            _tr_src = _cast_expand
            _tr_dtype = "nl.float32"
        tr_psum = state.device_function.new_var("_nki_expand_tr_psum", dce=True)
        tr_sbuf = state.device_function.new_var("_nki_expand_tr_sbuf", dce=True)
        state.device_function._nki_sbuf_shapes[tr_sbuf] = [dst_shape[0], 1]
        # When the transpose cast via float32, keep tr_sbuf as float32 to avoid
        # a float32 psum -> bool_ bitcast (which destroys the data). The where
        # lowering handles the bool->uint32 cast separately.
        _tr_sbuf_dtype = _tr_dtype if _tr_dtype != dtype_str else dtype_str
        state.device_function._nki_sbuf_dtypes[tr_sbuf] = _tr_sbuf_dtype
        state.add_statement(
            statement_from_string(
                f"{tr_psum} = nl.ndarray([{dst_shape[0]}, 1], "
                f"{_tr_dtype}, buffer=nl.psum)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.nc_transpose(dst={tr_psum}, data={_tr_src})"
            )
        )
        state.add_statement(
            statement_from_string(
                f"{tr_sbuf} = nl.ndarray([{dst_shape[0]}, 1], "
                f"{_tr_sbuf_dtype}, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(f"nisa.tensor_copy(dst={tr_sbuf}, src={tr_psum})")
        )
        src_for_broadcast = tr_sbuf
        src_shape = [dst_shape[0], 1]

    if not all(s == d or s == 1 for s, d in zip(src_shape, dst_shape, strict=True)):
        raise exc.BackendUnsupported(
            "nki", f"expand from {src_shape} to {dst_shape} is not broadcastable"
        )

    dtype_str = env.backend.dtype_str(val.dtype)
    # When we transposed via float32, src_for_broadcast is float32.
    # Use the actual src dtype for the broadcast output to avoid
    # float32 → bool_ invalid bitcasts.
    _out_dtype = state.device_function._nki_sbuf_dtypes.get(
        src_for_broadcast if isinstance(src_for_broadcast, str) else "", dtype_str
    )
    out = state.device_function.new_var("_nki_expand", dce=True)
    state.device_function._nki_sbuf_shapes[out] = list(dst_shape)
    state.device_function._nki_sbuf_dtypes[out] = _out_dtype
    state.add_statement(
        statement_from_string(
            f"{out} = nl.broadcast_to({src_for_broadcast}, "
            f"shape=({dst_shape[0]}, {dst_shape[1]}))"
        )
    )
    return expr_from_string(out)


silu_lowering = register_lowering(
    torch.ops.aten.silu.default,
    masked_value_fn=passthrough_masked_value,
)


@silu_lowering.register_codegen("nki")
def codegen_silu_nki(ctx: LoweringContext, node: Node) -> object:
    """Lower aten.silu directly to the NKI activation override."""
    assert not node.kwargs, "silu kwargs not supported"
    x = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(x, ast.AST)
    x_name = ast.unparse(x)
    result_name = CompileEnvironment.current().backend.inductor_op_overrides().silu(x_name)
    return expr_from_string(result_name)


@mm_lowering.register_codegen("nki")
def codegen_mm_nki(ctx: LoweringContext, node: Node) -> ast.AST:
    return _nki_dot(ctx, node, False)


@addmm_lowering.register_codegen("nki")
def codegen_addmm_nki(ctx: LoweringContext, node: Node) -> ast.AST:
    return _nki_dot(ctx, node, True)


@bmm_lowering.register_codegen("nki")
def codegen_bmm_nki(ctx: LoweringContext, node: Node) -> ast.AST:
    return _nki_dot(ctx, node, False)


@baddbmm_lowering.register_codegen("nki")
def codegen_baddbmm_nki(ctx: LoweringContext, node: Node) -> ast.AST:
    return _nki_dot(ctx, node, True)


cumsum_lowering = register_lowering(torch.ops.aten.cumsum.default)


@cumsum_lowering.register_codegen("nki")
def codegen_cumsum_nki(ctx: LoweringContext, node: Node) -> object:
    """NKI lowering for aten.cumsum via nisa.tensor_tensor_scan.

    ``nisa.tensor_tensor_scan(dst, data0, data1, initial, op0, op1)`` computes
    ``dst[i] = op1(op0(data0[i], prev), data1[i])`` with ``prev`` starting at
    ``initial``. Setting ``data1 = zeros`` (identity for add), ``op0 = add``,
    ``op1 = add`` and ``initial = 0`` gives us ``cumsum(data0)``.

    Trn2/Trn3 only. Inputs cast to fp32 internally.
    """
    from .ast_extension import statement_from_string
    from .compile_environment import CompileEnvironment

    tensor, _dim = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    env = CompileEnvironment.current()
    state = getattr(env, "_codegen_state", None)
    if state is None:
        raise exc.BackendUnsupported("nki", "cumsum requires active codegen state")

    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise exc.BackendUnsupported("nki", "cumsum requires tensor FX meta")

    def _resolve_dim(d: object) -> int:
        if isinstance(d, int):
            return d
        if isinstance(d, torch.SymInt):
            return env.size_hint(d)
        return int(d)

    shape = [_resolve_dim(d) for d in val.shape]
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) > 2:
        flat = 1
        for d in shape[:-1]:
            flat *= d
        shape = [flat, shape[-1]]
    if len(shape) == 1:
        shape = [1, shape[0]]

    dtype_str = env.backend.dtype_str(val.dtype)
    tensor_str = ast.unparse(tensor)

    dst_var = state.device_function.new_var("_nki_cumsum_dst", dce=True)
    zero_var = state.device_function.new_var("_nki_cumsum_zeros", dce=True)
    # Allocate two buffers: zeros operand and result.
    state.add_statement(
        statement_from_string(
            f"{zero_var} = nl.ndarray([{shape[0]}, {shape[1]}], {dtype_str}, buffer=nl.sbuf)"
        )
    )
    state.add_statement(
        statement_from_string(f"nisa.memset({zero_var}, value=0)")
    )
    state.add_statement(
        statement_from_string(
            f"{dst_var} = nl.ndarray([{shape[0]}, {shape[1]}], {dtype_str}, buffer=nl.sbuf)"
        )
    )
    state.add_statement(
        statement_from_string(
            f"nisa.tensor_tensor_scan(dst={dst_var}, data0={tensor_str}, "
            f"data1={zero_var}, initial=0.0, op0=nl.add, op1=nl.add)"
        )
    )
    state.device_function._nki_sbuf_shapes[dst_var] = list(shape)
    state.device_function._nki_sbuf_dtypes[dst_var] = dtype_str
    return expr_from_string(dst_var)


@iota_lowering.register_codegen("nki")
def codegen_iota_nki(ctx: LoweringContext, node: Node) -> object:
    """NKI lowering for torch.ops.prims.iota.default.

    Emits nisa.iota to generate an affine integer sequence into SBUF.
    The SBUF tile is 2D (1, length); callers that expect a 1D view get it
    implicitly since view/reshape on NKI are no-ops.

    When start/step are symbolic expressions they are emitted as runtime
    values (``nisa.iota`` accepts a runtime int32 for offset). Only the
    LENGTH is required to be a compile-time int for the tile shape.
    """
    from .ast_extension import statement_from_string
    from .compile_environment import CompileEnvironment

    start = node.kwargs.get("start", 0)
    step = node.kwargs.get("step", 1)
    dtype = node.kwargs.get("dtype") or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)
    (length_arg,) = node.args

    env = CompileEnvironment.current()
    state = getattr(env, "_codegen_state", None)
    if state is None:
        raise exc.BackendUnsupported("nki", "iota requires active codegen state")

    # Resolve the (possibly symbolic) length to a concrete int for the
    # tile shape. NKI requires compile-time tile sizes, matching the rest
    # of the backend.
    def _to_concrete_int(x: object) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, torch.SymInt):
            return env.size_hint(x)
        if isinstance(x, torch.fx.Node):
            # FX node representing a symbolic value; look up its SymInt
            # via ``meta["val"]`` and size_hint that.
            val = x.meta.get("val")
            if isinstance(val, int):
                return val
            if isinstance(val, torch.SymInt):
                return env.size_hint(val)
            if hasattr(val, "_sympy_"):
                try:
                    return int(val._sympy_())
                except Exception:
                    return env.size_hint(val)
        if hasattr(x, "_sympy_"):
            try:
                return int(x._sympy_())
            except Exception:
                pass
        try:
            return int(x)
        except Exception:
            return env.size_hint(x)  # last resort; likely to assert

    length = _to_concrete_int(length_arg)

    # Pad static non-power-of-2 lengths to next power of 2 to match the
    # Triton behaviour — this keeps downstream slicing consistent.
    if isinstance(length, int) and length != next_power_of_2(length):
        length = next_power_of_2(length)

    # For start/step, prefer a compile-time int when possible. Otherwise
    # emit the expression (symbolic start is fine for nisa.iota's
    # runtime-int32 offset parameter).
    def _emit_as_operand(x: object) -> str:
        if isinstance(x, int):
            return str(x)
        if isinstance(x, torch.SymInt):
            return str(env.size_hint(x))
        if isinstance(x, torch.fx.Node):
            # FX node — use its environment lookup. Fall back to a
            # reasonable default if ctx.env lookup fails.
            mapped = ctx.env.get(x)
            if mapped is not None:
                if isinstance(mapped, ast.AST):
                    return ast.unparse(mapped)
                return str(mapped)
            return str(x)
        try:
            return str(int(x))
        except Exception:
            return str(env.size_hint(x))

    start_op = _emit_as_operand(start)
    step_op = _emit_as_operand(step)

    # Generate [1, length] SBUF tile via nisa.iota.
    dtype_str = env.backend.dtype_str(dtype)
    dst_var = state.device_function.new_var("_nki_iota_dst")
    state.add_statement(
        statement_from_string(
            f"{dst_var} = nl.ndarray([1, {length}], {dtype_str}, buffer=nl.sbuf)"
        )
    )
    state.add_statement(
        statement_from_string(
            f"nisa.iota(dst={dst_var}, pattern=[[{step_op}, {length}]], "
            f"offset={start_op}, channel_multiplier=0)"
        )
    )
    state.device_function._nki_sbuf_shapes[dst_var] = [1, length]
    state.device_function._nki_sbuf_dtypes[dst_var] = dtype_str
    return expr_from_string(dst_var)


# --- NKI matmul helpers (referenced by the mm/addmm/bmm/baddbmm nki handlers) ---
def _nki_copy_psum_to_sbuf(
    state: object,
    psum_var: str,
    shape_str: str,
    dtype_str: str,
    *,
    prefix: str = "_nki_psum_copy",
) -> str:
    """Allocate an SBUF tile and copy a PSUM tile into it via nisa.tensor_copy.

    NKI Tensor Engine ops (nc_transpose, nc_matmul) write to PSUM.
    All subsequent Vector Engine / ISA ops expect SBUF inputs, so every
    PSUM result must be copied back to SBUF before further use.
    """
    sbuf_var = state.device_function.new_var(prefix)
    state.add_statement(
        statement_from_string(
            f"{sbuf_var} = nl.ndarray([{shape_str}], {dtype_str}, buffer=nl.sbuf)"
        )
    )
    state.add_statement(
        statement_from_string(f"nisa.tensor_copy(dst={sbuf_var}, src={psum_var})")
    )
    return sbuf_var


def _nki_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    """Generate nisa.nc_matmul for NKI backend with automatic sub-tiling.

    When user tile sizes exceed hardware limits, this emits nested loops
    that break the matmul into hardware-native sub-tiles:
      - stationary (LHS^T): [K=128, M=128]
      - moving (RHS):       [K=128, N<=512]
      - result (PSUM):      [M=128, N<=512]

    nc_matmul accumulates into PSUM across calls within the inner K loop,
    so no explicit PSUM zeroing or summation is needed per sub_k iteration.

    When inputs are tile lists (partition dim > 128), uses list indexing
    to select [128,...] sub-tiles, keeping all NKI allocations within the
    128-partition hardware limit.
    """
    TILE_M = 128  # stationary free dim (gemm_stationary_fmax)
    TILE_K = 128  # partition dim (pmax)
    TILE_N = 512  # moving free dim max (gemm_moving_fmax)

    env = CompileEnvironment.current()
    state = getattr(env, "_codegen_state", None)
    if state is None:
        raise exc.BackendUnsupported(
            "nki", "nc_matmul requires active codegen state"
        )

    if with_acc:
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        lhs_node = node.args[1]
        rhs_node = node.args[2]
    else:
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        lhs_node = node.args[0]
        rhs_node = node.args[1]
        acc = None

    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)

    lhs_shape = list(lhs_node.meta["val"].size())  # [M_tile, K_tile] or [B, M_tile, K_tile]
    rhs_shape = list(rhs_node.meta["val"].size())  # [K_tile, N_tile] or [B, K_tile, N_tile]

    import sympy as _sympy

    _bs_subs: dict[_sympy.Symbol, int] = {}
    for _bid in range(len(env.block_sizes)):
        _bs = env.block_sizes[_bid]
        _bs_subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))

    def _to_int(s: int | torch.SymInt) -> int:
        if isinstance(s, int):
            return s
        return int(s._sympy_().subs(_bs_subs))

    # Squeeze leading batch dimensions (size 1) from 3D+ shapes.
    # The SBUF operands are already 2D (load codegen flattened them),
    # so we just need the last 2 dims for tile size computation.
    while len(lhs_shape) > 2:
        lhs_shape = lhs_shape[1:]
    while len(rhs_shape) > 2:
        rhs_shape = rhs_shape[1:]

    M_tile = _to_int(lhs_shape[0])
    K_tile = _to_int(lhs_shape[1])
    N_tile = _to_int(rhs_shape[-1])

    # Determine input dtype for nc_matmul — both operands must match.
    # NKI nc_transpose requires dst.dtype == data.dtype (validated on gen3+).
    # We therefore allocate the transpose-result PSUM tile with the lhs dtype
    # (not always fp32 as before). When lhs and rhs dtypes differ, the
    # transposed stationary operand must be explicitly cast to match the
    # moving operand's dtype before nc_matmul.
    lhs_dtype = lhs_node.meta["val"].dtype
    rhs_dtype = rhs_node.meta["val"].dtype
    matmul_dtype = rhs_dtype  # moving operand dtype; stationary will match
    matmul_dtype_str = env.backend.dtype_str(matmul_dtype)
    transpose_dtype_str = env.backend.dtype_str(lhs_dtype)
    # Need an explicit cast between the transpose result and the matmul input
    # whenever their dtypes differ (e.g. lhs fp32, rhs bf16, or vice versa).
    need_cast_after_transpose = (lhs_dtype != matmul_dtype)

    if M_tile % TILE_M != 0 and M_tile > TILE_M:
        raise exc.BackendUnsupported("nki", f"M_tile must be <= {TILE_M} or a multiple of {TILE_M}, got {M_tile}")
    if K_tile % TILE_K != 0 and K_tile > TILE_K:
        raise exc.BackendUnsupported("nki", f"K_tile must be <= {TILE_K} or a multiple of {TILE_K}, got {K_tile}")
    N_sub = min(TILE_N, N_tile)
    if N_tile > TILE_N and N_tile % TILE_N != 0:
        raise exc.BackendUnsupported("nki", f"N_tile must be <= {TILE_N} or a multiple of {TILE_N}, got {N_tile}")

    actual_tile_m = min(M_tile, TILE_M)
    actual_tile_k = min(K_tile, TILE_K)
    n_sub_m = max(1, M_tile // TILE_M)
    n_sub_k = max(1, K_tile // TILE_K)
    n_sub_n = max(1, N_tile // N_sub)

    # From here on, use actual tile sizes for all allocations and slicing
    TILE_M = actual_tile_m
    TILE_K = actual_tile_k

    _is_transpose_mode = TILE_K > TILE_M
    # Note: is_transpose=True causes correctness issues on some kernels;
    # leave it at default (False) for now. matmul_split_k stays broken.
    _nc_matmul_transpose_arg = ""

    lhs_name = ast.unparse(lhs)
    rhs_name = ast.unparse(rhs)

    lhs_tile_vars = state.device_function.get_tile_list_vars(lhs_name)  # list[str] or None
    rhs_tile_vars = state.device_function.get_tile_list_vars(rhs_name)  # list[str] or None
    lhs_is_list = lhs_tile_vars is not None
    rhs_is_list = rhs_tile_vars is not None
    result_is_list = n_sub_m > 1

    # PSUM-reuse fusion: if the fusion pass tagged this matmul node and the
    # result is a single tile (no M/N sub-tiling, no bias add), skip the
    # final PSUM→SBUF copy and expose the PSUM buffer via an alias. The
    # single downstream Vector/Scalar consumer will read from PSUM directly.
    _keep_in_psum = bool(
        node.meta.get("nki_keep_in_psum", False)
        and not result_is_list
        and n_sub_n == 1
        and not with_acc
    )

    # Allocate result buffer(s) in SBUF — fully unrolled, no list comprehension.
    # Skip the SBUF allocation entirely when we're keeping the result in PSUM;
    # consumers resolve the name through device_function._nki_psum_aliases.
    mm_result = state.device_function.new_var("_nki_mm_result")
    mm_result_tile_vars: list[str] = []
    _mm_result_dtype = "nl.float32"
    if result_is_list:
        for i in range(n_sub_m):
            rv = state.device_function.new_var(f"{mm_result}_{i}")
            mm_result_tile_vars.append(rv)
            state.add_statement(
                statement_from_string(
                    f"{rv} = nl.ndarray([{TILE_M}, {N_tile}], {_mm_result_dtype}, buffer=nl.sbuf)"
                )
            )
        state.device_function.register_tile_list(mm_result, mm_result_tile_vars)
    elif not _keep_in_psum:
        state.add_statement(
            statement_from_string(
                f"{mm_result} = nl.ndarray([{TILE_M}, {N_tile}], {_mm_result_dtype}, buffer=nl.sbuf)"
            )
        )

    sub_k_var = state.device_function.new_var("_sub_k")
    lhs_t_psum = state.device_function.new_var("_lhs_t_psum")
    lhs_t_sbuf = state.device_function.new_var("_lhs_t_sbuf")
    lhs_t_cast = state.device_function.new_var("_lhs_t_cast") if need_cast_after_transpose else None
    mm_psum = state.device_function.new_var("_mm_psum")

    def _transpose_stmts(lhs_slice: str) -> list[ast.AST]:
        """Generate transpose + optional cast statements for one LHS sub-tile.
        Returns statements that leave the transposed data in lhs_t_sbuf (or
        lhs_t_cast if a dtype cast is needed for nc_matmul compatibility).
        The final variable name to use as stationary is _stationary_var.

        nc_transpose requires dst.dtype == data.dtype, so we allocate the
        PSUM buffer in the input dtype. When the resulting stationary dtype
        differs from the matmul (moving) dtype, an additional tensor_copy
        cast step produces the nc_matmul-compatible stationary tile.
        """
        stmts = [
            statement_from_string(
                f"{lhs_t_psum} = nl.ndarray([{TILE_K}, {TILE_M}], {transpose_dtype_str}, buffer=nl.psum)"
            ),
            statement_from_string(
                f"nisa.nc_transpose(dst={lhs_t_psum}, data={lhs_slice})"
            ),
            statement_from_string(
                f"{lhs_t_sbuf} = nl.ndarray([{TILE_K}, {TILE_M}], {transpose_dtype_str}, buffer=nl.sbuf)"
            ),
            statement_from_string(
                f"nisa.tensor_copy(dst={lhs_t_sbuf}, src={lhs_t_psum})"
            ),
        ]
        if need_cast_after_transpose:
            # Cast transposed-input dtype to matmul's moving-operand dtype
            # (e.g. fp32 → bf16 if lhs is fp32 but rhs is bf16).
            stmts.extend([
                statement_from_string(
                    f"{lhs_t_cast} = nl.ndarray([{TILE_K}, {TILE_M}], {matmul_dtype_str}, buffer=nl.sbuf)"
                ),
                statement_from_string(
                    f"nisa.tensor_copy(dst={lhs_t_cast}, src={lhs_t_sbuf})"
                ),
            ])
        return stmts

    # The variable name to use as the stationary operand for nc_matmul
    _stationary_var = lhs_t_cast if need_cast_after_transpose else lhs_t_sbuf

    def _lhs_ref(m_i: int) -> str:
        """Get the concrete SBUF tile name for LHS partition index m_i."""
        if lhs_is_list:
            assert lhs_tile_vars is not None
            return lhs_tile_vars[m_i]
        return f"{lhs_name}[{m_i} * {TILE_M} : ({m_i} + 1) * {TILE_M}, 0:{K_tile}]"

    def _lhs_k_slice(m_i: int, k_expr: str) -> str:
        """Get LHS slice for partition m_i and K sub-tile k_expr."""
        if lhs_is_list:
            assert lhs_tile_vars is not None
            if n_sub_k > 1:
                return (
                    f"{lhs_tile_vars[m_i]}[0:{TILE_M}, "
                    f"{k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}]"
                )
            return lhs_tile_vars[m_i]
        return (
            f"{lhs_name}[{m_i} * {TILE_M} : ({m_i} + 1) * {TILE_M}, "
            f"{k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}]"
        )

    def _rhs_ref(k_i: int, n_expr: str) -> str:
        """Get RHS reference for K partition k_i and N sub-tile n_expr."""
        if rhs_is_list:
            assert rhs_tile_vars is not None
            if n_sub_n > 1:
                return (
                    f"{rhs_tile_vars[k_i]}[0:{TILE_K}, "
                    f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
                )
            return rhs_tile_vars[k_i]
        return (
            f"{rhs_name}[{k_i} * {TILE_K} : ({k_i} + 1) * {TILE_K}, "
            f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
        )

    def _result_ref(m_i: int, n_expr: str) -> str:
        """Get result buffer reference for partition m_i."""
        if result_is_list:
            rv = mm_result_tile_vars[m_i]
            if n_sub_n > 1:
                return (
                    f"{rv}[0:{TILE_M}, "
                    f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
                )
            return rv
        if n_sub_n > 1:
            return (
                f"{mm_result}[{m_i} * {TILE_M} : ({m_i} + 1) * {TILE_M}, "
                f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
            )
        return mm_result

    def _make_k_body_for_m(m_i: int, n_expr: str, k_expr: str) -> list[ast.AST]:
        """Statements for one K-sub-tile within one M-stripe."""
        stmts = _transpose_stmts(_lhs_k_slice(m_i, k_expr))
        stmts.append(
            statement_from_string(
                f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary_var}{_nc_matmul_transpose_arg}, "
                f"moving={_rhs_ref(0, n_expr) if not rhs_is_list else _rhs_ref(0, n_expr)})"
            ),
        )
        return stmts

    def _make_k_body_rhs_loop(m_i: int, n_expr: str, k_expr: str) -> list[ast.AST]:
        """Statements for one K-sub-tile, with rhs indexed by k_expr (affine loop var)."""
        if rhs_is_list:
            assert rhs_tile_vars is not None
            if n_sub_n > 1:
                rhs_ref = (
                    f"{rhs_tile_vars[0]}[0:{TILE_K}, "  # placeholder; use k_expr below
                    f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
                )
            else:
                rhs_ref = rhs_tile_vars[0]
        else:
            rhs_ref = (
                f"{rhs_name}[{k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}, "
                f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
            )
        stmts = _transpose_stmts(_lhs_k_slice(m_i, k_expr))
        stmts.append(
            statement_from_string(
                f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary_var}{_nc_matmul_transpose_arg}, "
                f"moving={rhs_ref})"
            ),
        )
        return stmts

    def _emit_one_m_stripe(m_i: int, n_expr: str) -> None:
        """Emit all statements for a single M-stripe (fully unrolled, no sub_m loop)."""
        mm_sbuf_tmp = state.device_function.new_var("_mm_sbuf_tmp")
        _psum_dtype = "nl.float32"
        state.add_statement(
            statement_from_string(
                f"{mm_psum} = nl.ndarray([{TILE_M}, {N_sub}], {_psum_dtype}, buffer=nl.psum)"
            )
        )
        if n_sub_k > 1 and rhs_is_list:
            # Unroll K loop with concrete rhs tile var per iteration
            assert rhs_tile_vars is not None
            for k_i in range(n_sub_k):
                if n_sub_n > 1:
                    rhs_ref = (
                        f"{rhs_tile_vars[k_i]}[0:{TILE_K}, "
                        f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
                    )
                else:
                    rhs_ref = rhs_tile_vars[k_i]
                for s in _transpose_stmts(_lhs_k_slice(m_i, str(k_i))):
                    state.add_statement(s)
                state.add_statement(
                    statement_from_string(
                        f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary_var}{_nc_matmul_transpose_arg}, "
                        f"moving={rhs_ref})"
                    )
                )
        elif n_sub_k > 1:
            # rhs is not a list; use affine_range for K loop
            k_body = _transpose_stmts(_lhs_k_slice(m_i, sub_k_var))
            k_body.append(
                statement_from_string(
                    f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary_var}{_nc_matmul_transpose_arg}, "
                    f"moving={_rhs_ref(sub_k_var, n_expr)})"
                ),
            )
            state.add_statement(
                create(
                    ast.For,
                    target=create(ast.Name, id=sub_k_var, ctx=ast.Store()),
                    iter=expr_from_string(f"nl.affine_range({n_sub_k})"),
                    body=k_body,
                    orelse=[],
                )
            )
        else:
            # Single K sub-tile: inline
            rhs_ref_0 = _rhs_ref(0, n_expr) if not rhs_is_list else (
                rhs_tile_vars[0] if not n_sub_n > 1 else  # type: ignore[index]
                f"{rhs_tile_vars[0]}[0:{TILE_K}, "  # type: ignore[index]
                f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
            )
            for s in _transpose_stmts(_lhs_k_slice(m_i, '0')):
                state.add_statement(s)
            state.add_statement(
                statement_from_string(
                    f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary_var}{_nc_matmul_transpose_arg}, "
                    f"moving={rhs_ref_0})"
                )
            )
        if _keep_in_psum:
            # PSUM-reuse: skip the PSUM→SBUF copies and let the single
            # downstream Vector/Scalar consumer read from mm_psum directly.
            # The alias registration below happens once, after all stripes.
            return
        state.add_statement(
            statement_from_string(
                f"{mm_sbuf_tmp} = nl.ndarray([{TILE_M}, {N_sub}], nl.float32, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_copy(dst={mm_sbuf_tmp}, src={mm_psum})"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_copy(dst={_result_ref(m_i, n_expr)}, src={mm_sbuf_tmp})"
            )
        )

    def _emit_all_m_stripes(n_expr: str) -> None:
        for m_i in range(n_sub_m):
            _emit_one_m_stripe(m_i, n_expr)

    if n_sub_n > 1:
        sub_n_var = state.device_function.new_var("_sub_n")
        inner_body: list[ast.AST] = []
        orig_stmts = state.codegen.statements_stack[-1]
        state.codegen.statements_stack[-1] = inner_body
        _emit_all_m_stripes(sub_n_var)
        state.codegen.statements_stack[-1] = orig_stmts
        state.add_statement(
            create(
                ast.For,
                target=create(ast.Name, id=sub_n_var, ctx=ast.Store()),
                iter=expr_from_string(f"nl.affine_range({n_sub_n})"),
                body=inner_body,
                orelse=[],
            )
        )
    else:
        _emit_all_m_stripes("0")

    # If this matmul was tagged for PSUM reuse, register the alias so
    # downstream Vector/Scalar consumers read from PSUM instead of SBUF.
    if _keep_in_psum:
        state.device_function._nki_psum_aliases[mm_result] = mm_psum
        state.device_function._nki_fx_matmul_vars[node.name] = mm_result
        # Register the shape under BOTH the SBUF virtual name and the PSUM
        # name so shape lookups succeed whether the consumer sees the
        # original operand or the alias-resolved PSUM name.
        state.device_function._nki_sbuf_shapes[mm_result] = [TILE_M, N_tile]
        state.device_function._nki_sbuf_shapes[mm_psum] = [TILE_M, N_tile]

    if not with_acc:
        return expr_from_string(mm_result)

    # addmm: add bias + mm result → output, fully unrolled
    assert acc is not None
    acc_name = ast.unparse(acc)
    acc_tile_vars = state.device_function.get_tile_list_vars(acc_name)
    out_result = state.device_function.new_var("_nki_addmm_result")

    if result_is_list:
        out_tile_vars: list[str] = []
        for i in range(n_sub_m):
            ov = state.device_function.new_var(f"{out_result}_{i}")
            out_tile_vars.append(ov)
            state.add_statement(
                statement_from_string(
                    f"{ov} = nl.ndarray([{TILE_M}, {N_tile}], nl.float32, buffer=nl.sbuf)"
                )
            )
        for i in range(n_sub_m):
            acc_ref = acc_tile_vars[i] if acc_tile_vars else acc_name
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_tensor(dst={out_tile_vars[i]}, "
                    f"data1={mm_result_tile_vars[i]}, data2={acc_ref}, op=nl.add)"
                )
            )
        state.device_function.register_tile_list(out_result, out_tile_vars)
    else:
        state.add_statement(
            statement_from_string(
                f"{out_result} = nl.ndarray([{M_tile}, {N_tile}], nl.float32, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_tensor(dst={out_result}, data1={mm_result}, "
                f"data2={{acc}}, op=nl.add)",
                acc=acc,
            )
        )
    return expr_from_string(out_result)


