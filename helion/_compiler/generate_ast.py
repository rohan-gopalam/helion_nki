from __future__ import annotations

import ast
import collections
import contextlib
from typing import TYPE_CHECKING
from typing import NamedTuple

from torch.utils._ordered_set import OrderedSet

from .. import exc
from ..language._decorators import is_api_func
from ..runtime.config import Config
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import dead_assignment_elimination
from .ast_read_writes import dead_expression_elimination
from .ast_read_writes import definitely_does_not_have_side_effects
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .helper_function import CodegenInterface
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .loop_dependency_checker import LoopDependencyChecker
from .program_id import ForEachProgramID
from .tile_strategy import DeviceGridState
from .tile_strategy import DeviceLoopState
from .variable_origin import ArgumentOrigin

if TYPE_CHECKING:
    from collections.abc import Iterator

    import sympy

    from ..runtime import Config
    from .host_function import HostFunction
    from .loop_dependency_checker import LoopDependencyChecker
    from .tile_strategy import DeviceLoopOrGridState
    from .type_propagation import TensorType


class GenerateAST(NodeVisitor, CodegenInterface):
    def __init__(self, func: HostFunction, config: Config) -> None:
        # Initialize NodeVisitor first
        NodeVisitor.__init__(self)

        # Initialize our attributes
        self.host_function = func
        self.host_statements: list[ast.AST] = []
        self.module_statements: list[ast.stmt] = []
        self.statements_stack: list[list[ast.AST]] = [self.host_statements]
        self.on_device = False
        self.active_device_loops: dict[int, list[DeviceLoopOrGridState]] = (
            collections.defaultdict(list)
        )
        self.current_grid_state: DeviceGridState | None = None
        self.next_else_block: list[ast.AST] | None = None
        # Track var=const for NKI tensor_scalar (avoids tensor_tensor when one op is scalar)
        self._var_to_constant: dict[str, int | float | bool] = {}
        self._nki_sbuf_constant_values: dict[str, int | float | bool] = {}
        self._nki_sbuf_alloc_depth: dict[str, int] = {}
        self.fx_node_to_ast: dict[object, ast.AST | tuple[ast.AST, ...]] = {}

        # Now create device function and initialize CodegenInterface
        self.device_function = DeviceFunction(
            f"_helion_{func.name}",
            config,
            self,
        )
        CodegenInterface.__init__(self, self.device_function)

    def offset_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.index_var(block_idx)

    def mask_var(self, block_idx: int) -> str | None:
        if loops := self.active_device_loops[block_idx]:
            return loops[-1].strategy.mask_var(block_idx)
        return None

    def _lower_nki_mod_assign(
        self, target: str, value: ast.BinOp
    ) -> list[ast.AST] | None:
        try:
            env = CompileEnvironment.current()
        except Exception:
            return None
        if env.backend.codegen_name != "nki":
            return None

        lhs = ast.unparse(value.left)
        rhs = ast.unparse(value.right)
        lhs_shape = self.device_function._nki_sbuf_shapes.get(lhs)
        if lhs_shape is None:
            lookup = lhs
            while "_copy" in lookup:
                lookup = lookup[: lookup.rfind("_copy")]
                lhs_shape = self.device_function._nki_sbuf_shapes.get(lookup)
                if lhs_shape is not None:
                    lhs = lookup
                    break
        if lhs_shape is None:
            block_size = self.device_function._nki_iota_block_sizes.get(lhs)
            if block_size is not None:
                try:
                    lhs_shape = [1, int(block_size)]
                except ValueError:
                    lhs_shape = None
        if lhs_shape is None:
            for block_id, loops in self.active_device_loops.items():
                if not loops:
                    continue
                strategy = loops[-1].strategy
                if strategy.index_var(block_id) != lhs:
                    continue
                try:
                    block_size = env.block_sizes[block_id].from_config_assert(
                        self.device_function.config
                    )
                    lhs_shape = [1, int(block_size)]
                except Exception:
                    lhs_shape = None
                break
        if lhs_shape is None or len(lhs_shape) < 2:
            return None

        rhs_value: int | float | None = None
        rhs_value = self._constant_value_from_ast(value.right)
        if rhs_value == 0:
            return None

        tmp = self.device_function.new_var("_nki_mod_div_f32", dce=True)
        q = self.device_function.new_var("_nki_mod_div_i32", dce=True)
        prod = self.device_function.new_var("_nki_mod_prod", dce=True)
        self.device_function._nki_sbuf_shapes[tmp] = list(lhs_shape)
        self.device_function._nki_sbuf_shapes[q] = list(lhs_shape)
        self.device_function._nki_sbuf_shapes[prod] = list(lhs_shape)
        self.device_function._nki_sbuf_shapes[target] = list(lhs_shape)
        self.device_function._nki_sbuf_dtypes[tmp] = "nl.float32"
        self.device_function._nki_sbuf_dtypes[q] = "nl.int32"
        self.device_function._nki_sbuf_dtypes[prod] = "nl.int32"
        self.device_function._nki_sbuf_dtypes[target] = "nl.int32"

        shape = ", ".join(str(dim) for dim in lhs_shape)
        if rhs_value is not None:
            inv: object = 1.0 / float(rhs_value)
            rhs_operand: object = (
                int(rhs_value) if float(rhs_value).is_integer() else rhs_value
            )
        else:
            inv = f"1.0 / ({rhs})"
            rhs_operand = rhs

        return [
            statement_from_string(
                f"{tmp} = nl.ndarray([{shape}], nl.float32, buffer=nl.sbuf)"
            ),
            statement_from_string(
                f"nisa.tensor_scalar(dst={tmp}, data={lhs}, "
                f"op0=nl.multiply, operand0={inv})"
            ),
            statement_from_string(
                f"{prod} = nl.ndarray([{shape}], nl.int32, buffer=nl.sbuf)"
            ),
            statement_from_string(f"{q} = nl.floor({tmp}, dtype=nl.int32)"),
            statement_from_string(
                f"nisa.tensor_scalar(dst={prod}, data={q}, "
                f"op0=nl.multiply, operand0={rhs_operand})"
            ),
            statement_from_string(
                f"{target} = nl.ndarray([{shape}], nl.int32, buffer=nl.sbuf)"
            ),
            statement_from_string(
                f"nisa.tensor_tensor(dst={target}, data1={lhs}, data2={prod}, "
                f"op=nl.subtract)"
            ),
        ]

    def _phase_checker(self, root_id: int) -> LoopDependencyChecker:
        phase_idx = self.host_function.device_ir.phase_for_root(root_id)
        return self.host_function.device_ir.phases[phase_idx].loop_dependency_checker

    def _record_nki_sbuf_allocation(self, stmt: ast.Assign) -> None:
        if (
            len(stmt.targets) != 1
            or not isinstance(stmt.targets[0], ast.Name)
            or not isinstance(stmt.value, ast.Call)
            or not isinstance(stmt.value.func, ast.Attribute)
            or not isinstance(stmt.value.func.value, ast.Name)
            or stmt.value.func.value.id != "nl"
            or stmt.value.func.attr != "ndarray"
        ):
            return
        if not any(
            keyword.arg == "buffer"
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "nl"
            and keyword.value.attr == "sbuf"
            for keyword in stmt.value.keywords
        ):
            return
        self._nki_sbuf_alloc_depth[stmt.targets[0].id] = len(self.statements_stack)
        self._nki_sbuf_constant_values.pop(stmt.targets[0].id, None)

    def _constant_value_from_ast(
        self, expr: ast.AST
    ) -> int | float | bool | None:
        if isinstance(expr, ast.Constant) and isinstance(
            expr.value, (int, float, bool)
        ):
            return expr.value
        if (
            isinstance(expr, ast.UnaryOp)
            and isinstance(expr.operand, ast.Constant)
            and isinstance(expr.operand.value, (int, float))
        ):
            if isinstance(expr.op, ast.USub):
                return -expr.operand.value
            if isinstance(expr.op, ast.UAdd):
                return expr.operand.value
        if isinstance(expr, ast.Name):
            const = self.get_var_constant_value(expr.id)
            if isinstance(const, (int, float, bool)):
                return const
            return self._nki_sbuf_constant_values.get(expr.id)
        if isinstance(expr, ast.BinOp):
            left = self._constant_value_from_ast(expr.left)
            right = self._constant_value_from_ast(expr.right)
            if not isinstance(left, (int, float, bool)) or not isinstance(
                right, (int, float, bool)
            ):
                return None
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            if isinstance(expr.op, ast.Div) and right != 0:
                return left / right
            if isinstance(expr.op, ast.FloorDiv) and right != 0:
                return left // right
            if isinstance(expr.op, ast.Mod) and right != 0:
                return left % right
        return None

    def _record_nki_sbuf_write(self, stmt: ast.AST) -> None:
        call: ast.Call | None = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        if call is None:
            return

        written_names: list[str] = []
        for keyword in call.keywords:
            if keyword.arg == "dst" and isinstance(keyword.value, ast.Name):
                written_names.append(keyword.value.id)
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "nisa"
            and call.func.attr == "memset"
            and call.args
            and isinstance(call.args[0], ast.Name)
        ):
            written_names.append(call.args[0].id)

        for name in written_names:
            self._nki_sbuf_constant_values.pop(name, None)

        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "nisa"
            and call.func.attr == "memset"
            and call.args
            and isinstance(call.args[0], ast.Name)
        ):
            return

        value_expr = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "value"),
            None,
        )
        if value_expr is None:
            return
        value = self._constant_value_from_ast(value_expr)
        if isinstance(value, (int, float, bool)):
            self._nki_sbuf_constant_values[call.args[0].id] = value

    def _lower_nki_sbuf_reassign(
        self, target: str, source: str
    ) -> list[ast.AST] | None:
        if not self.on_device:
            return None
        try:
            env = CompileEnvironment.current()
        except Exception:
            return None
        if env.backend.codegen_name != "nki":
            return None
        if target == source or not target.startswith("_nki") or "_copy" in target:
            return None

        target_depth = self._nki_sbuf_alloc_depth.get(target)
        if target_depth is None or target_depth > len(self.statements_stack):
            return None

        sbuf_shapes = self.device_function._nki_sbuf_shapes
        target_shape = sbuf_shapes.get(target)
        source_shape = sbuf_shapes.get(source)
        if target_shape is None or source_shape is None or target_shape != source_shape:
            return None

        sbuf_dtypes = self.device_function._nki_sbuf_dtypes
        target_dtype = sbuf_dtypes.get(target)
        source_dtype = sbuf_dtypes.get(source)
        if (
            target_dtype is not None
            and source_dtype is not None
            and target_dtype != source_dtype
        ):
            return None

        return [statement_from_string(f"nisa.tensor_copy(dst={target}, src={source})")]

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        if stmt is None:
            return
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        # Track var=const for NKI tensor_scalar scalar operand resolution
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.BinOp)
            and isinstance(stmt.value.op, ast.Mod)
            and (lowered := self._lower_nki_mod_assign(stmt.targets[0].id, stmt.value))
            is not None
        ):
            self.statements_stack[-1].extend(lowered)
            return
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
            and (
                lowered := self._lower_nki_sbuf_reassign(
                    stmt.targets[0].id, stmt.value.id
                )
            )
            is not None
        ):
            self.statements_stack[-1].extend(lowered)
            return
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            const_value = self._constant_value_from_ast(stmt.value)
            if isinstance(const_value, (int, float, bool)):
                self._var_to_constant[stmt.targets[0].id] = const_value
            elif (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "float"
                and len(stmt.value.args) == 1
                and isinstance(stmt.value.args[0], ast.Constant)
                and isinstance(stmt.value.args[0].value, str)
            ):
                self._var_to_constant[stmt.targets[0].id] = float(
                    stmt.value.args[0].value
                )
            self._record_nki_sbuf_allocation(stmt)
        self._record_nki_sbuf_write(stmt)
        self.statements_stack[-1].append(stmt)

    def get_var_constant_value(self, var_name: str) -> int | float | bool | None:
        """Return constant value if var was assigned a literal, else None."""
        return self._var_to_constant.get(
            var_name, self._nki_sbuf_constant_values.get(var_name)
        )

    def record_fx_node_ast(self, node: object, value: object) -> None:
        if isinstance(value, ast.AST):
            self.fx_node_to_ast[node] = value
        elif isinstance(value, tuple) and all(isinstance(v, ast.AST) for v in value):
            self.fx_node_to_ast[node] = value

    def ast_for_fx_node(self, node: object) -> ast.AST | tuple[ast.AST, ...] | None:
        return self.fx_node_to_ast.get(node)

    def get_rng_seed_buffer_statements(self) -> list[ast.AST]:
        import_stmt = statement_from_string(
            "from torch._inductor import inductor_prims"
        )

        # Create host-side seed buffer with the required number of seeds
        seed_buffer_stmt = statement_from_string(
            f"_rng_seed_buffer = inductor_prims.seeds({self.device_function.rng_seed_count}, torch.accelerator.current_accelerator())"
        )

        return [import_stmt, seed_buffer_stmt]

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "v") -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod):
            try:
                env = CompileEnvironment.current()
                if env.backend.codegen_name == "nki":
                    from .backend import NKIOpOverrides

                    lhs = ast.unparse(expr.left)
                    rhs = ast.unparse(expr.right)
                    lowered = NKIOpOverrides.mod(lhs, rhs)
                    if lowered != f"{lhs} % {rhs}" and lowered.isidentifier():
                        return create(ast.Name, id=lowered, ctx=ast.Load())
            except Exception:
                pass
        with expr:
            varname = self.tmpvar(dce=dce, prefix=prefix)
            self.add_statement(
                statement_from_string(f"{varname} = {{expr}}", expr=expr)
            )
            return create(ast.Name, id=varname, ctx=ast.Load())

    def lift_symnode(
        self,
        expr: ast.AST,
        sym_expr: sympy.Expr,
        *,
        dce: bool = False,
        prefix: str = "symnode",
    ) -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr

        target_statements = self.statements_stack[-1]
        env = CompileEnvironment.current()
        # Identify every block dimension the symbolic value depends on so we know
        # which loop nests the expression depends on.
        dep_block_ids = {
            block_id
            for symbol in sym_expr.free_symbols
            if (block_id := env.get_block_id(symbol)) is not None
        }

        # Walk outward through the active device loops: as soon as we see a loop
        # whose block id appears in the dependency set we must stop, otherwise we
        # can safely hoist into that loop's outer prefix (which executes before the
        # loop body).
        for loop_state in reversed(self._active_loop_stack()):
            if dep_block_ids.intersection(loop_state.block_ids):
                break
            target_statements = loop_state.outer_prefix

        with expr:
            varname = self.tmpvar(dce=dce, prefix=prefix)
            # Emit the temporary into the chosen statement list so the symbolic
            # expression is computed exactly once at the appropriate scope.
            target_statements.append(
                statement_from_string(f"{varname} = {{expr}}", expr=expr)
            )
            # Reuse the temporary everywhere else in the kernel body.
            return create(ast.Name, id=varname, ctx=ast.Load())

    def _active_loop_stack(self) -> list[DeviceLoopState]:
        seen: set[int] = set()
        stack: list[DeviceLoopState] = []
        for loops in self.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, DeviceLoopState):
                    continue
                key = id(loop_state)
                if key not in seen:
                    stack.append(loop_state)
                    seen.add(key)
        return stack

    @contextlib.contextmanager
    def set_statements(self, new_statements: list[ast.AST] | None) -> Iterator[None]:
        if new_statements is None:
            yield
        else:
            expr_to_var_info = self.device_function.expr_to_var_info
            # We don't want to reuse vars assigned in a nested scope, so copy it
            self.device_function.expr_to_var_info = expr_to_var_info.copy()
            self.statements_stack.append(new_statements)
            try:
                yield
            finally:
                self.statements_stack.pop()
                self.device_function.expr_to_var_info = expr_to_var_info

    @contextlib.contextmanager
    def set_on_device(self) -> Iterator[None]:
        assert self.on_device is False
        self.on_device = True
        prior = self.host_statements
        self.host_statements = self.statements_stack[-1]
        try:
            yield
        finally:
            self.on_device = False
            self.host_statements = prior

    @contextlib.contextmanager
    def add_device_loop(self, device_loop: DeviceLoopState) -> Iterator[None]:
        with self.set_statements(device_loop.inner_statements):
            for idx in device_loop.block_ids:
                active_loops = self.active_device_loops[idx]
                active_loops.append(device_loop)
                if len(active_loops) > 1:
                    raise exc.NestedDeviceLoopsConflict
            try:
                yield
            finally:
                for idx in device_loop.block_ids:
                    self.active_device_loops[idx].pop()
                # DISABLED: this cleanup was causing jagged_layer_norm variance
                # pass to take an unmasked codegen path.  The _is_outer_jagged
                # detection in tile_strategy.py already uses active_device_loops
                # (not _nki_dyn_loops) for nesting detection, so this cleanup
                # is not needed for sequential-sibling differentiation.
                # TODO: investigate whether this can be re-enabled safely.
                # _dyn_loops_ga = getattr(self.device_function, "_nki_dyn_loops", None)
                # if _dyn_loops_ga is not None:
                #     for _bid_ga in device_loop.block_ids:
                #         _dyn_loops_ga.pop(_bid_ga, None)
        self.statements_stack[-1].extend(device_loop.outer_prefix)
        self.add_statement(device_loop.for_node)
        self.statements_stack[-1].extend(device_loop.outer_suffix)

    def set_active_loops(self, device_grid: DeviceLoopOrGridState) -> None:
        self.current_grid_state = (
            device_grid if isinstance(device_grid, DeviceGridState) else None
        )
        for idx in device_grid.block_ids:
            self.active_device_loops[idx] = [device_grid]

    def generic_visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        fields = {}
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                fields[field] = new_list = []
                with self.set_statements(
                    new_list
                    if old_value and isinstance(old_value[0], ast.stmt)
                    else None
                ):
                    for item in old_value:
                        new_list.append(self.visit(item))  # mutation in visit
            elif isinstance(old_value, ast.AST):
                fields[field] = self.visit(  # pyrefly: ignore[unsupported-operation]
                    old_value
                )
            else:
                fields[field] = old_value
        # pyrefly: ignore[bad-return, bad-argument-type]
        return node.new(fields)

    def visit_For(self, node: ast.For) -> ast.AST | None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            assert not node.orelse

            assert node._root_id is not None
            # Loop dependency checks were already run during lowering; phase checker kept for symmetry/debug.
            self._phase_checker(node._root_id)
            env = CompileEnvironment.current()
            is_nki = env.backend.name == "nki"

            if len(self.host_function.device_ir.root_ids) == 1 or is_nki:
                body = self.device_function.body
            else:
                assert len(self.host_function.device_ir.root_ids) > 1
                # Multiple top level for loops

                if node._root_id == 0:
                    self.device_function.set_pid(
                        ForEachProgramID(
                            self.device_function.new_var("pid_shared", dce=False),
                        )
                    )
                    self.device_function.body.extend(
                        # pyrefly: ignore [missing-attribute]
                        self.device_function.pid.codegen_pid_init()
                    )
                if node._root_id < len(self.host_function.device_ir.root_ids) - 1:
                    body = []
                else:
                    # This is the last top level for, dont emit more if statements
                    assert self.next_else_block is not None
                    body = self.next_else_block
            with (
                self.set_on_device(),
                self.set_statements(body),
            ):
                iter_node = node.iter
                assert isinstance(iter_node, ExtendedAST)
                with iter_node:
                    assert isinstance(iter_node, ast.Call)
                    args = []
                    kwargs = {}
                    for arg_node in iter_node.args:
                        assert not isinstance(arg_node, ast.Starred)
                        assert isinstance(arg_node, ExtendedAST)
                        assert arg_node._type_info is not None
                        args.append(arg_node._type_info.proxy())
                    for kwarg_node in iter_node.keywords:
                        assert kwarg_node.arg is not None
                        assert isinstance(kwarg_node.value, ExtendedAST)
                        assert kwarg_node.value._type_info is not None
                        kwargs[kwarg_node.arg] = kwarg_node.value._type_info.proxy()
                    fn_node = iter_node.func
                    assert isinstance(fn_node, ExtendedAST)
                    assert fn_node._type_info is not None
                    fn = fn_node._type_info.proxy()
                    assert is_api_func(fn)
                    env = CompileEnvironment.current()
                    codegen_fn = fn._codegen.get(env.codegen_name)
                    if codegen_fn is None:
                        codegen_fn = fn._codegen.get("common")
                    if codegen_fn is None:
                        raise exc.BackendImplementationMissing(
                            env.backend_name,
                            f"codegen for API function {fn.__qualname__}",
                        )
                    bound = fn._signature.bind(*args, **kwargs)
                    bound.apply_defaults()

                    from .inductor_lowering import CodegenState

                    state = CodegenState(
                        self,
                        fx_node=None,
                        proxy_args=[*bound.arguments.values()],
                        # pyrefly: ignore [bad-argument-type]
                        ast_args=None,
                    )

                    codegen_fn(state)
                assert node._root_id is not None
                root = self.host_function.device_ir.get_root(
                    self.device_function.config,
                    self.host_function.device_ir.root_ids[node._root_id],
                )
                grid_state = self.current_grid_state
                if env.backend.name == "nki":
                    env.backend.validate_nki_tensor_shapes(root)
                    # Run FX-graph fusion passes after validation: annotate
                    # nodes with metadata that downstream codegen consumes
                    # (e.g. matmul→activation PSUM reuse). Passes are read-only
                    # on graph structure; they only set node.meta entries.
                    from .nki_fusion import annotate_fx_graph
                    annotate_fx_graph(root)
                with env.set_codegen_state(state):
                    if (
                        isinstance(grid_state, DeviceGridState)
                        and grid_state.has_lane_loops()
                    ):
                        wrapped_body: list[ast.AST] = []
                        with self.set_statements(wrapped_body):
                            codegen_call_with_graph(self, root, [])
                        self.statements_stack[-1].extend(
                            grid_state.wrap_body(wrapped_body)
                        )
                    else:
                        codegen_call_with_graph(self, root, [])

                # Flush deferred RDIM definitions now that block sizes are determined
                # This ensures block size and rdim vars are defined in the correct order
                self.device_function.flush_deferred_rdim_defs(self)

                if isinstance(self.device_function.pid, ForEachProgramID):
                    self.device_function.pid.case_phases.append(
                        self.host_function.device_ir.phase_for_root(node._root_id)
                    )

                # If we are in a multi top level loop, for all loops except for the last one
                # emit ifthenelse blocks
                if (
                    not is_nki
                    and node._root_id < len(self.host_function.device_ir.root_ids) - 1
                ):
                    block = (
                        self.device_function.body
                        if self.next_else_block is None
                        else self.next_else_block
                    )
                    self.next_else_block = []
                    block.append(
                        create(
                            ast.If,
                            # pyrefly: ignore [missing-attribute]
                            test=self.device_function.pid.codegen_test(state),
                            body=body,
                            orelse=self.next_else_block,
                        )
                    )
            if node._root_id == len(self.host_function.device_ir.root_ids) - 1:
                if self.device_function.pid is not None:
                    persistent_body = self.device_function.pid.setup_persistent_kernel(
                        self.device_function
                    )
                    if persistent_body is not None:
                        # pyrefly: ignore [bad-assignment]
                        self.device_function.body = persistent_body
                self.device_function.dead_code_elimination()
                if not self.device_function.preamble and not self.device_function.body:
                    raise exc.EmptyDeviceLoopAfterDCE
                call_stmt = self.device_function.codegen_function_call()
                post_stmts = getattr(
                    self.device_function, "_nki_post_call_stmts", None
                )
                if post_stmts:
                    self.add_statement(call_stmt)
                    for s in post_stmts:
                        self.add_statement(s)
                    return None
                return call_stmt
            return None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        if isinstance(node.ctx, ast.Load) and node._type_info is not None:
            origin = node._type_info.origin
            if (
                isinstance(origin, ArgumentOrigin)
                and origin.name in self.host_function.constexpr_args
            ):
                return expr_from_string(
                    repr(self.host_function.constexpr_args[origin.name])
                )
            if origin.needs_rename():
                # `x` => `_source_module.x`
                return expr_from_string(origin.host_str())
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        from .type_propagation import CallableType
        from .type_propagation import SequenceType
        from .type_propagation import TileIndexType

        func_node = node.func
        assert isinstance(func_node, ExtendedAST)

        assert isinstance(node, ExtendedAST)
        env = CompileEnvironment.current()
        if self.on_device:
            pass
        elif isinstance(type_info := node._type_info, TileIndexType):
            block_info = env.block_sizes[type_info.block_id]
            return expr_from_string(
                self.host_function.literal_expr(
                    block_info.from_config(self.device_function.config)
                )
            )
        elif isinstance(type_info, SequenceType) and all(
            isinstance(x, TileIndexType) for x in type_info.unpack()
        ):
            values = type_info.unpack()
            # pyrefly: ignore [missing-attribute]
            block_infos = [env.block_sizes[x.block_id] for x in values]
            return expr_from_string(
                self.host_function.literal_expr(
                    [x.from_config(self.device_function.config) for x in block_infos]
                )
            )
        elif isinstance(fn_type_info := func_node._type_info, CallableType) and (
            is_api_func(api := fn_type_info.value)
        ):
            codegen_fn = api._codegen.get(env.codegen_name)
            if codegen_fn is None:
                codegen_fn = api._codegen.get("common")
            if codegen_fn is None:
                raise exc.BackendImplementationMissing(
                    env.backend_name,
                    f"codegen for API function {api.__qualname__}",
                )
            ast_args = []
            ast_kwargs = {}
            proxy_args = []
            proxy_kwargs = {}
            for arg in node.args:
                assert not isinstance(arg, ast.Starred)
                assert isinstance(arg, ExtendedAST)
                assert arg._type_info is not None
                ast_args.append(arg)
                proxy_args.append(arg._type_info.proxy())
            for kwarg in node.keywords:
                assert kwarg.arg is not None
                assert isinstance(kwarg.value, ExtendedAST)
                assert kwarg.value._type_info is not None
                ast_kwargs[kwarg.arg] = kwarg.value
                proxy_kwargs[kwarg.arg] = kwarg.value._type_info.proxy()
            ast_params = api._signature.bind(*ast_args, **ast_kwargs)
            proxy_params = api._signature.bind(*proxy_args, **proxy_kwargs)
            ast_params.apply_defaults()
            proxy_params.apply_defaults()
            # pyrefly: ignore [bad-return]
            return codegen_fn(
                CodegenState(
                    self,
                    None,
                    proxy_args=[*proxy_params.arguments.values()],
                    ast_args=[*ast_params.arguments.values()],
                )
            )
        return self.generic_visit(node)

    def host_dead_code_elimination(self) -> None:
        dce_vars: OrderedSet[str] = OrderedSet()
        for stmt in self.host_statements:
            if (
                isinstance(stmt, ast.Assign)
                and definitely_does_not_have_side_effects(stmt.value)
                and all(isinstance(name, ast.Name) for name in stmt.targets)
            ):
                for name in stmt.targets:
                    assert isinstance(name, ast.Name)
                    dce_vars.add(name.id)

        dead_assignment_elimination(self.host_statements, list(dce_vars))
        dead_expression_elimination(self.host_statements)


class TensorReference(NamedTuple):
    node: ast.AST
    name: str
    type_info: TensorType

    @property
    def is_host(self) -> bool:
        return self.type_info.origin.is_host()


class SubscriptIndexing(NamedTuple):
    tensor_ref: TensorReference
    index_expr: ast.AST
    mask_expr: ast.AST

    def has_mask(self) -> bool:
        return not (
            isinstance(self.mask_expr, ast.Constant) and self.mask_expr.value is None
        )


def emit_main_def() -> ast.stmt:
    return statement_from_string("""
if __name__ == "__main__":
    call()
    """)


def generate_ast(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.AST:
    with func:
        if len(func.device_ir.phases) > 1:
            if not str(config.pid_type).startswith("persistent"):
                raise exc.BarrierRequiresPersistent(config.pid_type)
        codegen = GenerateAST(func, config)
        with codegen.device_function:
            for stmt in func.body:
                codegen.add_statement(codegen.visit(stmt))
            kernel_def = codegen.device_function.codegen_function_def()
            codegen.host_dead_code_elimination()

            # Inject RNG seed buffer creation if needed
            rng_statements = (
                codegen.get_rng_seed_buffer_statements()
                if codegen.device_function.has_rng_ops()
                else []
            )
            final_host_statements = rng_statements + codegen.host_statements

            host_def = func.codegen_function_def(final_host_statements)

            call_def = []
            main_def = []
            if emit_repro_caller:
                call_def = [func.codegen_call_function()]
                main_def = [emit_main_def()]

            result = ast.Module(
                [
                    *func.codegen_imports(),
                    *codegen.module_statements,
                    *codegen.device_function.codegen_helper_functions(),
                    *kernel_def,
                    host_def,
                    *call_def,
                    *main_def,
                ],
                [],
            )
            # break circular reference for better GC
            del codegen.device_function.codegen
            return result
