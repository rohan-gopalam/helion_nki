from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import exc
from .._compat import min_dot_size
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import format_shape
from .._compiler.matmul_utils import _compute_out_dtype
from .._compiler.matmul_utils import _emit_tl_dot_scaled
from .._compiler.matmul_utils import emit_tl_dot_with_padding
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState


@_decorators.api(is_device_only=True)
def dot(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Performs a matrix multiplication of tensors with support for multiple dtypes.

    This operation performs matrix multiplication with inputs of various dtypes including
    float16, bfloat16, float32, int8, and FP8 formats (e4m3fn, e5m2). The computation is
    performed with appropriate precision based on the input dtypes.

    Args:
        mat1: First matrix (2D or 3D tensor of torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.float8_e4m3fn, or torch.float8_e5m2)
        mat2: Second matrix (2D or 3D tensor of torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.float8_e4m3fn, or torch.float8_e5m2)
        acc: The accumulator tensor (2D or 3D tensor of torch.float16, torch.float32, or torch.int32).
             If not None, the result is added to this tensor.
             If None, a new tensor is created with appropriate dtype based on inputs.
        out_dtype: Optional dtype that controls the output type of the multiplication prior
            to any accumulation. This maps directly to the Triton ``tl.dot`` ``out_dtype``
            argument and overrides the default promotion rules when provided.

    Returns:
        Result of matrix multiplication. If acc is provided, returns acc + (mat1 @ mat2).
        Otherwise returns (mat1 @ mat2) with promoted dtype.

    Example:
        >>> # FP8 example
        >>> a = torch.randn(32, 64, device="cuda").to(torch.float8_e4m3fn)
        >>> b = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
        >>> c = torch.zeros(32, 128, device="cuda", dtype=torch.float32)
        >>> result = hl.dot(a, b, acc=c)  # result is c + (a @ b)

        >>> # Float16 example
        >>> a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        >>> b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        >>> result = hl.dot(a, b)  # result dtype will be torch.float16

        >>> # Int8 example
        >>> a = torch.randint(-128, 127, (32, 64), device="cuda", dtype=torch.int8)
        >>> b = torch.randint(-128, 127, (64, 128), device="cuda", dtype=torch.int8)
        >>> acc = torch.zeros(32, 128, device="cuda", dtype=torch.int32)
        >>> result = hl.dot(a, b, acc=acc)  # int8 x int8 -> int32
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.dtype | None]:
    # Define supported dtypes
    supported_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )

    # Validate input types
    if mat1.dtype not in supported_dtypes:
        raise TypeError(
            f"hl.dot: mat1 must be one of {[str(d) for d in supported_dtypes]}, got {mat1.dtype}"
        )
    if mat2.dtype not in supported_dtypes:
        raise TypeError(
            f"hl.dot: mat2 must be one of {[str(d) for d in supported_dtypes]}, got {mat2.dtype}"
        )

    # Validate shapes for matrix multiplication
    if mat1.ndim not in (2, 3):
        raise ValueError(f"hl.dot: mat1 must be 2D or 3D tensor, got {mat1.ndim}D")
    if mat2.ndim not in (2, 3):
        raise ValueError(f"hl.dot: mat2 must be 2D or 3D tensor, got {mat2.ndim}D")

    # Check matrix multiplication compatibility
    if mat1.shape[-1] != mat2.shape[-2]:
        raise ValueError(
            f"hl.dot: incompatible matrix dimensions for multiplication: "
            f"{mat1.shape} @ {mat2.shape}"
        )

    # Check batch dimension compatibility (broadcastable or matching) if any input is 3D
    if mat1.ndim == 3 or mat2.ndim == 3:
        from itertools import zip_longest

        batch_shape_1 = mat1.shape[:-2] if mat1.ndim > 2 else ()
        batch_shape_2 = mat2.shape[:-2] if mat2.ndim > 2 else ()

        for lhs_dim, rhs_dim in zip_longest(
            reversed(batch_shape_1), reversed(batch_shape_2), fillvalue=1
        ):
            # Allow broadcasting with 1
            if str(lhs_dim) == "1" or str(rhs_dim) == "1":
                continue
            # Check if dimensions match
            if str(lhs_dim) != str(rhs_dim):
                raise exc.DotBatchDimensionMismatch(
                    lhs=format_shape(batch_shape_1),
                    rhs=format_shape(batch_shape_2),
                )

    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        raise TypeError(
            f"hl.dot: out_dtype must be a torch.dtype or None, got {type(out_dtype)}"
        )

    # Validate accumulator if provided
    if acc is not None:
        # Allow int32 accumulator for int8 inputs
        valid_acc_dtypes = (torch.float16, torch.float32, torch.int32)
        if acc.dtype not in valid_acc_dtypes:
            raise TypeError(
                f"hl.dot: acc must be one of {[str(d) for d in valid_acc_dtypes]}, got {acc.dtype}"
            )

        # Check int8 inputs require int32 accumulator
        if mat1.dtype == torch.int8 or mat2.dtype == torch.int8:
            if acc.dtype != torch.int32:
                raise TypeError(
                    f"hl.dot: int8 inputs require int32 accumulator, got {acc.dtype}"
                )

        # Check accumulator shape compatibility
        expected_shape = list(mat1.shape)
        expected_shape[-1] = mat2.shape[-1]

        if acc.ndim not in (2, 3):
            raise ValueError(f"hl.dot: acc must be 2D or 3D tensor, got {acc.ndim}D")

        if list(acc.shape) != expected_shape:
            raise ValueError(
                f"hl.dot: acc shape {list(acc.shape)} incompatible with result shape {expected_shape}"
            )

    # Apply min-dot-size constraints so autotuner won't pick invalid block_size
    enforce_dot_requirements(mat1, mat2)

    return (mat1, mat2, acc, out_dtype)


def enforce_dot_requirements(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    """Update config-spec min/max sizes for a dot/matmul.

    This ensures the autotuner does not select block sizes below the hardware
    minimums for the current device and dtypes, and constrains the batch
    dimension block size to 1 for 3D operands since Triton does not support
    3D dot operations.
    """

    # Last two dims are used for matmul
    lshape = lhs.size()
    rshape = rhs.size()
    m, k = lshape[-2], lshape[-1]
    k2, n = rshape[-2], rshape[-1]
    assert k == k2, f"Mismatched K dimensions for dot: {k} vs {k2}"

    a, b, c = min_dot_size(lhs.device, lhs.dtype, rhs.dtype)
    env = CompileEnvironment.current()
    for shape, min_size in ((m, a), (n, b), (k, c)):
        block_idx = env.get_block_id(shape)
        if block_idx is not None:
            env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)

    # Triton only supports 2D dot operations.  When the operands are 3D
    # (batched matmul), constrain the batch dimension block size to 1 so
    # the codegen can squeeze it away before emitting tl.dot.
    if len(lshape) == 3:
        for batch_dim in (lshape[0], rshape[0]):
            block_idx = env.get_block_id(batch_dim)
            if block_idx is not None:
                env.block_sizes[block_idx].update_max_block(1)


@_decorators.register_fake(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # Matrix multiplication shape computation
    result_shape = list(mat1.shape)
    result_shape[-1] = mat2.shape[-1]

    if acc is not None:
        return acc.new_empty(result_shape)

    # Determine output dtype using the helper function
    resolved_out_dtype = out_dtype or _compute_out_dtype(mat1.dtype, mat2.dtype)
    return torch.empty(result_shape, dtype=resolved_out_dtype, device=mat1.device)


@_decorators.codegen(dot, "triton")
def _(state: CodegenState) -> object:
    # Get the AST representations of our arguments
    lhs_ast = state.ast_arg(0)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)

    # Get the dtypes of the inputs from proxy args
    lhs_proxy = state.proxy_args[0]
    assert isinstance(lhs_proxy, FakeTensor), "lhs_proxy must be a FakeTensor"
    rhs_proxy = state.proxy_args[1]
    assert isinstance(rhs_proxy, FakeTensor), "rhs_proxy must be a FakeTensor"
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None

    lhs_dtype = lhs_proxy.dtype
    rhs_dtype = rhs_proxy.dtype
    acc_dtype: torch.dtype | None = None
    if acc_proxy is not None:
        assert isinstance(acc_proxy, FakeTensor), "acc_proxy must be a FakeTensor"
        acc_dtype = acc_proxy.dtype

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype), (
            "out_dtype must be a torch.dtype"
        )
        out_dtype = out_dtype_proxy

    # Check if accumulator is None
    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    lhs_shape: list[int | torch.SymInt] = list(lhs_proxy.shape)
    rhs_shape: list[int | torch.SymInt] = list(rhs_proxy.shape)
    acc_shape: list[int | torch.SymInt] | None = (
        list(acc_proxy.shape) if acc_proxy is not None else None
    )
    acc_arg = None if is_acc_none else acc_ast
    acc_dtype_arg = acc_dtype if not is_acc_none else None

    # Perform dot with optional padding
    return emit_tl_dot_with_padding(
        lhs_ast,
        rhs_ast,
        acc_arg,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_arg,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
        out_dtype=out_dtype,
    )


@_decorators.codegen(dot, "nki")
def _(state: CodegenState) -> object:
    from .._compiler.ast_extension import expr_from_string
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.ast_extension import create

    import sympy as _sympy

    lhs_ast = state.ast_arg(0)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)

    lhs_proxy = state.proxy_args[0]
    rhs_proxy = state.proxy_args[1]
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    assert isinstance(lhs_proxy, FakeTensor)
    assert isinstance(rhs_proxy, FakeTensor)

    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    env = CompileEnvironment.current()
    _bs_subs: dict[_sympy.Symbol, int] = {}
    for _bid in range(len(env.block_sizes)):
        _bs = env.block_sizes[_bid]
        _bs_subs[_bs.symbol()] = int(_bs.from_config_assert(state.config))

    def _to_int(s: int | torch.SymInt) -> int:
        if isinstance(s, int):
            return s
        return int(s._sympy_().subs(_bs_subs))

    lhs_shape = list(lhs_proxy.shape)
    rhs_shape = list(rhs_proxy.shape)
    # Squeeze leading batch dims for 3D+ shapes (batch_block=1)
    while len(lhs_shape) > 2:
        lhs_shape = lhs_shape[1:]
    while len(rhs_shape) > 2:
        rhs_shape = rhs_shape[1:]
    M_tile = _to_int(lhs_shape[0])
    K_tile = _to_int(lhs_shape[1])
    N_tile = _to_int(rhs_shape[-1])

    TILE_M = 128
    TILE_K = 128
    TILE_N = 512

    assert M_tile % TILE_M == 0 or M_tile < TILE_M, (
        f"M_tile={M_tile} must be <= {TILE_M} or a multiple of {TILE_M}"
    )
    assert K_tile % TILE_K == 0 or K_tile < TILE_K, (
        f"K_tile={K_tile} must be <= {TILE_K} or a multiple of {TILE_K}"
    )

    actual_tile_m = min(M_tile, TILE_M)
    actual_tile_k = min(K_tile, TILE_K)
    N_sub = min(TILE_N, N_tile)
    n_sub_m = max(1, M_tile // TILE_M)
    n_sub_k = max(1, K_tile // TILE_K)
    n_sub_n = max(1, N_tile // N_sub) if N_tile > TILE_N else 1

    # trn1 requires both nc_matmul inputs to have the same non-float32 dtype.
    # nc_transpose always produces float32 (PSUM), so cast back to input dtype
    # when inputs are fp16/bf16 (same fix as in aten_lowering._nki_dot).
    rhs_dtype = rhs_proxy.dtype
    need_cast_after_transpose = rhs_dtype != torch.float32
    matmul_dtype_str = env.backend.dtype_str(rhs_dtype)

    lhs_name = ast.unparse(lhs_ast)
    rhs_name = ast.unparse(rhs_ast)
    device_fn = state.device_function

    lhs_tile_vars = device_fn.get_tile_list_vars(lhs_name)
    rhs_tile_vars = device_fn.get_tile_list_vars(rhs_name)
    lhs_is_list = lhs_tile_vars is not None
    rhs_is_list = rhs_tile_vars is not None
    result_is_list = n_sub_m > 1

    mm_result = device_fn.new_var("_nki_dot_result")
    mm_result_tile_vars: list[str] = []
    if result_is_list:
        for i in range(n_sub_m):
            rv = device_fn.new_var(f"{mm_result}_{i}")
            mm_result_tile_vars.append(rv)
            state.add_statement(
                statement_from_string(
                    f"{rv} = nl.ndarray([{actual_tile_m}, {N_tile}], nl.float32, buffer=nl.sbuf)"
                )
            )
        device_fn.register_tile_list(mm_result, mm_result_tile_vars)
    else:
        state.add_statement(
            statement_from_string(
                f"{mm_result} = nl.ndarray([{actual_tile_m}, {N_tile}], nl.float32, buffer=nl.sbuf)"
            )
        )

    sub_k_var = device_fn.new_var("_dot_sub_k")
    lhs_t_psum = device_fn.new_var("_dot_lhs_t_psum")
    lhs_t_sbuf = device_fn.new_var("_dot_lhs_t_sbuf")
    lhs_t_cast = device_fn.new_var("_dot_lhs_t_cast") if need_cast_after_transpose else None
    mm_psum = device_fn.new_var("_dot_mm_psum")
    # stationary operand for nc_matmul: cast buffer if dtype cast needed
    _stationary = lhs_t_cast if need_cast_after_transpose else lhs_t_sbuf

    def _lhs_k_slice(m_i: int, k_expr: str) -> str:
        if lhs_is_list:
            assert lhs_tile_vars is not None
            if n_sub_k > 1:
                return (
                    f"{lhs_tile_vars[m_i]}[0:{actual_tile_m}, "
                    f"{k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}]"
                )
            return lhs_tile_vars[m_i]
        if n_sub_m > 1:
            return (
                f"{lhs_name}[{m_i} * {actual_tile_m} : ({m_i} + 1) * {actual_tile_m}, "
                f"{k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}]"
            )
        if n_sub_k > 1:
            return f"{lhs_name}[0:{actual_tile_m}, {k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}]"
        return lhs_name

    def _rhs_ref(k_expr: str, n_expr: str) -> str:
        if rhs_is_list:
            assert rhs_tile_vars is not None
            idx = int(k_expr) if k_expr.isdigit() else 0
            if n_sub_n > 1:
                return (
                    f"{rhs_tile_vars[idx]}[0:{TILE_K}, "
                    f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
                )
            return rhs_tile_vars[idx]
        if n_sub_k > 1 or n_sub_n > 1:
            return (
                f"{rhs_name}[{k_expr} * {TILE_K} : ({k_expr} + 1) * {TILE_K}, "
                f"{n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
            )
        return rhs_name

    def _result_ref(m_i: int, n_expr: str) -> str:
        if result_is_list:
            rv = mm_result_tile_vars[m_i]
            if n_sub_n > 1:
                return f"{rv}[0:{actual_tile_m}, {n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
            return rv
        if n_sub_n > 1:
            return f"{mm_result}[0:{actual_tile_m}, {n_expr} * {N_sub} : ({n_expr} + 1) * {N_sub}]"
        return mm_result

    def _emit_one_m_stripe(m_i: int, n_expr: str) -> None:
        mm_sbuf_tmp = device_fn.new_var("_dot_sbuf_tmp")
        state.add_statement(
            statement_from_string(
                f"{mm_psum} = nl.ndarray([{actual_tile_m}, {N_sub}], nl.float32, buffer=nl.psum)"
            )
        )
        if n_sub_k > 1:
            state.add_statement(
                create(
                    ast.For,
                    target=create(ast.Name, id=sub_k_var, ctx=ast.Store()),
                    iter=expr_from_string(f"nl.affine_range({n_sub_k})"),
                    body=[
                        statement_from_string(
                            f"{lhs_t_psum} = nl.ndarray([{actual_tile_k}, {actual_tile_m}], nl.float32, buffer=nl.psum)"
                        ),
                        statement_from_string(
                            f"nisa.nc_transpose(dst={lhs_t_psum}, data={_lhs_k_slice(m_i, sub_k_var)})"
                        ),
                        statement_from_string(
                            f"{lhs_t_sbuf} = nl.ndarray([{actual_tile_k}, {actual_tile_m}], nl.float32, buffer=nl.sbuf)"
                        ),
                        statement_from_string(
                            f"nisa.tensor_copy(dst={lhs_t_sbuf}, src={lhs_t_psum})"
                        ),
                        *(
                            [
                                statement_from_string(
                                    f"{lhs_t_cast} = nl.ndarray([{actual_tile_k}, {actual_tile_m}], {matmul_dtype_str}, buffer=nl.sbuf)"
                                ),
                                statement_from_string(f"nisa.memset({lhs_t_cast}, value=0)"),
                                statement_from_string(
                                    f"nisa.tensor_tensor(dst={lhs_t_cast}, data1={lhs_t_cast}, data2={lhs_t_sbuf}, op=nl.add)"
                                ),
                            ]
                            if need_cast_after_transpose
                            else []
                        ),
                        statement_from_string(
                            f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary}, moving={_rhs_ref(sub_k_var, n_expr)})"
                        ),
                    ],
                    orelse=[],
                )
            )
        else:
            state.add_statement(
                statement_from_string(
                    f"{lhs_t_psum} = nl.ndarray([{actual_tile_k}, {actual_tile_m}], nl.float32, buffer=nl.psum)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.nc_transpose(dst={lhs_t_psum}, data={_lhs_k_slice(m_i, '0')})"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"{lhs_t_sbuf} = nl.ndarray([{actual_tile_k}, {actual_tile_m}], nl.float32, buffer=nl.sbuf)"
                )
            )
            state.add_statement(
                statement_from_string(
                    f"nisa.tensor_copy(dst={lhs_t_sbuf}, src={lhs_t_psum})"
                )
            )
            if need_cast_after_transpose:
                state.add_statement(
                    statement_from_string(
                        f"{lhs_t_cast} = nl.ndarray([{actual_tile_k}, {actual_tile_m}], {matmul_dtype_str}, buffer=nl.sbuf)"
                    )
                )
                state.add_statement(
                    statement_from_string(f"nisa.memset({lhs_t_cast}, value=0)")
                )
                state.add_statement(
                    statement_from_string(
                        f"nisa.tensor_tensor(dst={lhs_t_cast}, data1={lhs_t_cast}, data2={lhs_t_sbuf}, op=nl.add)"
                    )
                )
            state.add_statement(
                statement_from_string(
                    f"nisa.nc_matmul(dst={mm_psum}, stationary={_stationary}, moving={_rhs_ref('0', n_expr)})"
                )
            )
        state.add_statement(
            statement_from_string(
                f"{mm_sbuf_tmp} = nl.ndarray([{actual_tile_m}, {N_sub}], nl.float32, buffer=nl.sbuf)"
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
        sub_n_var = device_fn.new_var("_dot_sub_n")
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

    if is_acc_none:
        return expr_from_string(mm_result)

    assert acc_proxy is not None and isinstance(acc_proxy, FakeTensor)
    acc_name = ast.unparse(acc_ast)
    acc_tile_vars = device_fn.get_tile_list_vars(acc_name)
    out_result = device_fn.new_var("_nki_dot_acc_result")

    if result_is_list:
        out_tile_vars: list[str] = []
        for i in range(n_sub_m):
            ov = device_fn.new_var(f"{out_result}_{i}")
            out_tile_vars.append(ov)
            state.add_statement(
                statement_from_string(
                    f"{ov} = nl.ndarray([{actual_tile_m}, {N_tile}], nl.float32, buffer=nl.sbuf)"
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
        device_fn.register_tile_list(out_result, out_tile_vars)
    else:
        state.add_statement(
            statement_from_string(
                f"{out_result} = nl.ndarray([{actual_tile_m}, {N_tile}], nl.float32, buffer=nl.sbuf)"
            )
        )
        state.add_statement(
            statement_from_string(
                f"nisa.tensor_tensor(dst={out_result}, data1={mm_result}, "
                f"data2={{acc}}, op=nl.add)",
                acc=acc_ast,
            )
        )
    return expr_from_string(out_result)


@_decorators.ref(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_out_dtype = out_dtype or _compute_out_dtype(
        mat1.dtype, mat2.dtype, None if acc is None else acc.dtype
    )

    is_fp8 = mat1.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or mat2.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if is_fp8:
        # Use torch._scaled_mm for FP8 operations
        # Ensure column-major for second operand as required by torch._scaled_mm
        mat2_t = mat2.T.contiguous().T
        scale_a = torch.tensor(1.0, device=mat1.device)
        scale_b = torch.tensor(1.0, device=mat2.device)

        result = torch._scaled_mm(
            mat1,
            mat2_t,
            scale_a,
            scale_b,
            use_fast_accum=False,
            out_dtype=resolved_out_dtype,
        )
    else:
        # For non-FP8 tensors, use regular matmul
        result = torch.mm(mat1, mat2, out_dtype=resolved_out_dtype)

    if acc is not None:
        return acc + result
    return result


VALID_SCALED_FORMATS = frozenset({"e2m1", "e4m3", "e5m2", "bf16", "fp16"})


@_decorators.api(is_device_only=True)
def dot_scaled(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Performs a block-scaled matrix multiplication using Triton's tl.dot_scaled.

    This operation performs matrix multiplication with block-scaled inputs in formats
    such as FP4 (e2m1), FP8 (e4m3, e5m2), BF16, and FP16. Each input tensor has an
    associated scale factor tensor and format string.

    Args:
        mat1: First matrix (2D tensor of packed data)
        mat1_scale: Scale factors for mat1 (2D tensor)
        mat1_format: Format string for mat1 (one of "e2m1", "e4m3", "e5m2", "bf16", "fp16")
        mat2: Second matrix (2D tensor of packed data)
        mat2_scale: Scale factors for mat2 (2D tensor)
        mat2_format: Format string for mat2 (one of "e2m1", "e4m3", "e5m2", "bf16", "fp16")
        acc: Optional accumulator tensor (2D, float32 or float16)
        out_dtype: Optional output dtype for the multiplication

    Returns:
        Result of block-scaled matrix multiplication.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(dot_scaled)
def _(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    str,
    torch.Tensor,
    torch.Tensor,
    str,
    torch.Tensor | None,
    torch.dtype | None,
]:
    if mat1_format not in VALID_SCALED_FORMATS:
        raise ValueError(
            f"hl.dot_scaled: mat1_format must be one of {sorted(VALID_SCALED_FORMATS)}, "
            f"got '{mat1_format}'"
        )
    if mat2_format not in VALID_SCALED_FORMATS:
        raise ValueError(
            f"hl.dot_scaled: mat2_format must be one of {sorted(VALID_SCALED_FORMATS)}, "
            f"got '{mat2_format}'"
        )

    if mat1.ndim != 2:
        raise ValueError(f"hl.dot_scaled: mat1 must be a 2D tensor, got {mat1.ndim}D")
    if mat2.ndim != 2:
        raise ValueError(f"hl.dot_scaled: mat2 must be a 2D tensor, got {mat2.ndim}D")

    if mat1_scale.ndim != 2:
        raise ValueError(
            f"hl.dot_scaled: mat1_scale must be a 2D tensor, got {mat1_scale.ndim}D"
        )
    if mat2_scale.ndim != 2:
        raise ValueError(
            f"hl.dot_scaled: mat2_scale must be a 2D tensor, got {mat2_scale.ndim}D"
        )

    if acc is not None:
        expected_shape = [mat1.shape[0], mat2.shape[-1]]
        if acc.ndim != 2:
            raise ValueError(f"hl.dot_scaled: acc must be a 2D tensor, got {acc.ndim}D")
        if list(acc.shape) != expected_shape:
            raise ValueError(
                f"hl.dot_scaled: acc shape {list(acc.shape)} incompatible with "
                f"result shape {expected_shape}"
            )
        valid_acc_dtypes = (torch.float16, torch.float32)
        if acc.dtype not in valid_acc_dtypes:
            raise TypeError(
                f"hl.dot_scaled: acc must be one of {[str(d) for d in valid_acc_dtypes]}, "
                f"got {acc.dtype}"
            )

    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        raise TypeError(
            f"hl.dot_scaled: out_dtype must be a torch.dtype or None, got {type(out_dtype)}"
        )

    # Enforce minimum block sizes so autotuner picks valid configs.
    enforce_dot_requirements(mat1, mat2)
    # K must be >= 32 because scale tensors have shape [dim, K // 32].
    env = CompileEnvironment.current()
    k_dim = mat1.shape[-1]
    k_block_idx = env.get_block_id(k_dim)
    if k_block_idx is not None:
        env.block_sizes[k_block_idx].update_min_block(32, allow_flattened=True)

    return (
        mat1,
        mat1_scale,
        mat1_format,
        mat2,
        mat2_scale,
        mat2_format,
        acc,
        out_dtype,
    )


@_decorators.register_fake(dot_scaled)
def _(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    result_shape = [mat1.shape[0], mat2.shape[-1]]
    if acc is not None:
        return acc.new_empty(result_shape)
    resolved_dtype = out_dtype or torch.float32
    return torch.empty(result_shape, dtype=resolved_dtype, device=mat1.device)


@_decorators.codegen(dot_scaled, "triton")
def _(state: CodegenState) -> object:
    lhs_ast = state.ast_arg(0)  # mat1
    lhs_scale_ast = state.ast_arg(1)  # mat1_scale
    lhs_format = state.proxy_args[2]  # "e2m1" etc (string, not AST)
    assert isinstance(lhs_format, str), "lhs_format must be a string"
    rhs_ast = state.ast_arg(3)  # mat2
    rhs_scale_ast = state.ast_arg(4)  # mat2_scale
    rhs_format = state.proxy_args[5]  # "e2m1" etc (string, not AST)
    assert isinstance(rhs_format, str), "rhs_format must be a string"
    acc_ast = state.ast_arg(6)  # acc
    out_dtype_proxy = state.proxy_args[7] if len(state.proxy_args) > 7 else None

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype), (
            "out_dtype must be a torch.dtype"
        )
        out_dtype = out_dtype_proxy

    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None
    return _emit_tl_dot_scaled(
        lhs_ast,
        lhs_scale_ast,
        lhs_format,
        rhs_ast,
        rhs_scale_ast,
        rhs_format,
        acc=None if is_acc_none else acc_ast,
        out_dtype=out_dtype,
    )


@_decorators.ref(dot_scaled)
def _(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    def _dequant(data: torch.Tensor, scale: torch.Tensor, fmt: str) -> torch.Tensor:
        data_f32 = data.to(torch.float32)
        # Scale is in e8m0 format (uint8): value = 2^(byte - 127)
        # e.g. byte=127 means 2^0=1.0, byte=0 means 2^(-127), byte=254 means 2^127
        scale_f32 = torch.pow(2.0, scale.to(torch.float32) - 127.0)
        k_data = data_f32.shape[-1]
        k_scale = scale_f32.shape[-1]
        if k_scale < k_data:
            repeat_factor = k_data // k_scale
            scale_f32 = scale_f32.repeat_interleave(repeat_factor, dim=-1)
        return data_f32 * scale_f32

    mat1_dequant = _dequant(mat1, mat1_scale, mat1_format)
    mat2_dequant = _dequant(mat2, mat2_scale, mat2_format)

    result = torch.mm(mat1_dequant, mat2_dequant)
    resolved_dtype = out_dtype or torch.float32
    result = result.to(resolved_dtype)

    if acc is not None:
        return acc + result
    return result
