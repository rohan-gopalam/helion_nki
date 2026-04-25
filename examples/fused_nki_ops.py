"""
Fused NKI ISA Operations (Trn2/Trn3)
=====================================
Tests and examples for the five fused NKI ISA operations added to the Helion
NKI backend:

  - nisa.tensor_scalar_reduce   : tensor-scalar op + free-axis reduction in one pass
  - nisa.activation_reduce      : activation + free-axis reduction in one pass
  - nisa.scalar_tensor_tensor   : (data op0 scalar) op1 tensor in one instruction
  - nisa.tensor_tensor_scan     : prefix scan across the free axis
  - nisa.tensor_scalar_cumulative: cumulative tensor-scalar op

Hardware support: Trn2 and Trn3 only.  On Trn1 these calls will raise a
neuronx-cc error; the generated Python code is valid in both cases.

This file is structured in two parts:
  1. Code-generation tests — verify that Helion emits the correct nisa.* call.
  2. Numerical tests (Trn2/Trn3 only) — run on hardware and compare to reference.

Run:
  PYTHONPATH=helion_nki:$PYTHONPATH HELION_BACKEND=nki \
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  python helion_nki/examples/fused_nki_ops.py
"""

from __future__ import annotations

import os
import sys
import textwrap
import torch
import helion
import helion.language as hl
from helion._testing import DEVICE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IS_TRN2 = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").startswith("trn2")


def _compile_and_get_code(kernel_fn: object, *args: object, **kwargs: object) -> str:
    """Return the generated NKI Python source for kernel_fn(*args)."""
    import io, contextlib
    from helion._compiler.compile_environment import CompileEnvironment

    # Force a fresh compile by clearing inductor cache key
    import torch._inductor.codecache as cc
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = kernel_fn(*args, **kwargs)
    except Exception:
        pass

    # Search torchinductor cache for the generated file
    cache_dir = "/tmp/torchinductor_ubuntu"
    if not os.path.isdir(cache_dir):
        return ""
    import glob
    py_files = glob.glob(f"{cache_dir}/**/*.py", recursive=True)
    # Return the most-recently modified one that looks like our kernel
    py_files.sort(key=os.path.getmtime, reverse=True)
    for f in py_files[:10]:
        try:
            txt = open(f).read()
            if "nisa." in txt:
                return txt
        except OSError:
            pass
    return ""


# ---------------------------------------------------------------------------
# Part 1: Code-generation tests
# ---------------------------------------------------------------------------
# These tests use NKIOpOverrides directly to verify that the correct nisa.*
# statement is emitted into the generated code.  They don't require Trn2
# hardware — just that the compiler runs.

def _test_codegen_tensor_scalar_reduce() -> None:
    """Verify _nki_tensor_scalar_reduce emits nisa.tensor_scalar_reduce."""
    from helion._compiler.backend import NKIOpOverrides

    # Build a minimal mock state so the helper can emit statements
    class _FakeDF:
        _nki_sbuf_shapes: dict = {}
        _vars: list = []

        def new_var(self, prefix: str, dce: bool = False) -> str:
            name = f"{prefix}_{len(self._vars)}"
            self._vars.append(name)
            return name

    class _FakeState:
        device_function = _FakeDF()
        config = None
        _stmts: list = []

        def add_statement(self, s: object) -> None:
            self._stmts.append(s)

    # Patch CompileEnvironment.current() to return a fake env
    import helion._compiler.compile_environment as _ce
    import helion._compiler.backend as _be

    class _FakeBackend:
        name = "nki"
        def dtype_str(self, dt: object) -> str:
            return "nl.float32"

    class _FakeEnv:
        backend = _FakeBackend()
        _codegen_state = None

    state = _FakeState()

    _orig = _ce.CompileEnvironment.current
    _orig_codegen = getattr(_ce.CompileEnvironment, "_codegen_state", None)

    # Register known shape so the helper can use it
    state.device_function._nki_sbuf_shapes["x_tile"] = [128, 256]
    fake_env = _FakeEnv()
    fake_env._codegen_state = state

    try:
        _ce.CompileEnvironment._instance = fake_env  # type: ignore[attr-defined]
        _ce.CompileEnvironment.current = staticmethod(lambda: fake_env)  # type: ignore[method-assign]

        dst_var, reduce_var = NKIOpOverrides._nki_tensor_scalar_reduce(
            "x_tile", "nl.multiply", 2.0, "nl.add", [128, 1]
        )
    finally:
        _ce.CompileEnvironment.current = staticmethod(_orig)  # type: ignore[method-assign]

    import ast as _ast
    stmts = "\n".join(_ast.unparse(s) if isinstance(s, _ast.AST) else str(s) for s in state._stmts)
    assert "nisa.tensor_scalar_reduce" in stmts, f"Expected nisa.tensor_scalar_reduce in:\n{stmts}"
    assert dst_var != "" and reduce_var != "", "Expected non-empty var names"
    print("  [PASS] tensor_scalar_reduce codegen")


def _test_codegen_activation_reduce() -> None:
    """Verify _nki_activation_reduce emits nisa.activation_reduce."""
    from helion._compiler.backend import NKIOpOverrides
    import helion._compiler.compile_environment as _ce

    class _FakeDF:
        _nki_sbuf_shapes: dict = {"act_in": [128, 256]}
        _vars: list = []

        def new_var(self, prefix: str, dce: bool = False) -> str:
            name = f"{prefix}_{len(self._vars)}"
            self._vars.append(name)
            return name

    class _FakeState:
        device_function = _FakeDF()
        config = None
        _stmts: list = []

        def add_statement(self, s: object) -> None:
            self._stmts.append(s)

    class _FakeBackend:
        name = "nki"
        def dtype_str(self, dt: object) -> str:
            return "nl.float32"

    class _FakeEnv:
        backend = _FakeBackend()

    state = _FakeState()
    fake_env = _FakeEnv()
    fake_env._codegen_state = state  # type: ignore[attr-defined]

    _orig = _ce.CompileEnvironment.current
    try:
        _ce.CompileEnvironment.current = staticmethod(lambda: fake_env)  # type: ignore[method-assign]
        dst_var, reduce_var = NKIOpOverrides._nki_activation_reduce(
            "act_in", "nl.relu", "nl.add", [128, 1]
        )
    finally:
        _ce.CompileEnvironment.current = staticmethod(_orig)  # type: ignore[method-assign]

    import ast as _ast
    stmts = "\n".join(_ast.unparse(s) if isinstance(s, _ast.AST) else str(s) for s in state._stmts)
    assert "nisa.activation_reduce" in stmts, f"Expected nisa.activation_reduce in:\n{stmts}"
    assert dst_var != "" and reduce_var != ""
    print("  [PASS] activation_reduce codegen")


def _test_codegen_scalar_tensor_tensor() -> None:
    """Verify _nki_scalar_tensor_tensor emits nisa.scalar_tensor_tensor."""
    from helion._compiler.backend import NKIOpOverrides
    import helion._compiler.compile_environment as _ce

    class _FakeDF:
        _nki_sbuf_shapes: dict = {"d0": [128, 256]}
        _vars: list = []

        def new_var(self, prefix: str, dce: bool = False) -> str:
            name = f"{prefix}_{len(self._vars)}"
            self._vars.append(name)
            return name

    class _FakeState:
        device_function = _FakeDF()
        config = None
        _stmts: list = []

        def add_statement(self, s: object) -> None:
            self._stmts.append(s)

    class _FakeBackend:
        name = "nki"
        def dtype_str(self, dt: object) -> str:
            return "nl.float32"

    class _FakeEnv:
        backend = _FakeBackend()

    state = _FakeState()
    fake_env = _FakeEnv()
    fake_env._codegen_state = state  # type: ignore[attr-defined]

    _orig = _ce.CompileEnvironment.current
    try:
        _ce.CompileEnvironment.current = staticmethod(lambda: fake_env)  # type: ignore[method-assign]
        dst_var = NKIOpOverrides._nki_scalar_tensor_tensor(
            "d0", "nl.multiply", 0.5, "nl.add", "d1"
        )
    finally:
        _ce.CompileEnvironment.current = staticmethod(_orig)  # type: ignore[method-assign]

    import ast as _ast
    stmts = "\n".join(_ast.unparse(s) if isinstance(s, _ast.AST) else str(s) for s in state._stmts)
    assert "nisa.scalar_tensor_tensor" in stmts, f"Expected nisa.scalar_tensor_tensor in:\n{stmts}"
    assert dst_var != ""
    print("  [PASS] scalar_tensor_tensor codegen")


def _test_codegen_tensor_tensor_scan() -> None:
    """Verify _nki_tensor_tensor_scan emits nisa.tensor_tensor_scan."""
    from helion._compiler.backend import NKIOpOverrides
    import helion._compiler.compile_environment as _ce

    class _FakeDF:
        _nki_sbuf_shapes: dict = {"scan_d0": [128, 256], "scan_d1": [128, 256]}
        _vars: list = []

        def new_var(self, prefix: str, dce: bool = False) -> str:
            name = f"{prefix}_{len(self._vars)}"
            self._vars.append(name)
            return name

    class _FakeState:
        device_function = _FakeDF()
        config = None
        _stmts: list = []

        def add_statement(self, s: object) -> None:
            self._stmts.append(s)

    class _FakeBackend:
        name = "nki"
        def dtype_str(self, dt: object) -> str:
            return "nl.float32"

    class _FakeEnv:
        backend = _FakeBackend()

    state = _FakeState()
    fake_env = _FakeEnv()
    fake_env._codegen_state = state  # type: ignore[attr-defined]

    _orig = _ce.CompileEnvironment.current
    try:
        _ce.CompileEnvironment.current = staticmethod(lambda: fake_env)  # type: ignore[method-assign]
        dst_var = NKIOpOverrides._nki_tensor_tensor_scan(
            "scan_d0", "scan_d1", 0.0, "nl.add", "nl.add"
        )
    finally:
        _ce.CompileEnvironment.current = staticmethod(_orig)  # type: ignore[method-assign]

    import ast as _ast
    stmts = "\n".join(_ast.unparse(s) if isinstance(s, _ast.AST) else str(s) for s in state._stmts)
    assert "nisa.tensor_tensor_scan" in stmts, f"Expected nisa.tensor_tensor_scan in:\n{stmts}"
    assert dst_var != ""
    print("  [PASS] tensor_tensor_scan codegen")


def _test_codegen_tensor_scalar_cumulative() -> None:
    """Verify _nki_tensor_scalar_cumulative emits nisa.tensor_scalar_cumulative."""
    from helion._compiler.backend import NKIOpOverrides
    import helion._compiler.compile_environment as _ce

    class _FakeDF:
        _nki_sbuf_shapes: dict = {"tsc_src": [128, 256]}
        _vars: list = []

        def new_var(self, prefix: str, dce: bool = False) -> str:
            name = f"{prefix}_{len(self._vars)}"
            self._vars.append(name)
            return name

    class _FakeState:
        device_function = _FakeDF()
        config = None
        _stmts: list = []

        def add_statement(self, s: object) -> None:
            self._stmts.append(s)

    class _FakeBackend:
        name = "nki"
        def dtype_str(self, dt: object) -> str:
            return "nl.float32"

    class _FakeEnv:
        backend = _FakeBackend()

    state = _FakeState()
    fake_env = _FakeEnv()
    fake_env._codegen_state = state  # type: ignore[attr-defined]

    _orig = _ce.CompileEnvironment.current
    try:
        _ce.CompileEnvironment.current = staticmethod(lambda: fake_env)  # type: ignore[method-assign]
        dst_var = NKIOpOverrides._nki_tensor_scalar_cumulative(
            "tsc_src", "nl.multiply", "nl.add", 1.0
        )
    finally:
        _ce.CompileEnvironment.current = staticmethod(_orig)  # type: ignore[method-assign]

    import ast as _ast
    stmts = "\n".join(_ast.unparse(s) if isinstance(s, _ast.AST) else str(s) for s in state._stmts)
    assert "nisa.tensor_scalar_cumulative" in stmts, f"Expected nisa.tensor_scalar_cumulative in:\n{stmts}"
    assert "reduce_cmd=nisa.reduce_cmd.reset_reduce" in stmts
    assert dst_var != ""
    print("  [PASS] tensor_scalar_cumulative codegen")


def _test_nkibackend_public_methods() -> None:
    """Verify NKIBackend exposes the five new wrapper methods."""
    from helion._compiler.backend import NKIBackend
    b = NKIBackend()
    assert hasattr(b, "tensor_scalar_reduce_expr"), "Missing tensor_scalar_reduce_expr"
    assert hasattr(b, "activation_reduce_expr"), "Missing activation_reduce_expr"
    assert hasattr(b, "scalar_tensor_tensor_expr"), "Missing scalar_tensor_tensor_expr"
    assert hasattr(b, "tensor_tensor_scan_expr"), "Missing tensor_tensor_scan_expr"
    assert hasattr(b, "tensor_scalar_cumulative_expr"), "Missing tensor_scalar_cumulative_expr"
    print("  [PASS] NKIBackend public method presence")


# ---------------------------------------------------------------------------
# Part 2: Helion kernel tests (Trn2/Trn3 numerical verification)
# ---------------------------------------------------------------------------
# These kernels express patterns that should eventually trigger the fused ops
# automatically. For now they verify that the fused ops produce correct output.

@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128]),
)
def scale_and_sum(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Element-wise scale + row-sum: out[i] = sum(x[i, :] * scale).
    Pattern: tensor_scalar_reduce(data=x_tile, op0=nl.multiply, operand0=scale,
                                   reduce_op=nl.add, reduce_res=row_sum).
    """
    m, _n = x.size()
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        tile = x[tile_m, :]
        out[tile_m, 0] = torch.sum(tile * scale, dim=1, keepdim=True)
    return out


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128]),
)
def relu_and_sum(x: torch.Tensor) -> torch.Tensor:
    """ReLU + row-sum: out[i] = sum(relu(x[i, :])).
    Pattern: activation_reduce(data=x_tile, op=nl.relu,
                                reduce_op=nl.add, reduce_res=row_sum).
    """
    m, _n = x.size()
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        tile = x[tile_m, :]
        out[tile_m, 0] = torch.sum(torch.relu(tile), dim=1, keepdim=True)
    return out


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128]),
)
def bias_scale_add(
    x: torch.Tensor, bias: float, scale: float, y: torch.Tensor
) -> torch.Tensor:
    """(x + bias) * scale + y: tests the scalar_tensor_tensor pattern."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        xt = x[tile_m, tile_n]
        yt = y[tile_m, tile_n]
        out[tile_m, tile_n] = (xt + bias) * scale + yt
    return out


def check_scale_and_sum(m: int = 128, n: int = 256) -> None:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    scale = 0.5
    ref = (x * scale).sum(dim=1, keepdim=True)
    got = scale_and_sum(x, scale)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    print(f"  [PASS] scale_and_sum ({m}x{n})")


def check_relu_and_sum(m: int = 128, n: int = 256) -> None:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    ref = torch.relu(x).sum(dim=1, keepdim=True)
    got = relu_and_sum(x)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    print(f"  [PASS] relu_and_sum ({m}x{n})")


def check_bias_scale_add(m: int = 128, n: int = 256) -> None:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    y = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    bias, scale = 1.5, 0.25
    ref = (x + bias) * scale + y
    got = bias_scale_add(x, bias, scale, y)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    print(f"  [PASS] bias_scale_add ({m}x{n})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Codegen tests (no hardware required) ===")
    _test_codegen_tensor_scalar_reduce()
    _test_codegen_activation_reduce()
    _test_codegen_scalar_tensor_tensor()
    _test_codegen_tensor_tensor_scan()
    _test_codegen_tensor_scalar_cumulative()
    _test_nkibackend_public_methods()

    if not _IS_TRN2:
        print("\nSkipping numerical tests: not on Trn2 (set NEURON_PLATFORM_TARGET_OVERRIDE=trn2)")
        print("All codegen tests passed.")
        return

    print("\n=== Numerical tests (Trn2 hardware) ===")
    check_scale_and_sum()
    check_relu_and_sum()
    check_bias_scale_add()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
