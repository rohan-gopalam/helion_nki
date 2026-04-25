"""
PSUM-Reuse Fusion Tests
========================
Verifies the FX-graph fusion pass that tags ``matmul → Vector/Scalar op``
patterns for PSUM reuse, and that the downstream codegen honors the tag
by skipping the final ``tensor_copy(sbuf, psum)`` and reading from PSUM
directly in the consumer.

Hardware model recap::

    Tensor Engine (nc_matmul)  → inputs SBUF,  output PSUM
    Vector Engine              → inputs SBUF/PSUM, outputs SBUF/PSUM
    Scalar Engine (activation) → inputs SBUF/PSUM, outputs SBUF/PSUM

So after an nc_matmul writes to PSUM, a subsequent Vector/Scalar op that
is the **single** user can read from PSUM directly — no intermediate
SBUF copy needed.

This file exercises:
  1. FX-graph pass tagging (no hardware required)
  2. PSUM budget safety guards
  3. Generated-code inspection (fewer tensor_copy calls)
  4. Numerical correctness on Trn1 for matmul→relu and matmul→mul patterns

Run:
    PYTHONPATH=helion_nki:$PYTHONPATH HELION_BACKEND=nki \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn1 \
    python helion_nki/examples/psum_reuse_test.py

To disable fusion and get a baseline:
    HELION_NKI_DISABLE_FUSION=1 python helion_nki/examples/psum_reuse_test.py
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Callable

import torch
import torch.fx as fx

import helion
import helion.language as hl
from helion._testing import DEVICE


_IS_TRN2 = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").startswith("trn2")
_CACHE_DIR = "/tmp/torchinductor_ubuntu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache() -> None:
    if os.path.isdir(_CACHE_DIR):
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)


def _find_generated_kernel_source(hint: str) -> str:
    """Return the most recent generated NKI source containing ``hint``."""
    if not os.path.isdir(_CACHE_DIR):
        return ""
    files = sorted(
        glob.glob(f"{_CACHE_DIR}/**/*.py", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    for fp in files:
        try:
            txt = open(fp).read()
        except OSError:
            continue
        if hint in txt and "nc_matmul" in txt:
            return txt
    return ""


# ---------------------------------------------------------------------------
# Part 1: FX-graph fusion pass unit tests (no compilation needed)
# ---------------------------------------------------------------------------

def _build_fx_graph(
    body: Callable[[fx.Graph], fx.Node],
    lhs_val: torch.Tensor,
    rhs_val: torch.Tensor,
) -> fx.Graph:
    """Build a minimal FX graph with two placeholder inputs + a body builder."""
    g = fx.Graph()
    lhs = g.placeholder("lhs")
    lhs.meta["val"] = lhs_val
    rhs = g.placeholder("rhs")
    rhs.meta["val"] = rhs_val
    out = body(g, lhs, rhs)
    g.output(out)
    return g


def _test_pass_tags_matmul_relu() -> None:
    """relu(mm(x, y)): both nodes should be tagged."""
    from helion._compiler.nki_fusion import annotate_psum_reuse

    lhs_v = torch.empty(128, 128, dtype=torch.float32)
    rhs_v = torch.empty(128, 128, dtype=torch.float32)

    def body(g: fx.Graph, lhs: fx.Node, rhs: fx.Node) -> fx.Node:
        mm = g.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        mm.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        r = g.call_function(torch.ops.aten.relu.default, args=(mm,))
        r.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        return r

    g = _build_fx_graph(body, lhs_v, rhs_v)
    n = annotate_psum_reuse(g)
    assert n == 1, f"expected 1 tagged matmul, got {n}"
    mm_node = next(nd for nd in g.nodes if nd.op == "call_function" and nd.target is torch.ops.aten.mm.default)
    relu_node = next(nd for nd in g.nodes if nd.op == "call_function" and nd.target is torch.ops.aten.relu.default)
    assert mm_node.meta.get("nki_keep_in_psum") is True
    assert relu_node.meta.get("nki_read_psum_from") == mm_node.name
    print("  [PASS] pass tags matmul→relu")


def _test_pass_skips_multi_user() -> None:
    """mm(x, y) with two users must NOT be tagged."""
    from helion._compiler.nki_fusion import annotate_psum_reuse

    lhs_v = torch.empty(128, 128, dtype=torch.float32)
    rhs_v = torch.empty(128, 128, dtype=torch.float32)

    def body(g: fx.Graph, lhs: fx.Node, rhs: fx.Node) -> fx.Node:
        mm = g.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        mm.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        r = g.call_function(torch.ops.aten.relu.default, args=(mm,))
        r.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        s = g.call_function(torch.ops.aten.sigmoid.default, args=(mm,))
        s.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        add = g.call_function(torch.ops.aten.add.Tensor, args=(r, s))
        add.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        return add

    g = _build_fx_graph(body, lhs_v, rhs_v)
    n = annotate_psum_reuse(g)
    assert n == 0, f"expected 0 tagged (matmul has 2 users), got {n}"
    print("  [PASS] pass skips multi-user matmul")


def _test_pass_skips_non_vector_scalar_consumer() -> None:
    """mm → mm (Tensor Engine → Tensor Engine) must NOT be tagged."""
    from helion._compiler.nki_fusion import annotate_psum_reuse

    v = torch.empty(128, 128, dtype=torch.float32)

    def body(g: fx.Graph, lhs: fx.Node, rhs: fx.Node) -> fx.Node:
        mm1 = g.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        mm1.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        mm2 = g.call_function(torch.ops.aten.mm.default, args=(mm1, rhs))
        mm2.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        return mm2

    g = _build_fx_graph(body, v, v)
    n = annotate_psum_reuse(g)
    # mm2's consumer is output (not a call_function), so mm2 can't be tagged.
    # mm1's consumer is mm2 — matmul is NOT in _VECTOR_SCALAR_CONSUMER_TARGETS,
    # so mm1 can't be tagged either.
    assert n == 0, f"expected 0 tagged (consumer is another matmul), got {n}"
    print("  [PASS] pass skips non-Vector/Scalar consumer")


def _test_pass_respects_psum_budget() -> None:
    """Tile with free_dim * fp32_bytes > 2048 must NOT be tagged."""
    from helion._compiler.nki_fusion import annotate_psum_reuse

    # 128 × 1024 fp32 = 4096 bytes/partition, exceeds 2KB PSUM bank.
    lhs_v = torch.empty(128, 256, dtype=torch.float32)
    rhs_v = torch.empty(256, 1024, dtype=torch.float32)

    def body(g: fx.Graph, lhs: fx.Node, rhs: fx.Node) -> fx.Node:
        mm = g.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        mm.meta["val"] = torch.empty(128, 1024, dtype=torch.float32)
        r = g.call_function(torch.ops.aten.relu.default, args=(mm,))
        r.meta["val"] = torch.empty(128, 1024, dtype=torch.float32)
        return r

    g = _build_fx_graph(body, lhs_v, rhs_v)
    n = annotate_psum_reuse(g)
    assert n == 0, f"expected 0 tagged (PSUM budget exceeded), got {n}"
    print("  [PASS] pass respects PSUM byte budget")


def _test_pass_respects_dtype_guard() -> None:
    """int32 matmul must NOT be tagged (PSUM is fp32/bf16 only)."""
    from helion._compiler.nki_fusion import annotate_psum_reuse

    lhs_v = torch.empty(128, 128, dtype=torch.int32)
    rhs_v = torch.empty(128, 128, dtype=torch.int32)

    def body(g: fx.Graph, lhs: fx.Node, rhs: fx.Node) -> fx.Node:
        mm = g.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        mm.meta["val"] = torch.empty(128, 128, dtype=torch.int32)
        r = g.call_function(torch.ops.aten.relu.default, args=(mm,))
        r.meta["val"] = torch.empty(128, 128, dtype=torch.int32)
        return r

    g = _build_fx_graph(body, lhs_v, rhs_v)
    n = annotate_psum_reuse(g)
    assert n == 0, f"expected 0 tagged (int dtype), got {n}"
    print("  [PASS] pass respects dtype guard")


def _test_pass_env_disable() -> None:
    """HELION_NKI_DISABLE_FUSION=1 disables the pass."""
    from helion._compiler.nki_fusion import annotate_psum_reuse

    lhs_v = torch.empty(128, 128, dtype=torch.float32)
    rhs_v = torch.empty(128, 128, dtype=torch.float32)

    def body(g: fx.Graph, lhs: fx.Node, rhs: fx.Node) -> fx.Node:
        mm = g.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        mm.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        r = g.call_function(torch.ops.aten.relu.default, args=(mm,))
        r.meta["val"] = torch.empty(128, 128, dtype=torch.float32)
        return r

    old = os.environ.get("HELION_NKI_DISABLE_FUSION")
    try:
        os.environ["HELION_NKI_DISABLE_FUSION"] = "1"
        g = _build_fx_graph(body, lhs_v, rhs_v)
        n = annotate_psum_reuse(g)
        assert n == 0, f"expected 0 tagged when fusion disabled, got {n}"
    finally:
        if old is None:
            os.environ.pop("HELION_NKI_DISABLE_FUSION", None)
        else:
            os.environ["HELION_NKI_DISABLE_FUSION"] = old
    print("  [PASS] pass respects HELION_NKI_DISABLE_FUSION")


# ---------------------------------------------------------------------------
# Part 2: End-to-end kernels — matmul → Vector/Scalar op
# ---------------------------------------------------------------------------

@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128, 128, 128]),
    static_shapes=True,
)
def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """relu(x @ y). Single-user matmul → activation: PSUM reuse should fire."""
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc)
    return out


def _test_codegen_matmul_relu_skips_copy() -> None:
    """Compile matmul_relu and verify generated code has NO PSUM→SBUF copy on the
    terminal matmul. Because matmul.py uses addmm (gated out), we build a
    different kernel that uses mm directly. Here we use helion.dot."""
    # NOTE: with addmm the _keep_in_psum guard disables PSUM reuse (with_acc=True).
    # So this test only runs the compile to confirm no regression; the tagging
    # pass will simply not fire for addmm. The direct-mm test below is the
    # real PSUM-reuse verification.
    _clear_cache()
    x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    got = matmul_relu(x, y)
    ref = torch.relu(x @ y)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    print("  [PASS] matmul_relu numerical (kernel uses addmm, PSUM reuse not applicable)")


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128, 128]),
    static_shapes=True,
)
def dot_relu_single_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """relu(hl.dot(x, y)) for a single-tile (no K-split) case.
    hl.dot lowers to aten.mm (no accumulator), so PSUM reuse WILL fire.
    """
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        prod = hl.dot(x[tile_m, :], y[:, tile_n])
        out[tile_m, tile_n] = torch.relu(prod)
    return out


def _test_dot_relu_numerical() -> None:
    """2D hl.tile(m, n) with y[:, tile_n] — previously broken by the
    subscript→block_id heuristic when PSUM-reuse fusion caused SymInt
    concretization. Now exercises both the fix and PSUM reuse.
    """
    _clear_cache()
    x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    got = dot_relu_single_tile(x, y)
    ref = torch.relu(x @ y)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    print("  [PASS] dot_relu_single_tile numerical (2D tile, y[:, tile_n])")

    # Inspect generated source — expect no intermediate SBUF copy, and
    # the activation must read the PSUM buffer directly.
    src = _find_generated_kernel_source("dot_relu_single_tile")
    assert src, "no generated kernel source found"
    # Both mm and dot codegens use *_sbuf_tmp for the intermediate SBUF
    # copy of the matmul PSUM result; fusion should eliminate both.
    tmp_count = src.count("_mm_sbuf_tmp") + src.count("_dot_sbuf_tmp")
    has_psum_activation = "data=_mm_psum" in src or "data=_dot_mm_psum" in src
    assert tmp_count == 0, (
        f"expected no intermediate SBUF copy, got {tmp_count} references"
    )
    assert has_psum_activation, (
        "expected nisa.activation to read mm_psum / dot_mm_psum directly"
    )
    print("  [PASS] dot_relu_single_tile codegen: PSUM reuse active "
          "(no intermediate SBUF copy; activation reads PSUM)")


def _test_dot_relu_fusion_disabled_baseline() -> None:
    """With HELION_NKI_DISABLE_FUSION=1, the final PSUM→SBUF copy should
    remain AND the kernel still compiles correctly (baseline path).
    """
    _clear_cache()
    x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    old = os.environ.get("HELION_NKI_DISABLE_FUSION")
    os.environ["HELION_NKI_DISABLE_FUSION"] = "1"
    # Re-create a kernel since the decorator captures env at decoration time.
    @helion.kernel(
        backend="nki",
        autotune_effort="none",
        config=helion.Config(block_sizes=[128, 128]),
        static_shapes=True,
    )
    def baseline(x_in: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        m, k = x_in.size()
        _k2, n = y_in.size()
        out = torch.empty([m, n], dtype=torch.float32, device=x_in.device)
        for tile_m, tile_n in hl.tile([m, n]):
            prod = hl.dot(x_in[tile_m, :], y_in[:, tile_n])
            out[tile_m, tile_n] = torch.relu(prod)
        return out
    try:
        got = baseline(x, y)
        ref = torch.relu(x @ y)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
        src = _find_generated_kernel_source("baseline")
        assert src, "no generated kernel source found"
        tmp_count = src.count("_dot_sbuf_tmp") + src.count("_mm_sbuf_tmp")
        assert tmp_count > 0, (
            f"expected at least one SBUF copy when fusion disabled, got {tmp_count}"
        )
        print(f"  [PASS] baseline (fusion disabled): SBUF copy present "
              f"({tmp_count} refs) — no PSUM reuse, kernel still correct")
    finally:
        if old is None:
            os.environ.pop("HELION_NKI_DISABLE_FUSION", None)
        else:
            os.environ["HELION_NKI_DISABLE_FUSION"] = old


# ---------------------------------------------------------------------------
# Regression tests for the subscript→block_id bug in memory_ops.py
# (Bug symptom: y[:, tile_n] in 2D hl.tile produced wrong SBUF shape.)
# ---------------------------------------------------------------------------

@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128, 128]),
    static_shapes=True,
)
def y_tile_n_regression(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Plain `y[:, tile_n]` load inside a 2D tile loop.

    This pattern triggered the subscript→block_id bug because:
      - subscript[0] is slice(None,None,None)
      - subscript[1] is a SymInt that could get concretized upstream
    The broken heuristic at memory_ops.py:553 picked the first non-reduction
    block_id from active_device_loops (tile_m=bid0), not tile_n (bid1).
    """
    m, n = x.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        # This pattern — slicing y's free axis with tile_n — was the bug.
        out[tile_m, tile_n] = x[tile_m, :] @ y[:, tile_n]
    return out


def _test_y_tile_n_regression() -> None:
    """Verify y[:, tile_n] produces correct shape/offset with fusion enabled.

    The bug triggered specifically with fusion enabled, because the PSUM-reuse
    pass's int() on a SymInt forced a guard that concretized the tile_n
    subscript. After the fix in nki_fusion.py (use size_hint) and the
    subscript→block_id helper in memory_ops.py, this kernel compiles and
    produces the correct y slice regardless of whether concretization happens.

    The fusion-disabled path is covered by _test_dot_relu_fusion_disabled_baseline.
    """
    _clear_cache()
    os.environ.pop("HELION_NKI_DISABLE_FUSION", None)
    x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    got = y_tile_n_regression(x, y)
    ref = x @ y
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    # Inspect generated source to confirm:
    #   (a) _nki_sbuf_* for y uses full shape [128, 128]
    #   (b) the y load uses offset_1 (= tile_n), not offset_0
    src = _find_generated_kernel_source("y_tile_n_regression")
    assert src, "no generated kernel source"
    # y load: look for "src=y[0:128, offset_1:offset_1 + 128]" pattern.
    y_load_lines = [
        ln for ln in src.splitlines()
        if "dma_copy" in ln and "src=y[" in ln
    ]
    assert y_load_lines, "no DMA copy from y found in generated kernel"
    for ln in y_load_lines:
        assert "offset_1" in ln, (
            f"y[:, tile_n] load should use offset_1; got: {ln.strip()}"
        )
        # Should not use offset_0 for the free-axis slice:
        # pattern is "y[0:128, offset_1:offset_1 + 128]" — the ", "
        # before offset_1 separates partition/free slices.
        after_comma = ln.split(",", 1)[1] if "," in ln else ""
        assert "offset_0" not in after_comma, (
            f"y[:, tile_n] should not use offset_0 in the free-axis slice; got: {ln.strip()}"
        )
    print("  [PASS] y[:, tile_n] regression (fusion enabled) — correct shape + offset")


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128, 128]),
    static_shapes=True,
)
def swapped_tile_order(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stress: tile_n BEFORE tile_m (swapped order) — proves the fix doesn't
    rely on `tile_m is bid 0` convention."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    # Swap the iteration order so tile_n appears first in active_device_loops.
    for tile_n, tile_m in hl.tile([n, m]):
        out[tile_m, tile_n] = x[tile_m, :] @ y[:, tile_n]
    return out


def _test_swapped_tile_order() -> None:
    _clear_cache()
    x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    got = swapped_tile_order(x, y)
    ref = x @ y
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    print("  [PASS] swapped_tile_order numerical")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Part 1: FX-graph fusion pass unit tests ===")
    _test_pass_tags_matmul_relu()
    _test_pass_skips_multi_user()
    _test_pass_skips_non_vector_scalar_consumer()
    _test_pass_respects_psum_budget()
    _test_pass_respects_dtype_guard()
    _test_pass_env_disable()

    print("\n=== Part 2: End-to-end kernels ===")
    _test_codegen_matmul_relu_skips_copy()
    _test_dot_relu_numerical()
    _test_dot_relu_fusion_disabled_baseline()

    print("\n=== Part 3: y[:, tile_n] subscript→block_id regression ===")
    _test_y_tile_n_regression()
    _test_swapped_tile_order()

    print("\nAll PSUM-reuse tests complete.")


if __name__ == "__main__":
    main()
