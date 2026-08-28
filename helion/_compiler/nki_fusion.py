"""FX-graph fusion passes for the NKI backend.

This module runs over the final FX graph **before** NKI codegen and
annotates nodes with metadata keys that later stages (``_nki_dot``,
``_nki_tensor_tensor``, etc.) honor to avoid emitting redundant
instructions.

The passes here operate purely on FX node metadata — they never
mutate the graph structure. This keeps them safe to run repeatedly
and easy to disable (set ``HELION_NKI_DISABLE_FUSION=1``).

Passes
------
annotate_psum_reuse(graph):
    Detects the pattern ``matmul → single_user_vector_or_scalar_op``
    and tags the matmul node with ``meta["nki_keep_in_psum"] = True``
    and the consumer node with ``meta["nki_read_psum_from"] =
    <matmul_fx_name>``. Downstream codegen uses these tags to read
    the matmul result out of PSUM directly and skip the final
    ``tensor_copy(sbuf, psum)``.

See HANDOFF_NKI_CHANGES.md for context on the PSUM/SBUF memory model.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.fx import Graph, Node


# ---------------------------------------------------------------------------
# Pattern descriptors
# ---------------------------------------------------------------------------

# Matmul ATen targets. nc_matmul writes to PSUM; its result is a candidate
# for PSUM-reuse whenever it has exactly one downstream user.
_MATMUL_TARGETS: tuple[object, ...] = (
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.baddbmm.default,
)


def _is_matmul_target(target: object) -> bool:
    """Return True if ``target`` is a matmul the fusion pass should consider.

    Covers:
      - aten.mm / bmm / addmm / baddbmm (handled by _nki_dot)
      - helion.language.dot (handled by matmul_ops.py NKI codegen)
    """
    if target in _MATMUL_TARGETS:
        return True
    # Lazy import to avoid circular dependency between language and compiler.
    try:
        from ..language.matmul_ops import dot as _hl_dot
    except Exception:
        return False
    return target is _hl_dot

# ATen ops whose NKI codegen goes through Vector or Scalar engine paths and
# therefore **can** read from PSUM directly (per the hardware manual:
#   Vector Engine: input SBUF or PSUM, output SBUF or PSUM
#   Scalar Engine: input SBUF or PSUM, output SBUF or PSUM
# Tensor Engine inputs must be SBUF, so we explicitly exclude matmul-style
# consumers.
_VECTOR_SCALAR_CONSUMER_TARGETS: frozenset[object] = frozenset({
    # elementwise arithmetic (Vector/Scalar)
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.maximum.default,
    torch.ops.aten.minimum.default,
    torch.ops.aten.neg.default,
    # activations (Scalar)
    torch.ops.aten.relu.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.log.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.abs.default,
    torch.ops.aten.square.default,
    # reductions (Vector engine tensor_reduce accepts PSUM input)
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.default,
    torch.ops.aten.mean.dim,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
})


# ---------------------------------------------------------------------------
# PSUM budget
# ---------------------------------------------------------------------------

# PSUM bank capacity is ~2KB per partition on Trn1. The moving-operand free
# dim of nc_matmul is capped at 512 fp32 elements (2048 bytes) per bank.
# If the downstream consumer's tile exceeds the per-partition byte budget we
# cannot keep it in PSUM, so fall back to the normal SBUF copy path.
_PSUM_BYTES_PER_PARTITION = 2048

_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def _tile_bytes_per_partition(val: "torch.Tensor") -> int | None:
    """Byte-size of the free-axis row of a 2D-squeezed tile.

    Returns None if the shape cannot be resolved at compile time.

    IMPORTANT: must NOT call ``int(symint)`` on a shape dim, because that
    forces a guard on the underlying sympy symbol, which propagates through
    the whole ShapeEnv and concretizes every SymInt sharing the symbol.
    That caused `y[:, tile_n]` loads to lose their tile_n symbol and the
    load codegen to pick the wrong block_id. Use ``size_hint`` via the
    CompileEnvironment for a non-guard-inducing size lookup.
    """
    if not isinstance(val, torch.Tensor):
        return None
    dtype_bytes = _DTYPE_BYTES.get(val.dtype)
    if dtype_bytes is None:
        return None
    # Squeeze leading 1-dims so we're comparing free-axis bytes consistently
    # with NKIOpOverrides._squeeze_shape_2d.
    shape = list(val.shape)
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) > 2:
        # Can't reason about >2D in a fusion pass; bail conservatively.
        return None
    if len(shape) == 0:
        return dtype_bytes
    free = shape[-1]
    if isinstance(free, int):
        return free * dtype_bytes
    if isinstance(free, torch.SymInt):
        # Use size_hint to peek at the value WITHOUT adding a guard on the
        # underlying sympy symbol. Calling ``int(symint)`` here would add a
        # specialization guard that propagates through ShapeEnv and can
        # concretize subscript SymInts in downstream load codegen. The NKI
        # load codegen is now robust against that concretization (see the
        # _nki_subscript_block_id helper in memory_ops.py), but using
        # size_hint is still the cheapest and most conservative choice.
        try:
            from .compile_environment import CompileEnvironment
            env = CompileEnvironment.current()
            free_int = int(env.size_hint(free))
        except Exception:
            return None
        return free_int * dtype_bytes
    return None


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

def _is_fusion_disabled() -> bool:
    return os.environ.get("HELION_NKI_DISABLE_FUSION", "") not in ("", "0", "false", "False")


def annotate_psum_reuse(graph: "Graph") -> int:
    """Tag matmul → single-user Vector/Scalar-op patterns for PSUM reuse.

    Returns the number of matmul nodes that were tagged.

    Safety guards:
      - Matmul must have exactly one FX user.
      - Consumer target must be in ``_VECTOR_SCALAR_CONSUMER_TARGETS``.
      - Matmul output dtype must be fp32 or bf16 (PSUM dtype constraint).
      - Per-partition byte size must fit in one PSUM bank (~2KB).
    """
    if _is_fusion_disabled():
        return 0

    tagged = 0
    _hl_dot = None
    try:
        from ..language.matmul_ops import dot as _hl_dot  # noqa: F811
    except Exception:
        pass

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if not _is_matmul_target(node.target):
            continue
        # For hl.dot, skip when an accumulator is supplied (with_acc=True path
        # goes through a separate codegen branch that doesn't benefit from
        # PSUM reuse since it emits a tensor_tensor add on top of the SBUF
        # result).
        if _hl_dot is not None and node.target is _hl_dot:
            acc_arg = node.args[2] if len(node.args) >= 3 else node.kwargs.get("acc")
            if acc_arg is not None:
                continue
        # For aten.addmm/baddbmm, skip as well (they emit bias-add on SBUF).
        if node.target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default):
            continue
        users = list(node.users)
        if len(users) != 1:
            continue
        consumer = users[0]
        if consumer.op != "call_function":
            continue
        if consumer.target not in _VECTOR_SCALAR_CONSUMER_TARGETS:
            continue

        # Matmul output dtype guard.
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        if val.dtype not in (torch.float32, torch.bfloat16):
            continue

        # PSUM budget guard.
        row_bytes = _tile_bytes_per_partition(val)
        if row_bytes is None or row_bytes > _PSUM_BYTES_PER_PARTITION:
            continue

        # All guards passed — tag both nodes.
        node.meta["nki_keep_in_psum"] = True
        consumer.meta["nki_read_psum_from"] = node.name
        tagged += 1

    return tagged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def annotate_fx_graph(graph: "Graph") -> dict[str, int]:
    """Run every NKI fusion pass on ``graph``.

    Returns a dict of ``pass_name -> num_sites_tagged`` for diagnostics.
    Passes only set ``node.meta`` entries; they never rewrite the graph.
    """
    return {
        "psum_reuse": annotate_psum_reuse(graph),
    }
