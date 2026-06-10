"""NKI subscript/indexing resolvers.

Moved verbatim from ``helion/language/memory_ops.py`` as part of the NKI
subpackage refactor. ``_nki_subscript_block_id`` is the canonical
subscript->block_id resolver used by both the load and store codegen and by
``helion/language/atomic_ops.py``.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileIdOrigin

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _nki_shifted_tile_subscript(
    fx_node: object,
    state: "CodegenState",
    env: "CompileEnvironment",
) -> str | None:
    """Recognize ``tile.index ± const`` subscripts and return a shifted slice.

    For patterns like ``y[:, tile1.index - x.size(1)]``, the index tensor is
    ``tile1.index - x.size(1)`` — a 1D tile of integers. Instead of emitting
    an indirect gather, observe that this is just a UNIFORM shift by a
    constant: the gather of ``y[p, tile1.index - C]`` is equivalent to the
    slice ``y[p, offset_1 - C : offset_1 - C + block_size]``.

    Supports:
      - ``aten.sub.Tensor(tile_index, const_int)`` → ``offset - const``
      - ``aten.add.Tensor(tile_index, const_int)`` → ``offset + const``
      - ``aten.sub.Tensor(const_int, tile_index)`` → ``const - offset``
        (only valid for `block_size == 1`; otherwise the slice direction
        is reversed which dma_copy doesn't support)

    Returns a slice string ``"start:end"`` or None if the pattern doesn't
    apply.
    """
    if not isinstance(fx_node, torch.fx.Node):
        return None

    target = getattr(fx_node, "target", None)
    target_name = str(target) if target else ""
    import operator as _operator
    is_add = (
        "add.Tensor" in target_name
        or "add.Scalar" in target_name
        or target is torch.ops.aten.add.Tensor
        or target is torch.ops.aten.add.Scalar
        or target is _operator.add
    )
    is_sub = (
        "sub.Tensor" in target_name
        or "sub.Scalar" in target_name
        or target is torch.ops.aten.sub.Tensor
        or target is torch.ops.aten.sub.Scalar
        or target is _operator.sub
    )
    if not (is_add or is_sub):
        return None

    args = fx_node.args
    if len(args) != 2:
        return None

    def _get_tile_index_block_id(node: object) -> int | None:
        """If ``node`` is ``tile.index`` (FX value is a 1D SymInt-tensor),
        return its block_id.  tile.index gets lowered as a tensor-valued
        node whose shape is ``[block_size]`` symbolically.
        Also recognizes scalar grid loop variables (block_size == 1).
        """
        if not isinstance(node, torch.fx.Node):
            return None
        val = node.meta.get("val")
        # Scalar grid loop variable (hl.grid): val is int or SymInt, block_size=1
        if isinstance(val, (int, torch.SymInt)):
            node_ast = state.codegen.ast_for_fx_node(node)
            if isinstance(node_ast, ast.AST):
                node_expr = ast.unparse(node_ast)
                for bid in state.codegen.active_device_loops:
                    try:
                        block_size = int(
                            env.block_sizes[bid].from_config_assert(state.config)
                        )
                    except Exception:
                        continue
                    if block_size == 1 and node_expr == state.codegen.offset_var(bid):
                        return bid
        if isinstance(val, torch.Tensor) and val.ndim == 1:
            size = val.size(0)
            if isinstance(size, torch.SymInt):
                bid = env.get_block_id(size)
                if bid is not None:
                    return bid
                sym_expr = size._sympy_()
                free = getattr(sym_expr, "free_symbols", None)
                if free:
                    for sym in free:
                        bid = env.get_block_id(sym)
                        if bid is not None:
                            return bid
                try:
                    size_hint = int(env.size_hint(size))
                except Exception:
                    size_hint = -1
                for bid in state.codegen.active_device_loops:
                    try:
                        block_size = int(
                            env.block_sizes[bid].from_config_assert(state.config)
                        )
                    except Exception:
                        continue
                    if block_size == size_hint:
                        return bid
            node_target = getattr(node, "target", None)
            node_target_name = str(node_target) if node_target else ""
            if "tile_index" in node_target_name and len(
                state.codegen.active_device_loops
            ) == 1:
                return next(iter(state.codegen.active_device_loops))
        return None

    def _get_const_int(node: object) -> int | None:
        if isinstance(node, int):
            return node
        if isinstance(node, torch.SymInt):
            try:
                return int(node._sympy_())
            except Exception:
                # Don't use size_hint here — it gives a fake concrete value
                # that may not represent the actual runtime value (e.g. 8192
                # for a loop variable that will be 0..1 at runtime).
                return None
        if isinstance(node, torch.fx.Node):
            val = node.meta.get("val")
            if isinstance(val, int):
                return val
            if isinstance(val, torch.SymInt):
                try:
                    return int(val._sympy_())
                except Exception:
                    return None  # Same: don't use size_hint
        try:
            return int(node)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return None

    lhs_bid = _get_tile_index_block_id(args[0])
    rhs_const = _get_const_int(args[1]) if lhs_bid is not None else None

    if lhs_bid is not None and lhs_bid in state.codegen.active_device_loops:
        offset_var = state.codegen.offset_var(lhs_bid)
        block_size = int(env.block_sizes[lhs_bid].from_config_assert(state.config))
        # Get the shift expression — either numeric const or AST expression
        # (e.g. tile_c.begin * chunk_size → "offset_5 * 128")
        _rhs_expr: str | None = None
        if rhs_const is not None:
            _rhs_expr = str(rhs_const)
        elif isinstance(args[1], torch.fx.Node):
            _rhs_ast = state.codegen.ast_for_fx_node(args[1])
            if isinstance(_rhs_ast, ast.AST):
                _candidate = ast.unparse(_rhs_ast)
                # Only use as a shift if it's a scalar expression — not an
                # SBUF tile variable. SBUF tiles (iota indices, loaded tiles,
                # etc.) cannot be used as integer offsets in DMA slice
                # expressions. Check both the sbuf_shapes dict and variable
                # naming conventions (_nki_*, indices_*).
                _sbuf_shapes = state.device_function._nki_sbuf_shapes
                # Block SBUF tile variables from being used as DMA slice offsets.
                # SBUF tiles cannot be used as Python integers in DMA slice
                # expressions — they cause NKI compile errors like
                # "'add' expected (int, int) ... got (int, object)".
                _is_sbuf_tile = (
                    _candidate in _sbuf_shapes
                    or _candidate.startswith(("_nki_", "indices_"))
                )
                if not _is_sbuf_tile:
                    _rhs_expr = _candidate
        if _rhs_expr is not None:
            if is_sub:
                start = f"{offset_var} - {_rhs_expr}"
            else:  # is_add
                start = f"{offset_var} + {_rhs_expr}"
            # Build the end expression.  When the shift is a numeric constant,
            # fold it with block_size arithmetically to avoid emitting
            # "(offset_var + N) + block_size" which NKI's symbolic tracer
            # cannot evaluate for compound affine expressions.
            try:
                _rhs_int = int(_rhs_expr)
                if is_sub:
                    end = f"{offset_var} - {_rhs_int - block_size}"
                else:
                    end = f"{offset_var} + {_rhs_int + block_size}"
            except (ValueError, TypeError):
                end = f"{start}+{block_size}"
            return f"{start}:{end}"

    # Reverse: const ± tile
    rhs_bid = _get_tile_index_block_id(args[1])
    lhs_const = _get_const_int(args[0]) if rhs_bid is not None else None
    if rhs_bid is not None and rhs_bid in state.codegen.active_device_loops:
        offset_var = state.codegen.offset_var(rhs_bid)
        block_size = int(env.block_sizes[rhs_bid].from_config_assert(state.config))
        # Get the LHS expression — either a numeric const or an AST expression
        # (e.g. tile_c.begin * chunk_size → "offset_5 * 128")
        _lhs_expr: str | None = None
        if lhs_const is not None:
            _lhs_expr = str(lhs_const)
        elif isinstance(args[0], torch.fx.Node):
            _lhs_ast = state.codegen.ast_for_fx_node(args[0])
            if isinstance(_lhs_ast, ast.AST):
                _candidate_lhs = ast.unparse(_lhs_ast)
                _sbuf_shapes_rev = state.device_function._nki_sbuf_shapes
                _is_sbuf_tile_lhs = (
                    _candidate_lhs in _sbuf_shapes_rev
                    or _candidate_lhs.startswith(("_nki_", "indices_"))
                )
                if not _is_sbuf_tile_lhs:
                    _lhs_expr = _candidate_lhs
        if _lhs_expr is not None:
            if is_add:
                start = f"{_lhs_expr} + {offset_var}"
                # Fold numeric LHS with block_size to avoid double-addition.
                try:
                    _lhs_int = int(_lhs_expr)
                    end = f"{offset_var} + {_lhs_int + block_size}"
                except (ValueError, TypeError):
                    end = f"{start}+{block_size}"
                return f"{start}:{end}"
            # const - tile: only safe when block_size == 1
            if is_sub and block_size == 1:
                return f"{_lhs_expr} - {offset_var}"

    return None


def _nki_subscript_block_id(
    sub_val: object,
    fx_node_i: object,
    env: CompileEnvironment,
) -> int | None:
    """Return the block_id a subscript element corresponds to, or None.

    Tries in order:
      1. The live SymInt in ``sub_val`` via ``env.get_block_id``.
      2. The FX subscript node's ``meta["val"]`` (which might still be a
         live SymInt even if ``sub_val`` has been concretized).
      3. Free sympy symbols of either — match each symbol to a registered
         block size via ``env.get_block_id``.
      4. The FX node's name (Helion names symnode getters after their debug
         names ``block_size_N`` / ``rdim_N`` per compile_environment.py:245).
      5. The FX node's first positional arg (which for ``_get_symnode`` is
         the debug name string).

    This is the canonical NKI-backend subscript→block_id resolver. It does
    NOT consult ``active_device_loops``: positional heuristics there are
    fragile (e.g. broke for ``y[:, tile_n]`` when subscripts concretize).
    """
    import sympy as _sp

    # 1. Direct SymInt lookup.
    if isinstance(sub_val, torch.SymInt):
        bid = env.get_block_id(sub_val)
        if bid is not None:
            return bid
        sym_expr = sub_val._sympy_()
        # Only match a bare symbol (e.g. u2), not a compound expression
        # like u2 + 1. Compound expressions containing a block symbol should
        # be handled by _nki_shifted_tile_subscript instead.
        if isinstance(sym_expr, _sp.Symbol):
            bid = env.get_block_id(sym_expr)
            if bid is not None:
                return bid
            # ``tile.id`` / ``tile.begin`` symbols carry a TileIdOrigin /
            # TileBeginOrigin (subclasses of GridOrigin). ``get_block_id``
            # only resolves exact GridOrigin (upstream keeps the subclasses
            # distinct because tile.end/count need different math), so a
            # tile.id subscript like ``h[tile_b.id, ...]`` would otherwise
            # not map to its block. Recover the block_id from the origin —
            # the NKI load/store codegen already emits the correct
            # ``offset`` / ``offset // block_size`` for these downstream.
            origin_info = HostFunction.current().expr_to_origin.get(sym_expr)
            if origin_info is not None and isinstance(
                origin_info.origin, (TileIdOrigin, TileBeginOrigin)
            ):
                return origin_info.origin.block_id

    # 1b. FakeTensor subscript (1D tile): look at size(0) for the block_id.
    # Handles compound subscripts like "tile_c * chunk_size + tile_m.index"
    # which trace as 1D FakeTensors whose size is the tile_m block_size.
    if isinstance(sub_val, torch.Tensor) and sub_val.ndim == 1:
        size0 = sub_val.size(0)
        if isinstance(size0, torch.SymInt):
            bid = env.get_block_id(size0)
            if bid is not None:
                return bid
            sym_expr0 = size0._sympy_()
            if isinstance(sym_expr0, _sp.Symbol):
                bid = env.get_block_id(sym_expr0)
                if bid is not None:
                    return bid
            # Also try free symbols (for expressions like u2 in tensor size)
            free0 = getattr(sym_expr0, "free_symbols", None)
            if free0:
                for sym in free0:
                    bid = env.get_block_id(sym)
                    if bid is not None:
                        return bid

    # 2. FX node's meta["val"] (survives even if sub_val was concretized).
    if isinstance(fx_node_i, torch.fx.Node):
        fx_val = fx_node_i.meta.get("val")
        if isinstance(fx_val, torch.SymInt):
            bid = env.get_block_id(fx_val)
            if bid is not None:
                return bid
            sym_expr = fx_val._sympy_()
            # Same as above: only match bare symbols.
            if isinstance(sym_expr, _sp.Symbol):
                bid = env.get_block_id(sym_expr)
                if bid is not None:
                    return bid
        elif isinstance(fx_val, _sp.Expr):
            free = getattr(fx_val, "free_symbols", None)
            if free:
                for sym in free:
                    bid = env.get_block_id(sym)
                    if bid is not None:
                        return bid

        # 3. FX node name: "block_size_N" -> block_id N. Tile symnodes get
        #    this name via device_ir._get_proxy_slot's debug_name.
        node_name = getattr(fx_node_i, "name", None) or ""
        if node_name.startswith("block_size_"):
            try:
                cand = int(node_name.removeprefix("block_size_").split("_")[0])
                if 0 <= cand < len(env.block_sizes):
                    return cand
            except (ValueError, IndexError):
                pass
        elif node_name.startswith("rdim_"):
            try:
                cand = int(node_name.removeprefix("rdim_").split("_")[0])
                if 0 <= cand < len(env.block_sizes):
                    return cand
            except (ValueError, IndexError):
                pass

        # 4. FX node args[0] may be the debug name string (when target is
        #    _get_symnode).
        args = getattr(fx_node_i, "args", ())
        if args:
            first = args[0]
            if isinstance(first, str):
                for prefix in ("block_size_", "rdim_"):
                    if first.startswith(prefix):
                        try:
                            cand = int(first.removeprefix(prefix).split("_")[0])
                            if 0 <= cand < len(env.block_sizes):
                                return cand
                        except (ValueError, IndexError):
                            pass

    return None
