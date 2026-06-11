# Guide: Refactor NKI load/store into a `nki/` subpackage (mirroring Pallas)

**Audience:** an agent/engineer extracting the inline NKI codegen out of the giant shared
`helion/language/memory_ops.py` into a dedicated `helion/_compiler/nki/` subpackage, exactly the way
Pallas already structures its backend.

**Goal:** the NKI `load`/`store` codegen registrations in `memory_ops.py` become THIN SHIMS that delegate to
`helion/_compiler/nki/codegen.py`, with all NKI helpers moved into the subpackage. Generated NKI code must
be **byte-for-byte identical before and after** — this is a pure code-move refactor, no behavior change.

---

## 1. How Pallas does it (the model to copy)

Pallas keeps its `@_decorators.codegen(load/store, "pallas")` functions in `memory_ops.py` to ~14 lines
each, and puts all real logic in a `helion/_compiler/pallas/` subpackage.

```
helion/_compiler/pallas/
├── __init__.py        # 3 lines — just a docstring; exports NOTHING (modules imported by path)
├── codegen.py         # 638 lines — load_expr / index_parts / sliced_value_for_store / vmem_name / ...
├── gather.py          # 300 lines — emit_gather / emit_scatter_store (indirect gather/scatter)
└── plan_tiling.py     # 520 lines — plan_tiling() analysis pass + IndirectGatherPattern/IndirectScatterPattern
```

**The load shim** (`memory_ops.py`, `@_decorators.codegen(load, "pallas")`) is literally:
```python
@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
    if tile_index_result is not None:
        return tile_index_result
    return pallas_codegen.load_expr(state, list(subscript), tensor)   # <-- delegate
```

**The store shim** does a little orchestration but still delegates the heavy lifting:
```python
@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    ...
    name = pallas_codegen.vmem_name(state, name)
    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    value = pallas_codegen.sliced_value_for_store(state, tensor, subscript, parts, value)
    ...
    from .._compiler.pallas.gather import emit_scatter_store          # lazy import
    from .._compiler.pallas.plan_tiling import IndirectScatterPattern
    ...
```

**Import conventions Pallas uses (copy these):**
- Top-of-file module alias for the common case:
  `from .._compiler.pallas import codegen as pallas_codegen` (memory_ops.py L62).
- **Lazy (function-body) imports** for the gather/scatter/pattern modules
  (`from .._compiler.pallas.gather import emit_scatter_store` inside the function) — avoids import cycles
  between `memory_ops` ↔ subpackage ↔ backend.
- `__init__.py` exports nothing; callers import concrete modules by path
  (`from .._compiler.pallas import codegen`, `from .._compiler.pallas.plan_tiling import plan_tiling`).
- The backend wires its analysis pass in `Backend.pre_codegen` (backend.py L2511):
  `from .pallas.plan_tiling import plan_tiling; plan_tiling(graphs, config, tile_strategy)`.
- Other call sites that need the subpackage (atomic_ops.py, device_function.py, backend.py) use the same
  by-path imports — so the subpackage is the single home for this logic.

Key takeaway: **the inline function is a dispatcher; the subpackage is the implementation.** Pallas's
indexing machinery (~1.5k lines, 25 gather + 14 scatter markers) is just as large as NKI's — it's only
*located* in a subpackage. NKI should match that location.

---

## 2. What to move (exact inventory, current line numbers in `helion/language/memory_ops.py`)

**Registered entry points (become shims, keep the `@_decorators.codegen(... "nki")` decorator in
memory_ops.py):**
- `@_decorators.codegen(load, "nki")`  — L7721, body ~3400 lines  → move body to `nki/codegen.py:load_expr`
- `@_decorators.codegen(store, "nki")` — L11125, body ~1400 lines → move body to `nki/codegen.py:store_stmt`

**Helper functions to move into the subpackage (all currently module-level in memory_ops.py):**
| Function | Line | Suggested home |
|---|---|---|
| `_nki_shifted_tile_subscript` | 6681 | `nki/indexing.py` |
| `_nki_indirect_gather` | 6907 | `nki/gather.py` |
| `_nki_lookup_sbuf_shape_dtype` | 7120 | `nki/sbuf.py` (small shared util) |
| `_nki_as_uint32_p1_vector` | 7137 | `nki/gather.py` (the >128 transpose-tiling helper) |
| `_nki_store_3d_row_scatter` | 7228 | `nki/gather.py` |
| `_nki_row_index_gather` | 7340 | `nki/gather.py` |
| `_nki_subscript_block_id` | 7591 | `nki/indexing.py` (the TileIdOrigin resolver) |

**Shared types already factored out (do NOT move, just import):**
- `IndirectAP`, `DynamicAP`, the `DimAccess` hierarchy live in `helion/language/_nki_dim_access.py`
  (imported at memory_ops.py L69-70). The subpackage imports from there.

**Note:** `_nki_subscript_block_id` is ALSO referenced by the store path and is the canonical resolver — put
it in a shared `nki/indexing.py` and import it into both `nki/codegen.py` and anywhere else that needs it.

---

## 3. Proposed subpackage layout

```
helion/_compiler/nki/
├── __init__.py        # docstring only; export nothing (match pallas)
├── codegen.py         # load_expr(state, subscript, tensor) -> ast.AST
│                       # store_stmt(state) -> None   (or a function the shim calls)
├── indexing.py        # _nki_subscript_block_id, _nki_shifted_tile_subscript  (subscript→block_id/slice)
├── gather.py          # _nki_indirect_gather, _nki_row_index_gather, _nki_store_3d_row_scatter,
│                       # _nki_as_uint32_p1_vector   (the indirect gather/scatter + index-vector machinery)
└── sbuf.py            # _nki_lookup_sbuf_shape_dtype  (+ any other tiny SBUF-shape utilities)
```
(Splitting into indexing/gather/sbuf is optional — you could put everything in `codegen.py` first and split
later. The *priority* is getting it OUT of the 12.5k-line shared file. Pallas uses 3 files; 3-4 is fine.)

---

## 4. Step-by-step procedure

1. **Baseline the codegen FIRST (this is the safety net).** With a clean tree, capture NKI codegen for a
   broad set of examples so you can prove byte-identity after the move:
   ```
   # for each example, with PYTHONPATH=/home/ubuntu/helion_port and HELION_BACKEND=nki:
   python /home/ubuntu/codegen_parity_sweep.py port <stem>   # emits to_triton_code
   ```
   Save outputs for: add, attention, gdn_fwd_h, jagged_hstu_attn, int4_gemm, embedding, gather_gemv,
   moe_matmul_ogs, layer_norm, long_sum, matmul, softmax, cross_entropy (covers contiguous / gather /
   scatter / shifted-iota / n-D-flatten / reduction paths).

2. **Create the subpackage skeleton.** `helion/_compiler/nki/__init__.py` (docstring only, like pallas's).

3. **Move the helpers** (Section 2 table) verbatim into the new modules. Keep their bodies BYTE-IDENTICAL —
   only the location changes. Fix their imports (they use `CompileEnvironment`, `statement_from_string`,
   `ast`, `torch`, `IndirectAP`/`DynamicAP` from `..._nki_dim_access` — note the relative-path depth changes
   from `helion/language/` to `helion/_compiler/nki/`, so `from ._nki_dim_access import IndirectAP` becomes
   `from ...language._nki_dim_access import IndirectAP`).

4. **Move the load/store bodies** into `nki/codegen.py` as `load_expr(state, subscript, tensor)` and
   `store_stmt(state)`. Replace the inline bodies in `memory_ops.py` with thin shims:
   ```python
   @_decorators.codegen(load, "nki")
   def _(state: CodegenState) -> ast.AST:
       from .._compiler.nki import codegen as nki_codegen   # lazy, mirrors pallas
       tensor = state.proxy_arg(0); subscript = state.proxy_arg(1)
       assert isinstance(tensor, torch.Tensor)
       assert isinstance(subscript, (list, tuple))
       return nki_codegen.load_expr(state, list(subscript), tensor)

   @_decorators.codegen(store, "nki")
   def _(state: CodegenState) -> None:
       from .._compiler.nki import codegen as nki_codegen
       nki_codegen.store_stmt(state)
   ```
   (If the store needs `value`/`extra_mask` from `state.ast_arg(2)`/`ast_args[3]`, read them in the shim and
   pass them in, or read them inside `store_stmt` from `state` — match whatever the current body does.)

5. **Watch for import cycles.** `memory_ops.py` is imported very early. The subpackage will import things
   from `memory_ops` (e.g. it currently calls sibling helpers). Two options, in order of preference:
   (a) move ALL `_nki_*` helpers together so the subpackage is self-contained and only imports from
   `_compiler/*` and `language/_nki_dim_access`; (b) where a cross-reference is unavoidable, use a lazy
   in-function import (pallas does this for gather/plan_tiling). Do NOT add `nki` imports at module top of
   `memory_ops.py` beyond the existing `_nki_dim_access` ones unless they're cycle-free.

6. **Update the known external call sites.** Three helpers are imported OUTSIDE `memory_ops.py` — these
   imports MUST be repointed to the new subpackage location or they break:
   - `helion/language/atomic_ops.py` L1786-1787 does
     `from .memory_ops import _nki_row_index_gather` and `from .memory_ops import _nki_subscript_block_id`
     (used at L1819, L2034, L2040 in atomic scatter/gather codegen). Repoint these to
     `from .._compiler.nki.gather import _nki_row_index_gather` /
     `from .._compiler.nki.indexing import _nki_subscript_block_id`.
   - `inductor_lowering.py` L1549 only MENTIONS `_nki_shifted_tile_subscript` in a comment — no code change,
     but update the comment if you rename.
   Re-grep after moving to be sure nothing else references them:
   `grep -rn "_nki_subscript_block_id\|_nki_row_index_gather\|_nki_shifted_tile_subscript\|_nki_as_uint32_p1_vector\|_nki_indirect_gather\|_nki_store_3d_row_scatter\|_nki_lookup_sbuf_shape_dtype" helion/ | grep -v "_compiler/nki/"`
   — every remaining hit must import from the new location. (`atomic_ops.py` keeping these as *lazy*
   in-function imports is the cleanest, mirroring how it currently imports them lazily.)

7. **Verify byte-identity (the gate).** Re-run the Section-1 captures and `diff` against the baseline. Every
   example must be **IDENTICAL**. Then run CPU-sim on a few to confirm they still execute:
   ```
   PYTHONPATH=/home/ubuntu/helion_port HELION_BACKEND=nki HELION_NKI_SIMULATE=1 \
     NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python /tmp/ref_wt/examples/gdn_fwd_h.py
   ```
   (See `/home/ubuntu/sim_sweep.py` for the env. NOTE: `helion` is pip-installed editable from
   `helion_nki`; you MUST set `PYTHONPATH=/home/ubuntu/helion_port` or you'll test the wrong tree.)

8. **Lint.** `ruff format` + `ruff check` the new files (line length 88, double quotes, sorted single-line
   imports, `from __future__ import annotations` at top of each module). CI gates on ruff + pyrefly.

---

## 5. Acceptance criteria

- [ ] `helion/_compiler/nki/` exists with `__init__.py` + codegen + helper modules.
- [ ] The two `@_decorators.codegen(load/store, "nki")` functions in `memory_ops.py` are thin shims (~5-15
      lines each) that delegate to the subpackage.
- [ ] All `_nki_*` helpers moved out of `memory_ops.py` (grep `^def _nki` in memory_ops returns nothing, or
      only ones genuinely shared with non-NKI code — there should be none).
- [ ] `import helion` works; `list_backends()` includes `'nki'`; no import cycles.
- [ ] **Byte-identical** NKI codegen for all baselined examples (the hard gate — this is a pure move).
- [ ] CPU-sim spot check (gdn_fwd_h, jagged_hstu_attn, embedding, matmul) still PASS.
- [ ] ruff clean.
- [ ] No change to any non-NKI backend (only `memory_ops.py` shims + new `nki/` files touched; the Triton/
      Pallas/CuTe/Metal load/store functions are untouched).

---

## 6. Why this matters (context for the agent)

This is the single biggest structural blocker to upstreaming the NKI backend to pytorch-labs/helion. NKI's
codegen is FUNCTIONALLY isolated already (separate per-backend `@_decorators.codegen("nki")` registrations,
all hardware logic in `nki_backend.py`), and its load/store logic is not more complex than Pallas's in
substance — but it currently lives INLINE in the 12.5k-line shared `memory_ops.py` instead of a subpackage.
Maintainers reviewing the PR will want the Pallas structure: a thin dispatch in the shared file, the
implementation in `helion/_compiler/nki/`. This refactor makes the diff reviewable and the shared-file
footprint small, without changing a single byte of generated code.
