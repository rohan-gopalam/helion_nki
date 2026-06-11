# NKI Backend Port onto upstream/main — Authoritative Implementation Plan

> Generated 2026-06-08 by a diff-grounded multi-agent analysis of the two real git trees
> (`fix-nki-kernel-compilation` @ `3056e85c` vs `upstream/main` @ `eb61d5b8`), then adversarially
> reviewed. The "Corrections" section and risk register OVERRIDE `NKI_UPSTREAM_MERGE_GUIDE.md`
> wherever they disagree — every override was verified against `git show`/`git diff`.
> Port working tree: `/home/ubuntu/helion_port` (branch `nki-port-v2`).

## 0. Orientation

**Goal.** Port the AWS Trainium **NKI backend** from the reference branch `fix-nki-kernel-compilation` (commit `3056e85c`) onto the newer `upstream/main` (commit `eb61d5b8`). The working port branch `nki-port-v2` is checked out and currently equals `upstream/main`. Merge base is `1bfe577d`.

**The two branches.**
- **Reference (definitive NKI source):** `fix-nki-kernel-compilation`. The entire NKI backend lives in **one file** — `helion/_compiler/backend.py`: `NKIOpOverrides` (class at line 636) + `NKIBackend(Backend)` (class at line 5809). It also carries NKI-only files `nki_fusion.py`, `_nki_dim_access.py`, `nki_search.py`, plus surgical edits across ~25 shared files. **Verified:** `nki_backend.py` does NOT exist on the reference — extracting it is a port-time refactor, not a copy.
- **Target (port onto):** `upstream/main`. NKI does not exist here. Upstream introduced the plugin registry (`backend_registry.py`, 71 lines), a `Backend` base class with **7 `@abc.abstractmethod` methods**, five backend subclasses (`TritonBackend` L947, `TileIRBackend` L1378, `PallasBackend` L1442, `CuteBackend` L3141, `MetalBackend` L4663), a **`type_propagation.py` → `type_info.py` extraction (BOTH files exist upstream; reference has only `type_propagation.py`)**, and `jagged_tile` already present (see §1.3).

**Hard constraints that shape every step.**
1. **Byte-for-byte identical codegen.** `to_triton_code(config)` for the 47 passing examples must be character-for-character identical to the reference. Refactor only when output is provably unchanged. Verbatim-copy the large codegen bodies via `git show` / `git checkout`, never manual paste.
2. **Plugin semantics.** No hard import of NKI classes and **no `torch_xla` import at module-load time** in any shared file. NKI registers through `backend_registry.register_compiler_backend()` via a lazy hook (P1.5).
3. **`MetalBackend` (upstream backend.py L4663) is the template** for a minimal full `Backend` subclass.
4. **No `pip install`, no agent-initiated `git commit` beyond what the user authorizes.** The editable install points at `/home/ubuntu/helion_nki`; the PORT tree is exercised via `PYTHONPATH=/home/ubuntu/helion_port`. **Never reinstall.** All verify commands prepend `PYTHONPATH=/home/ubuntu/helion_port`.
5. **One logical step = one commit.** Verify generated code before each commit. End every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Determinism preconditions for every codegen / byte-diff command (apply globally).** To make `to_triton_code()` reproducible across the reference worktree and the port tree:
```bash
export PYTHONHASHSEED=0
```
Use an identical `Config` object on both sides, constructed without `platform_target` (the reference branch's `Config` lacks that field — see §1.7) so the two configs are field-compatible. If a kernel's codegen consumes RNG, set `torch.manual_seed(0)` inside the generator script. Before trusting any diff, run the **reference-against-itself sanity check** in §Methodology.

**Cache discipline.** The Neuron compiler cache hides codegen bugs — a kernel can "pass" on stale binaries. Codegen diffs (Phase 1) are cache-immune, but any on-device run is not. **After every codegen-affecting change, before any on-device run:**
```bash
rm -rf /tmp/helion_nki_portv2_* /var/tmp/neuron-compile-cache/
export NEURON_CC_FLAGS='--no_cache'
```
Relaxed NKI tolerances (rtol 5e-2 / atol 1.5) can mask a numeric bug AND a stale-cache bug simultaneously — clear cache even though tolerances are loose.

**Testing invocation (no hardware, Phase 1).** Codegen is pure Python; `to_triton_code()` needs no Trainium:
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "..."
PYTHONPATH=/home/ubuntu/helion_port HELION_BACKEND=nki python -c "..."
```

---

## Corrections to the guide and the analyst specs (verified against git — these OVERRIDE all input specs)

- **`type_propagation.py` is where the reference NKI type deltas live; `type_info.py` is UPSTREAM-ONLY.** Verified: `type_info.py` does **not** exist on `fix-nki-kernel-compilation`. Upstream split types across both files. `JaggedTileIndexType` is defined in **upstream `type_info.py` L1141** AND in **reference `type_propagation.py` L1173** — do **NOT** re-add it; it is already present upstream. The three *real* NKI deltas to port into upstream `type_propagation.py` are: (a) `TensorType.propagate_assignment` trailing-singletons allowance (ref L543), (b) `CallableType` `size_hint` capture creating `SymIntType(origin, env.create_unbacked_symint(hint=hint))` (ref L858), (c) the `patch_tensor_factories` guard — see next bullet. (New step P1.3b.)
- **`patch_tensor_factories` moved off `host_function.py` entirely on upstream.** Verified: upstream `host_function.py` no longer imports or calls `patch_tensor_factories` (returns plain `contextlib.nullcontext()` for the profiler path). Upstream now guards it inside `type_propagation.py` L903–911 via `CompileEnvironment.current().backend.pad_factory_tensors_to_power_of_2`. **The reference's NKI guard (`backend_name != "nki"` in host_function.py L126) must NOT be ported as a host_function edit.** Instead, NKI flows in by overriding `pad_factory_tensors_to_power_of_2` to return `False` on `NKIBackend` (this property already exists on the upstream `Backend` base, L102). **The draft's P1.8 host_function.py guard is REMOVED.**
- **`Backend.pad_factory_tensors_to_power_of_2` is an 8th overridable property** (upstream backend.py L102) that `NKIBackend` MUST override (`return False`). It is not one of the 7 abstractmethods but is load-bearing for NKI (avoids power-of-2 factory padding).
- **`reduction_expr` signature differs between branches.** Verified: upstream `Backend.reduction_expr` (L536) takes `*, block_size_var, threads_in_group`. Reference `Backend.reduction_expr` (L212) takes `*, block_size_var, fake_input, fake_output`. Upstream has **zero** `fake_input`/`fake_output` references in backend.py; reference has 45. **NKI carries `fake_input`/`fake_output` through `NKIBackend.reduction_expr` (extracted verbatim in P1.6) and the `reduction_strategy.py` call chain (P1.12, ref callers at L108/118/131/136-137).** Python does not require the base signature to match an override's, so do **not** edit upstream's base `reduction_expr`; just ensure NKI's override + callers agree. Reconcile against upstream's new `threads_in_group`/`thread_linear_index_expr`/`reduction_threads_hint` additions in P1.12.
- **`_check_block_broadcast_compatibility` ALREADY has `ctx: LoweringContext` on upstream** (L536; call site L483). Reference lacks `ctx` (L509). **Do NOT add the `ctx` parameter — it is upstream's existing state.** The NKI work in P1.15 is to ensure NKI's `PointwiseLowering` codegen path passes `ctx` correctly and merges any NKI-specific `block_sizes_proven_equal` body, not a signature change. (Ordering reviewer's "double-patch" warning is valid; the draft's "add ctx param" action was wrong.)
- **`get_masked_value` 'mean' IS genuinely absent upstream.** Verified: upstream `ReductionLowering.get_masked_value` (L949) uses `{"sum","prod","min","max"}`; reference (L727) uses `{"sum","prod","min","max","mean"}`. **What it is:** this set lists reduction types that "preserve zeroness" — if masked-off (padding) lanes contribute 0, the masking optimization can fire. Adding `"mean"` lets a jagged/tiled mean reduction treat padding lanes as zero-contribution, which is exactly what `jagged_mean.py` needs (a historically-failing example). The reference change came from commit `4f8f8c98` ("passing more tests"), an NKI-motivated change. **CAVEAT — verify before porting (the reference made this change UNCONDITIONALLY, not NKI-guarded):** adding `"mean"` here also changes Triton/CuTe behavior, which our NKI-only byte-diff gate will NOT catch. Before P1.15: confirm `mean` is genuinely zero-preserving (then adding it unconditionally is a correct upstream-bug-fill) OR wrap it in `if backend.codegen_name == "nki"` to be conservative and match reference NKI output exactly without touching other backends. Do not blind-copy.
- **`ast_for_fx_node` / `record_fx_node_ast` live on `CodegenInterface` in `helper_function.py`** (reference L49/L52), **not** `generate_ast.py`. Upstream's `CodegenInterface` (L31) has only `statement_owner_node` (L42). Add both to `helper_function.py` (P1.2). Usage count in `memory_ops.py` is **32**, all inside function bodies (verified: first at L571, indented under defs) — so the shim is import-safe.
- **`jagged_tile` is already fully present on upstream.** Verified: `loops.py` upstream has `def jagged_tile`, `compile_environment.py` has `register_jagged_tile`, `type_info.py` has `JaggedTileIndexType`, `exc.py` has `InvalidJaggedTileUsage`, `language/__init__.py` exports `jagged_tile`. **Do NOT re-add these.** loops.py diffs: merge-base→reference = **109** changed lines (the NKI delta); merge-base→upstream = 230; **upstream→reference = 333** (the actual conflict surface). The draft's "421-line diff" figure was WRONG. The NKI-specific delta to apply is ~109 lines of NKI codegen behavior; the rest of the 333 is upstream refactor noise to absorb.
- **`set_codegen_state` / `_codegen_state` is genuinely ABSENT upstream** (grep count 0; reference count 7). Real, load-bearing work — add it (P1.4).
- **`config.py.old` DOES exist on the reference branch.** Do not port it (Do-Not-Port).
- **Backend abstract methods confirmed (7):** `name`(L77), `dtype_str`(L142), `acc_type`(L150), `function_decorator`(L603), `constexpr_type`(L620), `default_launcher_name`(L636), `library_imports`(L650). `NKIBackend` must implement all 7, plus override `pad_factory_tensors_to_power_of_2` and its NKI-specific `reduction_expr`.
- **Upstream backend resolution** is `self._backend = get_backend_class(settings.backend)()` (compile_environment.py L234), `backend_name` property at L1006. **No hardcoded `nki` dict entry exists or should be added.**
- **`BackendLiteral` is dynamically derived** from `list_backends()` (settings.py L363: `mapping={name: name for name in list_backends()}`). NKI appears automatically once registered (P1.5/P1.6). **Do NOT manually add `'nki'`.**
- **`import ast` is already at upstream backend.py L4** — no action.
- **There is no `helion/_compiler/memory_ops.py`** — only `helion/language/memory_ops.py`.
- **Blocked example set (from the guide's status table):** out of 51 runner-selected, **47 pass**; the 5 non-passing are `nvfp4_gemm`, `fused_linear_jsd`, `mamba2_chunk_scan`, `mamba2_chunk_state` (blocked), and `grpo_loss` (compile timeout). Treat all 5 as xfail; the 4-vs-5 ambiguity is resolved: 4 "blocked" + 1 "timeout" = 5 non-passing, 47 passing.

---

# PHASE 1 — Wiring + Codegen (no hardware; gate = byte-identical `to_triton_code`)

**Dependency spine.** Each must precede the next where the verify or runtime depends on it:
`P1.1 NKI-only files` → `P1.2 helper_function shim` → `P1.3 jagged/loops delta` → `P1.3b type_propagation NKI deltas` → `P1.4 compile_environment (set_codegen_state, guards)` → `P1.5 backend_registry hook` → `P1.6 nki_backend.py extraction (+pad_factory override, +reduction_expr)` → `P1.7 runtime/settings/config/output_header` → `P1.8 program_id (NKIProgramIDs only)` → `P1.9 device_function` → `P1.10 generate_ast` → `P1.11 device_ir` → `P1.12 reduction_strategy` → `P1.13 creation_ops` → `P1.14 aten_lowering` → `P1.15 inductor_lowering` → `P1.16 tile_strategy (a–i)` → `P1.17 memory_ops (the giant)` → `P1.18 language op codegens` → `P1.19 atomic_ops` → `P1.20 kernel.py` → `P1.21 _testing.py` → `P1.22 byte-for-byte gate`.

**Per-step mid-gates.** Every step below ends with an import/inspect/grep gate that fails *at that step* if the work is wrong — **no step relies on the final hardware sweep or the P1.22 byte-diff alone to detect failure.** Codegen-affecting steps (P1.10, P1.11, P1.12, P1.13, P1.14, P1.15, P1.16g, P1.17b/c, each P1.18 file, P1.19) ALSO run the targeted single-kernel byte-diff for the pattern they touch (see §Methodology) before their commit — regressions are localized at the introducing step, never deferred to P1.22.

Rationale for ordering: NKI-only data files and the `ast_for_fx_node` shim are pure prerequisites; the type deltas precede compile_environment; the registry + backend extraction must precede anything calling `get_backend_class('nki')`; `set_codegen_state` must precede every `with env.set_codegen_state(...)` site (device_ir, generate_ast, reduction, inductor_lowering, tile_strategy); `_nki_sbuf_shapes` is registered in `device_function` (P1.9) and **read** by `tile_strategy.codegen_device_loop` (P1.16g) — P1.9 strictly precedes P1.16; `memory_ops` (32× `ast_for_fx_node`) precedes `atomic_ops` (which calls `load._codegen['nki']`/`store._codegen['nki']`).

---

### P1.1 — Copy the three NKI-only support files
**Files:** `helion/_compiler/nki_fusion.py`, `helion/language/_nki_dim_access.py` (NEW; verbatim). *(Defer `autotuner/nki_search.py` to Phase 3.)*
**Action:** `git checkout fix-nki-kernel-compilation -- helion/_compiler/nki_fusion.py helion/language/_nki_dim_access.py`. **No manual edits** — use git checkout so not one whitespace differs. `_nki_dim_access.py` defines `IndirectAP`, `DynamicAP`, and the `DimAccess` hierarchy (Contiguous/Scalar/Indirect/Dynamic/FullSlice/StridedGather) used by `memory_ops.py` and `atomic_ops.py`. `nki_fusion.py` (passes `annotate_psum_reuse`, `annotate_tensor_scalar_reduce`, `annotate_activation_reduce`, `annotate_fx_graph`) lazy-imports `matmul_ops` inside a function body — safe from circular import.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.nki_fusion import annotate_fx_graph; from helion.language._nki_dim_access import DimAccess, IndirectAP, DynamicAP; print('ok')"
```
**Commit:** `port: add NKI-only support files (nki_fusion.py, _nki_dim_access.py)`

---

### P1.2 — Add `record_fx_node_ast`/`ast_for_fx_node` to `CodegenInterface`
**File:** `helion/_compiler/helper_function.py` (anchor: `class CodegenInterface(ABC)` upstream L31; keep upstream's `statement_owner_node` L42).
**Action:** Add the two methods verbatim from reference `helper_function.py` L49/L52. These are the dict-based recording hooks NKI load/store codegen calls **32 times** (all inside function bodies in memory_ops, verified import-safe). The three methods are orthogonal — keep all three.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.helper_function import CodegenInterface; assert hasattr(CodegenInterface,'record_fx_node_ast') and hasattr(CodegenInterface,'ast_for_fx_node') and hasattr(CodegenInterface,'statement_owner_node'); print('ok')"
```
**Commit:** `port: add record_fx_node_ast/ast_for_fx_node hooks to CodegenInterface`

---

### P1.3 — Reconcile the jagged_tile / loops / exc deltas (verify-first; ~109-line NKI delta)
**Files:** `helion/language/loops.py`, `helion/exc.py`, `helion/language/__init__.py`, `helion/language/tunable_ops.py`.

> **"Don't re-add jagged_tile" ≠ "don't make jagged_tile work on NKI."** We absolutely DO port the
> logic that makes jagged_tile generate correct NKI code. The distinction is purely mechanical: the
> jagged_tile *scaffolding* (the `jagged_tile` function, `JaggedTileIndexType`, `register_jagged_tile`,
> `InvalidJaggedTileUsage`, the export) was NKI-introduced on the old reference branch, but in the 829
> intervening commits **upstream independently added its own jagged_tile scaffolding**. So we must NOT
> copy the scaffolding (that would duplicate-define and conflict) — we graft the **NKI-behavior hunks**
> onto upstream's scaffolding. The NKI-behavior for jagged lives across THREE steps: this one (loops.py
> codegen/config deltas), P1.3b (type_propagation inference deltas), and P1.16g (tile_strategy
> `codegen_device_loop` dynamic-range block — the core jagged NKI codegen). Treat P1.3 as a 3-way diff:
> keep upstream's scaffolding, add only the NKI hunks, verify upstream's and the reference's jagged_tile
> are not semantically divergent in a way that breaks the NKI hunks.

**Action — guarded by inspection (jagged_tile SCAFFOLDING is ALREADY upstream; NKI BEHAVIOR is not):**
1. `git diff upstream/main fix-nki-kernel-compilation -- helion/language/loops.py` (333 lines; only ~109 are NKI-specific per merge-base→reference). Isolate the NKI-specific hunks (NKI codegen behavior, extra config choices) and apply only those. Update any NKI import that still reads `from .._compiler.type_propagation import …` to `from .._compiler.type_info import …` **only for symbols upstream relocated** (e.g. `JaggedTileIndexType`, `SymIntType`, `TileIndexType`); leave symbols still in `type_propagation` alone.
2. `exc.py`: `InvalidJaggedTileUsage` already exists upstream — no action unless the diff shows a message delta.
3. `language/__init__.py`: `jagged_tile` export already present — no action.
4. `tunable_ops.py`: ensure relocated type imports (`SymIntType` etc.) point at the file that actually defines them upstream (`type_info.py` for the moved ones); preserve NKI's `int → SymIntType` creation if the diff shows it missing.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.language import jagged_tile; from helion.exc import InvalidJaggedTileUsage; from helion._compiler.type_info import JaggedTileIndexType; print('ok')"
# residual mis-targeted imports of relocated symbols:
grep -rn 'from .._compiler.type_propagation import' helion/language/ || echo 'no residual type_propagation imports of moved symbols'
```
**Commit:** `port: reconcile NKI jagged_tile/loops delta against upstream`

---

### P1.3b — Port the three NKI deltas in `type_propagation.py` (NEW STEP; was omitted)
**File:** `helion/_compiler/type_propagation.py`. (Reviewers correct: these are NKI-specific and absent upstream.)
**Action — apply exactly these three hunks from reference; do NOT re-add `JaggedTileIndexType` (already upstream in type_info.py):**
1. **`TensorType.propagate_assignment` trailing-singletons** (ref L543–554). Replace the `if rhs_rank != 0 and lhs_rank != rhs_rank:` rank check with the NKI-guarded `allow_trailing_singletons` / `rhs_trailing_singletons` form. Guard: `CompileEnvironment.current().backend.name == "nki"`.
2. **`CallableType` size_hint capture** (ref L858–869). Replace `return SymIntType.new_unbacked(origin)` in the `_new_symint_on_host_fns()` host branch with the `tree_map_only(torch.SymInt, env.size_hint, …)` hint-computing form returning `SymIntType(origin, env.create_unbacked_symint(hint=hint))`. `contextlib` is already imported upstream (L5); confirm `tree_map_only` import.
3. **`patch_tensor_factories` guard — DO NOT EDIT here.** Upstream already guards this at L903–911 via `backend.pad_factory_tensors_to_power_of_2`. NKI participates by overriding that property to `False` in P1.6. (This replaces the reference's `device_loop_depth > 0 and backend_name != "nki"` form; the semantic is preserved via the backend property.)
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.type_propagation import TensorType, CallableType; import inspect; s=inspect.getsource(TensorType.propagate_assignment); assert 'trailing_singletons' in s; c=inspect.getsource(CallableType); assert 'create_unbacked_symint' in c; print('ok')"
```
**Commit:** `port: add NKI trailing-singleton + symint-hint deltas to type_propagation`

---

### P1.4 — `compile_environment.py`: `set_codegen_state` + NKI guards
**File:** `helion/_compiler/compile_environment.py` (anchors: `self._backend = get_backend_class(settings.backend)()` L234; `backend_name` property L1006).
**Action (registry path already correct — do NOT add a dict entry):**
1. Add `self._codegen_state: object | None = None` in `__init__` and a `set_codegen_state(state)` context manager (the keystone for statement-based NKI codegen; absent upstream, count 0). Must land before any `with env.set_codegen_state(...)` site (P1.10/P1.11/P1.12/P1.15/P1.16).
2. int64→int32 fake-tensor promotion in the fake-tensor construction path: `if self.backend_name == 'nki' and fake_dtype == torch.int64: fake_dtype = torch.int32`. Apply in **both** static and non-static-shape branches.
3. NKI reduction DMA bound check (skip power-of-2 expansion) in the reduction block-size source path: `if self.backend_name == 'nki': return max(1, block_size_info.size_hint())`.
4. shape_env `size_hint` fallback chain if the diff shows it absent.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.compile_environment import CompileEnvironment; assert hasattr(CompileEnvironment,'set_codegen_state'); print('ok')"
```
**Commit:** `port: add set_codegen_state + NKI int64/reduction guards to compile_environment`

---

### P1.5 — Register NKI via lazy hook in `backend_registry.py`
**File:** `helion/_compiler/backend_registry.py` (EXISTING file, 71 lines; anchor: after the `for _cls in _BUILTIN_BACKENDS: register_compiler_backend(_cls)` loop at L70–71 — the **last thing in the file**).
**Action:** Append (this MODIFIES the existing file; it is not a new file):
```python
def _maybe_register_nki() -> None:
    try:
        from . import nki_backend  # noqa: F401  (import registers NKIBackend)
    except ImportError:
        pass

_maybe_register_nki()
```
Lazy import inside the function preserves plugin semantics (no `torch_xla` at registry load). Do **not** modify `_BUILTIN_BACKENDS`. **Dev-time mitigation for masked non-ImportErrors** (SyntaxError/circular import in `nki_backend.py` would otherwise leave NKI silently absent): during development temporarily widen to `except Exception as e: import logging; logging.getLogger(__name__).warning("NKI register failed: %r", e)`; narrow back to `except ImportError` before the P1.6 commit.
**Verify (NKI must NOT yet appear — nki_backend.py doesn't exist; this must not break import):**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.backend_registry import list_backends, _maybe_register_nki; assert 'nki' not in list_backends(); _maybe_register_nki(); print('ok (lazy hook present, NKI absent until P1.6):', list_backends())"
```
**Commit:** `port: add lazy _maybe_register_nki hook to backend_registry`

---

### P1.6 — Create `nki_backend.py` by extracting NKIOpOverrides + NKIBackend
**File:** `helion/_compiler/nki_backend.py` (NEW — confirmed absent on both branches). Source = reference `backend.py` `class NKIOpOverrides` (L636) through end of `class NKIBackend(Backend)` (L5809→end of class).
**Action:**
1. Create the file. Copy `NKIOpOverrides` and `NKIBackend` **verbatim** (extract by line range from `git show fix-nki-kernel-compilation:helion/_compiler/backend.py`).
2. Imports: **must** include `from .backend import Backend` (NKIBackend subclasses it). Add `from .backend_registry import register_compiler_backend`. Add NKI deps (`compile_environment`, `ast_extension`, `program_id` for `NKIProgramIDs` once P1.8 lands, `_nki_dim_access`, etc.). Strip imports only used by the rest of the old `backend.py`. **Leave core `backend.py` untouched.**
3. At module end: `register_compiler_backend(NKIBackend)`.
4. Confirm all **7** abstract methods implemented: `name`(→`'nki'`), `dtype_str`(maps `torch.int64 → 'nl.int32'`), `acc_type`, `function_decorator`(→`'nki.jit'`), `constexpr_type`, `default_launcher_name`(→`'_default_nki_launcher'`), `library_imports`.
5. **Override `pad_factory_tensors_to_power_of_2` → `return False`** (load-bearing; replaces the reference's host_function/type_propagation guard mechanism — see §Corrections).
6. Confirm `NKIBackend.reduction_expr` carries the reference's `fake_input`/`fake_output` keyword params (it does in the verbatim copy); this is the override the P1.12 callers target.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.backend import Backend; print('core backend import ok')"
PYTHONPATH=/home/ubuntu/helion_port python -c "import helion; from helion._compiler.backend_registry import list_backends, get_backend_class; assert 'nki' in list_backends(); b=get_backend_class('nki')(); assert b.name=='nki' and b.function_decorator=='nki.jit' and b.default_launcher_name=='_default_nki_launcher' and b.pad_factory_tensors_to_power_of_2 is False; import torch; assert b.dtype_str(torch.int64)=='nl.int32'; print('ok')"
PYTHONPATH=/home/ubuntu/helion_port python -c "import helion; print(helion.Config(backend='nki'))"
```
**Commit:** `port: extract NKIOpOverrides+NKIBackend into nki_backend.py and register`

---

### P1.7 — runtime launcher, settings, config, output_header
**Files:** `helion/runtime/__init__.py`, `helion/runtime/settings.py`, `helion/runtime/config.py`, `helion/_compiler/output_header.py`.
**Action:**
1. `runtime/__init__.py`: insert `default_nki_launcher` between `def default_launcher` and the next launcher helper (search symbols; don't trust line numbers in this large file). **Verify `torch_xla` import is inside the function body**, not module scope. Keeps `xm.mark_step()`, int64→int32 index casting, LNC auto-bump for `dynamic_range` kernels, `HELION_NKI_PROFILE`.
2. `settings.py`: add `get_neuron_target()` (4-tier: config → `HELION_NEURON_TARGET` → `neuron-ls` autodetect → helpful error), and `platform_target: str | None = None` field + FIELD_DOCS entry. **Do NOT add `'nki'` to `BackendLiteral`** — it derives from `list_backends()` (L363) and appears automatically.
3. `config.py`: add `platform_target: str | None = None` param + `self.platform_target = …` + docstring. (Note: the reference `Config` lacks this field — keep byte-diff configs free of it; see §0 determinism.)
4. `output_header.py`: add `"_default_nki_launcher",` to the `disallowed_names` dict immediately after `"_default_cute_launcher",` (L37).
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.runtime import default_nki_launcher; import helion; print(helion.Config(platform_target='trn2').platform_target)"
PYTHONPATH=/home/ubuntu/helion_port bash -c "HELION_NEURON_TARGET=trn2 python -c 'from helion.runtime.settings import get_neuron_target; print(get_neuron_target())'"
grep -n _default_nki_launcher helion/_compiler/output_header.py
PYTHONPATH=/home/ubuntu/helion_port python -c "import torch_xla" 2>&1 | grep -q . && echo "torch_xla present" || echo "torch_xla absent — verifying helion still imports"; PYTHONPATH=/home/ubuntu/helion_port python -c "import helion; print('helion imports without torch_xla at module scope')"
```
**Commit:** `port: add default_nki_launcher, get_neuron_target, platform_target, output_header guard`

---

### P1.8 — `program_id.py` (NKIProgramIDs only — host_function guard REMOVED)
**File:** `helion/_compiler/program_id.py`. **(host_function.py edit dropped — see §Corrections; upstream no longer calls `patch_tensor_factories` there, and NKI's guard is the `pad_factory_tensors_to_power_of_2=False` override from P1.6.)**
**Action:** Append `class NKIProgramIDs(ProgramIDs)` (codegen → pass; codegen_grid → `'(1,)'`) after `CuteProgramIDs`. Orthogonal to upstream's CuTe refactor.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.program_id import NKIProgramIDs; print('ok')"
# confirm the host_function patch path is gone upstream (no NKI edit needed):
grep -n 'patch_tensor_factories' helion/_compiler/host_function.py || echo 'host_function has no patch_tensor_factories — correct, no NKI edit needed'
```
**Commit:** `port: add NKIProgramIDs to program_id`

---

### P1.9 — `device_function.py`: NKI tracking dicts, methods, printer
**File:** `helion/_compiler/device_function.py` (anchors: `self.dce_vars: list[str] = []` in `__init__`; `codegen_function_def` body around `sorted_arguments = self.sorted_args()` / `fn_def = create(...)`; `class HelionTritonPrinter`).
**Action (re-anchored onto upstream's ScratchArg/Pallas/CuTe expansions):**
1. `__init__`: add the 12 NKI tracking dicts (`_nki_sbuf_shapes`, `_nki_sbuf_dtypes`, `_nki_logical_shapes`, `_nki_iota_offsets`, `_nki_iota_block_sizes`, `_nki_tile_lists`, `_nki_scalar_arg_names`, `_nki_hbm_sources`, `_nki_arg_dtypes_override`, `_nki_iota_source_shapes`, `_nki_sbuf_alloc_exprs`, `_nki_sbuf_alloc_depths`) after `self.dce_vars`. **`_nki_sbuf_shapes` is read by P1.16g — must exist before tile_strategy runs.**
2. Append three methods before `codegen_function_def`: `_register_nki_dynamic_tensor_size_args` (~130 lines), `_should_rewrite_nki_sbuf_reassign` (~20), `_rewrite_nki_sbuf_reassignments` (~86).
3. In `codegen_function_def`: NKI dtype-normalization guard near argument processing; call `_register_nki_dynamic_tensor_size_args()` early; call `_rewrite_nki_sbuf_reassignments(fn_def.body, {}, 0)` after `fn_def` is built, guarded `if backend.name == 'nki'`.
4. Append `HelionNKIPrinter` after `HelionTritonPrinter`. **Watch-out:** if upstream routes through `backend.sympy_printer_expr()` rather than `texpr()`, the NKI-printer selection belongs in `NKIBackend.sympy_printer_expr` (P1.6), not a `texpr` guard. Inspect the diff and place accordingly.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.device_function import DeviceFunction; d=DeviceFunction.__dict__; assert all(m in d for m in ['_register_nki_dynamic_tensor_size_args','_should_rewrite_nki_sbuf_reassign','_rewrite_nki_sbuf_reassignments']); print('ok')"
PYTHONPATH=/home/ubuntu/helion_port python -c "import ast,inspect; from helion._compiler import device_function; ast.parse(inspect.getsource(device_function)); print('parses')"
```
**Commit:** `port: add NKI SBUF tracking dicts/methods and HelionNKIPrinter to device_function`

---

### P1.10 — `generate_ast.py`: NKI init dicts, methods, pre-codegen passes (~386-line diff)
**File:** `helion/_compiler/generate_ast.py` (anchors: `NodeVisitor.__init__(self)`; the `codegen_call_with_graph` invocation in the statement-building loop; `GraphInterpreter.run_node`; `codegen_call_with_graph`).
**Action (the guide's "2 guards" is wrong — verified ~386 changed lines):**
1. `__init__`: add `self._var_to_constant`, `self._nki_sbuf_constant_values`, `self._nki_sbuf_alloc_depth`, `self.fx_node_to_ast` alongside upstream's CuTe tracking vars.
2. Append four NKI methods: `_lower_nki_mod_assign` (~156, with its `codegen_name != 'nki'` guard), `_record_nki_sbuf_allocation` (~30), `_record_nki_sbuf_write` (~40, with `codegen_name != 'nki'` guard), `_constant_value_from_ast` (~80).
3. Just before the codegen call: `if env.backend.name == 'nki': env.backend.validate_nki_tensor_shapes(root); from .nki_fusion import annotate_fx_graph; annotate_fx_graph(root)` (read-only FX annotation; must not mutate graph structure).
4. Wrap the codegen call with `with env.set_codegen_state(state):`.
5. `GraphInterpreter.run_node` + `codegen_call_with_graph`: add `record_fx_node_ast` recordings (2 sites) and the `get_tile_list_vars`/`register_tile_list` phi-node handling. **Verify the actual reference location with `git diff` — these live where `GraphInterpreter`/`codegen_call_with_graph` are defined; place per the diff, do not double-add.** (Resolves the inductor_lowering-analyst mislocation.)
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.generate_ast import GenerateAST; assert hasattr(GenerateAST,'_lower_nki_mod_assign') and hasattr(GenerateAST,'_record_nki_sbuf_allocation'); print('ok')"
```
+ **targeted byte-diff** on `add.py` (DMA copy) per §Methodology — must be IDENTICAL before commit.
**Commit:** `port: add NKI tracking/methods/pre-codegen passes to generate_ast`

---

### P1.11 — `device_ir.py`: SiLU decomp guard + set_codegen_state wraps
**File:** `helion/_compiler/device_ir.py` (anchors: `_get_custom_decomp_table` + its `decomp_table.pop(torch.ops.aten.stack.default, None)`; per-type `def codegen(self, state: CodegenState)` on `ForLoopGraphInfo`/`IfGraphInfo`/`WhileLoopGraphInfo`/`HelperFunctionGraphInfo`).
**Action:**
1. In `_get_custom_decomp_table`, after the stack pop: `if CompileEnvironment.current().backend_name == 'nki': decomp_table.pop(torch.ops.aten.silu.default, None)` so NKIOpOverrides handles SiLU first-class.
2. Wrap each control-flow graph `codegen` with `env = CompileEnvironment.current()` + `with env.set_codegen_state(state), …`. **WhileLoopGraphInfo has TWO paths (condition + body)** — wrap both.
3. `JaggedTileIndexType` import + jagged handling already upstream — no action unless diff shows an NKI delta.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.device_ir import _get_custom_decomp_table; import ast,inspect; from helion._compiler import device_ir; ast.parse(inspect.getsource(device_ir)); print('ok')"
```
+ targeted byte-diff on a nested-loop / while kernel if one exists in the representative set; otherwise covered at P1.16g/P1.22.
**Commit:** `port: add NKI SiLU decomp guard and set_codegen_state wraps to device_ir`

---

### P1.12 — `reduction_strategy.py` (guide-omitted; 232 NKI lines)
**File:** `helion/_compiler/reduction_strategy.py` (re-anchor onto upstream's refactored hierarchy: `CachedReductionState`, `LoopedReductionStrategy`, imports `DeviceGridState/LoopDimInfo/ThreadAxisTracker/_to_sympy`).
**Action:** Accept upstream's refactor first, then re-anchor NKI guards:
- `fake_input`/`fake_output` params threaded into `call_reduction_function` and passed to `backend.reduction_expr(..., fake_input=fake_input, fake_output=fake_output)` (ref callers at L108/118/131/136-137). **Reconcile with upstream's new `threads_in_group`/`thread_linear_index_expr`/`reduction_threads_hint` machinery** — for NKI those return their NKI values; do not let the upstream thread-group keywords collide with NKI's fake_input/output keywords (they are distinct kwargs).
- `full_memset_stmt` accumulator init via `getattr(backend,'full_memset_stmt')`.
- `_nki_sbuf_shapes` registration for multi-user copy-var detection.
- deferred post-loop reduction with `_NKI_REDUCTION_OPS` (sum→nl.add, max→nl.maximum, …) and dynamic SBUF shape resolution.
All guarded `backend.name == 'nki'`.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "import helion._compiler.reduction_strategy; import ast,inspect; from helion._compiler import reduction_strategy; ast.parse(inspect.getsource(reduction_strategy)); print('ok')"
```
+ **targeted byte-diff** on `softmax.py` and `rms_norm.py` per §Methodology before commit.
**Commit:** `port: re-anchor NKI reduction codegen onto upstream reduction_strategy`

---

### P1.13 — `creation_ops.py` (guide-omitted; ~150 NKI lines)
**File:** `helion/language/creation_ops.py` (anchor: `_full_codegen`; keep upstream's `@codegen(full,'pallas')` variant).
**Action:** In `_full_codegen` add `nki_memset = getattr(backend,'full_memset_stmt')`; when `backend.name == 'nki'` emit the two-line ndarray+memset pattern; apply the `[1,N]→[N,1]` accumulator-transpose heuristic (scans device_ir graphs for reduction ops) and `_nki_sbuf_shapes` registration. Orthogonal to upstream's Pallas variant.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.language import full; print('ok')"
```
+ covered by the reduction byte-diffs (softmax/rms_norm exercise full+memset) before P1.14.
**Commit:** `port: add NKI full() memset/transpose codegen to creation_ops`

---

### P1.14 — `aten_lowering.py` (15 NKI handlers + 3 NKI-only lowering objects)
**File:** `helion/_compiler/aten_lowering.py`.
**Action:**
1. Add upstream's new imports (cute utils, `matmul_utils`, `strategies`, `contextlib`) and three helpers (`_requested_pure_matmul_role_lifecycle`, `_requested_tcgen05_flat_role_coordinates`, `_reject_tcgen05_flat_role_coordinates_fallback`).
2. Add upstream's new lowering objects **first** so they don't get shadowed: `scalar_tensor_lowering`, `where_lowering`, `argmax_lowering`, `argmin_lowering`, `arange_default_lowering`.
3. **Create the three NKI-only lowering objects** (`stack_lowering`, `silu_lowering`, `cumsum_lowering`) via `register_lowering(torch.ops.aten.<op>)` — upstream lacks them. **Create the object BEFORE applying its `@<obj>.register_codegen('nki')` decorator.** If upstream's lowering-registration architecture changed, match the current `register_lowering` signature.
4. Append the **15** `@<obj>.register_codegen('nki')` handlers verbatim: full, unsqueeze, squeeze, view, reshape (reshape decorates `codegen_view_nki` — intentional name mismatch), permute, stack, expand, silu, mm, addmm, bmm, baddbmm, cumsum, iota.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler import aten_lowering as a; assert all(hasattr(a,x) for x in ['full_lowering','stack_lowering','silu_lowering','cumsum_lowering']); print('ok')"
```
+ **targeted byte-diff** on `matmul.py` (mm/addmm) before commit.
**Commit:** `port: add upstream lowering objects + 15 NKI aten codegen handlers`

---

### P1.15 — `inductor_lowering.py` (NKI guards + reduction wrap + dtype + mod/remainder/fmod)
**File:** `helion/_compiler/inductor_lowering.py` (moderate re-anchor onto upstream's cute-argreduce/ctx refactor).
**Action — `_check_block_broadcast_compatibility` ALREADY has `ctx` upstream (L536); do NOT change its signature:**
1. Replace `INDUCTOR_PATCH` dict with `_patched_inductor_config()` context manager (adds `fast_math`) at all 3 call sites.
2. `PointwiseLowering`: ensure the NKI codegen path passes `ctx` to `_check_block_broadcast_compatibility` correctly and merges any NKI-specific `block_sizes_proven_equal` body — **no signature edit**; the `ctx` param is upstream's existing state.
3. `ReductionLowering.codegen`: merge upstream's `match_active_block_id` cute-argreduce block, then wrap **only** the single `strategy.codegen_reduction(...)` call with `with env.set_codegen_state(state):`. Confirm no early `return` between the wrap and the call moved it out of scope (read the upstream refactor first).
4. `ReductionLowering.get_masked_value`: add `'mean'` to `{'sum','prod','min','max'}` → `{'sum','prod','min','max','mean'}` (genuinely absent upstream at L949; needed for jagged mean masking — see §Corrections). **First decide guarded vs unconditional:** the reference added it unconditionally, which also affects Triton/CuTe and is NOT caught by the NKI-only byte-diff gate. Either confirm `mean` is genuinely zero-preserving (safe unconditional) or guard with `if backend.codegen_name == "nki"`. Verify the reference's intent against the matching NKI byte-diff (a mean/softmax-style kernel) before committing.
5. `GenerateASTFromInductor._default`: add NKI activation validation (`_NKI_ACTIVATION_NAMES`, raise `exc.BackendUnsupported`) **after** upstream's Metal namespace handling.
6. `to_dtype`: add NKI `cast_ast(..., src_dtype=src_dtype)` branch after the cute fp8 case.
7. Add 3 NKI-only methods `mod`/`remainder`/`fmod` (string-arg → `_default`).
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.inductor_lowering import _patched_inductor_config, PointwiseLowering, ReductionLowering; import inspect; assert 'ctx' in str(inspect.signature(PointwiseLowering._check_block_broadcast_compatibility)); s=inspect.getsource(ReductionLowering.get_masked_value); assert 'mean' in s; print('ok')"
```
+ targeted byte-diff on `softmax.py` (reduction wrap) + a pointwise mod kernel if present.
**Commit:** `port: re-anchor NKI guards in inductor_lowering (patch ctx, reduction wrap, mean, dtype, mod/remainder/fmod)`

---

### P1.16 — `tile_strategy.py` grid/device-loop codegen (~945-line diff; hard-semantic, multi-commit)
**File:** `helion/_compiler/tile_strategy.py`. Largest conflict surface alongside memory_ops. Split into the sub-commits below, each its own commit, **each with its own grep/parse gate**. Depends on `NKIProgramIDs` (P1.8), `set_codegen_state` (P1.4), and `_nki_sbuf_shapes` registration (P1.9 — must already exist).
- **P1.16a** — merge upstream imports + add NKI helpers (`_nki_body_leading_count`, `_count_leading_block_id_matches`, `_backend_loop_index_statements`, `_backend_grid_index_statements`) after upstream's lane-reduce helpers. *Gate:* `python -c "import helion._compiler.tile_strategy"`.
- **P1.16b** — `DeviceGridState`: keep `lane_loops: list[tuple[str, ...]]` (NKI flexible form), add upstream fields (`lane_loop_blocks`, `outer_prefix`, `outer_suffix`, `add_lane_loop`); `wrap_body()` branches on tuple length (2-tuple `_create_lane_loop`; 3-tuple NKI inline body_prefix). Convert `PersistentReductionState` to upstream's `@dataclass`. Add `EmitPipelineLoopState`/`ForiLoopState`. *Gate (verify class shape exists upstream before relying on it):* `python -c "from helion._compiler.tile_strategy import DeviceGridState, EmitPipelineLoopState, ForiLoopState; print('ok')"`.
- **P1.16c** — `TileStrategy` base: upstream `offset_prefix` CuTe-collision fix, `thread_block_size_exprs()`, `get_range_call_str()` `in`-operator.
- **P1.16d** — `BlockSizeTileStrategy` + `FlattenedTileStrategy` upstream methods.
- **P1.16e** — `_BaseNDTileStrategy` upstream methods + `codegen_grid()` refactor with the **early NKI branch** `if env.backend.name == 'nki': return self._codegen_grid_nki(...)`. *Gate (must pass before P1.16f):* `python -c "import inspect; from helion._compiler.tile_strategy import _BaseNDTileStrategy as B; assert \"backend.name == 'nki'\" in inspect.getsource(B.codegen_grid)"`.
- **P1.16f** — add the `_codegen_grid_nki` method (~270 lines). *Gate:* `grep -c '_codegen_grid_nki' helion/_compiler/tile_strategy.py` → expect ≥2 (def + call).
- **P1.16g** — `codegen_device_loop()` with NKI's 600+-line block kept **first** (jagged demotion, `_nki_dyn_loops`, register/counter setup; reads `_nki_sbuf_shapes` from P1.9), then upstream body; `mask_statement` handled as both single and `list`. *Gate:* parse + **targeted byte-diff on the jagged example** before commit.
- **P1.16h** — `_setup_block_size_constexpr` gains `block_idx: int | None = None`; NKI caches literal to `block_size_var_cache`; **update all call sites incl. refactored CuteNDTileStrategy**. *Gate:* `grep -c '_setup_block_size_constexpr' helion/_compiler/tile_strategy.py` and inspect each call passes/omits `block_idx` consistently.
- **P1.16i** — `NDTileStrategy.mask_var()` → `.get(block_idx)`; `CuteNDTileStrategy` renames + new methods.
**Commits:** one per sub-step, e.g. `port(tile_strategy): add NKI helpers + upstream lane-reduce infra`, …, `port(tile_strategy): codegen_device_loop NKI dynamic-range block`, etc.

---

### P1.17 — `memory_ops.py` NKI load/store (the giant; verbatim, multi-commit)
**File:** `helion/language/memory_ops.py`. Append **after upstream's last codegen (end of CuTe load codegen)**, in order. Depends on P1.1 (`_nki_dim_access`) and P1.2 (`ast_for_fx_node`, called 32×, all inside function bodies — verified import-safe).
- **P1.17a** — six helpers verbatim: `_nki_shifted_tile_subscript`, `_nki_indirect_gather`, `_nki_lookup_sbuf_shape_dtype`, `_nki_as_uint32_p1_vector`, `_nki_row_index_gather`, `_nki_subscript_block_id`.
- **P1.17b** — `@_decorators.codegen(load, 'nki')` verbatim.
- **P1.17c** — `@_decorators.codegen(store, 'nki')` verbatim.
**Do not refactor.** 20+ subscript patterns; any deviation is a silent numeric regression.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.language.memory_ops import _nki_shifted_tile_subscript,_nki_indirect_gather,_nki_lookup_sbuf_shape_dtype,_nki_as_uint32_p1_vector,_nki_row_index_gather,_nki_subscript_block_id; from helion.language.memory_ops import load, store; assert 'nki' in load._codegen and 'nki' in store._codegen; print('ok')"
```
+ **targeted byte-diff** on `add.py` (DMA copy) after P1.17c, and on the gather/`embedding.py` example.
**Commits:** `port(memory_ops): NKI helpers`, `port(memory_ops): NKI load codegen (verbatim)`, `port(memory_ops): NKI store codegen (verbatim)`.

---

### P1.18 — language op codegens (matmul/scan/_mask_to/rand/barrier/subscript)
**Files:** `matmul_ops.py`, `scan_ops.py`, `_tracing_ops.py`, `random_ops.py`, `barrier.py`, `view_ops.py`. Append each `@_decorators.codegen(<op>, 'nki')` after the last existing codegen for that op. **Verbatim.**
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.language.matmul_ops import dot; from helion.language.scan_ops import _associative_scan; from helion.language._tracing_ops import _mask_to; from helion.language.random_ops import rand; from helion.language.barrier import barrier; from helion.language.view_ops import subscript; print('ok')"
```
+ targeted byte-diff on `matmul.py` after `matmul_ops` lands.
**Commits:** one per file, e.g. `port: add NKI dot codegen to matmul_ops`, ….

---

### P1.19 — `atomic_ops.py` (depends on P1.17 — memory_ops load/store registered)
**File:** `helion/language/atomic_ops.py`.
**Action:** Add `from ._nki_dim_access import IndirectAP` (keep upstream's `_symint_expr`/`GridOrigin` imports). Append `@_decorators.codegen(atomic_add, 'nki')` verbatim. The handler makes **two direct calls** `load._codegen['nki'](load_state)` and `store._codegen['nki'](store_state)` — these `KeyError` if P1.17 isn't done. This is why P1.17 strictly precedes P1.19.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.language.atomic_ops import atomic_add; from helion.language.memory_ops import load, store; assert 'nki' in load._codegen and 'nki' in store._codegen; print('ok')"
```
+ targeted byte-diff on an `atomic_add` example (scatter/histogram).
**Commit:** `port: add NKI atomic_add codegen + IndirectAP import to atomic_ops`

---

### P1.20 — `runtime/kernel.py` NKI profiling + config completion
**File:** `helion/runtime/kernel.py` (re-anchor onto upstream's `KernelCompiler`/specialization-key/dist-utils refactor).
**Action:** Add `_complete_nki_partial_config()` and `_clone_config()` to the bound-kernel class; call `_clone_config` + `_complete_nki_partial_config` at the new normalization point in `to_triton_code()` and `compile()`; add `HELION_NKI_PROFILE` stderr timing.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port bash -c "HELION_NKI_PROFILE=1 python -c 'import helion; print(\"ok\")'"
```
**Commit:** `port: add NKI config completion + HELION_NKI_PROFILE to kernel.py`

---

### P1.21 — `_testing.py` NKI harness adjustments
**File:** `helion/_testing.py` (re-anchor onto upstream's refactored `run_example`).
**Action (order matters):**
1. DEVICE detection — add `if _get_backend() == 'nki': DEVICE = torch.device('cpu')` as the **first** check (before pallas, before `HALF_DTYPE`/`LONG_INT_TYPE` use it).
2. In `run_example`: NKI baseline-dict filter (select pytorch/torch only); tolerance override (rtol 5e-2 / atol 1.5, both fwd and bwd `assert_close`); benchmark skip (`if _get_backend()=='nki': print(skip, file=sys.stderr); return`).
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port bash -c "HELION_BACKEND=nki python -c 'from helion._testing import DEVICE; print(DEVICE)'"  # expect cpu
```
**Commit:** `port: add NKI DEVICE/baseline/tolerance/benchmark handling to _testing`

---

### P1.22 — **GATE: byte-for-byte codegen verification** (see §Methodology)
**Two sub-gates; do not advance from a to b until a is fully green.**
- **P1.22a** — the 6 representative kernels (one per pattern) must diff-empty.
- **P1.22b** — the full 47 passing examples must diff-empty (the 5 blocked examples are expected to error/xfail, not diff).
No commit until all required diffs are empty. (Most regressions are already caught by the per-step targeted diffs above; P1.22 is the comprehensive backstop, not the first line of defense.)
**Commit (only after green):** `port: verify byte-for-byte NKI codegen vs reference for 47 examples`

---

# PHASE 2 — Example sweep on hardware (Trainium)

Prereq each run: clear cache (`rm -rf /tmp/helion_nki_portv2_* /var/tmp/neuron-compile-cache/`) and `export NEURON_CC_FLAGS='--no_cache'`.

### P2.1 — Single-kernel on-device smoke
Run one pointwise + one matmul kernel end-to-end with `HELION_BACKEND=nki`; confirm output matches torch reference within NKI tolerances.
**Commit:** `test(nki): on-device smoke for pointwise+matmul`

### P2.2 — Full 47-example sweep
Run the full example suite `HELION_BACKEND=nki`. Expect **47 pass; the 5 blocked (`nvfp4_gemm`, `fused_linear_jsd`, `mamba2_chunk_scan`, `mamba2_chunk_state`, `grpo_loss`) xfail**. Triage any *new* failure back to the Phase-1 step that introduced it; fix and re-run with cleared cache.
**Commit:** `test(nki): full example sweep — 47 pass, blocked-set xfail`

---

# PHASE 3 — Autotuning

### P3.1 — Copy `autotuner/nki_search.py` and register
**Files:** `helion/autotuner/nki_search.py` (NEW, verbatim), `helion/autotuner/__init__.py`.
**Action:** `git checkout fix-nki-kernel-compilation -- helion/autotuner/nki_search.py`. Add two imports after `FiniteSearch`: `from .nki_search import NKIFiniteSearch as NKIFiniteSearch` and `from .nki_search import _generate_nki_configs as _generate_nki_configs`. **Verify no module-scope `torch_xla` import** in nki_search.py first (`git show fix-nki-kernel-compilation:helion/autotuner/nki_search.py | grep -n 'import torch_xla\|import_xla\|xm\.' | head`); if any is at module scope it must move into a function before this can be imported on a non-Trainium box.
**Verify:** `PYTHONPATH=/home/ubuntu/helion_port python -c "from helion.autotuner import NKIFiniteSearch; print('ok')"`
**Commit:** `port: add NKIFiniteSearch autotuner (Phase 3)`

### P3.2 — NKI benchmarking path
Replace the Phase-1 `_testing.py` benchmark skip with real NKI timing (`xm.mark_step()` + wall-clock), per the autotune memory notes.
**Commit:** `port(nki): enable NKI benchmarking with xm.mark_step timing`

---

## Guide-omitted-file steps (explicit)

These files are touched by the reference but absent/under-specified in the guide's file list; each is a real port step above:
- `helion/_compiler/type_propagation.py` — NKI trailing-singleton + symint-hint deltas (**P1.3b**, entirely omitted by the guide and the draft).
- `helion/_compiler/reduction_strategy.py` — 232 NKI lines (**P1.12**).
- `helion/language/creation_ops.py` — ~150 NKI lines (**P1.13**).
- `helion/_compiler/aten_lowering.py` — 3 NKI-only lowering objects + 15 handlers (**P1.14**).
- `helion/_compiler/inductor_lowering.py` — reduction wrap, `'mean'`, dtype, mod/remainder/fmod (**P1.15**).
- `helion/_compiler/program_id.py` — `NKIProgramIDs` (**P1.8**).
- `helion/_compiler/helper_function.py` — `record_fx_node_ast`/`ast_for_fx_node` (**P1.2**).
- `helion/language/{scan_ops,_tracing_ops,random_ops,barrier,view_ops}.py` — NKI codegens (**P1.18**).
- `helion/runtime/kernel.py` — config completion + profiling (**P1.20**).
**`helion/_compiler/host_function.py` is NOT a port step** — upstream removed the `patch_tensor_factories` call entirely; NKI's guard is the `pad_factory_tensors_to_power_of_2=False` override in P1.6 (the guide and draft both got this wrong).

---

## Byte-for-byte verification methodology

Codegen is pure Python — no hardware needed. For each kernel, generate `to_triton_code(config)` on **both** branches with the **same Config** (constructed WITHOUT `platform_target`, so reference and port configs are field-compatible) and `PYTHONHASHSEED=0`, then diff.

**Step 0 — reference-against-itself sanity check (run ONCE before trusting any diff).** Proves codegen is deterministic; if this fails, codegen has nondeterminism (dict/set iteration, unseeded RNG) and full-string diffing is invalid — fall back to structured assertions.
```bash
export PYTHONHASHSEED=0
git worktree add /tmp/ref_wt fix-nki-kernel-compilation
PYTHONPATH=/tmp/ref_wt HELION_BACKEND=nki python /tmp/gen_one.py examples/add.py > /tmp/r1.txt 2>&1
PYTHONPATH=/tmp/ref_wt HELION_BACKEND=nki python /tmp/gen_one.py examples/add.py > /tmp/r2.txt 2>&1
diff /tmp/r1.txt /tmp/r2.txt && echo "REFERENCE DETERMINISTIC" || echo "BLOCKER: nondeterministic codegen — use structured assertions"
```

**Step 1 — port vs reference per kernel.**
```bash
gen() {  # $1=worktree path, $2=example, $3=out file
  PYTHONPATH=$1 PYTHONHASHSEED=0 HELION_BACKEND=nki python /tmp/gen_one.py "$2" > "$3" 2>&1
}
# /tmp/gen_one.py: import example; build identical Config (no platform_target); torch.manual_seed(0); print(bound.to_triton_code(config))
gen /tmp/ref_wt <kernel> /tmp/ref.txt
gen /home/ubuntu/helion_port <kernel> /tmp/port.txt
diff -u /tmp/ref.txt /tmp/port.txt && echo "IDENTICAL"
```
If full-string diffing proves unstable, the fallback per kernel is a set of structured assertions (`@nki.jit` present, `nisa.dma_copy`/`nisa.nc_matmul` present as appropriate, function arg list identical, no `texpr` leakage).

**Representative kernels — one per codegen pattern** (the P1.22a set; each also used as the *targeted* diff at its introducing step):
- **pointwise / DMA copy** — `examples/add.py` → `memory_ops` load/store, `default_nki_launcher`, `generate_ast` (P1.10/P1.17).
- **matmul** — `examples/matmul.py` → `matmul_ops` `dot` (`nisa.nc_matmul`, PSUM reuse), `aten_lowering` mm/addmm (P1.14/P1.18).
- **reduction** — `examples/softmax.py` + `examples/rms_norm.py`/layernorm → `reduction_strategy`, `creation_ops` full+memset, `inductor_lowering` reduction wrap + `'mean'` (P1.12/P1.13/P1.15).
- **gather / indirect** — `examples/embedding.py` → `_nki_indirect_gather`/`_nki_row_index_gather`, `IndirectAP` (P1.17).
- **dynamic-loop / jagged** — the `jagged_*` example → `_codegen_grid_nki`, `codegen_device_loop` dynamic-range block, jagged type deltas (P1.16g/P1.3b).
- **atomic** — an `atomic_add` example (scatter/histogram) → `atomic_ops` + its `load._codegen['nki']`/`store._codegen['nki']` calls (P1.19).

After P1.22a passes, run the **full 47** (P1.22b). A non-empty diff localizes the regression to the codegen file owning that pattern. Snapshot expected outputs into `test/test_nki_backend.py` with `assertExpectedJournal` (`EXPECTTEST_ACCEPT=1` to seed) for regression detection. Clear the Neuron cache before any on-device confirmation; codegen diffs are cache-immune.

---

## Risk register (with mitigations)

| # | Risk | Source | Mitigation |
|---|------|--------|------------|
| R1 | `ast_for_fx_node` absent upstream → `AttributeError`/silent `None` ("cannot unparse None") on first of 32 calls. | memory_ops | P1.2 adds both methods to `CodegenInterface` in **helper_function.py** (verified home). P1.10 wires `record_fx_node_ast` at the same pipeline points as reference. |
| R2 | `set_codegen_state` nested at wrong level in control-flow codegens; WhileLoop has 2 paths. | core-glue | P1.4 adds the mechanism first; P1.11 wraps each `codegen` incl. both while paths; nested-loop kernel diffs at P1.16g/P1.22. |
| R3 | Lane-loop 3-tuple silently dropped by `wrap_body` → index-setup lost. | tile_strategy | P1.16b `wrap_body` branches on tuple length; jagged kernel targeted-diff at P1.16g. |
| R4 | Jagged demotion fails if outer `for_node` ref goes stale in upstream's refactored emission. | tile_strategy | NKI's 600-line block kept **first** in `codegen_device_loop` (P1.16g); jagged byte-diff before commit. |
| R5 | `_setup_block_size_constexpr` `block_idx` not threaded to every call site (incl. CuteNDTileStrategy). | tile_strategy | P1.16h updates all call sites; grep-count gate. |
| R6 | NKI-only lowering objects (stack/silu/cumsum) created wrong/after decorator → silent fallback. | aten_lowering | P1.14 creates each via `register_lowering` BEFORE its `register_codegen('nki')`; `hasattr` gate + matmul/silu diffs. |
| R7 | **WRONG in draft:** `_check_block_broadcast_compatibility` `ctx` already upstream — adding it would break. | inductor_lowering | P1.15 does NOT add `ctx`; only verifies NKI path passes it. `inspect.signature` gate confirms `ctx` present (upstream's). |
| R8 | `ReductionLowering` `set_codegen_state` wrap at wrong scope inside `match_active_block_id`. | inductor_lowering | P1.15 wraps only the single `strategy.codegen_reduction(...)` call after reading the upstream refactor; softmax diff. |
| R9 | `torch_xla` imported at module scope anywhere → import fails on dev box; breaks Triton too. | plugin constraint | P1.5 lazy hook; P1.7 launcher imports xla inside the function; P3.1 verifies nki_search has no module-scope xla. Gate: `import helion` works with no torch_xla. |
| R10 | `_maybe_register_nki` masks a non-ImportError (syntax/circular) → NKI silently absent. | registry | P1.5 dev-time widen-to-log; P1.6 gate asserts `'nki' in list_backends()` AND `helion.Config(backend='nki')` constructs. |
| R11 | int64→int32 promotion applied to only one fake-tensor branch → mixed int32/int64. | core-glue | P1.4 patches both static and non-static branches; launcher (P1.7) also casts indices; `dtype_str` gate (P1.6). |
| R12 | HelionNKIPrinter routed through `texpr` when upstream uses `backend.sympy_printer_expr()`. | core-glue | P1.9 inspects upstream injection point; if `sympy_printer_expr` exists, override on `NKIBackend` (P1.6). |
| R13 | **type_propagation NKI deltas omitted** (trailing-singleton, symint hint) → jagged type-inference reverts to wrong static bounds (silent). | guide+draft omission | **NEW P1.3b** ports both, with `inspect.getsource` gates; jagged example diffs at P1.16g/P1.22. |
| R14 | **patch_tensor_factories ported as a host_function edit** when upstream moved it to the backend property → double-mechanism / no-op. | guide+draft error | P1.8 drops the host_function edit; NKI guards via `pad_factory_tensors_to_power_of_2=False` (P1.6). Gate confirms host_function has no patch call. |
| R15 | **reduction_expr signature mismatch** (ref `fake_input/fake_output` vs upstream `threads_in_group`) → `TypeError` at reduction codegen. | backend-base delta | NKIBackend's own `reduction_expr` (verbatim, P1.6) carries `fake_input/fake_output`; P1.12 callers pass them and reconcile with upstream's thread-group kwargs (distinct keywords). |
| R16 | `get_masked_value` `'mean'` genuinely absent upstream → mean reductions mis-masked (silent numeric). | inductor_lowering | P1.15 adds `'mean'`; `inspect.getsource` gate; softmax/mean diff. |
| R17 | atomic_ops `load._codegen['nki']`/`store._codegen['nki']` KeyError if memory_ops not ported. | language ops | Strict order P1.17 → P1.19; P1.19 gate asserts both codegens registered. |
| R18 | `_nki_sbuf_shapes` read by P1.16g but registered in P1.9 — empty dict → SBUF alias detection skipped (silent wrong codegen). | tight coupling | P1.9 strictly precedes P1.16; jagged/reduction diffs catch it. |
| R19 | Nondeterministic codegen (dict/set iteration, RNG) → false byte-diffs OR masked real diff; `platform_target` field mismatch fails compatible-config construction. | byte-gate | §Methodology Step 0 reference-vs-reference sanity; `PYTHONHASHSEED=0`, `torch.manual_seed(0)`, Config built without `platform_target`; structured-assertion fallback. |
| R20 | Verbatim copies accidentally "tidied" → output drift. | memory_ops, language ops | Copy via `git checkout`/`git show … | sed -n` ranges, never manual paste; per-step targeted diffs + P1.22 backstop. |
| R21 | Neuron cache hides codegen bugs; relaxed tolerances compound it. | all | Cache-clear + `--no_cache` before every on-device run; byte-diff gate (cache-immune) before any hardware. |
| R22 | `type_propagation→type_info` import drift for relocated symbols. | guide-omitted | P1.3 greps for residual `type_propagation` imports of moved symbols; import-only test. Don't move symbols still defined in type_propagation. |

---

## Do-NOT-port list

- **`helion/runtime/config.py.old`** — stray backup present on the reference branch (verified). Never copy; if it appears, `rm` before committing.
- **The 5 permanently-blocked examples** (xfail in Phase 2, do not chase to "pass"): `nvfp4_gemm`, `fused_linear_jsd`, `mamba2_chunk_scan`, `mamba2_chunk_state` (blocked), `grpo_loss` (compile timeout). 47 pass + 5 non-passing = 52 known (out of 51 runner-selected ± example-set drift; confirm against the live sweep baseline).
- **Manual `'nki'` entry in `BackendLiteral`** — derived from `list_backends()` (settings.py L363); adding by hand is wrong.
- **Manual `'nki'` dict entry / NKIBackend import in `compile_environment.py`** — upstream uses `get_backend_class(settings.backend)()`; reference's dict approach must not be reintroduced.
- **Re-adding `jagged_tile`, `JaggedTileIndexType`, `register_jagged_tile`, `InvalidJaggedTileUsage`, the `jagged_tile` export** — already on upstream; only reconcile NKI-specific deltas (P1.3) and the type_propagation deltas (P1.3b). `JaggedTileIndexType` is duplicated (reference type_propagation.py vs upstream type_info.py) — keep upstream's.
- **Editing `host_function.py`** for `patch_tensor_factories` — upstream removed it there; the NKI guard is the `pad_factory_tensors_to_power_of_2=False` override (P1.6).
- **Changing upstream `Backend.reduction_expr` base signature** — NKI's override (P1.6) carries `fake_input/fake_output`; the base stays as upstream defined it.
- **Adding `ctx` to `_check_block_broadcast_compatibility`** — already upstream (L536).
- **`autotuner/nki_search.py` during Phases 1–2** — Phase 3 only.
- **Modifying upstream `backend.py`** during extraction — `nki_backend.py` is a new file; leave core `backend.py` untouched.

---

## Flat ordered execution checklist (top-to-bottom)

**Phase 1 — wiring + codegen**
1. P1.1 — Copy `nki_fusion.py` + `_nki_dim_access.py` (verbatim, `git checkout`).
2. P1.2 — Add `record_fx_node_ast`/`ast_for_fx_node` to `CodegenInterface` (helper_function.py).
3. P1.3 — Reconcile jagged_tile/loops/exc/tunable deltas (~109 NKI lines; most already upstream).
4. P1.3b — **type_propagation.py**: NKI trailing-singleton + CallableType symint-hint deltas (NOT JaggedTileIndexType; NOT patch guard).
5. P1.4 — `compile_environment.py`: `set_codegen_state` + int64→int32 (both branches) + reduction-DMA guards.
6. P1.5 — `backend_registry.py`: lazy `_maybe_register_nki` hook (modify existing 71-line file).
7. P1.6 — Create `nki_backend.py` (extract NKIOpOverrides+NKIBackend, register, 7 abstracts + `pad_factory_tensors_to_power_of_2=False` + `reduction_expr` with fake_input/output).
8. P1.7 — `default_nki_launcher` + `get_neuron_target` + `platform_target` (Config/Settings) + output_header guard.
9. P1.8 — `NKIProgramIDs` (program_id.py). **No host_function edit.**
10. P1.9 — `device_function.py`: 12 NKI dicts (incl. `_nki_sbuf_shapes`) + 3 methods + `HelionNKIPrinter`.
11. P1.10 — `generate_ast.py`: NKI init dicts + 4 methods + pre-codegen passes + `set_codegen_state` wrap + fx_node/tile_list recording. *(targeted diff: add.py)*
12. P1.11 — `device_ir.py`: SiLU decomp guard + `set_codegen_state` wraps (incl. both while paths).
13. P1.12 — `reduction_strategy.py`: re-anchor NKI reduction codegen; thread `fake_input/output`; reconcile with upstream thread-group kwargs. *(targeted diff: softmax, rms_norm)*
14. P1.13 — `creation_ops.py`: NKI full() memset/transpose.
15. P1.14 — `aten_lowering.py`: upstream lowering objects + 3 NKI-only objects + 15 handlers. *(targeted diff: matmul)*
16. P1.15 — `inductor_lowering.py`: patch_config ctx-mgr + reduction wrap + `'mean'` + dtype + mod/remainder/fmod (NO ctx signature change).
17. P1.16a — `tile_strategy.py`: imports + NKI helpers.
18. P1.16b — `tile_strategy.py`: DeviceGridState/PersistentReductionState/pipeline states (verify upstream classes exist).
19. P1.16c — `tile_strategy.py`: TileStrategy base (offset_prefix, thread_block_size_exprs, get_range_call_str).
20. P1.16d — `tile_strategy.py`: BlockSizeTileStrategy + FlattenedTileStrategy methods.
21. P1.16e — `tile_strategy.py`: `_BaseNDTileStrategy` methods + `codegen_grid` NKI branch (gate before P1.16f).
22. P1.16f — `tile_strategy.py`: `_codegen_grid_nki` method.
23. P1.16g — `tile_strategy.py`: `codegen_device_loop` NKI dynamic-range block (reads `_nki_sbuf_shapes`) + upstream body. *(targeted diff: jagged)*
24. P1.16h — `tile_strategy.py`: `_setup_block_size_constexpr` block_idx + all call sites.
25. P1.16i — `tile_strategy.py`: `mask_var().get()` + CuteNDTileStrategy renames/methods.
26. P1.17a — `memory_ops.py`: 6 NKI helpers (verbatim).
27. P1.17b — `memory_ops.py`: NKI load codegen (verbatim).
28. P1.17c — `memory_ops.py`: NKI store codegen (verbatim). *(targeted diff: add.py, embedding.py)*
29. P1.18a — `matmul_ops.py`: NKI dot codegen. *(targeted diff: matmul)*
30. P1.18b — `scan_ops.py`: NKI scan codegen.
31. P1.18c — `_tracing_ops.py`: NKI `_mask_to` codegen.
32. P1.18d — `random_ops.py`: NKI rand codegen.
33. P1.18e — `barrier.py`: NKI barrier codegen.
34. P1.18f — `view_ops.py`: NKI subscript codegen.
35. P1.19 — `atomic_ops.py`: IndirectAP import + NKI atomic_add codegen (requires P1.17). *(targeted diff: atomic example)*
36. P1.20 — `runtime/kernel.py`: NKI config completion + `HELION_NKI_PROFILE`.
37. P1.21 — `_testing.py`: NKI DEVICE/baseline/tolerance/benchmark-skip.
38. P1.22a — **GATE:** byte-for-byte diff for the 6 representative kernels (after Step-0 reference-vs-reference sanity).
39. P1.22b — **GATE:** byte-for-byte diff for the full 47 (blocked 5 xfail).

**Phase 2 — example sweep (hardware; clear cache + `--no_cache` each run)**
40. P2.1 — On-device smoke (pointwise + matmul).
41. P2.2 — Full 47-example sweep (47 pass; 5 blocked xfail).

**Phase 3 — autotuning**
42. P3.1 — Copy `autotuner/nki_search.py` (verify no module-scope torch_xla) + register in `autotuner/__init__.py`.
43. P3.2 — Real NKI benchmarking (`xm.mark_step` timing), replacing the Phase-1 skip.

Relevant absolute paths (all under `/home/ubuntu/helion_port/`): `helion/_compiler/{backend.py, nki_backend.py(new), backend_registry.py, compile_environment.py, helper_function.py, generate_ast.py, device_function.py, device_ir.py, reduction_strategy.py, aten_lowering.py, inductor_lowering.py, tile_strategy.py, program_id.py, type_propagation.py, type_info.py, nki_fusion.py(new), output_header.py}`, `helion/language/{memory_ops.py, matmul_ops.py, scan_ops.py, _tracing_ops.py, random_ops.py, barrier.py, view_ops.py, atomic_ops.py, creation_ops.py, loops.py, tunable_ops.py, _nki_dim_access.py(new), __init__.py}`, `helion/runtime/{__init__.py, settings.py, config.py, kernel.py}`, `helion/{exc.py, _testing.py}`, `helion/autotuner/{nki_search.py(new), __init__.py}`. **NOT a port step: `helion/_compiler/host_function.py`.** Reference for verbatim sources: `git show fix-nki-kernel-compilation:<path>`.

---

## Reviewer notes addressed (where a critique was wrong or partially wrong)

- **"correctness" reviewer (solid, no changes requested):** no action; the draft's core anchors it spot-checked (`get_backend_class` L234, `backend_name` L1006, output_header `_default_cute_launcher` L37, helper_function `statement_owner_node`) are all verified correct.
- **"completeness" — `backend.py` adds `import ast` and `fake_input/fake_output` on `reduction_expr`, "verify upstream already has these":** **Half-wrong.** `import ast` IS already on upstream (L4) — no action. But `fake_input/fake_output` are **NOT** on upstream's `reduction_expr` (verified count 0 vs reference's 45); upstream instead added `threads_in_group`/thread-group machinery. So this is real work, but it lives in `NKIBackend`'s own override (P1.6) + `reduction_strategy.py` callers (P1.12), not as a base-class edit. Captured as R15.
- **"completeness" — "type_propagation.py has 3 NKI mods" / "JaggedTileIndexType defined in type_propagation":** **Correct and incorporated as P1.3b.** Verified: type_info.py is upstream-only; reference's deltas are in type_propagation.py; JaggedTileIndexType is duplicated and must NOT be re-added (upstream type_info L1141 wins).
- **"completeness" — patch_tensor_factories "guards on `pad_factory_tensors_to_power_of_2`" mismatch:** **Correct.** Verified upstream removed it from host_function entirely (host_function has zero references) and guards via the backend property in type_propagation L903–911. P1.8's host_function edit is dropped; NKI uses `pad_factory_tensors_to_power_of_2=False`. Captured as R14.
- **"ordering" — "upstream's `ReductionLowering` already has mean handling … plan may double-add":** **WRONG.** Verified upstream `get_masked_value` (L949) is `{"sum","prod","min","max"}` with no `"mean"`; reference (L727) has `"mean"`. Adding it is real, single-add work (P1.15, R16).
- **"ordering" — `_check_block_broadcast_compatibility` "upstream ALREADY has ctx … plan risks double-patching":** **Correct.** Verified upstream L536 already takes `ctx`. Draft's "add ctx param" action removed; P1.15 now only ensures the NKI path passes `ctx`. Captured as R7.
- **"ordering" — loops.py "135 lines reference vs 318 vs upstream":** numbers slightly off; **verified** merge-base→reference = 109, merge-base→upstream = 230, upstream→reference = 333. The substance (isolate the ~109-line NKI delta from upstream-refactor noise) is correct and incorporated in P1.3. The draft's "421-line" figure was also wrong and is corrected.
- **"ordering" — "verify all 32 ast_for_fx_node sites are runtime-only, not module scope":** **Verified** — all 32 are inside function bodies (first at memory_ops L571, indented under defs), so the P1.2 shim is import-safe. Concern resolved, no extra step needed.
- **"completeness"/"ordering" — no mid-gates, byte-diff deferred to the end, P1.16 sub-commits unsequenced, RNG/seed nondeterminism, reference-vs-reference sanity, BackendLiteral dynamic-derivation, DeviceGridState class-existence:** **All valid and incorporated** — per-step targeted byte-diffs at every codegen step, P1.16a–i each have a grep/parse gate with an explicit P1.16e→P1.16f ordering gate, P1.22 split into 22a/22b, `PYTHONHASHSEED=0`+`torch.manual_seed(0)`+platform_target-free Config, Step-0 reference-vs-reference sanity check, BackendLiteral verified dynamic (settings L363), and P1.16b verifies upstream classes exist before relying on them.