# NKI Port — Commit Log

In-depth record of every commit made while executing `NKI_PORT_IMPLEMENTATION_PLAN.md`,
porting the NKI backend from `fix-nki-kernel-compilation` (`3056e85c`) onto `upstream/main`
(`eb61d5b8`) on branch `nki-port-v2`.

**How to read this:** newest entries are appended at the bottom. Each entry records the plan
step, what changed and why, how it was verified (with the actual gate result), and any
deviation from the plan or open follow-up. Commits are made only after the step's verification
gate passes.

**Conventions**
- Reference (NKI source of truth): `fix-nki-kernel-compilation`. Verbatim copies use `git show`/`git checkout`, never manual paste.
- Port tree tested via `PYTHONPATH=/home/ubuntu/helion_port` (editable install points elsewhere; never reinstall).
- Baseline (before any NKI work): `helion` imports OK; backends = `['triton','pallas','cute','tileir','metal']`; `nki` absent.

---

## Baseline — `eb61d5b8` (upstream/main, untouched)

- `import helion` succeeds from the port tree.
- `list_backends()` → `['triton','pallas','cute','tileir','metal']` (no `nki`).
- Working tree clean apart from the two planning docs.

---

## P1.1 — Copy NKI-only support files

**Files:** `helion/_compiler/nki_fusion.py` (NEW), `helion/language/_nki_dim_access.py` (NEW)

**What & why:** Copied both verbatim from the reference via `git checkout fix-nki-kernel-compilation -- …`
(no manual edits). These have no upstream equivalent, so there is no conflict surface.
- `_nki_dim_access.py` defines the `DimAccess` hierarchy (Contiguous/Scalar/Indirect/Dynamic/FullSlice/StridedGather)
  plus the `IndirectAP`/`DynamicAP` dataclasses (the post-sentinel-refactor representation) used by
  `memory_ops.py` and `atomic_ops.py` load/store codegen.
- `nki_fusion.py` provides the FX-graph fusion passes (`annotate_psum_reuse`, `annotate_tensor_scalar_reduce`,
  `annotate_activation_reduce`, `annotate_fx_graph`) that mark matmul→activation patterns for PSUM reuse.
  Its `matmul_ops` import is lazy (inside a function body) so no circular import at load time.

**Verification (gate passed):**
- `git diff fix-nki-kernel-compilation -- <both files>` → empty (byte-identical to reference).
- `from helion._compiler.nki_fusion import annotate_fx_graph; from helion.language._nki_dim_access import DimAccess, IndirectAP, DynamicAP` → OK.
- Regression: `import helion` still OK; backends unchanged `['triton','pallas','cute','tileir','metal']` (nki not yet registered — correct, that's P1.5/P1.6).

**Deviations/follow-ups:** none.

## P1.2 — Add record_fx_node_ast / ast_for_fx_node to CodegenInterface

**File:** `helion/_compiler/helper_function.py`

**What & why:** The NKI load/store codegen in `memory_ops.py` calls `ast_for_fx_node`/`record_fx_node_ast`
on the codegen interface 32 times (all inside function bodies — import-safe). Upstream's `CodegenInterface`
does not define them. Added both methods verbatim from the reference, immediately after `lift()`.
Both are no-op defaults on the base (record = pass, ast_for = return None); the real recording is wired in
`generate_ast.py` (P1.10). Kept upstream's `statement_owner_node` (and its `contextlib`/`Iterator`/`Node`
imports) untouched — the reference branch predates that method, but all three coexist cleanly.

**Insertion point:** after `CodegenInterface.lift`, before the module-level `extract_helper_function`.

**Verification (gate passed):**
- Inserted text is byte-identical to `fix-nki-kernel-compilation:helper_function.py` for both methods.
- Gate: `CodegenInterface` has all of `record_fx_node_ast`, `ast_for_fx_node`, `statement_owner_node`.
- Regression: `import helion` OK.

**Deviations/follow-ups:** none. (Note: did NOT port the reference's removal of `statement_owner_node` —
that method is upstream-new and orthogonal; keeping it is correct.)

## P1.3 — Reconcile jagged_tile / loops / exc / tunable deltas

**Files:** `helion/language/tunable_ops.py` (edited). **No edit needed:** `loops.py`, `exc.py`, `language/__init__.py`.

**MAJOR FINDING (plan correction):** The plan estimated a "~109-line NKI delta in loops.py." That figure was
the *merge-base → reference* diff, which only shows the reference *adding* `jagged_tile`. In the 829
intervening commits, **upstream independently added its own jagged_tile scaffolding — a strict superset** of
the reference's:
- `loops.py`: upstream's `jagged_tile` type_propagation handler is **byte-identical** to the reference's
  (verified via extracted-body diff); upstream's docstring is richer (more examples). Neither branch's
  `loops.py` contains any `nki`/`backend`-specific code. → **No edit.**
- `exc.py`: `InvalidJaggedTileUsage` already present upstream (L345). → **No edit.**
- `language/__init__.py`: `jagged_tile` export already present upstream (L26). → **No edit.**

**The ONE real NKI delta** (isolated via `merge-base → reference`, not the misleading `upstream → reference`
diff which conflates upstream's own additions like `signal`/`wait`/`CuteBackendUnavailable`): in
`tunable_ops._register_tunable_type`, the reference returns a `SymIntType(origin, env.create_unbacked_symint(default))`
for `int`-typed tunables instead of `NumericType.subtype(int).new_unbacked(origin)`. This makes an int
`register_tunable` participate as a symbolic int (needed so NKI tile/loop bounds derived from tunables get a
size hint). Added the `if python_type is int:` branch + `SymIntType` import.

**Verification (gate passed):**
- Edited region matches the reference verbatim.
- `create_unbacked_symint(hint=8192)` signature accepts the positional `default()` hint (compile_environment L667).
- `SymIntType`/`NumericType` import cleanly from `type_info` (and are re-exported by `type_propagation`).
- Gate: `jagged_tile`, `InvalidJaggedTileUsage`, `JaggedTileIndexType`, `register_tunable` all import.
- No residual mis-targeted `type_propagation` imports in `helion/language/`. Regression: `import helion` OK.

**Deviation from plan:** The `int → SymIntType` change is **unconditional (not NKI-guarded)** in the
reference, so it affects all backends. Applied as-is to match the reference (and per AGENTS.md "don't add
unnecessary guards"). **Open follow-up:** confirm at P1.22/P2 that this doesn't shift Triton codegen for
int tunables; if it does and that matters, revisit guarding. loops.py/exc.py/__init__.py edits from the
plan were correctly dropped.

## P1.3b — Port NKI type-inference deltas (trailing-singleton + symint-hint)

**File:** `helion/_compiler/type_info.py` (NOT `type_propagation.py` — see correction below).

**PLAN CORRECTION (file location):** The plan said port these into `type_propagation.py`. But upstream's
`type_propagation.py → type_info.py` split moved `TensorType` and `CallableType` into **`type_info.py`**
(`type_propagation` only re-exports them). So both NKI deltas were applied to `type_info.py`.

**PLAN CORRECTION (method names):** The reference's method was `TensorType.propagate_assignment`; upstream
renamed/reorganized it to **`TensorType.propagate_setitem`** (L429). The CallableType hunk lives in
**`CallableType.propagate_call`** (L705). Applied to the correct methods.

**Two deltas applied (verbatim from reference logic):**
1. **Trailing-singletons (in `propagate_setitem`):** when `backend.name == "nki"`, allow a higher-rank RHS to
   be assigned to a lower-rank LHS slice if the extra trailing dims are all size-1 (`rhs_trailing_singletons`).
   NKI's 2D SBUF model produces `[P, F, 1]`-style shapes that must assign into `[P, F]` lanes without a
   RankMismatch. This delta IS NKI-guarded.
2. **Symint-hint capture (in `propagate_call`):** for host-side `_new_symint_on_host_fns` calls, instead of
   `SymIntType.new_unbacked(origin)` (hint defaults to 8192), evaluate the function on size-hinted args to
   compute a concrete `hint`, then `SymIntType(origin, env.create_unbacked_symint(hint=hint))`. Gives NKI
   tile/loop bounds a realistic size hint. NOT backend-guarded in the reference (applies to all backends).
   Added `import contextlib` (was absent); `tree_map_only` already imported.

**NOT applied (correctly, per plan):**
- `JaggedTileIndexType` — already in upstream `type_info.py` (L… ); not duplicated (verified count 1/0).
- `patch_tensor_factories` guard — upstream already guards it at type_propagation.py L903–911 via
  `backend.pad_factory_tensors_to_power_of_2`; NKI participates by overriding that property to False in P1.6.
  No edit here.

**Verification (gate passed):**
- `trailing_singletons` present in `TensorType.propagate_setitem`; `create_unbacked_symint(hint=hint)` present
  in `CallableType.propagate_call`; `type_propagation` re-export of both classes intact.
- `JaggedTileIndexType` count: type_info=1, type_propagation=0 (not duplicated).
- Regression: `import helion` OK.

**Open follow-up:** delta #2 is cross-backend (unconditional) — confirm at P1.22 it doesn't shift Triton codegen.

## P1.4 — compile_environment.py: set_codegen_state + NKI int64/size_hint guards

**File:** `helion/_compiler/compile_environment.py`

**Applied (3 of the reference's NKI pieces):**
1. **`set_codegen_state` context manager + `self._codegen_state` attribute** (added after `has_barrier` /
   after `__init__`). The keystone for NKI statement-based codegen ("Option B"): during codegen the active
   state is parked on the env so NKI op-lowerings can emit `nisa.*` statements and return a result var name.
   Genuinely absent upstream (grep count 0). Added `from collections.abc import Iterator` under TYPE_CHECKING.
2. **int64→int32 fake-tensor promotion** in `_to_fake_tensor`: compute `fake_dtype` once (int64→int32 when
   `backend_name=='nki'`) and thread it through ALL THREE upstream branches — the new `FakeTensor` branch,
   `static_shapes`, and the `from_real_tensor` else-branch (with the pre-cast empty-tensor trick). Upstream
   refactored this method to 3 branches (reference had 2); the cast is applied to each so int64 never leaks.
3. **`size_hint` unbacked-symbol fallback**: before defaulting to 8192, try `expr.xreplace(var_hints)` (return
   if fully concrete) then `shape_env.size_hint(expr)`. Helps NKI bounds resolve to real hints.

**NOT applied here (handled elsewhere — plan corrections):**
- **`backend_factory` dict + `from .backend import NKIBackend`**: the reference hardcoded `"nki": NKIBackend`
  in a dict and hard-imported it. Upstream uses the registry (`get_backend_class(settings.backend)()`), and
  the plugin constraint FORBIDS a hard NKI import here. Skipped — NKI registers via backend_registry (P1.5/P1.6).
- **`jagged_tile_parent_ids`/`jagged_tile_mask_shapes` dicts + `register_jagged_tile`/`is_jagged_tile`**:
  ALREADY present upstream (L270-271, L1222-1226) from upstream's own jagged_tile work. Not re-added.
- **NKI reduction-DMA power-of-2 guard** in `ReductionLoopBlockSizeSource.from_config`: the reference inlined
  `if backend_name=='nki': return max(1, size_hint())`. Upstream replaced that whole path with a backend hook
  `backend.static_rdim_size(size)` (Pallas already returns exact `numel`; Triton/CuTe round to pow2). The
  correct NKI port is to override `NKIBackend.static_rdim_size` to return `numel` exact — **deferred to P1.6**.
  No edit needed in this file.

**Verification (gate passed):** `set_codegen_state` present; `_to_fake_tensor` has `fake_dtype`/`torch.int32`;
`size_hint` has the `hinted_expr` fallback; module AST-parses; `import helion` OK.

**Open follow-up:** ensure P1.6 adds `NKIBackend.static_rdim_size(numel) -> numel`.

## P1.5 — backend_registry.py: lazy _maybe_register_nki hook

**File:** `helion/_compiler/backend_registry.py` (existing file; appended after the builtin-register loop).

**What & why:** NKI is an optional plugin. Added `_maybe_register_nki()` which imports `nki_backend`
(side-effect: that module calls `register_compiler_backend(NKIBackend)` at load). The import is lazy/guarded
so on a non-Trainium box (no torch_xla) NKI is simply absent rather than breaking `import helion`. Did NOT
touch `_BUILTIN_BACKENDS` — NKI is not a builtin; it self-registers on import.

**Dev-time choice:** the except clause is intentionally widened to also catch non-ImportError (SyntaxError,
circular import) and log a warning, so a broken `nki_backend.py` surfaces instead of silently leaving NKI
unregistered. To be narrowed to `except ImportError` once nki_backend.py is stable (tracked for post-P1.6).

**Verification (gate passed):** `nki_backend.py` does not exist yet, so `'nki' not in list_backends()` and
`import helion` still works (ImportError swallowed). `_maybe_register_nki()` is idempotent. backends =
`['triton','pallas','cute','tileir','metal']` (nki appears only after P1.6).

## P1.6 — Create nki_backend.py (extract NKIOpOverrides + NKIBackend) [CENTERPIECE]

**File:** `helion/_compiler/nki_backend.py` (NEW, ~6750 lines). Core `backend.py` left UNTOUCHED.

**What & why:** On the reference branch the entire NKI backend lived inline in `backend.py`
(lines 636–7321: `NKIOpOverrides` 636→5768, module helper `_validate_nki_tensor_shape` 5769→5808,
`NKIBackend` 5809→7321). Upstream's `backend.py` has neither, and uses a plugin registry. So this step
**extracts** that 6686-line block verbatim into a standalone `nki_backend.py` that self-registers.

**How built (guarantees verbatim block):**
- Extracted lines 636–7321 from `fix-nki-kernel-compilation:backend.py` via `git show | sed -n`.
- Prepended a fresh module header: docstring + minimal module-level imports the block needs
  (`abc`, `ast`, `torch`, `exc`, `expr_from_string`, `Backend`, `register_compiler_backend`) plus a
  TYPE_CHECKING block (`OpsHandler`/`InductorOpOverrides`, `Config`, `BoundKernel`, `Argument`,
  `DeviceFunction`, `TileStrategy`). All heavy deps (CompileEnvironment, device_function, torch_xla)
  are lazy-imported inside method bodies (71 CompileEnvironment lazy-imports etc.), so the module imports
  cleanly with no torch_xla at load — plugin semantics preserved.
- Appended `register_compiler_backend(NKIBackend)` at module end.

**Two upstream-new Backend hooks overridden (the reference didn't need these; added by hand):**
- `pad_factory_tensors_to_power_of_2` → `False` (base default True). Replaces the reference's
  patch_tensor_factories guard, which upstream relocated onto this backend property (see P1.3b/P1.4).
- `static_rdim_size(numel)` → `numel` exact (base default `next_power_of_2(numel)` = 512 for 300).
  Replaces the reference's inline ReductionLoopBlockSizeSource NKI guard (deferred here from P1.4).

**Verification (gate passed):**
- AST parses; module imports directly.
- `NKIBackend()` instantiates → `__abstractmethods__` is EMPTY (all 7 of upstream's abstractmethods —
  name, dtype_str, acc_type, function_decorator, constexpr_type, default_launcher_name, library_imports —
  are concretely implemented in the extracted class).
- `import helion` now registers nki: `list_backends()` = `[...,'nki']`; `get_backend_class('nki')` resolves.
- `dtype_str(int64)='nl.int32'`; `pad_factory_tensors_to_power_of_2 is False`; `static_rdim_size(300)==300`.
- `git diff upstream/main -- backend.py` empty (core backend.py untouched).

**Known forward dependency (not a failure):** `NKIBackend.function_decorator` calls
`helion.runtime.settings.get_neuron_target`, which P1.7 adds. Not exercised by the P1.6 gate; resolved at P1.7.

## P1.7 — runtime launcher, settings (get_neuron_target + platform_target), config, output_header

**Files:** `helion/runtime/__init__.py`, `helion/runtime/settings.py`, `helion/runtime/config.py`,
`helion/_compiler/output_header.py`

**Applied:**
1. **`runtime/__init__.py`**: added `default_nki_launcher` verbatim from the reference, inserted right after
   `default_launcher`. Moves tensors to the XLA device, casts int64→int32, auto-bumps LNC to 2 for kernels
   containing `dynamic_range`, runs `xm.mark_step()`, and supports `HELION_NKI_PROFILE` timing. `torch_xla`
   is imported INSIDE the function body (plugin constraint preserved). Added `import time` (module scope;
   `os`/`sys` already upstream).
2. **`settings.py`**: added `get_neuron_target()` (config → `HELION_NEURON_TARGET` → `neuron-ls` autodetect →
   explicit RuntimeError) + `import subprocess`; added `platform_target: str | None = None` field to
   `_Settings` (placed last to avoid dataclass default-ordering issues) and its FIELD_DOCS entry.
   **Did NOT** touch `BackendLiteral`/`_get_backend` — upstream derives them from `list_backends()` so `nki`
   already appears (the reference's manual `"nki": "nki"` mapping entry is unnecessary here).
3. **`config.py`**: added `platform_target: str | None` class attr + `__init__` kwarg + `self.platform_target = …`.
4. **`output_header.py`**: added `"_default_nki_launcher"` to `disallowed_names`.

**This resolves the P1.6 forward dependency:** `NKIBackend.function_decorator` imports `get_neuron_target`,
which now exists. (function_decorator can only be fully exercised inside a live CompileEnvironment during a
real bind, which requires the codegen steps P1.10+; deferred to those gates.)

**Verification (gate passed):** `default_nki_launcher` importable; `Config(platform_target='trn2')` works;
`get_neuron_target()` resolves via env and via config arg; **`torch_xla` NOT in sys.modules after import**
(lazy import confirmed); all 4 files AST-parse; `_default_nki_launcher` in output_header disallowed_names.

**Plan deviation:** skipped the `BackendLiteral`/`_get_backend` mapping edits (registry-driven upstream).

## P1.8 — program_id.py: NKIProgramIDs (host_function edit dropped)

**File:** `helion/_compiler/program_id.py`. **No edit:** `helion/_compiler/host_function.py`.

**What & why:** Appended `class NKIProgramIDs(ProgramIDs)` after `CuteProgramIDs`. NKI compiles a single
program whose grid is always `(1,)` — all tiling happens via `nl.affine_range` loops inside the kernel — so
`codegen` is a no-op and `codegen_grid` returns `(1,)`. Verbatim from reference; orthogonal to upstream's
CuTe PID work.

**host_function.py NOT touched (plan-confirmed):** the reference guarded `patch_tensor_factories` in
host_function with `backend_name != "nki"`, but upstream removed `patch_tensor_factories` from host_function
entirely (grep count 0). NKI's equivalent guard is the `pad_factory_tensors_to_power_of_2=False` override
added to NKIBackend in P1.6.

**Verification (gate passed):** `NKIProgramIDs` imports and subclasses `ProgramIDs`; host_function has 0
`patch_tensor_factories` refs; `import helion` OK.
